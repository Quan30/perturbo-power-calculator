"""Dataset-neutral structural checks for external AnnData and MuData files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CustomDataMapping:
    perturbation_column: str
    control_value: str
    rna_modality: str | None = None
    counts_layer: str | None = None
    gene_id_column: str | None = None
    gene_symbol_column: str | None = None
    total_counts_column: str | None = None
    control_match: str = "exact"


@dataclass(frozen=True)
class CustomDataReport:
    n_cells: int
    n_genes: int
    count_source: str
    gene_id_source: str
    rna_modality: str | None
    warnings: tuple[str, ...]
    n_control_cells: int = 0


@dataclass(frozen=True)
class CustomBaselineData:
    counts: np.ndarray
    library_sizes: np.ndarray
    gene_ids: np.ndarray
    gene_names: np.ndarray


_MAPPING_FIELDS = (
    "rna_modality",
    "counts_layer",
    "perturbation_column",
    "control_value",
    "control_match",
    "gene_id_column",
    "gene_symbol_column",
    "total_counts_column",
)


def custom_mapping_to_options(mapping: CustomDataMapping) -> dict[str, str]:
    """Serialize a mapping into background-job command options."""
    return {
        field: str(value)
        for field in _MAPPING_FIELDS
        if (value := getattr(mapping, field)) is not None and str(value) != ""
    }


def custom_mapping_from_options(options: dict[str, str]) -> CustomDataMapping:
    """Reconstruct a mapping passed to the background worker."""
    try:
        perturbation_column = options["perturbation_column"]
        control_value = options["control_value"]
    except KeyError as exc:
        raise ValueError(f"Missing custom mapping option: {exc.args[0]}") from exc
    return CustomDataMapping(
        perturbation_column=perturbation_column,
        control_value=control_value,
        control_match=options.get("control_match", "exact"),
        rna_modality=options.get("rna_modality"),
        counts_layer=options.get("counts_layer"),
        gene_id_column=options.get("gene_id_column"),
        gene_symbol_column=options.get("gene_symbol_column"),
        total_counts_column=options.get("total_counts_column"),
    )


def control_cell_mask(labels: pd.Series, expression: str, mode: str = "exact") -> pd.Series:
    """Match controls identically in the preview, validation and fit worker.

    Matching is case-sensitive. Missing annotations never match; contains
    treats punctuation literally, while regex uses regular-expression search.
    """
    if mode not in {"exact", "contains", "regex"}:
        raise ValueError("Unknown control matching mode")
    if not expression or not expression.strip():
        raise ValueError("Negative-control expression must be nonempty")
    values = labels.astype("string")
    if mode == "exact":
        return values.eq(expression).fillna(False).astype(bool)
    if mode == "contains":
        return values.str.contains(expression, regex=False, na=False).astype(bool)
    try:
        pattern = re.compile(expression)
    except re.error as exc:
        raise ValueError(f"Invalid regular expression: {exc}") from exc
    # Evaluate each distinct label once, including when there are many cells.
    matches = {v: bool(pattern.search(v)) for v in values.dropna().unique()}
    return values.map(matches).fillna(False).astype(bool)


def _open_custom_data(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".h5ad":
        import anndata as ad

        return ad.read_h5ad(path, backed="r")
    if suffix == ".h5mu":
        import mudata as md

        return md.read_h5mu(path, backed=True)
    raise ValueError("Custom data must be an .h5ad or .h5mu file")


def _rna_data(data, mapping: CustomDataMapping):
    if not hasattr(data, "mod"):
        return data
    modality = (mapping.rna_modality or "").strip()
    if not modality:
        raise ValueError("MuData input requires an RNA modality name")
    if modality not in data.mod:
        raise ValueError(f"RNA modality '{modality}' was not found")
    return data.mod[modality]


def _close_backed_data(data) -> None:
    file_manager = getattr(data, "file", None)
    if file_manager is not None and hasattr(file_manager, "close"):
        file_manager.close()


def validate_custom_data_structure(data, mapping: CustomDataMapping) -> CustomDataReport:
    """Validate declared metadata without materializing the complete count matrix."""
    rna = _rna_data(data, mapping)
    modality = (mapping.rna_modality or "").strip() or None if hasattr(data, "mod") else None

    perturbation_column = mapping.perturbation_column
    if not perturbation_column.strip() or perturbation_column not in rna.obs.columns:
        raise ValueError(f"obs perturbation column '{perturbation_column}' was not found")
    control_value = mapping.control_value
    controls = control_cell_mask(rna.obs[perturbation_column], control_value, mapping.control_match)
    if not controls.any():
        raise ValueError(
            f"control value '{control_value}' was not found in obs['{perturbation_column}']"
        )

    if mapping.counts_layer:
        if mapping.counts_layer not in rna.layers:
            raise ValueError(f"counts layer '{mapping.counts_layer}' was not found")
        count_source = f"layers['{mapping.counts_layer}']"
    else:
        if getattr(rna, "X", None) is None:
            raise ValueError("X is empty; declare a counts layer")
        count_source = "X"

    if mapping.gene_id_column:
        if mapping.gene_id_column not in rna.var.columns:
            raise ValueError(f"var gene-ID column '{mapping.gene_id_column}' was not found")
        gene_ids = rna.var[mapping.gene_id_column]
        gene_id_source = f"var['{mapping.gene_id_column}']"
    else:
        gene_ids = pd.Series(rna.var_names)
        gene_id_source = "var_names"
    gene_ids = pd.Series(gene_ids, dtype="object")
    if gene_ids.isna().any():
        raise ValueError("Gene identifiers must be nonempty")
    normalized_ids = gene_ids.astype(str).str.strip()
    missing_tokens = {"", "nan", "none", "<na>"}
    if normalized_ids.str.lower().isin(missing_tokens).any():
        raise ValueError("Gene identifiers must be nonempty")
    if normalized_ids.duplicated().any():
        raise ValueError("Gene identifiers must be unique")

    if mapping.gene_symbol_column and mapping.gene_symbol_column not in rna.var.columns:
        raise ValueError(f"var gene-symbol column '{mapping.gene_symbol_column}' was not found")
    if mapping.total_counts_column and mapping.total_counts_column not in rna.obs.columns:
        raise ValueError(f"obs total-counts column '{mapping.total_counts_column}' was not found")

    warnings = []
    if not mapping.total_counts_column:
        warnings.append("No total-counts column declared; the fit will calculate it from raw counts.")
    if not mapping.gene_symbol_column:
        warnings.append("No gene-symbol column declared; stable gene IDs will also be display labels.")

    return CustomDataReport(
        n_cells=int(rna.n_obs),
        n_genes=int(rna.n_vars),
        count_source=count_source,
        gene_id_source=gene_id_source,
        rna_modality=modality,
        warnings=tuple(warnings),
        n_control_cells=int(controls.sum()),
    )


def validate_custom_data_file(path: str | Path, mapping: CustomDataMapping) -> CustomDataReport:
    """Open a custom dataset in backed mode, validate metadata, and close it."""
    path = Path(path).expanduser().resolve(strict=True)
    data = _open_custom_data(path)
    try:
        return validate_custom_data_structure(data, mapping)
    finally:
        _close_backed_data(data)


def _check_raw_counts(matrix) -> None:
    values = matrix.data if hasattr(matrix, 'toarray') else np.asarray(matrix)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Selected count matrix must be finite and nonnegative")
    if not np.allclose(values, np.rint(values), rtol=0, atol=1e-6):
        raise ValueError("Selected matrix must contain raw integer counts, not normalized expression")


def check_custom_count_sample(path: str | Path, mapping: CustomDataMapping, max_cells: int = 100) -> int:
    """Check bounded, evenly spaced control rows without loading other matrices.

    This is a preflight sample, not a guarantee about unsampled rows. The fit
    worker checks all control rows it actually uses before optimization.
    """
    import h5py
    from anndata.io import read_elem, sparse_dataset

    if max_cells < 1:
        raise ValueError('max_cells must be positive')
    with h5py.File(Path(path).expanduser(), 'r') as handle:
        group = handle['mod'][mapping.rna_modality] if 'mod' in handle else handle
        obs = read_elem(group['obs'])
        controls = control_cell_mask(obs[mapping.perturbation_column], mapping.control_value, mapping.control_match)
        indices = np.flatnonzero(controls.to_numpy())
        if not len(indices):
            raise ValueError('No cells match the declared negative-control value')
        if len(indices) > max_cells:
            indices = indices[np.linspace(0, len(indices) - 1, max_cells, dtype=int)]
        source = group['layers'][mapping.counts_layer] if mapping.counts_layer else group['X']
        matrix = sparse_dataset(source)[indices, :] if isinstance(source, h5py.Group) else source[indices, :]
        _check_raw_counts(matrix)
        totals = np.asarray(obs.iloc[indices][mapping.total_counts_column], dtype=float) if mapping.total_counts_column else np.asarray(matrix.sum(axis=1)).reshape(-1)
        if not np.isfinite(totals).all() or (totals < 0).any():
            raise ValueError('Total counts must be finite and nonnegative')
        # Zero counts are valid. A small sample cannot establish that every
        # control has zero counts; the worker checks the complete fit subset.
        return len(indices)


def prepare_custom_baseline_data(
    path: str | Path,
    mapping: CustomDataMapping,
    max_cells: int,
    random_seed: int = 42,
) -> CustomBaselineData:
    """Load validated control-cell arrays for a generic Cortado baseline fit."""
    if max_cells < 1:
        raise ValueError("max_cells must be positive")
    path = Path(path).expanduser().resolve(strict=True)
    data = _open_custom_data(path)
    try:
        validate_custom_data_structure(data, mapping)
        rna = _rna_data(data, mapping)
        controls = control_cell_mask(rna.obs[mapping.perturbation_column], mapping.control_value, mapping.control_match)
        indices = np.flatnonzero(controls.to_numpy())
        if not len(indices):
            raise ValueError("No cells match the declared negative-control value")
        if len(indices) > max_cells:
            rng = np.random.default_rng(random_seed)
            indices = np.sort(rng.choice(indices, max_cells, replace=False))

        matrix_source = rna.layers[mapping.counts_layer] if mapping.counts_layer else rna.X
        matrix = matrix_source[indices, :]
        counts = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        counts = np.asarray(counts)
        if counts.ndim != 2:
            raise ValueError("Selected count matrix must be finite, nonnegative, and two-dimensional")
        _check_raw_counts(counts)

        if mapping.total_counts_column:
            library_sizes = np.asarray(
                rna.obs.iloc[indices][mapping.total_counts_column], dtype=float
            ).reshape(-1)
        else:
            library_sizes = counts.sum(axis=1)
        if not np.isfinite(library_sizes).all() or (library_sizes < 0).any():
            raise ValueError("Total counts must be finite and nonnegative")
        if not np.any(library_sizes > 0):
            raise ValueError("Control cells have zero total counts")

        gene_ids = (
            np.asarray(rna.var[mapping.gene_id_column]).astype(str)
            if mapping.gene_id_column
            else np.asarray(rna.var_names).astype(str)
        )
        gene_names = (
            np.asarray(rna.var[mapping.gene_symbol_column]).astype(str)
            if mapping.gene_symbol_column
            else gene_ids.copy()
        )
        return CustomBaselineData(
            counts=np.asarray(np.rint(counts), dtype=np.float32),
            library_sizes=np.asarray(library_sizes, dtype=np.float32),
            gene_ids=np.asarray(gene_ids, dtype=str),
            gene_names=np.asarray(gene_names, dtype=str),
        )
    finally:
        _close_backed_data(data)
