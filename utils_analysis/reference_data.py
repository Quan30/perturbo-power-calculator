from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


_REQUIRED_PAIR_COLUMNS = {
    "gRNAgroup.chr",
    "gRNAgroup.start",
    "gRNAgroup.stop",
    "gRNAgroup",
    "general_group",
    "chr.targetgene",
    "start.targetgene",
    "stop.targetgene",
    "ENSG.targetgene",
    "targetgene_short_name",
}
_REQUIRED_DETECTION_COLUMNS = {
    "gene",
    "n_cells_total",
    "n_cells_detected",
    "fraction_cells_detected",
}
_EXCLUDED_DEFAULT_GROUPS = {"NTC", "TSS", "positive_ctrl"}


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def reference_depth_counts(path: str | Path) -> float:
    """Return the model's reference total expected counts per cell.

    ``exp(beta_0)`` is the expected count for each modeled gene at 1x depth,
    so their sum gives the absolute count scale corresponding to the existing
    depth multiplier of one.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline parameters not found: {path}")
    with np.load(path, allow_pickle=False) as params:
        if "beta_0" not in params.files:
            raise ValueError("baseline_params.npz is missing key: beta_0")
        beta_0 = np.asarray(params["beta_0"], dtype=float).reshape(-1)
    counts = float(np.exp(beta_0).sum())
    if not np.isfinite(counts) or counts <= 0:
        raise ValueError("Reference counts per cell must be finite and positive")
    return counts


def counts_to_depth_multiplier(counts_per_cell: float, reference_counts: float) -> float:
    """Convert an absolute expected count depth to the model's relative scale."""
    counts = float(counts_per_cell)
    reference = float(reference_counts)
    if not np.isfinite(counts) or not np.isfinite(reference) or counts <= 0 or reference <= 0:
        raise ValueError("Counts per cell and reference counts must be finite and positive")
    return counts / reference


def reference_depth_from_gene_reference(reference: pd.DataFrame) -> float:
    """Reuse a loaded gene reference instead of reopening its baseline archive."""
    if "reference_expression" not in reference.columns:
        raise ValueError("Gene reference is missing reference_expression")
    expression = pd.to_numeric(reference["reference_expression"], errors="coerce")
    counts = float(expression.sum())
    if not np.isfinite(counts) or counts <= 0:
        raise ValueError("Reference counts per cell must be finite and positive")
    return counts


def load_baseline_reference(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline parameters not found: {path}")

    with np.load(path, allow_pickle=False) as params:
        required = {"beta_0", "theta", "gene_names"}
        missing = sorted(required - set(params.files))
        if missing:
            raise ValueError(f"baseline_params.npz is missing keys: {', '.join(missing)}")
        beta_0 = np.asarray(params["beta_0"], dtype=float).reshape(-1)
        theta = np.asarray(params["theta"], dtype=float).reshape(-1)
        genes = np.asarray(params["gene_names"], dtype=object).astype(str).reshape(-1)

    if not (len(genes) == len(beta_0) == len(theta)):
        raise ValueError("gene_names, beta_0, and theta must have equal lengths")

    return pd.DataFrame(
        {
            "gene": genes,
            "beta_0": beta_0,
            "reference_expression": np.exp(beta_0),
            "theta": theta,
        }
    )



def load_gene_reference(path: str | Path) -> pd.DataFrame:
    """Load dataset-neutral gene-level baseline parameters.

    Replogle baselines carry stable Ensembl IDs in ``gene_ids`` and display
    symbols in ``gene_names``. Gasperini baselines only carry ``gene_names``,
    which are already stable Ensembl IDs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline parameters not found: {path}")

    with np.load(path, allow_pickle=False) as params:
        required = {"beta_0", "theta", "gene_names"}
        missing = sorted(required - set(params.files))
        if missing:
            raise ValueError(f"baseline_params.npz is missing keys: {', '.join(missing)}")

        beta_0 = np.asarray(params["beta_0"], dtype=float).reshape(-1)
        theta = np.asarray(params["theta"], dtype=float).reshape(-1)
        gene_names = np.asarray(params["gene_names"], dtype=object).astype(str).reshape(-1)
        if "gene_ids" in params.files:
            gene_ids = np.asarray(params["gene_ids"], dtype=object).astype(str).reshape(-1)
            gene_symbols = gene_names
        else:
            gene_ids = gene_names
            gene_symbols = gene_names

    if not (len(gene_ids) == len(gene_symbols) == len(beta_0) == len(theta)):
        raise ValueError("gene identifiers, beta_0, and theta must have equal lengths")

    return pd.DataFrame(
        {
            "gene_id": gene_ids,
            "gene_symbol": gene_symbols,
            "beta_0": beta_0,
            "reference_expression": np.exp(beta_0),
            "theta": theta,
        }
    )

def build_gasperini_cis_reference(
    pair_table_path: str | Path,
    baseline_path: str | Path,
    detection_path: str | Path,
) -> pd.DataFrame:
    pair_table_path = Path(pair_table_path)
    detection_path = Path(detection_path)
    if not pair_table_path.exists():
        raise FileNotFoundError(f"Gasperini pair table not found: {pair_table_path}")
    if not detection_path.exists():
        raise FileNotFoundError(f"Gene detection summary not found: {detection_path}")

    pairs = pd.read_csv(pair_table_path, sep="\t", low_memory=False)
    _require_columns(pairs, _REQUIRED_PAIR_COLUMNS, "Gasperini pair table")

    suffix = detection_path.suffix.lower()
    if suffix == ".parquet":
        detection = pd.read_parquet(detection_path)
    elif suffix == ".csv":
        detection = pd.read_csv(detection_path)
    elif suffix in {".tsv", ".txt"}:
        detection = pd.read_csv(detection_path, sep="\t")
    else:
        raise ValueError(f"Unsupported detection-summary format: {detection_path.suffix}")
    _require_columns(detection, _REQUIRED_DETECTION_COLUMNS, "Gene detection summary")

    baseline = load_baseline_reference(baseline_path)

    frame = pairs.rename(
        columns={
            "gRNAgroup": "element",
            "gRNAgroup.chr": "element_chr",
            "chr.targetgene": "gene_chr",
            "ENSG.targetgene": "gene",
            "targetgene_short_name": "gene_symbol",
        }
    ).copy()

    # Match the Gasperini cis-table convention used in the benchmark
    # inspection: distance is measured from element start to target-gene start.
    frame["element_position"] = pd.to_numeric(
        frame["gRNAgroup.start"], errors="coerce"
    )
    frame["gene_position"] = pd.to_numeric(
        frame["start.targetgene"], errors="coerce"
    )
    frame["distance_bp"] = np.where(
        frame["element_chr"].astype(str) == frame["gene_chr"].astype(str),
        np.abs(frame["gene_position"] - frame["element_position"]),
        np.nan,
    )

    keep = [
        "element",
        "general_group",
        "element_chr",
        "element_position",
        "gene",
        "gene_symbol",
        "gene_chr",
        "gene_position",
        "distance_bp",
    ]
    frame = frame[keep].drop_duplicates(subset=["element", "gene"], keep="first")
    frame = frame.merge(baseline, on="gene", how="inner", validate="many_to_one")
    frame = frame.merge(
        detection[list(_REQUIRED_DETECTION_COLUMNS)],
        on="gene",
        how="left",
        validate="many_to_one",
    )
    return frame.sort_values(["element", "distance_bp", "gene"], kind="stable").reset_index(drop=True)


def available_gasperini_elements(reference: pd.DataFrame) -> list[str]:
    _require_columns(reference, {"element", "general_group"}, "Cis reference table")
    eligible = reference.loc[~reference["general_group"].isin(_EXCLUDED_DEFAULT_GROUPS), "element"]
    return sorted(eligible.dropna().astype(str).unique().tolist())
