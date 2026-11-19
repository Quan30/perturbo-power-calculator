#!/usr/bin/env python3
"""Fit and save a control-only PerTurbo baseline from server-local single-cell data.

Requires the PerTurbo model package (importable as ``cortado``), which is not
bundled with this repository.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.baseline_catalog import validate_baseline
from services.custom_data import custom_mapping_from_options, prepare_custom_baseline_data


MISSING_MODEL_MESSAGE = (
    "The PerTurbo model package (importable as `cortado`) is not installed in this Python "
    "environment, so a new baseline cannot be fitted. Install it (see README: 'Fit a baseline "
    "from your own data'), or use one of the included baselines."
)


def require_model_package() -> None:
    if importlib.util.find_spec("cortado") is None:
        raise RuntimeError(MISSING_MODEL_MESSAGE)


def atomic_save_baseline(path: Path, arrays: dict[str, np.ndarray]) -> None:
    staging = path.with_name(path.stem + ".tmp.npz")
    np.savez(staging, **arrays)
    try:
        validate_baseline(staging)
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def fit_gasperini(input_path: Path, output_path: Path, options: dict[str, str]) -> None:
    require_model_package()
    import jax.numpy as jnp
    import mudata as md
    from cortado.api import CortadoData, SVIConfig, fit_control
    from cortado.preprocessing import compute_gene_clip_thresholds, count_gene_outliers_per_cell, winsorize_counts_to_gene_thresholds
    from cortado.utils import compute_size_factors

    data = md.read_h5mu(input_path)
    if not {"gene", "guide"} <= set(data.mod):
        raise ValueError("Gasperini MuData needs 'gene' and 'guide' modalities")
    rna = data.mod["gene"]
    guide = data.mod["guide"]
    if "guide_intended_target_pairs" not in guide.varm or "total_umis" not in rna.obs:
        raise ValueError("Gasperini data needs guide_intended_target_pairs and total_umis")
    thresholds = compute_gene_clip_thresholds(rna.X, percentile=99.0, threshold_floor=10.0)
    keep = np.asarray(count_gene_outliers_per_cell(rna.X, thresholds=thresholds) < 100)
    indices = np.flatnonzero(keep)
    if not len(indices):
        raise ValueError("No cells remain after Gasperini outlier filtering")
    rng = np.random.default_rng(42)
    max_cells = int(options.get("max_cells", "10000"))
    if max_cells > 0 and len(indices) > max_cells:
        indices = np.sort(rng.choice(indices, max_cells, replace=False))
    counts = rna.X[indices]
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    counts = winsorize_counts_to_gene_thresholds(np.asarray(counts), thresholds=thresholds).astype(np.float32)
    n = len(indices)
    library = np.asarray(rna.obs.iloc[indices]["total_umis"], dtype=float)
    if not np.isfinite(library).all() or (library < 0).any():
        raise ValueError("Gasperini total_umis must be finite and nonnegative")
    initial_sf = compute_size_factors(jnp.asarray(counts), library_size=jnp.asarray(library))
    # The control-stage model fixes perturbation effects at zero. A single
    # neutral perturbation column avoids constructing the huge guide matrix.
    cortado_data = CortadoData(
        counts=jnp.asarray(counts), pert_id=jnp.zeros((n, 1), dtype=jnp.int32),
        pert_names=["reference"], gene_names=list(map(str, rna.var_names)),
        size_factors=initial_sf,
    )
    control = fit_control(
        cortado_data, num_steps=int(options.get("steps", "2500")),
        svi_config=SVIConfig(step_size=float(options.get("step_size", ".003"))),
        model_name="negbin", use_observed_size_factors=False,
        progress=True,
    )
    atomic_save_baseline(output_path, {
        "beta_0": np.asarray(control.beta_0), "theta": np.asarray(control.theta),
        "size_factors": np.asarray(control.size_factors),
        "gene_names": np.asarray(rna.var_names.astype(str), dtype=str),
    })


def fit_replogle(input_path: Path, output_path: Path, options: dict[str, str]) -> None:
    require_model_package()
    import anndata as ad
    import jax.numpy as jnp
    from cortado.core import CortadoData, SVIConfig, fit_control
    from cortado.preprocessing import compute_gene_clip_thresholds, winsorize_counts_to_gene_thresholds

    adata = ad.read_h5ad(input_path, backed="r")
    if not {"gene_transcript", "UMI_count"} <= set(adata.obs) or "gene_name" not in adata.var:
        raise ValueError("Replogle data needs gene_transcript, UMI_count, and gene_name")
    indices = np.flatnonzero(adata.obs["gene_transcript"].astype(str).str.contains("non-targeting", case=False, na=False).to_numpy())
    if not len(indices):
        raise ValueError("No Replogle non-targeting control cells were found")
    rng = np.random.default_rng(42)
    max_cells = int(options.get("max_cells", "10000"))
    if max_cells > 0 and len(indices) > max_cells:
        indices = np.sort(rng.choice(indices, max_cells, replace=False))
    selected = adata[indices].to_memory()
    counts = selected.X.toarray() if hasattr(selected.X, "toarray") else np.asarray(selected.X)
    thresholds = compute_gene_clip_thresholds(counts, percentile=99.0)
    counts = winsorize_counts_to_gene_thresholds(counts, thresholds=thresholds).astype(np.float32)
    library = np.asarray(selected.obs["UMI_count"], dtype=float)
    if not np.isfinite(library).all() or (library < 0).any():
        raise ValueError("Replogle UMI_count must be finite and nonnegative")
    center = float(np.log1p(library).mean())
    sf = (np.log1p(library) - center).astype(np.float32).reshape(-1, 1)
    model_data = CortadoData(
        counts=jnp.asarray(counts), pert_id=jnp.zeros(len(indices), dtype=jnp.int32),
        pert_names=["control"], gene_names=list(map(str, selected.var_names)),
        size_factors=jnp.asarray(sf),
    )
    control = fit_control(
        model_data, num_steps=int(options.get("steps", "2000")),
        svi_config=SVIConfig(step_size=float(options.get("step_size", ".01"))),
        model_name="negbin", use_observed_size_factors=True,
        progress=True,
    )
    atomic_save_baseline(output_path, {
        "beta_0": np.asarray(control.beta_0), "theta": np.asarray(control.theta),
        "size_factors": sf, "gene_names": np.asarray(selected.var["gene_name"].astype(str), dtype=str),
        "gene_ids": np.asarray(selected.var_names.astype(str), dtype=str),
        "library_size_center_log_mean": np.float32(center),
    })


def _custom_runtime():
    require_model_package()
    import jax.numpy as jnp
    from cortado.core import CortadoData, SVIConfig, fit_control
    from cortado.preprocessing import compute_gene_clip_thresholds, winsorize_counts_to_gene_thresholds

    return SimpleNamespace(
        asarray=jnp.asarray,
        CortadoData=CortadoData,
        SVIConfig=SVIConfig,
        compute_gene_clip_thresholds=compute_gene_clip_thresholds,
        winsorize_counts_to_gene_thresholds=winsorize_counts_to_gene_thresholds,
        fit_control=fit_control,
    )


def fit_custom(
    input_path: Path,
    output_path: Path,
    options: dict[str, str],
    runtime=None,
) -> None:
    """Fit a control-only baseline from a declared AnnData/MuData mapping."""
    mapping = custom_mapping_from_options(options)
    prepared = prepare_custom_baseline_data(
        input_path,
        mapping,
        max_cells=int(options.get("max_cells", "10000")),
    )
    runtime = runtime or _custom_runtime()
    thresholds = runtime.compute_gene_clip_thresholds(prepared.counts, percentile=99.0)
    counts = runtime.winsorize_counts_to_gene_thresholds(
        prepared.counts, thresholds=thresholds
    ).astype(np.float32)
    log_library = np.log1p(prepared.library_sizes)
    center = float(log_library.mean())
    size_factors = (log_library - center).astype(np.float32).reshape(-1, 1)
    n_cells = counts.shape[0]
    model_data = runtime.CortadoData(
        counts=runtime.asarray(counts),
        pert_id=runtime.asarray(np.zeros(n_cells, dtype=np.int32)),
        pert_names=["control"],
        gene_names=prepared.gene_ids.astype(str).tolist(),
        size_factors=runtime.asarray(size_factors),
    )
    control = runtime.fit_control(
        model_data,
        num_steps=int(options.get("steps", "2000")),
        svi_config=runtime.SVIConfig(step_size=float(options.get("step_size", ".01"))),
        model_name="negbin",
        use_observed_size_factors=True,
        progress=True,
    )
    atomic_save_baseline(
        output_path,
        {
            "beta_0": np.asarray(control.beta_0),
            "theta": np.asarray(control.theta),
            "size_factors": np.asarray(control.size_factors),
            "gene_names": prepared.gene_names,
            "gene_ids": prepared.gene_ids,
            "library_size_center_log_mean": np.float32(center),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=["gasperini", "replogle", "custom"])
    parser.add_argument("--max-cells", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--rna-modality")
    parser.add_argument("--counts-layer")
    parser.add_argument("--perturbation-column")
    parser.add_argument("--control-value")
    parser.add_argument("--control-match", choices=["exact", "contains", "regex"], default="exact")
    parser.add_argument("--gene-id-column")
    parser.add_argument("--gene-symbol-column")
    parser.add_argument("--total-counts-column")
    args = parser.parse_args()
    if args.max_cells < 1 or (args.steps is not None and args.steps < 1):
        parser.error("max-cells and steps must be positive")
    opts = {"max_cells": str(args.max_cells)}
    if args.steps is not None:
        opts["steps"] = str(args.steps)
    if args.step_size is not None:
        opts["step_size"] = str(args.step_size)
    for key in (
        "rna_modality",
        "counts_layer",
        "perturbation_column",
        "control_value",
        "control_match",
        "gene_id_column",
        "gene_symbol_column",
        "total_counts_column",
    ):
        value = getattr(args, key)
        if value is not None:
            opts[key] = value
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    status = folder / "status.json"
    try:
        print(f"Fitting {args.dataset} baseline from {args.input}", flush=True)
        fitters = {
            "gasperini": fit_gasperini,
            "replogle": fit_replogle,
            "custom": fit_custom,
        }
        fitters[args.dataset](args.input, folder / "baseline_params.npz", opts)
        payload = {"status": "complete"}
        print("Baseline fit complete", flush=True)
    except Exception as exc:
        traceback.print_exc()
        payload = {"status": "failed", "error": str(exc)}
    temp = status.with_suffix(".tmp")
    temp.write_text(json.dumps(payload))
    temp.replace(status)
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
