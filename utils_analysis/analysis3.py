from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "dataset",
    "mode",
    "pair_id",
    "element",
    "gene",
    "gene_symbol",
    "baseline_expression",
    "effect_percent",
    "cells_per_guide",
    "depth_multiplier",
    "n_detected",
    "n_replicates",
    "empirical_power",
}


def load_analysis3_artifact(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Analysis 3 artifact not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unsupported Analysis 3 artifact format: {path.suffix}")
    return validate_analysis3_artifact(frame)


def validate_analysis3_artifact(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            "Analysis 3 artifact is missing required columns: " + ", ".join(missing)
        )

    out = frame.copy()
    numeric = [
        "baseline_expression",
        "effect_percent",
        "cells_per_guide",
        "depth_multiplier",
        "n_detected",
        "n_replicates",
        "empirical_power",
    ]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    non_power_numeric = [c for c in numeric if c != "empirical_power"]
    if out[non_power_numeric].isna().any().any():
        bad = [c for c in non_power_numeric if out[c].isna().any()]
        raise ValueError("Analysis 3 artifact contains non-numeric values in: " + ", ".join(bad))
    invalid_blank_power = out["empirical_power"].isna() & out["n_replicates"].ne(0)
    if invalid_blank_power.any():
        raise ValueError("empirical_power may be blank only when n_replicates is 0")
    if (out["baseline_expression"] < 0).any():
        raise ValueError("baseline_expression must be non-negative")
    if (out["effect_percent"] <= 0).any():
        raise ValueError("effect_percent must be positive")
    if (out["cells_per_guide"] <= 0).any():
        raise ValueError("cells_per_guide must be positive")
    if (out["depth_multiplier"] <= 0).any():
        raise ValueError("depth_multiplier must be positive")
    if (out["n_replicates"] < 0).any():
        raise ValueError("n_replicates must be non-negative")
    if ((out["n_detected"] < 0) | (out["n_detected"] > out["n_replicates"])).any():
        raise ValueError("n_detected must lie between 0 and n_replicates")
    finite_power = out["empirical_power"].notna()
    if ((out.loc[finite_power, "empirical_power"] < 0) | (out.loc[finite_power, "empirical_power"] > 1)).any():
        raise ValueError("empirical_power must lie in [0, 1]")

    observed = out["n_replicates"].gt(0)
    expected = out.loc[observed, "n_detected"] / out.loc[observed, "n_replicates"]
    if not np.allclose(
        out.loc[observed, "empirical_power"], expected, atol=1e-9, rtol=1e-9
    ):
        raise ValueError("empirical_power is inconsistent with n_detected / n_replicates")
    if (out.loc[~observed, "n_detected"] != 0).any():
        raise ValueError("n_detected must be 0 when n_replicates is 0")

    if "expected_replicates" in out.columns:
        out["expected_replicates"] = pd.to_numeric(out["expected_replicates"], errors="coerce")
        if out["expected_replicates"].isna().any() or (out["expected_replicates"] <= 0).any():
            raise ValueError("expected_replicates must be positive")
        if (out["n_replicates"] > out["expected_replicates"]).any():
            raise ValueError("n_replicates cannot exceed expected_replicates")

    duplicate_cols = [
        "dataset",
        "mode",
        "pair_id",
        "effect_percent",
        "cells_per_guide",
        "depth_multiplier",
    ]
    if out.duplicated(duplicate_cols).any():
        raise ValueError("Analysis 3 artifact contains duplicate pair/design rows")

    out["cells_per_guide"] = out["cells_per_guide"].astype(int)
    out["n_detected"] = out["n_detected"].astype(int)
    out["n_replicates"] = out["n_replicates"].astype(int)
    if "expected_replicates" in out.columns:
        out["expected_replicates"] = out["expected_replicates"].astype(int)
    return out.reset_index(drop=True)


def available_dataset_modes(frame: pd.DataFrame) -> list[tuple[str, str]]:
    validate_analysis3_artifact(frame)
    pairs = frame[["dataset", "mode"]].drop_duplicates()
    return sorted((str(r.dataset), str(r.mode)) for r in pairs.itertuples(index=False))


def available_pairs(frame: pd.DataFrame, dataset: str, mode: str) -> pd.DataFrame:
    """One searchable choice per measured perturbation–gene pair."""
    work = validate_analysis3_artifact(frame)
    work = work.loc[work["dataset"].astype(str).eq(dataset) & work["mode"].astype(str).eq(mode)]
    return work[["pair_id", "element", "gene", "gene_symbol", "baseline_expression"]].drop_duplicates("pair_id").sort_values(["element", "gene_symbol", "pair_id"]).reset_index(drop=True)


def pair_design_curves(frame: pd.DataFrame, pair_id: str, dataset: str, mode: str,
                       effect_percent: float, depth_multiplier: float,
                       cells_per_guide: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Power for one pair while varying cells or depth independently."""
    work = validate_analysis3_artifact(frame)
    work = work.loc[
        work["pair_id"].astype(str).eq(str(pair_id))
        & work["dataset"].astype(str).eq(dataset)
        & work["mode"].astype(str).eq(mode)
        & _match_float(work["effect_percent"], effect_percent)
    ]
    cells = work.loc[_match_float(work["depth_multiplier"], depth_multiplier)].sort_values("cells_per_guide")
    depths = work.loc[work["cells_per_guide"].eq(cells_per_guide)].sort_values("depth_multiplier")
    return cells.reset_index(drop=True), depths.reset_index(drop=True)


def _match_float(series: pd.Series, value: float) -> pd.Series:
    return np.isclose(pd.to_numeric(series, errors="coerce"), float(value), rtol=0, atol=1e-9)


def filter_pair_power(
    frame: pd.DataFrame,
    *,
    dataset: str,
    mode: str,
    effect_percent: float,
    cells_per_guide: int,
    depth_multiplier: float,
    min_expression: float = 0.0,
) -> pd.DataFrame:
    if min_expression < 0:
        raise ValueError("min_expression must be non-negative")
    work = validate_analysis3_artifact(frame)
    mask = (
        work["dataset"].astype(str).eq(str(dataset))
        & work["mode"].astype(str).eq(str(mode))
        & _match_float(work["effect_percent"], effect_percent)
        & work["cells_per_guide"].eq(int(cells_per_guide))
        & _match_float(work["depth_multiplier"], depth_multiplier)
        & (work["baseline_expression"] >= float(min_expression))
    )
    return work.loc[mask].sort_values("pair_id", kind="stable").reset_index(drop=True)



def design_replicate_progress(current: pd.DataFrame) -> dict[str, int | bool]:
    if "n_replicates" not in current.columns:
        raise ValueError("current design table must contain n_replicates")
    reps = pd.to_numeric(current["n_replicates"], errors="coerce").dropna().astype(int)
    completed = int(reps.max()) if len(reps) else 0
    if "expected_replicates" in current.columns:
        expected_vals = pd.to_numeric(current["expected_replicates"], errors="coerce").dropna().astype(int)
        expected = int(expected_vals.max()) if len(expected_vals) else completed
    else:
        expected = completed
    return {
        "completed": completed,
        "expected": expected,
        "is_provisional": completed < expected,
    }

def current_design_summary(current: pd.DataFrame, target_power: float) -> dict[str, float | int]:
    if not 0 < target_power <= 1:
        raise ValueError("target_power must be in (0, 1]")
    if "empirical_power" not in current.columns:
        raise ValueError("current design table must contain empirical_power")
    power = pd.to_numeric(current["empirical_power"], errors="coerce")
    power = power[np.isfinite(power)]
    n_pairs = int(len(power))
    n_powered = int((power >= float(target_power)).sum())
    return {
        "n_pairs": n_pairs,
        "n_powered": n_powered,
        "fraction_powered": (n_powered / n_pairs) if n_pairs else np.nan,
        "median_empirical_power": float(power.median()) if n_pairs else np.nan,
        "mean_empirical_power": float(power.mean()) if n_pairs else np.nan,
    }


def _base_subset(
    frame: pd.DataFrame,
    *,
    dataset: str,
    mode: str,
    effect_percent: float,
    min_expression: float,
) -> pd.DataFrame:
    work = validate_analysis3_artifact(frame)
    return work.loc[
        work["dataset"].astype(str).eq(str(dataset))
        & work["mode"].astype(str).eq(str(mode))
        & _match_float(work["effect_percent"], effect_percent)
        & (work["baseline_expression"] >= float(min_expression))
    ].copy()


def fraction_powered_vs_cells(
    frame: pd.DataFrame,
    *,
    dataset: str,
    mode: str,
    effect_percent: float,
    depth_multiplier: float,
    target_power: float,
    min_expression: float = 0.0,
) -> pd.DataFrame:
    work = _base_subset(
        frame,
        dataset=dataset,
        mode=mode,
        effect_percent=effect_percent,
        min_expression=min_expression,
    )
    work = work.loc[_match_float(work["depth_multiplier"], depth_multiplier)]
    rows = []
    for cpg, group in work.groupby("cells_per_guide", sort=True):
        summary = current_design_summary(group, target_power)
        rows.append({"cells_per_guide": int(cpg), **summary})
    return pd.DataFrame(rows)


def fraction_powered_vs_depth(
    frame: pd.DataFrame,
    *,
    dataset: str,
    mode: str,
    effect_percent: float,
    cells_per_guide: int,
    target_power: float,
    min_expression: float = 0.0,
) -> pd.DataFrame:
    work = _base_subset(
        frame,
        dataset=dataset,
        mode=mode,
        effect_percent=effect_percent,
        min_expression=min_expression,
    )
    work = work.loc[work["cells_per_guide"].eq(int(cells_per_guide))]
    rows = []
    for depth, group in work.groupby("depth_multiplier", sort=True):
        summary = current_design_summary(group, target_power)
        rows.append({"depth_multiplier": float(depth), **summary})
    return pd.DataFrame(rows)


def summarize_power_by_expression_decile(
    current: pd.DataFrame,
    target_power: float,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    required = {"baseline_expression", "empirical_power"}
    missing = sorted(required - set(current.columns))
    if missing:
        raise ValueError("current design table is missing: " + ", ".join(missing))

    work = current[["baseline_expression", "empirical_power"]].copy()
    work["baseline_expression"] = pd.to_numeric(work["baseline_expression"], errors="coerce")
    work["empirical_power"] = pd.to_numeric(work["empirical_power"], errors="coerce")
    work = work.loc[
        np.isfinite(work["baseline_expression"]) & np.isfinite(work["empirical_power"])
    ].sort_values("baseline_expression", kind="stable").reset_index(drop=True)

    cols = [
        "expression_bin",
        "expression_label",
        "n_pairs",
        "n_powered",
        "fraction_powered",
        "expression_min",
        "expression_max",
        "median_empirical_power",
        "mean_empirical_power",
    ]
    if work.empty:
        return pd.DataFrame(columns=cols)

    n_actual = min(int(n_bins), len(work))
    rows = []
    if work['baseline_expression'].nunique() == 1:
        groups = [(str(work['baseline_expression'].iloc[0]), work)]
    else:
        work['_quantile'] = pd.qcut(work['baseline_expression'], q=n_actual, duplicates='drop')
        groups = work.groupby('_quantile', observed=True, sort=True)
    for i, (interval, group) in enumerate(groups, start=1):
        powered = group["empirical_power"] >= float(target_power)
        rows.append(
            {
                "expression_bin": str(i),
                "expression_label": str(interval),
                "n_pairs": int(len(group)),
                "n_powered": int(powered.sum()),
                "fraction_powered": float(powered.mean()),
                "expression_min": float(group["baseline_expression"].min()),
                "expression_max": float(group["baseline_expression"].max()),
                "median_empirical_power": float(group["empirical_power"].median()),
                "mean_empirical_power": float(group["empirical_power"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def observed_effects_for_pairs(
    pair_power: pd.DataFrame,
    effects: pd.DataFrame,
    *,
    dataset: str,
    mode: str,
    fdr_threshold: float = 0.1,
    decreases_only: bool = True,
) -> pd.DataFrame:
    """Return observed Cortado effects for the pair universe used by Analysis 3.

    ``posterior_mean`` is interpreted as a natural-log fold change and converted
    to signed percent expression change. If ``empirical_fdr`` is present, only
    effects at or below ``fdr_threshold`` are retained.
    """
    required = {"element", "gene", "posterior_mean"}
    missing = sorted(required - set(effects.columns))
    if missing:
        raise ValueError("Observed-effect table is missing: " + ", ".join(missing))
    if not 0 < float(fdr_threshold) <= 1:
        raise ValueError("fdr_threshold must be in (0, 1]")

    pairs = validate_analysis3_artifact(pair_power)
    pairs = pairs.loc[
        pairs["dataset"].astype(str).eq(str(dataset))
        & pairs["mode"].astype(str).eq(str(mode)),
        ["element", "gene"],
    ].drop_duplicates()

    work = effects.copy()
    work["posterior_mean"] = pd.to_numeric(work["posterior_mean"], errors="coerce")
    work = work.loc[np.isfinite(work["posterior_mean"])]
    if "empirical_fdr" in work.columns:
        work["empirical_fdr"] = pd.to_numeric(work["empirical_fdr"], errors="coerce")
        work = work.loc[
            np.isfinite(work["empirical_fdr"])
            & (work["empirical_fdr"] <= float(fdr_threshold))
        ]

    work = pairs.merge(work, on=["element", "gene"], how="inner")
    work["observed_percent_change"] = 100.0 * (np.exp(work["posterior_mean"]) - 1.0)
    if decreases_only:
        work = work.loc[work["observed_percent_change"] < 0]
    return work.sort_values(["element", "gene"], kind="stable").reset_index(drop=True)



def observed_cis_effects_from_pair_table(
    pair_table: pd.DataFrame,
    effects: pd.DataFrame,
    *,
    fdr_threshold: float = 0.1,
    decreases_only: bool = False,
) -> pd.DataFrame:
    """Return significant observed effects across the full Gasperini cis-candidate universe.

    The cis universe is defined by the Gasperini pair table after removing NTC rows.
    Only element-gene pairs present in the Cortado effect table can contribute an
    observed effect. ``posterior_mean`` is interpreted as natural-log fold change.
    """
    pair_required = {"gRNAgroup", "general_group", "ENSG.targetgene"}
    pair_missing = sorted(pair_required - set(pair_table.columns))
    if pair_missing:
        raise ValueError("Gasperini pair table is missing: " + ", ".join(pair_missing))

    effect_required = {"element", "gene", "posterior_mean", "empirical_fdr"}
    effect_missing = sorted(effect_required - set(effects.columns))
    if effect_missing:
        raise ValueError("Observed-effect table is missing: " + ", ".join(effect_missing))
    if not 0 < float(fdr_threshold) <= 1:
        raise ValueError("fdr_threshold must be in (0, 1]")

    pairs = pair_table.loc[
        pair_table["general_group"].astype(str).ne("NTC"),
        ["gRNAgroup", "ENSG.targetgene"],
    ].rename(columns={"gRNAgroup": "element", "ENSG.targetgene": "gene"})
    pairs["element"] = pairs["element"].astype(str)
    pairs["gene"] = pairs["gene"].astype(str)
    pairs = pairs.drop_duplicates(["element", "gene"])

    work = effects[["element", "gene", "posterior_mean", "empirical_fdr"]].copy()
    work["element"] = work["element"].astype(str)
    work["gene"] = work["gene"].astype(str)
    work["posterior_mean"] = pd.to_numeric(work["posterior_mean"], errors="coerce")
    work["empirical_fdr"] = pd.to_numeric(work["empirical_fdr"], errors="coerce")
    work = work.loc[
        np.isfinite(work["posterior_mean"])
        & np.isfinite(work["empirical_fdr"])
        & (work["empirical_fdr"] <= float(fdr_threshold))
    ]

    out = pairs.merge(work, on=["element", "gene"], how="inner")
    out["observed_percent_change"] = 100.0 * (np.exp(out["posterior_mean"]) - 1.0)
    if decreases_only:
        out = out.loc[out["observed_percent_change"] < 0]
    return out.sort_values(["element", "gene"], kind="stable").reset_index(drop=True)

def expression_power_comparison(
    frame: pd.DataFrame,
    *,
    dataset: str,
    mode: str,
    depth_multiplier: float,
    target_power: float,
    min_expression: float = 0.0,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarize adequately-powered fraction by expression bin for every effect/cell design."""
    work = validate_analysis3_artifact(frame)
    work = work.loc[
        work["dataset"].astype(str).eq(str(dataset))
        & work["mode"].astype(str).eq(str(mode))
        & _match_float(work["depth_multiplier"], depth_multiplier)
        & (work["baseline_expression"] >= float(min_expression))
    ].copy()

    rows: list[pd.DataFrame] = []
    for (effect, cpg), group in work.groupby(
        ["effect_percent", "cells_per_guide"], sort=True
    ):
        summary = summarize_power_by_expression_decile(
            group, target_power=target_power, n_bins=n_bins
        )
        if summary.empty:
            continue
        summary.insert(0, "cells_per_guide", int(cpg))
        summary.insert(0, "effect_percent", float(effect))
        rows.append(summary)
    if not rows:
        return pd.DataFrame(
            columns=[
                "effect_percent",
                "cells_per_guide",
                "expression_bin",
                "n_pairs",
                "n_powered",
                "fraction_powered",
                "expression_min",
                "expression_max",
                "median_empirical_power",
                "mean_empirical_power",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def split_expression_power_by_cells(comparison: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Split expression-power summaries into one sorted panel per cells/guide design."""
    if comparison.empty:
        return {}
    required = {"cells_per_guide", "effect_percent", "expression_bin", "mean_empirical_power"}
    missing = sorted(required - set(comparison.columns))
    if missing:
        raise ValueError("expression-power comparison is missing: " + ", ".join(missing))

    panels: dict[int, pd.DataFrame] = {}
    for cpg, group in comparison.groupby("cells_per_guide", sort=True):
        panel = group.copy()
        panel["expression_bin"] = pd.to_numeric(panel["expression_bin"], errors="coerce")
        panel = panel.loc[
            panel["expression_bin"].notna()
            & pd.to_numeric(panel["mean_empirical_power"], errors="coerce").notna()
        ].sort_values(["effect_percent", "expression_bin"], kind="stable")
        if not panel.empty:
            panels[int(cpg)] = panel.reset_index(drop=True)
    return panels
