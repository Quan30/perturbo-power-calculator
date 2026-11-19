#!/usr/bin/env python3
"""Validate and calibrate analytical power against repeated simulations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.model_selection import GroupKFold


RAW_CRITICAL_Z = float(norm.ppf(0.975))


@dataclass(frozen=True)
class CalibrationResult:
    frame: pd.DataFrame
    fold_summary: pd.DataFrame
    metrics: pd.DataFrame
    model: dict


def power_from_ncp(ncp: np.ndarray | float, critical_z: float) -> np.ndarray:
    """Return two-sided normal-approximation power for noncentrality values."""
    values = np.asarray(ncp, dtype=float)
    return norm.sf(float(critical_z) - values) + norm.cdf(-float(critical_z) - values)


def fit_effective_z(
    ncp: np.ndarray,
    empirical_power: np.ndarray,
    weights: np.ndarray,
    *,
    bounds: tuple[float, float] = (1.96, 8.0),
) -> float:
    """Fit one effective discovery threshold by weighted least squares."""
    ncp = np.asarray(ncp, dtype=float)
    empirical_power = np.asarray(empirical_power, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(ncp)
        & np.isfinite(empirical_power)
        & np.isfinite(weights)
        & (ncp >= 0)
        & (empirical_power >= 0)
        & (empirical_power <= 1)
        & (weights > 0)
    )
    if not np.any(valid):
        raise ValueError("No valid rows are available to fit the calibration")

    def objective(critical_z: float) -> float:
        residual = power_from_ncp(ncp[valid], critical_z) - empirical_power[valid]
        return float(np.average(residual**2, weights=weights[valid]))

    fitted = minimize_scalar(objective, bounds=bounds, method="bounded")
    if not fitted.success:
        raise RuntimeError(f"Calibration fit failed: {fitted.message}")
    return float(fitted.x)


def cross_validated_calibration(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on training pairs and predict every held-out pair exactly once."""
    required = {"pair_id", "ncp", "empirical_power", "n_replicates"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Calibration table is missing: " + ", ".join(missing))
    if frame["pair_id"].nunique() < n_splits:
        raise ValueError("n_splits cannot exceed the number of unique pair_id values")

    result = frame.copy()
    result["calibrated_power"] = np.nan
    result["validation_fold"] = -1
    summaries: list[dict[str, float | int]] = []
    splitter = GroupKFold(n_splits=n_splits)
    groups = result["pair_id"].astype(str).to_numpy()

    for fold, (train, test) in enumerate(splitter.split(result, groups=groups), start=1):
        effective_z = fit_effective_z(
            result.iloc[train]["ncp"].to_numpy(float),
            result.iloc[train]["empirical_power"].to_numpy(float),
            result.iloc[train]["n_replicates"].to_numpy(float),
        )
        result.loc[result.index[test], "calibrated_power"] = power_from_ncp(
            result.iloc[test]["ncp"].to_numpy(float), effective_z
        )
        result.loc[result.index[test], "validation_fold"] = fold
        summaries.append(
            {
                "validation_fold": fold,
                "effective_z": effective_z,
                "n_training_rows": len(train),
                "n_validation_rows": len(test),
            }
        )

    result["validation_fold"] = result["validation_fold"].astype(int)
    return result, pd.DataFrame(summaries)


def _one_value(frame: pd.DataFrame, column: str):
    values = frame[column].dropna().unique().tolist()
    if len(values) != 1:
        raise ValueError(f"{column} must have exactly one value; found {values}")
    return values[0]


def prepare_calibration_frame(frame: pd.DataFrame, theta: np.ndarray) -> pd.DataFrame:
    """Reconstruct the app's analytical power for every empirical row."""
    required = {
        "dataset",
        "mode",
        "pair_id",
        "gene_index",
        "baseline_expression",
        "effect_percent",
        "cells_per_guide",
        "depth_multiplier",
        "fdr_threshold",
        "n_replicates",
        "empirical_power",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Empirical table is missing: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("Empirical table is empty")
    _one_value(frame, "dataset")
    _one_value(frame, "mode")
    _one_value(frame, "fdr_threshold")

    out = frame.copy().reset_index(drop=True)
    # Legacy Gasperini artifacts predate the explicit direction column and
    # contain standardized decreases only.
    if "direction" not in out.columns:
        out["direction"] = "decrease"
    direction = str(_one_value(out, "direction"))
    if direction not in {"decrease", "increase"}:
        raise ValueError("direction must be 'decrease' or 'increase'")
    theta = np.asarray(theta, dtype=float).reshape(-1)
    gene_index_numeric = pd.to_numeric(out["gene_index"], errors="coerce")
    if gene_index_numeric.isna().any() or not np.equal(gene_index_numeric, np.floor(gene_index_numeric)).all():
        raise ValueError("gene_index must contain integers")
    gene_index = gene_index_numeric.to_numpy(dtype=int)
    if (gene_index < 0).any() or (gene_index >= len(theta)).any():
        raise ValueError("gene_index contains values outside the baseline theta array")

    baseline_expression = pd.to_numeric(out["baseline_expression"], errors="coerce").to_numpy(float)
    effect_percent = pd.to_numeric(out["effect_percent"], errors="coerce").to_numpy(float)
    cells_per_guide = pd.to_numeric(out["cells_per_guide"], errors="coerce").to_numpy(float)
    depth_multiplier = pd.to_numeric(out["depth_multiplier"], errors="coerce").to_numpy(float)
    empirical_power = pd.to_numeric(out["empirical_power"], errors="coerce").to_numpy(float)
    n_replicates = pd.to_numeric(out["n_replicates"], errors="coerce").to_numpy(float)
    dispersion = theta[gene_index]

    valid = (
        np.isfinite(baseline_expression)
        & np.isfinite(effect_percent)
        & np.isfinite(cells_per_guide)
        & np.isfinite(depth_multiplier)
        & np.isfinite(empirical_power)
        & np.isfinite(n_replicates)
        & np.isfinite(dispersion)
        & (baseline_expression > 0)
        & (effect_percent > 0)
        & (effect_percent < 100)
        & (cells_per_guide > 0)
        & (depth_multiplier > 0)
        & (empirical_power >= 0)
        & (empirical_power <= 1)
        & (n_replicates > 0)
        & (dispersion > 0)
    )
    if not valid.all():
        bad = np.flatnonzero(~valid)[:10].tolist()
        raise ValueError(f"Invalid analytical or empirical inputs at row indices {bad}")

    if direction == "decrease":
        log_effect = np.abs(np.log1p(-effect_percent / 100.0))
    else:
        log_effect = np.log1p(effect_percent / 100.0)
    variance_per_arm = (
        1.0 / (cells_per_guide * baseline_expression * depth_multiplier)
        + 1.0 / (cells_per_guide * dispersion)
    )
    out["ncp"] = log_effect / np.sqrt(2.0 * variance_per_arm)
    out["raw_analytical_power"] = power_from_ncp(out["ncp"].to_numpy(float), RAW_CRITICAL_Z)
    return out


def _prediction_metrics(frame: pd.DataFrame, column: str, method: str) -> dict[str, float | int | str]:
    observed = frame["empirical_power"].to_numpy(float)
    predicted = frame[column].to_numpy(float)
    residual = predicted - observed
    predicted_powered = predicted >= 0.8
    observed_powered = observed >= 0.8
    return {
        "method": method,
        "n_rows": len(frame),
        "mean_empirical_power": float(observed.mean()),
        "mean_predicted_power": float(predicted.mean()),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(residual.mean()),
        "correlation": float(np.corrcoef(observed, predicted)[0, 1]),
        "predicted_powered_at_80": int(predicted_powered.sum()),
        "empirically_powered_at_80": int(observed_powered.sum()),
        "true_positive_at_80": int((predicted_powered & observed_powered).sum()),
        "false_positive_at_80": int((predicted_powered & ~observed_powered).sum()),
        "false_negative_at_80": int((~predicted_powered & observed_powered).sum()),
    }


def run_calibration(
    empirical_frame: pd.DataFrame,
    theta: np.ndarray,
    *,
    n_splits: int = 5,
) -> CalibrationResult:
    """Prepare, cross-validate, summarize, and fit the final calibration."""
    prepared = prepare_calibration_frame(empirical_frame, theta)
    validated, folds = cross_validated_calibration(prepared, n_splits=n_splits)
    metrics = pd.DataFrame(
        [
            _prediction_metrics(validated, "raw_analytical_power", "raw_analytical"),
            _prediction_metrics(validated, "calibrated_power", "calibrated_cross_validated"),
        ]
    )
    final_z = fit_effective_z(
        validated["ncp"].to_numpy(float),
        validated["empirical_power"].to_numpy(float),
        validated["n_replicates"].to_numpy(float),
    )
    model = {
        "model_type": "effective_critical_z",
        "effective_critical_z": final_z,
        "effective_two_sided_alpha": float(2.0 * norm.sf(final_z)),
        "raw_analytical_critical_z": RAW_CRITICAL_Z,
        "dataset": str(_one_value(validated, "dataset")),
        "mode": str(_one_value(validated, "mode")),
        "fdr_threshold": float(_one_value(validated, "fdr_threshold")),
        "n_rows": int(len(validated)),
        "n_unique_pairs": int(validated["pair_id"].nunique()),
        "cross_validation_folds": int(n_splits),
        "calibration_scope": {
            "direction": str(_one_value(validated, "direction")),
            "effect_percent": sorted(validated["effect_percent"].astype(float).unique().tolist()),
            "cells_per_guide": sorted(validated["cells_per_guide"].astype(int).unique().tolist()),
            "depth_multiplier": sorted(validated["depth_multiplier"].astype(float).unique().tolist()),
        },
    }
    return CalibrationResult(validated, folds, metrics, model)


def _calibration_curve(frame: pd.DataFrame, prediction: str, bins: int = 12) -> pd.DataFrame:
    work = frame[[prediction, "empirical_power", "n_replicates"]].copy()
    edges = np.linspace(0.0, 1.0, bins + 1)
    work["bin"] = pd.cut(work[prediction], edges, include_lowest=True, duplicates="drop")
    rows = []
    for _, group in work.groupby("bin", observed=True):
        weights = group["n_replicates"].to_numpy(float)
        rows.append(
            {
                "predicted": float(np.average(group[prediction], weights=weights)),
                "empirical": float(np.average(group["empirical_power"], weights=weights)),
                "n": len(group),
            }
        )
    return pd.DataFrame(rows)


def make_validation_figure(result: CalibrationResult) -> plt.Figure:
    """Create the four-panel figure supporting the calibration conclusion."""
    frame = result.frame
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    methods = [
        ("raw_analytical_power", "Raw analytical", "#d95f02"),
        ("calibrated_power", "Calibrated (held-out)", "#1b9e77"),
    ]
    for ax, (column, title, color) in zip(axes[0], methods):
        curve = _calibration_curve(frame, column)
        ax.plot([0, 1], [0, 1], color="0.55", linestyle="--", label="Perfect agreement")
        ax.plot(curve["predicted"], curve["empirical"], marker="o", color=color, linewidth=2)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted power", ylabel="Empirical power", title=title)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left")

    grouped = (
        frame.groupby("effect_percent", as_index=False)
        .agg(
            empirical=("empirical_power", "mean"),
            raw=("raw_analytical_power", "mean"),
            calibrated=("calibrated_power", "mean"),
        )
        .sort_values("effect_percent")
    )
    ax = axes[1, 0]
    positions = np.arange(len(grouped))
    width = 0.25
    ax.bar(positions - width, 100 * grouped["empirical"], width, label="Empirical", color="#7570b3")
    ax.bar(positions, 100 * grouped["raw"], width, label="Raw analytical", color="#d95f02")
    ax.bar(positions + width, 100 * grouped["calibrated"], width, label="Calibrated", color="#1b9e77")
    ax.set_xticks(positions, [f"{value:g}%" for value in grouped["effect_percent"]])
    ax.set(xlabel="Assumed expression change", ylabel="Mean power (%)", title="Mean power by effect size")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()

    ax = axes[1, 1]
    metric_lookup = result.metrics.set_index("method")
    labels = ["Empirical", "Raw analytical", "Calibrated"]
    values = [
        int((frame["empirical_power"] >= 0.8).sum()),
        int(metric_lookup.loc["raw_analytical", "predicted_powered_at_80"]),
        int(metric_lookup.loc["calibrated_cross_validated", "predicted_powered_at_80"]),
    ]
    bars = ax.bar(labels, values, color=["#7570b3", "#d95f02", "#1b9e77"])
    ax.set(ylabel="Design–pair rows", title="Rows reaching at least 80% power")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom")

    calibrated = metric_lookup.loc["calibrated_cross_validated"]
    raw = metric_lookup.loc["raw_analytical"]
    fig.suptitle(
        "Analytical power is optimistic; empirical calibration restores agreement\n"
        f"Held-out-pair MAE: {100 * raw['mae']:.1f} pp raw → {100 * calibrated['mae']:.1f} pp calibrated",
        fontsize=15,
    )
    return fig


def write_outputs(result: CalibrationResult, output_dir: Path) -> dict[str, Path]:
    """Write the requested figure, metrics, fold summary, and deployable model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "figure": output_dir / "power_calibration_validation.png",
        "metrics": output_dir / "power_calibration_metrics.csv",
        "folds": output_dir / "power_calibration_folds.csv",
        "model": output_dir / "calibration_model.json",
    }
    figure = make_validation_figure(result)
    figure.savefig(paths["figure"], dpi=200, bbox_inches="tight")
    plt.close(figure)
    result.metrics.to_csv(paths["metrics"], index=False)
    result.fold_summary.to_csv(paths["folds"], index=False)
    paths["model"].write_text(json.dumps(result.model, indent=2) + "\n")
    return paths


def _load_theta(path: Path) -> np.ndarray:
    with np.load(Path(path), allow_pickle=False) as baseline:
        if "theta" not in baseline.files:
            raise ValueError("Baseline NPZ is missing theta")
        theta = np.asarray(baseline["theta"], dtype=float).reshape(-1)
    if not len(theta) or not np.isfinite(theta).all() or (theta <= 0).any():
        raise ValueError("Baseline theta must be nonempty, finite, and positive")
    return theta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline NPZ containing theta")
    parser.add_argument("--empirical", required=True, type=Path, help="Empirical pair-power parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("calibration_results"))
    parser.add_argument("--folds", type=int, default=5, help="Pair-grouped cross-validation folds")
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    empirical = pd.read_parquet(args.empirical)
    result = run_calibration(empirical, _load_theta(args.baseline), n_splits=args.folds)
    paths = write_outputs(result, args.output_dir)

    print(f"Rows: {result.model['n_rows']:,}")
    print(f"Unique pairs: {result.model['n_unique_pairs']:,}")
    print(f"Effective critical z: {result.model['effective_critical_z']:.6f}")
    print(f"Effective two-sided alpha: {result.model['effective_two_sided_alpha']:.6g}")
    print(result.metrics.to_string(index=False))
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
