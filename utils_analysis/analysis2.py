from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
import re

import numpy as np
import pandas as pd


def baseline_expression_overview(reference: pd.DataFrame) -> dict[str, float | int]:
    """Simple summary before a user supplies a target-gene program."""
    expression = pd.to_numeric(reference["reference_expression"], errors="coerce")
    valid = expression[np.isfinite(expression) & (expression >= 0)]
    return {
        "n_genes": len(reference),
        "n_evaluable": len(valid),
        "n_expressed": int((valid > 0).sum()),
        "median_expression": float(valid.median()) if len(valid) else np.nan,
    }

from .power import nb_wald_power, percent_decrease_to_lfc


@dataclass(frozen=True)
class TargetMatchResult:
    matched: pd.DataFrame
    unmatched: list[str]
    ambiguous: list[str]


_BUILTIN_PROGRAM_REQUIRED_COLUMNS = {
    "tf",
    "ensembl_gene_id",
    "gene_symbol",
    "n_supporting_peaks",
}


def _validate_builtin_programs(frame: pd.DataFrame) -> None:
    missing = sorted(_BUILTIN_PROGRAM_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            "Built-in target-program table is missing required columns: "
            + ", ".join(missing)
        )


def available_builtin_tfs(frame: pd.DataFrame) -> list[str]:
    _validate_builtin_programs(frame)
    return sorted(frame["tf"].dropna().astype(str).unique().tolist())


def select_builtin_target_program(
    frame: pd.DataFrame,
    *,
    tf: str,
    min_supporting_peaks: int = 1,
) -> pd.DataFrame:
    _validate_builtin_programs(frame)
    if min_supporting_peaks < 1:
        raise ValueError("min_supporting_peaks must be at least 1")

    selected = frame.loc[
        frame["tf"].astype(str).eq(str(tf))
        & (pd.to_numeric(frame["n_supporting_peaks"], errors="coerce") >= int(min_supporting_peaks))
    ].copy()
    if selected.empty:
        return selected.reset_index(drop=True)

    selected["n_supporting_peaks"] = pd.to_numeric(
        selected["n_supporting_peaks"], errors="coerce"
    ).astype(int)
    selected = selected.sort_values(
        ["ensembl_gene_id", "n_supporting_peaks"],
        ascending=[True, False],
        kind="stable",
    )
    selected = selected.drop_duplicates(subset=["ensembl_gene_id"], keep="first")
    return selected.reset_index(drop=True)


def parse_target_gene_text(text: str) -> list[str]:
    tokens = [t.strip() for t in re.split(r"[\s,;]+", str(text)) if t.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        key = token.upper()
        if key not in seen:
            seen.add(key)
            out.append(token)
    return out


def parse_target_gene_table(frame: pd.DataFrame) -> list[str]:
    if frame is None or frame.empty or len(frame.columns) == 0:
        return []
    preferred = [
        "target_gene",
        "gene",
        "gene_id",
        "gene_symbol",
        "target",
    ]
    lower_to_col = {str(c).lower(): c for c in frame.columns}
    chosen = next((lower_to_col[name] for name in preferred if name in lower_to_col), frame.columns[0])
    values = [str(v) for v in frame[chosen].dropna().tolist()]
    return parse_target_gene_text("\n".join(values))


def match_target_genes(targets: Iterable[str], reference: pd.DataFrame) -> TargetMatchResult:
    required = {"gene_id", "gene_symbol", "reference_expression", "theta"}
    missing = sorted(required - set(reference.columns))
    if missing:
        raise ValueError(f"Gene reference is missing required columns: {', '.join(missing)}")

    ref = reference.copy()
    ref["_id_key"] = ref["gene_id"].astype(str).str.upper()
    ref["_symbol_key"] = ref["gene_symbol"].astype(str).str.upper()

    id_groups = {k: g for k, g in ref.groupby("_id_key", sort=False)}
    symbol_groups = {k: g for k, g in ref.groupby("_symbol_key", sort=False)}

    rows: list[pd.Series] = []
    unmatched: list[str] = []
    ambiguous: list[str] = []
    used_gene_ids: set[str] = set()

    for query in parse_target_gene_text("\n".join(map(str, targets))):
        key = query.upper()
        group = id_groups.get(key)
        source = "gene_id"
        if group is None:
            group = symbol_groups.get(key)
            source = "gene_symbol"

        if group is None or group.empty:
            unmatched.append(query)
            continue
        if source == "gene_symbol" and group["gene_id"].astype(str).nunique() > 1:
            ambiguous.append(query)
            continue

        row = group.iloc[0].copy()
        gid = str(row["gene_id"])
        if gid in used_gene_ids:
            continue
        used_gene_ids.add(gid)
        row["query"] = query
        row["match_source"] = source
        rows.append(row)

    if rows:
        matched = pd.DataFrame(rows).drop(columns=["_id_key", "_symbol_key"], errors="ignore")
        matched = matched.reset_index(drop=True)
    else:
        cols = [c for c in reference.columns] + ["query", "match_source"]
        matched = pd.DataFrame(columns=list(dict.fromkeys(cols)))

    return TargetMatchResult(matched=matched, unmatched=unmatched, ambiguous=ambiguous)


def evaluate_target_program(
    matched_reference: pd.DataFrame,
    effect_percent: float,
    cells_per_guide: int,
    depth_multiplier: float,
    target_power: float,
    min_reference_expression: float = 0.0,
    critical_z: float | None = None,
) -> pd.DataFrame:
    if cells_per_guide <= 0:
        raise ValueError("cells_per_guide must be positive")
    if depth_multiplier <= 0:
        raise ValueError("depth_multiplier must be positive")
    if not 0 < target_power <= 1:
        raise ValueError("target_power must be in (0, 1]")
    if min_reference_expression < 0:
        raise ValueError("min_reference_expression must be non-negative")

    out = matched_reference.copy()
    expr = pd.to_numeric(out.get("reference_expression"), errors="coerce")
    out = out.loc[expr >= float(min_reference_expression)].copy().reset_index(drop=True)

    if out.empty:
        out["effect_percent"] = pd.Series(dtype=float)
        out["lfc"] = pd.Series(dtype=float)
        out["analytical_power"] = pd.Series(dtype=float)
        out["raw_analytical_power"] = pd.Series(dtype=float)
        out["powered"] = pd.Series(dtype=bool)
        return out

    lfc = float(np.asarray(percent_decrease_to_lfc(effect_percent)))
    power_kwargs = dict(
        baseline_mean=pd.to_numeric(out["reference_expression"], errors="coerce").to_numpy(float),
        theta=pd.to_numeric(out["theta"], errors="coerce").to_numpy(float),
        cells_per_arm=float(cells_per_guide),
        lfc=lfc,
        depth_multiplier=float(depth_multiplier),
    )
    raw_power = nb_wald_power(**power_kwargs)
    power = nb_wald_power(**power_kwargs, critical_z=critical_z)
    out["effect_percent"] = float(effect_percent)
    out["lfc"] = lfc
    out["analytical_power"] = power
    out["raw_analytical_power"] = raw_power
    out["powered"] = np.isfinite(power) & (power >= float(target_power))
    return out


def summarize_target_program(
    result: pd.DataFrame,
    target_power: float,
    *,
    n_requested: int,
    n_matched_before_filter: int,
    n_unmatched: int,
    n_ambiguous: int,
) -> dict[str, int | float]:
    if "analytical_power" not in result.columns:
        raise ValueError("Result table must contain analytical_power")
    power = pd.to_numeric(result["analytical_power"], errors="coerce").to_numpy(float)
    valid = np.isfinite(power)
    n_evaluable = int(valid.sum())
    n_powered = int(np.sum(valid & (power >= float(target_power))))
    return {
        "n_requested": int(n_requested),
        "n_matched_before_filter": int(n_matched_before_filter),
        "n_after_expression_filter": int(len(result)),
        "n_evaluable": n_evaluable,
        "n_powered": n_powered,
        "fraction_powered": (n_powered / n_evaluable) if n_evaluable else np.nan,
        "n_unmatched": int(n_unmatched),
        "n_ambiguous": int(n_ambiguous),
    }


def summarize_power_by_expression_bins(
    result: pd.DataFrame,
    target_power: float,
) -> pd.DataFrame:
    """Summarize adequately powered targets across low/medium/high expression groups.

    Groups are equal-count bins after sorting by reference expression. Rows without
    finite expression or analytical power are excluded.
    """
    required = {"reference_expression", "analytical_power"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(
            "Result table is missing required columns: " + ", ".join(missing)
        )
    if not 0 < target_power <= 1:
        raise ValueError("target_power must be in (0, 1]")

    work = result[["reference_expression", "analytical_power"]].copy()
    work["reference_expression"] = pd.to_numeric(
        work["reference_expression"], errors="coerce"
    )
    work["analytical_power"] = pd.to_numeric(
        work["analytical_power"], errors="coerce"
    )
    work = work.loc[
        np.isfinite(work["reference_expression"])
        & np.isfinite(work["analytical_power"])
    ].sort_values("reference_expression", kind="stable").reset_index(drop=True)

    columns = [
        "expression_group",
        "n_targets",
        "n_powered",
        "fraction_powered",
        "expression_min",
        "expression_max",
    ]
    if work.empty:
        return pd.DataFrame(columns=columns)

    n_groups = min(3, len(work))
    labels = {1: ["All"], 2: ["Low", "High"], 3: ["Low", "Medium", "High"]}[n_groups]
    rows = []
    for label, idx in zip(labels, np.array_split(np.arange(len(work)), n_groups)):
        group = work.iloc[idx]
        powered = group["analytical_power"] >= float(target_power)
        n_targets = int(len(group))
        n_powered = int(powered.sum())
        rows.append(
            {
                "expression_group": label,
                "n_targets": n_targets,
                "n_powered": n_powered,
                "fraction_powered": n_powered / n_targets,
                "expression_min": float(group["reference_expression"].min()),
                "expression_max": float(group["reference_expression"].max()),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def fraction_powered_program_vs_cells(
    matched_reference: pd.DataFrame,
    effect_percent: float,
    cells_values: Iterable[int],
    depth_multiplier: float,
    target_power: float,
    min_reference_expression: float = 0.0,
    critical_z: float | None = None,
) -> pd.DataFrame:
    rows = []
    for cells in cells_values:
        result = evaluate_target_program(
            matched_reference,
            effect_percent,
            int(cells),
            depth_multiplier,
            target_power,
            min_reference_expression,
            critical_z,
        )
        power = pd.to_numeric(result["analytical_power"], errors="coerce").to_numpy(float)
        valid = np.isfinite(power)
        n_powered = int(np.sum(valid & (power >= float(target_power))))
        rows.append(
            {
                "cells_per_guide": int(cells),
                "n_evaluable": int(valid.sum()),
                "n_powered": n_powered,
                "fraction_powered": n_powered / int(valid.sum()) if valid.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def fraction_powered_program_vs_depth(
    matched_reference: pd.DataFrame,
    effect_percent: float,
    cells_per_guide: int,
    depth_values: Iterable[float],
    target_power: float,
    min_reference_expression: float = 0.0,
    critical_z: float | None = None,
) -> pd.DataFrame:
    rows = []
    for depth in depth_values:
        result = evaluate_target_program(
            matched_reference,
            effect_percent,
            cells_per_guide,
            float(depth),
            target_power,
            min_reference_expression,
            critical_z,
        )
        power = pd.to_numeric(result["analytical_power"], errors="coerce").to_numpy(float)
        valid = np.isfinite(power)
        n_powered = int(np.sum(valid & (power >= float(target_power))))
        rows.append(
            {
                "depth_multiplier": float(depth),
                "n_evaluable": int(valid.sum()),
                "n_powered": n_powered,
                "fraction_powered": n_powered / int(valid.sum()) if valid.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)
