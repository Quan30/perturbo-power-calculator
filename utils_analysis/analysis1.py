from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .power import nb_wald_power, percent_change_to_lfc


def select_variant_nearby(
    reference: pd.DataFrame, chromosome: str, position: int, window_bp: int,
    *, reference_build: str, variant_build: str,
) -> pd.DataFrame:
    """Use gene coordinates as candidate targets of a *hypothetical* variant effect."""
    if not reference_build or not variant_build or reference_build != variant_build:
        raise ValueError("Reference and variant genome build must be known and identical")
    if position < 1 or window_bp < 0:
        raise ValueError("Variant position must be positive and window non-negative")
    chromosome = "chr" + str(chromosome).removeprefix("chr")
    required = {"gene_chr", "gene_position", "gene"}
    if missing := required - set(reference.columns):
        raise ValueError("Reference lacks gene coordinates: " + ", ".join(sorted(missing)))
    genes = reference.drop_duplicates("gene").copy()
    genes["distance_bp"] = (pd.to_numeric(genes["gene_position"], errors="coerce") - position).abs()
    genes = genes.loc[(genes["gene_chr"].astype(str) == chromosome) & (genes["distance_bp"] <= window_bp)].copy()
    genes["element"] = f"{chromosome}:{position}"
    genes["element_chr"] = chromosome
    genes["element_position"] = position
    return genes.sort_values(["distance_bp", "gene"], kind="stable").reset_index(drop=True)


def gene_power_curves(
    reference: pd.DataFrame, element: str, gene_id: str, cells: np.ndarray,
    depths: np.ndarray, effect_percent: float, direction: str,
    fixed_cells: int, fixed_depth: float,
    critical_z: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chosen = reference.loc[reference["element"].astype(str).eq(element) & reference["gene"].astype(str).eq(gene_id)]
    if len(chosen) != 1:
        raise ValueError("Select exactly one eligible gene for this locus")
    gene = chosen.iloc[0]
    delta = percent_change_to_lfc(effect_percent, direction)
    common = dict(baseline_mean=float(gene["reference_expression"]), theta=float(gene["theta"]), lfc=delta)
    cells = np.asarray(cells, dtype=int)
    depths = np.asarray(depths, dtype=float)
    raw_cells = nb_wald_power(**common, cells_per_arm=cells, depth_multiplier=fixed_depth)
    raw_depths = nb_wald_power(**common, cells_per_arm=fixed_cells, depth_multiplier=depths)
    return (
        pd.DataFrame({
            "cells_per_guide": cells,
            "analytical_power": nb_wald_power(
                **common, cells_per_arm=cells, depth_multiplier=fixed_depth,
                critical_z=critical_z,
            ),
            "raw_analytical_power": raw_cells,
        }),
        pd.DataFrame({
            "depth_multiplier": depths,
            "analytical_power": nb_wald_power(
                **common, cells_per_arm=fixed_cells, depth_multiplier=depths,
                critical_z=critical_z,
            ),
            "raw_analytical_power": raw_depths,
        }),
    )


def _select_nearby(
    reference: pd.DataFrame,
    element: str | None,
    window_bp: float,
    min_detection_fraction: float,
) -> pd.DataFrame:
    required = {
        "element",
        "element_chr",
        "gene_chr",
        "distance_bp",
        "reference_expression",
        "theta",
        "fraction_cells_detected",
    }
    missing = sorted(required - set(reference.columns))
    if missing:
        raise ValueError(f"Reference table is missing required columns: {', '.join(missing)}")
    if window_bp < 0:
        raise ValueError("window_bp must be non-negative")
    if not 0 <= min_detection_fraction <= 1:
        raise ValueError("min_detection_fraction must be in [0, 1]")

    out = reference.copy() if element is None else reference.loc[reference["element"].astype(str) == str(element)].copy()
    out = out.loc[
        (out["element_chr"].astype(str) == out["gene_chr"].astype(str))
        & pd.to_numeric(out["distance_bp"], errors="coerce").notna()
        & (pd.to_numeric(out["distance_bp"], errors="coerce") <= float(window_bp))
        & (
            pd.to_numeric(out["fraction_cells_detected"], errors="coerce")
            >= float(min_detection_fraction)
        )
    ].copy()
    return out.sort_values(["distance_bp", "gene"], kind="stable").reset_index(drop=True)


def evaluate_cis_element(
    reference: pd.DataFrame,
    element: str,
    window_bp: float,
    effect_percent: float,
    cells_per_guide: int,
    depth_multiplier: float,
    target_power: float,
    min_detection_fraction: float = 0.0,
    direction: str = "decrease",
    critical_z: float | None = None,
) -> pd.DataFrame:
    if cells_per_guide <= 0:
        raise ValueError("cells_per_guide must be positive")
    if depth_multiplier <= 0:
        raise ValueError("depth_multiplier must be positive")
    if not 0 < target_power <= 1:
        raise ValueError("target_power must be in (0, 1]")

    out = _select_nearby(reference, element, window_bp, min_detection_fraction)
    if out.empty:
        out["effect_percent"] = pd.Series(dtype=float)
        out["lfc"] = pd.Series(dtype=float)
        out["analytical_power"] = pd.Series(dtype=float)
        out["raw_analytical_power"] = pd.Series(dtype=float)
        out["powered"] = pd.Series(dtype=bool)
        return out

    lfc = percent_change_to_lfc(effect_percent, direction)
    common = dict(
        baseline_mean=pd.to_numeric(out["reference_expression"], errors="coerce").to_numpy(float),
        theta=pd.to_numeric(out["theta"], errors="coerce").to_numpy(float),
        cells_per_arm=float(cells_per_guide),
        lfc=lfc,
        depth_multiplier=float(depth_multiplier),
    )
    raw_power = nb_wald_power(**common)
    power = nb_wald_power(**common, critical_z=critical_z)
    out["effect_percent"] = float(effect_percent)
    out["lfc"] = lfc
    out["analytical_power"] = power
    out["raw_analytical_power"] = raw_power
    out["powered"] = np.isfinite(power) & (power >= float(target_power))
    return out


def summarize_cis_power(result: pd.DataFrame, target_power: float) -> dict[str, int | float]:
    if "analytical_power" not in result.columns:
        raise ValueError("Result table must contain analytical_power")
    power = pd.to_numeric(result["analytical_power"], errors="coerce").to_numpy(float)
    valid = np.isfinite(power)
    n_nearby = int(len(result))
    n_evaluable = int(valid.sum())
    n_powered = int(np.sum(valid & (power >= float(target_power))))
    return {
        "n_nearby": n_nearby,
        "n_evaluable": n_evaluable,
        "n_powered": n_powered,
        "fraction_powered": (n_powered / n_evaluable) if n_evaluable else np.nan,
        "n_excluded_invalid": n_nearby - n_evaluable,
    }


def fraction_powered_vs_cells(
    reference: pd.DataFrame,
    element: str,
    window_bp: float,
    effect_percent: float,
    cells_values: Iterable[int],
    depth_multiplier: float,
    target_power: float,
    min_detection_fraction: float = 0.0,
    direction: str = "decrease",
    critical_z: float | None = None,
) -> pd.DataFrame:
    rows = []
    for cells in cells_values:
        result = evaluate_cis_element(
            reference,
            element,
            window_bp,
            effect_percent,
            int(cells),
            depth_multiplier,
            target_power,
            min_detection_fraction,
            direction,
            critical_z,
        )
        summary = summarize_cis_power(result, target_power)
        rows.append({"cells_per_guide": int(cells), **summary})
    return pd.DataFrame(rows)


def fraction_powered_vs_depth(
    reference: pd.DataFrame,
    element: str,
    window_bp: float,
    effect_percent: float,
    cells_per_guide: int,
    depth_values: Iterable[float],
    target_power: float,
    min_detection_fraction: float = 0.0,
    direction: str = "decrease",
    critical_z: float | None = None,
) -> pd.DataFrame:
    rows = []
    for depth in depth_values:
        result = evaluate_cis_element(
            reference,
            element,
            window_bp,
            effect_percent,
            cells_per_guide,
            float(depth),
            target_power,
            min_detection_fraction,
            direction,
            critical_z,
        )
        summary = summarize_cis_power(result, target_power)
        rows.append({"depth_multiplier": float(depth), **summary})
    return pd.DataFrame(rows)


def evaluate_cis_dataset(reference, elements, **settings):
    """Evaluate eligible pairs together with exactly the element-level filters."""
    eligible = reference.loc[reference['element'].astype(str).isin(elements)]
    return evaluate_cis_element(eligible, element=None, **settings)
