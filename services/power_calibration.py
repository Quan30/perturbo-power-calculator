from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PowerCalibration:
    effective_critical_z: float
    dataset: str
    mode: str
    direction: str
    locus_type: str
    effect_percent: tuple[float, ...]
    cells_per_guide: tuple[float, ...]
    depth_multiplier: tuple[float, ...]
    provisional: bool = False
    replicates_per_design: int | None = None


@dataclass(frozen=True)
class CalibrationSelection:
    status: str
    critical_z: float | None
    message: str
    provisional: bool = False
    replicates_per_design: int | None = None


def load_power_calibration(path: str | Path) -> PowerCalibration:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("model_type") != "effective_critical_z":
        raise ValueError("Unsupported power calibration model")
    critical_z = float(payload["effective_critical_z"])
    if not np.isfinite(critical_z) or critical_z <= 0:
        raise ValueError("effective_critical_z must be finite and positive")
    scope = payload["calibration_scope"]
    return PowerCalibration(
        effective_critical_z=critical_z,
        dataset=str(payload["dataset"]),
        mode=str(payload["mode"]),
        direction=str(scope["direction"]),
        locus_type=str(scope.get("locus_type", "enhancer")),
        effect_percent=tuple(float(x) for x in scope["effect_percent"]),
        cells_per_guide=tuple(float(x) for x in scope["cells_per_guide"]),
        depth_multiplier=tuple(float(x) for x in scope["depth_multiplier"]),
        provisional=bool(payload.get("provisional", False)),
        replicates_per_design=(
            int(payload["replicates_per_design"])
            if payload.get("replicates_per_design") is not None
            else None
        ),
    )


def _inside(value: float, supported: tuple[float, ...]) -> bool:
    return min(supported) <= float(value) <= max(supported)


def select_power_calibration(
    *,
    calibration: PowerCalibration,
    dataset: str,
    mode: str,
    direction: str,
    locus_type: str,
    effect_percent: float,
    cells_per_guide: int,
    depth_multiplier: float,
) -> CalibrationSelection:
    categorical_match = (
        dataset == calibration.dataset
        and mode == calibration.mode
        and direction == calibration.direction
        and locus_type == calibration.locus_type
    )
    if not categorical_match:
        return CalibrationSelection(
            status="unsupported",
            critical_z=None,
            message="No matching empirical calibration is available. This result uses the raw analytical estimate.",
        )

    provisional_note = ""
    if calibration.provisional:
        replicate_text = (
            f" from {calibration.replicates_per_design} simulation replicates per design"
            if calibration.replicates_per_design is not None
            else ""
        )
        provisional_note = f" Provisional calibration{replicate_text}; it will be replaced after the larger run."

    within_ranges = (
        _inside(effect_percent, calibration.effect_percent)
        and _inside(cells_per_guide, calibration.cells_per_guide)
        and _inside(depth_multiplier, calibration.depth_multiplier)
    )
    if within_ranges:
        return CalibrationSelection(
            status="validated",
            critical_z=calibration.effective_critical_z,
            message=(
                f"Empirically calibrated for the selected {calibration.dataset} "
                f"{calibration.mode} setting using held-out pairs.{provisional_note}"
            ),
            provisional=calibration.provisional,
            replicates_per_design=calibration.replicates_per_design,
        )
    return CalibrationSelection(
        status="extrapolated",
        critical_z=calibration.effective_critical_z,
        message=(
            "The matching empirical correction is applied, but one or more inputs are outside "
            "its validated ranges. Treat this estimate as an extrapolation."
            f"{provisional_note}"
        ),
        provisional=calibration.provisional,
        replicates_per_design=calibration.replicates_per_design,
    )


def select_matching_power_calibration(
    *,
    calibrations: list[PowerCalibration] | tuple[PowerCalibration, ...],
    dataset: str,
    mode: str,
    direction: str,
    locus_type: str,
    effect_percent: float,
    cells_per_guide: int,
    depth_multiplier: float,
) -> CalibrationSelection:
    """Select a dataset/mode/direction-specific calibration without transferring it."""
    for calibration in calibrations:
        if (
            dataset == calibration.dataset
            and mode == calibration.mode
            and direction == calibration.direction
            and locus_type == calibration.locus_type
        ):
            return select_power_calibration(
                calibration=calibration,
                dataset=dataset,
                mode=mode,
                direction=direction,
                locus_type=locus_type,
                effect_percent=effect_percent,
                cells_per_guide=cells_per_guide,
                depth_multiplier=depth_multiplier,
            )
    return CalibrationSelection(
        status="unsupported",
        critical_z=None,
        message="No matching empirical calibration is available. This result uses the raw analytical estimate.",
    )
