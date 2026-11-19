import numpy as np
import pandas as pd
import pytest

from scripts.validate_power_calibration import (
    cross_validated_calibration,
    fit_effective_z,
    prepare_calibration_frame,
    power_from_ncp,
    run_calibration,
    write_outputs,
)


def test_fit_effective_z_recovers_known_threshold():
    ncp = np.linspace(0.1, 8.0, 400)
    expected_z = 4.2
    empirical_power = power_from_ncp(ncp, expected_z)

    fitted_z = fit_effective_z(
        ncp,
        empirical_power,
        np.full(ncp.shape, 30.0),
    )

    assert fitted_z == pytest.approx(expected_z, abs=1e-3)


def test_cross_validation_holds_out_complete_pairs():
    rows = []
    for pair_index in range(10):
        for ncp in (1.0, 3.0, 5.0):
            rows.append(
                {
                    "pair_id": f"pair-{pair_index}",
                    "ncp": ncp,
                    "empirical_power": float(power_from_ncp(ncp, 4.0)),
                    "n_replicates": 30,
                }
            )
    frame = pd.DataFrame(rows)

    result, fold_summary = cross_validated_calibration(frame, n_splits=5)

    assert result["calibrated_power"].notna().all()
    assert result.groupby("pair_id")["validation_fold"].nunique().eq(1).all()
    assert len(fold_summary) == 5
    assert fold_summary["effective_z"].between(3.999, 4.001).all()


def test_prepare_calibration_frame_reconstructs_app_noncentrality():
    frame = pd.DataFrame(
        {
            "dataset": ["Gasperini"],
            "mode": ["cis"],
            "pair_id": ["element__gene"],
            "gene_index": [1],
            "baseline_expression": [2.0],
            "effect_percent": [25.0],
            "cells_per_guide": [100],
            "depth_multiplier": [1.0],
            "fdr_threshold": [0.1],
            "n_replicates": [30],
            "empirical_power": [0.4],
        }
    )
    theta = np.array([1.0, 4.0])
    expected_se = np.sqrt(2.0 * (1.0 / (100 * 2.0) + 1.0 / (100 * 4.0)))
    expected_ncp = abs(np.log(0.75)) / expected_se

    prepared = prepare_calibration_frame(frame, theta)

    assert prepared.loc[0, "ncp"] == pytest.approx(expected_ncp)
    assert prepared.loc[0, "raw_analytical_power"] == pytest.approx(
        float(power_from_ncp(expected_ncp, 1.959963984540054))
    )


def test_prepare_calibration_frame_reconstructs_increase_effect():
    frame = pd.DataFrame(
        {
            "dataset": ["Gasperini"],
            "mode": ["cis"],
            "direction": ["increase"],
            "pair_id": ["p1"],
            "gene_index": [0],
            "baseline_expression": [2.0],
            "effect_percent": [25.0],
            "cells_per_guide": [100],
            "depth_multiplier": [1.0],
            "fdr_threshold": [0.1],
            "n_replicates": [10],
            "empirical_power": [0.5],
        }
    )
    prepared = prepare_calibration_frame(frame, np.array([4.0]))
    expected_variance = 2 * (1 / (100 * 2.0) + 1 / (100 * 4.0))
    assert prepared.loc[0, "ncp"] == pytest.approx(
        np.log(1.25) / np.sqrt(expected_variance)
    )


def test_validation_figure_uses_direction_neutral_effect_label():
    from pathlib import Path

    script = (Path(__file__).resolve().parents[1] / "scripts" / "validate_power_calibration.py").read_text()
    assert 'xlabel="Assumed expression change"' in script


def test_run_and_write_outputs_create_reproducible_artifacts(tmp_path):
    rows = []
    theta = np.array([3.0, 5.0, 8.0, 10.0])
    for pair_index in range(4):
        for effect, empirical in ((10.0, 0.05), (25.0, 0.35), (50.0, 0.85)):
            rows.append(
                {
                    "dataset": "Gasperini",
                    "mode": "cis",
                    "pair_id": f"pair-{pair_index}",
                    "gene_index": pair_index,
                    "baseline_expression": 0.5 + pair_index,
                    "effect_percent": effect,
                    "cells_per_guide": 500,
                    "depth_multiplier": 1.0,
                    "fdr_threshold": 0.1,
                    "n_replicates": 30,
                    "empirical_power": empirical,
                }
            )
    result = run_calibration(pd.DataFrame(rows), theta, n_splits=2)

    paths = write_outputs(result, tmp_path)

    assert paths["figure"].is_file() and paths["figure"].stat().st_size > 1000
    assert paths["metrics"].is_file()
    assert paths["model"].is_file()
    metrics = pd.read_csv(paths["metrics"])
    assert set(metrics["method"]) == {"raw_analytical", "calibrated_cross_validated"}
    assert result.model["model_type"] == "effective_critical_z"
    assert result.model["calibration_scope"]["effect_percent"] == [10.0, 25.0, 50.0]
