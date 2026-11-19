import json

import numpy as np
import pandas as pd
import pytest

import services.power_calibration as power_calibration
from services.power_calibration import load_power_calibration, select_power_calibration
from utils_analysis.analysis1 import evaluate_cis_element
from utils_analysis.power import nb_wald_power


@pytest.fixture
def calibration_file(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "model_type": "effective_critical_z",
                "effective_critical_z": 4.272191700633306,
                "dataset": "Gasperini",
                "mode": "cis",
                "calibration_scope": {
                    "direction": "decrease",
                    "locus_type": "enhancer",
                    "effect_percent": [10.0, 25.0, 50.0],
                    "cells_per_guide": [100, 200, 500, 1000, 2000],
                    "depth_multiplier": [0.5, 1.0, 2.0],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_custom_critical_value_reduces_analytical_power():
    raw = nb_wald_power(2.0, 4.0, 500, np.log(0.75), depth_multiplier=1.0)
    calibrated = nb_wald_power(
        2.0,
        4.0,
        500,
        np.log(0.75),
        depth_multiplier=1.0,
        critical_z=4.272191700633306,
    )

    assert float(calibrated) < float(raw)
    with pytest.raises(ValueError, match="critical_z"):
        nb_wald_power(2.0, 4.0, 500, np.log(0.75), critical_z=0)


def test_calibration_selector_distinguishes_validated_extrapolated_and_unsupported(
    calibration_file,
):
    calibration = load_power_calibration(calibration_file)
    common = dict(
        calibration=calibration,
        dataset="Gasperini",
        mode="cis",
        direction="decrease",
        locus_type="enhancer",
        effect_percent=25,
        depth_multiplier=1.0,
    )

    validated = select_power_calibration(cells_per_guide=500, **common)
    extrapolated = select_power_calibration(cells_per_guide=5000, **common)
    unsupported_direction = select_power_calibration(
        cells_per_guide=500, **{**common, "direction": "increase"}
    )
    unsupported_locus = select_power_calibration(
        cells_per_guide=500, **{**common, "locus_type": "variant"}
    )

    assert validated.status == "validated"
    assert validated.critical_z == pytest.approx(4.272191700633306)
    assert extrapolated.status == "extrapolated"
    assert extrapolated.critical_z == pytest.approx(4.272191700633306)
    assert unsupported_direction.status == "unsupported"
    assert unsupported_direction.critical_z is None
    assert unsupported_locus.status == "unsupported"
    assert unsupported_locus.critical_z is None


def test_analysis1_uses_calibrated_power_but_preserves_raw_power():
    reference = pd.DataFrame(
        {
            "element": ["enhancer-1"],
            "gene": ["gene-1"],
            "element_chr": ["chr1"],
            "gene_chr": ["chr1"],
            "distance_bp": [1000],
            "reference_expression": [2.0],
            "theta": [4.0],
            "fraction_cells_detected": [0.8],
        }
    )

    result = evaluate_cis_element(
        reference,
        "enhancer-1",
        window_bp=50_000,
        effect_percent=25,
        cells_per_guide=500,
        depth_multiplier=1.0,
        target_power=0.8,
        critical_z=4.272191700633306,
    )

    assert result.loc[0, "analytical_power"] < result.loc[0, "raw_analytical_power"]
    assert result.loc[0, "powered"] == (result.loc[0, "analytical_power"] >= 0.8)


def test_matching_selector_uses_direction_specific_model_and_reports_provisional(tmp_path):
    def write_model(name, *, dataset, mode, direction, critical_z, provisional=False):
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "model_type": "effective_critical_z",
                    "effective_critical_z": critical_z,
                    "dataset": dataset,
                    "mode": mode,
                    "provisional": provisional,
                    "replicates_per_design": 10 if provisional else 30,
                    "calibration_scope": {
                        "direction": direction,
                        "locus_type": "enhancer",
                        "effect_percent": [10.0, 25.0, 50.0],
                        "cells_per_guide": [100, 500, 2000],
                        "depth_multiplier": [0.5, 1.0, 2.0],
                    },
                }
            ),
            encoding="utf-8",
        )
        return load_power_calibration(path)

    decrease = write_model(
        "decrease.json", dataset="Gasperini", mode="cis", direction="decrease",
        critical_z=4.27,
    )
    increase = write_model(
        "increase.json", dataset="Gasperini", mode="cis", direction="increase",
        critical_z=4.11, provisional=True,
    )

    selected = power_calibration.select_matching_power_calibration(
        calibrations=[decrease, increase], dataset="Gasperini", mode="cis",
        direction="increase", locus_type="enhancer", effect_percent=25,
        cells_per_guide=500, depth_multiplier=1.0,
    )

    assert selected.status == "validated"
    assert selected.critical_z == pytest.approx(4.11)
    assert selected.provisional is True
    assert selected.replicates_per_design == 10
    assert "10 simulation replicates" in selected.message


def test_matching_selector_does_not_transfer_replogle_model_to_custom_data(tmp_path):
    path = tmp_path / "replogle.json"
    path.write_text(
        json.dumps(
            {
                "model_type": "effective_critical_z",
                "effective_critical_z": 4.72,
                "dataset": "Replogle",
                "mode": "trans",
                "provisional": True,
                "replicates_per_design": 10,
                "calibration_scope": {
                    "direction": "decrease",
                    "locus_type": "target_program",
                    "effect_percent": [10.0, 25.0, 50.0],
                    "cells_per_guide": [100, 500, 2000],
                    "depth_multiplier": [0.5, 1.0, 2.0],
                },
            }
        ),
        encoding="utf-8",
    )
    model = load_power_calibration(path)

    selected = power_calibration.select_matching_power_calibration(
        calibrations=[model], dataset="Custom", mode="trans", direction="decrease",
        locus_type="target_program", effect_percent=25, cells_per_guide=500,
        depth_multiplier=1.0,
    )

    assert selected.status == "unsupported"
    assert selected.critical_z is None
    assert "raw analytical estimate" in selected.message
