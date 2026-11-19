import pandas as pd

from utils_analysis.analysis2 import baseline_expression_overview, evaluate_target_program


def test_baseline_expression_overview_counts_and_nonfinite_values():
    reference = pd.DataFrame({
        "gene_id": ["a", "b", "c", "d"],
        "reference_expression": [0, 1.5, 20.0, float("nan")],
    })
    overview = baseline_expression_overview(reference)
    assert overview["n_genes"] == 4
    assert overview["n_evaluable"] == 3
    assert overview["n_expressed"] == 2
    assert overview["median_expression"] == 1.5


def test_target_program_uses_calibration_and_preserves_raw_power():
    reference = pd.DataFrame(
        {
            "gene_id": ["ENSG1"],
            "gene_symbol": ["GENE1"],
            "query": ["GENE1"],
            "match_source": ["gene_symbol"],
            "reference_expression": [2.0],
            "theta": [4.0],
        }
    )

    result = evaluate_target_program(
        reference,
        effect_percent=25,
        cells_per_guide=500,
        depth_multiplier=1.0,
        target_power=0.8,
        critical_z=4.720386,
    )

    assert result.loc[0, "analytical_power"] < result.loc[0, "raw_analytical_power"]
    assert result.loc[0, "powered"] == (result.loc[0, "analytical_power"] >= 0.8)
