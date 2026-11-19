import pandas as pd

from utils_analysis.analysis3 import available_pairs, pair_design_curves


def test_pair_picker_and_design_curves_keep_unfinished_simulations_blank():
    rows = []
    for pair, element, gene in [("a", "enhancer-A", "GENE1"), ("b", "enhancer-B", "GENE2")]:
        for cells, depth, reps, detected in [(100, 1.0, 30, 24), (200, 1.0, 0, 0), (100, 2.0, 30, 28)]:
            rows.append({"dataset": "Gasperini", "mode": "cis", "pair_id": pair,
                         "element": element, "gene": gene, "gene_symbol": gene,
                         "baseline_expression": 2., "effect_percent": 25.,
                         "cells_per_guide": cells, "depth_multiplier": depth,
                         "n_detected": detected, "n_replicates": reps,
                         "empirical_power": detected / reps if reps else float("nan")})
    frame = pd.DataFrame(rows)
    assert available_pairs(frame, "Gasperini", "cis")["pair_id"].tolist() == ["a", "b"]
    cells_curve, depth_curve = pair_design_curves(frame, "a", "Gasperini", "cis", 25., 1., 100)
    assert len(cells_curve) == 2 and len(depth_curve) == 2
    assert pd.isna(cells_curve.loc[cells_curve["cells_per_guide"] == 200, "empirical_power"].iloc[0])
    assert depth_curve.loc[depth_curve["depth_multiplier"] == 2., "empirical_power"].iloc[0] == 28 / 30
