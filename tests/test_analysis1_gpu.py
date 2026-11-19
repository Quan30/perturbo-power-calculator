import numpy as np
import pandas as pd
import pytest

from utils_analysis.analysis1 import select_variant_nearby, gene_power_curves
from utils_analysis.power import percent_change_to_lfc


def test_increase_decrease_have_correct_signed_log_change():
    assert percent_change_to_lfc(25, "decrease") == pytest.approx(np.log(.75))
    assert percent_change_to_lfc(25, "increase") == pytest.approx(np.log(1.25))


def test_variant_uses_only_same_chromosome_and_unique_genes():
    reference = pd.DataFrame({"gene": ["A", "A", "B"], "gene_chr": ["chr1", "chr1", "chr2"],
                              "gene_position": [1050, 1050, 1050], "element": ["E1", "E2", "E3"]})
    found = select_variant_nearby(reference, "chr1", 1000, 100, reference_build="GRCh38", variant_build="GRCh38")
    assert found.gene.tolist() == ["A"]
    assert found.distance_bp.iloc[0] == 50


def test_variant_rejects_build_mismatch():
    with pytest.raises(ValueError, match="genome build"):
        select_variant_nearby(pd.DataFrame(), "chr1", 1000, 100, reference_build="GRCh38", variant_build="GRCh37")


def test_gene_curves_are_specific_to_selected_gene():
    reference = pd.DataFrame({"element": ["E", "E"], "gene": ["A", "B"],
        "gene_chr": ["chr1", "chr1"], "element_chr": ["chr1", "chr1"],
        "distance_bp": [50, 50], "fraction_cells_detected": [1., 1.],
        "reference_expression": [10., .01], "theta": [10., 10.]})
    cells, depth = gene_power_curves(reference, "E", "A", np.array([100, 200]),
                                    np.array([.5, 1.]), 25, "decrease", 200, 1.)
    assert len(cells) == len(depth) == 2
    assert cells.analytical_power.iloc[1] > cells.analytical_power.iloc[0]
