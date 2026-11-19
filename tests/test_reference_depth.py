import numpy as np
import pandas as pd
import pytest

from utils_analysis.reference_data import (
    counts_to_depth_multiplier,
    reference_depth_from_gene_reference,
    reference_depth_counts,
)


def _write_baseline(path, beta_0):
    beta_0 = np.asarray(beta_0, dtype=float)
    np.savez(
        path,
        beta_0=beta_0,
        theta=np.ones_like(beta_0),
        gene_names=np.asarray([f"g{i}" for i in range(len(beta_0))], dtype=object),
        size_factors=np.zeros(4),
    )


def test_reference_depth_is_sum_of_baseline_gene_means(tmp_path):
    path = tmp_path / "baseline.npz"
    _write_baseline(path, np.log([2.0, 3.0, 5.0]))

    assert reference_depth_counts(path) == pytest.approx(10.0)


def test_reference_counts_convert_to_one_x_depth():
    assert counts_to_depth_multiplier(16_620, 16_620) == pytest.approx(1.0)
    assert counts_to_depth_multiplier(8_310, 16_620) == pytest.approx(0.5)


def test_reference_depth_reuses_loaded_gene_reference():
    reference = pd.DataFrame({"reference_expression": [2.0, 3.0, 5.0]})
    assert reference_depth_from_gene_reference(reference) == pytest.approx(10.0)


@pytest.mark.parametrize("counts, reference", [(0, 100), (-1, 100), (100, 0), (100, np.nan)])
def test_depth_conversion_rejects_nonpositive_or_nonfinite_values(counts, reference):
    with pytest.raises(ValueError, match="positive"):
        counts_to_depth_multiplier(counts, reference)


def test_pages_present_absolute_counts_not_relative_percentage():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    for relative in ["pages/analysis1_cis_power.py", "pages/analysis2_target_program.py"]:
        text = (root / relative).read_text(encoding="utf-8")
        assert "Expected counts per cell" in text
        assert '"Relative sequencing depth (%)"' not in text
