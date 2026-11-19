from pathlib import Path

from utils_analysis.analysis3 import available_pairs, load_analysis3_artifact


def test_real_partial_simulation_artifact_has_selectable_pairs_and_provisional_design():
    path = Path(__file__).resolve().parent.parent / "artifacts" / "analysis3_pair_power.parquet"
    frame = load_analysis3_artifact(path)
    assert len(frame) == 148176
    assert frame["pair_id"].nunique() == 7056
    pairs = available_pairs(frame, "Gasperini", "cis")
    assert len(pairs) == 7056
    assert (frame["n_replicates"] == 29).any()
    assert "expected_replicates" in frame and (frame["expected_replicates"] == 30).all()
