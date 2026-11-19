import numpy as np
import pytest

from services.baseline_catalog import (
    baseline_catalog_signature,
    discover_baselines,
    order_baseline_choices,
    source_signature,
    validate_baseline,
)


def test_missing_dispersion_is_rejected(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, beta_0=[0.0], gene_names=["ENSG1"])
    with pytest.raises(ValueError, match="theta"):
        validate_baseline(p)


def test_mismatched_gene_arrays_are_rejected(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, beta_0=[0.0, 1.0], theta=[1.0, 1.0], gene_names=["ENSG1"])
    with pytest.raises(ValueError, match="length"):
        validate_baseline(p)


def test_discovery_finds_only_valid_baselines(tmp_path):
    sub = tmp_path / "dataset"
    sub.mkdir()
    np.savez(sub / "good.npz", beta_0=[0.0], theta=[1.0], gene_names=["ENSG1"])
    np.savez(sub / "bad.npz", beta_0=[0.0])
    (sub / "notes.txt").write_text("x")
    assert discover_baselines(tmp_path) == [sub / "good.npz"]
    assert source_signature(sub / "good.npz")[0] == str((sub / "good.npz").resolve())


def test_setup_selected_baseline_stays_first_choice(tmp_path):
    selected = tmp_path / "selected.npz"
    discovered = tmp_path / "discovered.npz"
    default = tmp_path / "default.npz"

    choices = order_baseline_choices(
        preferred=selected,
        discovered=[discovered, selected],
        default=default,
    )

    assert choices == [selected, discovered, default]


def test_catalog_signature_changes_only_with_npz_files(tmp_path):
    baseline = tmp_path / "baseline.npz"
    note = tmp_path / "note.txt"
    np.savez(baseline, beta_0=[0.0], theta=[1.0], gene_names=["ENSG1"])

    original = baseline_catalog_signature(tmp_path)
    note.write_text("unrelated")
    assert baseline_catalog_signature(tmp_path) == original

    np.savez(baseline, beta_0=[0.0, 1.0], theta=[1.0, 1.0], gene_names=["A", "B"])
    assert baseline_catalog_signature(tmp_path) != original
