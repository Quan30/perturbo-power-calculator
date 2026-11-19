from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scripts.fit_baseline import atomic_save_baseline, fit_custom


def test_atomic_baseline_output_preserves_previous_valid_file_on_failure(tmp_path):
    target = tmp_path / "baseline_params.npz"
    atomic_save_baseline(target, {
        "beta_0": np.array([0., 1.]), "theta": np.array([2., 3.]),
        "gene_names": np.array(["A", "B"]),
    })
    with pytest.raises(ValueError, match="inconsistent length"):
        atomic_save_baseline(target, {
            "beta_0": np.array([2.]), "theta": np.array([1., 2.]),
            "gene_names": np.array(["A", "B"]),
        })
    with np.load(target) as data:
        assert data["beta_0"].tolist() == [0., 1.]
    assert not (tmp_path / "baseline_params.tmp.npz").exists()


def test_custom_fit_writes_dataset_neutral_baseline(tmp_path):
    source = tmp_path / "custom.h5ad"
    output = tmp_path / "baseline_params.npz"
    ad.AnnData(
        X=np.array([[1, 2], [4, 5], [2, 3]], dtype=float),
        obs=pd.DataFrame(
            {"guide": ["NTC", "OTHER", "NTC"], "total": [3, 9, 5]},
            index=["c1", "c2", "c3"],
        ),
        var=pd.DataFrame(
            {"gene_id": ["ENSG1", "ENSG2"], "symbol": ["A", "B"]},
            index=["v1", "v2"],
        ),
    ).write_h5ad(source)
    captured = {}

    def make_data(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    runtime = SimpleNamespace(
        asarray=np.asarray,
        CortadoData=make_data,
        SVIConfig=lambda **kwargs: kwargs,
        compute_gene_clip_thresholds=lambda counts, percentile: np.full(counts.shape[1], 100),
        winsorize_counts_to_gene_thresholds=lambda counts, thresholds: counts,
        fit_control=lambda data, **kwargs: SimpleNamespace(
            beta_0=np.array([0.1, 0.2]),
            theta=np.array([2.0, 3.0]),
            size_factors=data.size_factors,
        ),
    )

    fit_custom(
        source,
        output,
        {
            "max_cells": "10",
            "steps": "5",
            "perturbation_column": "guide",
            "control_value": "NTC",
            "gene_id_column": "gene_id",
            "gene_symbol_column": "symbol",
            "total_counts_column": "total",
        },
        runtime=runtime,
    )

    assert captured["counts"].shape == (2, 2)
    assert captured["pert_names"] == ["control"]
    with np.load(output) as baseline:
        assert baseline["gene_ids"].tolist() == ["ENSG1", "ENSG2"]
        assert baseline["gene_names"].tolist() == ["A", "B"]
