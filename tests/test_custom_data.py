from dataclasses import asdict
from types import SimpleNamespace

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from services.custom_data import (
    CustomDataMapping,
    custom_mapping_from_options,
    custom_mapping_to_options,
    prepare_custom_baseline_data,
    validate_custom_data_file,
    validate_custom_data_structure,
)


def _adata():
    return SimpleNamespace(
        obs=pd.DataFrame(
            {
                "perturbation": ["NTC", "GATA1"],
                "total_counts": [1000, 1200],
            }
        ),
        var=pd.DataFrame({"gene_symbol": ["A", "B"]}, index=["ENSG1", "ENSG2"]),
        var_names=pd.Index(["ENSG1", "ENSG2"]),
        layers={"counts": object()},
        X=object(),
        n_obs=2,
        n_vars=2,
    )


def test_custom_anndata_mapping_reports_compatible_structure():
    report = validate_custom_data_structure(
        _adata(),
        CustomDataMapping(
            counts_layer="counts",
            perturbation_column="perturbation",
            control_value="NTC",
            gene_id_column=None,
            gene_symbol_column="gene_symbol",
            total_counts_column="total_counts",
        ),
    )

    assert report.n_cells == 2
    assert report.n_genes == 2
    assert report.count_source == "layers['counts']"
    assert report.gene_id_source == "var_names"


def test_custom_mapping_rejects_missing_control_label():
    with pytest.raises(ValueError, match="control value 'non-targeting'"):
        validate_custom_data_structure(
            _adata(),
            CustomDataMapping(
                counts_layer="counts",
                perturbation_column="perturbation",
                control_value="non-targeting",
            ),
        )


def test_custom_mudata_requires_and_uses_rna_modality():
    mdata = SimpleNamespace(mod={"rna": _adata()})
    mapping = CustomDataMapping(
        rna_modality="rna",
        counts_layer=None,
        perturbation_column="perturbation",
        control_value="NTC",
    )

    report = validate_custom_data_structure(mdata, mapping)
    assert report.rna_modality == "rna"
    assert report.count_source == "X"


@pytest.mark.parametrize("missing_id", [None, pd.NA])
def test_custom_mapping_rejects_missing_stable_gene_ids(missing_id):
    adata = _adata()
    adata.var_names = pd.Index(["ENSG1", missing_id])

    with pytest.raises(ValueError, match="nonempty"):
        validate_custom_data_structure(
            adata,
            CustomDataMapping(
                counts_layer="counts",
                perturbation_column="perturbation",
                control_value="NTC",
            ),
        )


def test_custom_mapping_job_options_round_trip():
    mapping = CustomDataMapping(
        rna_modality="rna",
        counts_layer="counts",
        perturbation_column="guide",
        control_value="NTC",
        gene_id_column="ensembl_id",
        gene_symbol_column="gene_name",
        total_counts_column="total_counts",
    )
    restored = custom_mapping_from_options(custom_mapping_to_options(mapping))
    assert asdict(restored) == asdict(mapping)


def test_prepare_custom_anndata_uses_only_controls_and_raw_count_layer(tmp_path):
    adata = ad.AnnData(
        X=np.log1p(np.array([[1, 2], [3, 4], [5, 6]], dtype=float)),
        obs=pd.DataFrame(
            {"guide": ["NTC", "GATA1", "NTC"], "total_counts": [3, 7, 11]},
            index=["c1", "c2", "c3"],
        ),
        var=pd.DataFrame(
            {"ensembl_id": ["ENSG1", "ENSG2"], "gene_name": ["A", "B"]},
            index=["v1", "v2"],
        ),
    )
    adata.layers["counts"] = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    path = tmp_path / "custom.h5ad"
    adata.write_h5ad(path)
    mapping = CustomDataMapping(
        counts_layer="counts",
        perturbation_column="guide",
        control_value="NTC",
        gene_id_column="ensembl_id",
        gene_symbol_column="gene_name",
        total_counts_column="total_counts",
    )

    report = validate_custom_data_file(path, mapping)
    prepared = prepare_custom_baseline_data(path, mapping, max_cells=10)

    assert report.n_cells == 3
    assert prepared.counts.tolist() == [[1.0, 2.0], [5.0, 6.0]]
    assert prepared.library_sizes.tolist() == [3.0, 11.0]
    assert prepared.gene_ids.tolist() == ["ENSG1", "ENSG2"]
    assert prepared.gene_names.tolist() == ["A", "B"]


def test_prepare_custom_mudata_uses_declared_rna_modality(tmp_path):
    rna = ad.AnnData(
        X=np.array([[1, 0], [2, 3]], dtype=float),
        obs=pd.DataFrame({"guide": ["NTC", "OTHER"]}, index=["c1", "c2"]),
        var=pd.DataFrame(index=["ENSG1", "ENSG2"]),
    )
    path = tmp_path / "custom.h5mu"
    md.MuData({"rna": rna}).write_h5mu(path)
    mapping = CustomDataMapping(
        rna_modality="rna",
        perturbation_column="guide",
        control_value="NTC",
    )

    prepared = prepare_custom_baseline_data(path, mapping, max_cells=10)
    assert prepared.counts.tolist() == [[1.0, 0.0]]
    assert prepared.library_sizes.tolist() == [1.0]


def test_prepare_custom_data_rejects_normalized_matrix(tmp_path):
    adata = ad.AnnData(
        X=np.array([[0.1, 1.2]], dtype=float),
        obs=pd.DataFrame({"guide": ["NTC"]}, index=["c1"]),
        var=pd.DataFrame(index=["ENSG1", "ENSG2"]),
    )
    path = tmp_path / "normalized.h5ad"
    adata.write_h5ad(path)

    with pytest.raises(ValueError, match="raw integer counts"):
        prepare_custom_baseline_data(
            path,
            CustomDataMapping(perturbation_column="guide", control_value="NTC"),
            max_cells=10,
        )


def test_validation_and_fit_use_the_same_exact_control_value():
    with pytest.raises(ValueError, match="control value ' NTC '"):
        validate_custom_data_structure(
            _adata(),
            CustomDataMapping(
                counts_layer="counts",
                perturbation_column="perturbation",
                control_value=" NTC ",
            ),
        )


def test_prepare_custom_backed_sparse_matrix_without_materializing_other_layers(tmp_path, monkeypatch):
    adata = ad.AnnData(
        X=sparse.csr_matrix([[1, 0], [2, 3], [4, 1]], dtype=np.float32),
        obs=pd.DataFrame({"guide": ["NTC", "OTHER", "NTC"]}, index=["c1", "c2", "c3"]),
        var=pd.DataFrame(index=["ENSG1", "ENSG2"]),
    )
    adata.layers["unused"] = sparse.csr_matrix(np.ones((3, 2), dtype=np.float32))
    path = tmp_path / "sparse.h5ad"
    adata.write_h5ad(path)

    def reject_whole_object_materialization(self, *args, **kwargs):
        raise AssertionError("prepare should read only the selected matrix")

    monkeypatch.setattr(ad.AnnData, "to_memory", reject_whole_object_materialization)

    prepared = prepare_custom_baseline_data(
        path,
        CustomDataMapping(perturbation_column="guide", control_value="NTC"),
        max_cells=10,
    )
    assert prepared.counts.tolist() == [[1.0, 0.0], [4.0, 1.0]]


def test_custom_control_label_survives_job_serialization_exactly():
    mapping = CustomDataMapping(perturbation_column="guide", control_value=" NTC ")
    restored = custom_mapping_from_options(custom_mapping_to_options(mapping))
    assert restored.control_value == " NTC "


def test_contains_and_regex_control_matching_exclude_missing_labels():
    from services import custom_data
    assert hasattr(custom_data, 'control_cell_mask')
    labels = pd.Series(['1_non-targeting_non-targeting', '2_non-targeting', 'GATA1', None])
    assert custom_data.control_cell_mask(labels, 'non-targeting', 'contains').tolist() == [True, True, False, False]
    assert custom_data.control_cell_mask(labels, r'^\d+_non-targeting', 'regex').tolist() == [True, True, False, False]
    assert custom_data.control_cell_mask(labels, 'non-targeting', 'exact').tolist() == [False, False, False, False]


def test_control_patterns_reject_empty_invalid_regex_and_unknown_modes():
    from services import custom_data
    assert hasattr(custom_data, 'control_cell_mask')
    for pattern, mode, message in [('', 'contains', 'nonempty'), ('[', 'regex', 'regular expression'), ('NTC', 'guess', 'matching mode')]:
        with pytest.raises(ValueError, match=message):
            custom_data.control_cell_mask(pd.Series(['NTC']), pattern, mode)


def test_pattern_mapping_round_trip_and_control_only_preparation(tmp_path):
    data = ad.AnnData(X=np.array([[1, 2], [3, 4], [5, 6]]), obs=pd.DataFrame({'guide': ['1_NTC', 'GATA1', '2_NTC']}, index=['c1','c2','c3']), var=pd.DataFrame(index=['E1','E2']))
    source = tmp_path/'patterns.h5ad'
    data.write_h5ad(source)
    mapping = CustomDataMapping(perturbation_column='guide', control_value='NTC', control_match='contains')
    restored = custom_mapping_from_options(custom_mapping_to_options(mapping))
    assert restored == mapping
    report = validate_custom_data_file(source, restored)
    assert report.n_control_cells == 2
    prepared = prepare_custom_baseline_data(source, restored, max_cells=10)
    assert prepared.counts.tolist() == [[1,2],[5,6]]


@pytest.mark.parametrize('encoding', ['dense', 'csr', 'csc'])
def test_count_sample_reads_only_selected_controls_and_layer(tmp_path, encoding):
    from services.custom_data import check_custom_count_sample
    raw = np.array([[1, 2], [0.25, 0.5], [3, 4]], dtype=float)
    matrix = raw if encoding == 'dense' else getattr(sparse, f'{encoding}_matrix')(raw)
    data = ad.AnnData(X=np.log1p(raw), obs=pd.DataFrame({'guide': ['NTC', 'HIT', 'NTC']}, index=['c1', 'c2', 'c3']))
    data.layers['raw'] = matrix
    source = tmp_path / f'{encoding}.h5ad'
    data.write_h5ad(source)
    mapping = CustomDataMapping(perturbation_column='guide', control_value='NTC', counts_layer='raw')
    assert check_custom_count_sample(source, mapping, max_cells=1) == 1
    assert check_custom_count_sample(source, mapping) == 2
    with pytest.raises(ValueError, match='raw integer counts'):
        check_custom_count_sample(source, CustomDataMapping(perturbation_column='guide', control_value='NTC'))


def test_count_sample_uses_mudata_modality_and_rejects_invalid_totals(tmp_path):
    from services.custom_data import check_custom_count_sample
    rna = ad.AnnData(X=np.array([[1, 2], [3, 4]]), obs=pd.DataFrame({'guide': ['NTC', 'HIT'], 'total': [-1, 7]}, index=['c1', 'c2']))
    source = tmp_path / 'sample.h5mu'
    md.MuData({'transcripts': rna}).write_h5mu(source)
    mapping = CustomDataMapping(rna_modality='transcripts', perturbation_column='guide', control_value='NTC')
    assert check_custom_count_sample(source, mapping) == 1
    with pytest.raises(ValueError, match='Total counts'):
        check_custom_count_sample(source, CustomDataMapping(**{**asdict(mapping), 'total_counts_column': 'total'}))


def test_zero_count_sample_does_not_reject_unsampled_positive_controls(tmp_path):
    from services.custom_data import check_custom_count_sample
    counts = np.zeros((101, 2))
    counts[99] = [1, 2]  # omitted by a 100-row evenly spaced sample
    data = ad.AnnData(X=counts, obs=pd.DataFrame({'guide': ['NTC'] * 101}, index=[f'c{i}' for i in range(101)]))
    source = tmp_path / 'zeros.h5ad'
    data.write_h5ad(source)
    mapping = CustomDataMapping(perturbation_column='guide', control_value='NTC')
    assert check_custom_count_sample(source, mapping) == 100
    assert prepare_custom_baseline_data(source, mapping, max_cells=101).library_sizes.sum() == 3
