from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from utils_analysis.analysis1 import evaluate_cis_element, evaluate_cis_dataset
from utils_analysis.analysis3 import summarize_power_by_expression_decile
from services.interface import preferred_simulation_dataset, median_design_value


def test_expression_quantiles_keep_ties_together():
    frame = pd.DataFrame({'baseline_expression': [1.] * 15 + [2.] * 10 + [3.] * 10,
                          'empirical_power': [.8] * 35})
    bins = summarize_power_by_expression_decile(frame, .8)
    assert bins.n_pairs.sum() == 35
    assert len(bins) < 10
    assert bins.expression_label.notna().all()
    for expression in [1., 2., 3.]:
        assert ((bins.expression_min <= expression) & (bins.expression_max >= expression)).sum() == 1


def test_constant_expression_is_one_bin():
    bins = summarize_power_by_expression_decile(pd.DataFrame({
        'baseline_expression': [2.] * 20, 'empirical_power': [.5] * 20}), .8)
    assert len(bins) == 1
    assert bins.n_pairs.iloc[0] == 20


def test_dataset_evaluation_matches_each_eligible_element():
    reference = pd.DataFrame({'element': ['A', 'B', 'NTC'], 'gene': ['G1', 'G2', 'G3'],
        'element_chr': ['chr1'] * 3, 'gene_chr': ['chr1'] * 3,
        'distance_bp': [10., 20., 30.], 'fraction_cells_detected': [1.] * 3,
        'reference_expression': [1., 2., 3.], 'theta': [5.] * 3})
    kwargs = dict(window_bp=100., effect_percent=50., cells_per_guide=200,
                  depth_multiplier=1., target_power=.8)
    all_pairs = evaluate_cis_dataset(reference, ['A', 'B'], **kwargs)
    separate = pd.concat([evaluate_cis_element(reference, e, **kwargs) for e in ['A', 'B']])
    assert set(all_pairs.element) == {'A', 'B'}
    assert np.allclose(all_pairs.sort_values('element').analytical_power,
                       separate.sort_values('element').analytical_power)


def test_defaults_use_available_dataset_and_design_median():
    assert preferred_simulation_dataset(['Gasperini', 'Replogle']) == 'Replogle'
    assert preferred_simulation_dataset(['Gasperini']) == 'Gasperini'
    assert median_design_value([100, 200, 300, 500, 900]) == 300
    assert median_design_value([100, 200, 500, 900]) == 200


def test_observed_significance_counts_only_joined_unique_pairs():
    from services.observed_pairs import observed_significance_by_gene
    candidates = pd.DataFrame({'element': ['A', 'B', 'C'], 'gene': ['G1', 'G1', 'G2']})
    observed = pd.DataFrame({'element': ['A', 'B', 'D'], 'gene': ['G1', 'G1', 'G9'],
                             'empirical_fdr': [.01, .2, .01]})
    counts = observed_significance_by_gene(candidates, observed, .1).set_index('gene')
    assert counts.loc['G1', 'observed_pairs'] == 2
    assert counts.loc['G1', 'significant_pairs'] == 1
    assert counts.loc['G2', 'observed_pairs'] == 0
    assert 'G9' not in counts.index
    with pytest.raises(ValueError, match='duplicate'):
        observed_significance_by_gene(candidates, pd.concat([observed, observed]), .1)
    observed.loc[0, 'empirical_fdr'] = -1
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        observed_significance_by_gene(candidates, observed, .1)


def test_home_cards_and_analysis2_defaults():
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    home = AppTest.from_file(str(root / 'app.py')).run()
    assert not home.exception
    assert [w.label for w in home.button] == ['Data Setup', 'Open Analysis 3', 'Open Analysis 2', 'Open Analysis 1']
    assert not home.expander
    app = AppTest.from_file(str(root / 'pages/analysis2_target_program.py'), default_timeout=60).run()
    assert not app.exception
    assert next(w for w in app.slider if w.label == 'Assumed expression decrease (%)').value == 50
    assert next(w for w in app.number_input if w.label == 'Minimum reference expression').value == .1
    assert next(w for w in app.number_input if w.label == 'Minimum supporting ChIP peaks').value == 1
    assert not app.sidebar.selectbox
    assert any(w.label == 'Gene to mark on expression plot' for w in app.selectbox)


def test_analysis3_pair_load_retains_explicit_loaded_identity():
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / 'pages/analysis3_screen_power.py'), default_timeout=60).run()
    assert not app.exception
    assert not any('Loaded pair:' in w.value for w in app.markdown)
    next(w for w in app.button if w.label == 'Load selected pair').click().run()
    loaded_label = next(w.value for w in app.markdown if 'Loaded pair:' in w.value)
    picker = next(w for w in app.selectbox if w.label == 'Enhancer / perturbation')
    picker.set_value(picker.options[1]).run()
    assert not app.exception
    assert next(w.value for w in app.markdown if 'Loaded pair:' in w.value) == loaded_label
    assert any('selection has changed' in w.value for w in app.info)
    next(w for w in app.button if w.label == 'Load selected pair').click().run()
    assert next(w.value for w in app.markdown if 'Loaded pair:' in w.value) != loaded_label
    next(w for w in app.selectbox if w.label == 'Cells per guide').set_value(100).run()
    assert not app.exception
    assert any('100 cells/guide' in w.value for w in app.caption)


def test_analysis1_detail_follows_gene_selected_in_overview():
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / 'pages/analysis1_cis_power.py'), default_timeout=60).run()
    assert not app.exception
    chosen = next(w for w in app.selectbox if w.label == 'Gene to inspect')
    assert next(w for w in app.selectbox if w.label == 'Gene').value == chosen.value
    chosen.select_index(1).run()
    assert not app.exception
    assert next(w for w in app.selectbox if w.label == 'Gene').value == next(w for w in app.selectbox if w.label == 'Gene to inspect').value
