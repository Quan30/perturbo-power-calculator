from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_changing_default_does_not_reuse_old_path_override(tmp_path):
    first = tmp_path / 'first.npz'
    second = tmp_path / 'second.npz'
    first.touch()
    second.touch()
    script = tmp_path / 'picker.py'
    script.write_text('from pathlib import Path\nimport streamlit as st\nfrom services.file_selection import choose_file\nselected = choose_file("Reference", st.session_state["default"], {".npz"}, Path("/tmp"), key="test_reference")\nst.session_state["chosen"] = selected\n')
    app = AppTest.from_file(str(script))
    app.session_state['default'] = str(first)
    app.run()
    assert app.session_state['chosen'] == str(first)
    app.session_state['default'] = str(second)
    app.run()
    assert app.session_state['chosen'] == str(second)


def test_analysis2_dataset_switch_selects_matching_bundled_baseline():
    page = Path(__file__).resolve().parents[1] / 'pages' / 'analysis2_target_program.py'
    app = AppTest.from_file(str(page), default_timeout=30).run()
    assert not app.exception
    assert next(w for w in app.selectbox if w.label == 'Baseline file in folder').value.name == 'replogle_baseline_params.npz'
    next(w for w in app.selectbox if w.label == 'Baseline dataset').select('Gasperini').run()
    assert not app.exception
    assert next(w for w in app.selectbox if w.label == 'Baseline file in folder').value.name == 'gasperini_baseline_params.npz'


def test_analysis2_summary_follows_custom_baseline_and_keeps_details_available(tmp_path, monkeypatch):
    import numpy as np
    first = tmp_path / 'first.npz'
    second = tmp_path / 'second.npz'
    for path, names in [(first, ['Demo_A', 'Demo_B']), (second, ['Demo_A', 'Demo_B', 'Demo_C'])]:
        np.savez(path, beta_0=np.zeros(len(names)), theta=np.ones(len(names)), gene_names=names, reference_depth_counts=1000.)
    monkeypatch.setenv('CORTADO_DATA_ROOT', str(tmp_path))
    page = Path(__file__).resolve().parents[1] / 'pages/analysis2_target_program.py'
    app = AppTest.from_file(str(page), default_timeout=30)
    app.session_state['gpu_dataset'] = 'Custom'
    app.session_state['gpu_baseline_path'] = str(first)
    app.run()
    assert not app.exception
    assert any('Active baseline' in w.value and 'first.npz' in w.value and '2 genes' in w.value for w in app.info)
    next(w for w in app.selectbox if w.label == 'Baseline file in folder').select(second).run()
    assert any('Active baseline' in w.value and 'second.npz' in w.value and '3 genes' in w.value for w in app.info)
    next(w for w in app.radio if w.label == 'Target program source').set_value('Custom gene list').run()
    next(w for w in app.text_area if w.label == 'Gene symbols or Ensembl IDs').set_value('Demo_A').run()
    assert not app.exception
    main = next(w for w in app.dataframe if 'Analytical power (%)' in w.value.columns)
    assert 'NB dispersion (θ)' not in main.value.columns
    details = next(w for w in app.dataframe if 'Matched by' in w.value.columns)
    assert 'Stable gene ID' in details.value.columns
