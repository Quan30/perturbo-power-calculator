"""Data Setup must recover a saved fit without a browser session."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def test_refresh_shows_saved_fit_status_and_log(tmp_path, monkeypatch):
    job_dir = tmp_path / "baseline_jobs" / "replogle_example"
    job_dir.mkdir(parents=True)
    (job_dir / "status.json").write_text(json.dumps({"status": "failed", "error": "example failure"}))
    (job_dir / "fit.log").write_text("Starting Replogle baseline\nExample error")
    monkeypatch.setenv("CORTADO_DATA_ROOT", str(tmp_path))

    page = Path(__file__).resolve().parents[1] / "pages" / "0_Data_Setup.py"
    app = AppTest.from_file(str(page)).run()

    assert not app.exception
    assert any(widget.label == "Select a fit job" for widget in app.selectbox)
    assert any("example failure" in message.value for message in app.error)
    assert any("Example error" in output.value for output in app.code)


def test_custom_raw_data_page_exposes_mapping_validator(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTADO_DATA_ROOT", str(tmp_path))
    page = Path(__file__).resolve().parents[1] / "pages" / "0_Data_Setup.py"

    app = AppTest.from_file(str(page)).run()
    app.selectbox[0].select("Custom").run()
    app.radio[0].set_value("Compute from raw data").run()

    assert not app.exception
    labels = [widget.label for widget in app.text_input]
    assert "Custom .h5ad or .h5mu file on server" in labels
    assert any("Choose a data file" in message.value for message in app.info)
    assert any(radio.label == "Where is the raw file?" for radio in app.radio)

    location = next(radio for radio in app.radio if radio.label == "Where is the raw file?")
    location.set_value("Upload from my computer").run()
    assert any(uploader.label == "Upload a custom .h5ad or .h5mu" for uploader in app.file_uploader)


def test_completed_job_does_not_replace_manually_selected_baseline(tmp_path, monkeypatch):
    manual = tmp_path / "manual.npz"
    np.savez(manual, beta_0=[0.0], theta=[1.0], gene_names=["ENSG1"])
    job_dir = tmp_path / "baseline_jobs" / "gasperini_example"
    job_dir.mkdir(parents=True)
    (job_dir / "status.json").write_text(json.dumps({"status": "complete"}))
    np.savez(job_dir / "baseline_params.npz", beta_0=[0.0], theta=[1.0], gene_names=["ENSG2"])
    monkeypatch.setenv("CORTADO_DATA_ROOT", str(tmp_path))

    page = Path(__file__).resolve().parents[1] / "pages" / "0_Data_Setup.py"
    app = AppTest.from_file(str(page))
    app.session_state["gpu_baseline_path"] = str(manual)
    app.session_state["gpu_dataset"] = "Replogle"
    app.run()

    assert not app.exception
    assert app.session_state["gpu_baseline_path"] == str(manual)
    assert app.session_state["gpu_dataset"] == "Replogle"
    assert any(button.label == "Use completed baseline" for button in app.button)


def test_completed_custom_job_activates_as_custom_dataset(tmp_path, monkeypatch):
    job_dir = tmp_path / "baseline_jobs" / "custom_example"
    job_dir.mkdir(parents=True)
    (job_dir / "status.json").write_text(json.dumps({"status": "complete"}))
    np.savez(job_dir / "baseline_params.npz", beta_0=[0.0], theta=[1.0], gene_names=["ENSG1"])
    monkeypatch.setenv("CORTADO_DATA_ROOT", str(tmp_path))

    page = Path(__file__).resolve().parents[1] / "pages" / "0_Data_Setup.py"
    app = AppTest.from_file(str(page)).run()
    next(button for button in app.button if button.label == "Use completed baseline").click().run()

    assert app.session_state["gpu_dataset"] == "Custom"
    assert any("Analysis 2" in message.value for message in app.success)
    assert not any("Analyses 1–2" in message.value for message in app.success)


def test_valid_custom_file_enables_custom_baseline_fit(tmp_path, monkeypatch):
    source = tmp_path / "custom.h5ad"
    ad.AnnData(
        X=np.array([[1, 2], [2, 3]], dtype=float),
        obs=pd.DataFrame(
            {"guide": ["NTC", "OTHER"], "total_counts": [3, 5]},
            index=["c1", "c2"],
        ),
        var=pd.DataFrame(
            {"gene_id": ["E1", "E2"], "symbol": ["A", "B"]},
            index=["v1", "v2"],
        ),
    ).write_h5ad(source)
    monkeypatch.setenv("CORTADO_DATA_ROOT", str(tmp_path))
    page = Path(__file__).resolve().parents[1] / "pages" / "0_Data_Setup.py"

    app = AppTest.from_file(str(page)).run()
    app.selectbox[0].select("Custom").run()
    next(r for r in app.radio if r.label == "Baseline source").set_value("Compute from raw data").run()
    next(w for w in app.text_input if w.label == "Custom .h5ad or .h5mu file on server").set_value(str(source)).run()
    next(r for r in app.radio if r.label == "Control matching rule").set_value("Exact label").run()
    values = {
        "Perturbation column in obs": "guide",
        "Negative-control value": "NTC",
        "Stable gene-ID column in var (optional)": "gene_id",
        "Gene-symbol column in var (optional)": "symbol",
        "Total-counts column in obs (optional)": "total_counts",
    }
    for label, value in values.items():
        next(w for w in app.selectbox if w.label == label).select(value).run()
    assert any("1 control cells" in message.value for message in app.caption)

    assert not app.exception
    next(b for b in app.button if b.label == "Check custom data structure").click().run()
    assert not app.exception
    assert any(b.label == "Compute custom baseline" for b in app.button)


def test_mudata_dropdowns_follow_selected_modality(tmp_path, monkeypatch):
    import mudata as md
    rna = ad.AnnData(X=np.ones((2, 2)), obs=pd.DataFrame({"guide": ["NTC", "OTHER"]}, index=["c1", "c2"]), var=pd.DataFrame({"symbol": ["A", "B"]}, index=["E1", "E2"]))
    rna.layers["raw_counts"] = np.ones((2, 2))
    protein = ad.AnnData(X=np.ones((2, 1)), obs=pd.DataFrame({"assignment": ["CONTROL", "HIT"]}, index=["c1", "c2"]), var=pd.DataFrame(index=["P1"]))
    source = tmp_path / "custom.h5mu"
    md.MuData({"transcriptome": rna, "protein": protein}).write_h5mu(source)
    monkeypatch.setenv("CORTADO_DATA_ROOT", str(tmp_path))
    page = Path(__file__).resolve().parents[1] / "pages" / "0_Data_Setup.py"
    app = AppTest.from_file(str(page)).run()
    app.selectbox[0].select("Custom").run()
    next(r for r in app.radio if r.label == "Baseline source").set_value("Compute from raw data").run()
    next(w for w in app.text_input if w.label == "Custom .h5ad or .h5mu file on server").set_value(str(source)).run()
    next(r for r in app.radio if r.label == "Control matching rule").set_value("Exact label").run()
    next(w for w in app.selectbox if w.label == "RNA modality (MuData only)").select("transcriptome").run()
    assert not app.exception
    assert "Layer: raw_counts" in next(w for w in app.selectbox if w.label == "Raw-count matrix").options
    assert "guide" in next(w for w in app.selectbox if w.label == "Perturbation column in obs").options
    next(w for w in app.selectbox if w.label == "RNA modality (MuData only)").select("protein").run()
    assert not app.exception
    assert "assignment" in next(w for w in app.selectbox if w.label == "Perturbation column in obs").options
    assert "guide" not in next(w for w in app.selectbox if w.label == "Perturbation column in obs").options


def test_layer_named_x_remains_distinct_from_main_matrix(tmp_path, monkeypatch):
    source = tmp_path / "layer_x.h5ad"
    data = ad.AnnData(X=np.array([[0.1, 0.2], [0.3, 0.4]]), obs=pd.DataFrame({"guide": ["NTC", "HIT"]}, index=["c1", "c2"]), var=pd.DataFrame(index=["E1", "E2"]))
    data.layers["X"] = np.array([[1, 2], [3, 4]])
    data.write_h5ad(source)
    monkeypatch.setenv("CORTADO_DATA_ROOT", str(tmp_path))
    page = Path(__file__).resolve().parents[1] / "pages" / "0_Data_Setup.py"
    app = AppTest.from_file(str(page)).run()
    app.selectbox[0].select("Custom").run()
    next(r for r in app.radio if r.label == "Baseline source").set_value("Compute from raw data").run()
    next(w for w in app.text_input if w.label == "Custom .h5ad or .h5mu file on server").set_value(str(source)).run()
    next(r for r in app.radio if r.label == "Control matching rule").set_value("Exact label").run()
    next(w for w in app.selectbox if w.label == "Raw-count matrix").select("X").run()
    next(b for b in app.button if b.label == "Check custom data structure").click().run()
    assert not app.exception
    assert any("counts from layers['X']" in w.value for w in app.success)


def test_pattern_preview_matches_300_controls_in_replogle_example(tmp_path, monkeypatch):
    # Do not depend on an ignored local browser upload in a fresh checkout.
    source = tmp_path / 'many_control_labels.h5ad'
    labels = [f'{i % 214}_non-targeting' for i in range(300)] + ['GATA1'] * 50
    ad.AnnData(X=np.ones((350, 2)), obs=pd.DataFrame({'gene_transcript': labels}, index=[f'c{i}' for i in range(350)]), var=pd.DataFrame(index=['E1', 'E2'])).write_h5ad(source)
    monkeypatch.setenv('CORTADO_DATA_ROOT', str(tmp_path))
    page = Path(__file__).resolve().parents[1] / 'pages' / '0_Data_Setup.py'
    app = AppTest.from_file(str(page)).run()
    app.selectbox[0].select('Custom').run()
    next(r for r in app.radio if r.label == 'Baseline source').set_value('Compute from raw data').run()
    next(w for w in app.text_input if w.label == 'Custom .h5ad or .h5mu file on server').set_value(str(source)).run()
    assert any(r.label == 'Control matching rule' for r in app.radio)
    next(r for r in app.radio if r.label == 'Control matching rule').set_value('Contains text').run()
    next(w for w in app.text_input if w.label == 'Negative-control expression').set_value('non-targeting').run()
    assert not app.exception
    assert any('300 control cells' in w.value for w in app.caption)
    next(b for b in app.button if b.label == 'Check custom data structure').click().run()
    assert any(b.label == 'Compute custom baseline' for b in app.button)
    next(r for r in app.radio if r.label == 'Control matching rule').set_value('Regular expression').run()
    next(w for w in app.text_input if w.label == 'Negative-control expression').set_value('[').run()
    assert not app.exception
    assert any('regular expression' in w.value for w in app.error)
    assert not any(b.label == 'Compute custom baseline' for b in app.button)


def test_job_log_clearly_identifies_selected_job_and_current_session(tmp_path, monkeypatch):
    import json
    jobs=tmp_path/'baseline_jobs'
    paths=[]
    for name,source in [('custom_old','old_input.h5ad'),('custom_current','independent_custom.h5mu')]:
        folder=jobs/name
        folder.mkdir(parents=True)
        (folder/'status.json').write_text(json.dumps({'status':'failed','error':'example'}))
        (folder/'job.json').write_text(json.dumps({'input':f'/data/{source}','dataset':'custom','started_at':'2026-09-30T20:00:00+00:00','options':{'steps':'100'}}))
        (folder/'fit.log').write_text(f'LOG FOR {source}')
        paths.append(folder)
    monkeypatch.setenv('CORTADO_DATA_ROOT',str(tmp_path))
    page=Path(__file__).resolve().parents[1]/'pages/0_Data_Setup.py'
    app=AppTest.from_file(str(page))
    app.session_state['gpu_fit_job_dir']=str(paths[1])
    app.session_state['gpu_job_choice']=paths[1]
    app.run()
    assert not app.exception
    choice=next(w for w in app.selectbox if w.label=='Select a fit job')
    assert any('independent_custom.h5mu' in label for label in choice.options)
    assert any('Most recently requested fit in this session' in w.value for w in app.info)
    assert any('LOG FOR independent_custom.h5mu' in w.value for w in app.code)

    choice.select(paths[0]).run()
    assert any('older job' in w.value.lower() for w in app.info)
    assert any('LOG FOR old_input.h5ad' in w.value for w in app.code)

    next(w for w in app.button if w.label == 'Show latest job').click().run()
    assert not app.exception
    assert next(w for w in app.selectbox if w.label == 'Select a fit job').value == paths[1]
    assert any('LOG FOR independent_custom.h5mu' in w.value for w in app.code)


def test_count_preflight_blocks_normalized_matrix_and_requires_recheck(tmp_path, monkeypatch):
    source = tmp_path / 'preflight.h5ad'
    data = ad.AnnData(X=np.array([[0.1, 0.2], [0.3, 0.4]]), obs=pd.DataFrame({'guide': ['NTC', 'HIT']}, index=['c1', 'c2']))
    data.layers['raw'] = np.array([[1, 2], [3, 4]])
    data.write_h5ad(source)
    monkeypatch.setenv('CORTADO_DATA_ROOT', str(tmp_path))
    page = Path(__file__).resolve().parents[1] / 'pages/0_Data_Setup.py'
    app = AppTest.from_file(str(page)).run()
    app.selectbox[0].select('Custom').run()
    next(w for w in app.radio if w.label == 'Baseline source').set_value('Compute from raw data').run()
    next(w for w in app.text_input if w.label == 'Custom .h5ad or .h5mu file on server').set_value(str(source)).run()
    next(w for w in app.text_input if w.label == 'Negative-control expression').set_value('NTC').run()
    next(w for w in app.button if w.label == 'Check custom data structure').click().run()
    assert any('Structure checked' in w.value for w in app.success)
    assert any('Raw-count sample failed' in w.value for w in app.error)
    assert not any(w.label == 'Compute custom baseline' for w in app.button)
    next(w for w in app.selectbox if w.label == 'Raw-count matrix').select('raw').run()
    assert any('check again' in w.value.lower() for w in app.info)
    next(w for w in app.button if w.label == 'Check custom data structure').click().run()
    assert any('Raw-count sample checked' in w.value for w in app.success)
    assert any(w.label == 'Compute custom baseline' for w in app.button)
    assert any('1 of 2' in w.value for w in app.caption)


def test_reset_setup_clears_selection_but_preserves_completed_jobs(tmp_path, monkeypatch):
    job_dir = tmp_path / 'baseline_jobs/custom_reset'
    job_dir.mkdir(parents=True)
    baseline = job_dir / 'baseline_params.npz'
    np.savez(baseline, beta_0=[0.0], theta=[1.0], gene_names=['A'])
    (job_dir / 'status.json').write_text(json.dumps({'status': 'complete'}))
    (job_dir / 'job.json').write_text(json.dumps({'input': '/data/readable_source.h5ad', 'dataset': 'custom'}))
    monkeypatch.setenv('CORTADO_DATA_ROOT', str(tmp_path))
    page = Path(__file__).resolve().parents[1] / 'pages/0_Data_Setup.py'
    app = AppTest.from_file(str(page))
    app.session_state['gpu_baseline_path'] = str(baseline)
    app.session_state['gpu_dataset'] = 'Custom'
    app.run()
    assert any('Active baseline' in w.value and 'readable_source.h5ad' in w.value and '1 genes' in w.value for w in app.info)
    assert any(w.label == 'Continue to Analysis 2' for w in app.button)
    next(w for w in app.button if w.label == 'Reset this setup').click().run()
    assert not app.exception
    assert 'gpu_baseline_path' not in app.session_state
    assert baseline.is_file()
    assert any(w.label == 'Use completed baseline' for w in app.button)


def test_job_refresh_timer_is_only_enabled_for_running_selected_job(tmp_path, monkeypatch):
    import os
    import streamlit as st
    intervals = []
    def fragment(*, run_every=None):
        intervals.append(run_every)
        return lambda function: function
    monkeypatch.setattr(st, 'fragment', fragment)
    folder = tmp_path / 'baseline_jobs/custom_timer'
    folder.mkdir(parents=True)
    state = folder / 'status.json'
    state.write_text(json.dumps({'status': 'running', 'pid': os.getpid()}))
    monkeypatch.setenv('CORTADO_DATA_ROOT', str(tmp_path))
    page = Path(__file__).resolve().parents[1] / 'pages/0_Data_Setup.py'
    app = AppTest.from_file(str(page)).run()
    assert intervals[-1] == 5
    next(w for w in app.checkbox if w.label.startswith('Automatically refresh')).uncheck().run()
    assert intervals[-1] is None
    state.write_text(json.dumps({'status': 'failed', 'error': 'example'}))
    next(w for w in app.checkbox if w.label.startswith('Automatically refresh')).check().run()
    assert intervals[-1] is None
    assert any('example' in w.value for w in app.error)


def test_builtin_fit_step_defaults_follow_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv('CORTADO_DATA_ROOT', str(tmp_path))
    page = Path(__file__).resolve().parents[1] / 'pages/0_Data_Setup.py'
    app = AppTest.from_file(str(page)).run()
    next(w for w in app.radio if w.label == 'Baseline source').set_value('Compute from raw data').run()
    assert next(w for w in app.number_input if w.label == 'Control fit steps').value == 2500
    app.selectbox[0].select('Replogle').run()
    assert next(w for w in app.number_input if w.label == 'Control fit steps').value == 2000
