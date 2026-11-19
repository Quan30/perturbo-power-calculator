import json
import subprocess
import os
from pathlib import Path

import pytest

from services.baseline_jobs import list_jobs, read_job, start_fit


class FakeProcess:
    pid = 123


def test_second_click_reuses_running_job(tmp_path, monkeypatch):
    source = tmp_path / "raw.h5mu"
    source.write_bytes(b"raw data location")
    launches = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: launches.append(args) or FakeProcess())
    monkeypatch.setattr(os, "kill", lambda pid, signal: None)
    first = start_fit(source, tmp_path / "jobs", "gasperini", {"steps": "10"})
    again = start_fit(source, tmp_path / "jobs", "gasperini", {"steps": "10"})
    assert first.output_path == again.output_path
    assert first.status == again.status == "running"
    assert len(launches) == 1


def test_failure_status_is_reported(tmp_path, monkeypatch):
    source = tmp_path / "raw.h5mu"
    source.write_bytes(b"raw data location")
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(os, "kill", lambda pid, signal: None)
    job = start_fit(source, tmp_path / "jobs", "gasperini", {})
    status_file = job.output_path.parent / "status.json"
    status_file.write_text(json.dumps({"status": "failed", "error": "invalid modality", "pid": 123}))
    assert read_job(job.output_path.parent).status == "failed"
    assert "invalid modality" in read_job(job.output_path.parent).error


def test_wrong_extension_never_launches(tmp_path):
    source = tmp_path / "raw.txt"
    source.write_text("x")
    with pytest.raises(ValueError, match="h5mu"):
        start_fit(source, tmp_path / "jobs", "gasperini", {})


def test_disappeared_worker_reports_failure(tmp_path, monkeypatch):
    source = tmp_path / "raw.h5ad"
    source.write_bytes(b"x")
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    job = start_fit(source, tmp_path / "jobs", "replogle", {})
    def gone(pid, signal):
        raise ProcessLookupError()
    monkeypatch.setattr(os, "kill", gone)
    assert read_job(job.output_path.parent).status == "failed"


def test_saved_jobs_are_visible_after_browser_session_is_lost(tmp_path):
    jobs = tmp_path / "baseline_jobs"
    older = jobs / "replogle_old"
    newer = jobs / "gasperini_new"
    for folder in (older, newer):
        folder.mkdir(parents=True)
        (folder / "status.json").write_text('{"status": "failed"}')
    os.utime(older / "status.json", (100, 100))
    os.utime(newer / "status.json", (200, 200))
    (jobs / "unstarted").mkdir()

    assert list_jobs(jobs) == [newer, older]


def test_job_log_starts_immediately_and_worker_is_unbuffered(tmp_path, monkeypatch):
    source = tmp_path / "raw.h5ad"
    source.write_bytes(b"raw")
    launched = []
    monkeypatch.setattr(subprocess, "Popen", lambda args, **kwargs: launched.append(args) or FakeProcess())
    monkeypatch.setattr(os, "kill", lambda pid, signal: None)

    job = start_fit(source, tmp_path / "jobs", "replogle", {"steps": "100"})

    assert launched[0][1] == "-u"
    assert "Starting Replogle baseline" in job.log_path.read_text()


def test_custom_job_accepts_mapping_options_and_h5ad(tmp_path, monkeypatch):
    source = tmp_path / "custom.h5ad"
    source.write_bytes(b"raw")
    launched = []
    monkeypatch.setattr(subprocess, "Popen", lambda args, **kwargs: launched.append(args) or FakeProcess())
    monkeypatch.setattr(os, "kill", lambda pid, signal: None)

    job = start_fit(
        source,
        tmp_path / "jobs",
        "custom",
        {
            "steps": "100",
            "perturbation_column": "guide",
            "control_value": "NTC",
            "control_match": "contains",
            "counts_layer": "counts",
        },
    )

    assert job.status == "running"
    assert "--perturbation-column" in launched[0]
    assert "guide" in launched[0]
    assert "--counts-layer" in launched[0]
    assert launched[0][launched[0].index("--control-match") + 1] == "contains"


def test_worker_launch_failure_is_saved_as_failed_job(tmp_path, monkeypatch):
    source = tmp_path / "raw.h5ad"
    source.write_bytes(b"raw")

    def fail_to_launch(*args, **kwargs):
        raise OSError("cannot execute worker")

    monkeypatch.setattr(subprocess, "Popen", fail_to_launch)
    with pytest.raises(OSError, match="cannot execute worker"):
        start_fit(source, tmp_path / "jobs", "replogle", {})

    job_dir = next((tmp_path / "jobs").iterdir())
    job = read_job(job_dir)
    assert job.status == "failed"
    assert "cannot execute worker" in job.error


def test_dash_prefixed_expression_reaches_worker_parser(tmp_path, monkeypatch):
    import sys
    from scripts import fit_baseline
    source = tmp_path / 'custom.h5ad'
    source.write_bytes(b'raw')
    launches = []
    monkeypatch.setattr(subprocess, 'Popen', lambda args, **kwargs: launches.append(args) or FakeProcess())
    monkeypatch.setattr(os, 'kill', lambda pid, signal: None)
    job = start_fit(source, tmp_path / 'jobs', 'custom', {'perturbation_column': 'guide', 'control_value': '-targeting', 'control_match': 'contains'})
    received = []
    monkeypatch.setattr(fit_baseline, 'fit_custom', lambda source, output, options: received.append(options))
    monkeypatch.setattr(sys, 'argv', launches[0][2:])
    fit_baseline.main()
    assert received[0]['control_value'] == '-targeting'
    assert received[0]['control_match'] == 'contains'


def test_finished_job_retains_source_and_settings_when_worker_replaces_status(tmp_path, monkeypatch):
    source = tmp_path / 'independent_custom.h5mu'
    source.write_bytes(b'raw')
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(os, 'kill', lambda pid, signal: None)
    job = start_fit(source, tmp_path / 'jobs', 'custom', {'perturbation_column': 'exposure_tag', 'control_value': 'sham', 'control_match': 'contains', 'steps': '100'})
    folder=job.output_path.parent
    (folder/'status.json').write_text('{"status":"failed","error":"example"}')
    saved=read_job(folder)
    assert saved.input_path == source.resolve()
    assert saved.dataset == 'custom'
    assert saved.options['steps'] == '100'
    assert saved.started_at


def test_legacy_job_label_recovers_filename_from_first_log_line(tmp_path):
    from services import baseline_jobs
    folder=tmp_path/'custom_abcd'
    folder.mkdir()
    (folder/'status.json').write_text('{"status":"failed","error":"example"}')
    (folder/'fit.log').write_text('Starting Custom baseline from /server/data/custom_123456abcdef_independent_custom.h5ad\nWorker output')
    assert hasattr(baseline_jobs, 'job_label')
    label=baseline_jobs.job_label(folder)
    assert 'independent_custom.h5ad' in label
    assert 'Failed' in label
    assert 'custom_123456abcdef' not in label
