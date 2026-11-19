"""Persistent local baseline-fit jobs for a self-hosted Streamlit process."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .baseline_catalog import source_signature, validate_baseline
from .custom_data import custom_mapping_from_options


@dataclass(frozen=True)
class JobInfo:
    status: str
    log_path: Path
    output_path: Path
    pid: int | None = None
    error: str = ""
    input_path: Path | None = None
    dataset: str = ""
    started_at: str = ""
    options: dict[str, str] = field(default_factory=dict)


def _write_state(job_dir: Path, data: dict) -> None:
    temp = job_dir / "status.tmp"
    temp.write_text(json.dumps(data))
    temp.replace(job_dir / "status.json")


def list_jobs(output_dir: Path) -> list[Path]:
    """Find saved fit jobs, newest first, without relying on browser state."""
    root = Path(output_dir)
    if not root.is_dir():
        return []
    states = [path for path in root.glob("*/status.json") if path.is_file()]
    return [path.parent for path in sorted(states, key=lambda p: p.stat().st_mtime_ns, reverse=True)]


def _read_metadata(job_dir: Path, state: dict) -> dict:
    metadata = {}
    if (job_dir / "job.json").exists():
        try:
            metadata = json.loads((job_dir / "job.json").read_text())
        except (ValueError, OSError):
            pass
    metadata.setdefault("input", state.get("input"))
    if not metadata.get("input") and (job_dir / "fit.log").exists():
        with (job_dir / "fit.log").open(errors="replace") as log:
            first = log.readline(4096).strip()
        if first.startswith("Starting ") and " baseline from " in first:
            metadata["input"] = first.split(" baseline from ", 1)[1]
    metadata.setdefault("dataset", job_dir.name.split("_", 1)[0])
    return metadata


def job_source_name(job: JobInfo) -> str:
    if job.input_path is None:
        return "Unknown source (older job)"
    # Browser uploads add an internal fingerprint; display the original name.
    return re.sub(r"^(?:custom|raw)_[0-9a-f]{12}_", "", job.input_path.name)


def job_label(job_dir: Path) -> str:
    """Readable selector text, including compatibility with saved older jobs."""
    try:
        job = read_job(job_dir)
        start = "start time unavailable"
        if job.started_at:
            start = datetime.fromisoformat(job.started_at).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        steps = f" · {job.options['steps']} steps" if job.options.get("steps") else ""
        return f"{job_source_name(job)} · {job.dataset.title()} · {job.status.title()} · {start}{steps} · #{Path(job_dir).name[-6:]}"
    except (OSError, ValueError, KeyError):
        return f"Unreadable saved job · {Path(job_dir).name[-6:]}"


def read_job(job_dir: Path) -> JobInfo:
    job_dir = Path(job_dir)
    state = json.loads((job_dir / "status.json").read_text())
    output_path = job_dir / "baseline_params.npz"
    if state["status"] == "complete":
        try:
            validate_baseline(output_path)
        except (OSError, ValueError, KeyError) as exc:
            state = {**state, "status": "failed", "error": f"Output baseline is invalid: {exc}"}
    elif state["status"] == "running" and state.get("pid"):
        try:
            os.kill(int(state["pid"]), 0)
        except ProcessLookupError:
            state = {**state, "status": "failed", "error": "Worker stopped without producing a result; see fit.log."}
        except PermissionError:
            pass
    metadata = _read_metadata(job_dir, state)
    return JobInfo(state["status"], job_dir / "fit.log", output_path, state.get("pid"), state.get("error", ""), Path(metadata["input"]) if metadata.get("input") else None, metadata.get("dataset", ""), metadata.get("started_at", ""), metadata.get("options", {}))


def start_fit(input_path: Path, output_dir: Path, dataset: str, options: dict[str, str]) -> JobInfo:
    input_path = Path(input_path).expanduser().resolve(strict=True)
    dataset = dataset.lower()
    expected = {
        "gasperini": {".h5mu"},
        "replogle": {".h5ad"},
        "custom": {".h5ad", ".h5mu"},
    }
    if dataset not in expected or input_path.suffix.lower() not in expected[dataset]:
        suffixes = " or ".join(sorted(expected.get(dataset, {".h5ad", ".h5mu"})))
        raise ValueError(f"{dataset} baseline input must be {suffixes}")
    if dataset == "custom":
        custom_mapping_from_options(options)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = json.dumps([source_signature(input_path), dataset, sorted(options.items())]).encode()
    job_dir = output_dir / (dataset + "_" + hashlib.sha256(fingerprint).hexdigest()[:16])
    job_dir.mkdir(parents=True, exist_ok=True)
    with (job_dir / "launch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = job_dir / "status.json"
        if state_path.exists():
            previous = read_job(job_dir)
            if previous.status in {"running", "complete"}:
                return previous
        args = [sys.executable, "-u", str(Path(__file__).resolve().parent.parent / "scripts" / "fit_baseline.py"),
                "--input", str(input_path), "--output-dir", str(job_dir), "--dataset", dataset]
        allowed_options = {
            "steps",
            "max_cells",
            "step_size",
            "rna_modality",
            "counts_layer",
            "perturbation_column",
            "control_value",
            "control_match",
            "gene_id_column",
            "gene_symbol_column",
            "total_counts_column",
        }
        for key, value in sorted(options.items()):
            if key not in allowed_options:
                raise ValueError(f"Unsupported baseline option: {key}")
            flag = "--" + key.replace("_", "-")
            if str(value).startswith("-"):
                args.append(f"{flag}={value}")
            else:
                args.extend([flag, str(value)])
        metadata = {"input": str(input_path), "dataset": dataset, "started_at": datetime.now(timezone.utc).isoformat(), "options": options}
        metadata_temp = job_dir / "job.tmp"
        metadata_temp.write_text(json.dumps(metadata))
        metadata_temp.replace(job_dir / "job.json")
        log_path = job_dir / "fit.log"
        _write_state(job_dir, {"status": "running", "input": str(input_path)})
        with log_path.open("w") as log:
            log.write(f"Starting {dataset.title()} baseline from {input_path}\n")
            log.flush()
            try:
                process = subprocess.Popen(
                    args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
            except Exception as exc:
                _write_state(
                    job_dir,
                    {
                        "status": "failed",
                        "input": str(input_path),
                        "error": f"Could not start baseline worker: {exc}",
                    },
                )
                raise
        if json.loads(state_path.read_text())["status"] == "running":
            _write_state(job_dir, {"status": "running", "pid": process.pid, "input": str(input_path)})
        return read_job(job_dir)
