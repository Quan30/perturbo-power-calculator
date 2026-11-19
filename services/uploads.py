"""Safe, chunked persistence for Streamlit dataset uploads."""
from __future__ import annotations

import hashlib
from pathlib import Path


def save_uploaded_dataset(
    uploaded,
    upload_dir: Path,
    allowed_suffixes: set[str],
    prefix: str,
) -> Path:
    """Save an uploaded dataset atomically and return its server-local path."""
    basename = Path(uploaded.name).name
    suffix = Path(basename).suffix.lower()
    normalized_allowed = {value.lower() for value in allowed_suffixes}
    if suffix not in normalized_allowed:
        expected = ", ".join(sorted(normalized_allowed))
        raise ValueError(f"Unsupported upload type {suffix or '(none)'}; expected {expected}")

    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    identity = f"{uploaded.name}|{uploaded.size}|{getattr(uploaded, 'file_id', '')}".encode()
    fingerprint = hashlib.sha256(identity).hexdigest()[:12]
    destination = upload_dir / f"{prefix}_{fingerprint}_{basename}"
    staging = destination.with_suffix(destination.suffix + ".part")
    uploaded.seek(0)
    with staging.open("wb") as dest:
        while chunk := uploaded.read(8 * 1024 * 1024):
            dest.write(chunk)
    staging.replace(destination)
    return destination
