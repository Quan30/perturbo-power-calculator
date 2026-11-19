"""Locate and validate fitted Cortado baselines on the app server."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BaselineInfo:
    path: Path
    n_genes: int
    has_gene_ids: bool


def source_signature(path: Path) -> tuple[str, int, int]:
    resolved = Path(path).expanduser().resolve(strict=True)
    stat = resolved.stat()
    return str(resolved), stat.st_mtime_ns, stat.st_size


def baseline_catalog_signature(root: Path) -> tuple[tuple[str, int, int], ...]:
    """Build a cheap cache key without opening any baseline arrays."""
    root = Path(root).expanduser()
    if not root.is_dir():
        return ()
    signatures = []
    for path in sorted(root.rglob("*.npz")):
        try:
            signatures.append(source_signature(path))
        except OSError:
            continue
    return tuple(signatures)


def order_baseline_choices(
    preferred: Path | None,
    discovered: list[Path],
    default: Path | None = None,
) -> list[Path]:
    """Return unique baseline choices without overriding the user's selection."""
    ordered: list[Path] = []
    for candidate in [preferred, *discovered, default]:
        if candidate is None:
            continue
        path = Path(candidate)
        if path not in ordered:
            ordered.append(path)
    return ordered


def validate_baseline(path: Path) -> BaselineInfo:
    path = Path(path).expanduser().resolve(strict=True)
    with np.load(path, allow_pickle=False) as params:
        missing = {"beta_0", "theta", "gene_names"} - set(params.files)
        if missing:
            raise ValueError("Baseline is missing: " + ", ".join(sorted(missing)))
        beta = np.asarray(params["beta_0"], dtype=float).reshape(-1)
        theta = np.asarray(params["theta"], dtype=float).reshape(-1)
        names = np.asarray(params["gene_names"]).astype(str).reshape(-1)
        ids = np.asarray(params["gene_ids"]).astype(str).reshape(-1) if "gene_ids" in params.files else names
        if not len(beta) or len({len(beta), len(theta), len(names), len(ids)}) != 1:
            raise ValueError("Baseline gene arrays have inconsistent length")
        if not np.isfinite(beta).all() or not np.isfinite(theta).all() or (theta <= 0).any():
            raise ValueError("Baseline beta_0 or theta has invalid values")
        if any(not value or value.lower() == "nan" for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("Baseline stable gene identifiers must be unique and nonempty")
        if "size_factors" in params.files and not np.isfinite(np.asarray(params["size_factors"], dtype=float)).all():
            raise ValueError("Baseline size_factors must be finite")
    return BaselineInfo(path, len(beta), "gene_ids" in params.files)


def discover_baselines(root: Path) -> list[Path]:
    root = Path(root).expanduser()
    if not root.is_dir():
        return []
    valid = []
    for path in sorted(root.rglob("*.npz")):
        try:
            validate_baseline(path)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        valid.append(path)
    return valid
