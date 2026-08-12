"""On-disk NPZ cache for pipeline stage outputs.

Cache files are keyed by (config_hash, stage, source_content_hash). The cache
skips recomputation when both the training config AND the source recording
content are unchanged. File content hashing (not mtime) is used so caches
survive branch switches that reset mtimes without changing content.

Stages currently supported:
    - preprocessed: output of preprocess_multichannel_eeg (n_channels, n_samples)
    - features: output of pairs_to_features (X_features, X_multichannel, y)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def source_content_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return a 16-char sha256 prefix of the file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class StageCache:
    """NPZ cache keyed by (config_hash, stage, source_content_hash)."""

    def __init__(self, cache_dir: Path, config_hash: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.config_hash = config_hash

    def _path(self, stage: str, source_hash: str) -> Path:
        return self.cache_dir / f"{self.config_hash}_{stage}_{source_hash}.npz"

    def get(self, stage: str, source_hash: str) -> dict[str, np.ndarray] | None:
        """Return cached arrays for (stage, source_hash) or None on miss."""
        path = self._path(stage, source_hash)
        if not path.exists():
            return None
        # Load only numpy arrays; object arrays are rejected for safety.
        with np.load(path) as npz:
            return {key: npz[key].copy() for key in npz.files}

    def put(self, stage: str, source_hash: str, **arrays: np.ndarray) -> Path:
        """Save arrays as NPZ under the (stage, source_hash) key."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(stage, source_hash)
        # Write directly via savez; keyword args become archive member names.
        np.savez(str(path), **arrays)  # type: ignore[call-overload]
        return path
