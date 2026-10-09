"""Load and parse Unicorn EEG recordings."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
SFREQ = 250.0


def recording_fingerprint(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return a content fingerprint (size + sha256) for duplicate detection."""
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _phase3_count(csv_path: Path) -> int:
    stim = pd.read_csv(csv_path, usecols=["stim"])["stim"].to_numpy(copy=False)
    nonzero = stim[stim != 0].astype(int)
    return int(np.sum((nonzero // 10) % 10 == 3))


def discover_recordings(
    data_dirs: list[Path],
    min_trials: int = 20,
    subject_merge: dict[str, str] | None = None,
) -> dict[str, list[Path]]:
    """Scan one or more data roots, drop duplicate files, and group by subject.

    Duplicate detection uses a content fingerprint so the same recording copied
    into both ``unicorn-data`` and ``MI_DATA_NEW`` is loaded once. Subject alias
    maps (e.g. ``subject0100_2`` → ``subject0100``) are applied after grouping.
    """
    if subject_merge is None:
        subject_merge = {}

    seen_fingerprints: set[str] = set()
    grouped: dict[str, list[Path]] = {}
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for csv_path in sorted(data_dir.glob("subject*/session*/*.csv")):
            if _phase3_count(csv_path) < min_trials:
                continue
            fingerprint = recording_fingerprint(csv_path)
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            raw_id = csv_path.parent.parent.name
            canonical = subject_merge.get(raw_id, raw_id)
            grouped.setdefault(canonical, []).append(csv_path)
    return grouped


def load_recording(
    csv_path: Path,
) -> tuple[np.ndarray, list[tuple[int, int, int]], float]:
    """Load a single recording: returns (n_channels × n_samples) data, events, sfreq."""
    df = pd.read_csv(csv_path, usecols=[*CHANNELS, "stim"])
    data = df[CHANNELS].to_numpy(copy=False).T
    stim = df["stim"].to_numpy(copy=False)

    nonzero_idx = np.flatnonzero(stim)
    stim_vals = stim[nonzero_idx].astype(int)
    events = [
        (int(idx), (val // 10) % 10, val % 10)
        for idx, val in zip(nonzero_idx, stim_vals)
    ]

    return data, events, SFREQ
