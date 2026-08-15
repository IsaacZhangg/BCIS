"""Load and parse Unicorn EEG recordings."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.epochs import extract_left_right_epochs, reject_bad_epochs
from src.preprocess import preprocess_multichannel_eeg

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


def get_recordings(data_dir: Path, min_trials: int = 100) -> list[Path]:
    """Find recordings with at least *min_trials* phase-3 imagery events.

    Args:
        data_dir: Root directory containing subject*/session*/*.csv files.
        min_trials: Minimum number of phase-3 trials to include a recording.
            Default 100 matches the original "complete recording" criterion.
    """
    matched = []
    for csv_path in sorted(data_dir.glob("subject*/session*/*.csv")):
        if _phase3_count(csv_path) >= min_trials:
            matched.append(csv_path)
    return matched


# Backward compatibility alias
get_complete_recordings = get_recordings


def get_recordings_by_subject(
    data_dir: Path, min_trials: int = 100
) -> dict[str, list[Path]]:
    """Group recordings by subject ID, filtering by minimum trial count."""
    grouped: dict[str, list[Path]] = {}
    for rec_path in get_recordings(data_dir, min_trials=min_trials):
        subject_id = rec_path.parent.parent.name
        grouped.setdefault(subject_id, []).append(rec_path)
    return grouped


def get_recordings_flexible(
    data_dir: Path,
    min_trials: int = 20,
) -> dict[str, list[Path]]:
    """Find recordings with >= min_trials phase-3 events, grouped by subject.

    Unlike get_complete_recordings (which requires exactly 100 trials), this
    accepts any recording with sufficient phase-3 events — needed for MI_DATA_NEW
    recordings that have 32 trials each.
    """
    grouped: dict[str, list[Path]] = {}
    for csv_path in sorted(data_dir.glob("subject*/session*/*.csv")):
        if _phase3_count(csv_path) >= min_trials:
            subject_id = csv_path.parent.parent.name
            grouped.setdefault(subject_id, []).append(csv_path)
    return grouped


def process_recording(
    rec_path: Path,
    sfreq: float,
    threshold_uv: float | None = None,
) -> tuple | None:
    """Load, preprocess, epoch, and artifact-reject a single recording."""
    data, events, _ = load_recording(rec_path)
    preprocessed = preprocess_multichannel_eeg(data, sfreq)

    left_pairs_by_channel: dict = {}
    right_pairs_by_channel: dict = {}
    for ch_idx, ch_name in enumerate(CHANNELS):
        signal = preprocessed[ch_idx]
        left, right = extract_left_right_epochs(
            signal,
            events,
            sfreq,
            task_duration=3.0,
            baseline_duration=1.0,
            skip_duration=0.25,
        )
        left_pairs_by_channel[ch_name] = left
        right_pairs_by_channel[ch_name] = right

    n_left_raw = len(left_pairs_by_channel[CHANNELS[0]])
    n_right_raw = len(right_pairs_by_channel[CHANNELS[0]])

    left_pairs_by_channel, right_pairs_by_channel, rej_l, rej_r = reject_bad_epochs(
        left_pairs_by_channel,
        right_pairs_by_channel,
        threshold_uv=threshold_uv,
    )

    n_left = len(left_pairs_by_channel[CHANNELS[0]])
    n_right = len(right_pairs_by_channel[CHANNELS[0]])

    if n_left == 0 or n_right == 0:
        return None

    return (
        left_pairs_by_channel,
        right_pairs_by_channel,
        n_left_raw,
        n_right_raw,
        rej_l,
        rej_r,
    )
