"""Load and parse Unicorn EEG recordings."""

import numpy as np
import pandas as pd
from pathlib import Path


CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
SFREQ = 250.0


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
        stim = pd.read_csv(csv_path, usecols=["stim"])["stim"].to_numpy(copy=False)
        nonzero = stim[stim != 0].astype(int)
        phase3_count = int(np.sum((nonzero // 10) % 10 == 3))
        if phase3_count >= min_trials:
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
