"""Load and parse Unicorn EEG recordings."""

from collections import defaultdict

import numpy as np
import pandas as pd
from pathlib import Path


# Channel names in order from CSV
CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
SFREQ = 250.0  # Sampling frequency in Hz


def load_recording(
    csv_path: Path,
) -> tuple[np.ndarray, list[tuple[int, int, int]], float]:
    """
    Load a single recording from CSV.

    Args:
        csv_path: Path to the CSV file

    Returns:
        data: EEG data array of shape (n_channels, n_samples)
        events: List of (sample_idx, phase, movement) tuples
        sfreq: Sampling frequency (250 Hz)
    """
    df = pd.read_csv(csv_path)
    data = df[CHANNELS].values.T
    stim = df["stim"].values

    (nonzero_idx,) = np.nonzero(stim)
    stim_vals = stim[nonzero_idx].astype(int)
    events = [
        (int(idx), (val // 10) % 10, val % 10)
        for idx, val in zip(nonzero_idx, stim_vals)
    ]

    return data, events, SFREQ


def get_complete_recordings(data_dir: Path) -> list[Path]:
    """
    Find all complete recordings (those with 100 imagery trials).

    Args:
        data_dir: Path to unicorn-data directory

    Returns:
        List of paths to complete recording CSVs
    """
    complete = []
    for csv_path in sorted(data_dir.glob("subject*/session*/*.csv")):
        stim = pd.read_csv(csv_path, usecols=["stim"])["stim"].values
        nonzero = stim[stim != 0].astype(int)
        phase3_count = np.sum((nonzero // 10) % 10 == 3)
        if phase3_count == 100:
            complete.append(csv_path)
    return complete


def get_recordings_by_subject(data_dir: Path) -> dict[str, list[Path]]:
    """Group complete recordings by subject ID.

    Calls :func:`get_complete_recordings` and groups the resulting paths by
    their parent subject directory name (e.g. ``"subject0001"``).

    Args:
        data_dir: Path to unicorn-data directory.

    Returns:
        Dict mapping subject ID strings to lists of recording paths.
    """
    grouped: dict[str, list[Path]] = defaultdict(list)
    for rec_path in get_complete_recordings(data_dir):
        subject_id = rec_path.parent.parent.name
        grouped[subject_id].append(rec_path)
    return dict(grouped)
