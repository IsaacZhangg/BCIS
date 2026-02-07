"""Load and parse Unicorn EEG recordings."""

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
    data = df[CHANNELS].values.T  # (n_channels, n_samples)

    stim = df["stim"].values
    mask = stim != 0
    indices = np.where(mask)[0]
    stim_nz = stim[mask].astype(int)
    phases = (stim_nz // 10) % 10
    movements = stim_nz % 10
    events = list(zip(indices.tolist(), phases.tolist(), movements.tolist()))

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
        stim = pd.read_csv(csv_path)["stim"].values
        stim_nz = stim[stim != 0].astype(int)
        imagery_count = np.sum((stim_nz // 10) % 10 == 3)
        if imagery_count == 100:
            complete.append(csv_path)
    return complete
