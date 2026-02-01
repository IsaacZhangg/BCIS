"""Load and parse Unicorn EEG recordings."""

import numpy as np
import pandas as pd
from pathlib import Path


# Channel names in order from CSV
CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
SFREQ = 250.0  # Sampling frequency in Hz


def load_recording(csv_path: Path) -> tuple[np.ndarray, list[tuple[int, int, int]], float]:
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

    # Extract EEG channels (columns 1-8, excluding timestamps and stim)
    data = df[CHANNELS].values.T  # Transpose to (n_channels, n_samples)

    # Parse events from stim column
    events = []
    stim = df["stim"].values
    for idx, val in enumerate(stim):
        if val != 0:
            stim_int = int(val)
            phase = (stim_int // 10) % 10
            movement = stim_int % 10
            events.append((idx, phase, movement))

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
        df = pd.read_csv(csv_path)
        stim = df["stim"].values

        # Count phase 3 (imagery) events
        phase3_count = 0
        for val in stim:
            if val != 0:
                phase = (int(val) // 10) % 10
                if phase == 3:
                    phase3_count += 1

        if phase3_count == 100:
            complete.append(csv_path)

    return complete
