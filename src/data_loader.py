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

    # Extract EEG channels (columns 1-8, excluding timestamps and stim)
    data = df[CHANNELS].values.T  # Transpose to (n_channels, n_samples)

    # Parse events from stim column
    events = parse_events(df["stim"].values)

    return data, events, SFREQ


def parse_events(stim_values: np.ndarray) -> list[tuple[int, int, int]]:
    """Parse event markers from stim column."""
    events = []

    for idx, value in enumerate(stim_values):
        if value != 0:
            stim_int = int(value)
            phase = extract_phase_from_stim(stim_int)
            movement = stim_int % 10
            events.append((idx, phase, movement))

    return events


def get_complete_recordings(data_dir: Path) -> list[Path]:
    """
    Find all complete recordings (those with 100 imagery trials).

    Args:
        data_dir: Path to unicorn-data directory

    Returns:
        List of paths to complete recording CSVs
    """
    complete_recordings = []

    for csv_path in sorted(data_dir.glob("subject*/session*/*.csv")):
        if is_complete_recording(csv_path):
            complete_recordings.append(csv_path)

    return complete_recordings


def is_complete_recording(csv_path: Path) -> bool:
    """Check if a recording has exactly 100 imagery trials."""
    df = pd.read_csv(csv_path)
    stim_values = df["stim"].values

    # Count phase 3 (imagery) events
    imagery_count = count_phase_events(stim_values, target_phase=3)

    return imagery_count == 100


def count_phase_events(stim_values: np.ndarray, target_phase: int) -> int:
    """Count events for a specific phase in the stim column."""
    count = 0
    for value in stim_values:
        if value != 0:
            phase = extract_phase_from_stim(int(value))
            if phase == target_phase:
                count += 1
    return count


def extract_phase_from_stim(stim_value: int) -> int:
    """Extract phase code from stim marker value."""
    return (stim_value // 10) % 10
