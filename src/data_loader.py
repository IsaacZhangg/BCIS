"""Load and parse Unicorn EEG recordings."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.epochs import extract_left_right_epochs, reject_bad_epochs
from src.preprocess import preprocess_multichannel_eeg

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


def get_complete_recordings(data_dir: Path) -> list[Path]:
    """Find all complete recordings (those with 100 imagery trials)."""
    complete = []
    for csv_path in sorted(data_dir.glob("subject*/session*/*.csv")):
        stim = pd.read_csv(csv_path, usecols=["stim"])["stim"].to_numpy(copy=False)
        nonzero = stim[stim != 0].astype(int)
        phase3_count = np.sum((nonzero // 10) % 10 == 3)
        if phase3_count == 100:
            complete.append(csv_path)
    return complete


def get_recordings_by_subject(data_dir: Path) -> dict[str, list[Path]]:
    """Group complete recordings by subject ID."""
    grouped: dict[str, list[Path]] = {}
    for rec_path in get_complete_recordings(data_dir):
        subject_id = rec_path.parent.parent.name
        grouped.setdefault(subject_id, []).append(rec_path)
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
