"""Cross-subject Riemannian transfer learning with Euclidean Alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data_loader import CHANNELS, get_recordings_by_subject, load_recording
from src.epochs import (
    compute_rejection_threshold,
    extract_left_right_epochs,
    reject_bad_epochs,
)
from src.preprocess import preprocess_multichannel_eeg


def _task_epochs(pairs_by_ch: dict) -> np.ndarray:
    """Convert per-channel epoch pairs to (n_trials, n_channels, n_samples)."""
    return np.array(
        [[trial[1] for trial in pairs_by_ch[ch]] for ch in CHANNELS]
    ).transpose(1, 0, 2)


def load_all_subjects(
    data_dirs: list[Path],
    sfreq: float = 250.0,
    subject_merge: dict[str, str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess data from multiple directories, merging by subject.

    Args:
        data_dirs: List of directories to scan for recordings.
        sfreq: Sampling frequency.
        subject_merge: Optional mapping of subject_id -> canonical_id for merging
            (e.g., {"subject0100_2": "subject0100"}).

    Returns:
        Dict mapping subject_id to (X_multichannel, y) where
        X_multichannel is (n_trials, 8, n_samples) and y is (n_trials,).
    """
    if subject_merge is None:
        subject_merge = {}

    all_recordings: dict[str, list[Path]] = {}
    for data_dir in data_dirs:
        for sid, paths in get_recordings_by_subject(data_dir).items():
            canonical = subject_merge.get(sid, sid)
            all_recordings.setdefault(canonical, []).extend(paths)

    subjects: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for sid, rec_paths in sorted(all_recordings.items()):
        all_left: dict[str, list] = {ch: [] for ch in CHANNELS}
        all_right: dict[str, list] = {ch: [] for ch in CHANNELS}

        for rec_path in rec_paths:
            data, events, rec_sfreq = load_recording(rec_path)
            if abs(rec_sfreq - sfreq) > 1e-6:
                continue
            preprocessed = preprocess_multichannel_eeg(data, sfreq)

            left_pairs: dict[str, list] = {}
            right_pairs: dict[str, list] = {}
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
                left_pairs[ch_name] = left
                right_pairs[ch_name] = right

            threshold = compute_rejection_threshold(left_pairs, right_pairs)
            left_pairs, right_pairs, _, _ = reject_bad_epochs(
                left_pairs,
                right_pairs,
                threshold_uv=threshold,
            )

            for ch in CHANNELS:
                all_left[ch].extend(left_pairs[ch])
                all_right[ch].extend(right_pairs[ch])

        n_left = len(all_left[CHANNELS[0]])
        n_right = len(all_right[CHANNELS[0]])
        if n_left == 0 or n_right == 0:
            continue

        X_mc = np.vstack([_task_epochs(all_left), _task_epochs(all_right)])
        y = np.array([0] * n_left + [1] * n_right)
        subjects[sid] = (X_mc, y)

    return subjects
