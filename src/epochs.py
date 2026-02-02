"""Extract epochs from continuous EEG data."""

import numpy as np


def extract_epochs(
    signal: np.ndarray,
    events: list[tuple[int, int, int]],
    sfreq: float,
    duration: float = 2.0,
) -> list[np.ndarray]:
    """
    Extract fixed-duration epochs starting at each event.

    Args:
        signal: 1D preprocessed signal array
        events: List of (sample_idx, phase, movement) tuples
        sfreq: Sampling frequency in Hz
        duration: Epoch duration in seconds

    Returns:
        List of epoch arrays
    """
    n_samples = int(duration * sfreq)
    epochs = []

    for sample_idx, phase, movement in events:
        # Check if epoch would extend past end of signal
        if sample_idx + n_samples <= len(signal):
            epoch = signal[sample_idx : sample_idx + n_samples]
            epochs.append(epoch)

    return epochs


def extract_labeled_epochs(
    signal: np.ndarray,
    events: list[tuple[int, int, int]],
    sfreq: float,
    duration: float = 2.0,
    offset: float = 0.0,
    focused_phase: int = 3,
    baseline_phase: int = 2,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Extract epochs and separate by focus label.

    Default: Phase 3 (imagery) = Focused, Phase 2 (prepare) = Baseline
    Alternative: Phase 3 (imagery) = Focused, Phase 5 (rest) = Baseline

    Args:
        signal: 1D preprocessed signal array
        events: List of (sample_idx, phase, movement) tuples
        sfreq: Sampling frequency in Hz
        duration: Epoch duration in seconds
        offset: Time offset from event start in seconds
        focused_phase: Phase number for focused condition (default 3 = imagery)
        baseline_phase: Phase number for baseline condition (default 2 = prepare)

    Returns:
        Tuple of (focused_epochs, baseline_epochs)
    """
    n_samples = int(duration * sfreq)
    offset_samples = int(offset * sfreq)
    focused = []
    baseline = []

    for sample_idx, phase, movement in events:
        # Apply offset
        start_idx = sample_idx + offset_samples

        # Check if epoch would extend past end of signal
        if start_idx + n_samples > len(signal):
            continue

        epoch = signal[start_idx : start_idx + n_samples]

        if phase == focused_phase:  # Imagery = Focused
            focused.append(epoch)
        elif phase == baseline_phase:  # Prepare or Rest = Baseline
            baseline.append(epoch)
        # Other phases are ignored

    return focused, baseline


def extract_erd_epochs(
    signal: np.ndarray,
    events: list[tuple[int, int, int]],
    sfreq: float,
    task_duration: float = 2.5,
    baseline_duration: float = 1.5,
    class1_phase: int = 3,
    class2_phase: int = 5,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """
    Extract task and baseline epochs for ERD computation.

    For each trial, extracts:
    - Baseline: Immediately before the event
    - Task: During the event

    Args:
        signal: 1D preprocessed signal array
        events: List of (sample_idx, phase, movement) tuples
        sfreq: Sampling frequency in Hz
        task_duration: Duration of task epoch in seconds (default 2.5 for more samples)
        baseline_duration: Duration of baseline epoch in seconds (default 1.5)
        class1_phase: Phase number for class 1 (default 3 = imagery)
        class2_phase: Phase number for class 2 (default 5 = rest)

    Returns:
        Tuple of (class1_pairs, class2_pairs) where each pair is (baseline, task)
    """
    task_samples = int(task_duration * sfreq)
    baseline_samples = int(baseline_duration * sfreq)

    class1_pairs = []
    class2_pairs = []

    for sample_idx, phase, movement in events:
        if phase not in [class1_phase, class2_phase]:
            continue

        # Extract baseline from immediately before event
        baseline_end = sample_idx
        baseline_start = baseline_end - baseline_samples
        if baseline_start < 0:
            continue

        baseline = signal[baseline_start:baseline_end]

        # Extract task epoch (skip first 0.2s to let pattern develop)
        task_start = sample_idx + int(0.2 * sfreq)
        task_end = task_start + task_samples
        if task_end > len(signal):
            continue

        task = signal[task_start:task_end]

        if phase == class1_phase:
            class1_pairs.append((baseline, task))
        elif phase == class2_phase:
            class2_pairs.append((baseline, task))

    return class1_pairs, class2_pairs


def extract_augmented_epochs(
    signal: np.ndarray,
    events: list[tuple[int, int, int]],
    sfreq: float,
    window_duration: float = 2.0,
    n_windows: int = 3,
    class1_phase: int = 3,
    class2_phase: int = 5,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """
    Extract multiple overlapping windows from each trial for data augmentation.

    This increases the training data by extracting multiple time-shifted windows
    from each trial.

    Args:
        signal: 1D preprocessed signal array
        events: List of (sample_idx, phase, movement) tuples
        sfreq: Sampling frequency in Hz
        window_duration: Duration of each window in seconds
        n_windows: Number of overlapping windows to extract per trial
        class1_phase: Phase number for class 1
        class2_phase: Phase number for class 2

    Returns:
        Tuple of (class1_pairs, class2_pairs) where each pair is (baseline, task)
    """
    window_samples = int(window_duration * sfreq)
    baseline_samples = int(1.0 * sfreq)

    class1_pairs = []
    class2_pairs = []

    for sample_idx, phase, movement in events:
        if phase not in [class1_phase, class2_phase]:
            continue

        # Extract baseline
        baseline_end = sample_idx
        baseline_start = baseline_end - baseline_samples
        if baseline_start < 0:
            continue

        baseline = signal[baseline_start:baseline_end]

        # Extract multiple windows with different start offsets
        for window_idx in range(n_windows):
            # Offset: 0.1s, 0.3s, 0.5s, etc.
            offset = 0.1 + window_idx * 0.2
            task_start = sample_idx + int(offset * sfreq)
            task_end = task_start + window_samples

            if task_end > len(signal):
                continue

            task = signal[task_start:task_end]

            if phase == class1_phase:
                class1_pairs.append((baseline, task))
            elif phase == class2_phase:
                class2_pairs.append((baseline, task))

    return class1_pairs, class2_pairs
