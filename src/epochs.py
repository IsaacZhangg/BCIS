"""Extract epochs from continuous EEG data."""

import numpy as np


def extract_left_right_epochs(
    signal: np.ndarray,
    events: list[tuple[int, int, int]],
    sfreq: float,
    task_duration: float = 1.5,
    baseline_duration: float = 1.0,
    phase: int = 3,
    left_movement: int = 1,
    right_movement: int = 2,
    skip_duration: float = 0.5,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """
    Extract left and right motor imagery epochs from specified phase.

    Args:
        signal: 1D preprocessed signal array
        events: List of (sample_idx, phase, movement) tuples
        sfreq: Sampling frequency in Hz
        task_duration: Duration of task epoch in seconds
        baseline_duration: Duration of baseline epoch in seconds
        phase: Phase number to extract epochs from (default 3)
        left_movement: Movement code for left class (default 1)
        right_movement: Movement code for right class (default 2)
        skip_duration: Time to skip after event start before task epoch (default 0.5s)

    Returns:
        Tuple of (left_pairs, right_pairs) where each pair is (baseline, task)
    """
    task_samples = int(task_duration * sfreq)
    baseline_samples = int(baseline_duration * sfreq)
    skip_samples = int(skip_duration * sfreq)

    left_pairs = []
    right_pairs = []

    for sample_idx, event_phase, movement in events:
        if event_phase != phase:
            continue

        baseline_start = sample_idx - baseline_samples
        if baseline_start < 0:
            continue

        task_start = sample_idx + skip_samples
        task_end = task_start + task_samples
        if task_end > len(signal):
            continue

        baseline = signal[baseline_start:sample_idx]
        task = signal[task_start:task_end]

        if movement == left_movement:
            left_pairs.append((baseline, task))
        elif movement == right_movement:
            right_pairs.append((baseline, task))

    return left_pairs, right_pairs


def _trial_max_ptps(
    pairs_by_channel: dict[str, list[tuple[np.ndarray, np.ndarray]]],
) -> list[float]:
    """Return max cross-channel PTP amplitude for each trial."""
    channels = list(pairs_by_channel.keys())
    n_trials = len(pairs_by_channel[channels[0]])
    return [
        max(float(np.ptp(pairs_by_channel[ch][i][1])) for ch in channels)
        for i in range(n_trials)
    ]


def adaptive_threshold(values: np.ndarray, n_mad: float) -> float:
    """Return median + n_mad * MAD for an array of values."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median + n_mad * mad


def compute_rejection_threshold(
    left_pairs_by_channel: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    right_pairs_by_channel: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    n_mad: float = 4.0,
) -> float:
    """Compute adaptive amplitude rejection threshold without applying it.

    Computes median + n_mad * MAD of peak-to-peak amplitudes across all trials
    and channels.  Use the returned value as ``threshold_uv`` in
    :func:`reject_bad_epochs` to apply a pre-computed threshold (e.g. one
    derived from training data only).

    Args:
        left_pairs_by_channel: Dict mapping channel names to (baseline, task) pairs for left trials.
        right_pairs_by_channel: Same for right trials.
        n_mad: Number of MADs above the median for the threshold.

    Returns:
        Adaptive threshold in µV.
    """
    all_ptps = np.array(
        _trial_max_ptps(left_pairs_by_channel) + _trial_max_ptps(right_pairs_by_channel)
    )
    return adaptive_threshold(all_ptps, n_mad)


def compute_trial_max_ptp(X_multichannel: np.ndarray) -> np.ndarray:
    """Compute max peak-to-peak amplitude across channels for each trial.

    Operates on the (n_trials, n_channels, n_samples) array format used by the
    CV functions, replicating the same PTP metric as :func:`reject_bad_epochs`.

    Args:
        X_multichannel: Array of shape (n_trials, n_channels, n_samples).

    Returns:
        Array of shape (n_trials,) with the max PTP per trial.
    """
    return np.max(np.ptp(X_multichannel, axis=2), axis=1)


def task_epochs(
    pairs_by_ch: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    channels: list[str],
) -> np.ndarray:
    """Convert per-channel epoch pairs to (n_trials, n_channels, n_samples)."""
    return np.array(
        [[trial[1] for trial in pairs_by_ch[ch]] for ch in channels]
    ).transpose(1, 0, 2)


def reject_bad_epochs(
    left_pairs_by_channel: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    right_pairs_by_channel: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    threshold_uv: float | None = None,
    flat_uv: float = 1.0,
    n_mad: float = 4.0,
) -> tuple[dict, dict, int, int]:
    """Drop trials whose task epoch is too large or too flat on any channel.

    A trial is rejected when its max peak-to-peak amplitude across channels
    exceeds *threshold_uv* (adaptive median + n_mad * MAD when ``None``), or
    when any channel is flatter than *flat_uv*.

    Returns:
        Tuple of (cleaned_left, cleaned_right, n_rejected_left, n_rejected_right).
    """
    channels = list(left_pairs_by_channel.keys())

    def _trial_ptp_range(pairs_by_channel):
        n_trials = len(pairs_by_channel[channels[0]])
        max_ptps, min_ptps = [], []
        for i in range(n_trials):
            ch_ptps = [float(np.ptp(pairs_by_channel[ch][i][1])) for ch in channels]
            max_ptps.append(max(ch_ptps))
            min_ptps.append(min(ch_ptps))
        return max_ptps, min_ptps

    left_metrics = _trial_ptp_range(left_pairs_by_channel)
    right_metrics = _trial_ptp_range(right_pairs_by_channel)

    if threshold_uv is None:
        threshold_uv = adaptive_threshold(
            np.array(left_metrics[0] + right_metrics[0]), n_mad
        )

    def _filter_pairs(pairs_by_channel, metrics):
        max_ptps, min_ptps = metrics
        good = [
            i
            for i in range(len(max_ptps))
            if max_ptps[i] <= threshold_uv and min_ptps[i] >= flat_uv
        ]
        cleaned = {ch: [pairs_by_channel[ch][i] for i in good] for ch in channels}
        return cleaned, len(max_ptps) - len(good)

    cleaned_left, n_rejected_left = _filter_pairs(left_pairs_by_channel, left_metrics)
    cleaned_right, n_rejected_right = _filter_pairs(
        right_pairs_by_channel, right_metrics
    )

    return cleaned_left, cleaned_right, n_rejected_left, n_rejected_right
