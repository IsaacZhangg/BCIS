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
    return [
        signal[idx : idx + n_samples]
        for idx, _, _ in events
        if idx + n_samples <= len(signal)
    ]


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

    for sample_idx, phase, _movement in events:
        start_idx = sample_idx + offset_samples
        if start_idx + n_samples > len(signal):
            continue

        epoch = signal[start_idx : start_idx + n_samples]

        if phase == focused_phase:
            focused.append(epoch)
        elif phase == baseline_phase:
            baseline.append(epoch)

    return focused, baseline


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


def reject_bad_epochs(
    left_pairs_by_channel: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    right_pairs_by_channel: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    threshold_uv: float | None = None,
    flat_uv: float = 1.0,
    n_mad: float = 4.0,
) -> tuple[dict, dict, int, int]:
    """Drop trials where any channel's task epoch exceeds amplitude thresholds.

    When *threshold_uv* is ``None`` (default), the threshold is computed
    adaptively as ``median + n_mad * MAD`` of peak-to-peak amplitudes across
    **all** trials and channels.  This adapts to each subject's signal
    characteristics rather than assuming a fixed scale.

    A trial is rejected if any channel has peak-to-peak amplitude above the
    threshold or below *flat_uv* in the task epoch.

    Args:
        left_pairs_by_channel: Dict mapping channel names to (baseline, task) pairs for left trials.
        right_pairs_by_channel: Same for right trials.
        threshold_uv: Fixed maximum peak-to-peak amplitude in µV.  When
            ``None``, the threshold is set adaptively using *n_mad*.
        flat_uv: Minimum peak-to-peak amplitude in µV (flat signal rejection).
        n_mad: Number of MADs above the median to set the adaptive threshold.
            Only used when *threshold_uv* is ``None``.

    Returns:
        Tuple of (cleaned_left, cleaned_right, n_rejected_left, n_rejected_right).
    """
    channels = list(left_pairs_by_channel.keys())

    # Collect all per-trial max-across-channels ptp values for adaptive threshold
    def _trial_ptps(pairs_by_channel: dict) -> list[float]:
        n_trials = len(pairs_by_channel[channels[0]])
        ptps = []
        for i in range(n_trials):
            max_ptp = max(float(np.ptp(pairs_by_channel[ch][i][1])) for ch in channels)
            ptps.append(max_ptp)
        return ptps

    if threshold_uv is None:
        all_ptps = np.array(
            _trial_ptps(left_pairs_by_channel) + _trial_ptps(right_pairs_by_channel)
        )
        median = float(np.median(all_ptps))
        mad = float(np.median(np.abs(all_ptps - median)))
        threshold_uv = median + n_mad * mad

    def _good_indices(pairs_by_channel: dict) -> list[int]:
        n_trials = len(pairs_by_channel[channels[0]])
        good = []
        for i in range(n_trials):
            ok = True
            for ch in channels:
                task = pairs_by_channel[ch][i][1]
                ptp = float(np.ptp(task))
                if ptp > threshold_uv or ptp < flat_uv:
                    ok = False
                    break
            if ok:
                good.append(i)
        return good

    left_good = _good_indices(left_pairs_by_channel)
    right_good = _good_indices(right_pairs_by_channel)

    n_left_orig = len(left_pairs_by_channel[channels[0]])
    n_right_orig = len(right_pairs_by_channel[channels[0]])

    cleaned_left = {
        ch: [left_pairs_by_channel[ch][i] for i in left_good] for ch in channels
    }
    cleaned_right = {
        ch: [right_pairs_by_channel[ch][i] for i in right_good] for ch in channels
    }

    return (
        cleaned_left,
        cleaned_right,
        n_left_orig - len(left_good),
        n_right_orig - len(right_good),
    )
