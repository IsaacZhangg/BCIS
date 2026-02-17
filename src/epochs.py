"""Extract epochs from continuous EEG data."""

import numpy as np
from scipy.signal import welch


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


def _compute_max_gradient(task: np.ndarray) -> float:
    """Return the maximum absolute sample-to-sample voltage difference."""
    return float(np.max(np.abs(np.diff(task))))


def _compute_hf_power(task: np.ndarray, sfreq: float, low: float, high: float) -> float:
    """Return mean spectral power in [low, high] Hz using Welch's method."""
    nperseg = min(int(sfreq / 2), len(task) // 2)
    nperseg = max(nperseg, 64)
    freqs, psd = welch(task, sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= low) & (freqs <= high)
    return float(np.mean(psd[mask]))


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


def _adaptive_threshold(values: np.ndarray, n_mad: float) -> float:
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
    return _adaptive_threshold(all_ptps, n_mad)


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


def reject_bad_epochs(
    left_pairs_by_channel: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    right_pairs_by_channel: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    threshold_uv: float | None = None,
    flat_uv: float = 1.0,
    n_mad: float = 4.0,
    sfreq: float | None = None,
    gradient_n_mad: float | None = 3.0,
    hf_power_n_mad: float | None = 3.0,
    hf_band: tuple[float, float] = (30.0, 45.0),
) -> tuple[dict, dict, int, int]:
    """Drop trials where any channel's task epoch exceeds rejection criteria.

    Three rejection criteria are applied (any channel triggers rejection):

    1. **Peak-to-peak amplitude**: Adaptive (median + n_mad * MAD) or fixed
       threshold.  Also rejects flat trials (ptp < flat_uv).
    2. **Gradient**: Maximum sample-to-sample voltage jump exceeds adaptive
       threshold (median + gradient_n_mad * MAD).  Catches electrode pops and
       movement artifacts.  Disabled when *gradient_n_mad* is ``None`` or
       *sfreq* is ``None``.
    3. **High-frequency power**: Trials with abnormally high power in
       *hf_band* (EMG contamination).  Uses adaptive median + hf_power_n_mad *
       MAD threshold.  Requires *sfreq*.  Disabled when *hf_power_n_mad* is
       ``None``.

    Args:
        left_pairs_by_channel: Dict mapping channel names to (baseline, task) pairs for left trials.
        right_pairs_by_channel: Same for right trials.
        threshold_uv: Fixed maximum peak-to-peak amplitude in µV.  When
            ``None``, the threshold is set adaptively using *n_mad*.
        flat_uv: Minimum peak-to-peak amplitude in µV (flat signal rejection).
        n_mad: Number of MADs above the median to set the adaptive threshold.
            Only used when *threshold_uv* is ``None``.
        sfreq: Sampling frequency in Hz.  Required for gradient and HF power
            rejection.
        gradient_n_mad: Number of MADs above the median max-gradient for adaptive
            gradient rejection.  Set to ``None`` to disable.
        hf_power_n_mad: Number of MADs above the median HF power for adaptive
            HF rejection.  Set to ``None`` to disable.
        hf_band: Frequency band for HF power rejection (default 30-45 Hz).

    Returns:
        Tuple of (cleaned_left, cleaned_right, n_rejected_left, n_rejected_right).
    """
    channels = list(left_pairs_by_channel.keys())

    def _trial_max_metric(pairs_by_channel, metric_fn):
        """Return max cross-channel metric value for each trial."""
        n_trials = len(pairs_by_channel[channels[0]])
        return [
            max(metric_fn(pairs_by_channel[ch][i][1]) for ch in channels)
            for i in range(n_trials)
        ]

    # Peak-to-peak amplitude threshold (adaptive or fixed)
    if threshold_uv is None:
        all_ptps = np.array(
            _trial_max_metric(left_pairs_by_channel, np.ptp)
            + _trial_max_metric(right_pairs_by_channel, np.ptp)
        )
        threshold_uv = _adaptive_threshold(all_ptps, n_mad)

    # Gradient adaptive threshold
    gradient_threshold: float | None = None
    if gradient_n_mad is not None and sfreq is not None:
        all_grads = np.array(
            _trial_max_metric(left_pairs_by_channel, _compute_max_gradient)
            + _trial_max_metric(right_pairs_by_channel, _compute_max_gradient)
        )
        gradient_threshold = _adaptive_threshold(all_grads, gradient_n_mad)

    # HF power adaptive threshold
    hf_threshold: float | None = None
    if hf_power_n_mad is not None and sfreq is not None:

        def hf_fn(task):
            return _compute_hf_power(task, sfreq, hf_band[0], hf_band[1])

        all_hf = np.array(
            _trial_max_metric(left_pairs_by_channel, hf_fn)
            + _trial_max_metric(right_pairs_by_channel, hf_fn)
        )
        hf_threshold = _adaptive_threshold(all_hf, hf_power_n_mad)

    # Apply all criteria
    def _is_good_trial(pairs_by_channel, trial_idx):
        """Return True if trial passes all rejection criteria across all channels."""
        for ch in channels:
            task = pairs_by_channel[ch][trial_idx][1]
            ptp = float(np.ptp(task))
            if ptp > threshold_uv or ptp < flat_uv:
                return False
            if (
                gradient_threshold is not None
                and _compute_max_gradient(task) > gradient_threshold
            ):
                return False
            if (
                hf_threshold is not None
                and _compute_hf_power(task, sfreq, hf_band[0], hf_band[1])
                > hf_threshold
            ):
                return False
        return True

    def _filter_pairs(pairs_by_channel):
        n_trials = len(pairs_by_channel[channels[0]])
        good = [i for i in range(n_trials) if _is_good_trial(pairs_by_channel, i)]
        cleaned = {ch: [pairs_by_channel[ch][i] for i in good] for ch in channels}
        return cleaned, n_trials - len(good)

    cleaned_left, n_rejected_left = _filter_pairs(left_pairs_by_channel)
    cleaned_right, n_rejected_right = _filter_pairs(right_pairs_by_channel)

    return cleaned_left, cleaned_right, n_rejected_left, n_rejected_right
