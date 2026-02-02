"""Feature extraction: multi-channel, multi-band power computation with ERD."""

import numpy as np
from scipy.signal import welch, hilbert


# Frequency bands for feature extraction
FREQUENCY_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "low_beta": (13, 20),
    "high_beta": (20, 30),
}


def compute_band_power(
    epoch: np.ndarray,
    sfreq: float,
    low_freq: float,
    high_freq: float,
) -> float:
    """
    Compute power in a frequency band using Welch's method.

    Args:
        epoch: 1D signal array (single epoch)
        sfreq: Sampling frequency in Hz
        low_freq: Lower bound of frequency band (Hz)
        high_freq: Upper bound of frequency band (Hz)

    Returns:
        Mean power in the frequency band
    """
    # nperseg must be <= len(epoch); use half the epoch for good frequency resolution
    nperseg = min(int(sfreq / 2), len(epoch) // 2)
    nperseg = max(nperseg, 64)  # Minimum for reasonable FFT
    freqs, psd = welch(epoch, sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= low_freq) & (freqs <= high_freq)
    return float(np.mean(psd[mask]))


def compute_theta_power(
    epoch: np.ndarray,
    sfreq: float,
    theta_low: float = 4.0,
    theta_high: float = 8.0,
) -> float:
    """Compute theta band power (backward compatible)."""
    return compute_band_power(epoch, sfreq, theta_low, theta_high)


def compute_hjorth_parameters(epoch: np.ndarray) -> tuple[float, float, float]:
    """
    Compute Hjorth parameters: Activity, Mobility, Complexity.

    These are efficient time-domain features for EEG.
    """
    # Activity = variance
    activity = np.var(epoch)

    # First derivative
    diff1 = np.diff(epoch)
    var_diff1 = np.var(diff1)

    # Second derivative
    diff2 = np.diff(diff1)
    var_diff2 = np.var(diff2)

    # Mobility = sqrt(var(diff1) / var(signal))
    mobility = np.sqrt(var_diff1 / (activity + 1e-10))

    # Complexity = mobility(diff1) / mobility(signal)
    mobility_diff1 = np.sqrt(var_diff2 / (var_diff1 + 1e-10))
    complexity = mobility_diff1 / (mobility + 1e-10)

    return activity, mobility, complexity


def compute_envelope_features(epoch: np.ndarray) -> tuple[float, float, float]:
    """
    Compute features from the signal envelope using Hilbert transform.
    """
    analytic_signal = hilbert(epoch)
    envelope = np.abs(analytic_signal)

    env_mean = np.mean(envelope)
    env_std = np.std(envelope)
    env_max = np.max(envelope)

    return env_mean, env_std, env_max


def extract_features(
    epochs: list[np.ndarray],
    sfreq: float,
    log_transform: bool = True,
) -> np.ndarray:
    """
    Extract theta power features from a list of epochs (backward compatible).

    Args:
        epochs: List of 1D epoch arrays
        sfreq: Sampling frequency in Hz
        log_transform: Whether to apply log transform to power values

    Returns:
        Feature array of shape (n_epochs, 1)
    """
    features = []

    for epoch in epochs:
        theta_power = compute_theta_power(epoch, sfreq)
        if log_transform:
            theta_power = np.log(theta_power + 1e-10)
        features.append([theta_power])

    return np.array(features)


def extract_erd_features(
    epoch_pairs_by_channel: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    sfreq: float,
) -> np.ndarray:
    """
    Extract comprehensive ERD features from all motor cortex channels.

    ERD = (baseline_power - task_power) / baseline_power * 100

    Args:
        epoch_pairs_by_channel: Dict mapping channel names to list of (baseline, task) pairs
        sfreq: Sampling frequency in Hz

    Returns:
        Feature array of shape (n_epochs, n_features)
    """
    # More bands for better discrimination
    bands = {
        "theta": (4, 8),
        "low_alpha": (8, 10),
        "high_alpha": (10, 13),
        "mu": (8, 12),
        "low_beta": (13, 20),
        "high_beta": (20, 30),
        "beta": (13, 30),
        "gamma": (30, 40),
    }

    # All relevant channels
    motor_channels = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

    channel_names = list(epoch_pairs_by_channel.keys())
    n_epochs = len(epoch_pairs_by_channel[channel_names[0]])

    all_features = []

    for epoch_idx in range(n_epochs):
        epoch_features = []
        channel_powers = {}

        for ch_name in motor_channels:
            if ch_name not in epoch_pairs_by_channel:
                continue

            baseline, task = epoch_pairs_by_channel[ch_name][epoch_idx]
            channel_powers[ch_name] = {}

            # Compute total power for normalization
            baseline_total = compute_band_power(baseline, sfreq, 1, 40)
            task_total = compute_band_power(task, sfreq, 1, 40)

            for band_name, (low, high) in bands.items():
                baseline_power = compute_band_power(baseline, sfreq, low, high)
                task_power = compute_band_power(task, sfreq, low, high)

                # ERD percentage
                erd = (baseline_power - task_power) / (baseline_power + 1e-10) * 100
                epoch_features.append(erd)

                # Log power ratio
                log_ratio = np.log((task_power + 1e-10) / (baseline_power + 1e-10))
                epoch_features.append(log_ratio)

                # Relative power (normalized)
                baseline_rel = baseline_power / (baseline_total + 1e-10)
                task_rel = task_power / (task_total + 1e-10)
                epoch_features.append(task_rel - baseline_rel)

                # Absolute log power (often more discriminative)
                epoch_features.append(np.log(task_power + 1e-10))
                epoch_features.append(np.log(baseline_power + 1e-10))

                # Store for inter-channel features
                channel_powers[ch_name][band_name] = {
                    'baseline': baseline_power,
                    'task': task_power,
                    'erd': erd,
                }

            # Add time-domain features from task epoch
            activity, mobility, complexity = compute_hjorth_parameters(task)
            epoch_features.extend([activity, mobility, complexity])

            # Envelope features
            env_mean, env_std, env_max = compute_envelope_features(task)
            epoch_features.extend([env_mean, env_std, env_max])

        # Inter-channel features: C3-C4 asymmetry for each band
        if "C3" in channel_powers and "C4" in channel_powers:
            for band_name in bands:
                c3_erd = channel_powers["C3"][band_name]['erd']
                c4_erd = channel_powers["C4"][band_name]['erd']
                epoch_features.append(c3_erd - c4_erd)

                # Power ratio
                c3_task = channel_powers["C3"][band_name]['task']
                c4_task = channel_powers["C4"][band_name]['task']
                epoch_features.append(np.log((c3_task + 1e-10) / (c4_task + 1e-10)))

        # Frontal-Parietal connectivity proxy (Fz vs Pz)
        if "Fz" in channel_powers and "Pz" in channel_powers:
            for band_name in ["theta", "mu", "beta"]:
                fz_task = channel_powers["Fz"][band_name]['task']
                pz_task = channel_powers["Pz"][band_name]['task']
                epoch_features.append(np.log((fz_task + 1e-10) / (pz_task + 1e-10)))

        # Central vs occipital (Cz vs Oz)
        if "Cz" in channel_powers and "Oz" in channel_powers:
            for band_name in ["mu", "beta"]:
                cz_task = channel_powers["Cz"][band_name]['task']
                oz_task = channel_powers["Oz"][band_name]['task']
                epoch_features.append(np.log((cz_task + 1e-10) / (oz_task + 1e-10)))

        all_features.append(epoch_features)

    return np.array(all_features)


def compute_lateralization_index(
    c3_signal: np.ndarray,
    c4_signal: np.ndarray,
    sfreq: float,
    low_freq: float,
    high_freq: float,
) -> float:
    """
    Compute lateralization index between C3 and C4.

    Lateralization Index = (C4_power - C3_power) / (C4_power + C3_power)

    Positive values indicate left hand imagery (more C3 desync).
    Negative values indicate right hand imagery (more C4 desync).

    Args:
        c3_signal: Signal from C3 electrode (left motor cortex)
        c4_signal: Signal from C4 electrode (right motor cortex)
        sfreq: Sampling frequency in Hz
        low_freq: Lower bound of frequency band
        high_freq: Upper bound of frequency band

    Returns:
        Lateralization index (-1 to 1)
    """
    c3_power = compute_band_power(c3_signal, sfreq, low_freq, high_freq)
    c4_power = compute_band_power(c4_signal, sfreq, low_freq, high_freq)

    return (c4_power - c3_power) / (c4_power + c3_power + 1e-10)


def extract_lateralization_features(
    epoch_pairs_by_channel: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    sfreq: float,
) -> np.ndarray:
    """
    Extract features optimized for left/right motor imagery discrimination.

    Focuses on C3-C4 lateralization in mu (8-12Hz) and beta (13-30Hz) bands.

    Args:
        epoch_pairs_by_channel: Dict mapping channel names to list of (baseline, task) pairs
        sfreq: Sampling frequency in Hz

    Returns:
        Feature array of shape (n_epochs, n_features)
    """
    # Key frequency bands for motor imagery
    bands = {
        "mu": (8, 12),
        "low_beta": (13, 20),
        "high_beta": (20, 30),
        "beta": (13, 30),
    }

    channel_names = list(epoch_pairs_by_channel.keys())
    n_epochs = len(epoch_pairs_by_channel[channel_names[0]])

    all_features = []

    for epoch_idx in range(n_epochs):
        epoch_features = []

        # Get C3 and C4 signals for this epoch
        c3_baseline, c3_task = epoch_pairs_by_channel["C3"][epoch_idx]
        c4_baseline, c4_task = epoch_pairs_by_channel["C4"][epoch_idx]

        # PRIMARY: Lateralization features (C3 vs C4)
        for band_name, (low, high) in bands.items():
            # Task lateralization index
            lat_idx_task = compute_lateralization_index(c3_task, c4_task, sfreq, low, high)
            epoch_features.append(lat_idx_task)

            # Baseline lateralization index
            lat_idx_baseline = compute_lateralization_index(c3_baseline, c4_baseline, sfreq, low, high)
            epoch_features.append(lat_idx_baseline)

            # Change in lateralization (task - baseline)
            epoch_features.append(lat_idx_task - lat_idx_baseline)

            # ERD asymmetry: difference in ERD between C3 and C4
            c3_baseline_power = compute_band_power(c3_baseline, sfreq, low, high)
            c3_task_power = compute_band_power(c3_task, sfreq, low, high)
            c4_baseline_power = compute_band_power(c4_baseline, sfreq, low, high)
            c4_task_power = compute_band_power(c4_task, sfreq, low, high)

            c3_erd = (c3_baseline_power - c3_task_power) / (c3_baseline_power + 1e-10) * 100
            c4_erd = (c4_baseline_power - c4_task_power) / (c4_baseline_power + 1e-10) * 100
            epoch_features.append(c3_erd - c4_erd)  # Asymmetry

            # Log power ratio (C3/C4) during task
            epoch_features.append(np.log((c3_task_power + 1e-10) / (c4_task_power + 1e-10)))

            # Individual channel ERDs
            epoch_features.append(c3_erd)
            epoch_features.append(c4_erd)

        # SECONDARY: Cz features (supplementary motor area)
        cz_baseline, cz_task = epoch_pairs_by_channel["Cz"][epoch_idx]
        for band_name, (low, high) in [("mu", (8, 12)), ("beta", (13, 30))]:
            cz_baseline_power = compute_band_power(cz_baseline, sfreq, low, high)
            cz_task_power = compute_band_power(cz_task, sfreq, low, high)
            cz_erd = (cz_baseline_power - cz_task_power) / (cz_baseline_power + 1e-10) * 100
            epoch_features.append(cz_erd)
            epoch_features.append(np.log(cz_task_power + 1e-10))

        # TERTIARY: Fz theta (attention/effort marker)
        fz_baseline, fz_task = epoch_pairs_by_channel["Fz"][epoch_idx]
        fz_theta_baseline = compute_band_power(fz_baseline, sfreq, 4, 8)
        fz_theta_task = compute_band_power(fz_task, sfreq, 4, 8)
        epoch_features.append(np.log((fz_theta_task + 1e-10) / (fz_theta_baseline + 1e-10)))

        # Time-domain features from C3 and C4
        for signal in [c3_task, c4_task]:
            activity, mobility, complexity = compute_hjorth_parameters(signal)
            epoch_features.extend([activity, mobility, complexity])

        all_features.append(epoch_features)

    return np.array(all_features)
