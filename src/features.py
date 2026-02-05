"""Feature extraction: multi-channel, multi-band power computation with ERD."""

import numpy as np
from scipy.signal import welch, hilbert, butter, filtfilt
from scipy.stats import entropy, skew, kurtosis


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

    # First and second derivatives
    first_derivative = np.diff(epoch)
    second_derivative = np.diff(first_derivative)

    var_first_derivative = np.var(first_derivative)
    var_second_derivative = np.var(second_derivative)

    # Mobility = sqrt(var(first_derivative) / var(signal))
    mobility = np.sqrt(var_first_derivative / (activity + 1e-10))

    # Complexity = mobility(first_derivative) / mobility(signal)
    mobility_first_derivative = np.sqrt(
        var_second_derivative / (var_first_derivative + 1e-10)
    )
    complexity = mobility_first_derivative / (mobility + 1e-10)

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


def bandpass_filter_signal(
    signal: np.ndarray, sfreq: float, low: float, high: float, order: int = 4
) -> np.ndarray:
    """Apply bandpass filter to signal."""
    nyq = sfreq / 2
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(order, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, signal)


def compute_statistical_features(epoch: np.ndarray) -> list[float]:
    """Compute statistical features from an epoch."""
    features = [
        np.mean(epoch),
        np.std(epoch),
        np.var(epoch),
        skew(epoch),
        kurtosis(epoch),
        np.ptp(epoch),  # peak-to-peak
        np.percentile(epoch, 25),
        np.percentile(epoch, 75),
    ]
    return features


def compute_spectral_entropy(epoch: np.ndarray, sfreq: float) -> float:
    """Compute spectral entropy."""
    nperseg = min(int(sfreq / 2), len(epoch) // 2)
    nperseg = max(nperseg, 64)
    freqs, psd = welch(epoch, sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    psd_norm = psd / (np.sum(psd) + 1e-10)
    return entropy(psd_norm + 1e-10)


def compute_line_length(epoch: np.ndarray) -> float:
    """Compute line length (sum of absolute differences)."""
    return np.sum(np.abs(np.diff(epoch)))


def compute_zero_crossings(epoch: np.ndarray) -> int:
    """Count zero crossings."""
    return np.sum(np.diff(np.sign(epoch - np.mean(epoch))) != 0)


def compute_filter_bank_features(epoch: np.ndarray, sfreq: float) -> list[float]:
    """
    Compute filter bank features (FBCSP-style sub-band powers).

    Divides signal into overlapping frequency bands and computes
    log-variance features used in FBCSP methods.
    """
    # Filter bank: overlapping bands from 4-40 Hz
    filter_banks = [
        (4, 8),
        (6, 10),
        (8, 12),
        (10, 14),
        (12, 16),
        (14, 18),
        (16, 20),
        (18, 22),
        (20, 24),
        (22, 26),
        (24, 28),
        (26, 30),
        (28, 32),
        (30, 34),
        (32, 36),
        (34, 38),
    ]

    features = []
    for low, high in filter_banks:
        try:
            filtered = bandpass_filter_signal(epoch, sfreq, low, high)
            # Log variance (standard FBCSP feature)
            log_var = np.log(np.var(filtered) + 1e-10)
            features.append(log_var)
        except Exception:
            features.append(0.0)

    return features


def compute_wavelet_features(epoch: np.ndarray) -> list[float]:
    """
    Compute simple wavelet-like multi-resolution features.

    Uses differencing at multiple scales instead of full wavelet transform.
    """
    features = []

    # Multi-scale analysis using differencing
    for scale in [1, 2, 4, 8, 16]:
        detail = np.diff(epoch, n=1)
        if len(detail) > scale:
            # Downsample by scale
            downsampled = detail[::scale]
            features.extend(
                [
                    np.log(np.var(downsampled) + 1e-10),
                    np.mean(np.abs(downsampled)),
                ]
            )
        else:
            features.extend([0.0, 0.0])

    return features


def compute_temporal_features(epoch: np.ndarray, sfreq: float) -> list[float]:
    """
    Compute additional temporal features for engagement detection.

    Phase 3 (imagery) vs Phase 5 (rest) should show different temporal dynamics.
    """
    features = []

    # Split epoch into segments for temporal analysis
    n_segments = 4
    segment_length = len(epoch) // n_segments

    segment_variances = []
    segment_mean_amplitudes = []

    for i in range(n_segments):
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length
        segment = epoch[start_idx:end_idx]

        segment_variances.append(np.var(segment))
        segment_mean_amplitudes.append(np.mean(np.abs(segment)))

    # Variance trend features
    variance_std = np.std(segment_variances)
    variance_ratio = segment_variances[-1] / (segment_variances[0] + 1e-10)
    features.extend([variance_std, variance_ratio])

    # Activity trend features
    amplitude_std = np.std(segment_mean_amplitudes)
    amplitude_ratio = segment_mean_amplitudes[-1] / (segment_mean_amplitudes[0] + 1e-10)
    features.extend([amplitude_std, amplitude_ratio])

    # Root mean square and signal energy
    rms = np.sqrt(np.mean(epoch**2))
    energy = np.sum(epoch**2)
    features.extend([rms, energy])

    # Autocorrelation at specific lags (rhythm indicators)
    lags_ms = [40, 100, 200]  # milliseconds
    lags_samples = [int(lag * sfreq / 1000) for lag in lags_ms]

    for lag in lags_samples:
        if len(epoch) > lag:
            autocorr = np.corrcoef(epoch[:-lag], epoch[lag:])[0, 1]
            features.append(0.0 if np.isnan(autocorr) else autocorr)
        else:
            features.append(0.0)

    return features


def compute_asymmetry_features(
    channels_data: dict[str, np.ndarray], sfreq: float
) -> list[float]:
    """
    Compute inter-hemispheric asymmetry features.

    Motor imagery typically shows lateralized patterns.
    """
    features = []

    bands = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "mu": (8, 12),
        "beta": (13, 30),
    }

    # C3-C4 asymmetry (critical for motor tasks)
    if "C3" in channels_data and "C4" in channels_data:
        c3_signal = channels_data["C3"]
        c4_signal = channels_data["C4"]

        for band_name, (low_freq, high_freq) in bands.items():
            c3_power = compute_band_power(c3_signal, sfreq, low_freq, high_freq)
            c4_power = compute_band_power(c4_signal, sfreq, low_freq, high_freq)

            # Asymmetry index: (right - left) / (right + left)
            total_power = c4_power + c3_power + 1e-10
            asymmetry_index = (c4_power - c3_power) / total_power
            features.append(asymmetry_index)

            # Log power ratio
            log_ratio = np.log((c3_power + 1e-10) / (c4_power + 1e-10))
            features.append(log_ratio)

    # Frontal-parietal asymmetry (attention/engagement indicator)
    if "Fz" in channels_data and "Pz" in channels_data:
        fz_signal = channels_data["Fz"]
        pz_signal = channels_data["Pz"]

        for band_name in ["theta", "alpha"]:
            low_freq, high_freq = bands[band_name]
            fz_power = compute_band_power(fz_signal, sfreq, low_freq, high_freq)
            pz_power = compute_band_power(pz_signal, sfreq, low_freq, high_freq)

            # Log power ratio
            log_ratio = np.log((fz_power + 1e-10) / (pz_power + 1e-10))
            features.append(log_ratio)

    return features


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
                    "baseline": baseline_power,
                    "task": task_power,
                    "erd": erd,
                }

            # Add time-domain features from task epoch
            activity, mobility, complexity = compute_hjorth_parameters(task)
            epoch_features.extend([activity, mobility, complexity])

            # Envelope features
            env_mean, env_std, env_max = compute_envelope_features(task)
            epoch_features.extend([env_mean, env_std, env_max])

            # Spectral entropy
            spec_entropy = compute_spectral_entropy(task, sfreq)
            epoch_features.append(spec_entropy)

            # Filter bank features (FBCSP-style) - key for BCI
            fb_features = compute_filter_bank_features(task, sfreq)
            epoch_features.extend(fb_features)

            # Temporal dynamics features
            temporal_feats = compute_temporal_features(task, sfreq)
            epoch_features.extend(temporal_feats)

        # Compute asymmetry features across channels for this epoch
        task_by_channel = {}
        for ch_name in motor_channels:
            if ch_name in epoch_pairs_by_channel:
                _, task = epoch_pairs_by_channel[ch_name][epoch_idx]
                task_by_channel[ch_name] = task

        asymmetry_feats = compute_asymmetry_features(task_by_channel, sfreq)
        epoch_features.extend(asymmetry_feats)

        # Inter-channel features: C3-C4 asymmetry for each band
        if "C3" in channel_powers and "C4" in channel_powers:
            for band_name in bands:
                c3_erd = channel_powers["C3"][band_name]["erd"]
                c4_erd = channel_powers["C4"][band_name]["erd"]
                epoch_features.append(c3_erd - c4_erd)

                # Power ratio
                c3_task = channel_powers["C3"][band_name]["task"]
                c4_task = channel_powers["C4"][band_name]["task"]
                epoch_features.append(np.log((c3_task + 1e-10) / (c4_task + 1e-10)))

        # Frontal-Parietal connectivity proxy (Fz vs Pz)
        if "Fz" in channel_powers and "Pz" in channel_powers:
            for band_name in ["theta", "mu", "beta"]:
                fz_task = channel_powers["Fz"][band_name]["task"]
                pz_task = channel_powers["Pz"][band_name]["task"]
                epoch_features.append(np.log((fz_task + 1e-10) / (pz_task + 1e-10)))

        # Central vs occipital (Cz vs Oz)
        if "Cz" in channel_powers and "Oz" in channel_powers:
            for band_name in ["mu", "beta"]:
                cz_task = channel_powers["Cz"][band_name]["task"]
                oz_task = channel_powers["Oz"][band_name]["task"]
                epoch_features.append(np.log((cz_task + 1e-10) / (oz_task + 1e-10)))

        all_features.append(epoch_features)

    return np.array(all_features)


def extract_realtime_features(
    window_by_channel: dict[str, np.ndarray],
    sfreq: float,
) -> np.ndarray:
    """
    Extract features from a single time window for real-time inference.

    Unlike extract_erd_features, this does NOT require a baseline epoch.
    Uses absolute features that can be computed from a single window.

    Args:
        window_by_channel: Dict mapping channel names to signal arrays
        sfreq: Sampling frequency in Hz

    Returns:
        1D feature vector
    """
    bands = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "low_beta": (13, 20),
        "high_beta": (20, 30),
        "beta": (13, 30),
        "gamma": (30, 40),
    }

    channels = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    features = []
    channel_powers = {}

    for ch_name in channels:
        if ch_name not in window_by_channel:
            continue

        signal = window_by_channel[ch_name]
        channel_powers[ch_name] = {}

        # Total power for normalization
        total_power = compute_band_power(signal, sfreq, 1, 40)

        for band_name, (low, high) in bands.items():
            power = compute_band_power(signal, sfreq, low, high)

            # Absolute log power
            features.append(np.log(power + 1e-10))

            # Relative power (proportion of total)
            rel_power = power / (total_power + 1e-10)
            features.append(rel_power)

            channel_powers[ch_name][band_name] = power

        # Theta/Alpha ratio (drowsiness indicator)
        theta = channel_powers[ch_name]["theta"]
        alpha = channel_powers[ch_name]["alpha"]
        features.append(theta / (alpha + 1e-10))

        # Theta/Beta ratio (attention indicator)
        beta = channel_powers[ch_name]["beta"]
        features.append(theta / (beta + 1e-10))

        # Hjorth parameters
        activity, mobility, complexity = compute_hjorth_parameters(signal)
        features.extend([activity, mobility, complexity])

        # Envelope features
        env_mean, env_std, env_max = compute_envelope_features(signal)
        features.extend([env_mean, env_std, env_max])

    # Inter-channel features
    if "C3" in channel_powers and "C4" in channel_powers:
        for band_name in bands:
            c3 = channel_powers["C3"][band_name]
            c4 = channel_powers["C4"][band_name]
            # Asymmetry
            features.append(np.log((c3 + 1e-10) / (c4 + 1e-10)))

    if "Fz" in channel_powers and "Pz" in channel_powers:
        for band_name in ["theta", "alpha", "beta"]:
            fz = channel_powers["Fz"][band_name]
            pz = channel_powers["Pz"][band_name]
            features.append(np.log((fz + 1e-10) / (pz + 1e-10)))

    return np.array(features)
