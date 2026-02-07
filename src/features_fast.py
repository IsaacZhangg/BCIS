"""Optimized feature extraction for real-time EEG engagement detection.

Key optimizations over features.py:
- Single Welch PSD per channel, band powers extracted by masking frequency bins
- Vectorized across all 8 channels using 2D numpy operations (no per-channel loops)
- Pre-computed Welch parameters as module constants
- Reduced feature count (~96 vs 169) by removing redundant/overlapping features
- Filter bank replaced by FFT bin masking (eliminates 16 bandpass filtfilt calls)

Drop-in replacement: extract_realtime_features_fast() has same interface as
extract_realtime_features() from features.py.
"""

import numpy as np
from scipy.signal import welch

# --- Constants (pre-computed at import time) ---

CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

# Welch parameters for 5s window (1250 samples)
_NPERSEG = 125  # 0.5s segments = sfreq/2
_NOVERLAP = _NPERSEG // 2

# Non-overlapping frequency bands (no redundant beta = low_beta + high_beta)
BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "low_beta": (13.0, 20.0),
    "high_beta": (20.0, 30.0),
    "gamma": (30.0, 40.0),
}
N_BANDS = len(BANDS)
BAND_NAMES = list(BANDS.keys())

# Total power range
TOTAL_BAND = (1.0, 40.0)

# FBCSP-style filter bank: 10 overlapping sub-bands covering 4-38 Hz
FILTER_BANK_BANDS = [
    (4, 8),
    (8, 12),
    (10, 14),
    (14, 18),
    (18, 22),
    (22, 26),
    (24, 28),
    (28, 32),
    (30, 34),
    (34, 38),
]
N_FB_BANDS = len(FILTER_BANK_BANDS)

# Pre-computed band indices for ratio calculations
_THETA_IDX = BAND_NAMES.index("theta")
_ALPHA_IDX = BAND_NAMES.index("alpha")
_LOW_BETA_IDX = BAND_NAMES.index("low_beta")
_HIGH_BETA_IDX = BAND_NAMES.index("high_beta")


def _welch_psd_batch(
    data_2d: np.ndarray, sfreq: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD for all channels at once.

    Args:
        data_2d: (n_channels, n_samples) array
        sfreq: Sampling frequency

    Returns:
        freqs: 1D frequency array
        psd: (n_channels, n_freqs) power spectral density
    """
    return welch(
        data_2d,
        fs=sfreq,
        nperseg=_NPERSEG,
        noverlap=_NOVERLAP,
        axis=-1,
    )


def _band_power_from_psd(
    psd: np.ndarray,
    freqs: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """Extract mean band power from pre-computed PSD by masking freq bins.

    Args:
        psd: (n_channels, n_freqs) or (n_freqs,) PSD array
        freqs: 1D frequency array
        low: Lower frequency bound
        high: Upper frequency bound

    Returns:
        Band power per channel (n_channels,) or scalar
    """
    mask = (freqs >= low) & (freqs <= high)

    if psd.ndim == 1:
        return np.mean(psd[mask])

    return np.mean(psd[:, mask], axis=1)


def _hjorth_parameters_batch(
    data_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Hjorth parameters for all channels at once.

    Args:
        data_2d: (n_channels, n_samples)

    Returns:
        activity: (n_channels,) variance
        mobility: (n_channels,) sqrt(var(d1) / var(signal))
        complexity: (n_channels,) mobility(d1) / mobility(signal)
    """
    activity = np.var(data_2d, axis=1)
    d1 = np.diff(data_2d, axis=1)
    d2 = np.diff(d1, axis=1)
    var_d1 = np.var(d1, axis=1)
    var_d2 = np.var(d2, axis=1)
    mobility = np.sqrt(var_d1 / (activity + 1e-10))
    mob_d1 = np.sqrt(var_d2 / (var_d1 + 1e-10))
    complexity = mob_d1 / (mobility + 1e-10)
    return activity, mobility, complexity


def _filter_bank_from_psd(
    psd: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """Compute FBCSP-style log-variance features from PSD (replaces 16 filtfilt calls).

    Instead of bandpass-filtering and computing log(var(filtered)), we use
    Parseval's theorem: the total power in a band (integral of PSD) is
    proportional to the variance of the bandpassed signal. We compute
    log(sum(PSD[band_mask])) which is equivalent to log(var) up to a
    constant scaling factor.

    Args:
        psd: (n_channels, n_freqs) power spectral density
        freqs: 1D frequency array

    Returns:
        (n_channels, n_fb_bands) log-power features
    """
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    result = np.empty((psd.shape[0], N_FB_BANDS))
    for i, (low, high) in enumerate(FILTER_BANK_BANDS):
        mask = (freqs >= low) & (freqs <= high)
        # Sum PSD in band * freq_resolution approximates variance of bandpassed signal
        band_power = np.sum(psd[:, mask], axis=1) * freq_resolution
        result[:, i] = np.log(band_power + 1e-10)
    return result


def _prepare_channel_data(
    window_by_channel: dict[str, np.ndarray],
    sfreq: float,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build 2D channel array and compute PSD, band powers, total power, Hjorth.

    Returns:
        present_channels, band_powers, total_power, (activity, mobility, complexity)
        packed as (present_channels, band_powers, total_power, hjorth_tuple, psd_data)
    """
    present_channels = [ch for ch in CHANNELS if ch in window_by_channel]
    n_ch = len(present_channels)
    n_samples = len(window_by_channel[present_channels[0]])
    data_2d = np.empty((n_ch, n_samples))
    for i, ch in enumerate(present_channels):
        data_2d[i] = window_by_channel[ch]

    freqs, psd = _welch_psd_batch(data_2d, sfreq)

    band_powers = np.empty((n_ch, N_BANDS))
    for j, (low, high) in enumerate(BANDS.values()):
        band_powers[:, j] = _band_power_from_psd(psd, freqs, low, high)

    total_power = _band_power_from_psd(psd, freqs, TOTAL_BAND[0], TOTAL_BAND[1])
    activity, mobility, complexity = _hjorth_parameters_batch(data_2d)

    return (
        present_channels,
        band_powers,
        total_power,
        activity,
        mobility,
        complexity,
        freqs,
        psd,
    )


def _per_channel_features(
    band_powers: np.ndarray,
    total_power: np.ndarray,
    activity: np.ndarray,
    mobility: np.ndarray,
    complexity: np.ndarray,
) -> list[float]:
    """Build per-channel feature list: log power, relative power, ratios, Hjorth.

    Per channel: log_power(5) + rel_power(5) + theta/alpha(1) + theta/beta(1) + hjorth(3) = 15
    """
    n_ch = band_powers.shape[0]
    log_powers = np.log(band_powers + 1e-10)
    rel_powers = band_powers / (total_power[:, None] + 1e-10)

    beta_power = band_powers[:, _LOW_BETA_IDX] + band_powers[:, _HIGH_BETA_IDX]
    theta_alpha_ratio = band_powers[:, _THETA_IDX] / (
        band_powers[:, _ALPHA_IDX] + 1e-10
    )
    theta_beta_ratio = band_powers[:, _THETA_IDX] / (beta_power + 1e-10)

    features = []
    for i in range(n_ch):
        features.extend(log_powers[i])
        features.extend(rel_powers[i])
        features.append(theta_alpha_ratio[i])
        features.append(theta_beta_ratio[i])
        features.append(activity[i])
        features.append(mobility[i])
        features.append(complexity[i])

    return features


def _inter_channel_features(
    present_channels: list[str],
    band_powers: np.ndarray,
) -> list[float]:
    """Compute C3-C4 asymmetry and Fz-Pz ratio features."""
    ch_map = {ch: i for i, ch in enumerate(present_channels)}
    features = []

    # C3-C4 asymmetry (5 bands)
    if "C3" in ch_map and "C4" in ch_map:
        c3_idx = ch_map["C3"]
        c4_idx = ch_map["C4"]
        for j in range(N_BANDS):
            features.append(
                np.log(
                    (band_powers[c3_idx, j] + 1e-10) / (band_powers[c4_idx, j] + 1e-10)
                )
            )

    # Fz-Pz ratio (theta, alpha, combined beta = 3)
    if "Fz" in ch_map and "Pz" in ch_map:
        fz_idx = ch_map["Fz"]
        pz_idx = ch_map["Pz"]

        for band_idx in [_THETA_IDX, _ALPHA_IDX]:
            features.append(
                np.log(
                    (band_powers[fz_idx, band_idx] + 1e-10)
                    / (band_powers[pz_idx, band_idx] + 1e-10)
                )
            )

        # Combined beta ratio
        fz_beta = (
            band_powers[fz_idx, _LOW_BETA_IDX] + band_powers[fz_idx, _HIGH_BETA_IDX]
        )
        pz_beta = (
            band_powers[pz_idx, _LOW_BETA_IDX] + band_powers[pz_idx, _HIGH_BETA_IDX]
        )
        features.append(np.log((fz_beta + 1e-10) / (pz_beta + 1e-10)))

    return features


def extract_realtime_features_fast(
    window_by_channel: dict[str, np.ndarray],
    sfreq: float,
) -> np.ndarray:
    """Extract features from a single time window for real-time inference.

    Optimized drop-in replacement for extract_realtime_features().
    Same interface: takes channel dict, returns 1D feature vector.

    Feature layout (128 features with all 8 channels):
    - Per channel (8 ch x 15 = 120):
        - 5 bands x log(power)
        - 5 bands x relative_power
        - theta/alpha ratio
        - theta/beta ratio
        - Hjorth: activity, mobility, complexity
    - Inter-channel (8):
        - C3-C4 asymmetry: 5 bands
        - Fz-Pz ratio: 3 bands (theta, alpha, combined beta)

    Args:
        window_by_channel: Dict mapping channel names to signal arrays
        sfreq: Sampling frequency in Hz

    Returns:
        1D feature vector
    """
    (
        present_channels,
        band_powers,
        total_power,
        activity,
        mobility,
        complexity,
        freqs,
        psd,
    ) = _prepare_channel_data(window_by_channel, sfreq)

    features = _per_channel_features(
        band_powers, total_power, activity, mobility, complexity
    )
    features.extend(_inter_channel_features(present_channels, band_powers))

    return np.array(features)


def extract_realtime_features_fast_v2(
    window_by_channel: dict[str, np.ndarray],
    sfreq: float,
) -> np.ndarray:
    """Extended feature set with filter bank from PSD.

    Adds FBCSP-style filter bank features on top of v1 features.
    Uses PSD bin masking instead of separate filtfilt calls.

    Args:
        window_by_channel: Dict mapping channel names to signal arrays
        sfreq: Sampling frequency in Hz

    Returns:
        1D feature vector
    """
    (
        present_channels,
        band_powers,
        total_power,
        activity,
        mobility,
        complexity,
        freqs,
        psd,
    ) = _prepare_channel_data(window_by_channel, sfreq)

    features = _per_channel_features(
        band_powers, total_power, activity, mobility, complexity
    )

    # Filter bank features (flattened): 10 per channel
    fb_features = _filter_bank_from_psd(psd, freqs)
    for i in range(fb_features.shape[0]):
        features.extend(fb_features[i])

    features.extend(_inter_channel_features(present_channels, band_powers))

    return np.array(features)
