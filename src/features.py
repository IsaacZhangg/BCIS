"""Feature extraction: multi-channel, multi-band power computation with ERD."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import coherence, welch
from scipy.stats import kurtosis, skew

if TYPE_CHECKING:
    from mne.decoding import CSP


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


def compute_spectral_entropy(
    epoch: np.ndarray,
    sfreq: float,
    low: float,
    high: float,
) -> float:
    """Compute normalised spectral entropy in [low, high] Hz.

    Returns a value in [0, 1]: low entropy indicates a strong narrow-band
    rhythm; high entropy indicates broadband noise.
    """
    nperseg = min(int(sfreq / 2), len(epoch) // 2)
    nperseg = max(nperseg, 64)
    freqs, psd = welch(epoch, sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= low) & (freqs <= high)
    psd_band = psd[mask]
    psd_norm = psd_band / (psd_band.sum() + 1e-10)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    max_entropy = np.log2(len(psd_norm)) if len(psd_norm) > 0 else 1.0
    return float(entropy / (max_entropy + 1e-10))


def compute_peak_frequency(
    epoch: np.ndarray,
    sfreq: float,
    low: float,
    high: float,
) -> float:
    """Return the peak frequency in [low, high] Hz."""
    nperseg = min(int(sfreq / 2), len(epoch) // 2)
    nperseg = max(nperseg, 64)
    freqs, psd = welch(epoch, sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= low) & (freqs <= high)
    freqs_band = freqs[mask]
    psd_band = psd[mask]
    return float(freqs_band[np.argmax(psd_band)])


def compute_c3c4_coherence(
    c3: np.ndarray,
    c4: np.ndarray,
    sfreq: float,
    low: float,
    high: float,
) -> float:
    """Mean magnitude-squared coherence between C3 and C4 in [low, high] Hz."""
    nperseg = min(int(sfreq / 2), len(c3) // 2)
    nperseg = max(nperseg, 64)
    freqs, coh = coherence(c3, c4, sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= low) & (freqs <= high)
    return float(np.mean(coh[mask]))


def compute_statistical_features(epoch: np.ndarray) -> tuple[float, float, float]:
    """Return (skewness, kurtosis, zero-crossing rate) for *epoch*."""
    sk = float(skew(epoch))
    ku = float(kurtosis(epoch))  # Fisher definition (normal → 0)
    zcr = float(np.sum(np.diff(np.sign(epoch)) != 0)) / (len(epoch) - 1)
    return sk, ku, zcr


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
    powers = [compute_theta_power(epoch, sfreq) for epoch in epochs]
    if log_transform:
        powers = [np.log(p + 1e-10) for p in powers]
    return np.array(powers).reshape(-1, 1)


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

    Raises:
        ValueError: If required channels (C3, C4, Cz, Fz) are missing from input
    """
    # Validate required channels exist
    required_channels = {"C3", "C4", "Cz", "Fz", "PO7", "PO8"}
    available_channels = set(epoch_pairs_by_channel.keys())
    missing_channels = required_channels - available_channels
    if missing_channels:
        raise ValueError(
            f"Missing required channels for lateralization features: {sorted(missing_channels)}. "
            f"Required: {sorted(required_channels)}, Available: {sorted(available_channels)}"
        )

    # Key frequency bands for motor imagery
    bands = {
        "mu": (8, 12),
        "low_beta": (13, 20),
        "high_beta": (20, 30),
        "beta": (13, 30),
    }

    n_epochs = len(epoch_pairs_by_channel["C3"])

    all_features = []

    for epoch_idx in range(n_epochs):
        epoch_features = []

        # Get C3 and C4 signals for this epoch
        c3_baseline, c3_task = epoch_pairs_by_channel["C3"][epoch_idx]
        c4_baseline, c4_task = epoch_pairs_by_channel["C4"][epoch_idx]

        # PRIMARY: Lateralization features (C3 vs C4)
        for _band_name, (low, high) in bands.items():
            # Compute powers once per band (avoids redundant Welch calls)
            c3_bp = compute_band_power(c3_baseline, sfreq, low, high)
            c3_tp = compute_band_power(c3_task, sfreq, low, high)
            c4_bp = compute_band_power(c4_baseline, sfreq, low, high)
            c4_tp = compute_band_power(c4_task, sfreq, low, high)

            # Lateralization indices: (C4 - C3) / (C4 + C3)
            lat_task = (c4_tp - c3_tp) / (c4_tp + c3_tp + 1e-10)
            lat_baseline = (c4_bp - c3_bp) / (c4_bp + c3_bp + 1e-10)

            # ERD per channel: % power decrease from baseline
            c3_erd = (c3_bp - c3_tp) / (c3_bp + 1e-10) * 100
            c4_erd = (c4_bp - c4_tp) / (c4_bp + 1e-10) * 100

            epoch_features.extend(
                [
                    lat_task,
                    lat_baseline,
                    lat_task - lat_baseline,
                    c3_erd - c4_erd,
                    np.log((c3_tp + 1e-10) / (c4_tp + 1e-10)),
                    c3_erd,
                    c4_erd,
                ]
            )

        # SECONDARY: Cz features (supplementary motor area)
        # Note: Cz uses only mu and beta bands (not all 4 bands like C3/C4) because:
        # - Cz sits over the supplementary motor area, not primary motor cortex
        # - Mu (8-12Hz) captures motor planning activity
        # - Beta (13-30Hz, combined) is sufficient for SMA; splitting into low/high adds noise
        cz_baseline, cz_task = epoch_pairs_by_channel["Cz"][epoch_idx]
        for band_name, (low, high) in [("mu", (8, 12)), ("beta", (13, 30))]:
            cz_baseline_power = compute_band_power(cz_baseline, sfreq, low, high)
            cz_task_power = compute_band_power(cz_task, sfreq, low, high)
            cz_erd = (
                (cz_baseline_power - cz_task_power) / (cz_baseline_power + 1e-10) * 100
            )
            epoch_features.append(cz_erd)
            epoch_features.append(np.log(cz_task_power + 1e-10))

        # TERTIARY: Fz theta (attention/effort marker)
        # Note: Fz uses only theta band (4-8Hz) because:
        # - Frontal theta is a well-established marker of cognitive effort and attention
        # - Motor imagery requires attention, and frontal theta increases with task demands
        # - Mu/beta bands at Fz don't reflect motor-specific activity (Fz is frontal, not motor)
        fz_baseline, fz_task = epoch_pairs_by_channel["Fz"][epoch_idx]
        fz_theta_baseline = compute_band_power(fz_baseline, sfreq, 4, 8)
        fz_theta_task = compute_band_power(fz_task, sfreq, 4, 8)
        epoch_features.append(
            np.log((fz_theta_task + 1e-10) / (fz_theta_baseline + 1e-10))
        )

        # Time-domain features from C3 and C4
        for signal in [c3_task, c4_task]:
            activity, mobility, complexity = compute_hjorth_parameters(signal)
            epoch_features.extend([activity, mobility, complexity])

        # --- NEW: Spectral entropy for C3/C4 in mu and beta (4 features) ---
        for sig, _name in [(c3_task, "C3"), (c4_task, "C4")]:
            epoch_features.append(compute_spectral_entropy(sig, sfreq, 8, 12))
            epoch_features.append(compute_spectral_entropy(sig, sfreq, 13, 30))

        # --- NEW: Peak frequency for C3/C4 in mu band (2 features) ---
        epoch_features.append(compute_peak_frequency(c3_task, sfreq, 8, 12))
        epoch_features.append(compute_peak_frequency(c4_task, sfreq, 8, 12))

        # --- NEW: C3-C4 coherence in mu and beta (2 features) ---
        epoch_features.append(compute_c3c4_coherence(c3_task, c4_task, sfreq, 8, 12))
        epoch_features.append(compute_c3c4_coherence(c3_task, c4_task, sfreq, 13, 30))

        # --- NEW: Posterior lateralization (PO7/PO8) (4 features) ---
        po7_task = epoch_pairs_by_channel["PO7"][epoch_idx][1]
        po8_task = epoch_pairs_by_channel["PO8"][epoch_idx][1]
        for low, high in [(8, 12), (13, 30)]:
            po7_power = compute_band_power(po7_task, sfreq, low, high)
            po8_power = compute_band_power(po8_task, sfreq, low, high)
            lat_po = (po8_power - po7_power) / (po8_power + po7_power + 1e-10)
            epoch_features.append(lat_po)
            epoch_features.append(np.log(po7_power + po8_power + 1e-10))

        # --- NEW: Band power ratios for C3 (2 features) ---
        c3_theta = compute_band_power(c3_task, sfreq, 4, 8)
        c3_mu = compute_band_power(c3_task, sfreq, 8, 12)
        c3_beta = compute_band_power(c3_task, sfreq, 13, 30)
        epoch_features.append(np.log((c3_mu + 1e-10) / (c3_theta + 1e-10)))
        epoch_features.append(np.log((c3_beta + 1e-10) / (c3_mu + 1e-10)))

        # --- NEW: Statistical features for C3/C4 (6 features) ---
        for sig in [c3_task, c4_task]:
            sk, ku, zcr = compute_statistical_features(sig)
            epoch_features.extend([sk, ku, zcr])

        all_features.append(epoch_features)

    return np.array(all_features)


def extract_csp_features(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    n_components: int = 4,
    freq_band: tuple[float, float] = (8, 30),
) -> tuple[np.ndarray, CSP]:
    """
    Extract CSP (Common Spatial Pattern) features for motor imagery.

    CSP finds spatial filters that maximize variance difference between classes,
    making it ideal for left/right motor imagery where the difference is in
    spatial distribution of mu/beta desynchronization.

    Args:
        X: Multichannel EEG data of shape (n_epochs, n_channels, n_samples)
        y: Labels of shape (n_epochs,)
        sfreq: Sampling frequency in Hz
        n_components: Number of CSP components (filters) to use
        freq_band: Frequency band to filter before CSP (default mu+beta)

    Returns:
        Tuple of (features array of shape (n_epochs, n_components), fitted CSP model)
    """
    import mne
    from mne.decoding import CSP

    # Bandpass filter to mu+beta range before CSP
    X_filtered = mne.filter.filter_data(
        X, sfreq, l_freq=freq_band[0], h_freq=freq_band[1], verbose=False
    )

    # Fit CSP - finds spatial filters maximizing class separability
    csp = CSP(
        n_components=n_components,
        reg="ledoit_wolf",  # Regularization for robust covariance estimation
        log=True,  # Log-transform variance features
        norm_trace=True,  # Normalize for scale invariance
    )
    features = csp.fit_transform(X_filtered, y)

    return features, csp
