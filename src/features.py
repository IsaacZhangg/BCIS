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
    """Compute mean power in [low_freq, high_freq] Hz using Welch's method."""
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
    activity = np.var(epoch)
    diff1 = np.diff(epoch)
    var_diff1 = np.var(diff1)
    diff2 = np.diff(diff1)
    var_diff2 = np.var(diff2)
    mobility = np.sqrt(var_diff1 / (activity + 1e-10))
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
    """Extract theta power features from a list of epochs (backward compatible)."""
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
    """Compute C3-C4 lateralization index: (C4-C3)/(C4+C3) power in band.

    Positive = left hand imagery (C3 desync), negative = right hand imagery (C4 desync).
    """
    c3_power = compute_band_power(c3_signal, sfreq, low_freq, high_freq)
    c4_power = compute_band_power(c4_signal, sfreq, low_freq, high_freq)

    return (c4_power - c3_power) / (c4_power + c3_power + 1e-10)


def extract_lateralization_features(
    epoch_pairs_by_channel: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    sfreq: float,
) -> np.ndarray:
    """Extract features optimized for left/right motor imagery discrimination.

    Focuses on C3-C4 lateralization in mu (8-12Hz) and beta (13-30Hz) bands.
    Returns feature array of shape (n_epochs, n_features).
    """
    required_channels = {"C3", "C4", "Cz", "Fz", "Pz", "PO7", "PO8"}
    available_channels = set(epoch_pairs_by_channel.keys())
    missing_channels = required_channels - available_channels
    if missing_channels:
        raise ValueError(
            f"Missing required channels for lateralization features: {sorted(missing_channels)}. "
            f"Required: {sorted(required_channels)}, Available: {sorted(available_channels)}"
        )

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

        c3_baseline, c3_task = epoch_pairs_by_channel["C3"][epoch_idx]
        c4_baseline, c4_task = epoch_pairs_by_channel["C4"][epoch_idx]
        cz_baseline, cz_task = epoch_pairs_by_channel["Cz"][epoch_idx]
        fz_baseline, fz_task_sig = epoch_pairs_by_channel["Fz"][epoch_idx]
        po7_baseline, po7_task = epoch_pairs_by_channel["PO7"][epoch_idx]
        pz_baseline, pz_task = epoch_pairs_by_channel["Pz"][epoch_idx]
        po8_baseline, po8_task = epoch_pairs_by_channel["PO8"][epoch_idx]

        # Surface Laplacian: sharpen spatial resolution for motor cortex channels
        # C3 neighbors in Unicorn montage: Fz, Cz, PO7
        # C4 neighbors in Unicorn montage: Cz, Pz, PO8
        c3_baseline_lap = c3_baseline - (fz_baseline + cz_baseline + po7_baseline) / 3
        c3_task_lap = c3_task - (fz_task_sig + cz_task + po7_task) / 3
        c4_baseline_lap = c4_baseline - (cz_baseline + pz_baseline + po8_baseline) / 3
        c4_task_lap = c4_task - (cz_task + pz_task + po8_task) / 3

        # Lateralization features from Laplacian-filtered C3/C4
        for c3b, c3t, c4b, c4t in [
            (c3_baseline_lap, c3_task_lap, c4_baseline_lap, c4_task_lap),
        ]:
            for _band_name, (low, high) in bands.items():
                c3_bp = compute_band_power(c3b, sfreq, low, high)
                c3_tp = compute_band_power(c3t, sfreq, low, high)
                c4_bp = compute_band_power(c4b, sfreq, low, high)
                c4_tp = compute_band_power(c4t, sfreq, low, high)

                lat_task = (c4_tp - c3_tp) / (c4_tp + c3_tp + 1e-10)
                lat_baseline = (c4_bp - c3_bp) / (c4_bp + c3_bp + 1e-10)

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

        # Cz (supplementary motor area): only mu + combined beta — SMA doesn't
        # benefit from the low/high beta split used for primary motor cortex
        for band_name, (low, high) in [("mu", (8, 12)), ("beta", (13, 30))]:
            cz_baseline_power = compute_band_power(cz_baseline, sfreq, low, high)
            cz_task_power = compute_band_power(cz_task, sfreq, low, high)
            cz_erd = (
                (cz_baseline_power - cz_task_power) / (cz_baseline_power + 1e-10) * 100
            )
            epoch_features.append(cz_erd)
            epoch_features.append(np.log(cz_task_power + 1e-10))

        # Fz theta (4-8Hz): frontal theta tracks cognitive effort/attention;
        # mu/beta at Fz don't carry motor-specific information
        fz_theta_baseline = compute_band_power(fz_baseline, sfreq, 4, 8)
        fz_theta_task = compute_band_power(fz_task_sig, sfreq, 4, 8)
        epoch_features.append(
            np.log((fz_theta_task + 1e-10) / (fz_theta_baseline + 1e-10))
        )

        for signal in [c3_task_lap, c4_task_lap]:
            activity, mobility, complexity = compute_hjorth_parameters(signal)
            epoch_features.extend([activity, mobility, complexity])

        # C3-C4 inter-hemispheric coherence (Laplacian-filtered)
        # Lower coherence during lateralized motor imagery
        for low, high in [(8, 12), (13, 30)]:
            coh_task = compute_c3c4_coherence(
                c3_task_lap, c4_task_lap, sfreq, low, high
            )
            coh_baseline = compute_c3c4_coherence(
                c3_baseline_lap, c4_baseline_lap, sfreq, low, high
            )
            epoch_features.append(coh_task)
            epoch_features.append(coh_task - coh_baseline)

        # Spectral entropy lateralization (Laplacian-filtered)
        # Narrow-band mu/beta suppression → lower entropy on contralateral side
        for low, high in [(8, 12), (13, 30)]:
            c3_ent = compute_spectral_entropy(c3_task_lap, sfreq, low, high)
            c4_ent = compute_spectral_entropy(c4_task_lap, sfreq, low, high)
            epoch_features.append(c3_ent - c4_ent)

        all_features.append(epoch_features)

    return np.array(all_features)


def extract_csp_features(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    n_components: int = 4,
    freq_band: tuple[float, float] = (8, 30),
) -> tuple[np.ndarray, CSP]:
    """Extract CSP features: bandpass to freq_band, then fit-transform CSP.

    Returns (features of shape (n_epochs, n_components), fitted CSP model).
    """
    import mne
    from mne.decoding import CSP

    X_filtered = mne.filter.filter_data(
        X, sfreq, l_freq=freq_band[0], h_freq=freq_band[1], verbose=False
    )

    csp = CSP(
        n_components=n_components,
        reg="oas",
        log=True,
        norm_trace=True,
    )
    features = csp.fit_transform(X_filtered, y)

    return features, csp
