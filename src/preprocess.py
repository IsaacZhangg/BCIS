"""EEG preprocessing: filtering and artifact removal."""

import numpy as np
import mne


def apply_car(data: np.ndarray) -> np.ndarray:
    """
    Apply Common Average Reference (CAR) spatial filter.

    Args:
        data: 2D array (n_channels, n_samples)

    Returns:
        CAR-filtered data
    """
    mean = np.mean(data, axis=0, keepdims=True)
    return data - mean


def apply_laplacian(data: np.ndarray, channel_names: list[str]) -> np.ndarray:
    """
    Apply surface Laplacian (approximate) for motor cortex channels.

    Uses local spatial gradients to reduce volume conduction.

    Args:
        data: 2D array (n_channels, n_samples)
        channel_names: List of channel names

    Returns:
        Laplacian-filtered data
    """
    # Define neighbors for each channel based on 10-20 system
    # Fz, C3, Cz, C4, Pz, PO7, Oz, PO8
    neighbors = {
        "C3": ["Fz", "Cz"],  # C3 surrounded by Fz, Cz
        "Cz": ["Fz", "C3", "C4", "Pz"],
        "C4": ["Fz", "Cz"],
    }

    result = data.copy()
    ch_idx = {name: i for i, name in enumerate(channel_names)}

    for ch_name, neighbor_names in neighbors.items():
        if ch_name not in ch_idx:
            continue
        valid_neighbors = [n for n in neighbor_names if n in ch_idx]
        if len(valid_neighbors) >= 2:
            neighbor_mean = np.mean([data[ch_idx[n]] for n in valid_neighbors], axis=0)
            result[ch_idx[ch_name]] = data[ch_idx[ch_name]] - neighbor_mean

    return result


def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
) -> np.ndarray:
    """
    Apply bandpass filter to remove DC drift and high-frequency noise.

    Args:
        data: 1D signal array
        sfreq: Sampling frequency in Hz
        l_freq: Low cutoff frequency (Hz)
        h_freq: High cutoff frequency (Hz)

    Returns:
        Filtered signal
    """
    # MNE filter expects 2D array (n_channels, n_samples)
    data_2d = data.reshape(1, -1)
    filtered = mne.filter.filter_data(
        data_2d, sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False
    )
    return filtered.flatten()


def notch_filter(
    data: np.ndarray,
    sfreq: float,
    freq: float = 60.0,
) -> np.ndarray:
    """
    Apply notch filter to remove power line interference.

    Args:
        data: 1D signal array
        sfreq: Sampling frequency in Hz
        freq: Frequency to notch out (Hz)

    Returns:
        Filtered signal
    """
    data_2d = data.reshape(1, -1)
    filtered = mne.filter.notch_filter(data_2d, sfreq, freqs=freq, verbose=False)
    return filtered.flatten()


def preprocess_eeg(data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Full preprocessing pipeline: bandpass + notch filter.

    Args:
        data: 1D signal array (single channel)
        sfreq: Sampling frequency in Hz

    Returns:
        Preprocessed signal
    """
    # Bandpass filter (1-40 Hz)
    filtered = bandpass_filter(data, sfreq, l_freq=1.0, h_freq=40.0)
    # Notch filter (60 Hz)
    filtered = notch_filter(filtered, sfreq, freq=60.0)
    return filtered


def preprocess_eeg_multichannel(
    data: np.ndarray,
    sfreq: float,
    channel_names: list[str],
    spatial_filter: str = "car",
) -> np.ndarray:
    """
    Full preprocessing pipeline for multichannel data with spatial filtering.

    Args:
        data: 2D array (n_channels, n_samples)
        sfreq: Sampling frequency in Hz
        channel_names: List of channel names
        spatial_filter: Type of spatial filter ("car", "laplacian", or "none")

    Returns:
        Preprocessed data (n_channels, n_samples)
    """
    # Bandpass filter each channel
    filtered = np.zeros_like(data)
    for ch_idx in range(data.shape[0]):
        filtered[ch_idx] = bandpass_filter(data[ch_idx], sfreq, l_freq=1.0, h_freq=40.0)
        filtered[ch_idx] = notch_filter(filtered[ch_idx], sfreq, freq=60.0)

    # Apply spatial filter
    if spatial_filter == "car":
        filtered = apply_car(filtered)
    elif spatial_filter == "laplacian":
        filtered = apply_laplacian(filtered, channel_names)

    return filtered
