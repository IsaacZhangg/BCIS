"""EEG preprocessing: filtering and artifact removal."""

import numpy as np
import mne


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
    return mne.filter.filter_data(
        data.reshape(1, -1), sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False
    ).flatten()


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
    return mne.filter.notch_filter(
        data.reshape(1, -1), sfreq, freqs=freq, verbose=False
    ).flatten()


def preprocess_eeg(data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Full preprocessing pipeline: bandpass + notch filter.

    Args:
        data: 1D signal array (single channel)
        sfreq: Sampling frequency in Hz

    Returns:
        Preprocessed signal
    """
    filtered = bandpass_filter(data, sfreq)
    return notch_filter(filtered, sfreq)
