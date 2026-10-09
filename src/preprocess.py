"""EEG preprocessing: bandpass and notch filtering."""

import mne
import numpy as np


def preprocess_multichannel_eeg(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Bandpass (1-40 Hz) and notch (60 Hz) filter (n_channels, n_samples) EEG.

    Filtering operates over the last axis, so each channel is transformed
    independently while avoiding per-channel Python overhead.
    """
    filtered = mne.filter.filter_data(
        data,
        sfreq,
        l_freq=1.0,
        h_freq=40.0,
        verbose=False,
    )
    return mne.filter.notch_filter(filtered, sfreq, freqs=60.0, verbose=False)
