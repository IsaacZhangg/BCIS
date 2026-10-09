"""Tests for preprocessing functionality."""

import numpy as np
from scipy.signal import welch

from src.preprocess import preprocess_multichannel_eeg


def test_preprocess_removes_dc_offset_and_60hz():
    """Bandpass removes the DC offset and the notch attenuates 60 Hz per channel."""
    sfreq = 250.0
    n_channels = 8
    n_samples = 1000
    rng = np.random.default_rng(123)
    t = np.arange(n_samples) / sfreq
    raw = (
        73000
        + rng.standard_normal((n_channels, n_samples)) * 10
        + np.sin(2 * np.pi * 10 * t)[None, :]
        + 0.5 * np.sin(2 * np.pi * 60 * t)[None, :]
    )

    processed = preprocess_multichannel_eeg(raw, sfreq)

    assert processed.shape == raw.shape
    assert np.all(np.abs(processed.mean(axis=1)) < 50)
    freqs, psd_before = welch(raw - raw.mean(axis=1, keepdims=True), sfreq, nperseg=250)
    _, psd_after = welch(processed, sfreq, nperseg=250)
    idx_60 = int(np.argmin(np.abs(freqs - 60)))
    assert np.all(psd_after[:, idx_60] < psd_before[:, idx_60] * 0.1)
