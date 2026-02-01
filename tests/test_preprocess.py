"""Tests for preprocessing functionality."""

import numpy as np

from src.preprocess import bandpass_filter, notch_filter, preprocess_eeg


def test_bandpass_filter_removes_dc_offset():
    """Test that bandpass filter removes DC component."""
    sfreq = 250.0
    # Create signal with DC offset + 10 Hz sine wave
    t = np.arange(0, 2, 1/sfreq)
    dc_offset = 73000  # Typical EEG offset in microvolts
    signal = dc_offset + 10 * np.sin(2 * np.pi * 10 * t)

    filtered = bandpass_filter(signal, sfreq, l_freq=1.0, h_freq=40.0)

    # DC offset should be removed (mean near zero)
    assert abs(np.mean(filtered)) < 100  # Much less than 73000


def test_notch_filter_removes_60hz():
    """Test that notch filter attenuates 60 Hz."""
    sfreq = 250.0
    t = np.arange(0, 2, 1/sfreq)
    # Signal with 10 Hz + 60 Hz noise
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)

    filtered = notch_filter(signal, sfreq, freq=60.0)

    # Compute power at 60 Hz before and after
    from scipy.signal import welch
    _, psd_before = welch(signal, sfreq, nperseg=250)
    _, psd_after = welch(filtered, sfreq, nperseg=250)

    # 60 Hz power should be reduced
    idx_60 = 60  # At 1 Hz resolution, index 60 is 60 Hz
    assert psd_after[idx_60] < psd_before[idx_60] * 0.1  # 90% reduction


def test_preprocess_eeg_full_pipeline():
    """Test full preprocessing pipeline."""
    sfreq = 250.0
    n_samples = 1000
    # Simulate raw EEG: DC offset + signal + 60 Hz noise
    np.random.seed(42)
    raw = 73000 + np.random.randn(n_samples) * 10 + 0.5 * np.sin(2 * np.pi * 60 * np.arange(n_samples) / sfreq)

    processed = preprocess_eeg(raw, sfreq)

    # Should be zero-centered and cleaned
    assert abs(np.mean(processed)) < 50
    assert processed.shape == raw.shape
