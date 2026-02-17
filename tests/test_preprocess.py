"""Tests for preprocessing functionality."""

import numpy as np
from scipy.signal import welch

from src.preprocess import (
    apply_asr,
    bandpass_filter,
    common_average_reference,
    notch_filter,
    preprocess_eeg,
    preprocess_multichannel_eeg,
)


def test_bandpass_filter_removes_dc_offset():
    """Test that bandpass filter removes DC component."""
    sfreq = 250.0
    # Create signal with DC offset + 10 Hz sine wave
    t = np.arange(0, 2, 1 / sfreq)
    dc_offset = 73000  # Typical EEG offset in microvolts
    signal = dc_offset + 10 * np.sin(2 * np.pi * 10 * t)

    filtered = bandpass_filter(signal, sfreq, l_freq=1.0, h_freq=40.0)

    # DC offset should be removed (mean near zero)
    assert abs(np.mean(filtered)) < 100  # Much less than 73000


def test_notch_filter_removes_60hz():
    """Test that notch filter attenuates 60 Hz."""
    sfreq = 250.0
    t = np.arange(0, 2, 1 / sfreq)
    # Signal with 10 Hz + 60 Hz noise
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)

    filtered = notch_filter(signal, sfreq, freq=60.0)

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
    raw = (
        73000
        + np.random.randn(n_samples) * 10
        + 0.5 * np.sin(2 * np.pi * 60 * np.arange(n_samples) / sfreq)
    )

    processed = preprocess_eeg(raw, sfreq)

    # Should be zero-centered and cleaned
    assert abs(np.mean(processed)) < 50
    assert processed.shape == raw.shape


def test_multichannel_preprocess_matches_per_channel_pipeline():
    """Batch preprocessing should match per-channel preprocessing semantics."""
    sfreq = 250.0
    n_channels = 8
    n_samples = 1000
    rng = np.random.default_rng(123)
    raw = (
        73000
        + rng.standard_normal((n_channels, n_samples)) * 10
        + 0.5 * np.sin(2 * np.pi * 60 * np.arange(n_samples) / sfreq)[None, :]
    )

    batched = preprocess_multichannel_eeg(raw, sfreq)
    per_channel = np.vstack(
        [preprocess_eeg(raw[ch], sfreq) for ch in range(n_channels)]
    )

    np.testing.assert_allclose(batched, per_channel, atol=1e-10, rtol=1e-8)


def test_common_average_reference_removes_common_signal():
    """CAR removes signal shared across all channels."""
    n_channels, n_samples = 8, 500
    rng = np.random.default_rng(42)

    # Independent channel activity
    unique = rng.standard_normal((n_channels, n_samples)) * 5

    # Common-mode noise (same on every channel)
    common = rng.standard_normal(n_samples) * 50

    data = unique + common

    filtered = common_average_reference(data)

    # Mean across channels should be ~0 at every time point
    assert np.allclose(filtered.mean(axis=0), 0, atol=1e-10)

    # Shape preserved
    assert filtered.shape == data.shape

    # Channel-specific signal variance is mostly retained
    # (common was 50 amplitude, unique was 5; after CAR the unique part dominates)
    for ch in range(n_channels):
        corr = np.corrcoef(filtered[ch], unique[ch])[0, 1]
        assert corr > 0.5


def test_apply_asr_preserves_shape_and_cleans():
    """ASR returns same-shape data and reduces artifact amplitude."""
    sfreq = 250.0
    n_channels = 8
    n_samples = 5000  # 20s — enough for ASR calibration
    ch_names = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    rng = np.random.default_rng(42)

    # Clean background EEG (~10 µV)
    clean = rng.standard_normal((n_channels, n_samples)) * 10

    # Inject a large artifact burst on channel 0 (samples 2000-2250)
    artifact = clean.copy()
    artifact[0, 2000:2250] += 500  # 500 µV spike

    cleaned = apply_asr(artifact, sfreq, ch_names)

    assert cleaned.shape == artifact.shape

    # Artifact region should be reduced toward clean background levels
    artifact_ptp = np.ptp(artifact[0, 2000:2250])
    cleaned_ptp = np.ptp(cleaned[0, 2000:2250])
    assert cleaned_ptp < artifact_ptp
