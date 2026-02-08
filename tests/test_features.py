"""Tests for feature extraction functionality."""

import numpy as np
import pytest

from src.features import (
    compute_c3c4_coherence,
    compute_lateralization_index,
    compute_peak_frequency,
    compute_spectral_entropy,
    compute_statistical_features,
    compute_theta_power,
    extract_csp_features,
    extract_features,
    extract_lateralization_features,
)


def test_compute_theta_power_detects_theta():
    """Test that theta power is higher for theta-dominant signal."""
    sfreq = 250.0
    t = np.arange(0, 2, 1 / sfreq)  # 2 seconds

    # Signal with strong 6 Hz (theta) component
    theta_signal = np.sin(2 * np.pi * 6 * t)

    # Signal with strong 20 Hz (beta) component
    beta_signal = np.sin(2 * np.pi * 20 * t)

    theta_power_high = compute_theta_power(theta_signal, sfreq)
    theta_power_low = compute_theta_power(beta_signal, sfreq)

    # Theta signal should have higher theta power
    assert theta_power_high > theta_power_low * 10


def test_compute_theta_power_returns_scalar():
    """Test that theta power returns a single scalar value."""
    sfreq = 250.0
    signal = np.random.randn(500)  # 2 seconds

    power = compute_theta_power(signal, sfreq)

    assert isinstance(power, float)
    assert power > 0  # Power is always positive


def test_extract_features_correct_shape():
    """Test that extract_features returns correct array shape."""
    sfreq = 250.0
    epochs = [np.random.randn(500) for _ in range(10)]

    features = extract_features(epochs, sfreq)

    # Should be (n_epochs, 1) - one feature per epoch
    assert features.shape == (10, 1)


def test_extract_features_log_transform():
    """Test that log transform is applied (values should be reasonable range)."""
    sfreq = 250.0
    epochs = [np.random.randn(500) * 10 for _ in range(10)]

    features = extract_features(epochs, sfreq, log_transform=True)

    # Log-transformed values should be in reasonable range (not huge like raw power)
    assert np.all(features < 100)
    assert np.all(features > -100)


def test_compute_lateralization_index_detects_asymmetry():
    """Test that lateralization index detects left/right power difference."""
    sfreq = 250.0
    t = np.arange(0, 1.5, 1 / sfreq)

    # C3 has strong 10Hz (mu), C4 has weak signal
    c3_signal = np.sin(2 * np.pi * 10 * t) * 2
    c4_signal = np.sin(2 * np.pi * 10 * t) * 0.5

    # Lateralization index: (C4 - C3) / (C4 + C3)
    # Should be negative when C3 > C4 (right hand imagery pattern)
    lat_idx = compute_lateralization_index(c3_signal, c4_signal, sfreq, 8, 12)
    assert lat_idx < 0


def test_extract_lateralization_features_correct_shape():
    """Test that lateralization features have correct shape."""
    sfreq = 250.0
    n_epochs = 5
    n_samples_baseline = 250
    n_samples_task = 375

    epoch_pairs_by_channel = {
        ch: [
            (np.random.randn(n_samples_baseline), np.random.randn(n_samples_task))
            for _ in range(n_epochs)
        ]
        for ch in ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    }

    features = extract_lateralization_features(epoch_pairs_by_channel, sfreq)

    # Should be (n_epochs, n_features)
    assert features.shape[0] == n_epochs
    assert features.shape[1] > 0


def test_compute_lateralization_index_equal_power():
    """Test that lateralization index is ~0 when C3 and C4 have equal power."""
    sfreq = 250.0
    t = np.arange(0, 1.5, 1 / sfreq)

    # C3 and C4 have identical signals (equal power)
    c3_signal = np.sin(2 * np.pi * 10 * t)
    c4_signal = np.sin(2 * np.pi * 10 * t)

    lat_idx = compute_lateralization_index(c3_signal, c4_signal, sfreq, 8, 12)

    # Lateralization index should be approximately 0 when power is equal
    assert abs(lat_idx) < 0.01


def test_extract_lateralization_features_expected_count():
    """Test that lateralization features returns the expected number of features."""
    sfreq = 250.0
    n_epochs = 3
    n_samples_baseline = 250
    n_samples_task = 375

    epoch_pairs_by_channel = {
        ch: [
            (np.random.randn(n_samples_baseline), np.random.randn(n_samples_task))
            for _ in range(n_epochs)
        ]
        for ch in ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    }

    features = extract_lateralization_features(epoch_pairs_by_channel, sfreq)

    # Expected feature count breakdown:
    # - 4 bands x 7 features per band = 28 (lateralization features for C3/C4)
    # - 2 bands x 2 features per band = 4 (Cz features: mu and beta)
    # - 1 feature (Fz theta ratio)
    # - 2 channels x 3 Hjorth params = 6 (C3 and C4 time-domain features)
    # Total = 28 + 4 + 1 + 6 = 39
    expected_feature_count = 39
    assert features.shape == (n_epochs, expected_feature_count)


def test_extract_lateralization_features_missing_channels():
    """Test that missing required channels raises a clear error."""
    sfreq = 250.0
    n_epochs = 2

    # Missing C4, Fz, PO7, PO8 channels
    epoch_pairs_by_channel = {
        ch: [(np.random.randn(250), np.random.randn(375)) for _ in range(n_epochs)]
        for ch in ["C3", "Cz"]
    }

    with pytest.raises(ValueError, match="Missing required channels"):
        extract_lateralization_features(epoch_pairs_by_channel, sfreq)


def test_extract_csp_features_correct_shape():
    """Test CSP feature extraction returns correct shape."""
    sfreq = 250.0
    n_epochs = 20
    n_channels = 8
    n_samples = 375

    # Multichannel data: (n_epochs, n_channels, n_samples)
    X = np.random.randn(n_epochs, n_channels, n_samples)
    y = np.array([0] * 10 + [1] * 10)  # Binary labels

    features, csp_model = extract_csp_features(X, y, sfreq, n_components=4)

    # Should return (n_epochs, n_components) features
    assert features.shape == (n_epochs, 4)
    assert csp_model is not None


# ---------- New feature function tests ----------


def test_spectral_entropy_normalized():
    """Spectral entropy is in [0, 1] and a sinusoid has lower entropy than noise."""
    sfreq = 250.0
    t = np.arange(0, 1.5, 1 / sfreq)

    sinusoid = np.sin(2 * np.pi * 10 * t)
    noise = np.random.default_rng(42).standard_normal(len(t))

    se_sin = compute_spectral_entropy(sinusoid, sfreq, 8, 12)
    se_noise = compute_spectral_entropy(noise, sfreq, 8, 12)

    assert 0 <= se_sin <= 1
    assert 0 <= se_noise <= 1
    assert se_sin < se_noise


def test_peak_frequency_known_peak():
    """Peak frequency of a 10 Hz sinusoid should be near 10 Hz."""
    sfreq = 250.0
    t = np.arange(0, 1.5, 1 / sfreq)
    signal = np.sin(2 * np.pi * 10 * t)

    pf = compute_peak_frequency(signal, sfreq, 8, 12)

    assert abs(pf - 10.0) < 1.0


def test_c3c4_coherence_range():
    """Identical signals should have coherence near 1.0."""
    sfreq = 250.0
    t = np.arange(0, 1.5, 1 / sfreq)
    signal = np.sin(2 * np.pi * 10 * t)

    coh = compute_c3c4_coherence(signal, signal, sfreq, 8, 12)

    assert 0.9 <= coh <= 1.0


def test_statistical_features_gaussian():
    """Gaussian noise should have skewness ~0 and kurtosis ~0 (Fisher)."""
    rng = np.random.default_rng(42)
    signal = rng.standard_normal(10000)

    sk, ku, zcr = compute_statistical_features(signal)

    assert abs(sk) < 0.1
    assert abs(ku) < 0.2
    assert 0 < zcr < 1
