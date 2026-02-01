"""Tests for feature extraction functionality."""

import numpy as np

from src.features import compute_theta_power, extract_features


def test_compute_theta_power_detects_theta():
    """Test that theta power is higher for theta-dominant signal."""
    sfreq = 250.0
    t = np.arange(0, 2, 1/sfreq)  # 2 seconds

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
