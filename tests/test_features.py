"""Tests for feature extraction functionality."""

import numpy as np
import pytest

from src.features import (
    compute_c3c4_coherence,
    compute_spectral_entropy,
    extract_lateralization_features,
)


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
    # - 4 bands x 7 features per band = 28 (Laplacian C3/C4 lateralization)
    # - 2 bands x 2 features per band = 4 (Cz features: mu and beta)
    # - 1 feature (Fz theta ratio)
    # - 2 channels x 3 Hjorth params = 6 (Laplacian C3/C4 time-domain)
    # - 2 bands x 2 features per band = 4 (C3-C4 coherence: task + delta)
    # - 2 bands x 1 feature per band = 2 (spectral entropy lateralization)
    # Total = 28 + 4 + 1 + 6 + 4 + 2 = 45
    expected_feature_count = 45
    assert features.shape == (n_epochs, expected_feature_count)


def test_extract_lateralization_features_missing_channels():
    """Test that missing required channels raises a clear error."""
    sfreq = 250.0
    n_epochs = 2

    # Missing C4, Fz, PO7, PO8, Pz channels
    epoch_pairs_by_channel = {
        ch: [(np.random.randn(250), np.random.randn(375)) for _ in range(n_epochs)]
        for ch in ["C3", "Cz"]
    }

    with pytest.raises(ValueError, match="Missing required channels"):
        extract_lateralization_features(epoch_pairs_by_channel, sfreq)


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


def test_c3c4_coherence_range():
    """Identical signals should have coherence near 1.0."""
    sfreq = 250.0
    t = np.arange(0, 1.5, 1 / sfreq)
    signal = np.sin(2 * np.pi * 10 * t)

    coh = compute_c3c4_coherence(signal, signal, sfreq, 8, 12)

    assert 0.9 <= coh <= 1.0
