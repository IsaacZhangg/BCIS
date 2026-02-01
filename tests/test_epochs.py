"""Tests for epoch extraction functionality."""

import numpy as np

from src.epochs import extract_epochs, extract_labeled_epochs


def test_extract_epochs_correct_shape():
    """Test that epochs have correct shape."""
    # Simulated preprocessed signal (2 seconds * 250 Hz * 10 = 5000 samples)
    sfreq = 250.0
    signal = np.random.randn(5000)

    # Events at sample 500 and 1500
    events = [(500, 3, 1), (1500, 5, 1)]

    epochs = extract_epochs(signal, events, sfreq, duration=2.0)

    # Should get 2 epochs, each 500 samples (2 sec * 250 Hz)
    assert len(epochs) == 2
    assert all(e.shape == (500,) for e in epochs)


def test_extract_epochs_skips_truncated():
    """Test that epochs near end of signal are skipped."""
    sfreq = 250.0
    signal = np.random.randn(1000)  # Only 4 seconds of data

    # Event at sample 900 - not enough room for 2 sec epoch
    events = [(100, 3, 1), (900, 3, 1)]

    epochs = extract_epochs(signal, events, sfreq, duration=2.0)

    # Should only get 1 epoch (second one would be truncated)
    assert len(epochs) == 1


def test_extract_labeled_epochs_separates_classes():
    """Test that labeled epochs correctly separate focused vs not-focused."""
    sfreq = 250.0
    signal = np.random.randn(10000)

    # Phase 3 = focused, Phase 5 = not focused
    events = [
        (100, 3, 1),   # focused
        (1000, 5, 1),  # not focused
        (2000, 3, 2),  # focused
        (3000, 5, 2),  # not focused
        (4000, 1, 1),  # other phase - should be ignored
    ]

    focused, not_focused = extract_labeled_epochs(signal, events, sfreq)

    assert len(focused) == 2
    assert len(not_focused) == 2
