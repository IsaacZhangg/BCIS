"""Tests for epoch extraction functionality."""

import numpy as np

from src.epochs import extract_epochs, extract_labeled_epochs, extract_left_right_epochs


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

    focused, not_focused = extract_labeled_epochs(signal, events, sfreq, baseline_phase=5)

    assert len(focused) == 2
    assert len(not_focused) == 2


def test_extract_left_right_epochs_separates_by_movement():
    """Test that left/right epochs are separated by movement code in phase 3."""
    sfreq = 250.0
    signal = np.random.randn(20000)

    # Phase 3 events with movement 1 (left) and 2 (right)
    events = [
        (500, 3, 1),    # left
        (2000, 3, 2),   # right
        (4000, 3, 1),   # left
        (6000, 3, 2),   # right
        (8000, 4, 1),   # phase 4 - should be ignored
        (10000, 3, 1),  # left
    ]

    left_pairs, right_pairs = extract_left_right_epochs(signal, events, sfreq)

    assert len(left_pairs) == 3
    assert len(right_pairs) == 2
    # Each pair should be (baseline, task) tuple
    assert len(left_pairs[0]) == 2
    assert len(right_pairs[0]) == 2


def test_extract_left_right_epochs_correct_durations():
    """Test that baseline and task epochs have correct durations."""
    sfreq = 250.0
    signal = np.random.randn(20000)

    events = [(2000, 3, 1), (5000, 3, 2)]

    left_pairs, right_pairs = extract_left_right_epochs(
        signal, events, sfreq,
        task_duration=1.5,
        baseline_duration=1.0
    )

    # Baseline should be 1.0s * 250Hz = 250 samples
    # Task should be 1.5s * 250Hz = 375 samples
    baseline, task = left_pairs[0]
    assert baseline.shape == (250,)
    assert task.shape == (375,)
