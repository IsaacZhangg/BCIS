"""Tests for real-time simulation."""

import numpy as np
import pytest
from pathlib import Path


def test_compute_engagement_score_range():
    """Test that engagement score is in 0-100 range."""
    from src.simulate_realtime import compute_engagement_score

    # Mock probability
    score = compute_engagement_score(0.75)
    assert 0 <= score <= 100

    score_low = compute_engagement_score(0.0)
    assert score_low == 0

    score_high = compute_engagement_score(1.0)
    assert score_high == 100


def test_check_alert_consecutive_windows():
    """Test alert triggers after consecutive low scores."""
    from src.simulate_realtime import check_alert

    # Not enough consecutive low scores
    history = [80, 75, 25, 70]  # Only one below threshold
    assert check_alert(history, threshold=30, consecutive=3) == False

    # Enough consecutive low scores
    history = [80, 25, 20, 15]  # Three consecutive below threshold
    assert check_alert(history, threshold=30, consecutive=3) == True


def test_sliding_window_extraction():
    """Test that sliding windows are correctly extracted."""
    from src.simulate_realtime import extract_sliding_windows

    # 20 seconds of fake data at 250 Hz = 5000 samples
    n_samples = 5000
    data = {ch: np.random.randn(n_samples) for ch in
            ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]}

    windows = extract_sliding_windows(data, sfreq=250.0, window_sec=5.0, step_sec=5.0)

    # 20 seconds / 5 second step = 4 windows
    assert len(windows) == 4

    # Each window should have 1250 samples per channel
    for window in windows:
        for ch_name, signal in window.items():
            assert len(signal) == 1250
