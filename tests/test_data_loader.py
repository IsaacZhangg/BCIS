"""Tests for data loading functionality."""

from pathlib import Path

from src.data_loader import load_recording, get_complete_recordings


def test_load_recording_returns_data_and_events():
    """Test that load_recording returns EEG data and event markers."""
    data_dir = Path("Data/unicorn-data")
    csv_path = data_dir / "subject0001/session000/recording_2025-11-12-21.33.31.csv"

    data, events, sfreq = load_recording(csv_path)

    # Check data shape: (n_channels, n_samples)
    assert data.ndim == 2
    assert data.shape[0] == 8  # 8 EEG channels
    assert data.shape[1] > 0

    # Check events: list of (sample_idx, phase, movement)
    assert len(events) > 0
    assert all(len(e) == 3 for e in events)

    # Check sampling frequency
    assert sfreq == 250.0


def test_get_complete_recordings_finds_all_subjects():
    """Test that we find all complete recordings (100 imagery trials)."""
    data_dir = Path("Data/unicorn-data")

    recordings = get_complete_recordings(data_dir)

    # Should find 10 complete recordings
    assert len(recordings) == 10
    assert all(Path(r).exists() for r in recordings)


def test_get_recordings_flexible_finds_mi_data_new():
    """Flexible loader finds MI_DATA_NEW recordings with >= 20 phase-3 trials."""
    from src.data_loader import get_recordings_flexible

    data_dir = Path("Data/MI_DATA_NEW")
    grouped = get_recordings_flexible(data_dir, min_trials=20)

    # subject0104 has 2 sessions with 32 trials each
    assert "subject0104" in grouped
    assert len(grouped["subject0104"]) == 2

    # subject0106 has 1 session with 32 trials
    assert "subject0106" in grouped
    assert len(grouped["subject0106"]) == 1

    # subject0105 has 0 trials — should be excluded
    assert "subject0105" not in grouped


def test_get_recordings_flexible_respects_min_trials():
    """Recordings below min_trials are excluded."""
    from src.data_loader import get_recordings_flexible

    data_dir = Path("Data/MI_DATA_NEW")
    grouped = get_recordings_flexible(data_dir, min_trials=50)
    assert "subject0106" not in grouped
    assert "subject0104" not in grouped
