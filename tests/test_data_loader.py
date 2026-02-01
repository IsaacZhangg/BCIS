"""Tests for data loading functionality."""

from pathlib import Path

from src.data_loader import load_recording, get_complete_recordings


def test_load_recording_returns_data_and_events():
    """Test that load_recording returns EEG data and event markers."""
    data_dir = Path("unicorn-data")
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
    data_dir = Path("unicorn-data")

    recordings = get_complete_recordings(data_dir)

    # Should find 10 complete recordings
    assert len(recordings) == 10
    assert all(Path(r).exists() for r in recordings)
