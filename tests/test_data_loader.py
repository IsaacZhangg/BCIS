"""Tests for data loading functionality."""

from pathlib import Path

from src.data_loader import (
    get_complete_recordings,
    get_recordings,
    get_recordings_by_subject,
    load_recording,
)


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


def test_get_recordings_default_finds_complete():
    """Test that get_recordings with default min_trials=100 finds complete recordings."""
    data_dir = Path("Data/unicorn-data")

    recordings = get_recordings(data_dir)

    # Default min_trials=100 should find exactly the same as the old function
    assert len(recordings) == 10
    assert all(Path(r).exists() for r in recordings)


def test_get_recordings_low_threshold_finds_more():
    """Test that lowering min_trials finds recordings that have fewer trials."""
    data_dir = Path("Data/unicorn-data")

    recordings_strict = get_recordings(data_dir, min_trials=100)
    recordings_loose = get_recordings(data_dir, min_trials=30)

    # Loose threshold should find at least as many as strict
    assert len(recordings_loose) >= len(recordings_strict)
    # All strict recordings should also appear in loose results
    assert set(recordings_strict).issubset(set(recordings_loose))


def test_get_recordings_by_subject_groups():
    """Test that get_recordings_by_subject groups recordings correctly."""
    data_dir = Path("Data/unicorn-data")

    grouped = get_recordings_by_subject(data_dir)

    # Should have subjects
    assert len(grouped) > 0
    # Each key should be a subject directory name
    for sid, paths in grouped.items():
        assert sid.startswith("subject")
        assert len(paths) >= 1
        for p in paths:
            assert p.parent.parent.name == sid


def test_backward_compat_alias():
    """Test that get_complete_recordings still works as backward compat alias."""
    data_dir = Path("Data/unicorn-data")

    recordings = get_complete_recordings(data_dir)

    assert len(recordings) == 10
    assert all(Path(r).exists() for r in recordings)
