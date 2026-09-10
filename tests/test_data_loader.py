"""Test data loading with synthetic recordings, without participant data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    discover_recordings,
    get_complete_recordings,
    get_recordings,
    get_recordings_by_subject,
    get_recordings_flexible,
    load_recording,
)


def _write_synthetic_recording(path: Path, seed: int = 0, n_trials: int = 40) -> None:
    """Write compact CSVs for loader tests, not signal-processing benchmarks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = 4 * (n_trials + 1)
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_samples, 8))
    stim = np.zeros(n_samples, dtype=int)
    for i in range(n_trials):
        event_idx = 4 * (i + 1)
        movement = 1 if i % 2 == 0 else 2
        stim[event_idx - 1] = 20 + movement
        stim[event_idx] = 30 + movement
    df = pd.DataFrame(data, columns=["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"])
    df.insert(0, "timestamps", np.arange(n_samples) / 250.0)
    df["stim"] = stim
    df.to_csv(path, index=False)


@pytest.fixture
def recordings_dir(tmp_path: Path) -> Path:
    """Include complete, short, and empty recordings across several subjects."""
    recordings = [
        ("subject_alpha/session000/recording_complete.csv", 100),
        ("subject_alpha/session001/recording_short.csv", 32),
        ("subject_beta/session000/recording_complete.csv", 100),
        ("subject_short/session000/recording_short.csv", 19),
        ("subject_empty/session000/recording_empty.csv", 0),
    ]
    for seed, (relative_path, n_trials) in enumerate(recordings):
        _write_synthetic_recording(tmp_path / relative_path, seed, n_trials)
    return tmp_path


def test_load_recording_returns_data_and_events(tmp_path: Path) -> None:
    csv_path = tmp_path / "recording_synthetic.csv"
    _write_synthetic_recording(csv_path, n_trials=2)

    data, events, sfreq = load_recording(csv_path)

    assert data.shape == (8, 12)
    assert np.isfinite(data).all()
    assert events == [(3, 2, 1), (4, 3, 1), (7, 2, 2), (8, 3, 2)]
    assert sfreq == 250.0


def test_get_recordings_default_finds_complete(recordings_dir: Path) -> None:
    recordings = get_recordings(recordings_dir)

    assert set(recordings) == {
        recordings_dir / "subject_alpha/session000/recording_complete.csv",
        recordings_dir / "subject_beta/session000/recording_complete.csv",
    }


def test_get_recordings_low_threshold_finds_more(recordings_dir: Path) -> None:
    recordings_strict = get_recordings(recordings_dir, min_trials=100)
    recordings_loose = get_recordings(recordings_dir, min_trials=30)

    assert set(recordings_strict) < set(recordings_loose)
    assert set(recordings_loose) - set(recordings_strict) == {
        recordings_dir / "subject_alpha/session001/recording_short.csv"
    }


def test_get_recordings_by_subject_groups(recordings_dir: Path) -> None:
    grouped = get_recordings_by_subject(recordings_dir, min_trials=30)

    assert grouped == {
        "subject_alpha": [
            recordings_dir / "subject_alpha/session000/recording_complete.csv",
            recordings_dir / "subject_alpha/session001/recording_short.csv",
        ],
        "subject_beta": [
            recordings_dir / "subject_beta/session000/recording_complete.csv"
        ],
    }


def test_backward_compat_alias(recordings_dir: Path) -> None:
    assert get_complete_recordings(recordings_dir) == get_recordings(recordings_dir)
    assert len(get_complete_recordings(recordings_dir)) == 2


def test_get_recordings_flexible_finds_short_sessions(recordings_dir: Path) -> None:
    grouped = get_recordings_flexible(recordings_dir, min_trials=20)

    assert set(grouped) == {"subject_alpha", "subject_beta"}
    assert len(grouped["subject_alpha"]) == 2
    assert len(grouped["subject_beta"]) == 1


def test_get_recordings_flexible_respects_min_trials(recordings_dir: Path) -> None:
    grouped = get_recordings_flexible(recordings_dir, min_trials=50)

    assert set(grouped) == {"subject_alpha", "subject_beta"}
    assert all(len(paths) == 1 for paths in grouped.values())
    assert all(
        path.name == "recording_complete.csv"
        for paths in grouped.values()
        for path in paths
    )


def test_discover_recordings_dedups_identical_copies(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    rec_a = dir_a / "subject_demo/session000/recording.csv"
    rec_b = dir_b / "subject_demo/session000/recording.csv"
    _write_synthetic_recording(rec_a, seed=1)
    _write_synthetic_recording(rec_b, seed=1)

    grouped = discover_recordings([dir_a, dir_b], min_trials=20)

    assert grouped == {"subject_demo": [rec_a]}


def test_discover_recordings_merges_subject_aliases(tmp_path: Path) -> None:
    rec_a = tmp_path / "subject_demo/session001/recording.csv"
    rec_b = tmp_path / "subject_alias/session001/recording.csv"
    _write_synthetic_recording(rec_a, seed=1)
    _write_synthetic_recording(rec_b, seed=2)

    grouped = discover_recordings(
        [tmp_path], min_trials=20, subject_merge={"subject_alias": "subject_demo"}
    )

    assert set(grouped) == {"subject_demo"}
    assert set(grouped["subject_demo"]) == {rec_a, rec_b}


def test_discover_recordings_combines_roots_with_filtering(tmp_path: Path) -> None:
    """Filter empty recordings, deduplicate copies, and retain distinct sessions."""
    dir_a = tmp_path / "original"
    dir_b = tmp_path / "additional"
    rec_a = dir_a / "subject_demo/session001/recording.csv"
    duplicate = dir_b / "subject_demo/session001/recording.csv"
    rec_b = dir_b / "subject_alias/session002/recording.csv"
    empty = dir_b / "subject_empty/session001/recording.csv"
    _write_synthetic_recording(rec_a, seed=1)
    _write_synthetic_recording(duplicate, seed=1)
    _write_synthetic_recording(rec_b, seed=2)
    _write_synthetic_recording(empty, n_trials=0)

    grouped = discover_recordings(
        [dir_a, dir_b],
        min_trials=20,
        subject_merge={"subject_alias": "subject_demo"},
    )

    assert grouped == {"subject_demo": [rec_a, rec_b]}
