"""Tests for epoch extraction functionality."""

import numpy as np

from src.epochs import (
    extract_left_right_epochs,
    reject_bad_epochs,
)


def test_extract_left_right_epochs_separates_by_movement():
    """Test that left/right epochs are separated by movement code in phase 3."""
    sfreq = 250.0
    signal = np.random.randn(20000)

    # Phase 3 events with movement 1 (left) and 2 (right)
    events = [
        (500, 3, 1),  # left
        (2000, 3, 2),  # right
        (4000, 3, 1),  # left
        (6000, 3, 2),  # right
        (8000, 4, 1),  # phase 4 - should be ignored
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
        signal, events, sfreq, task_duration=1.5, baseline_duration=1.0
    )

    # Baseline should be 1.0s * 250Hz = 250 samples
    # Task should be 1.5s * 250Hz = 375 samples
    baseline, task = left_pairs[0]
    assert baseline.shape == (250,)
    assert task.shape == (375,)


def test_extract_left_right_epochs_skips_baseline_before_signal_start():
    """Test that events where baseline would start before signal start are skipped."""
    sfreq = 250.0
    signal = np.random.randn(20000)

    # Event at sample 100 with baseline_duration=1.0s (250 samples)
    # Baseline would need to start at sample -150, so should be skipped
    events = [
        (100, 3, 1),  # baseline would start at -150, skipped
        (2000, 3, 1),  # valid left
    ]

    left_pairs, right_pairs = extract_left_right_epochs(
        signal, events, sfreq, baseline_duration=1.0
    )

    # Only second event should produce an epoch
    assert len(left_pairs) == 1
    assert len(right_pairs) == 0


def test_extract_left_right_epochs_skips_task_past_signal_end():
    """Test that events where task would extend past signal end are skipped."""
    sfreq = 250.0
    signal = np.random.randn(5000)  # 20 seconds of data

    # With task_duration=1.5s and skip_duration=0.5s:
    # Task needs 375 + 125 = 500 samples after event
    # Event at 4600: task_end = 4600 + 125 + 375 = 5100 > 5000, skipped
    events = [
        (2000, 3, 2),  # valid right
        (4600, 3, 2),  # task would extend past signal end, skipped
    ]

    left_pairs, right_pairs = extract_left_right_epochs(
        signal,
        events,
        sfreq,
        task_duration=1.5,
        baseline_duration=1.0,
        skip_duration=0.5,
    )

    # Only first event should produce an epoch
    assert len(left_pairs) == 0
    assert len(right_pairs) == 1


# ---------- reject_bad_epochs tests ----------


def _make_pairs(n_trials, amplitude=10.0, channels=("C3", "C4")):
    """Helper: create pairs_by_channel dicts with controlled amplitude."""
    rng = np.random.default_rng(0)
    return {
        ch: [
            (rng.standard_normal(250) * amplitude, rng.standard_normal(375) * amplitude)
            for _ in range(n_trials)
        ]
        for ch in channels
    }


def test_reject_bad_epochs_drops_high_amplitude():
    """Trials with peak-to-peak > fixed threshold_uv are rejected."""
    channels = ("C3", "C4")
    left = _make_pairs(3, amplitude=10.0, channels=channels)
    right = _make_pairs(2, amplitude=10.0, channels=channels)

    # Inject one bad left trial (index 1) with extreme amplitude
    bad_task = np.zeros(375)
    bad_task[0] = 200.0  # ptp = 200 > 150
    left["C3"][1] = (left["C3"][1][0], bad_task)

    cleaned_l, cleaned_r, rej_l, rej_r = reject_bad_epochs(
        left, right, threshold_uv=150.0
    )

    assert rej_l == 1
    assert rej_r == 0
    assert len(cleaned_l["C3"]) == 2
    assert len(cleaned_r["C3"]) == 2


def test_reject_bad_epochs_drops_flat_trials():
    """Flat trials (ptp < flat_uv) are rejected."""
    channels = ("C3", "C4")
    left = _make_pairs(2, amplitude=10.0, channels=channels)
    right = _make_pairs(2, amplitude=10.0, channels=channels)

    # Make one right trial flat on one channel
    flat_task = np.ones(375) * 5.0  # ptp = 0 < 1
    right["C4"][0] = (right["C4"][0][0], flat_task)

    cleaned_l, cleaned_r, rej_l, rej_r = reject_bad_epochs(
        left, right, threshold_uv=150.0
    )

    assert rej_l == 0
    assert rej_r == 1


def test_reject_bad_epochs_keeps_clean_data():
    """All-clean data passes through unchanged (adaptive threshold)."""
    channels = ("C3", "C4")
    left = _make_pairs(5, amplitude=10.0, channels=channels)
    right = _make_pairs(5, amplitude=10.0, channels=channels)

    cleaned_l, cleaned_r, rej_l, rej_r = reject_bad_epochs(left, right)

    assert rej_l == 0
    assert rej_r == 0
    assert len(cleaned_l["C3"]) == 5
    assert len(cleaned_r["C3"]) == 5


def test_reject_bad_epochs_adaptive_drops_outlier():
    """Adaptive threshold (median + n_mad * MAD) drops only extreme outliers."""
    channels = ("C3", "C4")
    # 20 uniform trials at amplitude=10 → typical ptp ~60-70
    left = _make_pairs(10, amplitude=10.0, channels=channels)
    right = _make_pairs(10, amplitude=10.0, channels=channels)

    # Inject one massive outlier into left trial 0
    bad_task = np.zeros(375)
    bad_task[0] = 1000.0  # ptp = 1000, far above any adaptive threshold
    left["C3"][0] = (left["C3"][0][0], bad_task)

    cleaned_l, cleaned_r, rej_l, rej_r = reject_bad_epochs(left, right)

    assert rej_l == 1  # only the outlier rejected
    assert rej_r == 0
    assert len(cleaned_l["C3"]) == 9


def test_reject_tighter_n_mad():
    """An outlier that passes n_mad=4 but fails n_mad=3 is correctly rejected."""
    channels = ("C3", "C4")
    # 20 uniform trials at amplitude=10
    left = _make_pairs(10, amplitude=10.0, channels=channels)
    right = _make_pairs(10, amplitude=10.0, channels=channels)

    all_ptps = [
        float(np.ptp(pairs[i][1]))
        for ch in channels
        for pairs in [left[ch], right[ch]]
        for i in range(len(pairs))
    ]
    ptps_arr = np.array(all_ptps)
    median = float(np.median(ptps_arr))
    mad = float(np.median(np.abs(ptps_arr - median)))

    # Craft a trial that exceeds median + 3*MAD but not median + 4*MAD
    target_ptp = median + 3.5 * mad
    bad_task = np.zeros(375)
    bad_task[0] = target_ptp
    left["C3"][0] = (left["C3"][0][0], bad_task)

    # With n_mad=4 (old default), should pass
    _, _, rej_l_loose, _ = reject_bad_epochs(left, right, n_mad=4.0)

    # With n_mad=3 (new default), should be rejected
    _, _, rej_l_tight, _ = reject_bad_epochs(left, right, n_mad=3.0)

    assert rej_l_tight > rej_l_loose
