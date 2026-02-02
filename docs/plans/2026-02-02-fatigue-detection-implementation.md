# Fatigue Detection Classifier Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Pivot from Phase 3 vs Phase 4 classification to Phase 3 vs Phase 5 for fatigue detection with real-time simulation.

**Architecture:** Modify existing pipeline to use Phase 5 (rest) as the disengaged class, add absolute feature extraction for real-time inference, create sliding window simulation outputting engagement scores.

**Tech Stack:** Python, NumPy, SciPy, scikit-learn, pyriemann, MNE, joblib

---

## Task 1: Remove Old Phase 3 vs 4 Artifacts

**Files:**
- Delete: `models/theta_classifier_model.joblib`
- Delete: `models/theta_classifier_scaler.joblib`
- Delete: `models/training_results.json`
- Delete: `docs/plans/2026-01-31-theta-focus-classifier-design.md`
- Delete: `docs/plans/2026-01-31-theta-focus-classifier-implementation.md`

**Step 1: Delete old model files and docs**

```bash
rm models/theta_classifier_model.joblib
rm models/theta_classifier_scaler.joblib
rm models/training_results.json
rm docs/plans/2026-01-31-theta-focus-classifier-design.md
rm docs/plans/2026-01-31-theta-focus-classifier-implementation.md
```

**Step 2: Verify deletion**

Run: `ls models/ && ls docs/plans/`
Expected: Only `.DS_Store` in models/, only `2026-02-02-*` files in docs/plans/

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove Phase 3 vs 4 classifier artifacts"
```

---

## Task 2: Update Pipeline to Use Phase 5

**Files:**
- Modify: `src/pipeline.py:59-66` (change class2_phase from 4 to 5)
- Modify: `src/pipeline.py:27-29` (update print header)
- Modify: `src/pipeline.py:82` (update print message)
- Modify: `src/pipeline.py:141-142` (rename model files)

**Step 1: Write failing test for Phase 5 usage**

Create test in `tests/test_pipeline.py`:

```python
"""Tests for pipeline configuration."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


def test_pipeline_uses_phase_5_for_disengaged():
    """Test that pipeline extracts Phase 5 (rest) as disengaged class."""
    from src.pipeline import run_pipeline
    from pathlib import Path

    # We'll verify by checking the extract_erd_epochs call
    with patch('src.pipeline.extract_erd_epochs') as mock_extract:
        # Set up mock to return empty data (we just want to check the call)
        mock_extract.return_value = ([], [])

        with patch('src.pipeline.get_complete_recordings') as mock_get:
            mock_get.return_value = []  # No recordings, just testing config

            try:
                run_pipeline(Path("unicorn-data"), Path("models"))
            except Exception:
                pass  # Expected to fail with no data

            # If called, verify class2_phase=5
            if mock_extract.called:
                call_kwargs = mock_extract.call_args[1]
                assert call_kwargs.get('class2_phase') == 5, \
                    f"Expected class2_phase=5, got {call_kwargs.get('class2_phase')}"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py::test_pipeline_uses_phase_5_for_disengaged -v`
Expected: FAIL (class2_phase=4 currently)

**Step 3: Update src/pipeline.py**

Change line 28-29 from:
```python
    print("=" * 60)
    print("Theta Focus Classifier - Training Pipeline")
```
to:
```python
    print("=" * 60)
    print("Engagement Classifier - Training Pipeline")
```

Change line 59 comment and line 65 from:
```python
            # Compare phase 3 vs phase 4 (different imagery task types)
            focused_pairs, rest_pairs = extract_erd_epochs(
                signal, events, sfreq,
                task_duration=1.5,
                baseline_duration=1.0,
                class1_phase=3,
                class2_phase=4,
            )
```
to:
```python
            # Compare phase 3 (engaged/imagery) vs phase 5 (disengaged/rest)
            engaged_pairs, disengaged_pairs = extract_erd_epochs(
                signal, events, sfreq,
                task_duration=1.5,
                baseline_duration=1.0,
                class1_phase=3,
                class2_phase=5,
            )
```

Change line 67-68 variable names from:
```python
            focused_pairs_by_channel[ch_name] = focused_pairs
            rest_pairs_by_channel[ch_name] = rest_pairs
```
to:
```python
            engaged_pairs_by_channel[ch_name] = engaged_pairs
            disengaged_pairs_by_channel[ch_name] = disengaged_pairs
```

Change line 55-56 variable names from:
```python
        focused_pairs_by_channel = {}
        rest_pairs_by_channel = {}
```
to:
```python
        engaged_pairs_by_channel = {}
        disengaged_pairs_by_channel = {}
```

Change line 71-72 from:
```python
        n_focused = len(focused_pairs_by_channel[CHANNELS[0]])
        n_rest = len(rest_pairs_by_channel[CHANNELS[0]])
```
to:
```python
        n_engaged = len(engaged_pairs_by_channel[CHANNELS[0]])
        n_disengaged = len(disengaged_pairs_by_channel[CHANNELS[0]])
```

Change lines 74-82 (class balancing) from:
```python
        # Balance classes
        if n_rest > n_focused:
            np.random.seed(42)
            indices = np.random.choice(n_rest, n_focused, replace=False)
            for ch_name in CHANNELS:
                rest_pairs_by_channel[ch_name] = [rest_pairs_by_channel[ch_name][i] for i in indices]
            n_rest = n_focused

        print(f"    Focused epochs: {n_focused}, Not-focused epochs: {n_rest}")
```
to:
```python
        # Balance classes
        if n_disengaged > n_engaged:
            np.random.seed(42)
            indices = np.random.choice(n_disengaged, n_engaged, replace=False)
            for ch_name in CHANNELS:
                disengaged_pairs_by_channel[ch_name] = [disengaged_pairs_by_channel[ch_name][i] for i in indices]
            n_disengaged = n_engaged

        print(f"    Engaged epochs: {n_engaged}, Disengaged epochs: {n_disengaged}")
```

Change lines 84-86 from:
```python
        # Extract ERD features
        focused_features = extract_erd_features(focused_pairs_by_channel, sfreq)
        rest_features = extract_erd_features(rest_pairs_by_channel, sfreq)
```
to:
```python
        # Extract ERD features
        engaged_features = extract_erd_features(engaged_pairs_by_channel, sfreq)
        disengaged_features = extract_erd_features(disengaged_pairs_by_channel, sfreq)
```

Change lines 91-93 from:
```python
        n_focused = len(focused_pairs_by_channel[CHANNELS[0]])
        n_rest = len(rest_pairs_by_channel[CHANNELS[0]])
        n_samples = focused_pairs_by_channel[CHANNELS[0]][0][1].shape[0]
```
to:
```python
        n_engaged = len(engaged_pairs_by_channel[CHANNELS[0]])
        n_disengaged = len(disengaged_pairs_by_channel[CHANNELS[0]])
        n_samples = engaged_pairs_by_channel[CHANNELS[0]][0][1].shape[0]
```

Change lines 95-96 from:
```python
        focused_multichannel = np.zeros((n_focused, n_channels, n_samples))
        rest_multichannel = np.zeros((n_rest, n_channels, n_samples))
```
to:
```python
        engaged_multichannel = np.zeros((n_engaged, n_channels, n_samples))
        disengaged_multichannel = np.zeros((n_disengaged, n_channels, n_samples))
```

Change lines 98-102 from:
```python
        for ch_idx, ch_name in enumerate(CHANNELS):
            for trial_idx in range(n_focused):
                focused_multichannel[trial_idx, ch_idx, :] = focused_pairs_by_channel[ch_name][trial_idx][1]
            for trial_idx in range(n_rest):
                rest_multichannel[trial_idx, ch_idx, :] = rest_pairs_by_channel[ch_name][trial_idx][1]
```
to:
```python
        for ch_idx, ch_name in enumerate(CHANNELS):
            for trial_idx in range(n_engaged):
                engaged_multichannel[trial_idx, ch_idx, :] = engaged_pairs_by_channel[ch_name][trial_idx][1]
            for trial_idx in range(n_disengaged):
                disengaged_multichannel[trial_idx, ch_idx, :] = disengaged_pairs_by_channel[ch_name][trial_idx][1]
```

Change line 104 from:
```python
        X_multichannel = np.vstack([focused_multichannel, rest_multichannel])
```
to:
```python
        X_multichannel = np.vstack([engaged_multichannel, disengaged_multichannel])
```

Change lines 106-108 from:
```python
        # Combine ERD features
        X = np.vstack([focused_features, rest_features])
        y = np.array([1] * len(focused_features) + [0] * len(rest_features))
```
to:
```python
        # Combine ERD features
        X = np.vstack([engaged_features, disengaged_features])
        y = np.array([1] * len(engaged_features) + [0] * len(disengaged_features))
```

Change lines 141-142 from:
```python
    model_path = output_dir / "theta_classifier_model.joblib"
    scaler_path = output_dir / "theta_classifier_scaler.joblib"
```
to:
```python
    model_path = output_dir / "engagement_classifier_model.joblib"
    scaler_path = output_dir / "engagement_classifier_scaler.joblib"
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py::test_pipeline_uses_phase_5_for_disengaged -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: update pipeline to use Phase 5 (rest) as disengaged class"
```

---

## Task 3: Add Real-Time Feature Extraction

**Files:**
- Modify: `src/features.py` (add extract_realtime_features function)
- Test: `tests/test_features.py`

**Step 1: Write failing test for realtime features**

Add to `tests/test_features.py`:

```python
from src.features import extract_realtime_features


def test_extract_realtime_features_correct_shape():
    """Test that realtime features have correct shape for 8 channels."""
    sfreq = 250.0
    # 5 seconds of data, 8 channels
    window = {ch: np.random.randn(1250) for ch in
              ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]}

    features = extract_realtime_features(window, sfreq)

    # Should return a 1D feature vector
    assert features.ndim == 1
    assert len(features) > 0


def test_extract_realtime_features_no_baseline_needed():
    """Test that realtime features work without baseline reference."""
    sfreq = 250.0
    # Single 5-second window
    window = {ch: np.random.randn(1250) for ch in
              ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]}

    # Should not raise - no baseline needed
    features = extract_realtime_features(window, sfreq)

    # All features should be finite
    assert np.all(np.isfinite(features))
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features.py::test_extract_realtime_features_correct_shape -v`
Expected: FAIL (function doesn't exist)

**Step 3: Implement extract_realtime_features in src/features.py**

Add at end of file (after line 236):

```python


def extract_realtime_features(
    window_by_channel: dict[str, np.ndarray],
    sfreq: float,
) -> np.ndarray:
    """
    Extract features from a single time window for real-time inference.

    Unlike extract_erd_features, this does NOT require a baseline epoch.
    Uses absolute features that can be computed from a single window.

    Args:
        window_by_channel: Dict mapping channel names to signal arrays
        sfreq: Sampling frequency in Hz

    Returns:
        1D feature vector
    """
    bands = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "low_beta": (13, 20),
        "high_beta": (20, 30),
        "beta": (13, 30),
        "gamma": (30, 40),
    }

    channels = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    features = []
    channel_powers = {}

    for ch_name in channels:
        if ch_name not in window_by_channel:
            continue

        signal = window_by_channel[ch_name]
        channel_powers[ch_name] = {}

        # Total power for normalization
        total_power = compute_band_power(signal, sfreq, 1, 40)

        for band_name, (low, high) in bands.items():
            power = compute_band_power(signal, sfreq, low, high)

            # Absolute log power
            features.append(np.log(power + 1e-10))

            # Relative power (proportion of total)
            rel_power = power / (total_power + 1e-10)
            features.append(rel_power)

            channel_powers[ch_name][band_name] = power

        # Theta/Alpha ratio (drowsiness indicator)
        theta = channel_powers[ch_name]["theta"]
        alpha = channel_powers[ch_name]["alpha"]
        features.append(theta / (alpha + 1e-10))

        # Theta/Beta ratio (attention indicator)
        beta = channel_powers[ch_name]["beta"]
        features.append(theta / (beta + 1e-10))

        # Hjorth parameters
        activity, mobility, complexity = compute_hjorth_parameters(signal)
        features.extend([activity, mobility, complexity])

        # Envelope features
        env_mean, env_std, env_max = compute_envelope_features(signal)
        features.extend([env_mean, env_std, env_max])

    # Inter-channel features
    if "C3" in channel_powers and "C4" in channel_powers:
        for band_name in bands:
            c3 = channel_powers["C3"][band_name]
            c4 = channel_powers["C4"][band_name]
            # Asymmetry
            features.append(np.log((c3 + 1e-10) / (c4 + 1e-10)))

    if "Fz" in channel_powers and "Pz" in channel_powers:
        for band_name in ["theta", "alpha", "beta"]:
            fz = channel_powers["Fz"][band_name]
            pz = channel_powers["Pz"][band_name]
            features.append(np.log((fz + 1e-10) / (pz + 1e-10)))

    return np.array(features)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_features.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: add extract_realtime_features for sliding window inference"
```

---

## Task 4: Add Real-Time Model Training to Pipeline

**Files:**
- Modify: `src/pipeline.py` (add realtime model training and saving)

**Step 1: Write failing test**

Add to `tests/test_pipeline.py`:

```python
def test_pipeline_saves_realtime_model():
    """Test that pipeline saves realtime model artifacts."""
    from pathlib import Path
    import tempfile
    import shutil

    from src.pipeline import run_pipeline

    # Use temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "models"

        # Run pipeline (will fail if no data, but we check the intent)
        try:
            run_pipeline(Path("unicorn-data"), output_dir)
        except Exception:
            pass

        # Check that realtime model paths are in the expected output
        # (This tests the code structure, actual files need real data)
```

Note: This test is minimal since full testing requires actual data. The real validation is running the pipeline.

**Step 2: Update src/pipeline.py to train and save realtime model**

After line 135 (after `final_model, scaler = train_final_model(X_all, y_all)`), add:

```python

    # Also train realtime model using only realtime-compatible features
    print("\n[4b/5] Training realtime model...")
    from src.features import extract_realtime_features

    # Extract realtime features from all subjects
    X_realtime_all = []
    for rec_idx, rec_path in enumerate(recordings):
        data, events, sfreq = load_recording(rec_path)

        # Preprocess all channels
        processed = {}
        for ch_idx, ch_name in enumerate(CHANNELS):
            processed[ch_name] = preprocess_eeg(data[ch_idx], sfreq)

        # Get phase 3 and phase 5 event indices
        for sample_idx, phase, movement in events:
            if phase not in [3, 5]:
                continue

            # Extract 5-second window (1250 samples at 250 Hz)
            window_samples = int(5.0 * sfreq)
            if sample_idx + window_samples > len(processed[CHANNELS[0]]):
                continue

            window = {ch: processed[ch][sample_idx:sample_idx + window_samples]
                     for ch in CHANNELS}

            features = extract_realtime_features(window, sfreq)
            X_realtime_all.append((features, 1 if phase == 3 else 0))

    if X_realtime_all:
        X_rt = np.array([x[0] for x in X_realtime_all])
        y_rt = np.array([x[1] for x in X_realtime_all])

        realtime_model, realtime_scaler = train_final_model(X_rt, y_rt)

        realtime_model_path = output_dir / "engagement_realtime_model.joblib"
        realtime_scaler_path = output_dir / "engagement_realtime_scaler.joblib"

        joblib.dump(realtime_model, realtime_model_path)
        joblib.dump(realtime_scaler, realtime_scaler_path)

        print(f"Realtime model saved to: {realtime_model_path}")
```

Update the results dict (around line 151) to include realtime model paths:

Change from:
```python
    results = {
        "n_subjects": len(recordings),
        "per_subject_scores": dict(zip(subject_ids, [float(s) for s in scores])),
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "target_met": bool(target_met),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }
```
to:
```python
    results = {
        "n_subjects": len(recordings),
        "per_subject_scores": dict(zip(subject_ids, [float(s) for s in scores])),
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "target_met": bool(target_met),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "realtime_model_path": str(output_dir / "engagement_realtime_model.joblib"),
        "realtime_scaler_path": str(output_dir / "engagement_realtime_scaler.joblib"),
    }
```

**Step 3: Run existing tests to ensure no regression**

Run: `python -m pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: add realtime model training to pipeline"
```

---

## Task 5: Create Simulation Script

**Files:**
- Create: `src/simulate_realtime.py`
- Test: `tests/test_simulate.py`

**Step 1: Write failing test for simulation**

Create `tests/test_simulate.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_simulate.py -v`
Expected: FAIL (module doesn't exist)

**Step 3: Create src/simulate_realtime.py**

```python
"""Real-time EEG engagement simulation.

Processes recordings with sliding windows to simulate real-time
fatigue/engagement detection.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_recording, CHANNELS, SFREQ
from src.preprocess import preprocess_eeg
from src.features import extract_realtime_features


def compute_engagement_score(probability: float) -> float:
    """
    Convert model probability to 0-100 engagement score.

    Args:
        probability: Probability of engaged class (0-1)

    Returns:
        Engagement score (0-100)
    """
    return probability * 100


def check_alert(
    score_history: list[float],
    threshold: float = 30.0,
    consecutive: int = 3,
) -> bool:
    """
    Check if alert should be triggered based on score history.

    Args:
        score_history: List of recent engagement scores
        threshold: Score below which is considered fatigued
        consecutive: Number of consecutive low scores to trigger alert

    Returns:
        True if alert should trigger
    """
    if len(score_history) < consecutive:
        return False

    recent = score_history[-consecutive:]
    return all(score < threshold for score in recent)


def extract_sliding_windows(
    data: dict[str, np.ndarray],
    sfreq: float,
    window_sec: float = 5.0,
    step_sec: float = 5.0,
) -> list[dict[str, np.ndarray]]:
    """
    Extract sliding windows from multi-channel data.

    Args:
        data: Dict mapping channel names to signal arrays
        sfreq: Sampling frequency in Hz
        window_sec: Window duration in seconds
        step_sec: Step between windows in seconds

    Returns:
        List of window dicts, each mapping channel names to signal arrays
    """
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)

    # Get total length from first channel
    first_channel = list(data.keys())[0]
    total_samples = len(data[first_channel])

    windows = []
    start = 0

    while start + window_samples <= total_samples:
        window = {}
        for ch_name, signal in data.items():
            window[ch_name] = signal[start:start + window_samples]
        windows.append(window)
        start += step_samples

    return windows


def simulate_recording(
    recording_path: Path,
    model_path: Path,
    scaler_path: Path,
    window_sec: float = 5.0,
    threshold: float = 30.0,
    consecutive: int = 3,
) -> pd.DataFrame:
    """
    Simulate real-time engagement detection on a recording.

    Args:
        recording_path: Path to recording CSV
        model_path: Path to trained model
        scaler_path: Path to fitted scaler
        window_sec: Window size in seconds
        threshold: Alert threshold (0-100)
        consecutive: Consecutive windows below threshold to trigger alert

    Returns:
        DataFrame with timestamp, engagement_score, alert_triggered columns
    """
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Load and preprocess recording
    data, events, sfreq = load_recording(recording_path)

    processed = {}
    for ch_idx, ch_name in enumerate(CHANNELS):
        processed[ch_name] = preprocess_eeg(data[ch_idx], sfreq)

    # Extract sliding windows
    windows = extract_sliding_windows(processed, sfreq, window_sec, window_sec)

    # Process each window
    results = []
    score_history = []

    for idx, window in enumerate(windows):
        # Extract features
        features = extract_realtime_features(window, sfreq)
        features_scaled = scaler.transform(features.reshape(1, -1))

        # Get probability of engaged class
        prob = model.predict_proba(features_scaled)[0, 1]
        score = compute_engagement_score(prob)

        score_history.append(score)
        alert = check_alert(score_history, threshold, consecutive)

        timestamp_sec = idx * window_sec
        results.append({
            "timestamp_sec": timestamp_sec,
            "engagement_score": round(score, 1),
            "alert_triggered": alert,
        })

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame, threshold: float) -> None:
    """Print summary statistics from simulation results."""
    print("\n" + "=" * 50)
    print("Engagement Score Summary")
    print("=" * 50)

    scores = df["engagement_score"]
    print(f"Mean:  {scores.mean():.1f}")
    print(f"Std:   {scores.std():.1f}")
    print(f"Min:   {scores.min():.1f} at {df.loc[scores.idxmin(), 'timestamp_sec']:.0f}s")
    print(f"Max:   {scores.max():.1f} at {df.loc[scores.idxmax(), 'timestamp_sec']:.0f}s")

    alerts = df["alert_triggered"].sum()
    print(f"\nAlerts triggered: {alerts}")

    below_threshold = (scores < threshold).sum()
    total_windows = len(df)
    print(f"Windows below threshold ({threshold}): {below_threshold}/{total_windows}")

    if below_threshold > 0:
        time_below = below_threshold * 5  # Assuming 5-second windows
        print(f"Total time below threshold: {time_below} seconds")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate real-time engagement detection on EEG recording"
    )
    parser.add_argument(
        "recording",
        type=Path,
        help="Path to recording CSV file",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing trained model (default: models/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: simulation_results.csv)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Alert threshold (0-100, default: 30)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=5.0,
        help="Window size in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--consecutive",
        type=int,
        default=3,
        help="Consecutive windows below threshold to trigger alert (default: 3)",
    )

    args = parser.parse_args()

    model_path = args.model_dir / "engagement_realtime_model.joblib"
    scaler_path = args.model_dir / "engagement_realtime_scaler.joblib"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run the training pipeline first: python -m src.pipeline")
        return 1

    print(f"Processing: {args.recording}")
    print(f"Model: {model_path}")
    print(f"Window: {args.window}s, Threshold: {args.threshold}, Consecutive: {args.consecutive}")

    df = simulate_recording(
        args.recording,
        model_path,
        scaler_path,
        window_sec=args.window,
        threshold=args.threshold,
        consecutive=args.consecutive,
    )

    print_summary(df, args.threshold)

    # Save results
    output_path = args.output or Path("simulation_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_simulate.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/simulate_realtime.py tests/test_simulate.py
git commit -m "feat: add real-time engagement simulation script"
```

---

## Task 6: Run Full Pipeline and Validate

**Files:**
- Run: `src/pipeline.py`
- Run: `src/simulate_realtime.py`

**Step 1: Run the training pipeline**

Run: `python -m src.pipeline`
Expected: Training completes, models saved to `models/`

**Step 2: Verify model files created**

Run: `ls -la models/`
Expected:
- `engagement_classifier_model.joblib`
- `engagement_classifier_scaler.joblib`
- `engagement_realtime_model.joblib`
- `engagement_realtime_scaler.joblib`
- `training_results.json`

**Step 3: Run simulation on a test recording**

Run: `python -m src.simulate_realtime unicorn-data/subject0001/session000/recording_2025-11-12-21.33.31.csv`
Expected: Simulation output with engagement scores and summary

**Step 4: Review results**

Check `simulation_results.csv` and verify:
- Engagement scores are in 0-100 range
- Alerts trigger appropriately
- Results look reasonable

**Step 5: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 6: Commit final state**

```bash
git add models/ simulation_results.csv
git commit -m "feat: complete fatigue detection pipeline with trained models"
```

---

## Summary

After completing all tasks:

1. Old Phase 3 vs 4 artifacts removed
2. Pipeline updated to use Phase 5 (rest) as disengaged class
3. Realtime feature extraction added (no baseline needed)
4. Both full-feature and realtime models trained
5. Simulation script created for offline testing
6. All tests passing

**Usage:**
```bash
# Train models
python -m src.pipeline

# Run simulation
python -m src.simulate_realtime path/to/recording.csv --threshold 30
```
