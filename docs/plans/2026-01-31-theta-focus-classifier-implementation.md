# Theta Focus Classifier Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a classifier that uses frontal midline theta power to classify focused vs not-focused states with >90% LOSO accuracy.

**Architecture:** Load CSV data, bandpass filter (1-40 Hz), extract 2-second epochs at phase markers, compute theta power (4-8 Hz) from Fz electrode, train logistic regression with leave-one-subject-out cross-validation.

**Tech Stack:** Python, uv, MNE (filtering), scipy (Welch PSD), scikit-learn (classification), numpy, pandas, joblib (model serialization)

---

## Setup

### Task 0: Initialize Project

**Step 1: Create project structure**

```bash
mkdir -p "./src"
mkdir -p "./models"
mkdir -p "./tests"
```

**Step 2: Initialize uv and install dependencies**

```bash
cd "."
uv init --name theta-focus-classifier
uv add numpy pandas scipy mne scikit-learn joblib
uv add --dev pytest ruff
```

**Step 3: Create src/__init__.py**

Create: `src/__init__.py`

```python
"""Theta-based focus classifier for EEG data."""
```

---

## Core Implementation

### Task 1: Data Loading Module

**Files:**
- Create: `src/data_loader.py`
- Create: `tests/test_data_loader.py`

**Step 1: Write the failing test**

Create `tests/test_data_loader.py`:

```python
"""Tests for data loading functionality."""

import numpy as np
from pathlib import Path

from src.data_loader import load_recording, get_complete_recordings


def test_load_recording_returns_data_and_events():
    """Test that load_recording returns EEG data and event markers."""
    data_dir = Path("./unicorn-data")
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
    data_dir = Path("./unicorn-data")

    recordings = get_complete_recordings(data_dir)

    # Should find 10 complete recordings
    assert len(recordings) == 10
    assert all(Path(r).exists() for r in recordings)
```

**Step 2: Run test to verify it fails**

```bash
cd "."
uv run pytest tests/test_data_loader.py -v
```

Expected: FAIL with "ModuleNotFoundError" or "cannot import name"

**Step 3: Write minimal implementation**

Create `src/data_loader.py`:

```python
"""Load and parse Unicorn EEG recordings."""

import numpy as np
import pandas as pd
from pathlib import Path


# Channel names in order from CSV
CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
SFREQ = 250.0  # Sampling frequency in Hz


def load_recording(csv_path: Path) -> tuple[np.ndarray, list[tuple[int, int, int]], float]:
    """
    Load a single recording from CSV.

    Args:
        csv_path: Path to the CSV file

    Returns:
        data: EEG data array of shape (n_channels, n_samples)
        events: List of (sample_idx, phase, movement) tuples
        sfreq: Sampling frequency (250 Hz)
    """
    df = pd.read_csv(csv_path)

    # Extract EEG channels (columns 1-8, excluding timestamps and stim)
    data = df[CHANNELS].values.T  # Transpose to (n_channels, n_samples)

    # Parse events from stim column
    events = []
    stim = df["stim"].values
    for idx, val in enumerate(stim):
        if val != 0:
            stim_int = int(val)
            phase = (stim_int // 10) % 10
            movement = stim_int % 10
            events.append((idx, phase, movement))

    return data, events, SFREQ


def get_complete_recordings(data_dir: Path) -> list[Path]:
    """
    Find all complete recordings (those with 100 imagery trials).

    Args:
        data_dir: Path to unicorn-data directory

    Returns:
        List of paths to complete recording CSVs
    """
    complete = []

    for csv_path in sorted(data_dir.glob("subject*/session*/*.csv")):
        df = pd.read_csv(csv_path)
        stim = df["stim"].values

        # Count phase 3 (imagery) events
        phase3_count = 0
        for val in stim:
            if val != 0:
                phase = (int(val) // 10) % 10
                if phase == 3:
                    phase3_count += 1

        if phase3_count == 100:
            complete.append(csv_path)

    return complete
```

**Step 4: Run test to verify it passes**

```bash
cd "."
uv run pytest tests/test_data_loader.py -v
```

Expected: PASS

---

### Task 2: Preprocessing Module

**Files:**
- Create: `src/preprocess.py`
- Create: `tests/test_preprocess.py`

**Step 1: Write the failing test**

Create `tests/test_preprocess.py`:

```python
"""Tests for preprocessing functionality."""

import numpy as np

from src.preprocess import bandpass_filter, notch_filter, preprocess_eeg


def test_bandpass_filter_removes_dc_offset():
    """Test that bandpass filter removes DC component."""
    sfreq = 250.0
    # Create signal with DC offset + 10 Hz sine wave
    t = np.arange(0, 2, 1/sfreq)
    dc_offset = 73000  # Typical EEG offset in microvolts
    signal = dc_offset + 10 * np.sin(2 * np.pi * 10 * t)

    filtered = bandpass_filter(signal, sfreq, l_freq=1.0, h_freq=40.0)

    # DC offset should be removed (mean near zero)
    assert abs(np.mean(filtered)) < 100  # Much less than 73000


def test_notch_filter_removes_60hz():
    """Test that notch filter attenuates 60 Hz."""
    sfreq = 250.0
    t = np.arange(0, 2, 1/sfreq)
    # Signal with 10 Hz + 60 Hz noise
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)

    filtered = notch_filter(signal, sfreq, freq=60.0)

    # Compute power at 60 Hz before and after
    from scipy.signal import welch
    _, psd_before = welch(signal, sfreq, nperseg=250)
    _, psd_after = welch(filtered, sfreq, nperseg=250)

    # 60 Hz power should be reduced
    idx_60 = 60  # At 1 Hz resolution, index 60 is 60 Hz
    assert psd_after[idx_60] < psd_before[idx_60] * 0.1  # 90% reduction


def test_preprocess_eeg_full_pipeline():
    """Test full preprocessing pipeline."""
    sfreq = 250.0
    n_samples = 1000
    # Simulate raw EEG: DC offset + signal + 60 Hz noise
    np.random.seed(42)
    raw = 73000 + np.random.randn(n_samples) * 10 + 0.5 * np.sin(2 * np.pi * 60 * np.arange(n_samples) / sfreq)

    processed = preprocess_eeg(raw, sfreq)

    # Should be zero-centered and cleaned
    assert abs(np.mean(processed)) < 50
    assert processed.shape == raw.shape
```

**Step 2: Run test to verify it fails**

```bash
cd "."
uv run pytest tests/test_preprocess.py -v
```

Expected: FAIL with "cannot import name"

**Step 3: Write minimal implementation**

Create `src/preprocess.py`:

```python
"""EEG preprocessing: filtering and artifact removal."""

import numpy as np
import mne


def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
) -> np.ndarray:
    """
    Apply bandpass filter to remove DC drift and high-frequency noise.

    Args:
        data: 1D signal array
        sfreq: Sampling frequency in Hz
        l_freq: Low cutoff frequency (Hz)
        h_freq: High cutoff frequency (Hz)

    Returns:
        Filtered signal
    """
    # MNE filter expects 2D array (n_channels, n_samples)
    data_2d = data.reshape(1, -1)
    filtered = mne.filter.filter_data(
        data_2d, sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False
    )
    return filtered.flatten()


def notch_filter(
    data: np.ndarray,
    sfreq: float,
    freq: float = 60.0,
) -> np.ndarray:
    """
    Apply notch filter to remove power line interference.

    Args:
        data: 1D signal array
        sfreq: Sampling frequency in Hz
        freq: Frequency to notch out (Hz)

    Returns:
        Filtered signal
    """
    data_2d = data.reshape(1, -1)
    filtered = mne.filter.notch_filter(
        data_2d, sfreq, freqs=freq, verbose=False
    )
    return filtered.flatten()


def preprocess_eeg(data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Full preprocessing pipeline: bandpass + notch filter.

    Args:
        data: 1D signal array (single channel)
        sfreq: Sampling frequency in Hz

    Returns:
        Preprocessed signal
    """
    # Bandpass filter (1-40 Hz)
    filtered = bandpass_filter(data, sfreq, l_freq=1.0, h_freq=40.0)
    # Notch filter (60 Hz)
    filtered = notch_filter(filtered, sfreq, freq=60.0)
    return filtered
```

**Step 4: Run test to verify it passes**

```bash
cd "."
uv run pytest tests/test_preprocess.py -v
```

Expected: PASS

---

### Task 3: Epoch Extraction Module

**Files:**
- Create: `src/epochs.py`
- Create: `tests/test_epochs.py`

**Step 1: Write the failing test**

Create `tests/test_epochs.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
cd "."
uv run pytest tests/test_epochs.py -v
```

Expected: FAIL with "cannot import name"

**Step 3: Write minimal implementation**

Create `src/epochs.py`:

```python
"""Extract epochs from continuous EEG data."""

import numpy as np


def extract_epochs(
    signal: np.ndarray,
    events: list[tuple[int, int, int]],
    sfreq: float,
    duration: float = 2.0,
) -> list[np.ndarray]:
    """
    Extract fixed-duration epochs starting at each event.

    Args:
        signal: 1D preprocessed signal array
        events: List of (sample_idx, phase, movement) tuples
        sfreq: Sampling frequency in Hz
        duration: Epoch duration in seconds

    Returns:
        List of epoch arrays
    """
    n_samples = int(duration * sfreq)
    epochs = []

    for sample_idx, phase, movement in events:
        # Check if epoch would extend past end of signal
        if sample_idx + n_samples <= len(signal):
            epoch = signal[sample_idx : sample_idx + n_samples]
            epochs.append(epoch)

    return epochs


def extract_labeled_epochs(
    signal: np.ndarray,
    events: list[tuple[int, int, int]],
    sfreq: float,
    duration: float = 2.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Extract epochs and separate by focus label.

    Phase 3 (imagery) = Focused
    Phase 5 (rest) = Not Focused

    Args:
        signal: 1D preprocessed signal array
        events: List of (sample_idx, phase, movement) tuples
        sfreq: Sampling frequency in Hz
        duration: Epoch duration in seconds

    Returns:
        Tuple of (focused_epochs, not_focused_epochs)
    """
    n_samples = int(duration * sfreq)
    focused = []
    not_focused = []

    for sample_idx, phase, movement in events:
        # Check if epoch would extend past end of signal
        if sample_idx + n_samples > len(signal):
            continue

        epoch = signal[sample_idx : sample_idx + n_samples]

        if phase == 3:  # Imagery = Focused
            focused.append(epoch)
        elif phase == 5:  # Rest = Not Focused
            not_focused.append(epoch)
        # Other phases are ignored

    return focused, not_focused
```

**Step 4: Run test to verify it passes**

```bash
cd "."
uv run pytest tests/test_epochs.py -v
```

Expected: PASS

---

### Task 4: Feature Extraction Module

**Files:**
- Create: `src/features.py`
- Create: `tests/test_features.py`

**Step 1: Write the failing test**

Create `tests/test_features.py`:

```python
"""Tests for feature extraction functionality."""

import numpy as np

from src.features import compute_theta_power, extract_features


def test_compute_theta_power_detects_theta():
    """Test that theta power is higher for theta-dominant signal."""
    sfreq = 250.0
    t = np.arange(0, 2, 1/sfreq)  # 2 seconds

    # Signal with strong 6 Hz (theta) component
    theta_signal = np.sin(2 * np.pi * 6 * t)

    # Signal with strong 20 Hz (beta) component
    beta_signal = np.sin(2 * np.pi * 20 * t)

    theta_power_high = compute_theta_power(theta_signal, sfreq)
    theta_power_low = compute_theta_power(beta_signal, sfreq)

    # Theta signal should have higher theta power
    assert theta_power_high > theta_power_low * 10


def test_compute_theta_power_returns_scalar():
    """Test that theta power returns a single scalar value."""
    sfreq = 250.0
    signal = np.random.randn(500)  # 2 seconds

    power = compute_theta_power(signal, sfreq)

    assert isinstance(power, float)
    assert power > 0  # Power is always positive


def test_extract_features_correct_shape():
    """Test that extract_features returns correct array shape."""
    sfreq = 250.0
    epochs = [np.random.randn(500) for _ in range(10)]

    features = extract_features(epochs, sfreq)

    # Should be (n_epochs, 1) - one feature per epoch
    assert features.shape == (10, 1)


def test_extract_features_log_transform():
    """Test that log transform is applied (values should be reasonable range)."""
    sfreq = 250.0
    epochs = [np.random.randn(500) * 10 for _ in range(10)]

    features = extract_features(epochs, sfreq, log_transform=True)

    # Log-transformed values should be in reasonable range (not huge like raw power)
    assert np.all(features < 100)
    assert np.all(features > -100)
```

**Step 2: Run test to verify it fails**

```bash
cd "."
uv run pytest tests/test_features.py -v
```

Expected: FAIL with "cannot import name"

**Step 3: Write minimal implementation**

Create `src/features.py`:

```python
"""Feature extraction: theta power computation."""

import numpy as np
from scipy.signal import welch


# Theta band frequency range
THETA_LOW = 4.0  # Hz
THETA_HIGH = 8.0  # Hz


def compute_theta_power(
    epoch: np.ndarray,
    sfreq: float,
    theta_low: float = THETA_LOW,
    theta_high: float = THETA_HIGH,
) -> float:
    """
    Compute theta band power for a single epoch using Welch's method.

    Args:
        epoch: 1D signal array (single epoch)
        sfreq: Sampling frequency in Hz
        theta_low: Lower bound of theta band (Hz)
        theta_high: Upper bound of theta band (Hz)

    Returns:
        Mean power in theta band
    """
    # Welch's method with 1-second window, 50% overlap
    nperseg = int(sfreq)  # 1 second = 250 samples
    freqs, psd = welch(epoch, sfreq, nperseg=nperseg, noverlap=nperseg // 2)

    # Find indices for theta band
    theta_mask = (freqs >= theta_low) & (freqs <= theta_high)

    # Mean power in theta band
    theta_power = np.mean(psd[theta_mask])

    return float(theta_power)


def extract_features(
    epochs: list[np.ndarray],
    sfreq: float,
    log_transform: bool = True,
) -> np.ndarray:
    """
    Extract theta power features from a list of epochs.

    Args:
        epochs: List of 1D epoch arrays
        sfreq: Sampling frequency in Hz
        log_transform: Whether to apply log transform to power values

    Returns:
        Feature array of shape (n_epochs, 1)
    """
    features = []

    for epoch in epochs:
        theta_power = compute_theta_power(epoch, sfreq)
        if log_transform:
            theta_power = np.log(theta_power + 1e-10)  # Small epsilon to avoid log(0)
        features.append([theta_power])

    return np.array(features)
```

**Step 4: Run test to verify it passes**

```bash
cd "."
uv run pytest tests/test_features.py -v
```

Expected: PASS

---

### Task 5: Training Pipeline

**Files:**
- Create: `src/train.py`
- Create: `tests/test_train.py`

**Step 1: Write the failing test**

Create `tests/test_train.py`:

```python
"""Tests for training pipeline."""

import numpy as np

from src.train import train_loso_cv, train_final_model


def test_train_loso_cv_returns_scores():
    """Test that LOSO CV returns accuracy scores for each subject."""
    # Simulated data: 5 subjects, 20 epochs each
    np.random.seed(42)
    n_subjects = 5
    n_epochs_per_class = 10

    X_by_subject = []
    y_by_subject = []

    for i in range(n_subjects):
        # Create separable data: focused has higher feature values
        focused = np.random.randn(n_epochs_per_class, 1) + 2
        not_focused = np.random.randn(n_epochs_per_class, 1) - 2

        X = np.vstack([focused, not_focused])
        y = np.array([1] * n_epochs_per_class + [0] * n_epochs_per_class)

        X_by_subject.append(X)
        y_by_subject.append(y)

    scores, mean_acc, std_acc = train_loso_cv(X_by_subject, y_by_subject)

    # Should return one score per subject
    assert len(scores) == n_subjects
    # With separable data, accuracy should be high
    assert mean_acc > 0.8
    assert 0 <= std_acc <= 1


def test_train_final_model_returns_model():
    """Test that final model training returns a fitted model."""
    np.random.seed(42)

    # Simple separable data
    X = np.vstack([
        np.random.randn(50, 1) + 2,  # focused
        np.random.randn(50, 1) - 2,  # not focused
    ])
    y = np.array([1] * 50 + [0] * 50)

    model, scaler = train_final_model(X, y)

    # Model should be able to predict
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    assert len(predictions) == len(y)

    # Should have high accuracy on training data
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.9
```

**Step 2: Run test to verify it fails**

```bash
cd "."
uv run pytest tests/test_train.py -v
```

Expected: FAIL with "cannot import name"

**Step 3: Write minimal implementation**

Create `src/train.py`:

```python
"""Training pipeline with LOSO cross-validation."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_loso_cv(
    X_by_subject: list[np.ndarray],
    y_by_subject: list[np.ndarray],
) -> tuple[list[float], float, float]:
    """
    Train with Leave-One-Subject-Out cross-validation.

    Args:
        X_by_subject: List of feature arrays, one per subject
        y_by_subject: List of label arrays, one per subject

    Returns:
        Tuple of (per_subject_scores, mean_accuracy, std_accuracy)
    """
    n_subjects = len(X_by_subject)
    scores = []

    for test_idx in range(n_subjects):
        # Prepare train/test split
        X_train_list = []
        y_train_list = []

        for i in range(n_subjects):
            if i != test_idx:
                X_train_list.append(X_by_subject[i])
                y_train_list.append(y_by_subject[i])

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test = X_by_subject[test_idx]
        y_test = y_by_subject[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train logistic regression
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

        # Evaluate
        accuracy = model.score(X_test_scaled, y_test)
        scores.append(accuracy)

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)

    return scores, mean_acc, std_acc


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[LogisticRegression, StandardScaler]:
    """
    Train final model on all data.

    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)

    Returns:
        Tuple of (fitted model, fitted scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    return model, scaler
```

**Step 4: Run test to verify it passes**

```bash
cd "."
uv run pytest tests/test_train.py -v
```

Expected: PASS

---

### Task 6: Main Pipeline Script

**Files:**
- Create: `src/pipeline.py`

**Step 1: Write the main pipeline script**

Create `src/pipeline.py`:

```python
"""Main pipeline: load data, preprocess, extract features, train, evaluate."""

import json
from pathlib import Path

import joblib
import numpy as np

from src.data_loader import load_recording, get_complete_recordings, CHANNELS
from src.preprocess import preprocess_eeg
from src.epochs import extract_labeled_epochs
from src.features import extract_features
from src.train import train_loso_cv, train_final_model


def run_pipeline(data_dir: Path, output_dir: Path) -> dict:
    """
    Run the full training pipeline.

    Args:
        data_dir: Path to unicorn-data directory
        output_dir: Path to save model and results

    Returns:
        Dictionary with results
    """
    print("=" * 60)
    print("Theta Focus Classifier - Training Pipeline")
    print("=" * 60)

    # Step 1: Find complete recordings
    print("\n[1/5] Finding complete recordings...")
    recordings = get_complete_recordings(data_dir)
    print(f"Found {len(recordings)} complete recordings")

    # Step 2: Load and preprocess data for each subject
    print("\n[2/5] Loading and preprocessing data...")
    X_by_subject = []
    y_by_subject = []
    subject_ids = []

    for rec_path in recordings:
        subject_id = rec_path.parent.parent.name
        print(f"  Processing {subject_id}...")

        # Load recording
        data, events, sfreq = load_recording(rec_path)

        # Get Fz channel (index 0)
        fz_idx = CHANNELS.index("Fz")
        fz_signal = data[fz_idx]

        # Preprocess
        fz_processed = preprocess_eeg(fz_signal, sfreq)

        # Extract labeled epochs
        focused_epochs, not_focused_epochs = extract_labeled_epochs(
            fz_processed, events, sfreq, duration=2.0
        )

        # Balance classes by subsampling not-focused
        n_focused = len(focused_epochs)
        if len(not_focused_epochs) > n_focused:
            np.random.seed(42)
            indices = np.random.choice(len(not_focused_epochs), n_focused, replace=False)
            not_focused_epochs = [not_focused_epochs[i] for i in indices]

        print(f"    Focused epochs: {len(focused_epochs)}, Not-focused epochs: {len(not_focused_epochs)}")

        # Extract features
        focused_features = extract_features(focused_epochs, sfreq, log_transform=True)
        not_focused_features = extract_features(not_focused_epochs, sfreq, log_transform=True)

        # Combine
        X = np.vstack([focused_features, not_focused_features])
        y = np.array([1] * len(focused_features) + [0] * len(not_focused_features))

        X_by_subject.append(X)
        y_by_subject.append(y)
        subject_ids.append(subject_id)

    # Step 3: LOSO Cross-validation
    print("\n[3/5] Running Leave-One-Subject-Out cross-validation...")
    scores, mean_acc, std_acc = train_loso_cv(X_by_subject, y_by_subject)

    print("\nPer-subject accuracy:")
    for sid, score in zip(subject_ids, scores):
        status = "PASS" if score >= 0.8 else "FAIL"
        print(f"  {sid}: {score:.1%} [{status}]")

    print(f"\nMean accuracy: {mean_acc:.1%} (+/- {std_acc:.1%})")

    # Step 4: Check if target met
    target_met = mean_acc >= 0.90
    print(f"\nTarget (>90%): {'MET' if target_met else 'NOT MET'}")

    # Step 5: Train final model on all data
    print("\n[4/5] Training final model on all data...")
    X_all = np.vstack(X_by_subject)
    y_all = np.concatenate(y_by_subject)

    final_model, scaler = train_final_model(X_all, y_all)

    # Step 6: Save model and scaler
    print("\n[5/5] Saving model...")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "theta_classifier_model.joblib"
    scaler_path = output_dir / "theta_classifier_scaler.joblib"

    joblib.dump(final_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

    # Save results as JSON
    results = {
        "n_subjects": len(recordings),
        "per_subject_scores": dict(zip(subject_ids, [float(s) for s in scores])),
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "target_met": target_met,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }

    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    data_dir = Path("./unicorn-data")
    output_dir = Path("./models")

    results = run_pipeline(data_dir, output_dir)
```

**Step 2: Run the full pipeline**

```bash
cd "."
uv run python -m src.pipeline
```

Expected: Pipeline runs and outputs accuracy results

---

### Task 7: Run All Tests

**Step 1: Run full test suite**

```bash
cd "."
uv run pytest tests/ -v
```

Expected: All tests pass

**Step 2: Run linting**

```bash
cd "."
uv run ruff check src/ tests/
```

Expected: No errors (or fix any that appear)

---

## Summary

After completing all tasks:

1. **If accuracy >= 90%**: Success! Ready to adapt for real-time inference.
2. **If accuracy < 90%**: Implement fallback (Task 8 below).

---

### Task 8 (Fallback): Add Theta/Beta Ratio Feature

**Only implement if accuracy < 90%**

**Files:**
- Modify: `src/features.py`
- Modify: `tests/test_features.py`

Add `compute_theta_beta_ratio()` function and update `extract_features()` to include it as a second feature. Re-run pipeline to evaluate improvement.
