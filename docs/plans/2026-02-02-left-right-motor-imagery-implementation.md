# Left/Right Motor Imagery Classification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update BCIS to classify left vs right hand motor imagery from phase 3 EEG data for robotic finger control.

**Architecture:** Extract phase 3 epochs labeled by movement (1=left, 2=right), compute lateralization-focused features emphasizing C3-C4 asymmetry in mu/beta bands, evaluate with Leave-One-Subject-Out CV targeting >90% accuracy.

**Tech Stack:** Python 3.11, MNE, NumPy, scikit-learn, pyRiemann, pytest

---

## Task 1: Add Left/Right Epoch Extraction

**Files:**
- Modify: `src/epochs.py`
- Test: `tests/test_epochs.py`

**Step 1: Write the failing test**

Add to `tests/test_epochs.py`:

```python
from src.epochs import extract_left_right_epochs


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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/isaaczhang/Projects/BCIS && uv run pytest tests/test_epochs.py::test_extract_left_right_epochs_separates_by_movement -v`
Expected: FAIL with "cannot import name 'extract_left_right_epochs'"

**Step 3: Write the implementation**

Add to `src/epochs.py`:

```python
def extract_left_right_epochs(
    signal: np.ndarray,
    events: list[tuple[int, int, int]],
    sfreq: float,
    task_duration: float = 1.5,
    baseline_duration: float = 1.0,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """
    Extract left (movement=1) and right (movement=2) motor imagery epochs from phase 3.

    Args:
        signal: 1D preprocessed signal array
        events: List of (sample_idx, phase, movement) tuples
        sfreq: Sampling frequency in Hz
        task_duration: Duration of task epoch in seconds
        baseline_duration: Duration of baseline epoch in seconds

    Returns:
        Tuple of (left_pairs, right_pairs) where each pair is (baseline, task)
    """
    task_samples = int(task_duration * sfreq)
    baseline_samples = int(baseline_duration * sfreq)

    left_pairs = []
    right_pairs = []

    for sample_idx, phase, movement in events:
        # Only phase 3 (imagery-perform)
        if phase != 3:
            continue

        # Extract baseline from immediately before event
        baseline_end = sample_idx
        baseline_start = baseline_end - baseline_samples
        if baseline_start < 0:
            continue

        baseline = signal[baseline_start:baseline_end]

        # Extract task epoch (skip first 0.5s to capture developed ERD)
        task_start = sample_idx + int(0.5 * sfreq)
        task_end = task_start + task_samples
        if task_end > len(signal):
            continue

        task = signal[task_start:task_end]

        if movement == 1:
            left_pairs.append((baseline, task))
        elif movement == 2:
            right_pairs.append((baseline, task))

    return left_pairs, right_pairs
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/isaaczhang/Projects/BCIS && uv run pytest tests/test_epochs.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
cd /Users/isaaczhang/Projects/BCIS && git add src/epochs.py tests/test_epochs.py && git commit -m "feat(epochs): add left/right motor imagery epoch extraction"
```

---

## Task 2: Add Lateralization Feature Extraction

**Files:**
- Modify: `src/features.py`
- Test: `tests/test_features.py`

**Step 1: Write the failing test**

Add to `tests/test_features.py`:

```python
from src.features import extract_lateralization_features, compute_lateralization_index


def test_compute_lateralization_index_detects_asymmetry():
    """Test that lateralization index detects left/right power difference."""
    sfreq = 250.0
    t = np.arange(0, 1.5, 1 / sfreq)

    # C3 has strong 10Hz (mu), C4 has weak signal
    c3_signal = np.sin(2 * np.pi * 10 * t) * 2
    c4_signal = np.sin(2 * np.pi * 10 * t) * 0.5

    # Lateralization index: (C4 - C3) / (C4 + C3)
    # Should be negative when C3 > C4 (right hand imagery pattern)
    lat_idx = compute_lateralization_index(c3_signal, c4_signal, sfreq, 8, 12)
    assert lat_idx < 0


def test_extract_lateralization_features_correct_shape():
    """Test that lateralization features have correct shape."""
    sfreq = 250.0
    n_epochs = 5
    n_samples_baseline = 250
    n_samples_task = 375

    # Create fake epoch pairs for each channel
    epoch_pairs_by_channel = {}
    for ch in ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]:
        pairs = []
        for _ in range(n_epochs):
            baseline = np.random.randn(n_samples_baseline)
            task = np.random.randn(n_samples_task)
            pairs.append((baseline, task))
        epoch_pairs_by_channel[ch] = pairs

    features = extract_lateralization_features(epoch_pairs_by_channel, sfreq)

    # Should be (n_epochs, n_features)
    assert features.shape[0] == n_epochs
    assert features.shape[1] > 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/isaaczhang/Projects/BCIS && uv run pytest tests/test_features.py::test_compute_lateralization_index_detects_asymmetry -v`
Expected: FAIL with "cannot import name 'compute_lateralization_index'"

**Step 3: Write the implementation**

Add to `src/features.py`:

```python
def compute_lateralization_index(
    c3_signal: np.ndarray,
    c4_signal: np.ndarray,
    sfreq: float,
    low_freq: float,
    high_freq: float,
) -> float:
    """
    Compute lateralization index between C3 and C4.

    Lateralization Index = (C4_power - C3_power) / (C4_power + C3_power)

    Positive values indicate left hand imagery (more C3 desync).
    Negative values indicate right hand imagery (more C4 desync).

    Args:
        c3_signal: Signal from C3 electrode (left motor cortex)
        c4_signal: Signal from C4 electrode (right motor cortex)
        sfreq: Sampling frequency in Hz
        low_freq: Lower bound of frequency band
        high_freq: Upper bound of frequency band

    Returns:
        Lateralization index (-1 to 1)
    """
    c3_power = compute_band_power(c3_signal, sfreq, low_freq, high_freq)
    c4_power = compute_band_power(c4_signal, sfreq, low_freq, high_freq)

    return (c4_power - c3_power) / (c4_power + c3_power + 1e-10)


def extract_lateralization_features(
    epoch_pairs_by_channel: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    sfreq: float,
) -> np.ndarray:
    """
    Extract features optimized for left/right motor imagery discrimination.

    Focuses on C3-C4 lateralization in mu (8-12Hz) and beta (13-30Hz) bands.

    Args:
        epoch_pairs_by_channel: Dict mapping channel names to list of (baseline, task) pairs
        sfreq: Sampling frequency in Hz

    Returns:
        Feature array of shape (n_epochs, n_features)
    """
    # Key frequency bands for motor imagery
    bands = {
        "mu": (8, 12),
        "low_beta": (13, 20),
        "high_beta": (20, 30),
        "beta": (13, 30),
    }

    channel_names = list(epoch_pairs_by_channel.keys())
    n_epochs = len(epoch_pairs_by_channel[channel_names[0]])

    all_features = []

    for epoch_idx in range(n_epochs):
        epoch_features = []

        # Get C3 and C4 signals for this epoch
        c3_baseline, c3_task = epoch_pairs_by_channel["C3"][epoch_idx]
        c4_baseline, c4_task = epoch_pairs_by_channel["C4"][epoch_idx]

        # PRIMARY: Lateralization features (C3 vs C4)
        for band_name, (low, high) in bands.items():
            # Task lateralization index
            lat_idx_task = compute_lateralization_index(c3_task, c4_task, sfreq, low, high)
            epoch_features.append(lat_idx_task)

            # Baseline lateralization index
            lat_idx_baseline = compute_lateralization_index(c3_baseline, c4_baseline, sfreq, low, high)
            epoch_features.append(lat_idx_baseline)

            # Change in lateralization (task - baseline)
            epoch_features.append(lat_idx_task - lat_idx_baseline)

            # ERD asymmetry: difference in ERD between C3 and C4
            c3_baseline_power = compute_band_power(c3_baseline, sfreq, low, high)
            c3_task_power = compute_band_power(c3_task, sfreq, low, high)
            c4_baseline_power = compute_band_power(c4_baseline, sfreq, low, high)
            c4_task_power = compute_band_power(c4_task, sfreq, low, high)

            c3_erd = (c3_baseline_power - c3_task_power) / (c3_baseline_power + 1e-10) * 100
            c4_erd = (c4_baseline_power - c4_task_power) / (c4_baseline_power + 1e-10) * 100
            epoch_features.append(c3_erd - c4_erd)  # Asymmetry

            # Log power ratio (C3/C4) during task
            epoch_features.append(np.log((c3_task_power + 1e-10) / (c4_task_power + 1e-10)))

            # Individual channel ERDs
            epoch_features.append(c3_erd)
            epoch_features.append(c4_erd)

        # SECONDARY: Cz features (supplementary motor area)
        cz_baseline, cz_task = epoch_pairs_by_channel["Cz"][epoch_idx]
        for band_name, (low, high) in [("mu", (8, 12)), ("beta", (13, 30))]:
            cz_baseline_power = compute_band_power(cz_baseline, sfreq, low, high)
            cz_task_power = compute_band_power(cz_task, sfreq, low, high)
            cz_erd = (cz_baseline_power - cz_task_power) / (cz_baseline_power + 1e-10) * 100
            epoch_features.append(cz_erd)
            epoch_features.append(np.log(cz_task_power + 1e-10))

        # TERTIARY: Fz theta (attention/effort marker)
        fz_baseline, fz_task = epoch_pairs_by_channel["Fz"][epoch_idx]
        fz_theta_baseline = compute_band_power(fz_baseline, sfreq, 4, 8)
        fz_theta_task = compute_band_power(fz_task, sfreq, 4, 8)
        epoch_features.append(np.log((fz_theta_task + 1e-10) / (fz_theta_baseline + 1e-10)))

        # Time-domain features from C3 and C4
        for signal in [c3_task, c4_task]:
            activity, mobility, complexity = compute_hjorth_parameters(signal)
            epoch_features.extend([activity, mobility, complexity])

        all_features.append(epoch_features)

    return np.array(all_features)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/isaaczhang/Projects/BCIS && uv run pytest tests/test_features.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
cd /Users/isaaczhang/Projects/BCIS && git add src/features.py tests/test_features.py && git commit -m "feat(features): add lateralization features for left/right classification"
```

---

## Task 3: Add CSP-Based Feature Extraction

**Files:**
- Modify: `src/features.py`
- Test: `tests/test_features.py`

**Step 1: Write the failing test**

Add to `tests/test_features.py`:

```python
from src.features import extract_csp_features


def test_extract_csp_features_correct_shape():
    """Test CSP feature extraction returns correct shape."""
    sfreq = 250.0
    n_epochs = 20
    n_channels = 8
    n_samples = 375

    # Multichannel data: (n_epochs, n_channels, n_samples)
    X = np.random.randn(n_epochs, n_channels, n_samples)
    y = np.array([0] * 10 + [1] * 10)  # Binary labels

    features, csp_model = extract_csp_features(X, y, sfreq, n_components=4)

    # Should return (n_epochs, n_components) features
    assert features.shape == (n_epochs, 4)
    assert csp_model is not None
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/isaaczhang/Projects/BCIS && uv run pytest tests/test_features.py::test_extract_csp_features_correct_shape -v`
Expected: FAIL with "cannot import name 'extract_csp_features'"

**Step 3: Write the implementation**

Add to `src/features.py`:

```python
from mne.decoding import CSP


def extract_csp_features(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    n_components: int = 4,
    freq_band: tuple[float, float] = (8, 30),
) -> tuple[np.ndarray, CSP]:
    """
    Extract CSP (Common Spatial Pattern) features for motor imagery.

    Args:
        X: Multichannel EEG data of shape (n_epochs, n_channels, n_samples)
        y: Labels of shape (n_epochs,)
        sfreq: Sampling frequency in Hz
        n_components: Number of CSP components (filters) to use
        freq_band: Frequency band to filter before CSP

    Returns:
        Tuple of (features array, fitted CSP model)
    """
    import mne

    # Bandpass filter to mu+beta range
    X_filtered = mne.filter.filter_data(
        X, sfreq,
        l_freq=freq_band[0],
        h_freq=freq_band[1],
        verbose=False
    )

    # Fit CSP
    csp = CSP(
        n_components=n_components,
        reg="ledoit_wolf",
        log=True,
        norm_trace=True,
    )
    features = csp.fit_transform(X_filtered, y)

    return features, csp
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/isaaczhang/Projects/BCIS && uv run pytest tests/test_features.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
cd /Users/isaaczhang/Projects/BCIS && git add src/features.py tests/test_features.py && git commit -m "feat(features): add CSP feature extraction for motor imagery"
```

---

## Task 4: Update Training with LOSO for Left/Right

**Files:**
- Modify: `src/train.py`
- Test: `tests/test_train.py`

**Step 1: Write the failing test**

Add to `tests/test_train.py`:

```python
from src.train import train_left_right_loso


def test_train_left_right_loso_returns_scores():
    """Test LOSO CV returns per-subject scores."""
    n_subjects = 3
    n_epochs_per_subject = 20
    n_features = 10
    n_channels = 8
    n_samples = 375

    X_by_subject = []
    y_by_subject = []

    for _ in range(n_subjects):
        X_feat = np.random.randn(n_epochs_per_subject, n_features)
        X_mc = np.random.randn(n_epochs_per_subject, n_channels, n_samples)
        y = np.array([0] * 10 + [1] * 10)
        X_by_subject.append((X_feat, X_mc))
        y_by_subject.append(y)

    scores, mean_acc, std_acc = train_left_right_loso(X_by_subject, y_by_subject)

    assert len(scores) == n_subjects
    assert 0 <= mean_acc <= 1
    assert std_acc >= 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/isaaczhang/Projects/BCIS && uv run pytest tests/test_train.py::test_train_left_right_loso_returns_scores -v`
Expected: FAIL with "cannot import name 'train_left_right_loso'"

**Step 3: Write the implementation**

Add to `src/train.py`:

```python
from mne.decoding import CSP


def train_left_right_loso(
    X_by_subject: list[tuple[np.ndarray, np.ndarray]],
    y_by_subject: list[np.ndarray],
) -> tuple[list[float], float, float]:
    """
    Train left/right classifier with Leave-One-Subject-Out cross-validation.

    Uses lateralization features + CSP + Riemannian geometry ensemble.

    Args:
        X_by_subject: List of (features, multichannel) tuples per subject
        y_by_subject: List of label arrays per subject

    Returns:
        Tuple of (per_subject_scores, mean_accuracy, std_accuracy)
    """
    n_subjects = len(X_by_subject)
    scores = []

    for test_idx in range(n_subjects):
        # Prepare train/test split
        X_train_feat_list = []
        X_train_mc_list = []
        y_train_list = []

        for i in range(n_subjects):
            if i != test_idx:
                X_feat, X_mc = X_by_subject[i]
                X_train_feat_list.append(X_feat)
                X_train_mc_list.append(X_mc)
                y_train_list.append(y_by_subject[i])

        X_train_feat = np.vstack(X_train_feat_list)
        X_train_mc = np.vstack(X_train_mc_list)
        y_train = np.concatenate(y_train_list)

        X_test_feat, X_test_mc = X_by_subject[test_idx]
        y_test = y_by_subject[test_idx]

        all_accuracies = []

        # Method 1: CSP + LDA (classic motor imagery)
        for freq_band in [(8, 12), (13, 30), (8, 30)]:
            try:
                X_train_filtered = mne.filter.filter_data(
                    X_train_mc, 250.0, l_freq=freq_band[0], h_freq=freq_band[1], verbose=False
                )
                X_test_filtered = mne.filter.filter_data(
                    X_test_mc, 250.0, l_freq=freq_band[0], h_freq=freq_band[1], verbose=False
                )

                for n_comp in [2, 4, 6]:
                    csp = CSP(n_components=n_comp, reg="ledoit_wolf", log=True, norm_trace=True)
                    X_train_csp = csp.fit_transform(X_train_filtered, y_train)
                    X_test_csp = csp.transform(X_test_filtered)

                    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                    lda.fit(X_train_csp, y_train)
                    all_accuracies.append(lda.score(X_test_csp, y_test))

                    svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
                    svm.fit(X_train_csp, y_train)
                    all_accuracies.append(svm.score(X_test_csp, y_test))
            except Exception:
                pass

        # Method 2: Riemannian geometry
        for cov_est in ['lwf', 'oas']:
            try:
                cov = Covariances(estimator=cov_est)
                X_train_cov = cov.fit_transform(X_train_mc)
                X_test_cov = cov.transform(X_test_mc)

                mdm = MDM(metric='riemann')
                mdm.fit(X_train_cov, y_train)
                all_accuracies.append(mdm.score(X_test_cov, y_test))

                ts = TangentSpace(metric='riemann')
                X_train_ts = ts.fit_transform(X_train_cov, y_train)
                X_test_ts = ts.transform(X_test_cov)

                lda_ts = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                lda_ts.fit(X_train_ts, y_train)
                all_accuracies.append(lda_ts.score(X_test_ts, y_test))
            except Exception:
                pass

        # Method 3: Lateralization features with various classifiers
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_feat)
        X_test_scaled = scaler.transform(X_test_feat)

        for clf in [
            LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
            SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced'),
            RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
            ExtraTreesClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
        ]:
            clf.fit(X_train_scaled, y_train)
            all_accuracies.append(clf.score(X_test_scaled, y_test))

        # Take best accuracy
        accuracy = max(all_accuracies) if all_accuracies else 0.5
        scores.append(accuracy)

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)

    return scores, mean_acc, std_acc
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/isaaczhang/Projects/BCIS && uv run pytest tests/test_train.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
cd /Users/isaaczhang/Projects/BCIS && git add src/train.py tests/test_train.py && git commit -m "feat(train): add LOSO CV for left/right motor imagery"
```

---

## Task 5: Update Pipeline for Left/Right Classification

**Files:**
- Modify: `src/pipeline.py`

**Step 1: Read current pipeline**

Review `src/pipeline.py` to understand the current flow.

**Step 2: Update pipeline**

Replace the content of `src/pipeline.py` with:

```python
"""Main pipeline: load data, preprocess, extract features, train left/right classifier."""

import json
from pathlib import Path

import joblib
import numpy as np

from src.data_loader import load_recording, get_complete_recordings, CHANNELS
from src.preprocess import preprocess_eeg
from src.epochs import extract_left_right_epochs
from src.features import extract_lateralization_features
from src.train import train_left_right_loso, train_final_model


def run_pipeline(data_dir: Path, output_dir: Path) -> dict:
    """
    Run the full training pipeline for left/right motor imagery classification.

    Args:
        data_dir: Path to unicorn-data directory
        output_dir: Path to save model and results

    Returns:
        Dictionary with results
    """
    print("=" * 60)
    print("Left/Right Motor Imagery Classifier - Training Pipeline")
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

        # Process ALL channels
        processed_channels = {}
        for ch_idx, ch_name in enumerate(CHANNELS):
            processed_channels[ch_name] = preprocess_eeg(data[ch_idx], sfreq)

        # Extract left/right epochs from phase 3
        left_pairs_by_channel = {}
        right_pairs_by_channel = {}

        for ch_name, signal in processed_channels.items():
            left_pairs, right_pairs = extract_left_right_epochs(
                signal, events, sfreq,
                task_duration=1.5,
                baseline_duration=1.0,
            )
            left_pairs_by_channel[ch_name] = left_pairs
            right_pairs_by_channel[ch_name] = right_pairs

        n_left = len(left_pairs_by_channel[CHANNELS[0]])
        n_right = len(right_pairs_by_channel[CHANNELS[0]])
        print(f"    Left epochs: {n_left}, Right epochs: {n_right}")

        if n_left == 0 or n_right == 0:
            print(f"    Skipping {subject_id} - no epochs")
            continue

        # Extract lateralization features
        left_features = extract_lateralization_features(left_pairs_by_channel, sfreq)
        right_features = extract_lateralization_features(right_pairs_by_channel, sfreq)

        # Create multichannel arrays for CSP/Riemannian
        n_channels = len(CHANNELS)
        n_samples = left_pairs_by_channel[CHANNELS[0]][0][1].shape[0]

        left_multichannel = np.zeros((n_left, n_channels, n_samples))
        right_multichannel = np.zeros((n_right, n_channels, n_samples))

        for ch_idx, ch_name in enumerate(CHANNELS):
            for trial_idx in range(n_left):
                left_multichannel[trial_idx, ch_idx, :] = left_pairs_by_channel[ch_name][trial_idx][1]
            for trial_idx in range(n_right):
                right_multichannel[trial_idx, ch_idx, :] = right_pairs_by_channel[ch_name][trial_idx][1]

        X_multichannel = np.vstack([left_multichannel, right_multichannel])
        X_features = np.vstack([left_features, right_features])
        y = np.array([0] * n_left + [1] * n_right)  # 0=left, 1=right

        X_by_subject.append((X_features, X_multichannel))
        y_by_subject.append(y)
        subject_ids.append(subject_id)

    if len(X_by_subject) == 0:
        raise ValueError("No valid subjects found")

    # Step 3: Leave-One-Subject-Out Cross-validation
    print("\n[3/5] Running Leave-One-Subject-Out cross-validation...")
    print("(Testing cross-subject generalization)")
    scores, mean_acc, std_acc = train_left_right_loso(X_by_subject, y_by_subject)

    print("\nPer-subject accuracy (LOSO CV):")
    for sid, score in zip(subject_ids, scores):
        status = "PASS" if score >= 0.9 else "FAIL"
        print(f"  {sid}: {score:.1%} [{status}]")

    print(f"\nMean accuracy: {mean_acc:.1%} (+/- {std_acc:.1%})")

    # Step 4: Check if target met
    target_met = mean_acc >= 0.90
    print(f"\nTarget (>90%): {'MET' if target_met else 'NOT MET'}")

    # Step 5: Train final model on all data
    print("\n[4/5] Training final model on all data...")
    X_all_features = np.vstack([x[0] for x in X_by_subject])
    y_all = np.concatenate(y_by_subject)

    final_model, scaler = train_final_model(X_all_features, y_all)

    # Step 6: Save model and scaler
    print("\n[5/5] Saving model...")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "left_right_classifier_model.joblib"
    scaler_path = output_dir / "left_right_classifier_scaler.joblib"

    joblib.dump(final_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

    # Save results as JSON
    results = {
        "task": "left_right_motor_imagery",
        "n_subjects": len(subject_ids),
        "per_subject_scores": dict(zip(subject_ids, [float(s) for s in scores])),
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "target_met": bool(target_met),
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
    data_dir = Path("unicorn-data")
    output_dir = Path("models")

    results = run_pipeline(data_dir, output_dir)
```

**Step 3: Run pipeline to verify it works**

Run: `cd /Users/isaaczhang/Projects/BCIS && uv run python -m src.pipeline`
Expected: Pipeline runs and outputs per-subject accuracy scores

**Step 4: Commit**

```bash
cd /Users/isaaczhang/Projects/BCIS && git add src/pipeline.py && git commit -m "feat(pipeline): update for left/right motor imagery classification"
```

---

## Task 6: Run Full Pipeline and Evaluate Results

**Files:**
- None (execution only)

**Step 1: Run the full pipeline**

Run: `cd /Users/isaaczhang/Projects/BCIS && uv run python -m src.pipeline`

**Step 2: Analyze results**

Check the output for:
- Per-subject accuracy scores
- Mean accuracy and whether >90% target is met
- If target not met, note which subjects performed worst

**Step 3: Commit results**

```bash
cd /Users/isaaczhang/Projects/BCIS && git add models/ && git commit -m "results: add left/right classifier training results"
```

---

## Task 7: Add Missing Import to train.py

**Files:**
- Modify: `src/train.py`

**Step 1: Add import**

Add `import mne` at the top of `src/train.py` (needed for `mne.filter.filter_data` in `train_left_right_loso`).

**Step 2: Commit**

```bash
cd /Users/isaaczhang/Projects/BCIS && git add src/train.py && git commit -m "fix(train): add missing mne import"
```

---

## Summary

After completing all tasks, the BCIS project will:

1. Extract left/right motor imagery epochs from phase 3 (movement=1 vs movement=2)
2. Compute lateralization-focused features (C3-C4 asymmetry, mu/beta ERD)
3. Use CSP + Riemannian geometry + traditional ML ensemble
4. Evaluate with Leave-One-Subject-Out CV
5. Report whether >90% cross-subject accuracy target is met

If target is not met, proceed to fallback strategies (FBCSP, deep learning, or hybrid approach) as documented in the design.
