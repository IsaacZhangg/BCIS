# Cross-Subject Riemannian Transfer Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cross-subject Riemannian transfer learning with Euclidean Alignment and LOSO CV evaluation, loading data from both `Data/unicorn-data/` and `Data/MI_DATA_NEW/`.

**Architecture:** New `src/transfer.py` module handles multi-directory data loading, Euclidean Alignment via pyriemann's `TLCenter`, and leave-one-subject-out CV. Integrates into `pipeline.py` as an optional final step. Existing within-subject pipeline is untouched.

**Tech Stack:** pyriemann 0.10 (`Covariances`, `TLCenter`, `TangentSpace`), scikit-learn (`LogisticRegression`), numpy, existing `src/data_loader.py` and `src/preprocess.py`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/transfer.py` | Create | EA alignment, LOSO CV, multi-dir loading, comparison output |
| `tests/test_transfer.py` | Create | Tests for all transfer.py functions |
| `src/pipeline.py` | Modify (line ~456, after cross-session eval) | Add optional call to transfer evaluation |

---

### Task 1: Multi-Directory Data Loading

**Files:**
- Create: `src/transfer.py`
- Create: `tests/test_transfer.py`

- [ ] **Step 1: Write the failing test for `load_all_subjects`**

```python
"""Tests for cross-subject Riemannian transfer learning."""

import numpy as np
import pytest

from src.transfer import load_all_subjects


def test_load_all_subjects_returns_dict(tmp_path):
    """load_all_subjects returns a dict mapping subject_id to (X_mc, y) tuples."""
    # Create minimal fake data: 2 subjects, each with 10 trials (5L/5R), 8 channels, 1250 samples
    sfreq = 250.0
    n_samples_per_trial = int(sfreq * 5)  # 5s per trial (baseline+task+margins)
    n_trials = 10

    for subj_name in ["subject0001", "subject0002"]:
        subj_dir = tmp_path / subj_name / "session001"
        subj_dir.mkdir(parents=True)

        # Build a CSV with stim markers for 10 phase-3 trials
        n_total = n_samples_per_trial * n_trials + 5000  # padding
        rng = np.random.default_rng(42)
        data = rng.standard_normal((n_total, 8)) * 50  # ~50uV scale
        stim = np.zeros(n_total)

        # Place 10 phase-3 events (5 left=31, 5 right=32) spaced apart
        for i in range(n_trials):
            event_idx = 1000 + i * n_samples_per_trial
            stim[event_idx] = 31 if i < 5 else 32

        import pandas as pd
        df = pd.DataFrame(data, columns=["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"])
        df["stim"] = stim
        df["timestamps"] = np.arange(n_total) / sfreq
        df = df[["timestamps", "Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8", "stim"]]
        df.to_csv(subj_dir / "recording_test.csv", index=False)

    result = load_all_subjects([tmp_path], sfreq=sfreq)

    assert isinstance(result, dict)
    assert len(result) == 2
    for sid, (X_mc, y) in result.items():
        assert X_mc.ndim == 3  # (n_trials, n_channels, n_samples)
        assert X_mc.shape[1] == 8
        assert len(y) == X_mc.shape[0]
        assert set(np.unique(y)) == {0, 1}


def test_load_all_subjects_merges_sessions(tmp_path):
    """Sessions for the same subject are pooled into one entry."""
    sfreq = 250.0
    n_samples_per_trial = int(sfreq * 5)
    n_trials = 10

    subj_dir_s1 = tmp_path / "subject0001" / "session001"
    subj_dir_s2 = tmp_path / "subject0001" / "session002"
    subj_dir_s1.mkdir(parents=True)
    subj_dir_s2.mkdir(parents=True)

    rng = np.random.default_rng(42)
    for sess_dir in [subj_dir_s1, subj_dir_s2]:
        n_total = n_samples_per_trial * n_trials + 5000
        data = rng.standard_normal((n_total, 8)) * 50
        stim = np.zeros(n_total)
        for i in range(n_trials):
            event_idx = 1000 + i * n_samples_per_trial
            stim[event_idx] = 31 if i < 5 else 32

        import pandas as pd
        df = pd.DataFrame(data, columns=["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"])
        df["stim"] = stim
        df["timestamps"] = np.arange(n_total) / sfreq
        df = df[["timestamps", "Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8", "stim"]]
        df.to_csv(sess_dir / "recording_test.csv", index=False)

    result = load_all_subjects([tmp_path], sfreq=sfreq)

    # Should merge both sessions into one subject entry
    assert "subject0001" in result
    X_mc, y = result["subject0001"]
    # Two sessions of 10 trials each (minus any rejected)
    assert X_mc.shape[0] == len(y)
    assert X_mc.shape[0] >= 15  # At least 15 out of 20 survive rejection
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transfer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.transfer'`

- [ ] **Step 3: Implement `load_all_subjects`**

Create `src/transfer.py`:

```python
"""Cross-subject Riemannian transfer learning with Euclidean Alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data_loader import CHANNELS, get_recordings_by_subject, load_recording
from src.epochs import (
    compute_rejection_threshold,
    extract_left_right_epochs,
    reject_bad_epochs,
)
from src.preprocess import preprocess_multichannel_eeg


def _task_epochs(pairs_by_ch: dict) -> np.ndarray:
    """Convert per-channel epoch pairs to (n_trials, n_channels, n_samples)."""
    return np.array(
        [[trial[1] for trial in pairs_by_ch[ch]] for ch in CHANNELS]
    ).transpose(1, 0, 2)


def load_all_subjects(
    data_dirs: list[Path],
    sfreq: float = 250.0,
    subject_merge: dict[str, str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess data from multiple directories, merging by subject.

    Args:
        data_dirs: List of directories to scan for recordings.
        sfreq: Sampling frequency.
        subject_merge: Optional mapping of subject_id -> canonical_id for merging
            (e.g., {"subject0100_2": "subject0100"}).

    Returns:
        Dict mapping subject_id to (X_multichannel, y) where
        X_multichannel is (n_trials, 8, n_samples) and y is (n_trials,).
    """
    if subject_merge is None:
        subject_merge = {}

    # Collect all recordings grouped by canonical subject ID
    all_recordings: dict[str, list[Path]] = {}
    for data_dir in data_dirs:
        for sid, paths in get_recordings_by_subject(data_dir).items():
            canonical = subject_merge.get(sid, sid)
            all_recordings.setdefault(canonical, []).extend(paths)

    subjects: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for sid, rec_paths in sorted(all_recordings.items()):
        all_left: dict[str, list] = {ch: [] for ch in CHANNELS}
        all_right: dict[str, list] = {ch: [] for ch in CHANNELS}

        for rec_path in rec_paths:
            data, events, rec_sfreq = load_recording(rec_path)
            if abs(rec_sfreq - sfreq) > 1e-6:
                continue
            preprocessed = preprocess_multichannel_eeg(data, sfreq)

            left_pairs: dict[str, list] = {}
            right_pairs: dict[str, list] = {}
            for ch_idx, ch_name in enumerate(CHANNELS):
                signal = preprocessed[ch_idx]
                left, right = extract_left_right_epochs(
                    signal, events, sfreq,
                    task_duration=3.0, baseline_duration=1.0, skip_duration=0.25,
                )
                left_pairs[ch_name] = left
                right_pairs[ch_name] = right

            # Artifact rejection per recording
            threshold = compute_rejection_threshold(left_pairs, right_pairs)
            left_pairs, right_pairs, _, _ = reject_bad_epochs(
                left_pairs, right_pairs, threshold_uv=threshold,
            )

            for ch in CHANNELS:
                all_left[ch].extend(left_pairs[ch])
                all_right[ch].extend(right_pairs[ch])

        n_left = len(all_left[CHANNELS[0]])
        n_right = len(all_right[CHANNELS[0]])
        if n_left == 0 or n_right == 0:
            continue

        X_mc = np.vstack([_task_epochs(all_left), _task_epochs(all_right)])
        y = np.array([0] * n_left + [1] * n_right)
        subjects[sid] = (X_mc, y)

    return subjects
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_transfer.py::test_load_all_subjects_returns_dict tests/test_transfer.py::test_load_all_subjects_merges_sessions -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/transfer.py tests/test_transfer.py
git commit -m "Add multi-directory data loading for cross-subject transfer"
```

---

### Task 2: Euclidean Alignment

**Files:**
- Modify: `src/transfer.py`
- Modify: `tests/test_transfer.py`

- [ ] **Step 1: Write the failing test for `align_subjects`**

Add to `tests/test_transfer.py`:

```python
from src.transfer import align_subjects


def test_align_subjects_centers_covariances():
    """After EA, each subject's mean covariance should be close to identity."""
    rng = np.random.default_rng(42)
    n_channels = 8

    # Create 3 subjects with different covariance structures
    subjects = {}
    for i in range(3):
        # Random SPD matrix as subject-specific "mean"
        A = rng.standard_normal((n_channels, n_channels))
        cov_mean = A @ A.T + np.eye(n_channels)
        # Generate 20 trials from this subject's distribution
        X = rng.standard_normal((20, n_channels, 375))
        X = np.einsum("ij,njt->nit", np.linalg.cholesky(cov_mean), X)
        y = np.array([0] * 10 + [1] * 10)
        subjects[f"subj{i:02d}"] = (X, y)

    aligned_covs, aligned_labels, subject_ids = align_subjects(subjects, sfreq=250.0)

    assert aligned_covs.ndim == 3  # (total_trials, 8, 8)
    assert aligned_covs.shape[0] == 60  # 3 subjects * 20 trials
    assert aligned_covs.shape[1] == n_channels
    assert aligned_covs.shape[2] == n_channels
    assert len(aligned_labels) == 60
    assert len(subject_ids) == 60

    # Check each subject's aligned covariances are centered near identity
    for i in range(3):
        mask = np.array(subject_ids) == f"subj{i:02d}"
        subj_covs = aligned_covs[mask]
        mean_cov = np.mean(subj_covs, axis=0)
        # After EA, mean should be close to identity (not exact due to finite samples)
        np.testing.assert_allclose(mean_cov, np.eye(n_channels), atol=0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transfer.py::test_align_subjects_centers_covariances -v`
Expected: FAIL — `ImportError: cannot import name 'align_subjects'`

- [ ] **Step 3: Implement `align_subjects`**

Add to `src/transfer.py`:

```python
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from scipy.linalg import fractional_matrix_power


def align_subjects(
    subjects: dict[str, tuple[np.ndarray, np.ndarray]],
    sfreq: float = 250.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute covariance matrices and apply Euclidean Alignment across subjects.

    EA re-centers each subject's covariance distribution to the identity matrix,
    removing inter-subject variability from electrode impedance and placement.

    Args:
        subjects: Dict mapping subject_id to (X_multichannel, y).
        sfreq: Sampling frequency (unused, kept for API consistency).

    Returns:
        Tuple of:
        - aligned_covs: (total_trials, n_channels, n_channels) aligned SPD matrices
        - labels: (total_trials,) class labels
        - subject_ids: list of subject_id per trial (for LOSO indexing)
    """
    cov_estimator = Covariances(estimator="lwf")

    all_covs = []
    all_labels = []
    all_subject_ids: list[str] = []

    for sid in sorted(subjects.keys()):
        X_mc, y = subjects[sid]
        # Estimate covariance matrices: (n_trials, n_channels, n_channels)
        covs = cov_estimator.fit_transform(X_mc)

        # Euclidean Alignment: center subject's covariances to identity
        # 1. Compute Riemannian mean of this subject's covariances
        ref = mean_covariance(covs, metric="riemann")
        # 2. Compute ref^{-1/2}
        ref_inv_sqrt = fractional_matrix_power(ref, -0.5).real
        # 3. Re-center: C_aligned = ref^{-1/2} @ C @ ref^{-1/2}
        covs_aligned = ref_inv_sqrt @ covs @ ref_inv_sqrt.T

        all_covs.append(covs_aligned)
        all_labels.append(y)
        all_subject_ids.extend([sid] * len(y))

    return np.vstack(all_covs), np.concatenate(all_labels), all_subject_ids
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_transfer.py::test_align_subjects_centers_covariances -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/transfer.py tests/test_transfer.py
git commit -m "Add Euclidean Alignment for cross-subject covariance normalization"
```

---

### Task 3: Leave-One-Subject-Out CV

**Files:**
- Modify: `src/transfer.py`
- Modify: `tests/test_transfer.py`

- [ ] **Step 1: Write the failing test for `loso_cv`**

Add to `tests/test_transfer.py`:

```python
from src.transfer import loso_cv


def test_loso_cv_returns_per_subject_scores():
    """LOSO CV returns a score for each subject."""
    rng = np.random.default_rng(42)
    n_channels = 8

    subjects = {}
    for i in range(4):
        X = rng.standard_normal((20, n_channels, 375))
        y = np.array([0] * 10 + [1] * 10)
        subjects[f"subj{i:02d}"] = (X, y)

    scores = loso_cv(subjects, sfreq=250.0)

    assert isinstance(scores, dict)
    assert len(scores) == 4
    for sid, score in scores.items():
        assert 0.0 <= score <= 1.0
        assert sid.startswith("subj")


def test_loso_cv_on_random_data_near_chance():
    """On random data, LOSO accuracy should be near chance (~50%)."""
    rng = np.random.default_rng(99)
    n_channels = 8

    subjects = {}
    for i in range(5):
        X = rng.standard_normal((30, n_channels, 375))
        y = np.array([0] * 15 + [1] * 15)
        subjects[f"subj{i:02d}"] = (X, y)

    scores = loso_cv(subjects, sfreq=250.0)

    mean_acc = np.mean(list(scores.values()))
    # Random data should give ~50% +/- 20%
    assert 0.3 <= mean_acc <= 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transfer.py::test_loso_cv_returns_per_subject_scores tests/test_transfer.py::test_loso_cv_on_random_data_near_chance -v`
Expected: FAIL — `ImportError: cannot import name 'loso_cv'`

- [ ] **Step 3: Implement `loso_cv`**

Add to `src/transfer.py`:

```python
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression


def loso_cv(
    subjects: dict[str, tuple[np.ndarray, np.ndarray]],
    sfreq: float = 250.0,
) -> dict[str, float]:
    """Leave-one-subject-out CV with Euclidean Alignment.

    For each held-out subject:
    1. Align all subjects independently (EA per subject)
    2. Train TangentSpace + LR on aligned data from all other subjects
    3. Test on held-out subject's aligned data

    Args:
        subjects: Dict mapping subject_id to (X_multichannel, y).
        sfreq: Sampling frequency.

    Returns:
        Dict mapping subject_id to LOSO accuracy.
    """
    cov_estimator = Covariances(estimator="lwf")

    # Pre-compute aligned covariances per subject
    aligned_per_subject: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sid in sorted(subjects.keys()):
        X_mc, y = subjects[sid]
        covs = cov_estimator.fit_transform(X_mc)
        ref = mean_covariance(covs, metric="riemann")
        ref_inv_sqrt = fractional_matrix_power(ref, -0.5).real
        covs_aligned = ref_inv_sqrt @ covs @ ref_inv_sqrt.T
        aligned_per_subject[sid] = (covs_aligned, y)

    subject_ids = sorted(subjects.keys())
    scores: dict[str, float] = {}

    for held_out in subject_ids:
        # Train on all subjects except held_out
        train_covs = []
        train_labels = []
        for sid in subject_ids:
            if sid == held_out:
                continue
            covs, y = aligned_per_subject[sid]
            train_covs.append(covs)
            train_labels.append(y)

        X_train = np.vstack(train_covs)
        y_train = np.concatenate(train_labels)

        X_test, y_test = aligned_per_subject[held_out]

        # Classify in tangent space
        pipe_ts = TangentSpace(metric="riemann")
        X_train_ts = pipe_ts.fit_transform(X_train)
        X_test_ts = pipe_ts.transform(X_test)

        clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000)
        clf.fit(X_train_ts, y_train)
        scores[held_out] = float(clf.score(X_test_ts, y_test))

    return scores
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_transfer.py -v -k "loso"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/transfer.py tests/test_transfer.py
git commit -m "Add leave-one-subject-out CV with Euclidean Alignment"
```

---

### Task 4: LOSO + Fine-Tuning

**Files:**
- Modify: `src/transfer.py`
- Modify: `tests/test_transfer.py`

- [ ] **Step 1: Write the failing test for `loso_cv` with `fine_tune=True`**

Add to `tests/test_transfer.py`:

```python
def test_loso_cv_fine_tune_returns_scores():
    """LOSO CV with fine-tuning returns a score for each subject."""
    rng = np.random.default_rng(42)
    n_channels = 8

    subjects = {}
    for i in range(4):
        X = rng.standard_normal((20, n_channels, 375))
        y = np.array([0] * 10 + [1] * 10)
        subjects[f"subj{i:02d}"] = (X, y)

    scores = loso_cv(subjects, sfreq=250.0, fine_tune=True)

    assert isinstance(scores, dict)
    assert len(scores) == 4
    for sid, score in scores.items():
        assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transfer.py::test_loso_cv_fine_tune_returns_scores -v`
Expected: FAIL — `TypeError: loso_cv() got an unexpected keyword argument 'fine_tune'`

- [ ] **Step 3: Add `fine_tune` parameter to `loso_cv`**

Update the `loso_cv` function signature and add fine-tuning logic. Replace the existing function in `src/transfer.py`:

```python
def loso_cv(
    subjects: dict[str, tuple[np.ndarray, np.ndarray]],
    sfreq: float = 250.0,
    fine_tune: bool = False,
) -> dict[str, float]:
    """Leave-one-subject-out CV with Euclidean Alignment.

    For each held-out subject:
    1. Align all subjects independently (EA per subject)
    2. Train TangentSpace + LR on aligned data from all other subjects
    3. Test on held-out subject's aligned data
    4. (If fine_tune) Re-center tangent space reference on held-out subject's mean

    Args:
        subjects: Dict mapping subject_id to (X_multichannel, y).
        sfreq: Sampling frequency.
        fine_tune: If True, re-center the tangent space reference point
            to the held-out subject's Riemannian mean before testing.

    Returns:
        Dict mapping subject_id to LOSO accuracy.
    """
    cov_estimator = Covariances(estimator="lwf")

    # Pre-compute aligned covariances per subject
    aligned_per_subject: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sid in sorted(subjects.keys()):
        X_mc, y = subjects[sid]
        covs = cov_estimator.fit_transform(X_mc)
        ref = mean_covariance(covs, metric="riemann")
        ref_inv_sqrt = fractional_matrix_power(ref, -0.5).real
        covs_aligned = ref_inv_sqrt @ covs @ ref_inv_sqrt.T
        aligned_per_subject[sid] = (covs_aligned, y)

    subject_ids = sorted(subjects.keys())
    scores: dict[str, float] = {}

    for held_out in subject_ids:
        # Train on all subjects except held_out
        train_covs = []
        train_labels = []
        for sid in subject_ids:
            if sid == held_out:
                continue
            covs, y = aligned_per_subject[sid]
            train_covs.append(covs)
            train_labels.append(y)

        X_train = np.vstack(train_covs)
        y_train = np.concatenate(train_labels)

        X_test, y_test = aligned_per_subject[held_out]

        if fine_tune:
            # Re-center tangent space on held-out subject's mean
            test_ref = mean_covariance(X_test, metric="riemann")
            pipe_ts = TangentSpace(metric="riemann")
            # Fit on training data
            pipe_ts.fit(X_train)
            # Override the reference point with the held-out subject's mean
            pipe_ts.reference_ = test_ref
            X_train_ts = TangentSpace(metric="riemann").fit_transform(X_train)
            X_test_ts = pipe_ts.transform(X_test)
        else:
            pipe_ts = TangentSpace(metric="riemann")
            X_train_ts = pipe_ts.fit_transform(X_train)
            X_test_ts = pipe_ts.transform(X_test)

        clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000)
        clf.fit(X_train_ts, y_train)
        scores[held_out] = float(clf.score(X_test_ts, y_test))

    return scores
```

- [ ] **Step 4: Run all transfer tests to verify they pass**

Run: `uv run pytest tests/test_transfer.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/transfer.py tests/test_transfer.py
git commit -m "Add fine-tuning option to LOSO CV"
```

---

### Task 5: Standalone Entry Point and Pipeline Integration

**Files:**
- Modify: `src/transfer.py`
- Modify: `src/pipeline.py:456` (after cross-session eval block)
- Modify: `tests/test_transfer.py`

- [ ] **Step 1: Write the failing test for `run_transfer_evaluation`**

Add to `tests/test_transfer.py`:

```python
from src.transfer import run_transfer_evaluation


def test_run_transfer_evaluation_returns_results(tmp_path):
    """run_transfer_evaluation returns a results dict with expected keys."""
    rng = np.random.default_rng(42)
    n_channels = 8

    # Create 3 fake subjects
    for subj_name in ["subject0001", "subject0002", "subject0003"]:
        subj_dir = tmp_path / subj_name / "session001"
        subj_dir.mkdir(parents=True)

        n_samples_per_trial = 1250
        n_trials = 10
        n_total = n_samples_per_trial * n_trials + 5000
        data = rng.standard_normal((n_total, 8)) * 50
        stim = np.zeros(n_total)
        for i in range(n_trials):
            event_idx = 1000 + i * n_samples_per_trial
            stim[event_idx] = 31 if i < 5 else 32

        import pandas as pd
        df = pd.DataFrame(data, columns=["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"])
        df["stim"] = stim
        df["timestamps"] = np.arange(n_total) / 250.0
        df = df[["timestamps", "Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8", "stim"]]
        df.to_csv(subj_dir / "recording_test.csv", index=False)

    results = run_transfer_evaluation(data_dirs=[tmp_path])

    assert "loso_scores" in results
    assert "loso_ft_scores" in results
    assert "loso_mean" in results
    assert "loso_ft_mean" in results
    assert len(results["loso_scores"]) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transfer.py::test_run_transfer_evaluation_returns_results -v`
Expected: FAIL — `ImportError: cannot import name 'run_transfer_evaluation'`

- [ ] **Step 3: Implement `run_transfer_evaluation` and `__main__` block**

Add to `src/transfer.py`:

```python
def run_transfer_evaluation(
    data_dirs: list[Path] | None = None,
    sfreq: float = 250.0,
    subject_merge: dict[str, str] | None = None,
    within_subject_scores: dict[str, float] | None = None,
) -> dict:
    """Run cross-subject transfer learning evaluation and print comparison.

    Args:
        data_dirs: Directories to scan. Defaults to Data/unicorn-data + Data/MI_DATA_NEW.
        sfreq: Sampling frequency.
        subject_merge: Subject ID merging map.
        within_subject_scores: Optional dict of subject_id -> nested CV accuracy
            for comparison table.

    Returns:
        Results dict with LOSO and LOSO+FT scores.
    """
    if data_dirs is None:
        data_dirs = [Path("Data/unicorn-data"), Path("Data/MI_DATA_NEW")]
    if subject_merge is None:
        subject_merge = {"subject0100_2": "subject0100"}

    print("=" * 68)
    print("Cross-Subject Transfer Learning (Euclidean Alignment + LOSO)")
    print("=" * 68)

    print("\nLoading subjects from all data directories...")
    subjects = load_all_subjects(data_dirs, sfreq=sfreq, subject_merge=subject_merge)
    print(f"Loaded {len(subjects)} subjects: {', '.join(sorted(subjects.keys()))}")
    for sid in sorted(subjects.keys()):
        X_mc, y = subjects[sid]
        n_left = int(np.sum(y == 0))
        n_right = int(np.sum(y == 1))
        print(f"  {sid}: {len(y)} trials ({n_left}L/{n_right}R)")

    print("\nRunning LOSO CV (no fine-tuning)...")
    loso_scores = loso_cv(subjects, sfreq=sfreq, fine_tune=False)

    print("Running LOSO CV (with fine-tuning)...")
    loso_ft_scores = loso_cv(subjects, sfreq=sfreq, fine_tune=True)

    # Print comparison table
    loso_mean = float(np.mean(list(loso_scores.values())))
    loso_ft_mean = float(np.mean(list(loso_ft_scores.values())))

    header = f"\n{'Subject':<16}"
    if within_subject_scores:
        header += f"{'Within-Subj':>12}"
    header += f"{'LOSO':>10}{'LOSO+FT':>10}"
    print(header)
    print("-" * len(header))

    for sid in sorted(loso_scores.keys()):
        row = f"  {sid:<14}"
        if within_subject_scores and sid in within_subject_scores:
            row += f"{within_subject_scores[sid]:>11.1%}"
        elif within_subject_scores:
            row += f"{'n/a':>12}"
        row += f"{loso_scores[sid]:>9.1%}{loso_ft_scores[sid]:>9.1%}"
        print(row)

    print(f"\n  LOSO mean:       {loso_mean:.1%}")
    print(f"  LOSO+FT mean:    {loso_ft_mean:.1%}")
    if within_subject_scores:
        ws_mean = float(np.mean(list(within_subject_scores.values())))
        print(f"  Within-subj mean: {ws_mean:.1%}")

    return {
        "loso_scores": loso_scores,
        "loso_ft_scores": loso_ft_scores,
        "loso_mean": loso_mean,
        "loso_ft_mean": loso_ft_mean,
        "n_subjects": len(subjects),
    }


if __name__ == "__main__":
    results = run_transfer_evaluation()
```

- [ ] **Step 4: Run all transfer tests**

Run: `uv run pytest tests/test_transfer.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/transfer.py tests/test_transfer.py
git commit -m "Add run_transfer_evaluation entry point with comparison table"
```

- [ ] **Step 6: Integrate into pipeline.py**

Add to `src/pipeline.py` after the cross-session evaluation block (after line ~487, before step 5). Add the import at top of file alongside existing imports:

Add import:
```python
from src.transfer import run_transfer_evaluation
```

Add after the cross-session eval block (after `runtime_seconds["cross_session_eval"] = ...`):

```python
    # Step 4b: Cross-subject transfer evaluation
    step_start = time.perf_counter()
    print("\n[4b/5] Cross-subject transfer evaluation (Euclidean Alignment + LOSO)")
    try:
        transfer_data_dirs = [data_dir]
        mi_new_dir = Path("Data/MI_DATA_NEW")
        if mi_new_dir.exists():
            transfer_data_dirs.append(mi_new_dir)
        within_scores = {
            sid: float(s) for sid, s in zip(subject_ids, nested_scores)
        }
        transfer_results = run_transfer_evaluation(
            data_dirs=transfer_data_dirs,
            sfreq=sfreq,
            within_subject_scores=within_scores,
        )
        results["transfer_results"] = transfer_results
    except Exception as e:
        print(f"  Transfer evaluation failed: {e}")
    runtime_seconds["transfer_eval"] = time.perf_counter() - step_start
```

- [ ] **Step 7: Run existing pipeline tests to confirm no regressions**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add src/transfer.py src/pipeline.py tests/test_transfer.py
git commit -m "Integrate cross-subject transfer evaluation into pipeline"
```

---

### Task 6: Lint, Format, and Final Verification

**Files:**
- All changed files

- [ ] **Step 1: Lint and format**

```bash
ruff check --fix src/transfer.py tests/test_transfer.py
ruff format src/transfer.py tests/test_transfer.py
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS (including new tests)

- [ ] **Step 3: Run transfer evaluation standalone**

Run: `uv run python -m src.transfer`
Expected: Prints comparison table with LOSO scores for all subjects from both data directories.

- [ ] **Step 4: Commit any formatting changes**

```bash
git add -A
git commit -m "Lint and format transfer learning module"
```
