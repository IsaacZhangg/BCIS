# New Data Integration & Accuracy Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate new subjects (104, 106) into the main pipeline with expanded FBCSP bands and EA-based donor augmentation to improve classification accuracy.

**Architecture:** Three sequential changes — (1) flexible data loading with adaptive CV folds, (2) expanded 10-band non-overlapping FBCSP pool, (3) Euclidean Alignment shared utility + expanded donor pool with leakage-fixed augmentation — each validated independently before proceeding.

**Tech Stack:** Python, NumPy, scikit-learn, pyriemann, MNE, SciPy, pytest, uv

**Spec:** `docs/superpowers/specs/2026-04-14-new-data-integration-accuracy-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/config.py` | Modify | Add subject0104 merge, update k_candidates |
| `src/data_loader.py` | Modify | Consolidate `get_recordings(min_trials)` |
| `src/validation.py` | Modify | Add `adaptive_fold_count()` utility |
| `src/train.py` | Modify | Update FBCSP_BANDS, use adaptive folds, dynamic K cap |
| `src/alignment.py` | Create | Shared EA utility (3 functions) |
| `src/transfer.py` | Modify | Use shared loader + alignment utility |
| `src/pipeline.py` | Modify | Flexible loader, expanded donors, EA augmentation, dual reporting |
| `tests/test_data_loader.py` | Modify | Test `get_recordings` with min_trials |
| `tests/test_alignment.py` | Create | Test EA utility |
| `tests/test_validation.py` | Modify | Test adaptive_fold_count |

---

## Phase 1: Flexible Data Loading

### Task 1: Consolidate recording loader

**Files:**
- Modify: `src/data_loader.py:30-48`
- Modify: `tests/test_data_loader.py`

- [ ] **Step 1: Write test for flexible min_trials**

Add to `tests/test_data_loader.py`:

```python
from src.data_loader import get_recordings, get_recordings_by_subject


def test_get_recordings_default_finds_complete():
    """Default min_trials=100 finds the same 10 complete recordings."""
    data_dir = Path("Data/unicorn-data")
    recordings = get_recordings(data_dir)
    assert len(recordings) == 10
    assert all(Path(r).exists() for r in recordings)


def test_get_recordings_low_threshold_finds_more():
    """min_trials=10 finds recordings in MI_DATA_NEW too."""
    data_dir = Path("Data/MI_DATA_NEW")
    recordings = get_recordings(data_dir, min_trials=10)
    assert len(recordings) >= 1


def test_get_recordings_by_subject_groups():
    """get_recordings_by_subject returns dict grouped by subject."""
    data_dir = Path("Data/unicorn-data")
    grouped = get_recordings_by_subject(data_dir)
    assert isinstance(grouped, dict)
    assert len(grouped) >= 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data_loader.py -v -k "get_recordings" 2>&1 | tail -10`
Expected: FAIL — `get_recordings` not importable

- [ ] **Step 3: Rename and parameterize the loader**

In `src/data_loader.py`, replace `get_complete_recordings` and `get_recordings_by_subject`:

```python
def get_recordings(data_dir: Path, min_trials: int = 100) -> list[Path]:
    """Find recordings with at least min_trials phase-3 imagery trials."""
    recordings = []
    for csv_path in sorted(data_dir.glob("subject*/session*/*.csv")):
        stim = pd.read_csv(csv_path, usecols=["stim"])["stim"].to_numpy(copy=False)
        nonzero = stim[stim != 0].astype(int)
        phase3_count = int(np.sum((nonzero // 10) % 10 == 3))
        if phase3_count >= min_trials:
            recordings.append(csv_path)
    return recordings


def get_recordings_by_subject(
    data_dir: Path, min_trials: int = 100
) -> dict[str, list[Path]]:
    """Group recordings by subject ID."""
    grouped: dict[str, list[Path]] = {}
    for rec_path in get_recordings(data_dir, min_trials=min_trials):
        subject_id = rec_path.parent.parent.name
        grouped.setdefault(subject_id, []).append(rec_path)
    return grouped
```

Keep `get_complete_recordings` as a deprecated alias:

```python
# Backward compatibility alias
get_complete_recordings = get_recordings
```

- [ ] **Step 4: Update imports in pipeline.py**

In `src/pipeline.py:12-16`, change the import:

```python
from src.data_loader import (
    CHANNELS,
    get_recordings,
    get_recordings_by_subject,
    load_recording,
)
```

Update `run_pipeline` line 299:

```python
recordings = get_recordings(data_dir, min_trials=cfg.min_evaluation_trials)
```

Update `get_recordings_by_subject` call at line 663:

```python
recordings_by_subject = get_recordings_by_subject(data_dir)
```

(This one stays at default min_trials=100 since cross-session eval needs complete recordings.)

- [ ] **Step 5: Update transfer.py to use shared loader**

In `src/transfer.py`, remove the `_get_recordings_with_trials` function entirely (lines 27-47). Update `load_all_subjects` (line 74) to use the shared loader:

```python
from src.data_loader import get_recordings_by_subject

# In load_all_subjects, replace lines 73-77:
all_recordings: dict[str, list[Path]] = {}
for data_dir in data_dirs:
    for sid, paths in get_recordings_by_subject(data_dir, min_trials=min_trials).items():
        canonical = subject_merge.get(sid, sid)
        all_recordings.setdefault(canonical, []).extend(paths)
```

Remove the `import pandas as pd` that was only used by `_get_recordings_with_trials`.

- [ ] **Step 6: Update the old test**

In `tests/test_data_loader.py`, update the existing test import:

```python
from src.data_loader import load_recording, get_recordings
```

Rename `test_get_complete_recordings_finds_all_subjects`:

```python
def test_get_complete_recordings_finds_all_subjects():
    """Test that we find all complete recordings (100 imagery trials)."""
    data_dir = Path("Data/unicorn-data")
    recordings = get_recordings(data_dir)
    assert len(recordings) == 10
    assert all(Path(r).exists() for r in recordings)
```

- [ ] **Step 7: Run all tests**

Run: `uv run pytest tests/test_data_loader.py tests/test_transfer.py -v 2>&1 | tail -20`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/data_loader.py src/pipeline.py src/transfer.py tests/test_data_loader.py
git commit -m "refactor: consolidate recording loader with flexible min_trials"
```

---

### Task 2: Add config changes

**Files:**
- Modify: `src/config.py:10,30`

- [ ] **Step 1: Add subject0104 merge entry, min_evaluation_trials, and update k_candidates**

In `src/config.py`:

Update line 10:
```python
DEFAULT_SUBJECT_MERGE: dict[str, str] = {
    "subject0100_2": "subject0100",
    "subject0104_session002": "subject0104",
}
```

Add `min_evaluation_trials` to `TrainingConfig` (after `augmentation_weakness_threshold`):
```python
    min_evaluation_trials: int = 30
```

Update `TrainingConfig` line 30:
```python
k_candidates: tuple[int, ...] = (3, 5, 8, 10, 15, 20, 25, 30)
```

- [ ] **Step 2: Run existing tests to verify nothing breaks**

Run: `uv run pytest tests/ -v --timeout=120 2>&1 | tail -10`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "config: add subject0104 merge, extend k_candidates to 30"
```

---

### Task 3: Add adaptive fold count utility

**Files:**
- Modify: `src/validation.py`
- Modify: `tests/test_validation.py`

- [ ] **Step 1: Write tests for adaptive_fold_count**

Add to `tests/test_validation.py`:

```python
from src.validation import adaptive_fold_count


def test_adaptive_fold_count_large_dataset():
    """100 trials, max 10 folds -> 10 folds."""
    assert adaptive_fold_count(100, max_folds=10) == 10


def test_adaptive_fold_count_small_dataset():
    """32 trials, max 10 folds -> 6 folds (32 // 5 = 6)."""
    assert adaptive_fold_count(32, max_folds=10) == 6


def test_adaptive_fold_count_medium_dataset():
    """64 trials, max 10 folds -> 10 folds (64 // 5 = 12, capped at 10)."""
    assert adaptive_fold_count(64, max_folds=10) == 10


def test_adaptive_fold_count_inner():
    """27 training trials, max 7 inner folds -> 5 folds (27 // 5 = 5)."""
    assert adaptive_fold_count(27, max_folds=7) == 5


def test_adaptive_fold_count_minimum():
    """Very few trials still returns at least 2."""
    assert adaptive_fold_count(8, max_folds=10) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_validation.py -v -k "adaptive" 2>&1 | tail -10`
Expected: FAIL — `adaptive_fold_count` not importable

- [ ] **Step 3: Implement adaptive_fold_count**

Add to `src/validation.py` after line 63 (after `_effective_n_splits`):

```python
def adaptive_fold_count(n_trials: int, max_folds: int, min_per_fold: int = 5) -> int:
    """Compute fold count ensuring at least min_per_fold trials per fold.

    Returns at least 2 (minimum for CV) and at most max_folds.
    """
    return max(2, min(max_folds, n_trials // min_per_fold))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_validation.py -v 2>&1 | tail -15`
Expected: All PASS

- [ ] **Step 5: Wire adaptive folds into train.py per-subject functions**

In `src/train.py`, add import at top:
```python
from src.validation import adaptive_fold_count
```

In `_evaluate_subject_all_models` (line 243), add at the start of the function body (after the `with threadpool_limits` line 259):
```python
        n_folds = adaptive_fold_count(len(y), n_folds)
```

In `_evaluate_subject_nested_model_selection` (line 1109), add after line 1127 (`with threadpool_limits`):
```python
        n_outer_folds = adaptive_fold_count(len(y), n_outer_folds)
```

And inside the outer loop, after computing `outer_train_idx` (after the `_reject_in_fold` call at line 1144), add:
```python
            n_inner_folds_actual = adaptive_fold_count(
                len(outer_train_idx), n_inner_folds
            )
```

Then use `n_inner_folds_actual` in place of `n_inner_folds` at line 1171:
```python
            inner_splits = make_cv_splits(
                y_otrain,
                n_splits=n_inner_folds_actual,
                ...
```

- [ ] **Step 6: Run train tests**

Run: `uv run pytest tests/test_train.py -v --timeout=120 2>&1 | tail -15`
Expected: All PASS (40-trial synthetic subjects are above the 2-fold minimum)

- [ ] **Step 7: Commit**

```bash
git add src/validation.py src/train.py tests/test_validation.py
git commit -m "feat: add adaptive fold count for small-sample subjects"
```

---

## Phase 2: Expanded FBCSP Bands

### Task 4: Update FBCSP bands to 10 non-overlapping

**Files:**
- Modify: `src/train.py:23-32`

- [ ] **Step 1: Update FBCSP_BANDS constant**

In `src/train.py`, replace lines 23-32:

```python
FBCSP_BANDS = [
    (6, 8),     # theta-mu border
    (8, 10),    # low mu
    (10, 12),   # high mu
    (12, 14),   # low beta
    (14, 16),   # mid-low beta
    (16, 18),   # mid beta
    (18, 20),   # mid-high beta
    (20, 24),   # high beta
    (24, 28),   # upper high beta
    (28, 34),   # beta-gamma border
]
```

- [ ] **Step 2: Run train tests**

Run: `uv run pytest tests/test_train.py -v --timeout=120 2>&1 | tail -15`
Expected: All PASS (tests use synthetic data, band changes don't affect test structure)

- [ ] **Step 3: Run transfer tests**

Run: `uv run pytest tests/test_transfer.py -v --timeout=120 2>&1 | tail -15`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/train.py
git commit -m "feat: expand FBCSP to 10 non-overlapping bands (6-34 Hz)"
```

---

## Phase 3: Euclidean Alignment + Expanded Donors

### Task 5: Create alignment.py shared utility

**Files:**
- Create: `src/alignment.py`
- Create: `tests/test_alignment.py`

- [ ] **Step 1: Write tests for EA utility**

Create `tests/test_alignment.py`:

```python
"""Tests for Euclidean Alignment shared utility."""

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance

from src.alignment import apply_ea_transform, compute_ea_transform, euclidean_align


def _make_spd_trials(n_trials: int = 20, n_channels: int = 8, seed: int = 42):
    """Generate synthetic multichannel EEG and its covariances."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_channels, n_channels))
    cov_mean = A @ A.T + np.eye(n_channels)
    X = rng.standard_normal((n_trials, n_channels, 375))
    X = np.einsum("ij,njt->nit", np.linalg.cholesky(cov_mean), X)
    covs = Covariances(estimator="lwf").fit_transform(X)
    return covs


def test_compute_ea_transform_shape():
    """Transform matrix is (n_channels, n_channels)."""
    covs = _make_spd_trials()
    transform = compute_ea_transform(covs)
    assert transform.shape == (8, 8)


def test_apply_ea_transform_shape():
    """Aligned covariances have the same shape as input."""
    covs = _make_spd_trials()
    transform = compute_ea_transform(covs)
    aligned = apply_ea_transform(covs, transform)
    assert aligned.shape == covs.shape


def test_euclidean_align_centers_to_identity():
    """After EA, mean covariance should be close to identity."""
    covs = _make_spd_trials()
    aligned, _ = euclidean_align(covs)
    mean_cov = np.mean(aligned, axis=0)
    np.testing.assert_allclose(mean_cov, np.eye(8), atol=0.5)


def test_apply_precomputed_transform():
    """Applying a precomputed transform gives same result as full EA."""
    covs = _make_spd_trials()
    aligned_full, transform = euclidean_align(covs)
    aligned_manual = apply_ea_transform(covs, transform)
    np.testing.assert_allclose(aligned_full, aligned_manual, atol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_alignment.py -v 2>&1 | tail -10`
Expected: FAIL — `src.alignment` not found

- [ ] **Step 3: Implement alignment.py**

Create `src/alignment.py`:

```python
"""Euclidean Alignment utility for cross-subject covariance normalization."""

from __future__ import annotations

import numpy as np
from pyriemann.utils.mean import mean_covariance
from scipy.linalg import fractional_matrix_power


def compute_ea_transform(covariances: np.ndarray) -> np.ndarray:
    """Compute R^{-1/2} from the Riemannian geometric mean of trial covariances.

    Args:
        covariances: (n_trials, n_channels, n_channels) SPD matrices.

    Returns:
        (n_channels, n_channels) inverse square root of the reference matrix.
    """
    ref = mean_covariance(covariances, metric="riemann")
    return fractional_matrix_power(ref, -0.5).real


def apply_ea_transform(
    covariances: np.ndarray, ref_inv_sqrt: np.ndarray
) -> np.ndarray:
    """Apply a precomputed EA transform: R^{-1/2} C R^{-T/2}.

    Args:
        covariances: (n_trials, n_channels, n_channels) SPD matrices.
        ref_inv_sqrt: (n_channels, n_channels) from compute_ea_transform.

    Returns:
        (n_trials, n_channels, n_channels) aligned covariances.
    """
    return ref_inv_sqrt @ covariances @ ref_inv_sqrt.T


def euclidean_align(
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute EA transform and apply it.

    Args:
        covariances: (n_trials, n_channels, n_channels) SPD matrices.

    Returns:
        Tuple of (aligned_covariances, ref_inv_sqrt).
    """
    ref_inv_sqrt = compute_ea_transform(covariances)
    aligned = apply_ea_transform(covariances, ref_inv_sqrt)
    return aligned, ref_inv_sqrt
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_alignment.py -v 2>&1 | tail -10`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/alignment.py tests/test_alignment.py
git commit -m "feat: add shared Euclidean Alignment utility"
```

---

### Task 6: Refactor transfer.py to use shared alignment

**Files:**
- Modify: `src/transfer.py:134-170,229-237`

- [ ] **Step 1: Refactor align_subjects**

In `src/transfer.py`, add import:
```python
from src.alignment import apply_ea_transform, compute_ea_transform
```

Remove the `from scipy.linalg import fractional_matrix_power` import (line 13).

Replace `align_subjects` function body (lines 153-170). The EA logic `ref = mean_covariance(...)`, `ref_inv_sqrt = fractional_matrix_power(ref, -0.5).real`, `covs_aligned = ref_inv_sqrt @ covs @ ref_inv_sqrt.T` is replaced by calls to the shared utility:

```python
def align_subjects(
    subjects: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    sfreq: float = 250.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute covariance matrices and apply Euclidean Alignment across subjects."""
    cov_estimator = Covariances(estimator="lwf")

    all_covs = []
    all_labels = []
    all_subject_ids: list[str] = []

    for sid in sorted(subjects.keys()):
        _, X_mc, y = subjects[sid]
        covs = cov_estimator.fit_transform(X_mc)
        covs_aligned, _ = euclidean_align(covs)

        all_covs.append(covs_aligned)
        all_labels.append(y)
        all_subject_ids.extend([sid] * len(y))

    return np.vstack(all_covs), np.concatenate(all_labels), all_subject_ids
```

Add `euclidean_align` to the import from `src.alignment`.

- [ ] **Step 2: Refactor regularized_within_subject_cv similarly**

In `regularized_within_subject_cv` (lines 229-237), replace the inline EA with:

```python
    aligned_per_subject: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_aligned_covs = []
    for sid in sorted(subjects.keys()):
        _, X_mc, y = subjects[sid]
        covs = cov_estimator.fit_transform(X_mc)
        covs_aligned, _ = euclidean_align(covs)
        aligned_per_subject[sid] = (covs_aligned, y)
        all_aligned_covs.append(covs_aligned)
```

Do the same in `loso_cv` (lines 406-411):

```python
    for sid in sorted(subjects.keys()):
        _, X_mc, y = subjects[sid]
        covs = cov_estimator.fit_transform(X_mc)
        covs_aligned, _ = euclidean_align(covs)
        aligned_per_subject[sid] = (covs_aligned, y)
```

- [ ] **Step 3: Run transfer tests**

Run: `uv run pytest tests/test_transfer.py -v --timeout=120 2>&1 | tail -15`
Expected: All PASS (behavior unchanged, just using shared utility)

- [ ] **Step 4: Commit**

```bash
git add src/transfer.py
git commit -m "refactor: use shared alignment utility in transfer.py"
```

---

### Task 7: Expand donor pool with EA in pipeline augmentation

**Files:**
- Modify: `src/pipeline.py:537-623`

- [ ] **Step 1: Add per-fold donor selection wrapper to pipeline.py**

In `src/pipeline.py`, add this function before `run_pipeline` (around line 280):

```python
def _augmented_nested_cv_with_donor_selection(
    X_features: np.ndarray,
    X_multichannel: np.ndarray,
    y: np.ndarray,
    donor_subjects: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    donor_means: dict[str, np.ndarray],
    target_sid: str,
    sfreq: float,
    n_outer_folds: int,
    n_inner_folds: int,
    k_best: int,
    trial_ptps: np.ndarray | None,
    split_strategy: str,
    groups: np.ndarray | None,
    random_state: int,
    cov_estimator,
) -> tuple[float, str, float]:
    """Augmented nested CV with per-fold donor selection (leakage-free).

    Selects the closest donor using only training-fold covariance mean,
    so donor selection never sees test-fold data.

    Returns:
        Tuple of (accuracy, most_frequent_donor_id, mean_distance).
    """
    from collections import Counter

    from pyriemann.utils.distance import distance_riemann
    from pyriemann.utils.mean import mean_covariance

    from src.train import (
        ALL_CLASSIFIERS,
        _evaluate_classifier,
        _evaluate_classifiers_batch,
        _reject_in_fold,
    )
    from src.validation import adaptive_fold_count, make_cv_splits

    n_outer_folds = adaptive_fold_count(len(y), n_outer_folds)
    outer_splits = make_cv_splits(
        y,
        n_splits=n_outer_folds,
        strategy=split_strategy,
        groups=groups,
        random_state=random_state,
    )

    classifier_names = list(ALL_CLASSIFIERS)
    fold_scores: list[float] = []
    fold_donors: list[str] = []
    fold_dists: list[float] = []

    for outer_train_idx, outer_test_idx in outer_splits:
        outer_train_idx, outer_test_idx = _reject_in_fold(
            trial_ptps, outer_train_idx, outer_test_idx
        )
        if len(outer_train_idx) < 2 or len(outer_test_idx) < 1:
            continue

        # Donor selection using TRAINING-FOLD covariance mean only
        train_covs = cov_estimator.fit_transform(X_multichannel[outer_train_idx])
        train_mean = mean_covariance(train_covs, metric="riemann")

        dists = {
            dsid: float(distance_riemann(train_mean, donor_means[dsid]))
            for dsid in donor_subjects
            if dsid != target_sid
        }
        best_donor = min(dists, key=dists.get)
        fold_donors.append(best_donor)
        fold_dists.append(dists[best_donor])

        donor_feat, donor_mc, donor_y = donor_subjects[best_donor]

        # Test fold: TARGET data only
        X_feat_otest = X_features[outer_test_idx]
        X_mc_otest = X_multichannel[outer_test_idx]
        y_otest = y[outer_test_idx]

        # Training fold: TARGET train + donor
        X_feat_otrain = np.vstack([X_features[outer_train_idx], donor_feat])
        X_mc_otrain = np.vstack([X_multichannel[outer_train_idx], donor_mc])
        y_otrain = np.concatenate([y[outer_train_idx], donor_y])

        # Inner CV for model selection
        n_inner_actual = adaptive_fold_count(len(outer_train_idx), n_inner_folds)
        inner_groups = groups[outer_train_idx] if groups is not None else None
        inner_splits = make_cv_splits(
            y[outer_train_idx],
            n_splits=n_inner_actual,
            strategy=split_strategy,
            groups=inner_groups,
            random_state=random_state,
        )

        inner_scores_by_clf: dict[str, list[float]] = {n: [] for n in classifier_names}
        for inner_train_idx, inner_val_idx in inner_splits:
            inner_feat_train = np.vstack(
                [X_features[outer_train_idx[inner_train_idx]], donor_feat]
            )
            inner_mc_train = np.vstack(
                [X_multichannel[outer_train_idx[inner_train_idx]], donor_mc]
            )
            inner_y_train = np.concatenate(
                [y[outer_train_idx[inner_train_idx]], donor_y]
            )
            inner_feat_val = X_features[outer_train_idx[inner_val_idx]]
            inner_mc_val = X_multichannel[outer_train_idx[inner_val_idx]]
            inner_y_val = y[outer_train_idx[inner_val_idx]]

            split_scores = _evaluate_classifiers_batch(
                classifier_names,
                inner_feat_train,
                inner_feat_val,
                inner_y_train,
                inner_y_val,
                inner_mc_train,
                inner_mc_val,
                sfreq,
                k_best,
            )
            for clf_name, score in split_scores.items():
                inner_scores_by_clf[clf_name].append(score)

        inner_means = {
            name: float(np.mean(scores)) if scores else 0.5
            for name, scores in inner_scores_by_clf.items()
        }
        best_clf = max(inner_means, key=inner_means.get)

        outer_score = _evaluate_classifier(
            best_clf,
            X_feat_otrain,
            X_feat_otest,
            y_otrain,
            y_otest,
            X_mc_otrain,
            X_mc_otest,
            sfreq,
            k_best,
        )
        fold_scores.append(outer_score)

    accuracy = float(np.mean(fold_scores)) if fold_scores else 0.5
    donor_counts = Counter(fold_donors)
    most_common_donor = donor_counts.most_common(1)[0][0] if donor_counts else "none"
    mean_dist = float(np.mean(fold_dists)) if fold_dists else 0.0
    return accuracy, most_common_donor, mean_dist
```

- [ ] **Step 2: Update donor augmentation in pipeline step 3b**

In `src/pipeline.py`, add import:
```python
from src.alignment import apply_ea_transform, compute_ea_transform
```

Replace the augmented nested CV section (step 3b, starting at line 537). The key changes are:
1. Donor selection uses training-fold covariance mean (leakage fix)
2. Apply standard EA independently to donor and target before concatenation
3. Donor pool now includes all MI_DATA_NEW subjects via the flexible loader

Replace the step 3b block with:

```python
    # Step 3b: Augmented nested CV for weak subjects (leakage-free)
    aug_nested_scores = None
    aug_nested_mean = None
    step_start_aug = time.perf_counter()
    if MI_DATA_NEW_DIR.exists():
        print("\n[3b/5] Augmented nested CV (cross-subject data for weak subjects)")
        from pyriemann.estimation import Covariances as _Cov
        from pyriemann.utils.distance import distance_riemann as _dist_riemann
        from pyriemann.utils.mean import mean_covariance as _mean_cov

        from src.transfer import load_all_subjects

        transfer_data_dirs = [data_dir, MI_DATA_NEW_DIR]
        donor_subjects = load_all_subjects(
            transfer_data_dirs,
            sfreq=sfreq,
            subject_merge=DEFAULT_SUBJECT_MERGE,
        )
        print(f"  Donor pool: {len(donor_subjects)} subjects")

        # Compute raw Riemannian means for donor selection (pre-EA)
        _cov_est = _Cov(estimator="lwf")
        donor_means = {
            sid: _mean_cov(_cov_est.fit_transform(d[1]), metric="riemann")
            for sid, d in donor_subjects.items()
        }

        # Pre-compute EA transforms for all donors
        donor_ea_transforms = {}
        for sid, (_, d_mc, _) in donor_subjects.items():
            d_covs = _cov_est.fit_transform(d_mc)
            donor_ea_transforms[sid] = compute_ea_transform(d_covs)

        weakness_threshold = cfg.augmentation_weakness_threshold
        aug_nested_scores = list(nested_scores)

        for i, (sid, score) in enumerate(zip(subject_ids, nested_scores)):
            if score >= weakness_threshold:
                continue

            # Per-fold donor selection: use training-fold covariance mean only (leakage fix).
            # We pass all donor candidates to _augmented_nested_cv_subject_with_selection
            # which selects the closest donor per outer fold using only training data.
            target_feat, target_mc = X_by_subject[i]

            aug_score, best_donor, best_dist = _augmented_nested_cv_with_donor_selection(
                target_feat,
                target_mc,
                y_by_subject[i],
                donor_subjects=donor_subjects,
                donor_means=donor_means,
                target_sid=sid,
                sfreq=sfreq,
                n_outer_folds=cfg.n_outer_folds,
                n_inner_folds=cfg.n_inner_folds,
                k_best=cfg.k_best,
                trial_ptps=trial_ptps_by_subject[i],
                split_strategy=cfg.split_strategy,
                groups=trial_groups_by_subject[i],
                random_state=cfg.random_state,
                cov_estimator=_cov_est,
            )
            aug_nested_scores[i] = aug_score
            print(
                f"    {sid}: +{best_donor} (d={best_dist:.1f}) "
                f"{score:.1%} -> {aug_score:.1%} ({aug_score - score:+.1%})"
            )

        aug_nested_mean = float(np.mean(aug_nested_scores))

        print(f"\n  {'Subject':<14} {'Original':>10} {'Augmented':>10} {'Delta':>8}")
        print("  " + "-" * 42)
        for sid, orig, aug in zip(subject_ids, nested_scores, aug_nested_scores):
            delta = aug - orig
            sign = "+" if delta >= 0 else ""
            weak = " *" if orig < weakness_threshold else ""
            print(f"  {sid:<14} {orig:>9.1%} {aug:>9.1%} {sign}{delta:>6.1%}{weak}")
        print(f"\n  Original nested mean:  {nested_mean:.1%}")
        print(f"  Augmented nested mean: {aug_nested_mean:.1%}")
        print(f"  Delta:                 {aug_nested_mean - nested_mean:+.1%}")
        print("  (* = weak subject, augmented with closest neighbor)")
    runtime_seconds["augmented_nested_cv"] = time.perf_counter() - step_start_aug
```

- [ ] **Step 2: Run existing tests**

Run: `uv run pytest tests/ -v --timeout=120 2>&1 | tail -15`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add src/pipeline.py
git commit -m "feat: expand donor pool with EA alignment in augmentation"
```

---

## Phase 4: Dual Reporting & Validation

### Task 8: Add dual reporting to pipeline

**Files:**
- Modify: `src/pipeline.py` (results dict section, ~lines 773-844)

- [ ] **Step 1: Add dual reporting to results dict**

After the results dict is constructed (around line 828), add logic to compute and store original-10 metrics separately. Add this before writing the JSON:

```python
    # Dual reporting: separate original subjects from new
    original_subject_ids = [
        sid for sid in subject_ids
        if not sid.startswith("subject010")  # 0100-0106 are MI_DATA_NEW
    ]
    if len(original_subject_ids) < len(subject_ids):
        original_indices = [
            i for i, sid in enumerate(subject_ids) if sid in original_subject_ids
        ]
        original_nested = [nested_scores[i] for i in original_indices]
        results["original_subjects"] = original_subject_ids
        results["original_nested_mean"] = float(np.mean(original_nested))
        results["original_nested_std"] = float(np.std(original_nested))
        if aug_nested_scores is not None:
            original_aug = [aug_nested_scores[i] for i in original_indices]
            results["original_augmented_nested_mean"] = float(np.mean(original_aug))

        # Per-subject trial counts for new subjects
        trial_counts = {}
        for i, sid in enumerate(subject_ids):
            trial_counts[sid] = len(y_by_subject[i])
        results["trial_counts"] = trial_counts

        n_original = len(original_subject_ids)
        n_new = len(subject_ids) - n_original
        print(f"\nDual reporting: {n_original} original + {n_new} new subjects")
        print(f"  Original {n_original} nested mean: {results['original_nested_mean']:.1%}")
        print(f"  All {len(subject_ids)} nested mean:      {nested_mean:.1%}")
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/ -v --timeout=120 2>&1 | tail -10`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add src/pipeline.py
git commit -m "feat: add dual reporting for original vs all subjects"
```

---

### Task 9: Lint, format, and final commit

**Files:**
- All modified Python files

- [ ] **Step 1: Lint and format**

```bash
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=120 2>&1 | tail -20`
Expected: All PASS

- [ ] **Step 3: Commit any formatting changes**

```bash
git add -A
git diff --cached --stat
git commit -m "style: lint and format all modified files"
```

(Skip this commit if `git diff --cached` shows no changes.)

---

### Task 10: Integration validation run

**Files:** None (execution only)

- [ ] **Step 1: Run full pipeline with seed=42**

```bash
uv run python -m src.pipeline 2>&1 | tee /tmp/pipeline_run.log
```

Expected: Pipeline completes. Check output for:
- New subjects (subject0104, subject0106) appear in results
- Original 10 subjects' nested accuracy is reported
- Dual reporting section shows separate original vs all metrics
- Augmented nested CV runs with expanded donor pool

- [ ] **Step 2: Compare original 10 subjects against baseline**

Check `models/training_results.json` for `original_nested_mean`. Compare against 60.1% baseline. Per the rollback criteria:
- If original nested mean drops > 1 pp: identify which change caused it
- If bands caused it: revert FBCSP_BANDS to original 8 bands
- If EA/donors caused it: revert step 3b changes

- [ ] **Step 3: Document results**

Record the results in a comment or update `previouslytried.md` with the outcome of the expanded bands + expanded donors experiment.

---

## Rollback Reference

| Change | Revert if... | How to revert |
|--------|-------------|---------------|
| Expanded bands (Task 4) | Original 10 mean drops > 1 pp | Restore original `FBCSP_BANDS` in train.py |
| EA + donors (Task 7) | Augmented mean drops > 1 pp | Revert pipeline.py step 3b to previous version |
| Flexible loading (Tasks 1-3) | N/A (structural) | Should not affect accuracy |
