# Phase 1: Infrastructure That Unlocks Accuracy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the BCI pipeline into modular, testable components that enable faster accuracy experimentation.

**Architecture:** Extract band caching, artifact rejection, and classifier logic from train.py (1580 lines) into focused modules. Add unified experiment tracking and a CLI layer for partial pipeline runs. All 61 existing tests must pass at every commit.

**Tech Stack:** Python 3.11, scikit-learn, MNE-Python, pyriemann, NumPy, pytest

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `src/band_cache.py` | Band-filtered data caching, subsetting, FBCSP constants | New |
| `src/rejection.py` | Two-layer artifact rejection (fold-time PTP + epoch-time multi-criteria) | New |
| `src/classifiers.py` | Classifier registry with protocol-based adapters | New |
| `src/experiments.py` | Structured experiment tracking, comparison, CLI | New |
| `src/cli.py` | CLI argument parsing, RunConfig dataclass | New |
| `src/train.py` | CV orchestration only (slimmed from 1580 to ~500 lines) | Modified |
| `src/pipeline.py` | Stage-based pipeline coordinator (slimmed from 852 to ~350 lines) | Modified |
| `src/config.py` | Adds `config_hash()` method | Modified |
| `src/epochs.py` | Epoch-time rejection re-exported from rejection.py | Modified |
| `src/transfer.py` | Absorbs donor selection from pipeline.py | Modified |
| `.gitignore` | Adds `cache/`, `experiments/` | Modified |
| `tests/test_band_cache.py` | Tests for band cache module | New |
| `tests/test_rejection.py` | Tests for rejection module | New |
| `tests/test_classifiers.py` | Tests for classifier registry | New |
| `tests/test_experiments.py` | Tests for experiment tracking | New |
| `tests/test_cli.py` | Tests for CLI and RunConfig | New |
| `tests/test_pipeline_integration.py` | Integration tests for staged pipeline | New |

---

### Task 1: Extract `src/band_cache.py`

Move band-filtering logic out of train.py into its own module. This is the simplest extraction — pure functions with no cross-dependencies.

**Files:**
- Create: `src/band_cache.py`
- Create: `tests/test_band_cache.py`
- Modify: `src/train.py` (remove moved code, add re-exports)

- [ ] **Step 1: Write failing tests for band_cache**

Create `tests/test_band_cache.py`:

```python
"""Tests for band-filtered data caching."""

import numpy as np
import pytest

from src.band_cache import (
    FBCSP_BANDS,
    N_CSP_COMPONENTS,
    BandCache,
    precompute_bandpassed,
    subset_band_cache,
)

N_TRIALS = 10
N_CHANNELS = 8
N_SAMPLES = 375  # 1.5s at 250Hz
SFREQ = 250.0


def _make_multichannel(n_trials: int = N_TRIALS) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_trials, N_CHANNELS, N_SAMPLES))


class TestPrecomputeBandpassed:
    def test_returns_dict_keyed_by_band_tuples(self):
        X = _make_multichannel()
        cache = precompute_bandpassed(X, SFREQ)
        assert isinstance(cache, dict)
        for key in cache:
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_each_band_preserves_shape(self):
        X = _make_multichannel()
        cache = precompute_bandpassed(X, SFREQ)
        for band, X_band in cache.items():
            assert X_band.shape == X.shape

    def test_custom_bands(self):
        X = _make_multichannel()
        bands = [(8, 12), (12, 30)]
        cache = precompute_bandpassed(X, SFREQ, bands=bands)
        assert set(cache.keys()) == {(8, 12), (12, 30)}

    def test_default_bands_match_constant(self):
        X = _make_multichannel()
        cache = precompute_bandpassed(X, SFREQ)
        assert set(cache.keys()) == set(FBCSP_BANDS)


class TestSubsetBandCache:
    def test_subsets_by_trial_indices(self):
        X = _make_multichannel(20)
        cache = precompute_bandpassed(X, SFREQ)
        indices = np.array([0, 5, 10, 15])
        subset = subset_band_cache(cache, indices)
        for band in cache:
            assert subset[band].shape[0] == 4
            np.testing.assert_array_equal(subset[band], cache[band][indices])

    def test_preserves_all_bands(self):
        X = _make_multichannel()
        cache = precompute_bandpassed(X, SFREQ)
        indices = np.array([0, 1, 2])
        subset = subset_band_cache(cache, indices)
        assert set(subset.keys()) == set(cache.keys())


class TestConstants:
    def test_fbcsp_bands_are_8_bands(self):
        assert len(FBCSP_BANDS) == 8

    def test_bands_cover_8_to_30_hz(self):
        assert FBCSP_BANDS[0][0] == 8
        assert FBCSP_BANDS[-1][1] == 30

    def test_n_csp_components(self):
        assert N_CSP_COMPONENTS == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_band_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.band_cache'`

- [ ] **Step 3: Create `src/band_cache.py`**

Move the following from `src/train.py` into `src/band_cache.py`:
- Lines 23-32: `FBCSP_BANDS` constant
- Line 34: `N_CSP_COMPONENTS` constant
- Line 39: `BandCache` type alias
- Lines 42-60: `_precompute_bandpassed` function (rename to `precompute_bandpassed`)
- Lines 63-65: `_subset_band_cache` function (rename to `subset_band_cache`)

```python
"""Band-filtered data caching for FBCSP pipelines."""

import mne
import numpy as np

FBCSP_BANDS: list[tuple[float, float]] = [
    (8, 10),
    (10, 12),
    (12, 14),
    (14, 16),
    (16, 18),
    (18, 20),
    (20, 24),
    (24, 30),
]

N_CSP_COMPONENTS: int = 3

BandCache = dict[tuple[float, float], np.ndarray]


def precompute_bandpassed(
    X: np.ndarray,
    sfreq: float,
    bands: list[tuple[float, float]] = FBCSP_BANDS,
) -> BandCache:
    """Bandpass every trial for each FBCSP band once.

    Filtering is trial-wise along the time axis, so cached per-band tensors can be
    safely indexed for CV train/test splits without changing leakage boundaries.
    """
    filtered_by_band: BandCache = {}
    for low, high in bands:
        try:
            filtered_by_band[(low, high)] = mne.filter.filter_data(
                X, sfreq, low, high, verbose=False
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
    return filtered_by_band


def subset_band_cache(cache: BandCache, indices: np.ndarray) -> BandCache:
    """Slice a precomputed band cache by trial indices."""
    return {band: X_band[indices] for band, X_band in cache.items()}
```

- [ ] **Step 4: Update `src/train.py` to import from `band_cache`**

Replace the moved code in `src/train.py` with imports and re-exports for backward compatibility:

At the top of `src/train.py`, replace lines 23-39 and lines 42-65 with:

```python
from src.band_cache import (
    FBCSP_BANDS,
    N_CSP_COMPONENTS,
    BandCache,
    precompute_bandpassed,
    subset_band_cache,
)

# Backward-compatible aliases (tests import these private names)
_precompute_bandpassed = precompute_bandpassed
_subset_band_cache = subset_band_cache
```

Keep `DEFAULT_K_CANDIDATES`, `ALL_CLASSIFIERS`, and the `ParallelBackend`/`CacheScope` type aliases in train.py. Remove the duplicate `CacheScope` import from config.py if present — train.py already re-exports it.

- [ ] **Step 5: Run all tests to verify nothing breaks**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS (61 existing tests + new band_cache tests)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/band_cache.py src/train.py tests/test_band_cache.py
uv run ruff format src/band_cache.py src/train.py tests/test_band_cache.py
git add src/band_cache.py tests/test_band_cache.py src/train.py
git commit -m "Extract band_cache module from train.py"
```

---

### Task 2: Extract `src/rejection.py`

Move fold-time rejection from train.py into a dedicated module. The epoch-time multi-criteria rejection stays in epochs.py but is re-exported from rejection.py for discoverability.

**Files:**
- Create: `src/rejection.py`
- Create: `tests/test_rejection.py`
- Modify: `src/train.py` (remove `_reject_in_fold`, add import)

- [ ] **Step 1: Write failing tests for rejection**

Create `tests/test_rejection.py`:

```python
"""Tests for artifact rejection module."""

import numpy as np
import pytest

from src.rejection import reject_in_fold


class TestRejectInFold:
    def test_none_ptps_returns_unchanged(self):
        train_idx = np.array([0, 1, 2, 3])
        test_idx = np.array([4, 5])
        clean_train, clean_test = reject_in_fold(None, train_idx, test_idx)
        np.testing.assert_array_equal(clean_train, train_idx)
        np.testing.assert_array_equal(clean_test, test_idx)

    def test_removes_high_amplitude_trials(self):
        # 10 trials, one extreme outlier at index 2
        ptps = np.array([50, 60, 500, 55, 65, 45, 70, 50, 55, 60], dtype=float)
        train_idx = np.arange(8)
        test_idx = np.array([8, 9])
        clean_train, clean_test = reject_in_fold(ptps, train_idx, test_idx)
        assert 2 not in clean_train

    def test_removes_flat_trials(self):
        ptps = np.array([50, 60, 0.5, 55, 65, 45, 70, 50, 55, 60], dtype=float)
        train_idx = np.arange(8)
        test_idx = np.array([8, 9])
        clean_train, clean_test = reject_in_fold(ptps, train_idx, test_idx)
        assert 2 not in clean_train

    def test_threshold_from_training_only(self):
        # Train trials are low amplitude, test has one high trial
        # Threshold is computed from train only
        ptps = np.array([10, 12, 11, 13, 10, 200], dtype=float)
        train_idx = np.array([0, 1, 2, 3, 4])
        test_idx = np.array([5])
        clean_train, clean_test = reject_in_fold(ptps, train_idx, test_idx)
        # Test trial at 200 should be rejected (way above train median + n_mad*MAD)
        assert len(clean_test) == 0

    def test_custom_n_mad(self):
        ptps = np.array([50, 60, 120, 55, 65, 45, 70, 50, 55, 60], dtype=float)
        train_idx = np.arange(8)
        test_idx = np.array([8, 9])
        # Very tight threshold should reject more
        clean_tight, _ = reject_in_fold(ptps, train_idx, test_idx, n_mad=1.0)
        clean_loose, _ = reject_in_fold(ptps, train_idx, test_idx, n_mad=10.0)
        assert len(clean_tight) <= len(clean_loose)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rejection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.rejection'`

- [ ] **Step 3: Create `src/rejection.py`**

Move `_reject_in_fold` (train.py lines 174-210) into `src/rejection.py` as `reject_in_fold`:

```python
"""Artifact rejection for EEG trials.

Two-layer API:
- reject_in_fold: Fold-time rejection using per-trial max PTP + index filtering.
  Threshold computed from training indices only, applied to both splits.
- Epoch-time multi-criteria rejection (PTP, gradient, HF-power) remains in
  src/epochs.py (reject_bad_epochs). Re-exported here for discoverability.
"""

import numpy as np

from src.epochs import reject_bad_epochs  # re-export for discoverability


def reject_in_fold(
    trial_ptps: np.ndarray | None,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_mad: float = 3.5,
    flat_uv: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter train/test indices using an adaptive amplitude threshold.

    The threshold is computed from training indices only, then applied to both
    splits.  When *trial_ptps* is ``None``, indices are returned unchanged.

    Args:
        trial_ptps: Max peak-to-peak amplitude per trial, or ``None``.
        train_idx: Training fold indices.
        test_idx: Test fold indices.
        n_mad: Number of MADs above median for the threshold.
        flat_uv: Minimum PTP below which a trial is considered flat.

    Returns:
        (clean_train_idx, clean_test_idx).
    """
    if trial_ptps is None:
        return train_idx, test_idx

    train_ptps = trial_ptps[train_idx]
    median = float(np.median(train_ptps))
    mad = float(np.median(np.abs(train_ptps - median)))
    threshold = median + n_mad * mad

    def _keep(idx: np.ndarray) -> np.ndarray:
        ptps = trial_ptps[idx]
        mask = (ptps >= flat_uv) & (ptps <= threshold)
        return idx[mask]

    return _keep(train_idx), _keep(test_idx)
```

- [ ] **Step 4: Update `src/train.py` to import from `rejection`**

Replace `_reject_in_fold` in `src/train.py` (lines 174-210) with:

```python
from src.rejection import reject_in_fold

# Backward-compatible alias (tests and pipeline.py import this private name)
_reject_in_fold = reject_in_fold
```

- [ ] **Step 5: Run all tests to verify nothing breaks**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/rejection.py src/train.py tests/test_rejection.py
uv run ruff format src/rejection.py src/train.py tests/test_rejection.py
git add src/rejection.py tests/test_rejection.py src/train.py
git commit -m "Extract rejection module from train.py"
```

---

### Task 3: Extract `src/classifiers.py`

Create the classifier registry with protocol-based adapters. This is the highest-leverage extraction — it enables Phase 2 accuracy experiments.

**Files:**
- Create: `src/classifiers.py`
- Create: `tests/test_classifiers.py`
- Modify: `src/train.py` (use registry in `_evaluate_classifiers_batch`)

- [ ] **Step 1: Write failing tests for classifier registry**

Create `tests/test_classifiers.py`:

```python
"""Tests for classifier registry and adapters."""

import numpy as np
import pytest

from src.classifiers import (
    CLASSIFIER_REGISTRY,
    ClassifierAdapter,
    get_classifier,
    register_classifier,
)

N_TRIALS = 40
N_FEATURES = 10
N_CHANNELS = 8
N_SAMPLES = 375
SFREQ = 250.0


def _make_data(n: int = N_TRIALS):
    rng = np.random.default_rng(42)
    X_feat = rng.standard_normal((n, N_FEATURES))
    X_mc = rng.standard_normal((n, N_CHANNELS, N_SAMPLES))
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X_feat, X_mc, y


class TestRegistry:
    def test_builtin_classifiers_registered(self):
        assert "lda" in CLASSIFIER_REGISTRY
        assert "riemann" in CLASSIFIER_REGISTRY
        assert "svm" in CLASSIFIER_REGISTRY
        assert "ensemble" in CLASSIFIER_REGISTRY

    def test_get_classifier_returns_instance(self):
        clf = get_classifier("lda")
        assert isinstance(clf, ClassifierAdapter)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            get_classifier("nonexistent")

    def test_custom_registration(self):
        @register_classifier("test_dummy")
        class DummyClassifier(ClassifierAdapter):
            name = "test_dummy"

            def fit(self, X_feat_train, y_train, X_mc_train, sfreq, config, band_cache=None):
                pass

            def score(self, X_feat_test, y_test, X_mc_test):
                return 0.5

            def predict_proba(self, X_feat_test, X_mc_test):
                n = X_feat_test.shape[0]
                return np.full((n, 2), 0.5)

            def export(self):
                return {"type": "dummy"}

        assert "test_dummy" in CLASSIFIER_REGISTRY
        # Clean up
        del CLASSIFIER_REGISTRY["test_dummy"]


class TestLDAClassifier:
    def test_fit_score_roundtrip(self):
        X_feat, X_mc, y = _make_data()
        clf = get_classifier("lda")
        clf.fit(X_feat[:30], y[:30], X_mc[:30], SFREQ, config=None)
        score = clf.score(X_feat[30:], y[30:], X_mc[30:])
        assert 0.0 <= score <= 1.0

    def test_export_has_required_keys(self):
        X_feat, X_mc, y = _make_data()
        clf = get_classifier("lda")
        clf.fit(X_feat[:30], y[:30], X_mc[:30], SFREQ, config=None)
        exported = clf.export()
        assert "csp_models" in exported
        assert "classifier" in exported


class TestRiemannClassifier:
    def test_fit_score_roundtrip(self):
        X_feat, X_mc, y = _make_data()
        clf = get_classifier("riemann")
        clf.fit(X_feat[:30], y[:30], X_mc[:30], SFREQ, config=None)
        score = clf.score(X_feat[30:], y[30:], X_mc[30:])
        assert 0.0 <= score <= 1.0

    def test_export_has_pipeline(self):
        X_feat, X_mc, y = _make_data()
        clf = get_classifier("riemann")
        clf.fit(X_feat[:30], y[:30], X_mc[:30], SFREQ, config=None)
        exported = clf.export()
        assert "pipeline" in exported


class TestSVMClassifier:
    def test_fit_score_roundtrip(self):
        X_feat, X_mc, y = _make_data()
        clf = get_classifier("svm")
        clf.fit(X_feat[:30], y[:30], X_mc[:30], SFREQ, config=None)
        score = clf.score(X_feat[30:], y[30:], X_mc[30:])
        assert 0.0 <= score <= 1.0

    def test_predict_proba_shape(self):
        X_feat, X_mc, y = _make_data()
        clf = get_classifier("svm")
        clf.fit(X_feat[:30], y[:30], X_mc[:30], SFREQ, config=None)
        proba = clf.predict_proba(X_feat[30:], X_mc[30:])
        assert proba.shape == (10, 2)


class TestEnsembleClassifier:
    def test_fit_score_roundtrip(self):
        X_feat, X_mc, y = _make_data()
        clf = get_classifier("ensemble")
        clf.fit(X_feat[:30], y[:30], X_mc[:30], SFREQ, config=None)
        score = clf.score(X_feat[30:], y[30:], X_mc[30:])
        assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_classifiers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.classifiers'`

- [ ] **Step 3: Create `src/classifiers.py` with registry and adapter base**

```python
"""Classifier registry with protocol-based adapters.

Each classifier family (FBCSP-based, Riemannian, Ensemble) exposes fit, score,
predict_proba, and export methods. Adding a new classifier = one adapter class +
one @register_classifier decorator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.band_cache import (
    FBCSP_BANDS,
    N_CSP_COMPONENTS,
    BandCache,
    precompute_bandpassed,
    subset_band_cache,
)

DEFAULT_K_CANDIDATES = (3, 5, 8, 10, 15, 20, 25)

CLASSIFIER_REGISTRY: dict[str, type[ClassifierAdapter]] = {}


def register_classifier(name: str):
    """Decorator to register a classifier adapter class."""

    def decorator(cls: type[ClassifierAdapter]) -> type[ClassifierAdapter]:
        CLASSIFIER_REGISTRY[name] = cls
        return cls

    return decorator


def get_classifier(name: str) -> ClassifierAdapter:
    """Instantiate a registered classifier by name."""
    if name not in CLASSIFIER_REGISTRY:
        raise KeyError(f"Unknown classifier: {name!r}. Available: {list(CLASSIFIER_REGISTRY)}")
    return CLASSIFIER_REGISTRY[name]()


class ClassifierAdapter(ABC):
    """Base class for classifier adapters."""

    name: str

    @abstractmethod
    def fit(
        self,
        X_feat_train: np.ndarray,
        y_train: np.ndarray,
        X_mc_train: np.ndarray,
        sfreq: float,
        config: Any | None = None,
        band_cache: BandCache | None = None,
    ) -> None: ...

    @abstractmethod
    def score(
        self,
        X_feat_test: np.ndarray,
        y_test: np.ndarray,
        X_mc_test: np.ndarray,
    ) -> float: ...

    @abstractmethod
    def predict_proba(
        self,
        X_feat_test: np.ndarray,
        X_mc_test: np.ndarray,
    ) -> np.ndarray: ...

    @abstractmethod
    def export(self) -> dict[str, Any]: ...


def _safe_k_values(
    n_features: int,
    n_train_trials: int,
    k_best: int = 10,
    k_candidates: tuple[int, ...] = DEFAULT_K_CANDIDATES,
) -> list[int]:
    """Return conservative K values to reduce overfitting on small folds."""
    if n_features <= 0:
        return []
    k_cap = min(n_features, max(2, n_train_trials // 3))
    candidates = sorted(k for k in k_candidates if 1 <= k <= k_cap)
    return candidates or [min(k_best, k_cap)]


def _extract_fbcsp_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    sfreq: float,
    bands: list[tuple[float, float]] = FBCSP_BANDS,
    n_components: int = N_CSP_COMPONENTS,
    train_cache: BandCache | None = None,
    test_cache: BandCache | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[CSP, tuple[float, float]]]]:
    """Extract FBCSP features, using precomputed band cache if available."""
    if train_cache is not None and test_cache is not None:
        return _extract_fbcsp_from_cache(
            train_cache, test_cache, y_train, bands, n_components,
            n_train=X_train.shape[0], n_test=X_test.shape[0],
        )

    train_filtered = precompute_bandpassed(X_train, sfreq, bands=bands)
    test_filtered = precompute_bandpassed(X_test, sfreq, bands=bands)
    return _extract_fbcsp_from_cache(
        train_filtered, test_filtered, y_train, bands, n_components,
        n_train=X_train.shape[0], n_test=X_test.shape[0],
    )


def _extract_fbcsp_from_cache(
    train_cache: BandCache,
    test_cache: BandCache,
    y_train: np.ndarray,
    bands: list[tuple[float, float]],
    n_components: int,
    n_train: int,
    n_test: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[CSP, tuple[float, float]]]]:
    """Extract FBCSP features from pre-filtered band caches."""
    csp_models: list[tuple[CSP, tuple[float, float]]] = []
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for low, high in bands:
        band = (low, high)
        if band not in train_cache or band not in test_cache:
            continue
        try:
            csp = CSP(n_components=n_components, reg="oas", log=True, norm_trace=True)
            train_parts.append(csp.fit_transform(train_cache[band], y_train))
            test_parts.append(csp.transform(test_cache[band]))
            csp_models.append((csp, band))
        except (ValueError, np.linalg.LinAlgError):
            continue

    if not train_parts:
        return np.empty((n_train, 0)), np.empty((n_test, 0)), []

    return np.hstack(train_parts), np.hstack(test_parts), csp_models


class _FBCSPBase(ClassifierAdapter):
    """Shared FBCSP feature extraction for LDA/SVM/Ensemble adapters."""

    def __init__(self):
        self._fbcsp_train: np.ndarray | None = None
        self._fbcsp_test: np.ndarray | None = None
        self._csp_models: list[tuple[CSP, tuple[float, float]]] = []
        self._selector: SelectKBest | None = None
        self._scaler: StandardScaler | None = None
        self._sfreq: float = 250.0

    def _prepare_features(
        self,
        X_feat_train: np.ndarray,
        y_train: np.ndarray,
        X_mc_train: np.ndarray,
        sfreq: float,
        band_cache: BandCache | None = None,
        X_feat_test: np.ndarray | None = None,
        X_mc_test: np.ndarray | None = None,
        test_cache: BandCache | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Extract FBCSP, concat with handcrafted, select, scale."""
        self._sfreq = sfreq

        if X_mc_test is not None:
            fbcsp_train, fbcsp_test, self._csp_models = _extract_fbcsp_features(
                X_mc_train, X_mc_test, y_train, sfreq,
                train_cache=band_cache, test_cache=test_cache,
            )
            X_train_combined = np.hstack([fbcsp_train, X_feat_train])
            X_test_combined = np.hstack([fbcsp_test, X_feat_test])
        else:
            # Train-only mode (final model training)
            fbcsp_train, _, self._csp_models = _extract_fbcsp_features(
                X_mc_train, X_mc_train, y_train, sfreq,
                train_cache=band_cache, test_cache=band_cache,
            )
            X_train_combined = np.hstack([fbcsp_train, X_feat_train])
            X_test_combined = None

        k_values = _safe_k_values(
            n_features=X_train_combined.shape[1],
            n_train_trials=len(y_train),
        )
        if not k_values:
            k = min(10, X_train_combined.shape[1])
        else:
            k = k_values[-1]

        self._selector = SelectKBest(f_classif, k=k)
        X_train_sel = self._selector.fit_transform(X_train_combined, y_train)

        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train_sel)

        if X_test_combined is not None:
            X_test_sel = self._selector.transform(X_test_combined)
            X_test_scaled = self._scaler.transform(X_test_sel)
            return X_train_scaled, X_test_scaled

        return X_train_scaled, None

    def _base_export(self) -> dict[str, Any]:
        return {
            "csp_models": self._csp_models,
            "selector": self._selector,
            "scaler": self._scaler,
            "sfreq": self._sfreq,
        }


@register_classifier("lda")
class LDAClassifier(_FBCSPBase):
    name = "lda"

    def __init__(self):
        super().__init__()
        self._clf: LinearDiscriminantAnalysis | None = None
        self._X_test_scaled: np.ndarray | None = None

    def fit(self, X_feat_train, y_train, X_mc_train, sfreq, config=None, band_cache=None):
        X_train_scaled, _ = self._prepare_features(
            X_feat_train, y_train, X_mc_train, sfreq, band_cache=band_cache,
        )
        _, counts = np.unique(y_train, return_counts=True)
        priors = counts / counts.sum()
        self._clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors)
        self._clf.fit(X_train_scaled, y_train)

    def score(self, X_feat_test, y_test, X_mc_test):
        X_test_combined = self._transform_test(X_feat_test, X_mc_test)
        return float(self._clf.score(X_test_combined, y_test))

    def predict_proba(self, X_feat_test, X_mc_test):
        X_test_combined = self._transform_test(X_feat_test, X_mc_test)
        return self._clf.predict_proba(X_test_combined)

    def export(self):
        result = self._base_export()
        result["classifier"] = self._clf
        return result

    def _transform_test(self, X_feat_test, X_mc_test):
        import mne
        parts = []
        for csp, (low, high) in self._csp_models:
            X_filt = mne.filter.filter_data(X_mc_test, self._sfreq, low, high, verbose=False)
            parts.append(csp.transform(X_filt))
        fbcsp_test = np.hstack(parts) if parts else np.empty((X_mc_test.shape[0], 0))
        X_combined = np.hstack([fbcsp_test, X_feat_test])
        X_sel = self._selector.transform(X_combined)
        return self._scaler.transform(X_sel)


@register_classifier("riemann")
class RiemannianClassifier(ClassifierAdapter):
    name = "riemann"

    def __init__(self):
        self._pipeline = None
        self._sfreq: float = 250.0

    def fit(self, X_feat_train, y_train, X_mc_train, sfreq, config=None, band_cache=None):
        self._sfreq = sfreq
        self._pipeline = make_pipeline(
            Covariances(estimator="lwf"),
            TangentSpace(metric="riemann"),
            LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000),
        )
        self._pipeline.fit(X_mc_train, y_train)

    def score(self, X_feat_test, y_test, X_mc_test):
        return float(self._pipeline.score(X_mc_test, y_test))

    def predict_proba(self, X_feat_test, X_mc_test):
        return self._pipeline.predict_proba(X_mc_test)

    def export(self):
        return {"pipeline": self._pipeline, "sfreq": self._sfreq}


@register_classifier("svm")
class SVMClassifier(_FBCSPBase):
    name = "svm"

    def __init__(self):
        super().__init__()
        self._clf: SVC | None = None

    def fit(self, X_feat_train, y_train, X_mc_train, sfreq, config=None, band_cache=None):
        X_train_scaled, _ = self._prepare_features(
            X_feat_train, y_train, X_mc_train, sfreq, band_cache=band_cache,
        )
        self._clf = SVC(kernel="rbf", C=20.0, gamma="scale", probability=True, random_state=42)
        self._clf.fit(X_train_scaled, y_train)

    def score(self, X_feat_test, y_test, X_mc_test):
        X_test_combined = self._transform_test(X_feat_test, X_mc_test)
        return float(self._clf.score(X_test_combined, y_test))

    def predict_proba(self, X_feat_test, X_mc_test):
        X_test_combined = self._transform_test(X_feat_test, X_mc_test)
        return self._clf.predict_proba(X_test_combined)

    def export(self):
        result = self._base_export()
        result["classifier"] = self._clf
        return result

    def _transform_test(self, X_feat_test, X_mc_test):
        import mne
        parts = []
        for csp, (low, high) in self._csp_models:
            X_filt = mne.filter.filter_data(X_mc_test, self._sfreq, low, high, verbose=False)
            parts.append(csp.transform(X_filt))
        fbcsp_test = np.hstack(parts) if parts else np.empty((X_mc_test.shape[0], 0))
        X_combined = np.hstack([fbcsp_test, X_feat_test])
        X_sel = self._selector.transform(X_combined)
        return self._scaler.transform(X_sel)


@register_classifier("ensemble")
class EnsembleClassifier(ClassifierAdapter):
    """Soft-voting ensemble of LDA + SVM on shared FBCSP features."""

    name = "ensemble"

    def __init__(self):
        self._lda = LDAClassifier()
        self._svm = SVMClassifier()

    def fit(self, X_feat_train, y_train, X_mc_train, sfreq, config=None, band_cache=None):
        self._lda.fit(X_feat_train, y_train, X_mc_train, sfreq, config, band_cache)
        self._svm.fit(X_feat_train, y_train, X_mc_train, sfreq, config, band_cache)

    def score(self, X_feat_test, y_test, X_mc_test):
        proba = self.predict_proba(X_feat_test, X_mc_test)
        preds = np.argmax(proba, axis=1)
        return float(np.mean(preds == y_test))

    def predict_proba(self, X_feat_test, X_mc_test):
        proba_lda = self._lda.predict_proba(X_feat_test, X_mc_test)
        proba_svm = self._svm.predict_proba(X_feat_test, X_mc_test)
        return (proba_lda + proba_svm) / 2

    def export(self):
        return self._lda.export()
```

- [ ] **Step 4: Run new classifier tests**

Run: `uv run pytest tests/test_classifiers.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run all existing tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS (existing tests still use train.py's functions directly — no changes needed yet)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/classifiers.py tests/test_classifiers.py
uv run ruff format src/classifiers.py tests/test_classifiers.py
git add src/classifiers.py tests/test_classifiers.py
git commit -m "Add classifier registry with protocol-based adapters"
```

---

### Task 4: Create `src/experiments.py`

Unified experiment tracking with structured JSON and CLI.

**Files:**
- Create: `src/experiments.py`
- Create: `tests/test_experiments.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_experiments.py`:

```python
"""Tests for experiment tracking."""

import json
from pathlib import Path

import pytest

from src.experiments import (
    compare_experiments,
    list_experiments,
    load_experiment,
    save_experiment,
)


@pytest.fixture
def exp_dir(tmp_path):
    return tmp_path / "experiments"


class TestSaveLoad:
    def test_save_creates_json_file(self, exp_dir):
        save_experiment(
            name="test_baseline",
            results={"nested_mean": 0.589, "augmented_mean": 0.646},
            config_diff={},
            verdict="baseline",
            seed=42,
            experiments_dir=exp_dir,
        )
        files = list(exp_dir.glob("*.json"))
        assert len(files) == 1
        assert "test_baseline" in files[0].name

    def test_load_roundtrip(self, exp_dir):
        save_experiment(
            name="test_roundtrip",
            results={"nested_mean": 0.60},
            config_diff={"k_best": [10, 15]},
            verdict="neutral",
            seed=42,
            experiments_dir=exp_dir,
        )
        loaded = load_experiment("test_roundtrip", experiments_dir=exp_dir)
        assert loaded["name"] == "test_roundtrip"
        assert loaded["results"]["nested_mean"] == 0.60
        assert loaded["config_diff"]["k_best"] == [10, 15]
        assert loaded["verdict"] == "neutral"

    def test_load_nonexistent_raises(self, exp_dir):
        with pytest.raises(FileNotFoundError):
            load_experiment("nonexistent", experiments_dir=exp_dir)


class TestListExperiments:
    def test_list_empty_dir(self, exp_dir):
        result = list_experiments(experiments_dir=exp_dir)
        assert result == []

    def test_list_returns_summaries(self, exp_dir):
        save_experiment("exp_a", {"nested_mean": 0.55}, {}, "negative", 42, exp_dir)
        save_experiment("exp_b", {"nested_mean": 0.62}, {}, "positive", 42, exp_dir)
        result = list_experiments(experiments_dir=exp_dir)
        assert len(result) == 2
        names = {e["name"] for e in result}
        assert names == {"exp_a", "exp_b"}


class TestCompareExperiments:
    def test_compare_two_experiments(self, exp_dir):
        save_experiment("baseline", {"nested_mean": 0.589}, {}, "baseline", 42, exp_dir)
        save_experiment("new_idea", {"nested_mean": 0.610}, {"k_best": [10, 15]}, "positive", 42, exp_dir)
        comparison = compare_experiments("baseline", "new_idea", experiments_dir=exp_dir)
        assert "nested_mean" in comparison
        assert comparison["nested_mean"]["delta"] == pytest.approx(0.021, abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_experiments.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create `src/experiments.py`**

```python
"""Structured experiment tracking.

Each experiment run produces a JSON file in the experiments/ directory.
Provides save, load, list, and compare operations.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_EXPERIMENTS_DIR = Path("experiments")


def save_experiment(
    name: str,
    results: dict,
    config_diff: dict,
    verdict: str,
    seed: int,
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR,
    baseline_ref: str | None = None,
    seed_type: str = "single",
) -> Path:
    """Save an experiment result as structured JSON."""
    experiments_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filename = f"{date_prefix}_{name}.json"
    path = experiments_dir / filename

    data = {
        "name": name,
        "timestamp": timestamp,
        "seed": seed,
        "seed_type": seed_type,
        "config_diff": config_diff,
        "results": results,
        "verdict": verdict,
    }
    if baseline_ref is not None:
        data["baseline_ref"] = baseline_ref

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def load_experiment(
    name: str,
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR,
) -> dict:
    """Load an experiment by name (matches any file containing the name)."""
    if not experiments_dir.exists():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")

    matches = list(experiments_dir.glob(f"*{name}*.json"))
    if not matches:
        raise FileNotFoundError(f"No experiment matching {name!r} in {experiments_dir}")

    # Use the most recent if multiple matches
    path = sorted(matches)[-1]
    with open(path) as f:
        return json.load(f)


def list_experiments(
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR,
) -> list[dict]:
    """List all experiments as summary dicts."""
    if not experiments_dir.exists():
        return []

    summaries = []
    for path in sorted(experiments_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        summaries.append({
            "name": data["name"],
            "timestamp": data.get("timestamp", ""),
            "verdict": data.get("verdict", ""),
            "nested_mean": data.get("results", {}).get("nested_mean"),
            "file": path.name,
        })
    return summaries


def compare_experiments(
    name_a: str,
    name_b: str,
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR,
) -> dict:
    """Compare two experiments, returning per-metric deltas."""
    exp_a = load_experiment(name_a, experiments_dir)
    exp_b = load_experiment(name_b, experiments_dir)

    comparison = {}
    results_a = exp_a.get("results", {})
    results_b = exp_b.get("results", {})

    all_keys = set(results_a) | set(results_b)
    for key in sorted(all_keys):
        val_a = results_a.get(key)
        val_b = results_b.get(key)
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            comparison[key] = {
                "a": val_a,
                "b": val_b,
                "delta": val_b - val_a,
            }

    return comparison


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.experiments [list|compare <a> <b>]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "list":
        for exp in list_experiments():
            nested = exp.get("nested_mean")
            nested_str = f"{nested:.1%}" if nested is not None else "N/A"
            print(f"  {exp['name']:<30} {nested_str:>8}  [{exp['verdict']}]")
    elif command == "compare" and len(sys.argv) >= 4:
        result = compare_experiments(sys.argv[2], sys.argv[3])
        for key, vals in result.items():
            print(f"  {key:<20} {vals['a']:>8.3f} -> {vals['b']:>8.3f}  ({vals['delta']:+.3f})")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_experiments.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/experiments.py tests/test_experiments.py
uv run ruff format src/experiments.py tests/test_experiments.py
git add src/experiments.py tests/test_experiments.py
git commit -m "Add unified experiment tracking module"
```

---

### Task 5: Create `src/cli.py` and add `config_hash()` to TrainingConfig

CLI argument layer with RunConfig dataclass, separate from ML hyperparameters.

**Files:**
- Create: `src/cli.py`
- Create: `tests/test_cli.py`
- Modify: `src/config.py` (add `config_hash()`)

- [ ] **Step 1: Write failing tests**

Create `tests/test_cli.py`:

```python
"""Tests for CLI argument parsing and RunConfig."""

import pytest

from src.cli import RunConfig, parse_args
from src.config import TrainingConfig


class TestRunConfig:
    def test_defaults(self):
        rc = RunConfig()
        assert rc.stages == ["all"]
        assert rc.subjects is None
        assert rc.classifiers is None
        assert rc.experiment_name is None
        assert rc.export_models is True

    def test_custom_stages(self):
        rc = RunConfig(stages=["cv"])
        assert rc.stages == ["cv"]

    def test_custom_subjects(self):
        rc = RunConfig(subjects=["subject0001", "subject0003"])
        assert len(rc.subjects) == 2


class TestParseArgs:
    def test_no_args_gives_defaults(self):
        rc = parse_args([])
        assert rc.stages == ["all"]

    def test_stages_flag(self):
        rc = parse_args(["--stages", "cv,features"])
        assert rc.stages == ["cv", "features"]

    def test_subjects_flag(self):
        rc = parse_args(["--subjects", "0001,0003"])
        assert rc.subjects == ["0001", "0003"]

    def test_classifier_flag(self):
        rc = parse_args(["--classifier", "lda,svm"])
        assert rc.classifiers == ["lda", "svm"]

    def test_experiment_name_flag(self):
        rc = parse_args(["--experiment-name", "test_run"])
        assert rc.experiment_name == "test_run"

    def test_no_export_flag(self):
        rc = parse_args(["--no-export"])
        assert rc.export_models is False


class TestConfigHash:
    def test_same_config_same_hash(self):
        a = TrainingConfig()
        b = TrainingConfig()
        assert a.config_hash() == b.config_hash()

    def test_different_config_different_hash(self):
        a = TrainingConfig(k_best=10)
        b = TrainingConfig(k_best=15)
        assert a.config_hash() != b.config_hash()

    def test_hash_is_deterministic(self):
        cfg = TrainingConfig()
        assert cfg.config_hash() == cfg.config_hash()

    def test_hash_is_string(self):
        cfg = TrainingConfig()
        h = cfg.config_hash()
        assert isinstance(h, str)
        assert len(h) == 16  # truncated hex digest
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Add `config_hash()` to TrainingConfig in `src/config.py`**

Add to `src/config.py` after the `to_dict` method (line 43):

```python
    def config_hash(self) -> str:
        """Return a 16-char hex hash for cache invalidation."""
        import hashlib
        import json

        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
```

- [ ] **Step 4: Create `src/cli.py`**

```python
"""CLI argument parsing and execution configuration.

RunConfig controls WHAT to run (stages, subjects, classifiers).
TrainingConfig controls HOW to run (ML hyperparameters).
They are separate so TrainingConfig stays hashable for cache invalidation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    """Execution parameters for a pipeline run."""

    stages: list[str] = field(default_factory=lambda: ["all"])
    subjects: list[str] | None = None
    classifiers: list[str] | None = None
    experiment_name: str | None = None
    cache_dir: Path = Path("cache")
    export_models: bool = True


def parse_args(argv: list[str] | None = None) -> RunConfig:
    """Parse CLI arguments into a RunConfig."""
    parser = argparse.ArgumentParser(
        description="BCI Motor Imagery Training Pipeline"
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated stages to run (e.g., 'cv', 'features,cv')",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject IDs to include (e.g., '0001,0003')",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=None,
        help="Comma-separated classifiers to evaluate (e.g., 'lda,svm')",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment run",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Directory for stage caches",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip model export step",
    )

    args = parser.parse_args(argv)

    return RunConfig(
        stages=args.stages.split(","),
        subjects=args.subjects.split(",") if args.subjects else None,
        classifiers=args.classifier.split(",") if args.classifier else None,
        experiment_name=args.experiment_name,
        cache_dir=args.cache_dir,
        export_models=not args.no_export,
    )
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: ALL PASS

- [ ] **Step 6: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check --fix src/cli.py src/config.py tests/test_cli.py
uv run ruff format src/cli.py src/config.py tests/test_cli.py
git add src/cli.py src/config.py tests/test_cli.py
git commit -m "Add CLI layer with RunConfig and config_hash"
```

---

### Task 6: Update `.gitignore`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add cache and experiments directories**

Append to `.gitignore`:

```
# Pipeline cache (regenerated from data)
cache/

# Experiment results (tracked separately)
experiments/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "Add cache/ and experiments/ to .gitignore"
```

---

### Task 7: Verify train.py uses new modules correctly

After Tasks 1-2, train.py already imports from band_cache and rejection via backward-compatible aliases. This task verifies the wiring is correct and all tests pass.

**Important:** Do NOT remove FBCSP extraction functions (`_extract_fbcsp_features`, `_extract_fbcsp_features_prefiltered`, `_extract_fbcsp_features_pretrained`, `_safe_k_values`) from train.py. The classifier registry (Task 3) provides a *new parallel interface* — the existing CV orchestration functions still use train.py's FBCSP functions directly. Migrating the CV loops to use the registry is a future task.

**Files:**
- Verify: `src/train.py`

- [ ] **Step 1: Run all tests to verify wiring**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS — band_cache and rejection aliases work; classifiers.py is independent

- [ ] **Step 2: Verify import structure**

Run: `uv run python -c "from src.band_cache import precompute_bandpassed; from src.rejection import reject_in_fold; from src.classifiers import get_classifier; from src.experiments import list_experiments; from src.cli import RunConfig; print('All new modules importable')"`
Expected: "All new modules importable"

- [ ] **Step 3: Commit checkpoint**

```bash
git add -A
git commit -m "Checkpoint: verify all new modules integrate correctly"
```

---

### Task 8: Move `_augmented_nested_cv_subject` from pipeline.py to train.py

This function is CV orchestration logic that belongs in train.py, not pipeline.py.

**Files:**
- Modify: `src/train.py` (add function)
- Modify: `src/pipeline.py` (import from train.py instead of defining locally)

- [ ] **Step 1: Move the function**

Cut `_augmented_nested_cv_subject` from `src/pipeline.py` (lines 161-278) and paste it into `src/train.py` (at the end, before the `if __name__` block if present).

Update its internal imports — it currently does lazy imports from `src.train` (lines 182-187). Since it's now inside train.py, replace those with direct references:

```python
# Remove these lazy imports (they were needed because this was in pipeline.py):
# from src.train import ALL_CLASSIFIERS, _evaluate_classifier, _evaluate_classifiers_batch, _reject_in_fold
# from src.validation import make_cv_splits

# These are already available in train.py's scope
```

- [ ] **Step 2: Update pipeline.py to import from train.py**

In `src/pipeline.py`, replace the local `_augmented_nested_cv_subject` function with:

```python
from src.train import _augmented_nested_cv_subject
```

Add it to the existing `from src.train import (...)` block at the top.

- [ ] **Step 3: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff check --fix src/train.py src/pipeline.py
uv run ruff format src/train.py src/pipeline.py
git add src/train.py src/pipeline.py
git commit -m "Move _augmented_nested_cv_subject to train.py"
```

---

### Task 9: Move donor selection logic from pipeline.py to transfer.py

The Riemannian distance donor selection in pipeline.py (step 3b) is transfer learning logic.

**Files:**
- Modify: `src/transfer.py`
- Modify: `src/pipeline.py`
- Create: `tests/test_transfer_donor.py` (or add to existing test_transfer.py)

- [ ] **Step 1: Write failing test for donor selection**

Add to `tests/test_transfer.py`:

```python
def test_select_nearest_donor():
    """Test that select_nearest_donor returns the closest subject by Riemannian distance."""
    from src.transfer import select_nearest_donor

    rng = np.random.default_rng(42)
    # Create fake covariance-like data for 3 subjects
    # Subject A and B are similar, C is different
    base = np.eye(8)
    subjects = {
        "A": (rng.standard_normal((20, 10)), rng.standard_normal((20, 8, 375)) + 0.01, np.array([0]*10 + [1]*10)),
        "B": (rng.standard_normal((20, 10)), rng.standard_normal((20, 8, 375)) + 0.01, np.array([0]*10 + [1]*10)),
        "C": (rng.standard_normal((20, 10)), rng.standard_normal((20, 8, 375)) * 5, np.array([0]*10 + [1]*10)),
    }
    target_mc = rng.standard_normal((20, 8, 375)) + 0.01

    donor_id, distance = select_nearest_donor("target", target_mc, subjects)
    assert donor_id in ("A", "B", "C")
    assert isinstance(distance, float)
    assert distance > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transfer.py::test_select_nearest_donor -v`
Expected: FAIL — `ImportError: cannot import name 'select_nearest_donor'`

- [ ] **Step 3: Add `select_nearest_donor` to `src/transfer.py`**

Add at the end of `src/transfer.py`:

```python
def select_nearest_donor(
    target_id: str,
    target_mc: np.ndarray,
    donor_subjects: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[str, float]:
    """Select the nearest donor by Riemannian distance between covariance means.

    Args:
        target_id: Subject ID to exclude from donors.
        target_mc: Target multichannel EEG (n_trials, n_channels, n_samples).
        donor_subjects: Dict mapping subject_id -> (features, multichannel, labels).

    Returns:
        (best_donor_id, distance).
    """
    from pyriemann.estimation import Covariances
    from pyriemann.utils.distance import distance_riemann
    from pyriemann.utils.mean import mean_covariance

    cov_est = Covariances(estimator="lwf")
    target_mean = mean_covariance(cov_est.fit_transform(target_mc), metric="riemann")

    best_donor = None
    best_dist = float("inf")

    for sid, (_, mc, _) in donor_subjects.items():
        if sid == target_id:
            continue
        donor_mean = mean_covariance(cov_est.fit_transform(mc), metric="riemann")
        dist = float(distance_riemann(target_mean, donor_mean))
        if dist < best_dist:
            best_dist = dist
            best_donor = sid

    if best_donor is None:
        raise ValueError("No donors available (all excluded)")

    return best_donor, best_dist
```

- [ ] **Step 4: Update pipeline.py to use `select_nearest_donor`**

In `src/pipeline.py` step 3b (around lines 537-608), replace the inline Riemannian distance computation with:

```python
from src.transfer import select_nearest_donor
```

Replace the donor selection block (lines 558-584) with:

```python
best_donor, donor_dist = select_nearest_donor(
    sid, X_by_subject[i][1], donor_subjects
)
donor_feat, donor_mc, donor_y = donor_subjects[best_donor]
```

Remove the inline pyriemann imports (`_Cov`, `_dist_riemann`, `_mean_cov`) that were only used for donor selection.

- [ ] **Step 5: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/transfer.py src/pipeline.py tests/test_transfer.py
uv run ruff format src/transfer.py src/pipeline.py tests/test_transfer.py
git add src/transfer.py src/pipeline.py tests/test_transfer.py
git commit -m "Move donor selection logic to transfer.py"
```

---

### Task 10: Integration test for pipeline

Verify the full pipeline still works end-to-end with the refactored modules.

**Files:**
- Create: `tests/test_pipeline_integration.py`

- [ ] **Step 1: Write integration test**

Create `tests/test_pipeline_integration.py`:

```python
"""Integration tests for the refactored pipeline.

These tests verify that the module extractions (band_cache, rejection,
classifiers, experiments) work correctly when composed together through
the existing train.py orchestration functions.
"""

import numpy as np
import pytest

from src.band_cache import precompute_bandpassed, subset_band_cache
from src.classifiers import get_classifier, CLASSIFIER_REGISTRY
from src.rejection import reject_in_fold
from src.train import (
    train_within_subject_cv_all_models,
    train_nested_model_selection_cv,
    cross_session_evaluate,
)


N_TRIALS = 40
N_FEATURES = 10
N_CHANNELS = 8
N_SAMPLES = 375
SFREQ = 250.0


def _make_subject_data(n_trials: int = N_TRIALS, seed: int = 42):
    rng = np.random.default_rng(seed)
    X_feat = rng.standard_normal((n_trials, N_FEATURES))
    X_mc = rng.standard_normal((n_trials, N_CHANNELS, N_SAMPLES))
    y = np.array([0] * (n_trials // 2) + [1] * (n_trials // 2))
    return X_feat, X_mc, y


class TestBandCacheIntegration:
    def test_cache_subset_in_cv_fold(self):
        """Band cache subsetting works correctly in a CV fold context."""
        _, X_mc, y = _make_subject_data()
        cache = precompute_bandpassed(X_mc, SFREQ)

        train_idx = np.arange(30)
        test_idx = np.arange(30, 40)

        train_cache = subset_band_cache(cache, train_idx)
        test_cache = subset_band_cache(cache, test_idx)

        for band in cache:
            np.testing.assert_array_equal(
                train_cache[band], cache[band][train_idx]
            )
            np.testing.assert_array_equal(
                test_cache[band], cache[band][test_idx]
            )


class TestRejectionIntegration:
    def test_rejection_before_classifier(self):
        """Rejection + classifier pipeline works end-to-end."""
        X_feat, X_mc, y = _make_subject_data()
        ptps = np.abs(X_mc).max(axis=(1, 2))
        # Make one trial an outlier
        ptps[5] = ptps.max() * 100

        train_idx = np.arange(30)
        test_idx = np.arange(30, 40)
        clean_train, clean_test = reject_in_fold(ptps, train_idx, test_idx)

        assert 5 not in clean_train
        assert len(clean_train) < len(train_idx)


class TestClassifierIntegration:
    @pytest.mark.parametrize("name", ["lda", "riemann", "svm", "ensemble"])
    def test_classifier_fit_score(self, name):
        """Each registered classifier can fit and score."""
        X_feat, X_mc, y = _make_subject_data()
        clf = get_classifier(name)
        clf.fit(X_feat[:30], y[:30], X_mc[:30], SFREQ)
        score = clf.score(X_feat[30:], y[30:], X_mc[30:])
        assert 0.0 <= score <= 1.0


class TestCVIntegration:
    def test_all_models_cv_still_works(self):
        """train_within_subject_cv_all_models produces valid results after refactor."""
        X_feat, X_mc, y = _make_subject_data()
        results = train_within_subject_cv_all_models(
            [(X_feat, X_mc)],
            [y],
            sfreq=SFREQ,
            n_folds=3,
            enable_band_cache=True,
        )
        for name in ("lda", "riemann", "svm", "ensemble"):
            scores, mean, std = results[name]
            assert len(scores) == 1
            assert 0.0 <= mean <= 1.0

    def test_nested_cv_still_works(self):
        """train_nested_model_selection_cv produces valid results after refactor."""
        X_feat, X_mc, y = _make_subject_data()
        scores, mean, std, methods = train_nested_model_selection_cv(
            [(X_feat, X_mc)],
            [y],
            sfreq=SFREQ,
            n_outer_folds=3,
            n_inner_folds=2,
            enable_band_cache=True,
        )
        assert len(scores) == 1
        assert 0.0 <= mean <= 1.0
        assert len(methods) == 1
        assert methods[0] in ("lda", "riemann", "svm", "ensemble")


class TestNoLeakage:
    def test_random_data_below_70_percent(self):
        """Random data should not exceed 70% accuracy (leakage check)."""
        rng = np.random.default_rng(99)
        X_feat = rng.standard_normal((60, N_FEATURES))
        X_mc = rng.standard_normal((60, N_CHANNELS, N_SAMPLES))
        y = np.array([0] * 30 + [1] * 30)

        results = train_within_subject_cv_all_models(
            [(X_feat, X_mc)],
            [y],
            sfreq=SFREQ,
            n_folds=5,
            enable_band_cache=True,
        )
        for name in ("lda", "riemann", "svm", "ensemble"):
            _, mean, _ = results[name]
            assert mean < 0.70, f"{name} scored {mean:.1%} on random data — possible leakage"
```

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest tests/test_pipeline_integration.py -v`
Expected: ALL PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff check --fix tests/test_pipeline_integration.py
uv run ruff format tests/test_pipeline_integration.py
git add tests/test_pipeline_integration.py
git commit -m "Add integration tests for refactored pipeline modules"
```

---

### Task 11: Final cleanup — update test imports and remove compatibility shims

Once all tests pass with the new module structure, update test files to import from the canonical locations and remove backward-compatible aliases from train.py.

**Files:**
- Modify: `tests/test_train.py`
- Modify: `src/train.py` (remove shims)
- Modify: `src/pipeline.py` (update imports)

- [ ] **Step 1: Update test_train.py imports**

In `tests/test_train.py`, update the import block (lines 6-22):

Replace:
```python
from src.train import (
    _evaluate_classifiers_batch,
    _precompute_bandpassed,
    _reject_in_fold,
    ...
)
```

With:
```python
from src.band_cache import precompute_bandpassed
from src.rejection import reject_in_fold
from src.train import (
    _evaluate_classifiers_batch,
    cross_session_evaluate,
    predict,
    predict_riemann,
    train_final_model,
    train_final_model_riemann,
    train_final_model_svm,
    train_nested_model_selection_cv,
    train_within_subject_cv_all_models,
    train_within_subject_cv,
    train_within_subject_cv_ensemble,
    train_within_subject_cv_riemann,
    train_within_subject_cv_svm,
)
```

Then find-and-replace throughout the file:
- `_precompute_bandpassed(` → `precompute_bandpassed(`
- `_reject_in_fold(` → `reject_in_fold(`

- [ ] **Step 2: Update pipeline.py imports**

In `src/pipeline.py`, update imports to use canonical module names:

```python
from src.rejection import reject_in_fold
from src.band_cache import precompute_bandpassed, subset_band_cache
```

Replace any remaining `_reject_in_fold` calls with `reject_in_fold`.

- [ ] **Step 3: Remove backward-compatible aliases from train.py**

In `src/train.py`, remove:
```python
# Backward-compatible aliases
_precompute_bandpassed = precompute_bandpassed
_subset_band_cache = subset_band_cache
_reject_in_fold = reject_in_fold
```

Keep `_evaluate_classifiers_batch` and `_evaluate_classifier` as-is in train.py — they're part of the CV orchestration and still used by pipeline.py and tests.

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/train.py src/pipeline.py tests/test_train.py
uv run ruff format src/train.py src/pipeline.py tests/test_train.py
git add src/train.py src/pipeline.py tests/test_train.py
git commit -m "Remove compatibility shims, use canonical imports"
```

---

### Task 12: Update README.md

Update documentation to reflect the new module structure.

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the project structure section**

Add the new modules to the README's file listing. Update any references to train.py's responsibilities to note the new modules. Mention the experiment tracking CLI and partial pipeline runs.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Update README with new module structure"
```

---

## Deferred to Follow-Up Plan

The following spec sections require the module extractions in this plan to be completed first. They will be addressed in a separate implementation plan:

1. **Stage-based pipeline caching (spec 1.3):** NPZ file caching for preprocessed data, epochs, features, and band-filtered tensors. Requires the cache hash infrastructure (Task 5's `config_hash()`) and the extracted modules to be stable.

2. **Full pipeline.py slimming to ~350 lines (spec 1.4):** Refactoring `run_pipeline()` into a stage coordinator that uses `RunConfig` flags (`--stages`, `--subjects`, `--classifier`). Requires the CLI layer (Task 5) and experiment tracking (Task 4) to be integrated into the pipeline flow.

3. **Migrating CV orchestration to use classifier registry:** The existing CV loops in train.py (`_evaluate_subject_all_models`, `_evaluate_classifiers_batch`, etc.) still use inline classifier construction. Migrating them to use the registry from classifiers.py is a separate effort that can happen incrementally.

These are intentionally deferred — this plan focuses on the module extractions and new interfaces. The follow-up plan will wire them into the pipeline.
