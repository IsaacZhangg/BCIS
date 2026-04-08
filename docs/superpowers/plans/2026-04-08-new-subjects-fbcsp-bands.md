# New Subjects Integration & FBCSP Band Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate MI_DATA_NEW subjects (104, 106) into the main 4-classifier evaluation pipeline and add subject-specific FBCSP band optimization via inner CV selection.

**Architecture:** Extend the data loader with a flexible discovery function, refactor the pipeline to load MI_DATA_NEW subjects at the subject level (multi-recording merge), and add a band candidate dimension to the nested CV inner loop. Band selection propagates to final model training.

**Tech Stack:** Python, scikit-learn, MNE, pyriemann (all existing dependencies — no new ones)

**Spec:** `docs/superpowers/specs/2026-04-08-new-subjects-pipeline-improvements-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/data_loader.py` | Modify | Add `get_recordings_flexible` for MI_DATA_NEW discovery |
| `tests/test_data_loader.py` | Modify | Tests for flexible loader |
| `src/config.py` | Modify | Add `fbcsp_band_candidates` to `TrainingConfig` |
| `src/train.py` | Modify | Add `FBCSP_BAND_CANDIDATES`, band-aware `_evaluate_classifiers_batch`, nested CV band selection, propagate to final models |
| `tests/test_train.py` | Modify | Tests for band candidates and band-aware evaluation |
| `src/pipeline.py` | Modify | Load MI_DATA_NEW subjects into main pipeline, pass band config through nested CV and final model training |

---

### Task 1: Flexible Recording Discovery in data_loader.py

**Files:**
- Modify: `src/data_loader.py`
- Modify: `tests/test_data_loader.py`

- [ ] **Step 1: Write failing test for `get_recordings_flexible`**

```python
# tests/test_data_loader.py — add at end of file

def test_get_recordings_flexible_finds_mi_data_new():
    """Flexible loader finds MI_DATA_NEW recordings with >= 20 phase-3 trials."""
    from src.data_loader import get_recordings_flexible

    data_dir = Path("Data/MI_DATA_NEW")
    grouped = get_recordings_flexible(data_dir, min_trials=20)

    # subject0104 has 2 sessions with 32 trials each
    assert "subject0104" in grouped
    assert len(grouped["subject0104"]) == 2

    # subject0106 has 1 session with 32 trials
    assert "subject0106" in grouped
    assert len(grouped["subject0106"]) == 1

    # subject0105 has 0 trials — should be excluded
    assert "subject0105" not in grouped


def test_get_recordings_flexible_respects_min_trials():
    """Recordings below min_trials are excluded."""
    from src.data_loader import get_recordings_flexible

    data_dir = Path("Data/MI_DATA_NEW")
    # With threshold of 50, subject0106 (32 trials) should be excluded
    grouped = get_recordings_flexible(data_dir, min_trials=50)
    assert "subject0106" not in grouped
    # subject0104 individual sessions have 32 trials each — also excluded
    assert "subject0104" not in grouped
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data_loader.py::test_get_recordings_flexible_finds_mi_data_new tests/test_data_loader.py::test_get_recordings_flexible_respects_min_trials -v`
Expected: FAIL with `ImportError: cannot import name 'get_recordings_flexible'`

- [ ] **Step 3: Implement `get_recordings_flexible`**

Add to `src/data_loader.py` after `get_recordings_by_subject`:

```python
def get_recordings_flexible(
    data_dir: Path,
    min_trials: int = 20,
) -> dict[str, list[Path]]:
    """Find recordings with >= min_trials phase-3 events, grouped by subject.

    Unlike get_complete_recordings (which requires exactly 100 trials), this
    accepts any recording with sufficient phase-3 events — needed for MI_DATA_NEW
    recordings that have 32 trials each.
    """
    grouped: dict[str, list[Path]] = {}
    for csv_path in sorted(data_dir.glob("subject*/session*/*.csv")):
        stim = pd.read_csv(csv_path, usecols=["stim"])["stim"].to_numpy(copy=False)
        nonzero = stim[stim != 0].astype(int)
        phase3_count = int(np.sum((nonzero // 10) % 10 == 3))
        if phase3_count >= min_trials:
            subject_id = csv_path.parent.parent.name
            grouped.setdefault(subject_id, []).append(csv_path)
    return grouped
```

Also add `import pandas as pd` at the top of `data_loader.py` (it's already imported — verify).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data_loader.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_loader.py tests/test_data_loader.py
git commit -m "Add get_recordings_flexible for MI_DATA_NEW discovery"
```

---

### Task 2: Add FBCSP Band Candidates to Config and Train

**Files:**
- Modify: `src/config.py`
- Modify: `src/train.py`

- [ ] **Step 1: Add `FBCSP_BAND_CANDIDATES` to `train.py`**

Add after the existing `FBCSP_BANDS` constant (line 32):

```python
FBCSP_BAND_CANDIDATES: dict[str, list[tuple[float, float]]] = {
    "standard": [
        (8, 10), (10, 12), (12, 14), (14, 16),
        (16, 18), (18, 20), (20, 24), (24, 30),
    ],
    "high_mu": [
        (9, 11), (11, 13), (13, 15), (15, 18),
        (18, 22), (22, 26), (26, 30),
    ],
    "wide_mu": [
        (8, 12), (10, 14), (12, 16),
        (16, 20), (20, 24), (24, 30),
    ],
}
```

- [ ] **Step 2: Add `fbcsp_band_candidates` to `TrainingConfig`**

Add field to `TrainingConfig` in `src/config.py` (after `augmentation_weakness_threshold`):

```python
    fbcsp_band_candidates: tuple[str, ...] = ("standard", "high_mu", "wide_mu")
```

- [ ] **Step 3: Run existing tests to verify nothing breaks**

Run: `uv run pytest tests/ -v`
Expected: All existing tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/config.py src/train.py
git commit -m "Add FBCSP band candidate configurations"
```

---

### Task 3: Band-Aware FBCSP Feature Extraction

**Files:**
- Modify: `src/train.py`
- Modify: `tests/test_train.py`

The key change: `_evaluate_classifiers_batch` and `_evaluate_classifier` need to accept a `bands` parameter so the nested CV can evaluate different band configurations.

- [ ] **Step 1: Write failing test for band-aware evaluation**

Add to `tests/test_train.py`:

```python
def test_evaluate_classifiers_batch_with_custom_bands():
    """_evaluate_classifiers_batch accepts a custom bands parameter."""
    from src.train import FBCSP_BAND_CANDIDATES

    rng = np.random.default_rng(99)
    X_feat_train = rng.standard_normal((30, N_FEATURES))
    X_feat_test = rng.standard_normal((10, N_FEATURES))
    X_mc_train = rng.standard_normal((30, N_CHANNELS, N_SAMPLES))
    X_mc_test = rng.standard_normal((10, N_CHANNELS, N_SAMPLES))
    y_train = np.array([0] * 15 + [1] * 15)
    y_test = np.array([0] * 5 + [1] * 5)

    # Should work with each band candidate
    for band_name, bands in FBCSP_BAND_CANDIDATES.items():
        scores = _evaluate_classifiers_batch(
            ["lda", "svm"],
            X_feat_train, X_feat_test,
            y_train, y_test,
            X_mc_train, X_mc_test,
            sfreq=250.0,
            k_best=10,
            bands=bands,
        )
        assert "lda" in scores
        assert "svm" in scores
        assert 0 <= scores["lda"] <= 1
        assert 0 <= scores["svm"] <= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py::test_evaluate_classifiers_batch_with_custom_bands -v`
Expected: FAIL with `TypeError: _evaluate_classifiers_batch() got an unexpected keyword argument 'bands'`

- [ ] **Step 3: Add `bands` parameter to `_evaluate_classifiers_batch`**

Modify the function signature at `src/train.py:961`:

```python
def _evaluate_classifiers_batch(
    names: list[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_mc_train: np.ndarray,
    X_mc_test: np.ndarray,
    sfreq: float,
    k_best: int,
    prefiltered_train: BandCache | None = None,
    prefiltered_test: BandCache | None = None,
    bands: list[tuple[float, float]] | None = None,
) -> dict[str, float]:
```

Then where FBCSP features are extracted (around line 1001-1012), pass the bands through:

```python
    fbcsp_names = requested.intersection({"lda", "svm", "ensemble"})
    if not fbcsp_names:
        return scores

    effective_bands = bands if bands is not None else FBCSP_BANDS

    if prefiltered_train is not None and prefiltered_test is not None:
        fbcsp_train, fbcsp_test, _ = _extract_fbcsp_features_prefiltered(
            prefiltered_train,
            prefiltered_test,
            y_train,
            bands=effective_bands,
            n_train_trials=X_mc_train.shape[0],
            n_test_trials=X_mc_test.shape[0],
        )
    else:
        fbcsp_train, fbcsp_test, _ = _extract_fbcsp_features(
            X_mc_train, X_mc_test, y_train, sfreq, bands=effective_bands
        )
```

- [ ] **Step 4: Add `bands` parameter to `_evaluate_classifier`**

Modify `_evaluate_classifier` at `src/train.py:1079` to pass through `bands`:

```python
def _evaluate_classifier(
    name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_mc_train: np.ndarray,
    X_mc_test: np.ndarray,
    sfreq: float,
    k_best: int,
    prefiltered_train: BandCache | None = None,
    prefiltered_test: BandCache | None = None,
    bands: list[tuple[float, float]] | None = None,
) -> float:
    """Train and score a single classifier on pre-split data."""
    scores = _evaluate_classifiers_batch(
        [name],
        X_train, X_test, y_train, y_test,
        X_mc_train, X_mc_test, sfreq, k_best,
        prefiltered_train=prefiltered_train,
        prefiltered_test=prefiltered_test,
        bands=bands,
    )
    return scores[name]
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_train.py -v`
Expected: All tests PASS (existing tests use default bands=None, new test passes with explicit bands)

- [ ] **Step 6: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "Add bands parameter to classifier evaluation functions"
```

---

### Task 4: Band Selection in Nested CV Inner Loop

**Files:**
- Modify: `src/train.py` (function `_evaluate_subject_nested_model_selection`)
- Modify: `tests/test_train.py`

This is the core change: the inner CV loop now also iterates over band candidates for FBCSP-based classifiers, selecting the best (classifier, band_config) pair.

- [ ] **Step 1: Write failing test for band selection in nested CV**

Add to `tests/test_train.py`:

```python
def test_nested_cv_returns_band_config():
    """Nested CV returns per-subject selected band config name."""
    from src.train import FBCSP_BAND_CANDIDATES

    n_subjects = 2
    X_by_subject = [
        _make_subject_data(np.random.default_rng(42 + i)) for i in range(n_subjects)
    ]
    y_by_subject = [LABELS for _ in range(n_subjects)]

    scores, mean_acc, std_acc, methods, band_configs = train_nested_model_selection_cv(
        X_by_subject, y_by_subject,
        n_outer_folds=3, n_inner_folds=3,
        band_candidates=FBCSP_BAND_CANDIDATES,
    )

    assert len(band_configs) == n_subjects
    for bc in band_configs:
        assert bc in FBCSP_BAND_CANDIDATES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py::test_nested_cv_returns_band_config -v`
Expected: FAIL with `ValueError: too many values to unpack` (returns 4 values, we expect 5)

- [ ] **Step 3: Modify `_evaluate_subject_nested_model_selection` to search over band candidates**

In `src/train.py`, modify `_evaluate_subject_nested_model_selection` (starts at line 1109). The key changes:

1. Accept `band_candidates: dict[str, list[tuple[float, float]]] | None = None`
2. Precompute band cache for the **union** of all candidate bands
3. In the inner CV loop, for each inner split, evaluate every (classifier, band_config) pair
4. Return the winning band config name alongside the winning classifier

```python
def _evaluate_subject_nested_model_selection(
    X_features: np.ndarray,
    X_multichannel: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    n_outer_folds: int,
    n_inner_folds: int,
    k_best: int,
    trial_ptps: np.ndarray | None,
    split_strategy: SplitStrategy,
    groups: np.ndarray | None,
    random_state: int,
    enable_band_cache: bool,
    cache_scope: CacheScope,
    max_blas_threads_per_worker: int,
    band_candidates: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[float, str, str]:
    """Nested model-selection CV for a single subject.

    Returns (score, best_classifier_method, best_band_config_name).
    """
    classifier_names = list(ALL_CLASSIFIERS)

    # Resolve band candidates
    if band_candidates is None:
        effective_candidates = {"standard": FBCSP_BANDS}
    else:
        effective_candidates = band_candidates

    with threadpool_limits(limits=max_blas_threads_per_worker or None):
        # Precompute band cache for the union of all candidate bands
        all_bands = sorted(
            {band for bands in effective_candidates.values() for band in bands}
        )
        subject_band_cache = (
            _precompute_bandpassed(X_multichannel, sfreq, bands=all_bands)
            if enable_band_cache and cache_scope == "subject"
            else None
        )

        outer_splits = make_cv_splits(
            y,
            n_splits=n_outer_folds,
            strategy=split_strategy,
            groups=groups,
            random_state=random_state,
        )
        fold_scores: list[float] = []
        fold_winners: list[str] = []
        fold_band_winners: list[str] = []

        for outer_train_idx, outer_test_idx in outer_splits:
            outer_train_idx, outer_test_idx = _reject_in_fold(
                trial_ptps, outer_train_idx, outer_test_idx
            )
            if len(outer_train_idx) < 2 or len(outer_test_idx) < 1:
                continue

            X_feat_otrain = X_features[outer_train_idx]
            X_feat_otest = X_features[outer_test_idx]
            X_mc_otrain = X_multichannel[outer_train_idx]
            X_mc_otest = X_multichannel[outer_test_idx]
            y_otrain = y[outer_train_idx]
            y_otest = y[outer_test_idx]

            # Build per-fold band caches from subject cache or fresh
            outer_train_cache: BandCache | None = None
            outer_test_cache: BandCache | None = None
            if enable_band_cache and subject_band_cache is not None:
                outer_train_cache = _subset_band_cache(
                    subject_band_cache, outer_train_idx
                )
                outer_test_cache = _subset_band_cache(
                    subject_band_cache, outer_test_idx
                )
            elif enable_band_cache:
                outer_train_cache = _precompute_bandpassed(
                    X_mc_otrain, sfreq, bands=all_bands
                )
                outer_test_cache = _precompute_bandpassed(
                    X_mc_otest, sfreq, bands=all_bands
                )

            inner_groups = groups[outer_train_idx] if groups is not None else None
            inner_splits = make_cv_splits(
                y_otrain,
                n_splits=n_inner_folds,
                strategy=split_strategy,
                groups=inner_groups,
                random_state=random_state,
            )

            # Score every (classifier, band_config) pair in inner CV
            # Riemann doesn't use FBCSP bands, so score it once
            riemann_inner_scores: list[float] = []
            # (clf_name, band_name) -> list of inner fold scores
            fbcsp_inner_scores: dict[tuple[str, str], list[float]] = {}
            for clf_name in classifier_names:
                if clf_name == "riemann":
                    continue
                for band_name in effective_candidates:
                    fbcsp_inner_scores[(clf_name, band_name)] = []

            for inner_train_idx, inner_val_idx in inner_splits:
                # Riemann: evaluate once (band-independent)
                riemann_score = _evaluate_classifiers_batch(
                    ["riemann"],
                    X_feat_otrain[inner_train_idx],
                    X_feat_otrain[inner_val_idx],
                    y_otrain[inner_train_idx],
                    y_otrain[inner_val_idx],
                    X_mc_otrain[inner_train_idx],
                    X_mc_otrain[inner_val_idx],
                    sfreq,
                    k_best,
                )
                riemann_inner_scores.append(riemann_score["riemann"])

                # FBCSP classifiers: evaluate per band config
                for band_name, bands in effective_candidates.items():
                    inner_train_band_cache: BandCache | None = None
                    inner_val_band_cache: BandCache | None = None
                    if outer_train_cache is not None:
                        inner_train_band_cache = _subset_band_cache(
                            outer_train_cache, inner_train_idx
                        )
                        inner_val_band_cache = _subset_band_cache(
                            outer_train_cache, inner_val_idx
                        )
                    fbcsp_clf_names = [n for n in classifier_names if n != "riemann"]
                    split_scores = _evaluate_classifiers_batch(
                        fbcsp_clf_names,
                        X_feat_otrain[inner_train_idx],
                        X_feat_otrain[inner_val_idx],
                        y_otrain[inner_train_idx],
                        y_otrain[inner_val_idx],
                        X_mc_otrain[inner_train_idx],
                        X_mc_otrain[inner_val_idx],
                        sfreq,
                        k_best,
                        prefiltered_train=inner_train_band_cache,
                        prefiltered_test=inner_val_band_cache,
                        bands=bands,
                    )
                    for clf_name in fbcsp_clf_names:
                        fbcsp_inner_scores[(clf_name, band_name)].append(
                            split_scores[clf_name]
                        )

            # Find the best (classifier, band) pair
            riemann_mean = (
                float(np.mean(riemann_inner_scores))
                if riemann_inner_scores
                else 0.5
            )
            best_score = riemann_mean
            best_clf = "riemann"
            best_band = "standard"

            for (clf_name, band_name), scores_list in fbcsp_inner_scores.items():
                mean_score = float(np.mean(scores_list)) if scores_list else 0.5
                if mean_score > best_score or (
                    mean_score == best_score and band_name == "standard"
                ):
                    best_score = mean_score
                    best_clf = clf_name
                    best_band = band_name

            # Tie-break: prefer "standard" band config
            if best_clf != "riemann" and best_band != "standard":
                standard_score = float(
                    np.mean(fbcsp_inner_scores.get((best_clf, "standard"), [0.5]))
                )
                if abs(best_score - standard_score) < 1e-10:
                    best_band = "standard"

            # Retrain best on outer train, test on outer test
            outer_bands = (
                effective_candidates[best_band] if best_clf != "riemann" else None
            )
            outer_score = _evaluate_classifier(
                best_clf,
                X_feat_otrain, X_feat_otest,
                y_otrain, y_otest,
                X_mc_otrain, X_mc_otest,
                sfreq, k_best,
                prefiltered_train=outer_train_cache,
                prefiltered_test=outer_test_cache,
                bands=outer_bands,
            )
            fold_scores.append(outer_score)
            fold_winners.append(best_clf)
            fold_band_winners.append(best_band)

        score = float(np.mean(fold_scores)) if fold_scores else 0.5
        winner_counts = Counter(fold_winners)
        best_method = winner_counts.most_common(1)[0][0] if winner_counts else "lda"
        band_counts = Counter(fold_band_winners)
        best_band_config = (
            band_counts.most_common(1)[0][0] if band_counts else "standard"
        )
        return score, best_method, best_band_config
```

- [ ] **Step 4: Update `train_nested_model_selection_cv` to pass `band_candidates` and return band configs**

Modify the public function at `src/train.py:1233`:

```python
def train_nested_model_selection_cv(
    X_by_subject: list[tuple[np.ndarray, np.ndarray]],
    y_by_subject: list[np.ndarray],
    sfreq: float = 250.0,
    n_outer_folds: int = 10,
    n_inner_folds: int = 5,
    k_best: int = 10,
    trial_ptps_by_subject: list[np.ndarray] | None = None,
    split_strategy: SplitStrategy = "stratified_group",
    trial_groups_by_subject: list[np.ndarray] | None = None,
    trial_group_size: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "loky",
    max_blas_threads_per_worker: int = 1,
    enable_band_cache: bool = True,
    cache_scope: CacheScope = "subject",
    band_candidates: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[list[float], float, float, list[str], list[str]]:
```

Update the return to include band configs:

```python
    scores = [score for score, _, _ in per_subject_results]
    best_methods = [method for _, method, _ in per_subject_results]
    best_band_configs = [band for _, _, band in per_subject_results]

    mean_acc = float(np.mean(scores))
    std_acc = float(np.std(scores))
    return scores, mean_acc, std_acc, best_methods, best_band_configs
```

Pass `band_candidates=band_candidates` to each `_evaluate_subject_nested_model_selection` call in both the sequential and parallel paths.

- [ ] **Step 5: Fix all callers that unpack 4 values to unpack 5**

In `src/pipeline.py:485`, the call site unpacks 4 values:
```python
    nested_scores, nested_mean, nested_std, nested_methods = (
        train_nested_model_selection_cv(...)
    )
```
Change to:
```python
    nested_scores, nested_mean, nested_std, nested_methods, nested_band_configs = (
        train_nested_model_selection_cv(...)
    )
```

In `src/pipeline.py`, the augmented nested CV function `_augmented_nested_cv_subject` also calls `_evaluate_classifiers_batch` and `_evaluate_classifier` — add `bands=None` as their default keeps them working.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_train.py -v`
Expected: All tests PASS including the new `test_nested_cv_returns_band_config`

- [ ] **Step 7: Commit**

```bash
git add src/train.py src/pipeline.py tests/test_train.py
git commit -m "Add FBCSP band selection to nested CV inner loop"
```

---

### Task 5: Propagate Band Config to Final Model Training

**Files:**
- Modify: `src/train.py` (`train_final_model`, `train_final_model_svm`)
- Modify: `src/pipeline.py` (Step 5 model saving)
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write failing test for band-aware final model**

Add to `tests/test_train.py`:

```python
def test_train_final_model_with_custom_bands():
    """train_final_model accepts a bands parameter."""
    from src.train import FBCSP_BAND_CANDIDATES

    rng = np.random.default_rng(77)
    X_feat, X_mc = _make_subject_data(rng)
    y = LABELS

    bands = FBCSP_BAND_CANDIDATES["high_mu"]
    model = train_final_model(X_feat, X_mc, y, bands=bands)

    assert "bands" in model
    assert model["bands"] == bands

    preds = predict(model, X_feat, X_mc)
    assert len(preds) == len(y)


def test_train_final_model_svm_with_custom_bands():
    """train_final_model_svm accepts a bands parameter."""
    from src.train import FBCSP_BAND_CANDIDATES

    rng = np.random.default_rng(78)
    X_feat, X_mc = _make_subject_data(rng)
    y = LABELS

    bands = FBCSP_BAND_CANDIDATES["wide_mu"]
    model = train_final_model_svm(X_feat, X_mc, y, bands=bands)

    assert "bands" in model
    assert model["bands"] == bands
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train.py::test_train_final_model_with_custom_bands tests/test_train.py::test_train_final_model_svm_with_custom_bands -v`
Expected: FAIL with `TypeError: train_final_model() got an unexpected keyword argument 'bands'`

- [ ] **Step 3: Add `bands` parameter to `train_final_model` and `train_final_model_svm`**

Modify `train_final_model` at `src/train.py:666`:

```python
def train_final_model(
    X_features: np.ndarray,
    X_multichannel: np.ndarray,
    y: np.ndarray,
    sfreq: float = 250.0,
    k_best: int = 10,
    bands: list[tuple[float, float]] | None = None,
) -> dict:
    effective_bands = bands if bands is not None else FBCSP_BANDS
    fbcsp_features, _, csp_models = _extract_fbcsp_features(
        X_multichannel, X_multichannel, y, sfreq, bands=effective_bands
    )
    # ... rest unchanged ...
    return {
        "csp_models": csp_models,
        "selector": selector,
        "scaler": scaler,
        "classifier": classifier,
        "sfreq": sfreq,
        "k_best": k,
        "bands": effective_bands,
    }
```

Same pattern for `train_final_model_svm` at `src/train.py:1394`.

The `predict` function already uses `model["csp_models"]` which stores the fitted CSP objects with their bands, so no change needed there.

- [ ] **Step 4: Update pipeline Step 5 to pass selected bands**

In `src/pipeline.py` (Step 5, around line 718), where final models are trained, look up the per-subject band config and pass it:

```python
    for i, (sid, (X_features, X_multichannel), y, method) in enumerate(
        zip(subject_ids, X_by_subject, y_by_subject, nested_methods_display)
    ):
        # ... artifact rejection code unchanged ...

        # Look up the selected band config for this subject
        selected_band_name = nested_band_configs[i] if nested_band_configs else "standard"
        from src.train import FBCSP_BAND_CANDIDATES
        selected_bands = FBCSP_BAND_CANDIDATES.get(selected_band_name, FBCSP_BANDS)

        if method == "SVM":
            model = train_final_model_svm(
                X_features, X_multichannel, y, sfreq=sfreq, k_best=cfg.k_best,
                bands=selected_bands,
            )
        elif method == "Riemann":
            model = train_final_model_riemann(X_multichannel, y, sfreq=sfreq)
        elif method == "Ensemble":
            model = train_final_model(
                X_features, X_multichannel, y, sfreq=sfreq, k_best=cfg.k_best,
                bands=selected_bands,
            )
        else:
            model = train_final_model(
                X_features, X_multichannel, y, sfreq=sfreq, k_best=cfg.k_best,
                bands=selected_bands,
            )
```

- [ ] **Step 5: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/train.py src/pipeline.py tests/test_train.py
git commit -m "Propagate selected FBCSP band config to final model training"
```

---

### Task 6: Integrate MI_DATA_NEW Subjects into Main Pipeline

**Files:**
- Modify: `src/pipeline.py`

This is the main pipeline refactoring: after loading unicorn-data subjects (existing per-recording loop), load MI_DATA_NEW subjects via subject-level aggregation.

- [ ] **Step 1: Add MI_DATA_NEW subject loading to `run_pipeline`**

After the existing recording loop (around `src/pipeline.py:440`, after `runtime_seconds["load_preprocess"]`), add a new block to load MI_DATA_NEW subjects:

```python
    # Load additional subjects from MI_DATA_NEW (subject-level merge)
    if MI_DATA_NEW_DIR.exists():
        from src.data_loader import get_recordings_flexible

        mi_new_grouped = get_recordings_flexible(MI_DATA_NEW_DIR, min_trials=20)

        # Apply subject merge mapping
        merged_grouped: dict[str, list[Path]] = {}
        for sid, paths in mi_new_grouped.items():
            canonical = DEFAULT_SUBJECT_MERGE.get(sid, sid)
            merged_grouped.setdefault(canonical, []).extend(paths)

        # Skip subjects already loaded from unicorn-data and empty subjects
        for sid in sorted(merged_grouped.keys()):
            if sid in subject_ids:
                continue

            rec_paths = merged_grouped[sid]
            print(f"  Processing {sid} (MI_DATA_NEW, {len(rec_paths)} recording(s))...")

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

                # Per-recording artifact rejection
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
                print(f"    Skipping {sid} — no clean epochs")
                continue

            print(f"    Epochs: {n_left}L/{n_right}R (merged from {len(rec_paths)} recording(s))")

            X_feat, X_mc, y = _pairs_to_features(all_left, all_right, sfreq)
            trial_ptps = compute_trial_max_ptp(X_mc)

            X_by_subject.append((X_feat, X_mc))
            y_by_subject.append(y)
            subject_ids.append(sid)
            trial_ptps_by_subject.append(trial_ptps)
            trial_groups_by_subject.append(
                build_classwise_trial_groups(y, group_size=cfg.trial_group_size)
            )
```

- [ ] **Step 2: Pass band candidates from config into nested CV call**

At the `train_nested_model_selection_cv` call site (~line 485), add:

```python
    from src.train import FBCSP_BAND_CANDIDATES

    nested_scores, nested_mean, nested_std, nested_methods, nested_band_configs = (
        train_nested_model_selection_cv(
            X_by_subject, y_by_subject,
            sfreq=sfreq,
            n_outer_folds=cfg.n_outer_folds,
            n_inner_folds=cfg.n_inner_folds,
            k_best=cfg.k_best,
            trial_ptps_by_subject=trial_ptps_by_subject,
            split_strategy=cfg.split_strategy,
            trial_groups_by_subject=trial_groups_by_subject,
            trial_group_size=cfg.trial_group_size,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            parallel_backend=cfg.parallel_backend,
            max_blas_threads_per_worker=cfg.max_blas_threads_per_worker,
            enable_band_cache=cfg.enable_band_cache,
            cache_scope=cfg.cache_scope,
            band_candidates=FBCSP_BAND_CANDIDATES,
        )
    )
```

- [ ] **Step 3: Add band config to results output**

In the results dict construction (~line 773), add:

```python
        "nested_band_configs": {
            sid: bc for sid, bc in zip(subject_ids, nested_band_configs)
        },
```

And in the per-subject results table printing, include the band config:

```python
        print(
            f"  {sid:<12} {fs:>9.1%} {rs:>9.1%} {ss:>9.1%} {es:>9.1%} "
            f"{bs:>9.1%} {ns:>9.1%}  [{nm}, {bc}, {status}]"
        )
```

where `bc` comes from `nested_band_configs`.

- [ ] **Step 4: Run the full pipeline to verify it works**

Run: `uv run python -m src.pipeline`
Expected: Pipeline completes, shows ~15 subjects in the results table, band configs reported per subject.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py
git commit -m "Integrate MI_DATA_NEW subjects into main pipeline with band selection"
```

---

### Task 7: Run Leakage Review and Full Test Suite

**Files:** None modified — verification only.

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run pipeline end-to-end**

Run: `uv run python -m src.pipeline`
Expected: Complete with results for all subjects including 0104 and 0106.

- [ ] **Step 3: Lint and format**

Run: `uv run ruff check --fix src/ tests/ && uv run ruff format src/ tests/`
Expected: Clean output

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -u
git commit -m "Lint and format after pipeline integration"
```

---

### Task 8: Update previouslytried.md and README.md

**Files:**
- Modify: `previouslytried.md`
- Modify: `README.md`

- [ ] **Step 1: Record the experiment in previouslytried.md**

Add a new section after the existing experiments:

```markdown
## Round 5: New Subjects & FBCSP Band Optimization

### Experiment 1: Integrate MI_DATA_NEW subjects into main pipeline

- **What:** Added subjects 0100-0102, 0104, 0106 from MI_DATA_NEW to the main 4-classifier evaluation. Subject 0105 excluded (empty recording). Subject 0104 sessions merged (64 trials total). Subject 0106 has 32 trials.
- **Result:** [FILL IN after pipeline run — nested mean, augmented mean, per-subject breakdown]

### Experiment 2: Subject-specific FBCSP band optimization

- **What:** Inner CV now selects from 3 band configurations ("standard", "high_mu", "wide_mu") per subject. Selection is per-subject, per-outer-fold. Tie-breaking favors "standard".
- **Result:** [FILL IN after pipeline run — which subjects selected non-standard bands, accuracy impact]
```

- [ ] **Step 2: Update README.md with new results table**

Update the results section with the new subject count and accuracy numbers from the pipeline run.

- [ ] **Step 3: Commit**

```bash
git add previouslytried.md README.md
git commit -m "Log Round 5 experiments and update results"
```
