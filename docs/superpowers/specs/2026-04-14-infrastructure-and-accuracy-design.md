# Infrastructure That Unlocks Accuracy

**Date:** 2026-04-14
**Approach:** Targeted refactoring (Phase 1) that directly enables faster accuracy experimentation (Phase 2).
**Scope:** Big overhaul. All existing tests must pass throughout.

---

## Phase 1: Infrastructure

### 1.1 Break up train.py

train.py (1580 lines) is split into four focused modules:

| New module | Responsibility | Approx lines |
|---|---|---|
| `src/classifiers.py` | Classifier registry with model-family adapters. Each classifier family (FBCSP-based, Riemannian, Ensemble) exposes `fit`, `score`, `predict`, and `export` methods. Adding a new classifier = one adapter + one registration call. | ~300 |
| `src/band_cache.py` | `_precompute_bandpassed()`, `_subset_band_cache()`, band-filtered data management, and `cache_scope` behavior (`"subject"` vs `"outer_fold"`). Currently interleaved with CV logic. | ~150 |
| `src/rejection.py` | Two-layer rejection API: (1) fold-time rejection using per-trial max PTP + index filtering (from train.py's `_reject_in_fold`), and (2) epoch-time multi-criteria rejection using PTP, gradient, and HF-power on `(baseline, task)` pairs (from epochs.py's `reject_bad_epochs`). Both layers preserved, consolidated into one module. | ~200 |
| `src/train.py` (slimmed) | CV orchestration, nested model selection, cross-session eval, final model fit/predict. Calls into the above modules. | ~500 |

**Classifier registry design:**

FBCSP models do not fit a simple sklearn `Pipeline` -- they combine handcrafted features with multichannel EEG, fit CSP models internally, and save non-pipeline state for inference. The registry uses a protocol-based adapter instead of a bare callable:

```python
# src/classifiers.py
from typing import Protocol, Any
import numpy as np

class ClassifierAdapter(Protocol):
    """Protocol for classifier families. Each adapter handles its own
    feature extraction, fitting, and prediction logic."""

    name: str

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            handcrafted_train: np.ndarray, config: TrainingConfig,
            band_cache: dict | None = None) -> None: ...

    def score(self, X_test: np.ndarray, y_test: np.ndarray,
              handcrafted_test: np.ndarray) -> float: ...

    def predict_proba(self, X_test: np.ndarray,
                      handcrafted_test: np.ndarray) -> np.ndarray: ...

    def export(self) -> dict[str, Any]:
        """Return serializable state for joblib export."""
        ...

CLASSIFIER_REGISTRY: dict[str, type[ClassifierAdapter]] = {}

def register_classifier(name: str):
    """Decorator to register a classifier adapter class."""
    def decorator(cls: type[ClassifierAdapter]) -> type[ClassifierAdapter]:
        CLASSIFIER_REGISTRY[name] = cls
        return cls
    return decorator

@register_classifier("fbcsp_lda")
class FBCSPLDAClassifier:
    name = "fbcsp_lda"
    def fit(self, X_train, y_train, handcrafted_train, config, band_cache=None):
        # CSP extraction + handcrafted concat + SelectKBest + LDA
        ...
    def score(self, X_test, y_test, handcrafted_test): ...
    def predict_proba(self, X_test, handcrafted_test): ...
    def export(self): ...

@register_classifier("riemann")
class RiemannianClassifier:
    name = "riemann"
    # Covariances -> TangentSpace -> LogisticRegression
    # Does not use band_cache or handcrafted features
    ...
```

**Why protocol-based:** FBCSP classifiers need `band_cache` and `handcrafted` features; Riemannian classifiers need neither. The protocol accommodates both without forcing a lowest-common-denominator interface. The `export()` method handles the non-pipeline state that FBCSP models need for inference.

**Why this split:** The classifier registry is the highest-leverage change. Every Phase 2 accuracy experiment (stacking ensemble, subject-specific bands, new classifiers) becomes a single adapter class + registration, not surgery across CV loops.

### 1.2 Unified experiment tracking

Current state: results in three disconnected places (previouslytried.md, research/results.tsv, models/training_results.json).

**New structure:**

```
experiments/
  2026-04-14_baseline.json
  2026-04-14_csp4-components.json
  ...
```

Each experiment file:

```json
{
  "name": "csp4-components",
  "timestamp": "2026-04-14T10:32:00",
  "config_diff": {"n_csp_components": [3, 4]},
  "seed": 42,
  "seed_type": "single",
  "results": {
    "nested_mean": 0.589,
    "augmented_mean": 0.646,
    "per_subject": {}
  },
  "baseline_comparison": "+0.0% nested, -0.2% augmented",
  "baseline_ref": "2026-04-14_baseline.json",
  "verdict": "neutral"
}
```

**Baseline semantics:** Each experiment records `seed_type` (`"single"` or `"multi"`) and `baseline_ref` (the experiment file it's compared against). Multi-seed experiments include `seeds`, `mean`, and `std` fields. This avoids the current ambiguity between single-seed (62.0%), multi-seed (58.9% +/- 1.4%), nested, and augmented baselines.

**New module: `src/experiments.py`**
- `save_experiment(name, config_diff, results, verdict, baseline_ref)` -- writes structured JSON
- `load_experiment(name)` -- reads back
- `compare_experiments(name_a, name_b)` -- side-by-side diff with baseline alignment
- `list_experiments()` -- summary table

**CLI interface:**

```
uv run python -m src.experiments list
uv run python -m src.experiments compare csp4-components baseline
```

`previouslytried.md` stays as human narrative. The auto-research framework writes to `experiments/` so every run is structured and comparable.

### 1.3 Partial pipeline runs with caching

The pipeline becomes stage-based. Each stage caches its output; stages can be skipped if inputs haven't changed.

| Stage | Cache output | Skip condition |
|---|---|---|
| 1. Load & preprocess | `cache/preprocessed_{subject}_{recording}.npz` | Data files unchanged |
| 2. Epoch extraction | `cache/epochs_{subject}_{recording}.npz` | Preprocessing cache hit + same epoch config |
| 3. Feature extraction | `cache/features_{subject}_{recording}.npz` (handcrafted features) | Epoch cache hit + same feature config |
| 3b. Band filtering | `cache/bands_{subject}_{recording}.npz` (per-band filtered trials) | Epoch cache hit + same band config |
| 4. Cross-validation | `experiments/{name}.json` | Never skipped |
| 5. Model export | `models/*.joblib` | Explicit opt-in only |

**Cache key design:** Cache files are keyed by `{subject}_{recording}` (not subject-only) to avoid collisions between cross-session recordings and transfer-merged subjects. Each cache file includes a metadata header with the config hash so stale caches are detected.

**Cache namespaces for different modes:** The main no-holdout path caches features first, then rejects in-fold via PTP. The holdout and transfer paths reject epochs before feature extraction. These produce different cached artifacts and use separate namespace prefixes (`cache/main/...` vs `cache/holdout/...` vs `cache/transfer/...`).

**Band-filtered tensor caching (stage 3b):** FBCSP/CSP extraction happens inside CV folds and is the dominant cost. Caching the band-filtered tensors (output of `_precompute_bandpassed()`) means fold-time work only does CSP fitting + projection, not re-filtering. This is where the real speedup comes from for classifier-only experiments.

**Cache invalidation:** Hash-based. Each stage computes a hash of its config parameters + input file mtimes. If the hash matches the cached output's metadata, skip. Simple and deterministic.

**CLI flags:**

```
uv run python -m src.pipeline --stages cv --subjects 0001,0003
uv run python -m src.pipeline --stages features,cv --classifier fbcsp_lda
uv run python -m src.pipeline --stages cv --classifier new_thing_im_testing
```

**Target speedup:** Testing a new classifier goes from full pipeline (~20 min) to CV on cached bands+features (~2-3 min). Testing a new feature extractor goes from full pipeline to epochs+features+CV (~5 min).

`cache/` is added to `.gitignore`.

### 1.4 Slim down pipeline.py and add CLI layer

pipeline.py (852 lines) is split into orchestration and CLI:

| Current concern in pipeline.py | Moves to |
|---|---|
| Augmented nested CV logic (`_augmented_nested_cv_subject`) | `train.py` (CV orchestration) |
| Donor selection via Riemannian distance | `transfer.py` (already has the transfer module) |
| Result printing and formatting | `experiments.py` (unified tracking) |
| Stage orchestration | Stays in `pipeline.py` |
| Cross-session eval orchestration | Stays in `pipeline.py` |
| Transfer eval orchestration | Stays in `pipeline.py` |
| Holdout handling | Stays in `pipeline.py` |
| CLI argument parsing | New `src/cli.py` |

**Revised target:** pipeline.py at ~350 lines (accounting for cross-session, transfer, and holdout orchestration that legitimately belongs here). Separate `src/cli.py` (~100 lines) for argument parsing and validation.

**CLI argument layer:** The new `src/cli.py` module handles `--stages`, `--subjects`, `--classifier`, `--experiment-name`, `--cache-dir`, and `--export` flags. These are parsed into a `RunConfig` dataclass (separate from `TrainingConfig`) that controls pipeline execution without modifying the ML hyperparameters in `TrainingConfig`.

```python
# src/cli.py
@dataclass(frozen=True)
class RunConfig:
    stages: list[str]          # which stages to run
    subjects: list[str] | None # filter to specific subjects
    classifiers: list[str] | None  # filter to specific classifiers
    experiment_name: str | None
    cache_dir: Path
    export_models: bool
```

**Why separate from TrainingConfig:** `TrainingConfig` defines ML hyperparameters (seed, folds, bands). `RunConfig` defines execution parameters (what to run, where to cache). Keeping them separate means `TrainingConfig` stays unchanged and hashable for cache invalidation.

### 1.5 Testing strategy

- All 61 existing tests pass throughout the refactor. No "break now, fix later."
- **Compatibility shims during migration:** When moving functions to new modules (e.g., `_reject_in_fold` from `train.py` to `rejection.py`), the old module re-exports the symbol so existing test imports don't break. Shims are removed only after all test files are updated.
- New modules get their own test files: `test_classifiers.py`, `test_band_cache.py`, `test_rejection.py`, `test_experiments.py`, `test_cli.py`.
- **New integration tests:** `test_pipeline_integration.py` covers `run_pipeline()` end-to-end with synthetic data, cache hit/miss behavior, stage filtering, and multi-mode artifact handling.
- No-leakage tests are sacred and never weakened.

---

## Phase 2: Accuracy Improvements

### 2.1 New subject integration

Subjects 104/106 (and future subjects) integrated into the pipeline. Currently `load_all_subjects()` can ingest arbitrary-trial recordings from multiple directories, but the main within-subject path still depends on `get_complete_recordings(data_dir)` from one directory. Phase 1's CLI subject filtering enables running the main pipeline across both data directories seamlessly.

Experiment tracker compares "10 subjects" vs "12 subjects" results side by side with explicit baseline references.

### 2.2 Untested algorithmic ideas

Each is a focused experiment enabled by the classifier registry. Ordered by expected impact (highest first):

1. **Subject-specific FBCSP bands** -- noted as "computationally expensive" but never tried. Band cache + partial runs make it feasible. **Overfitting risk:** with ~100 trials per subject, sweeping many band sets inside inner CV will overfit. Mitigation: constrain search to 3-4 pre-defined band sets (e.g., narrow motor, wide motor, individual alpha peak +/- 2 Hz) rather than free-form search. Prior evidence: adding a single extra band (6,8) already hurt, so the search space must be tight.

2. **Stacking ensemble** -- replace simple soft voting with a meta-learner trained on out-of-fold base predictions. **Leakage prevention:** base classifiers produce predictions on outer-train data via inner CV; those predictions become features for the meta-learner, which is evaluated on the outer-test fold. This requires restructuring `_evaluate_classifiers_batch()` to return out-of-fold predictions, not just scores. **Prior evidence:** 3-way voting with Riemannian was worse, but a trained stacker with calibrated probabilities is a different approach.

3. **Adaptive epoch rejection thresholds** -- per-subject n_mad tuning via inner CV. **Prior evidence:** nearby n_mad values around 3.5 were tested and neutral. The more interesting variant: bring gradient and HF-power metrics from `reject_bad_epochs()` into fold-time CV (currently only PTP is used in-fold). This requires computing per-trial gradient and HF-power metrics alongside the existing `trial_ptps`.

4. ~~**Per-subject hyperparameter optimization**~~ -- **Deprioritized.** Per-subject SVM C selection was already tried and found neutral. Per-subject CSP components may have slightly more headroom, but the evidence is weak. Only attempt after higher-priority ideas are exhausted.

### 2.3 Better use of new data for transfer learning

More subjects in the donor pool enables:

1. **Weighted multi-donor augmentation** -- weight donor contributions by Riemannian distance (closer donors contribute more). **Caution:** the prior top-2 donor experiment reduced augmented mean from 64.6% to 63.5% because weak donors hurt weak subjects. Weighting must aggressively suppress low-quality donors (e.g., hard distance cutoff + exponential decay) rather than just averaging more donors in. Success criterion: augmented mean must not regress for any individual subject compared to single-best-donor.

2. **Systematic donor selection comparison** -- compare Riemannian distance vs. accuracy-based selection. **Leakage constraint for accuracy-based:** donor ranking must be computed from outer-train data only (use inner CV accuracy of augmented classifier to rank donors). ~~Demographic selection~~ removed -- no demographic metadata exists in the codebase.

---

## File changes summary

**New files:**
- `src/classifiers.py` -- classifier registry with protocol-based adapters
- `src/band_cache.py` -- band-filtered data caching and cache_scope management
- `src/rejection.py` -- two-layer rejection (fold-time PTP + epoch-time multi-criteria)
- `src/experiments.py` -- experiment tracking, comparison, and CLI
- `src/cli.py` -- CLI argument parsing and RunConfig
- `tests/test_classifiers.py`
- `tests/test_band_cache.py`
- `tests/test_rejection.py`
- `tests/test_experiments.py`
- `tests/test_cli.py`
- `tests/test_pipeline_integration.py`

**Modified files:**
- `src/train.py` -- slimmed to ~500 lines, imports from classifiers/band_cache/rejection
- `src/pipeline.py` -- slimmed to ~350 lines, stage-based with caching
- `src/transfer.py` -- absorbs donor selection from pipeline.py
- `src/epochs.py` -- epoch-time rejection logic moves to rejection.py (re-exports for compatibility)
- `.gitignore` -- add `cache/`, `experiments/`

**Modified (previously listed as unchanged):**
- `src/config.py` -- no ML hyperparameter changes, but `TrainingConfig` gains a `config_hash()` method for cache invalidation

**Unchanged files:**
- `src/data_loader.py`
- `src/preprocess.py`
- `src/features.py`
- `src/validation.py`
- `src/runtime_output.py`
- All existing test files (updated imports only, compatibility shims in place)

**Preserved files:**
- `previouslytried.md` -- stays as human narrative
- `research/` -- stays, auto-research framework updated to write to experiments/
- `models/training_results.json` -- replaced by experiments/ but not deleted
