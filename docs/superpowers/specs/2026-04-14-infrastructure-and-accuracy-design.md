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
| `src/classifiers.py` | Classifier factory/registry. Each classifier (FBCSP+LDA, Riemannian, SVM, Ensemble) is a registered builder function. Adding a new classifier = one function + one registration call. | ~200 |
| `src/band_cache.py` | `_precompute_bandpassed()` and band-filtered data management. Currently interleaved with CV logic. | ~100 |
| `src/rejection.py` | `_reject_in_fold()`, `compute_trial_max_ptp()`, and adaptive threshold logic. Currently split between train.py and epochs.py. Consolidated here. | ~150 |
| `src/train.py` (slimmed) | CV orchestration, nested model selection, cross-session eval, final model fit/predict. Calls into the above modules. | ~500 |

**Classifier registry design:**

```python
# src/classifiers.py
from typing import Callable
from sklearn.pipeline import Pipeline

# Builder takes (config, n_features) and returns a fitted-ready Pipeline
ClassifierBuilder = Callable[[TrainingConfig, int], Pipeline]

CLASSIFIER_REGISTRY: dict[str, ClassifierBuilder] = {}

def register_classifier(name: str):
    """Decorator to register a classifier builder."""
    def decorator(fn: ClassifierBuilder) -> ClassifierBuilder:
        CLASSIFIER_REGISTRY[name] = fn
        return fn
    return decorator

@register_classifier("fbcsp_lda")
def build_fbcsp_lda(config: TrainingConfig, n_features: int) -> Pipeline:
    ...

@register_classifier("riemann")
def build_riemann(config: TrainingConfig, n_features: int) -> Pipeline:
    ...
```

**Why this split:** The classifier registry is the highest-leverage change. Every Phase 2 accuracy experiment (stacking ensemble, subject-specific bands, new classifiers) becomes a single function + registration, not surgery across CV loops.

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
  "results": {
    "nested_mean": 0.589,
    "augmented_mean": 0.646,
    "per_subject": {}
  },
  "baseline_comparison": "+0.0% nested, -0.2% augmented",
  "verdict": "neutral",
  "seed": 42
}
```

**New module: `src/experiments.py`**
- `save_experiment(name, config_diff, results, verdict)` -- writes structured JSON
- `load_experiment(name)` -- reads back
- `compare_experiments(name_a, name_b)` -- side-by-side diff
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
| 1. Load & preprocess | `cache/preprocessed_{subject}.npz` | Data files unchanged |
| 2. Epoch extraction | `cache/epochs_{subject}.npz` | Preprocessing cache hit + same epoch config |
| 3. Feature extraction | `cache/features_{subject}.npz` | Epoch cache hit + same feature config |
| 4. Cross-validation | `experiments/{name}.json` | Never skipped |
| 5. Model export | `models/*.joblib` | Explicit opt-in only |

**Cache invalidation:** Hash-based. Each stage computes a hash of its config parameters + input file mtimes. If the hash matches the cached output's metadata, skip. Simple and deterministic.

**CLI flags:**

```
uv run python -m src.pipeline --stages cv --subjects 0001,0003
uv run python -m src.pipeline --stages features,cv --classifier fbcsp_lda
uv run python -m src.pipeline --stages cv --classifier new_thing_im_testing
```

**Target speedup:** Testing a new classifier or feature set goes from full pipeline (~20 min) to CV-only on cached features (~2 min).

`cache/` is added to `.gitignore`.

### 1.4 Slim down pipeline.py

pipeline.py (852 lines) becomes a thin CLI entry point and stage coordinator:

| Current concern in pipeline.py | Moves to |
|---|---|
| Augmented nested CV logic (`_augmented_nested_cv_subject`) | `train.py` (CV orchestration) |
| Donor selection via Riemannian distance | `transfer.py` (already has the transfer module) |
| Result printing and formatting | `experiments.py` (unified tracking) |
| Stage orchestration | Stays in `pipeline.py` |

**Target:** ~250 lines, wiring stages together and parsing CLI args.

### 1.5 Testing strategy

- All 61 existing tests pass throughout the refactor. No "break now, fix later."
- When code moves to new modules, existing tests stay and pass via updated imports.
- New modules get their own test files: `test_classifiers.py`, `test_band_cache.py`, `test_rejection.py`, `test_experiments.py`.
- No-leakage tests are sacred and never weakened.

---

## Phase 2: Accuracy Improvements

### 2.1 New subject integration

Subjects 104/106 (and future subjects) integrated into the pipeline. Partial pipeline means new subjects are preprocessed and cached without re-running existing ones. Experiment tracker compares "10 subjects" vs "12 subjects" results side by side.

### 2.2 Untested algorithmic ideas

Each is a focused experiment enabled by the classifier registry:

1. **Subject-specific FBCSP bands** -- noted as "computationally expensive" but never tried. Partial runs + band cache make it feasible. Sweep per-subject band ranges in inner CV.
2. **Stacking ensemble** -- CV predictions from individual classifiers as features for a meta-learner (e.g., LogisticRegression), replacing simple soft voting.
3. **Per-subject hyperparameter optimization** -- different C values or CSP components per subject. The registry makes this a loop over subjects, not a rewrite.
4. **Adaptive epoch rejection thresholds** -- per-subject n_mad tuning rather than fixed 3.5 for all subjects. Inner CV selects optimal threshold.

### 2.3 Better use of new data for transfer learning

More subjects in the donor pool enables:

1. **Weighted multi-donor augmentation** -- weight donor contributions by Riemannian distance (closer donors contribute more). Already designed, not yet implemented.
2. **Systematic donor selection comparison** -- use experiment tracking to compare Riemannian distance vs. accuracy-based vs. demographic donor selection strategies.

---

## File changes summary

**New files:**
- `src/classifiers.py` -- classifier registry and builders
- `src/band_cache.py` -- band-filtered data caching
- `src/rejection.py` -- in-fold artifact rejection
- `src/experiments.py` -- experiment tracking and CLI
- `tests/test_classifiers.py`
- `tests/test_band_cache.py`
- `tests/test_rejection.py`
- `tests/test_experiments.py`

**Modified files:**
- `src/train.py` -- slimmed to ~500 lines, imports from new modules
- `src/pipeline.py` -- slimmed to ~250 lines, stage-based with caching and CLI
- `src/transfer.py` -- absorbs donor selection from pipeline.py
- `src/epochs.py` -- rejection logic moves to rejection.py
- `.gitignore` -- add `cache/`, `experiments/`

**Unchanged files:**
- `src/config.py`
- `src/data_loader.py`
- `src/preprocess.py`
- `src/features.py`
- `src/validation.py`
- `src/runtime_output.py`
- All existing test files (updated imports only)

**Preserved files:**
- `previouslytried.md` -- stays as human narrative
- `research/` -- stays, auto-research framework updated to write to experiments/
- `models/training_results.json` -- replaced by experiments/ but not deleted
