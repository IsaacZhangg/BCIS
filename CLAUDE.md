# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Left/right motor imagery BCI classifier for controlling a robotic 6th finger. Classifies EEG signals from an 8-channel consumer-grade Unicorn headset (250Hz) into left vs. right motor imagery using a four-classifier pipeline with nested cross-validation.

## Commands

```bash
# Install dependencies
uv sync

# Run full pipeline (load data, preprocess, extract features, train, evaluate, save model)
uv run python -m src.pipeline

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_train.py -v

# Lint and format
ruff check --fix src/ tests/
ruff format src/ tests/
```

## Data

EEG recordings live in `unicorn-data/` (gitignored).

- **Structure**: `unicorn-data/subject{NNNN}/session{NNN}/recording_*.csv`
- **Columns**: timestamp, Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, stim
- **Event encoding**: stim value = `phase * 10 + movement` (phase 3 only; movement 1=left, 2=right)
- **Trials**: each complete recording has exactly 100 phase-3 trials (50 left, 50 right)

### Key Channels

| Channel | Role |
|---------|------|
| C3, C4 | Motor cortex (primary discriminative pair) |
| Cz | Supplementary motor area |
| Fz | Frontal theta / attention |
| PO7, Pz, PO8 | Surface Laplacian neighbors |

## Architecture

EEG data flows through a linear pipeline:

```
CSV files (8ch, 250Hz Unicorn headset)
  -> data_loader.py: parse recordings, extract events
  -> preprocess.py: bandpass (1-40Hz) + notch (60Hz) filtering via MNE
  -> epochs.py: extract paired baseline/task windows + per-trial PTP computation
  -> features.py: Surface Laplacian filtered C3/C4 lateralization indices,
                  ERD, Hjorth params, coherence, spectral entropy,
                  Cz motor area, frontal theta (45 features)
  -> train.py: four classifiers + nested model-selection CV
  -> pipeline.py: orchestrates the above, optional held-out split,
                  cross-session detection, saves models
```

### Feature Engineering (`features.py`)

**Surface Laplacian** sharpens C3/C4 spatial resolution before all feature extraction:
- `C3_lap = C3 - mean(Fz, Cz, PO7)`
- `C4_lap = C4 - mean(Cz, Pz, PO8)`

**45 handcrafted features** computed from Laplacian-filtered signals:
- C3/C4 lateralization indices
- Event-related desynchronization (ERD)
- Hjorth parameters
- C3-C4 coherence
- Spectral entropy
- Cz motor area features
- Frontal theta

**FBCSP features**: 8 motor-focused frequency bands [(8,10), (10,12), ..., (24,30)] x 3 CSP components (OAS regularization) = up to 24 spatial features. Concatenated with the 45 handcrafted features.

### Classifiers (`train.py`)

Four classifiers are evaluated in parallel:

1. **FBCSP + LDA**: FBCSP + handcrafted features -> SelectKBest(f_classif, k from {3,5,8,10,15,20,25}) -> StandardScaler -> shrinkage LDA with class priors estimated from training fold
2. **Riemannian**: Covariances(OAS) -> TangentSpace(riemann) -> LogisticRegression(C=0.1). Parameter-free (no frequency band tuning), complements FBCSP on some subjects
3. **FBCSP + SVM**: Same FBCSP + handcrafted features -> SelectKBest(k) -> StandardScaler -> SVC(RBF, C=10.0, gamma='scale'). Tests nonlinear decision boundaries
4. **Ensemble**: Soft voting (LDA + SVM probability averaging). Riemannian excluded because its lower accuracy drags down the vote

All four use `_evaluate_classifier()` as a shared helper.

### Artifact Rejection

Rejection happens **inside each CV fold**, not globally:

- `epochs.py: compute_trial_max_ptp()` computes max peak-to-peak amplitude across channels per trial
- `train.py: _reject_in_fold()` computes threshold (median + 3.5xMAD) from **training-fold trials only**, then applies to both train and test indices (no test-fold leakage)
- Flat signal rejection: trials with ptp < 1uV are also rejected
- For deployment models, global rejection is applied once
- Gradient and HF power criteria are available but disabled (they reject trials containing discriminative motor imagery signal)

### Cross-Validation (`train.py`)

**Nested model-selection CV** (`train_nested_model_selection_cv()`):
- Outer loop: 10-fold per subject
- Inner loop: 7-fold evaluates all 4 classifiers on outer-train
- Picks best classifier per outer fold, retrains on full outer-train, evaluates on outer-test
- In-fold artifact rejection applied at the outer level
- Provides unbiased estimate of the "best-of-4" strategy

**Cross-session evaluation** (`cross_session_evaluate()`):
- Trains on session A, evaluates on session B (no CV -- independent test)
- `data_loader.py: get_recordings_by_subject()` groups recordings by subject

**Held-out split** (`pipeline.py: run_pipeline(holdout_fraction=0.2)`):
- Splits epochs before artifact rejection
- Computes threshold from train only via `epochs.py: compute_rejection_threshold()`
- Default 0.0 preserves existing CV-only behavior

### Performance Optimization

- **Subject-level parallelism**: `joblib.Parallel` (n_jobs=-1, loky backend) with BLAS thread capping via `threadpoolctl`
- **Bandpass caching**: When `enable_band_cache=True` with `cache_scope="subject"`, bandpass filtering is precomputed once per subject and sliced by CV indices, avoiding redundant MNE `filter_data` calls
- **Timing**: Pipeline prints per-step wall-clock times and saves `runtime_seconds` in results JSON

## Current Configuration

| Parameter | Value |
|-----------|-------|
| Epoch timing | skip 0.25s, task 3.0s, baseline 1.0s |
| trial_group_size | 1 (pure StratifiedKFold) |
| Artifact threshold | median + 3.5xMAD (in-fold) |
| FBCSP bands | 8 bands, (8-30Hz), 3 CSP components, OAS reg |
| Feature selection | SelectKBest(k from {3,5,8,10,15,20,25}) |
| LDA | Shrinkage, class priors from training fold |
| Riemannian | LogisticRegression(C=0.1) |
| SVM | SVC(RBF, C=10.0, gamma='scale') |
| Parallelism | n_jobs=-1, loky backend, band cache on |

## Current Performance

- **Nested selection mean accuracy**: 61.2% (unbiased) across 10 subjects
- **Best-of mean**: 63.2% (optimistic, shown for reference only)
- **FBCSP+LDA mean**: 59.3%
- **SVM mean**: 58.8%
- **Subjects above chance (>=60%)**: 5 of 10
- **Bottleneck**: Signal quality with the 8-channel consumer-grade Unicorn headset

## Available but Unused Utilities

- `preprocess.py: common_average_reference()` -- CAR spatial filter. Tested but hurts CSP with only 8 channels (reduces rank 8->7).
- `preprocess.py: apply_asr()` -- Artifact Subspace Reconstruction via asrpy (includes numpy 2.x compatibility patch). Tested but removes discriminative motor imagery variance along with artifacts.
