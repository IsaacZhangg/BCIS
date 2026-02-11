# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Architecture

This is a left/right motor imagery BCI classifier for controlling a robotic 6th finger. EEG data flows through a linear pipeline:

```
CSV files (8ch, 250Hz Unicorn headset)
  -> data_loader.py: parse recordings, extract events
  -> preprocess.py: bandpass (1-40Hz) + notch (60Hz) filtering via MNE
  -> epochs.py: extract paired baseline/task windows + per-trial PTP computation
  -> features.py: Surface Laplacian filtered C3/C4 lateralization indices,
                  ERD, Hjorth params, coherence, spectral entropy,
                  Cz motor area, frontal theta (45 features)
  -> train.py: four classifiers evaluated in parallel:
      1. FBCSP (8 bands x 4 CSP) + handcrafted -> SelectKBest(nested CV k) -> StandardScaler -> LDA
      2. Riemannian: Covariances(OAS) -> TangentSpace(riemann) -> LogisticRegression
      3. FBCSP + SVM: same FBCSP features -> SelectKBest(k) -> StandardScaler -> SVC(RBF, C=10)
      4. Ensemble: soft voting (LDA + SVM probability averaging)
  -> train.py: in-fold artifact rejection (threshold from train fold only)
  -> train.py (nested CV): nested model-selection CV picks best classifier
                          per outer fold (unbiased); cross-session evaluate
  -> pipeline.py: orchestrates the above, optional held-out split,
                  cross-session detection, saves models using nested CV selection
```

**Event encoding**: stim value = `phase * 10 + movement` (phase 3 only; movement 1=left, 2=right). Each complete recording has exactly 100 phase-3 trials (50 left, 50 right).

**FBCSP + LDA pipeline**: `train.py` uses Filter-Bank CSP across 8 motor-focused frequency bands [(8,10), (10,12), ..., (24,30)] with 4 CSP components each (OAS regularization), producing up to 32 spatial features. These are concatenated with 45 handcrafted features from `features.py`, reduced via nested CV SelectKBest(f_classif, k from {3,5,8,10,15,20}), scaled, and classified with shrinkage LDA using class priors estimated from training fold frequencies.

**Riemannian pipeline**: `train.py` also provides a Riemannian geometry classifier -- OAS covariance estimation -> Riemannian tangent space projection -> Logistic Regression. This is parameter-free (no frequency band tuning) and complements FBCSP on some subjects.

**FBCSP + SVM pipeline**: `train.py` provides an FBCSP + SVM(RBF) classifier -- same FBCSP + handcrafted feature pipeline as LDA but with SVC(kernel='rbf', C=10.0, gamma='scale'). Tests nonlinear decision boundaries that LDA misses.

**Ensemble pipeline**: Soft voting between LDA and SVM -- averages their predicted probabilities (both use `predict_proba`) and takes argmax. Riemannian is excluded from the ensemble because its lower overall accuracy drags down the vote.

**In-fold artifact rejection**: `train.py: _reject_in_fold()` computes the amplitude rejection threshold (median + 3.5xMAD) from training-fold trials only, then applies it to both train and test indices. This prevents the rejection threshold from leaking test-fold information. `epochs.py: compute_trial_max_ptp()` computes max peak-to-peak amplitude across channels for each trial. The pipeline passes `trial_ptps_by_subject` through all CV functions.

**Nested model-selection CV**: `train.py: train_nested_model_selection_cv()` provides an unbiased estimate of the "best-of-4" strategy. Outer 10-fold loop per subject; inner 7-fold loop evaluates all 4 classifiers on outer-train, picks best, retrains on full outer-train, evaluates on outer-test. Uses `_evaluate_classifier()` shared helper for all 4 classifiers. In-fold rejection is applied at the outer level.

**Cross-session evaluation**: `train.py: cross_session_evaluate()` trains on session A, evaluates on session B (no CV -- independent test). Returns per-classifier accuracy. `data_loader.py: get_recordings_by_subject()` groups recordings by subject for detection.

**Held-out split**: `pipeline.py: run_pipeline(holdout_fraction=0.2)` splits epochs before artifact rejection, computes threshold from train only via `epochs.py: compute_rejection_threshold()`, applies to both splits. Default 0.0 preserves existing behavior.

**Artifact rejection**: Rejection now happens **inside each CV fold** rather than globally. `epochs.py: compute_trial_max_ptp()` computes per-trial max PTP amplitudes. `train.py: _reject_in_fold()` uses median + 3.5xMAD threshold from training-fold trials only, plus flat signal rejection (ptp < 1uV). For deployment models, global rejection is applied once. Gradient and HF power criteria are available but disabled by default -- they were found to reject trials containing discriminative motor imagery signal.

**Surface Laplacian**: `features.py` applies approximate Surface Laplacian to sharpen C3/C4 spatial resolution: C3_lap = C3 - mean(Fz, Cz, PO7), C4_lap = C4 - mean(Cz, Pz, PO8). All lateralization, Hjorth, coherence, and entropy features are computed from these Laplacian-filtered signals.

**Key channels**: C3 and C4 (motor cortex, primary discriminative pair), Cz (supplementary motor area), Fz (frontal theta/attention). PO7, Pz, PO8 serve as Laplacian neighbors.

## Data

EEG recordings live in `unicorn-data/` (gitignored). Structure: `unicorn-data/subject{NNNN}/session{NNN}/recording_*.csv`. Each CSV has columns: timestamp, Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, stim.

## Available but unused utilities

- `preprocess.py: common_average_reference()` -- CAR spatial filter. Tested but hurts CSP with only 8 channels (reduces rank 8->7).
- `preprocess.py: apply_asr()` -- Artifact Subspace Reconstruction via asrpy (includes numpy 2.x compatibility patch). Tested but removes discriminative motor imagery variance along with artifacts.

## Current Status

Four-classifier pipeline (FBCSP+LDA, Riemannian, FBCSP+SVM, Ensemble) with nested model-selection CV, optional held-out split, and cross-session evaluation infrastructure. **In-fold artifact rejection** (threshold from training fold only, via `_reject_in_fold()`) and Surface Laplacian spatial filtering. 45 handcrafted features (Laplacian-filtered C3/C4 lateralization, ERD, Hjorth, C3-C4 coherence, spectral entropy, Cz motor area, frontal theta) + 32 FBCSP features (8 motor bands x 4 CSP with OAS reg), reduced via nested CV SelectKBest for LDA (k from {3,5,8,10,15,20}). LDA uses class priors estimated from training fold. Epoch timing: skip 0.25s, task 3.0s, baseline 1.0s. Nested selection mean accuracy 55.2% (unbiased) across 10 subjects, 3 above chance (>=60%). Best-of mean 58.5% (optimistic, shown for reference). FBCSP+LDA mean 53.6%, SVM mean 55.2%. Signal quality remains the bottleneck with the 8-channel consumer-grade Unicorn headset. Git branch `P3LR` with PR base `P3P5`.
