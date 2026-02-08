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
  → data_loader.py: parse recordings, extract events
  → preprocess.py: bandpass (1-40Hz) + notch (60Hz) filtering via MNE
  → epochs.py: extract paired baseline/task windows + adaptive artifact rejection
  → features.py: Surface Laplacian filtered C3/C4 lateralization indices,
                  ERD, Hjorth params, Cz motor area, frontal theta (39 features)
  → train.py: four classifiers evaluated in parallel:
      1. FBCSP (10 bands × 4 CSP) + handcrafted → SelectKBest(nested CV k) → StandardScaler → LDA
      2. Riemannian: Covariances(OAS) → TangentSpace(riemann) → LogisticRegression
      3. FBCSP + SVM: same FBCSP features → SelectKBest(k=10) → StandardScaler → SVC(RBF)
      4. Ensemble: soft voting (LDA + SVM probability averaging)
  → pipeline.py: orchestrates the above, runs all four classifiers, saves best per subject
```

**Event encoding**: stim value = `phase * 10 + movement` (phase 3 only; movement 1=left, 2=right). Each complete recording has exactly 100 phase-3 trials (50 left, 50 right).

**FBCSP + LDA pipeline**: `train.py` uses Filter-Bank CSP across 10 frequency bands [(4,8), (8,10), ..., (30,40)] with 4 CSP components each, producing up to 40 spatial features. These are concatenated with 39 handcrafted features from `features.py`, reduced via nested CV SelectKBest(f_classif, k from {5,8,10,15,20}), scaled, and classified with shrinkage LDA using class priors estimated from training fold frequencies.

**Riemannian pipeline**: `train.py` also provides a Riemannian geometry classifier — OAS covariance estimation → Riemannian tangent space projection → Logistic Regression. This is parameter-free (no frequency band tuning) and complements FBCSP on some subjects.

**FBCSP + SVM pipeline**: `train.py` provides an FBCSP + SVM(RBF) classifier — same FBCSP + handcrafted feature pipeline as LDA but with SVC(kernel='rbf', C=1.0, gamma='scale') and fixed k=10. Tests nonlinear decision boundaries that LDA misses. Replaces the previous Transfer learning pipeline which averaged below chance (46.9%).

**Ensemble pipeline**: Soft voting between LDA and SVM — averages their predicted probabilities (both use `predict_proba`) and takes argmax. Riemannian is excluded from the ensemble because its lower overall accuracy drags down the vote.

**Artifact rejection**: `epochs.py` uses adaptive amplitude-only rejection: peak-to-peak amplitude via median + 4×MAD threshold, plus flat signal rejection (ptp < 1µV). Gradient and HF power criteria are available but disabled by default — they were found to reject trials containing discriminative motor imagery signal.

**Surface Laplacian**: `features.py` applies approximate Surface Laplacian to sharpen C3/C4 spatial resolution: C3_lap = C3 − mean(Fz, Cz, PO7), C4_lap = C4 − mean(Cz, Pz, PO8). All lateralization and Hjorth features are computed from these Laplacian-filtered signals.

**Key channels**: C3 and C4 (motor cortex, primary discriminative pair), Cz (supplementary motor area), Fz (frontal theta/attention). PO7, Pz, PO8 serve as Laplacian neighbors.

## Data

EEG recordings live in `unicorn-data/` (gitignored). Structure: `unicorn-data/subject{NNNN}/session{NNN}/recording_*.csv`. Each CSV has columns: timestamp, Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, stim.

## Available but unused utilities

- `preprocess.py: common_average_reference()` — CAR spatial filter. Tested but hurts CSP with only 8 channels (reduces rank 8→7).
- `preprocess.py: apply_asr()` — Artifact Subspace Reconstruction via asrpy (includes numpy 2.x compatibility patch). Tested but removes discriminative motor imagery variance along with artifacts.

## Current Status

Four-classifier pipeline (FBCSP+LDA, Riemannian, FBCSP+SVM, Ensemble) with amplitude-only adaptive artifact rejection and Surface Laplacian spatial filtering. 39 handcrafted features (Laplacian-filtered C3/C4 lateralization, ERD, Hjorth, Cz motor area, frontal theta) + 40 FBCSP features, reduced via nested CV SelectKBest for LDA (k from {5,8,10,15,20}) and fixed k=10 for SVM. LDA uses class priors estimated from training fold. Best-of mean accuracy 61.2% across 10 subjects, 3 above chance (>=60%). The pipeline selects the best of four classifiers per subject. Signal quality remains the bottleneck with the 8-channel consumer-grade Unicorn headset. Git branch `P3LR` with PR base `P3P5`.
