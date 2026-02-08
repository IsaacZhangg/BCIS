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
  → features.py: lateralization indices (C3 vs C4), CSP, Hjorth params, frontal theta (39 features)
  → train.py: two classifiers evaluated in parallel:
      1. FBCSP (10 bands × 4 CSP) + handcrafted → SelectKBest(k=10) → StandardScaler → LDA
      2. Riemannian: Covariances(OAS) → TangentSpace(riemann) → LogisticRegression
  → pipeline.py: orchestrates the above, runs both classifiers, saves best per subject
```

**Event encoding**: stim value = `phase * 10 + movement` (phase 3 only; movement 1=left, 2=right). Each complete recording has exactly 100 phase-3 trials (50 left, 50 right).

**FBCSP + LDA pipeline**: `train.py` uses Filter-Bank CSP across 10 frequency bands [(4,8), (8,10), ..., (30,40)] with 4 CSP components each, producing up to 40 spatial features. These are concatenated with 39 handcrafted features from `features.py`, reduced to 10 via SelectKBest(f_classif), scaled, and classified with shrinkage LDA.

**Riemannian pipeline**: `train.py` also provides a Riemannian geometry classifier — OAS covariance estimation → Riemannian tangent space projection → Logistic Regression. This is parameter-free (no frequency band tuning) and complements FBCSP on some subjects.

**Artifact rejection**: `epochs.py` includes adaptive rejection using median + 4×MAD threshold on peak-to-peak amplitude, dropping only statistical outliers per subject rather than using a fixed µV threshold.

**Key channels**: C3 and C4 (motor cortex, primary discriminative pair), Cz (supplementary motor area), Fz (frontal theta/attention).

## Data

EEG recordings live in `unicorn-data/` (gitignored). Structure: `unicorn-data/subject{NNNN}/session{NNN}/recording_*.csv`. Each CSV has columns: timestamp, Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, stim.

## Available but unused utilities

- `preprocess.py: common_average_reference()` — CAR spatial filter. Tested but hurts CSP with only 8 channels (reduces rank 8→7).
- `preprocess.py: apply_asr()` — Artifact Subspace Reconstruction via asrpy (includes numpy 2.x compatibility patch). Tested but removes discriminative motor imagery variance along with artifacts.

## Current Status

Dual-classifier pipeline (FBCSP+LDA and Riemannian) with adaptive artifact rejection. 4/10 subjects above chance, best-of mean accuracy 60.1%. Best subject: subject0006 at 91%. The pipeline selects the better classifier per subject. Signal quality remains the bottleneck with the 8-channel consumer-grade Unicorn headset. Git branch `P3LR` with PR base `P3P5`.
