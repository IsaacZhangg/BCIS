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
  → epochs.py: extract paired baseline (1.0s) / task (1.5s) windows from phase-3 imagery events
  → features.py: lateralization indices (C3 vs C4), CSP, Hjorth params, frontal theta
  → train.py: 40+ classifier ensemble with multiple aggregation strategies
  → pipeline.py: orchestrates the above, runs within-subject 10-fold CV, saves model
```

**Event encoding**: stim value = `phase * 10 + movement` (phase 3 only; movement 1=left, 2=right). Each complete recording has exactly 100 phase-3 trials (50 left, 50 right).

**Ensemble approach**: `train.py` builds 40+ classifiers per fold (Filter-Bank CSP, Riemannian geometry via pyRiemann, SVM variants, Random Forest, LDA, Gradient Boosting, MLP, k-NN) and selects the best aggregation strategy (average, median, weighted, voting, Borda count, etc.) per subject. The final saved model is a Random Forest trained on handcrafted features from `features.py`.

**Key channels**: C3 and C4 (motor cortex, primary discriminative pair), Cz (supplementary motor area), Fz (frontal theta/attention).

## Data

EEG recordings live in `unicorn-data/` (gitignored). Structure: `unicorn-data/subject{NNNN}/session{NNN}/recording_*.csv`. Each CSV has columns: timestamp, Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, stim.

## Current Status

87.6% mean accuracy across 10 subjects (target: 90%). High variance across subjects (83-98%). Git branch `P3LR` with PR base `P3P5`.
