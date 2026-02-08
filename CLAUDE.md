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
  → epochs.py: extract paired baseline (1.0s) / task (1.8s) windows from phase-3 imagery events
  → features.py: lateralization indices (C3 vs C4), CSP, Hjorth params, frontal theta (39 features)
  → train.py: FBCSP (10 bands × 4 CSP components = 40 features) + handcrafted features → SelectKBest(k=20) → StandardScaler → LDA
  → pipeline.py: orchestrates the above, runs within-subject 10-fold CV, saves per-subject models
```

**Event encoding**: stim value = `phase * 10 + movement` (phase 3 only; movement 1=left, 2=right). Each complete recording has exactly 100 phase-3 trials (50 left, 50 right).

**FBCSP + LDA pipeline**: `train.py` uses Filter-Bank CSP across 10 frequency bands [(4,8), (8,10), ..., (30,40)] with 4 CSP components each, producing up to 40 spatial features. These are concatenated with 39 handcrafted features from `features.py`, reduced to 20 via SelectKBest(f_classif), scaled, and classified with shrinkage LDA. Per-subject models are saved as `models/{subject_id}_fbcsp_lda.joblib`.

**Key channels**: C3 and C4 (motor cortex, primary discriminative pair), Cz (supplementary motor area), Fz (frontal theta/attention).

## Data

EEG recordings live in `unicorn-data/` (gitignored). Structure: `unicorn-data/subject{NNNN}/session{NNN}/recording_*.csv`. Each CSV has columns: timestamp, Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, stim.

## Current Status

Consolidated from a 40+ classifier ensemble + separate LGBM to a single clean FBCSP + LDA pipeline. Only 2/10 subjects (subject0006: 88%, subject0010: 82%) show above-chance accuracy; the other 8 are at chance level (~42-57%). Mean accuracy: 57.2%. This is honest reporting — signal quality varies by subject with consumer-grade EEG. Git branch `P3LR` with PR base `P3P5`.
