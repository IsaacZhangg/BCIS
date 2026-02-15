# BCIS — Left/Right Motor Imagery BCI Classifier

## Purpose

Classifies left vs right hand motor imagery from 8-channel EEG (g.tec Unicorn) for controlling a robotic 6th finger. Consumer-grade hardware with limited signal quality is the primary constraint — accuracy improvements matter most.

## Architecture

Pipeline: CSV → data_loader → preprocess → epochs → features → train (4 classifiers) → nested CV model selection.

```
src/
  config.py        # All experiment parameters (channels, bands, thresholds)
  pipeline.py      # Entry point — orchestrates the full pipeline
  data_loader.py   # CSV parsing, event marker extraction
  preprocess.py    # Bandpass + notch filtering (MNE)
  epochs.py        # Baseline/task windowing, per-trial PTP
  features.py      # 45 handcrafted + CSP features
  train.py         # 4 classifiers, nested CV, parallel dispatch
  validation.py    # CV split policies
tests/             # Mirrors src/ — one test file per module
models/            # Per-subject .joblib files + training_results.json
unicorn-data/      # Raw EEG recordings (gitignored)
```

## Commands

```bash
uv run python -m src.pipeline          # Full pipeline
uv run pytest tests/ -v                # All tests
uv run pytest tests/test_train.py -v   # Single module
```

## Key Design Decisions

- **In-fold artifact rejection**: Thresholds computed from training fold only — never leak test data.
- **Nested CV**: Inner 7-fold picks best classifier per subject, outer 10-fold evaluates — unbiased accuracy estimate.
- **Bandpass caching**: Filtered signals precomputed per subject to avoid redundant work across CV folds.
- All experiment parameters live in `src/config.py` — change there, not scattered across files.

## Context Documents

Read these when relevant to the task at hand:

- `previouslytried.md` — Exhaustive log of parameter sweeps and experiments already attempted. **Check before proposing changes** to avoid re-running failed approaches.
- `docs/plans/` — Design docs and implementation plans for major features.
- `README.md` — Results table, hardware details, data format, inference API.
