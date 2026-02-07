# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EEG-based fatigue/engagement detection system for BCI applications. Classifies Phase 3 (motor imagery = engaged) vs Phase 5 (rest = disengaged) from Unicorn Hybrid Black EEG recordings.

**Current Status:** 92.8% mean accuracy across 10 subjects using within-subject cross-validation.

## Commands

```bash
# Install dependencies
uv sync

# Run full training pipeline
uv run python -m src.pipeline

# Run real-time simulation (batch mode)
uv run python -m src.simulate_realtime unicorn-data/<recording>.csv

# Run real-time simulation (streaming mode with sliding windows)
uv run python -m src.simulate_realtime unicorn-data/<recording>.csv --streaming --slide 1.0
# Additional options: --threshold 30 --window 5.0 --consecutive 3

# Run classifier benchmarks (latency + accuracy comparison)
uv run python -m src.benchmark_classifiers

# Run pipeline profiler (bottleneck analysis)
uv run python -m src.profile_pipeline

# Run tests
uv run pytest tests/ -v
uv run pytest tests/test_pipeline.py -v       # Single file
uv run pytest tests/test_pipeline.py::test_X -v  # Single test

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Architecture

### Training Pipeline (`src/pipeline.py`)

```
Raw CSV → Load → Preprocess → Extract Epochs → Extract Features → Train → Save Models
```

1. **data_loader.py** - Load Unicorn CSV with 8 EEG channels + stim markers. Filters for "complete" recordings (exactly 100 imagery trials).
2. **preprocess.py** - Bandpass (1-40 Hz), notch (60 Hz), CAR spatial filter
3. **epochs.py** - Extract augmented baseline+task epoch pairs for Phase 3 and Phase 5 with sliding windows
4. **features.py** - Three feature extraction paths:
   - **ERD features** (`extract_erd_features`): Event-related desynchronization using baseline/task pairs. Used for training.
   - **Riemannian features**: Covariance matrices from multichannel epoch data, processed via pyriemann. Built in `pipeline.py:create_multichannel_arrays`.
   - **Realtime features** (`extract_realtime_features`): Absolute band powers, Hjorth, envelope, ratios. No baseline needed.
5. **features_fast.py** - Optimized drop-in replacement for realtime features. Uses single Welch PSD per channel with FFT bin masking instead of 16 separate filtfilt calls. Also includes `extract_realtime_features_fast_v2` with filter bank features from PSD.
6. **train.py** - Within-subject CV with 20+ classifiers using weighted ensemble voting. Three model families: Riemannian (MDM, tangent space+LDA/SVM), traditional (RF, ExtraTrees, GBM, SVM, LDA), and combined (ERD+Riemannian tangent features).

### Dual Model Design

- **Full-feature model** (`models/engagement_classifier_model.joblib`): ERD features, requires baseline epochs. Used for validation.
- **Realtime model** (`models/engagement_realtime_model.joblib`): Absolute features only, trained on pooled and balanced data from all subjects. Used for streaming inference.

Both models are `RandomForestClassifier(n_estimators=200)` with `StandardScaler`. Each has a companion `*_scaler.joblib`.

### Streaming Architecture

Three-layer design for real-time inference:

1. **eeg_stream.py** - Abstract `EEGStream` base with concrete implementations:
   - `FileEEGStream`: Replay CSV sample-by-sample (testing/simulation)
   - `RandomEEGStream`: Synthetic EEG from summed sinusoids (unit testing)
   - `LiveEEGStream`: Unicorn Hybrid Black via UnicornPy SDK (production)
2. **realtime_engine.py** - `RealtimeEngine` orchestrates sample-by-sample processing:
   - `RingBuffer` accumulates multichannel samples
   - `IncrementalFilter` applies IIR bandpass/notch with persistent state (no edge artifacts between chunks)
   - Emits `WindowResult` at configurable slide intervals via `on_result` callback
3. **simulate_realtime.py** - CLI entry point. Supports both legacy batch mode and `--streaming` mode using `FileEEGStream + RealtimeEngine`.

### Feature Selection in Training

`train.py` uses `SelectKBest(f_classif)` at 30%, 50%, and 70% of features. All three percentages produce separate model predictions that feed into the ensemble.

## EEG Data Format

- **Channels:** Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 (8 channels)
- **Sample Rate:** 250 Hz
- **Location:** `unicorn-data/` directory (subject*/session*/*.csv)

### Stim Column Encoding

Each non-zero stim value: `stim = trial_number * 100 + phase * 10 + movement`, decoded as `phase = (stim // 10) % 10`, `movement = stim % 10`.

**Phase codes:** 1=Video, 2=Instruction, 3=Imagery (engaged), 4=Movement, 5=Rest (disengaged)

**Movement codes:** 1=Left, 2=Right

A complete recording has exactly 100 Phase 3 (imagery) markers (50 per hand).

## Key Dependencies

- **pyriemann** - Riemannian geometry for BCI (covariance estimation, MDM, tangent space)
- **MNE** - EEG signal processing (used in preprocessing)
- **scikit-learn** - ML classifiers, feature selection, cross-validation
- **XGBoost / LightGBM** - Gradient boosting (used in benchmark_classifiers)
- **scipy** - Signal processing (Welch PSD, IIR filters, Hilbert transform)
