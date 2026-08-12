# BCIS - EEG-Based Fatigue Detection System

A real-time fatigue and engagement detection system using EEG signals from the Unicorn Hybrid Black headset. The system classifies brain states to detect when a patient becomes disengaged during motor imagery sessions, enabling proactive intervention.

## Overview

This project implements a classifier that distinguishes between:
- **Phase 3 (Motor Imagery)**: Engaged brain state
- **Phase 5 (Rest)**: Disengaged/fatigued brain state

The system achieves **92.8% mean accuracy** across 10 subjects using within-subject cross-validation with ensemble methods.

## Features

- **Proactive Fatigue Detection**: Identifies disengagement before the user notices
- **Streaming Inference**: Sample-by-sample processing with incremental filtering and sliding windows
- **Dual Model Architecture**:
  - Full-feature model (ERD-based) for offline validation
  - Realtime model for streaming without baseline
- **Riemannian Geometry**: MDM and tangent space classifiers via pyriemann
- **Weighted Ensemble Voting**: 20+ classifiers across three model families
- **Comprehensive Signal Processing**: Bandpass, notch, and CAR spatial filtering

## Project Structure

```
BCIS/
├── src/
│   ├── data_loader.py            # Load & parse Unicorn EEG recordings
│   ├── preprocess.py             # Bandpass, notch, CAR spatial filters
│   ├── epochs.py                 # Extract baseline-task epoch pairs (ERD)
│   ├── features.py               # Multi-band power + Hjorth + envelope features
│   ├── features_fast.py          # Optimized realtime features via Welch PSD
│   ├── train.py                  # Within-subject CV + ensemble models
│   ├── pipeline.py               # Main training orchestration
│   ├── eeg_stream.py             # EEG stream abstractions (file, random, live)
│   ├── realtime_engine.py        # Sample-by-sample processing engine
│   ├── simulate_realtime.py      # CLI for real-time simulation
│   ├── benchmark_classifiers.py  # Classifier latency + accuracy comparison
│   └── profile_pipeline.py       # Pipeline bottleneck analysis
├── tests/                        # Test suite (9 test modules)
├── benchmarks/                   # Feature extraction benchmarks
├── models/                       # Trained models + scalers + training_results.json (tracked)
└── unicorn-data/                 # EEG recordings (tracked)
```

## Installation

**Requirements:** Python 3.11+ and [uv](https://docs.astral.sh/uv/)

```bash
git clone <repository-url>
cd BCIS
git checkout P3P5
uv sync
```

Recordings live in `unicorn-data/subject{NNNN}/session{NNN}/recording_*.csv` and are committed. Trained models and `models/training_results.json` are also committed so a clone can load them without re-running the pipeline.

## Usage

### Train Models

```bash
uv run python -m src.pipeline
```

This will:
1. Load EEG recordings from `unicorn-data/`
2. Preprocess signals (bandpass, notch, CAR)
3. Extract epochs and features (ERD, Riemannian, realtime)
4. Train ensemble classifiers with within-subject cross-validation
5. Save models to `models/`

### Loading a Trained Model

```python
import joblib

# Offline ERD model (needs a baseline window)
model = joblib.load("models/engagement_classifier_model.joblib")
scaler = joblib.load("models/engagement_classifier_scaler.joblib")

# Streaming model (absolute powers only)
rt_model = joblib.load("models/engagement_realtime_model.joblib")
rt_scaler = joblib.load("models/engagement_realtime_scaler.joblib")
```

### Real-Time Simulation

**Batch mode** (process all windows at once):

```bash
uv run python -m src.simulate_realtime unicorn-data/<recording>.csv
```

**Streaming mode** (sample-by-sample with sliding windows):

```bash
uv run python -m src.simulate_realtime unicorn-data/<recording>.csv --streaming --slide 1.0
```

Options:
- `--streaming`: Enable sample-by-sample streaming mode
- `--slide`: Slide interval in seconds (default: 1.0)
- `--threshold`: Engagement score threshold for alerts (default: 30)
- `--window`: Window size in seconds (default: 5.0)
- `--consecutive`: Consecutive low windows before alert (default: 3)

### Benchmarking

```bash
# Classifier latency + accuracy comparison
uv run python -m src.benchmark_classifiers

# Pipeline bottleneck analysis
uv run python -m src.profile_pipeline
```

### Run Tests

```bash
uv run pytest tests/ -v
```

## Architecture

### Training Pipeline

```
Raw CSV → Load → Preprocess → Extract Epochs → Extract Features → Train → Save Models
```

1. **data_loader** - Load Unicorn CSV with 8 EEG channels + stim markers. Filters for complete recordings (exactly 100 imagery trials).
2. **preprocess** - Bandpass (1-40 Hz), notch (60 Hz), CAR spatial filter.
3. **epochs** - Extract augmented baseline+task epoch pairs for Phase 3 and Phase 5 with sliding windows.
4. **features** - Three feature extraction paths:
   - **ERD features**: Event-related desynchronization using baseline/task pairs (training).
   - **Riemannian features**: Covariance matrices processed via pyriemann (training).
   - **Realtime features**: Absolute band powers, Hjorth parameters, envelope, ratios (inference).
5. **train** - Within-subject CV with 20+ classifiers using weighted ensemble voting. Three model families: Riemannian (MDM, tangent space+LDA/SVM), traditional (RF, ExtraTrees, GBM, SVM, LDA), and combined (ERD+Riemannian tangent features).

### Dual Model Design

| Model | Features | Use Case |
|-------|----------|----------|
| `engagement_classifier_model.joblib` | ERD (requires baseline) | Offline validation |
| `engagement_realtime_model.joblib` | Absolute powers only | Streaming inference |

Both are `RandomForestClassifier(n_estimators=200)` with `StandardScaler`.

### Streaming Architecture

Three-layer design for real-time inference:

1. **EEGStream** (`eeg_stream.py`) - Abstract base with concrete implementations:
   - `FileEEGStream`: Replay CSV sample-by-sample (testing/simulation)
   - `RandomEEGStream`: Synthetic EEG from summed sinusoids (unit testing)
   - `LiveEEGStream`: Unicorn Hybrid Black via UnicornPy SDK (production)
2. **RealtimeEngine** (`realtime_engine.py`) - Sample-by-sample processing:
   - `RingBuffer` accumulates multichannel samples
   - `IncrementalFilter` applies IIR bandpass/notch with persistent state (no edge artifacts)
   - Emits `WindowResult` at configurable slide intervals via `on_result` callback
3. **simulate_realtime** - CLI entry point supporting both batch and streaming modes

## Technical Details

### EEG Channels
8 channels from Unicorn Hybrid Black: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 at 250 Hz.

### Signal Processing
- Bandpass filter: 1-40 Hz
- Notch filter: 60 Hz
- Spatial filter: Common Average Reference (CAR)

### Feature Sets

**ERD Features** (training):
- Log band powers (delta, theta, alpha, beta)
- Relative band powers
- Band asymmetries (C3 vs C4, Fz vs Pz)
- Inter-channel ratios

**Realtime Features** (inference):
- Absolute/relative band powers
- Theta/Alpha and Theta/Beta ratios
- Hjorth parameters (activity, mobility, complexity)
- Envelope features

**Optimized Realtime** (`features_fast.py`):
- Single Welch PSD per channel with FFT bin masking
- Filter bank features from PSD

### Model Ensemble
Weighted voting classifier combining 20+ classifiers:
- Riemannian: MDM, tangent space + LDA/SVM
- Traditional: Random Forest, Extra Trees, GBM, SVM, LDA, Logistic Regression
- Combined: ERD + Riemannian tangent features

Feature selection via `SelectKBest(f_classif)` at 30%, 50%, and 70% thresholds.

## Model Performance

| Metric | Value |
|--------|-------|
| Mean Accuracy | 92.8% |
| Std Dev | ±3.76% |
| Subjects | 10 |
| Target Met | Yes (≥90%) |

## Dependencies

- **pyriemann** - Riemannian geometry for BCI (covariance, MDM, tangent space)
- **MNE** - EEG signal processing
- **scikit-learn** - ML classifiers, feature selection, cross-validation
- **XGBoost / LightGBM** - Gradient boosting classifiers
- **scipy** - Signal processing (Welch PSD, IIR filters, Hilbert transform)
- **pandas / numpy** - Data handling
- **joblib** - Model serialization
