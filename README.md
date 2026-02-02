# BCIS - EEG-Based Fatigue Detection System

A real-time fatigue and engagement detection system using EEG signals from the Unicorn BCI headset. The system classifies brain states to detect when a patient becomes disengaged during motor imagery sessions, enabling proactive intervention.

## Overview

This project implements a classifier that distinguishes between:
- **Phase 3 (Motor Imagery)**: Engaged brain state
- **Phase 5 (Rest)**: Disengaged/fatigued brain state

The system achieves **92.8% mean accuracy** across subjects using ensemble machine learning methods.

## Features

- **Proactive Fatigue Detection**: Identifies disengagement before the user notices
- **Sliding Window Processing**: Real-time simulation with 5-second updates
- **Dual Model Architecture**:
  - Full-feature model (ERD-based) for validation
  - Realtime-only model for streaming without baseline
- **Configurable Alerts**: Customizable threshold and consecutive window parameters
- **Comprehensive Signal Processing**: Bandpass, notch, and spatial filtering

## Project Structure

```
BCIS/
├── src/                          # Core source code
│   ├── data_loader.py           # Load & parse Unicorn EEG recordings
│   ├── preprocess.py            # Bandpass, notch, CAR, Laplacian filters
│   ├── epochs.py                # Extract baseline-task epoch pairs (ERD)
│   ├── features.py              # Multi-band power + Hjorth + envelope features
│   ├── train.py                 # Within-subject CV + ensemble models
│   ├── pipeline.py              # Main training orchestration
│   └── simulate_realtime.py     # Sliding window engagement simulation
├── tests/                        # Test suite
├── models/                       # Trained model artifacts
├── unicorn-data/                 # EEG recordings (12 subjects)
└── docs/plans/                   # Design documentation
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd BCIS

# Install dependencies with uv
uv sync
```

## Usage

### Train Models

```bash
python -m src.pipeline
```

This will:
1. Load EEG recordings from `unicorn-data/`
2. Preprocess and extract features
3. Train ensemble classifiers with within-subject cross-validation
4. Save models to `models/`

### Run Real-Time Simulation

```bash
python -m src.simulate_realtime unicorn-data/subject0001/session000/recording_*.csv \
  --threshold 30 \
  --window 5.0 \
  --consecutive 3
```

Options:
- `--threshold`: Engagement score threshold for alerts (default: 30)
- `--window`: Window size in seconds (default: 5.0)
- `--consecutive`: Consecutive low windows before alert (default: 3)

### Run Tests

```bash
python -m pytest tests/ -v
```

## Technical Details

### EEG Channels
8 channels from Unicorn headset: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 at 250 Hz sampling rate.

### Signal Processing
- Bandpass filter: 1-40 Hz
- Notch filter: 60 Hz (AC power)
- Spatial filters: Common Average Reference (CAR), Laplacian

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

### Model Ensemble
Voting classifier combining:
- Random Forest
- Gradient Boosting
- Extra Trees
- Riemannian MDM
- SVM, LDA, Logistic Regression

## Model Performance

| Metric | Value |
|--------|-------|
| Mean Accuracy | 92.8% |
| Std Dev | ±3.76% |
| Subjects | 10 |
| Target Met | Yes (≥90%) |

## Dependencies

- numpy, scipy, pandas
- scikit-learn, xgboost
- pyriemann (Riemannian geometry classifiers)
- mne (neuroscience signal processing)
- torch (deep learning capable)
- joblib (model serialization)

## License

[Add license information]
