# BCIS - Left/Right Motor Imagery BCI Classifier

A brain-computer interface (BCI) system that classifies left vs right hand motor imagery from EEG signals. Designed for real-time control of a robotic 6th finger, where **left imagery = finger down** and **right imagery = finger up**.

Built on a **Filter-Bank Common Spatial Patterns (FBCSP) + Linear Discriminant Analysis (LDA)** pipeline, evaluated with within-subject 10-fold stratified cross-validation.

## Table of Contents

- [Results](#results)
- [Hardware](#hardware)
- [Pipeline Overview](#pipeline-overview)
- [Architecture](#architecture)
  - [Data Loading](#1-data-loading)
  - [Preprocessing](#2-preprocessing)
  - [Epoch Extraction](#3-epoch-extraction)
  - [Feature Extraction](#4-feature-extraction)
  - [Training & Classification](#5-training--classification)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Data Format](#data-format)
- [Model Format](#model-format)
- [Testing](#testing)
- [Dependencies](#dependencies)

## Results

**57.2% mean accuracy** across 10 subjects (within-subject 10-fold stratified CV).

2 of 10 subjects show above-chance classification. This is consistent with the literature on consumer-grade EEG -- signal quality and motor imagery aptitude vary significantly between individuals.

| Subject | Accuracy | Status |
|---------|----------|--------|
| subject0001 | 54% | chance |
| subject0002 | 51% | chance |
| subject0004 | 42% | chance |
| subject0005 | 45% | chance |
| subject0006 | **88%** | signal |
| subject0007 | 54% | chance |
| subject0008 | 57% | chance |
| subject0009 | 42% | chance |
| subject0010 | **82%** | signal |
| subject0011 | 57% | chance |

Subjects with signal (>=60% accuracy) are viable candidates for real-time 6th finger control. The remaining subjects perform at chance level (~42-57%), likely due to low signal-to-noise ratio from the consumer headset or difficulty producing distinguishable motor imagery patterns.

## Hardware

- **Headset**: [g.tec Unicorn Hybrid Black](https://www.unicorn-bi.com/) -- consumer-grade EEG
- **Channels**: 8 dry electrodes (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8)
- **Sampling rate**: 250 Hz
- **Resolution**: 24-bit ADC

## Pipeline Overview

```
CSV files (8 channels, 250 Hz, Unicorn headset)
  │
  ├─ data_loader.py ── Parse recordings, extract event markers
  │
  ├─ preprocess.py ─── Bandpass (1-40 Hz) + Notch (60 Hz) filtering
  │
  ├─ epochs.py ──────── Extract paired baseline (1.0s) / task (1.8s) windows
  │
  ├─ features.py ────── 39 handcrafted features:
  │                        Lateralization indices, ERD asymmetry,
  │                        Hjorth parameters, frontal theta
  │
  ├─ train.py ─────────  FBCSP (10 bands × 4 CSP components = 40 features)
  │                      + 39 handcrafted features
  │                      → SelectKBest(k=20)
  │                      → StandardScaler
  │                      → LDA (shrinkage)
  │
  └─ pipeline.py ────── Orchestrates everything, runs 10-fold CV,
                         saves per-subject models
```

## Architecture

### 1. Data Loading

**Module**: `src/data_loader.py`

Parses Unicorn EEG recordings from CSV files. Each recording contains 8 EEG channels plus a stimulus marker column.

**Event encoding**: The `stim` column encodes events as `phase * 10 + movement`:
- Phase 3 = motor imagery period (the only phase used for classification)
- Movement 1 = left hand imagery
- Movement 2 = right hand imagery
- Example: `stim = 31` means phase 3, left imagery; `stim = 32` means phase 3, right imagery

Each complete recording contains exactly **100 phase-3 trials** (50 left, 50 right).

### 2. Preprocessing

**Module**: `src/preprocess.py`

Two-stage filtering applied per channel using MNE-Python's FIR filters:

1. **Bandpass filter** (1-40 Hz) -- removes DC drift below 1 Hz and high-frequency noise above 40 Hz, retaining the physiologically relevant EEG bands (delta through low gamma)
2. **Notch filter** (60 Hz) -- removes power line interference (North American AC frequency)

### 3. Epoch Extraction

**Module**: `src/epochs.py`

Extracts time-locked windows around each motor imagery event marker. Each trial produces a **paired baseline + task epoch**:

```
       ←── 1.0s ──→←── event marker
       [  baseline  ][ 0.5s skip ][ ─── 1.8s task ─── ]
```

- **Baseline**: 1.0 second immediately before the event marker (resting state reference)
- **Skip**: 0.5 seconds after the event marker (accounts for reaction time / instruction processing)
- **Task**: 1.8 seconds of motor imagery signal (the classification target)

Trials are discarded if the baseline would start before the signal beginning or the task would extend past the signal end.

### 4. Feature Extraction

**Module**: `src/features.py`

Two feature sets are computed and later concatenated:

#### Handcrafted Features (39 total)

Computed from baseline-vs-task comparisons using Welch's method for spectral analysis:

| Feature Group | Channels | Bands | Features per Band | Count |
|---|---|---|---|---|
| C3/C4 lateralization | C3, C4 | mu (8-12 Hz), low beta (13-20 Hz), high beta (20-30 Hz), beta (13-30 Hz) | lateralization (task), lateralization (baseline), lateralization difference, ERD asymmetry, log power ratio, ERD C3, ERD C4 | 4 × 7 = **28** |
| Cz supplementary motor area | Cz | mu (8-12 Hz), beta (13-30 Hz) | ERD, log power | 2 × 2 = **4** |
| Fz frontal theta | Fz | theta (4-8 Hz) | log(theta_task / theta_baseline) | **1** |
| Time-domain | C3, C4 | -- | Hjorth activity, mobility, complexity | 2 × 3 = **6** |

**Key formulas**:
- **Lateralization Index**: `(C4_power - C3_power) / (C4_power + C3_power)` -- positive for left imagery (contralateral C3 desynchronization), negative for right
- **ERD (Event-Related Desynchronization)**: `(baseline_power - task_power) / baseline_power × 100` -- percentage power decrease during motor imagery
- **Hjorth parameters**: Activity (variance), Mobility (mean frequency), Complexity (bandwidth) -- efficient time-domain descriptors

#### FBCSP Features (up to 40)

Computed in `src/train.py` during the training loop (fitted per fold to avoid data leakage):

- **10 frequency bands**: (4-8), (8-10), (10-12), (12-14), (14-16), (16-18), (18-20), (20-24), (24-30), (30-40) Hz
- **4 CSP components per band**: spatial filters maximizing variance ratio between left and right classes
- **Log-variance transformation**: CSP features are log-transformed for Gaussian-like distribution
- **Ledoit-Wolf regularization**: robust covariance estimation for small sample sizes

### 5. Training & Classification

**Module**: `src/train.py`

Per-subject, per-fold pipeline:

1. **FBCSP extraction** -- bandpass filter to each of 10 bands, fit CSP on training set, transform both train and test (up to 40 features)
2. **Feature concatenation** -- FBCSP features (40) + handcrafted features (39) = ~79 features
3. **Feature selection** -- `SelectKBest(f_classif, k=20)` picks the 20 most discriminative features by ANOVA F-score
4. **Scaling** -- `StandardScaler` zero-centers and unit-normalizes features
5. **Classification** -- `LDA(solver='lsqr', shrinkage='auto')` with automatic Ledoit-Wolf shrinkage for robust covariance estimation

**Evaluation**: 10-fold stratified cross-validation, per subject. CSP models are re-fitted on each fold's training set to prevent information leakage.

**Deployment models**: After CV evaluation, a final model is trained on all data per subject and saved as a joblib dict containing the full inference pipeline (CSP models, feature selector, scaler, LDA classifier).

## Project Structure

```
BCIS/
├── src/
│   ├── __init__.py
│   ├── pipeline.py          # Main pipeline orchestrator (entry point)
│   ├── train.py              # FBCSP + LDA training, CV, inference
│   ├── data_loader.py        # CSV loading, event parsing
│   ├── preprocess.py         # Bandpass + notch filtering
│   ├── epochs.py             # Epoch extraction (baseline + task pairs)
│   └── features.py           # Handcrafted features + CSP
├── tests/
│   ├── test_data_loader.py   # Data loading and recording discovery
│   ├── test_preprocess.py    # Filter correctness (DC removal, 60Hz attenuation)
│   ├── test_epochs.py        # Epoch extraction and boundary conditions
│   ├── test_features.py      # Feature shapes, values, missing channel errors
│   └── test_train.py         # CV output, leakage check, model completeness
├── models/                    # Per-subject trained models (.joblib)
│   ├── subject0001_fbcsp_lda.joblib
│   ├── ...
│   ├── subject0011_fbcsp_lda.joblib
│   └── training_results.json # Accuracy metrics and model paths
├── unicorn-data/              # EEG recordings (gitignored)
├── pyproject.toml             # Project config and dependencies
├── uv.lock                    # Dependency lock file
├── CLAUDE.md                  # Claude Code project instructions
└── README.md
```

## Setup

**Requirements**: Python 3.11+ and [uv](https://docs.astral.sh/uv/)

```bash
# Clone the repository
git clone <repo-url>
cd BCIS

# Install dependencies
uv sync
```

### Data Setup

EEG recordings are not included in the repository. Place Unicorn headset CSV exports in the `unicorn-data/` directory following this structure:

```
unicorn-data/
├── subject0001/
│   └── session001/
│       └── recording_001.csv
├── subject0002/
│   └── session001/
│       └── recording_001.csv
└── ...
```

## Usage

### Run the full pipeline

Loads data, preprocesses, extracts features, runs 10-fold CV per subject, trains final models, and saves results:

```bash
uv run python -m src.pipeline
```

Output:
- Per-subject `.joblib` models in `models/`
- `models/training_results.json` with accuracy metrics

### Use a trained model for inference

```python
import joblib
import numpy as np
from src.train import predict
from src.features import extract_lateralization_features
from src.preprocess import preprocess_eeg
from src.data_loader import CHANNELS

# Load a per-subject model
model = joblib.load("models/subject0006_fbcsp_lda.joblib")

# Prepare your data:
# X_features: handcrafted features, shape (n_trials, 39)
# X_multichannel: raw multichannel EEG, shape (n_trials, 8, n_samples)
predictions = predict(model, X_features, X_multichannel)
# predictions: array of 0 (left) or 1 (right)
```

### Lint and format

```bash
ruff check --fix src/ tests/
ruff format src/ tests/
```

## Data Format

Each CSV file has the following columns:

| Column | Description |
|--------|-------------|
| `timestamp` | Sample timestamp |
| `Fz` | Frontal midline electrode (theta/attention) |
| `C3` | Left motor cortex (primary discriminative channel) |
| `Cz` | Central midline / supplementary motor area |
| `C4` | Right motor cortex (primary discriminative channel) |
| `Pz` | Parietal midline |
| `PO7` | Left parieto-occipital |
| `Oz` | Occipital midline |
| `PO8` | Right parieto-occipital |
| `stim` | Event marker (0 = no event, nonzero = phase×10 + movement) |

**Key channels for motor imagery**: C3 and C4 sit over the left and right primary motor cortex respectively. During left hand motor imagery, C3 (contralateral) shows mu/beta desynchronization; during right hand imagery, C4 does. This lateralization pattern is the primary discriminative signal.

## Model Format

Each saved `.joblib` file is a Python dictionary containing the full inference pipeline:

| Key | Type | Description |
|-----|------|-------------|
| `csp_models` | `list[tuple[CSP, tuple[float, float]]]` | Fitted CSP spatial filters per frequency band |
| `selector` | `SelectKBest` | Fitted feature selector (top 20 by ANOVA F-score) |
| `scaler` | `StandardScaler` | Fitted feature normalizer |
| `classifier` | `LinearDiscriminantAnalysis` | Fitted LDA with shrinkage |
| `sfreq` | `float` | Sampling frequency (250.0 Hz) |
| `k_best` | `int` | Number of selected features (20) |

## Testing

The test suite validates each pipeline stage with both correctness checks and edge cases:

```bash
# Run all tests
uv run pytest tests/ -v

# Run a specific test file
uv run pytest tests/test_train.py -v
```

**Test coverage**:
- `test_data_loader.py` -- CSV parsing, event extraction, complete recording discovery
- `test_preprocess.py` -- DC offset removal, 60 Hz attenuation (>90% power reduction), full pipeline
- `test_epochs.py` -- Epoch shapes, class separation, boundary conditions (signal start/end)
- `test_features.py` -- Feature dimensions, lateralization index symmetry, missing channel errors, CSP output shapes
- `test_train.py` -- CV score ranges, data leakage detection (random data <= 70%), model dict completeness, round-trip predict

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| [MNE-Python](https://mne.tools/) | >=1.11.0 | EEG signal processing, FIR filters, CSP implementation |
| [scikit-learn](https://scikit-learn.org/) | >=1.8.0 | LDA classifier, SelectKBest, StratifiedKFold, StandardScaler |
| [NumPy](https://numpy.org/) | >=2.4.2 | Array operations |
| [SciPy](https://scipy.org/) | >=1.17.0 | Welch's method for spectral analysis |
| [pandas](https://pandas.pydata.org/) | >=3.0.0 | CSV loading |
| [joblib](https://joblib.readthedocs.io/) | >=1.5.3 | Model serialization |

**Dev dependencies**: pytest (>=9.0.2), ruff (>=0.14.14)
