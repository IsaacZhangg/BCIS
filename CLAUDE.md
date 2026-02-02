# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_features.py

# Run a specific test
uv run pytest tests/test_features.py::test_extract_erd_features

# Lint and format
uv run ruff check .
uv run ruff format .

# Run the training pipeline
uv run python -m src.pipeline
```

## Project Overview

EEG motor imagery classifier achieving **90.4% accuracy** (within-subject 10-fold CV). Distinguishes focused (phase 3 imagery) from not-focused (phase 4 execution) states using ERD features and Riemannian geometry.

| Class | Phase | Description |
|-------|-------|-------------|
| **1 (Focused)** | Phase 3 | Motor imagery performance |
| **0 (Not-Focused)** | Phase 4 | Actual motor execution |

## Architecture

### Data Flow

```
unicorn-data/subject*/session*/*.csv
        ↓
    data_loader.py    → (n_channels, n_samples), events list
        ↓
    preprocess.py     → Bandpass 1-40Hz + Notch 60Hz per channel
        ↓
    epochs.py         → Extract task/baseline pairs per phase
        ↓
    features.py       → ERD features + multichannel arrays for Riemannian
        ↓
    train.py          → Within-subject CV with Riemannian + traditional ML ensemble
        ↓
    models/*.joblib   → Final model + scaler
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `data_loader.py` | Load CSV, extract 8 EEG channels, parse stim markers into events |
| `preprocess.py` | MNE-based filtering (bandpass + notch) |
| `epochs.py` | `extract_erd_epochs()` returns (baseline, task) pairs for two phases |
| `features.py` | `extract_erd_features()` for traditional ML, multichannel arrays for Riemannian |
| `train.py` | `train_within_subject_cv_riemannian()` is the main evaluation function |
| `pipeline.py` | Orchestrates everything, saves to `models/` |

### Classifiers Used

| Method | Type |
|--------|------|
| MDM (Minimum Distance to Mean) | Riemannian |
| Tangent Space + LDA | Riemannian |
| Tangent Space + SVM | Riemannian |
| Random Forest | Traditional ML |
| Extra Trees | Traditional ML |
| Gradient Boosting | Traditional ML |
| SVM (RBF kernel, multiple C values) | Traditional ML |
| LDA | Traditional ML |

**Covariance Estimators:** LWF, OAS, SCM (for Riemannian methods)

**Feature Selection:** SelectKBest with F-statistic at k=30%, 50%, 70%, 90%

## Data Format Reference

### Unicorn CSV Structure

| Column | Description |
|--------|-------------|
| EEG 1-8 | Raw EEG voltage (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8) |
| stim | Stimulus marker (0 = no event) |

### Electrode Positions

```
        Fz
         |
    C3--Cz--C4
         |
        Pz
       / | \
    PO7  Oz  PO8
```

| Channel | Position | Role |
|---------|----------|------|
| Fz | Frontal midline | Frontal activity |
| C3 | Left motor cortex | Right hand imagery |
| Cz | Central midline | General motor |
| C4 | Right motor cortex | Left hand imagery |
| Pz | Parietal midline | Attention, integration |
| PO7 | Left parieto-occipital | Visual processing |
| Oz | Occipital midline | Visual processing |
| PO8 | Right parieto-occipital | Visual processing |

### Stimulus Marker Encoding

Each non-zero `stim` value encodes: `trial_number · phase · movement`

| Component | Position | Description |
|-----------|----------|-------------|
| **trial_number** | Leading digits | Sequential trial index (1, 2, 3, ...) |
| **phase** | Second-to-last digit | Which part of the trial |
| **movement** | Last digit | Movement class (1=left, 2=right) |

**Phase Codes:**

| Code | Phase | Description |
|------|-------|-------------|
| 1 | Video | Video presentation |
| 2 | Instruction | Instruction display |
| 3 | Imagery-perform | **Target for decoding (focused)** |
| 4 | Actual movement | Physical execution (not-focused) |
| 5 | Rest | Rest period |

**Examples:**
- `131` → trial=1, phase=3 (imagery), movement=1 (left)
- `1042` → trial=10, phase=4 (execution), movement=2 (right)

**Parsing:**
```python
phase = (stim // 10) % 10
movement = stim % 10
trial = stim // 100
```

**Session Statistics:**
- 100 imagery-perform markers per session (50 per hand)
- `stim = 0` for most rows (continuous recording)

## Hardware Reference

### Unicorn Hybrid Black EEG Headset

| Specification | Value |
|---------------|-------|
| Channels | 8 |
| Sampling Rate | 250 Hz |
| Resolution | 24-bit |
| Connectivity | Bluetooth |
| Electrode System | 10-20 standard |

**Data Location:** `unicorn-data/subject####/session###/recording_*.csv`

**Complete recordings:** 10 subjects with 100 imagery trials each

## EEG Fundamentals

### Frequency Bands

| Band | Frequency | Project Use |
|------|-----------|-------------|
| Delta | 1-4 Hz | Low-frequency activity |
| Theta | 4-8 Hz | Focus indicator |
| Low Alpha | 8-10 Hz | ERD component |
| High Alpha | 10-13 Hz | ERD component |
| Mu | 8-12 Hz | **Motor imagery primary** |
| Low Beta | 13-20 Hz | ERD secondary |
| High Beta | 20-30 Hz | ERD secondary |
| Gamma | 30-40 Hz | High-frequency activity |

### Event-Related Desynchronization (ERD)

Key biomarker for motor imagery classification:

```
ERD% = (baseline_power - task_power) / baseline_power × 100
```

- **Positive ERD** = power decrease during task (typical for motor imagery)
- Strongest in **mu (8-13 Hz)** and **beta (13-30 Hz)** bands
- **Contralateral**: Left hand imagery → right motor cortex (C4) ERD

### Signal Processing

| Filter | Parameters | Purpose |
|--------|------------|---------|
| Bandpass | 1-40 Hz | Remove DC drift and high-frequency noise |
| Notch | 60 Hz | Remove power line interference |

## Code Examples

### Loading and Processing Data

```python
from src.data_loader import load_recording, get_complete_recordings
from src.preprocess import preprocess_eeg
from src.epochs import extract_erd_epochs
from src.features import extract_erd_features

# Find and load data
recordings = get_complete_recordings("unicorn-data/")
data, events, sfreq = load_recording(recordings[0])

# Preprocess
filtered = preprocess_eeg(data, sfreq)

# Extract epochs
task_epochs, baseline_epochs = extract_erd_epochs(filtered, events, sfreq)

# Extract features
X = extract_erd_features(task_epochs, baseline_epochs, sfreq)
```

### Loading Trained Model

```python
import joblib

model = joblib.load("models/theta_classifier_model.joblib")
scaler = joblib.load("models/theta_classifier_scaler.joblib")

X_scaled = scaler.transform(X_new)
predictions = model.predict(X_scaled)
```

### Computing ERD

```python
from scipy.signal import welch

def compute_erd(task_epoch, baseline_epoch, sfreq, band=(8, 13)):
    freqs, psd_task = welch(task_epoch, sfreq, nperseg=sfreq//2)
    _, psd_base = welch(baseline_epoch, sfreq, nperseg=sfreq//2)

    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    power_task = psd_task[band_mask].mean()
    power_base = psd_base[band_mask].mean()

    return (power_base - power_task) / power_base * 100
```

## Trained Model Files

| File | Description |
|------|-------------|
| `models/theta_classifier_model.joblib` | Random Forest classifier |
| `models/theta_classifier_scaler.joblib` | StandardScaler |
| `models/training_results.json` | Performance metrics |

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Bandpass | 1-40 Hz | Retain EEG bands, remove artifacts |
| Notch | 60 Hz | Remove power line noise |
| Epoch duration | 2 seconds | Sufficient for ERD computation |
| CV folds | 10 | Robust performance estimate |

## Resources

### Project Documentation

- `docs/plans/2026-01-31-theta-focus-classifier-design.md` - Architecture and methodology
- `docs/plans/2026-01-31-theta-focus-classifier-implementation.md` - Task-by-task implementation

### External References

| Resource | URL |
|----------|-----|
| MNE-Python | https://mne.tools/stable/ |
| scikit-learn | https://scikit-learn.org/stable/ |
| pyRiemann | https://pyriemann.readthedocs.io/ |

### Key Papers

- Glaser et al., "Machine Learning for Neural Decoding" (eNeuro, 2020)
- Barachant et al., "Riemannian Geometry Applied to BCI Classification" (2012)
