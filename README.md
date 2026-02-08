# BCIS - Left/Right Motor Imagery BCI Classifier

A brain-computer interface (BCI) system that classifies left vs right hand motor imagery from EEG signals. Designed for real-time control of a robotic 6th finger, where **left imagery = finger down** and **right imagery = finger up**.

Built on a triple-classifier pipeline — **FBCSP + LDA**, **Riemannian tangent-space**, and **Transfer learning (Riemannian domain adaptation)** — with three-criteria adaptive artifact rejection. The pipeline evaluates all three classifiers per subject and selects the best one. Evaluated with within-subject 10-fold stratified cross-validation.

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

**58.9% best-of mean accuracy** across 10 subjects (within-subject 10-fold stratified CV, best of three classifiers per subject).

3 of 10 subjects show above-chance classification. This is consistent with the literature on consumer-grade EEG -- signal quality and motor imagery aptitude vary significantly between individuals.

| Subject | FBCSP+LDA | Riemann | Transfer | Best | Status |
|---------|-----------|---------|----------|------|--------|
| subject0001 | 50.1% | 53.1% | 47.2% | 53.1% | chance |
| subject0002 | 38.5% | 52.9% | 36.8% | 52.9% | chance |
| subject0004 | 36.1% | 30.1% | 37.4% | 37.4% | chance |
| subject0005 | 56.2% | 43.8% | 47.7% | 56.2% | chance |
| subject0006 | **83.8%** | 31.2% | 45.2% | **83.8%** | signal |
| subject0007 | 56.1% | 44.3% | 39.3% | 56.1% | chance |
| subject0008 | 52.5% | 46.2% | 53.8% | 53.8% | chance |
| subject0009 | 46.8% | 56.5% | 46.4% | 56.5% | chance |
| subject0010 | **76.5%** | 65.0% | 67.5% | **76.5%** | signal |
| subject0011 | 56.4% | **62.3%** | 47.1% | **62.3%** | signal |

Subjects with signal (>=60% accuracy) are viable candidates for real-time 6th finger control. The remaining subjects perform at chance level (~37-57%), likely due to low signal-to-noise ratio from the consumer headset or difficulty producing distinguishable motor imagery patterns. The Riemannian classifier complements FBCSP+LDA on subjects 0001, 0002, 0009, and 0011. The Transfer classifier (cross-subject domain adaptation) wins for subjects 0004 and 0008 where within-subject data is insufficient.

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
  │                      + 3-criteria adaptive artifact rejection
  │                        (amplitude + gradient + HF power)
  │
  ├─ features.py ────── 59 handcrafted features:
  │                        Lateralization indices, ERD asymmetry,
  │                        Hjorth parameters, frontal theta,
  │                        spectral entropy, peak frequency,
  │                        C3-C4 coherence, posterior lateralization,
  │                        band power ratios, statistical features
  │
  ├─ train.py ─────────  Three classifiers evaluated per subject:
  │                      1. FBCSP (10 bands × 4 CSP = 40 features)
  │                         + 59 handcrafted → SelectKBest(k=15)
  │                         → StandardScaler → LDA (shrinkage)
  │                      2. Riemannian: Covariances(OAS)
  │                         → TangentSpace(riemann) → LogisticRegression
  │                      3. Transfer: TLCenter domain adaptation
  │                         → TangentSpace → LogisticRegression
  │
  └─ pipeline.py ────── Orchestrates everything, runs all three classifiers,
                         saves best model per subject
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

Two-stage temporal filtering applied per channel using MNE-Python's FIR filters:

1. **Bandpass filter** (1-40 Hz) -- removes DC drift below 1 Hz and high-frequency noise above 40 Hz, retaining the physiologically relevant EEG bands (delta through low gamma)
2. **Notch filter** (60 Hz) -- removes power line interference (North American AC frequency)

Also provides `common_average_reference()` (CAR) and `apply_asr()` (Artifact Subspace Reconstruction) as optional spatial filters, though neither is active in the pipeline -- CAR reduces rank below acceptable levels for 8-channel CSP, and ASR removes discriminative motor imagery variance.

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

After extraction, **three-criteria adaptive artifact rejection** drops trials that fail any of:

1. **Peak-to-peak amplitude** -- exceeds `median + 3×MAD` (adaptive per subject) or below 1 µV (flat signal)
2. **Gradient** -- maximum sample-to-sample voltage jump exceeds `median + 3×MAD` (electrode pops, movement artifacts)
3. **High-frequency power** -- abnormally high 30-45 Hz power exceeds `median + 3×MAD` (EMG contamination)

All thresholds are adaptive to each subject's signal characteristics rather than using fixed values.

### 4. Feature Extraction

**Module**: `src/features.py`

Two feature sets are computed and later concatenated:

#### Handcrafted Features (59 total)

Computed from baseline-vs-task comparisons using Welch's method for spectral analysis:

| Feature Group | Channels | Bands | Features per Band | Count |
|---|---|---|---|---|
| C3/C4 lateralization | C3, C4 | mu (8-12 Hz), low beta (13-20 Hz), high beta (20-30 Hz), beta (13-30 Hz) | lateralization (task), lateralization (baseline), lateralization difference, ERD asymmetry, log power ratio, ERD C3, ERD C4 | 4 × 7 = **28** |
| Cz supplementary motor area | Cz | mu (8-12 Hz), beta (13-30 Hz) | ERD, log power | 2 × 2 = **4** |
| Fz frontal theta | Fz | theta (4-8 Hz) | log(theta_task / theta_baseline) | **1** |
| Time-domain (Hjorth) | C3, C4 | -- | activity, mobility, complexity | 2 × 3 = **6** |
| Spectral entropy | C3, C4 | mu, beta | normalized entropy | 2 × 2 = **4** |
| Peak frequency | C3, C4 | mu (8-12 Hz) | individual peak freq | 2 × 1 = **2** |
| C3-C4 coherence | C3, C4 | mu, beta | magnitude-squared coherence | 2 × 1 = **2** |
| Posterior lateralization | PO7, PO8 | mu, beta | lateralization index, log total power | 2 × 2 = **4** |
| Band power ratios | C3 | mu/theta, beta/mu | log ratio | **2** |
| Statistical features | C3, C4 | -- | skewness, kurtosis, zero-crossing rate | 2 × 3 = **6** |

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

Three classifiers are evaluated per subject. The best per-subject classifier is selected for deployment.

#### Classifier 1: FBCSP + LDA

Per-subject, per-fold pipeline (10-fold stratified CV):

1. **FBCSP extraction** -- bandpass filter to each of 10 bands, fit CSP on training set, transform both train and test (up to 40 features)
2. **Feature concatenation** -- FBCSP features (40) + handcrafted features (59) = ~99 features
3. **Feature selection** -- `SelectKBest(f_classif, k=15)` picks the 15 most discriminative features by ANOVA F-score
4. **Scaling** -- `StandardScaler` zero-centers and unit-normalizes features
5. **Classification** -- `LDA(solver='lsqr', shrinkage='auto')` with automatic Ledoit-Wolf shrinkage for robust covariance estimation

#### Classifier 2: Riemannian Tangent Space

Per-subject, per-fold pipeline (10-fold stratified CV, no frequency band tuning or feature engineering):

1. **Covariance estimation** -- `Covariances(estimator='oas')` computes Oracle Approximating Shrinkage covariance matrices from multichannel EEG (8×8 SPD matrices)
2. **Tangent space projection** -- `TangentSpace(metric='riemann')` maps SPD matrices to Euclidean tangent space (36 features for 8 channels)
3. **Classification** -- `LogisticRegression(C=1.0, solver='lbfgs')`

#### Classifier 3: Transfer Learning (Riemannian Domain Adaptation)

Leave-one-subject-out CV with within-target k-fold:

1. **Covariance estimation** -- `Covariances(estimator='oas')` as above
2. **Domain encoding** -- `encode_domains()` labels each trial with source/target domain
3. **Riemannian re-centering** -- `TLCenter(target_domain='target')` re-centers each domain's covariance distribution to identity in Riemannian space, aligning cross-subject geometric means
4. **Tangent space projection** -- `TangentSpace(metric='riemann')`
5. **Classification** -- `LogisticRegression(C=1.0, solver='lbfgs')`

This classifier pools data from all other subjects as source data, helping subjects at chance by leveraging cross-subject motor imagery patterns.

**Evaluation**: 10-fold stratified cross-validation per subject (classifiers 1 & 2), leave-one-subject-out with within-target k-fold (classifier 3). CSP models are re-fitted on each fold's training set to prevent information leakage.

**Deployment models**: After CV evaluation, the best-scoring classifier is trained on all data per subject and saved as a `.joblib` file. FBCSP+LDA models contain the full inference pipeline (CSP models, feature selector, scaler, LDA classifier). Riemannian models contain a single sklearn Pipeline. Transfer models contain the covariance estimator, TLCenter, tangent space, and classifier.

## Project Structure

```
BCIS/
├── src/
│   ├── __init__.py
│   ├── pipeline.py          # Main pipeline orchestrator (entry point)
│   ├── train.py              # FBCSP+LDA & Riemannian training, CV, inference
│   ├── data_loader.py        # CSV loading, event parsing
│   ├── preprocess.py         # Bandpass + notch filtering, CAR, ASR
│   ├── epochs.py             # Epoch extraction + adaptive artifact rejection
│   └── features.py           # Handcrafted features + CSP
├── tests/
│   ├── test_data_loader.py   # Data loading and recording discovery
│   ├── test_preprocess.py    # Filter correctness (DC removal, 60Hz attenuation)
│   ├── test_epochs.py        # Epoch extraction and boundary conditions
│   ├── test_features.py      # Feature shapes, values, missing channel errors
│   └── test_train.py         # CV output, leakage check, model completeness
├── models/                    # Per-subject trained models (.joblib)
│   ├── subject0006_fbcsp_lda.joblib  # FBCSP+LDA where it wins
│   ├── subject0011_riemann.joblib    # Riemannian where it wins
│   ├── ...
│   └── training_results.json # Accuracy metrics, method comparison, model paths
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

# FBCSP+LDA model
model = joblib.load("models/subject0006_fbcsp_lda.joblib")
from src.train import predict
predictions = predict(model, X_features, X_multichannel)

# Riemannian model
model = joblib.load("models/subject0011_riemann.joblib")
from src.train import predict_riemann
predictions = predict_riemann(model, X_multichannel)

# Transfer model
model = joblib.load("models/subject0001_transfer.joblib")
from src.train import predict_transfer
predictions = predict_transfer(model, X_multichannel)

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

### FBCSP+LDA models (`*_fbcsp_lda.joblib`)

| Key | Type | Description |
|-----|------|-------------|
| `csp_models` | `list[tuple[CSP, tuple[float, float]]]` | Fitted CSP spatial filters per frequency band |
| `selector` | `SelectKBest` | Fitted feature selector (top 15 by ANOVA F-score) |
| `scaler` | `StandardScaler` | Fitted feature normalizer |
| `classifier` | `LinearDiscriminantAnalysis` | Fitted LDA with shrinkage |
| `sfreq` | `float` | Sampling frequency (250.0 Hz) |
| `k_best` | `int` | Number of selected features (15) |

### Riemannian models (`*_riemann.joblib`)

| Key | Type | Description |
|-----|------|-------------|
| `pipeline` | `sklearn.pipeline.Pipeline` | Full pipeline: Covariances → TangentSpace → LogisticRegression |

### Transfer models (`*_transfer.joblib`)

| Key | Type | Description |
|-----|------|-------------|
| `cov_estimator` | `Covariances` | Fitted OAS covariance estimator |
| `tlc` | `TLCenter` | Fitted Riemannian domain re-centering transform |
| `ts` | `TangentSpace` | Fitted tangent space projection |
| `classifier` | `LogisticRegression` | Fitted logistic regression classifier |

## Testing

The test suite validates each pipeline stage with both correctness checks and edge cases:

```bash
# Run all tests
uv run pytest tests/ -v

# Run a specific test file
uv run pytest tests/test_train.py -v
```

**Test coverage** (46 tests):
- `test_data_loader.py` -- CSV parsing, event extraction, complete recording discovery
- `test_preprocess.py` -- DC offset removal, 60 Hz attenuation (>90% power reduction), full pipeline, CAR common-mode removal, ASR artifact cleaning
- `test_epochs.py` -- Epoch shapes, class separation, boundary conditions (signal start/end), artifact rejection (amplitude, flat signal, adaptive outlier, gradient, HF power, criteria disable)
- `test_features.py` -- Feature dimensions (59 features), lateralization index symmetry, missing channel errors, CSP output shapes, spectral entropy, peak frequency, C3-C4 coherence, statistical features
- `test_train.py` -- FBCSP+LDA CV scores, leakage check, model completeness, round-trip predict; Riemannian CV scores, leakage check, model round-trip; Transfer CV scores, leakage check, model round-trip

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| [MNE-Python](https://mne.tools/) | >=1.11.0 | EEG signal processing, FIR filters, CSP implementation |
| [scikit-learn](https://scikit-learn.org/) | >=1.8.0 | LDA, LogisticRegression, SelectKBest, StratifiedKFold, StandardScaler |
| [pyriemann](https://pyriemann.readthedocs.io/) | >=0.7 | Riemannian geometry: covariance estimation, tangent space projection |
| [asrpy](https://github.com/DiGyt/asrpy) | >=0.0.8 | Artifact Subspace Reconstruction (optional, includes numpy 2.x compat patch) |
| [NumPy](https://numpy.org/) | >=2.4.2 | Array operations |
| [SciPy](https://scipy.org/) | >=1.17.0 | Welch's method for spectral analysis |
| [pandas](https://pandas.pydata.org/) | >=3.0.0 | CSV loading |
| [joblib](https://joblib.readthedocs.io/) | >=1.5.3 | Model serialization |

**Dev dependencies**: pytest (>=9.0.2), ruff (>=0.14.14)
