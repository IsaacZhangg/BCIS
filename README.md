# BCIS - Left/Right Motor Imagery BCI Classifier

A brain-computer interface (BCI) system that classifies left vs right hand motor imagery from EEG signals. Designed for real-time control of a robotic 6th finger, where **left imagery = finger down** and **right imagery = finger up**.

Built on a four-classifier pipeline — **FBCSP + LDA**, **Riemannian tangent-space**, **FBCSP + SVM (RBF)**, and **LDA+SVM Ensemble** — with **in-fold adaptive artifact rejection** (threshold computed from training fold only), **group-aware stratified CV** (to reduce temporal-neighbor leakage), and Surface Laplacian spatial filtering. The pipeline uses **nested model-selection CV** to select the best classifier per subject without selection bias, and supports optional held-out evaluation and cross-session validation.

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

**61.2% nested selection mean accuracy** (unbiased) across 10 subjects, with 63.2% best-of mean (optimistic, for reference). Evaluated with within-subject nested model-selection CV: outer 10-fold selects the best classifier per fold via inner 7-fold CV, with **in-fold artifact rejection** (threshold computed from training fold only), eliminating both post-hoc selection bias and rejection threshold leakage.

5 of 10 subjects show above-chance classification (>=60%). This is consistent with the literature on consumer-grade EEG -- signal quality and motor imagery aptitude vary significantly between individuals.

| Subject | FBCSP+LDA | Riemann | SVM | Ensemble | Best (optimistic) | Nested (unbiased) | Selected | Status |
|---------|-----------|---------|-----|----------|-------------------|-------------------|----------|--------|
| subject0001 | 48.5% | 41.9% | 49.3% | 45.7% | 49.3% | **45.9%** | SVM | chance |
| subject0002 | 40.1% | 41.7% | 43.2% | 35.4% | 43.2% | **41.0%** | SVM | chance |
| subject0004 | 42.8% | 29.9% | 56.7% | 44.2% | 56.7% | **54.7%** | SVM | chance |
| subject0005 | 56.8% | 53.4% | 55.6% | 55.5% | 56.8% | **55.9%** | FBCSP | chance |
| subject0006 | 90.0% | 42.0% | 87.0% | 85.0% | 90.0% | **88.0%** | FBCSP | signal |
| subject0007 | 64.0% | 45.0% | 50.0% | 61.0% | 64.0% | **66.0%** | FBCSP | signal |
| subject0008 | 69.8% | 52.8% | 57.1% | 63.4% | 69.8% | **65.6%** | FBCSP | signal |
| subject0009 | 49.1% | 56.1% | 54.9% | 47.6% | 56.1% | **52.3%** | Riemann | chance |
| subject0010 | 78.0% | 60.3% | 85.3% | 83.1% | 85.3% | **82.1%** | FBCSP | signal |
| subject0011 | 53.6% | 60.4% | 48.7% | 49.6% | 60.4% | **60.4%** | Riemann | signal |

Subjects with signal (>=60% accuracy) are viable candidates for real-time 6th finger control. The remaining subjects perform at chance level (~41-56%), likely due to low signal-to-noise ratio from the consumer headset or difficulty producing distinguishable motor imagery patterns. The Riemannian classifier complements FBCSP on some subjects. The nested CV selects the classifier method per subject without the inflated accuracy of post-hoc selection.

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
  ├─ epochs.py ──────── Extract paired baseline (1.0s) / task (3.0s) windows
  │                      + per-trial PTP computation for in-fold rejection
  │
  ├─ features.py ────── 45 handcrafted features:
  │                        Surface Laplacian filtered C3/C4,
  │                        Lateralization indices, ERD asymmetry,
  │                        Hjorth params, C3-C4 coherence,
  │                        spectral entropy, Cz motor area, frontal theta
  │
  ├─ train.py ─────────  Four classifiers evaluated per subject:
  │                      In-fold artifact rejection (threshold from train only)
  │                      1. FBCSP (8 bands × 4 CSP = 32 features)
  │                         + 45 handcrafted → SelectKBest(nested CV k)
  │                         → StandardScaler → LDA (shrinkage + class priors)
  │                      2. Riemannian: Covariances(OAS)
  │                         → TangentSpace(riemann) → LogisticRegression
  │                      3. FBCSP + SVM: same FBCSP features
  │                         → SelectKBest(k) → SVC(RBF, C=10)
  │                      4. Ensemble: LDA + SVM soft voting
  │
  └─ pipeline.py ────── Orchestrates everything, runs all four classifiers
                         + nested model-selection CV, optional held-out split,
                         cross-session detection, saves models per subject
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
       [  baseline  ][ skip ][ ────── 3.0s task ────── ]
```

- **Baseline**: 1.0 second immediately before the event marker (resting state reference)
- **Skip**: 0.25 seconds after the event marker (accounts for initial visual evoked potential)
- **Task**: 3.0 seconds of motor imagery signal (the classification target)

Trials are discarded if the baseline would start before the signal beginning or the task would extend past the signal end.

After extraction, per-trial **max peak-to-peak amplitude** is computed across channels via `compute_trial_max_ptp()`. Artifact rejection is performed **inside each CV fold** rather than globally: the threshold (`median + 3.5×MAD`) is computed from training-fold trials only, then applied to both train and test splits. This prevents the rejection threshold from leaking test-fold information. Flat trials (PTP < 1 µV) are also removed.

For deployment models (trained on all data), global rejection is applied once using the full-dataset threshold. Gradient and high-frequency power criteria are available in the code but disabled by default -- they were found to reject trials containing discriminative motor imagery signal.

### 4. Feature Extraction

**Module**: `src/features.py`

Two feature sets are computed and later concatenated:

#### Handcrafted Features (45 total)

Computed from baseline-vs-task comparisons using Welch's method for spectral analysis. C3/C4 features use **Surface Laplacian** spatial filtering to sharpen motor cortex signals: C3_lap = C3 − mean(Fz, Cz, PO7), C4_lap = C4 − mean(Cz, Pz, PO8).

| Feature Group | Channels | Bands | Features per Band | Count |
|---|---|---|---|---|
| C3/C4 lateralization | C3_lap, C4_lap | mu (8-12 Hz), low beta (13-20 Hz), high beta (20-30 Hz), beta (13-30 Hz) | lateralization (task), lateralization (baseline), lateralization difference, ERD asymmetry, log power ratio, ERD C3, ERD C4 | 4 × 7 = **28** |
| Cz supplementary motor area | Cz | mu (8-12 Hz), beta (13-30 Hz) | ERD, log power | 2 × 2 = **4** |
| Fz frontal theta | Fz | theta (4-8 Hz) | log(theta_task / theta_baseline) | **1** |
| Time-domain (Hjorth) | C3_lap, C4_lap | -- | activity, mobility, complexity | 2 × 3 = **6** |
| C3-C4 coherence | C3_lap, C4_lap | mu (8-12 Hz), beta (13-30 Hz) | coherence (task), coherence change (task - baseline) | 2 × 2 = **4** |
| Spectral entropy lateralization | C3_lap, C4_lap | mu (8-12 Hz), beta (13-30 Hz) | C3 entropy - C4 entropy | 2 × 1 = **2** |

**Key formulas**:
- **Lateralization Index**: `(C4_power - C3_power) / (C4_power + C3_power)` -- positive for left imagery (contralateral C3 desynchronization), negative for right
- **ERD (Event-Related Desynchronization)**: `(baseline_power - task_power) / baseline_power × 100` -- percentage power decrease during motor imagery
- **Hjorth parameters**: Activity (variance), Mobility (mean frequency), Complexity (bandwidth) -- efficient time-domain descriptors

#### FBCSP Features (up to 32)

Computed in `src/train.py` during the training loop (fitted per fold to avoid data leakage):

- **8 motor-focused frequency bands**: (8-10), (10-12), (12-14), (14-16), (16-18), (18-20), (20-24), (24-30) Hz
- **4 CSP components per band**: spatial filters maximizing variance ratio between left and right classes
- **Log-variance transformation**: CSP features are log-transformed for Gaussian-like distribution
- **OAS regularization**: Oracle Approximating Shrinkage for robust covariance estimation with small samples

### 5. Training & Classification

**Module**: `src/train.py`

Four classifiers are evaluated per subject. The best per-subject classifier is selected via **nested model-selection CV** (inner 7-fold selects, outer 10-fold evaluates) to avoid selection bias.

#### Classifier 1: FBCSP + LDA

Per-subject, per-fold pipeline (group-aware 10-fold stratified CV):

1. **FBCSP extraction** -- bandpass filter to each of 8 motor bands, fit CSP (OAS reg) on training set, transform both train and test (up to 32 features)
2. **Feature concatenation** -- FBCSP features (32) + handcrafted features (45) = ~77 features
3. **Feature selection** -- `SelectKBest(f_classif)` with k selected via nested 5-fold CV from {3, 5, 8, 10, 15, 20}
4. **Scaling** -- `StandardScaler` zero-centers and unit-normalizes features
5. **Classification** -- `LDA(solver='lsqr', shrinkage='auto', priors=...)` with Ledoit-Wolf shrinkage and class priors estimated from training fold frequencies

#### Classifier 2: Riemannian Tangent Space

Per-subject, per-fold pipeline (group-aware 10-fold stratified CV, no frequency band tuning or feature engineering):

1. **Covariance estimation** -- `Covariances(estimator='oas')` computes Oracle Approximating Shrinkage covariance matrices from multichannel EEG (8×8 SPD matrices)
2. **Tangent space projection** -- `TangentSpace(metric='riemann')` maps SPD matrices to Euclidean tangent space (36 features for 8 channels)
3. **Classification** -- `LogisticRegression(C=1.0, solver='lbfgs')`

#### Classifier 3: FBCSP + SVM (RBF)

Per-subject, per-fold pipeline (group-aware 10-fold stratified CV):

1. **FBCSP extraction** -- same as Classifier 1
2. **Feature concatenation** -- FBCSP features (32) + handcrafted features (45) = ~77 features
3. **Feature selection** -- `SelectKBest(f_classif, k=max_allowed)`
4. **Scaling** -- `StandardScaler`
5. **Classification** -- `SVC(kernel='rbf', C=10.0, gamma='scale')` -- tests nonlinear decision boundaries that LDA misses

This classifier replaces a previous Transfer learning (Riemannian domain adaptation) classifier that averaged below chance (46.9%) with only 10 subjects.

#### Classifier 4: LDA + SVM Ensemble (Soft Voting)

Per-subject, per-fold pipeline (group-aware 10-fold stratified CV):

1. **Shared FBCSP extraction** -- same as Classifiers 1 and 3
2. **Independent pipelines** -- LDA and SVM each run their full SelectKBest → Scaler → Classifier pipeline
3. **Probability averaging** -- `(LDA_proba + SVM_proba) / 2` averages predicted class probabilities from both classifiers
4. **Prediction** -- `argmax` of averaged probabilities

Riemannian is excluded from the ensemble because its lower overall accuracy (50.2%) drags down the vote.

**Evaluation**: Group-aware stratified cross-validation per subject for all four classifiers individually, plus nested model-selection CV for unbiased best-of-4 estimation. **In-fold artifact rejection** computes the amplitude threshold from training indices only and applies it to both train and test (via `_reject_in_fold()`), preventing threshold leakage. CSP models are re-fitted on each fold's training set to prevent information leakage. The pipeline also supports optional held-out evaluation (`holdout_fraction` parameter) and cross-session validation for subjects with multiple recordings.

**Deployment models**: After CV evaluation, the classifier selected by nested CV is trained on all data per subject and saved as a `.joblib` file. FBCSP+LDA, FBCSP+SVM, and Ensemble models contain the full inference pipeline (CSP models, feature selector, scaler, classifier). Riemannian models contain a single sklearn Pipeline. Ensemble winners are saved as their LDA component for deployment.

## Project Structure

```
BCIS/
├── src/
│   ├── __init__.py
│   ├── config.py            # Typed experiment configuration (CV/feature settings)
│   ├── validation.py        # Leakage-resistant split policies (stratified/grouped)
│   ├── pipeline.py          # Main pipeline orchestrator (entry point, held-out, cross-session)
│   ├── train.py              # FBCSP+LDA & Riemannian training, in-fold rejection, nested CV, cross-session eval
│   ├── data_loader.py        # CSV loading, event parsing
│   ├── preprocess.py         # Bandpass + notch filtering, CAR, ASR
│   ├── epochs.py             # Epoch extraction + per-trial PTP computation
│   └── features.py           # Handcrafted features + CSP
├── tests/
│   ├── test_data_loader.py   # Data loading and recording discovery
│   ├── test_preprocess.py    # Filter correctness (DC removal, 60Hz attenuation)
│   ├── test_epochs.py        # Epoch extraction and boundary conditions
│   ├── test_features.py      # Feature shapes, values, missing channel errors
│   └── test_train.py         # CV output, leakage check, model completeness, in-fold rejection
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

Loads data, preprocesses, extracts features, runs CV per subject (including nested model-selection CV), trains final models, and saves results:

```bash
# Default mode (no held-out split, backward compatible)
uv run python -m src.pipeline

# With held-out evaluation (20% of trials reserved as independent test set)
uv run python -c "
from pathlib import Path
from src.config import TrainingConfig
from src.pipeline import run_pipeline
cfg = TrainingConfig(
    split_strategy='stratified_group',
    trial_group_size=5,
    n_folds=10,
    n_outer_folds=10,
    n_inner_folds=5,
)
run_pipeline(
    Path('unicorn-data'),
    Path('models'),
    holdout_fraction=0.2,
    training_config=cfg,
    quiet_output=True,  # clean console output (default)
)
"
```

Set `BCIS_VERBOSE_LOGS=1` if you want full third-party logs for debugging.

Output:
- Per-subject `.joblib` models in `models/`
- `models/training_results.json` with accuracy metrics + serialized training config

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

# FBCSP+SVM model (uses same predict function as LDA)
model = joblib.load("models/subject0010_svm.joblib")
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

### FBCSP+LDA models (`*_fbcsp_lda.joblib`)

| Key | Type | Description |
|-----|------|-------------|
| `csp_models` | `list[tuple[CSP, tuple[float, float]]]` | Fitted CSP spatial filters per frequency band |
| `selector` | `SelectKBest` | Fitted feature selector (top 10 by ANOVA F-score) |
| `scaler` | `StandardScaler` | Fitted feature normalizer |
| `classifier` | `LinearDiscriminantAnalysis` | Fitted LDA with shrinkage |
| `sfreq` | `float` | Sampling frequency (250.0 Hz) |
| `k_best` | `int` | Number of selected features (10) |

### FBCSP+SVM models (`*_svm.joblib`)

Same structure as FBCSP+LDA models, but `classifier` is `SVC` instead of `LinearDiscriminantAnalysis`.

### Riemannian models (`*_riemann.joblib`)

| Key | Type | Description |
|-----|------|-------------|
| `pipeline` | `sklearn.pipeline.Pipeline` | Full pipeline: Covariances → TangentSpace → LogisticRegression |
| `sfreq` | `float` | Sampling frequency (250.0 Hz) |

## Testing

The test suite validates each pipeline stage with both correctness checks and edge cases:

```bash
# Run all tests
uv run pytest tests/

# Run a specific test file
uv run pytest tests/test_train.py
```

Pytest is configured for clean output by default (`-q --tb=short`) and suppresses known non-actionable MNE filter-length warnings.

**Test coverage** (60 tests):
- `test_data_loader.py` -- CSV parsing, event extraction, complete recording discovery
- `test_preprocess.py` -- DC offset removal, 60 Hz attenuation (>90% power reduction), full pipeline, CAR common-mode removal, ASR artifact cleaning
- `test_epochs.py` -- Epoch shapes, class separation, boundary conditions (signal start/end), artifact rejection (amplitude, flat signal, adaptive outlier, gradient, HF power, criteria disable)
- `test_features.py` -- Feature dimensions (45 features), lateralization index symmetry, missing channel errors, CSP output shapes, spectral entropy, peak frequency, C3-C4 coherence, statistical features
- `test_train.py` -- FBCSP+LDA CV scores, leakage check, model completeness, round-trip predict; Riemannian CV scores, leakage check, model round-trip; SVM CV scores, leakage check, model round-trip; Ensemble CV scores, leakage check; Nested model-selection CV scores, leakage check; Cross-session evaluate scores, leakage check; In-fold rejection (outlier removal, None passthrough, integration with CV)
- `test_validation.py` -- classwise trial grouping, group-aware split integrity, grouped-split fallback behavior

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| [MNE-Python](https://mne.tools/) | >=1.11.0 | EEG signal processing, FIR filters, CSP implementation |
| [scikit-learn](https://scikit-learn.org/) | >=1.8.0 | LDA, LogisticRegression, SelectKBest, StratifiedKFold/StratifiedGroupKFold, StandardScaler |
| [pyriemann](https://pyriemann.readthedocs.io/) | >=0.7 | Riemannian geometry: covariance estimation, tangent space projection |
| [asrpy](https://github.com/DiGyt/asrpy) | >=0.0.8 | Artifact Subspace Reconstruction (optional, includes numpy 2.x compat patch) |
| [NumPy](https://numpy.org/) | >=2.4.2 | Array operations |
| [SciPy](https://scipy.org/) | >=1.17.0 | Welch's method for spectral analysis |
| [pandas](https://pandas.pydata.org/) | >=3.0.0 | CSV loading |
| [joblib](https://joblib.readthedocs.io/) | >=1.5.3 | Model serialization |

**Dev dependencies**: pytest (>=9.0.2), ruff (>=0.14.14)
