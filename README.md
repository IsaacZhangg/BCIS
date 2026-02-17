# BCIS - Left/Right Motor Imagery BCI Classifier

Classifies left vs right hand motor imagery from 8-channel EEG for controlling a robotic 6th finger (left = down, right = up).

Uses four classifiers — FBCSP+LDA, Riemannian, FBCSP+SVM, and LDA+SVM Ensemble — with nested model-selection CV to pick the best per subject without selection bias.

## Results

**61.2% nested selection accuracy** (unbiased) across 10 subjects. 5 of 10 above chance (>=60%).

| Subject | FBCSP+LDA | Riemann | SVM | Ensemble | Nested (unbiased) | Selected |
|---------|-----------|---------|-----|----------|--------------------|----------|
| subject0001 | 48.5% | 41.9% | 49.3% | 45.7% | **45.9%** | SVM |
| subject0002 | 40.1% | 41.7% | 43.2% | 35.4% | **41.0%** | SVM |
| subject0004 | 42.8% | 29.9% | 56.7% | 44.2% | **54.7%** | SVM |
| subject0005 | 56.8% | 53.4% | 55.6% | 55.5% | **55.9%** | FBCSP |
| subject0006 | 90.0% | 42.0% | 87.0% | 85.0% | **88.0%** | FBCSP |
| subject0007 | 64.0% | 45.0% | 50.0% | 61.0% | **66.0%** | FBCSP |
| subject0008 | 69.8% | 52.8% | 57.1% | 63.4% | **65.6%** | FBCSP |
| subject0009 | 49.1% | 56.1% | 54.9% | 47.6% | **52.3%** | Riemann |
| subject0010 | 78.0% | 60.3% | 85.3% | 83.1% | **82.1%** | FBCSP |
| subject0011 | 53.6% | 60.4% | 48.7% | 49.6% | **60.4%** | Riemann |

Signal quality with the consumer-grade 8-channel Unicorn headset remains the primary bottleneck.

## Hardware

**Headset**: [g.tec Unicorn Hybrid Black](https://www.unicorn-bi.com/) — 8 dry electrodes (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8), 250 Hz, 24-bit ADC.

## Pipeline

```
CSV files (8ch, 250Hz)
  -> data_loader.py    Parse recordings, extract event markers
  -> preprocess.py     Bandpass (1-40Hz) + notch (60Hz) filtering (batched over channels)
  -> epochs.py         Baseline (1.0s) / task (3.0s) windows, per-trial PTP
  -> features.py       45 handcrafted features (Laplacian C3/C4, ERD, Hjorth, coherence, entropy)
  -> train.py          4 classifiers with in-fold artifact rejection + nested CV
  -> pipeline.py       Orchestration, model saving, optional held-out & cross-session eval
```

### Classifiers

1. **FBCSP+LDA**: 8 motor bands x 4 CSP + 45 handcrafted features -> SelectKBest (nested CV k) -> StandardScaler -> shrinkage LDA
2. **Riemannian**: Covariances (OAS) -> TangentSpace (Riemann) -> LogisticRegression
3. **FBCSP+SVM**: Same FBCSP features -> SelectKBest -> SVC (RBF, C=20)
4. **Ensemble**: LDA + SVM soft voting (averaged probabilities)

### Key design choices

- **In-fold artifact rejection**: Threshold (median + 3.5xMAD) computed from training fold only, preventing leakage
- **Surface Laplacian**: Sharpens C3/C4 spatial resolution using neighboring electrodes
- **Nested CV**: Inner 7-fold selects best classifier, outer 10-fold evaluates — unbiased estimate
- **Subject-level parallelism**: joblib dispatch with BLAS thread capping
- **Bandpass caching**: Precomputed per subject, sliced by CV indices
- **Redundant-work elimination**: batched multichannel preprocessing, shared calibrated SVM fits in shared-eval paths, and no duplicate FBCSP filtering when training final models

## Project Structure

```
BCIS/
├── src/
│   ├── config.py          # Experiment configuration
│   ├── pipeline.py        # Main entry point
│   ├── train.py           # Classifiers, CV, nested selection, parallel dispatch
│   ├── data_loader.py     # CSV loading, event parsing
│   ├── preprocess.py      # Temporal filtering
│   ├── epochs.py          # Epoch extraction, PTP computation
│   ├── features.py        # Handcrafted + CSP features
│   └── validation.py      # Split policies
├── tests/                 # 60 tests across all pipeline stages
├── models/                # Per-subject .joblib models + training_results.json
├── unicorn-data/          # EEG recordings (gitignored)
└── pyproject.toml
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd BCIS
uv sync
```

Place Unicorn CSV exports in `unicorn-data/subject{NNNN}/session{NNN}/recording_*.csv`.

## Usage

```bash
# Run full pipeline
uv run python -m src.pipeline

# Run tests
uv run pytest tests/ -v

# Lint and format
ruff check --fix src/ tests/
ruff format src/ tests/
```

### Inference

```python
import joblib
from src.train import predict, predict_riemann

# FBCSP+LDA model
model = joblib.load("models/subject0006_fbcsp_lda.joblib")
predictions = predict(model, X_features, X_multichannel)  # 0=left, 1=right

# SVM model (same predict interface as FBCSP+LDA)
model = joblib.load("models/subject0004_svm.joblib")
predictions = predict(model, X_features, X_multichannel)

# Riemannian model
model = joblib.load("models/subject0011_riemann.joblib")
predictions = predict_riemann(model, X_multichannel)

# Ensemble-selected subjects save an FBCSP+LDA model for deployment
model = joblib.load("models/subject0007_ensemble_lda.joblib")
predictions = predict(model, X_features, X_multichannel)
```

## Data Format

Each CSV has columns: `timestamp`, `Fz`, `C3`, `Cz`, `C4`, `Pz`, `PO7`, `Oz`, `PO8`, `stim`.

Event encoding: `stim = phase * 10 + movement` (phase 3 only; movement 1=left, 2=right). Each recording has 100 phase-3 trials (50 left, 50 right).

## Dependencies

Core: MNE-Python, scikit-learn, pyriemann, NumPy, SciPy, pandas, joblib, threadpoolctl. Dev: pytest, ruff. See `pyproject.toml` for versions.
