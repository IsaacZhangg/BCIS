# BCIS - Brain-Computer Interface Classifier

A machine learning pipeline that reads EEG brain signals and classifies whether someone is imagining moving their **left hand** or **right hand**. This powers a robotic 6th finger: left imagery = finger down, right imagery = finger up.

## How It Works

1. **Record** — 8-channel EEG via a [g.tec Unicorn](https://www.unicorn-bi.com/) headset (dry electrodes, 250 Hz)
2. **Clean** — Bandpass filter (1–40 Hz), notch filter (60 Hz), surface Laplacian to sharpen spatial signals
3. **Extract** — 45 features per trial (ERD, Hjorth parameters, coherence, entropy, CSP spatial filters)
4. **Classify** — 4 classifiers compete; nested cross-validation picks the best one per subject
5. **Deploy** — Saved models output `0` (left) or `1` (right) in real time

## Results

**53.4% nested selection accuracy** (unbiased) across 15 subjects. 5 of 15 above chance (>=60%): subject0006, subject0007, subject0008, subject0010, subject0101. Augmented nested mean: 56.3%. LOSO transfer mean: 51.3%.

Note: the 15-subject mean is not comparable to the previous 10-subject 62.0% because the new subjects (mostly at chance) drag the mean down.

Top performers:

| Subject | Accuracy | Best Classifier |
|---------|----------|-----------------|
| subject0006 | **88.0%** | FBCSP+LDA |
| subject0010 | **82.1%** | FBCSP+LDA |
| subject0008 | **65.6%** | FBCSP+LDA |
| subject0007 | **64.0%** | FBCSP+LDA |
| subject0101 | **64.3%** | — |

<details>
<summary>Full results table</summary>

| Subject | FBCSP+LDA | Riemann | SVM | Ensemble | Nested (unbiased) | Selected |
|---------|-----------|---------|-----|----------|--------------------|----------|
| subject0001 | 48.5% | 41.9% | 49.3% | 45.7% | **45.9%** | SVM |
| subject0002 | 40.1% | 41.7% | 43.2% | 35.4% | **41.0%** | SVM |
| subject0004 | 42.8% | 29.9% | 56.7% | 44.2% | **55.7%** | SVM |
| subject0005 | 56.8% | 53.4% | 55.6% | 55.5% | **50.6%** | FBCSP |
| subject0006 | 90.0% | 42.0% | 87.0% | 85.0% | **88.0%** | FBCSP |
| subject0007 | 64.0% | 45.0% | 50.0% | 61.0% | **64.0%** | FBCSP |
| subject0008 | 69.8% | 52.8% | 57.1% | 63.4% | **65.6%** | FBCSP |
| subject0009 | 49.1% | 56.1% | 54.9% | 47.6% | **52.3%** | Riemann |
| subject0010 | 78.0% | 60.3% | 85.3% | 83.1% | **82.1%** | FBCSP |
| subject0011 | 53.6% | 60.4% | 48.7% | 49.6% | **60.4%** | Riemann |
| subject0100 | — | — | — | — | **33.8%** | — |
| subject0101 | — | — | — | — | **64.3%** | — |
| subject0102 | — | — | — | — | **31.7%** | — |
| subject0104 | — | — | — | — | **37.8%** | — |
| subject0106 | — | — | — | — | **36.7%** | — |

</details>

Signal quality with the consumer-grade 8-channel headset remains the primary bottleneck.

## The Four Classifiers

| Classifier | What It Does |
|------------|--------------|
| **FBCSP+LDA** | Applies filter-bank Common Spatial Patterns to isolate motor-related brain activity, then classifies with Linear Discriminant Analysis |
| **Riemannian** | Works directly with covariance matrices on a curved (Riemannian) manifold — no hand-crafted features needed |
| **FBCSP+SVM** | Same spatial filtering as FBCSP+LDA, but uses a Support Vector Machine (RBF kernel) for classification |
| **Ensemble** | Averages the confidence scores of LDA and SVM for a combined vote |

A **nested cross-validation** scheme (inner 7-fold selects the best classifier, outer 10-fold evaluates) ensures the reported accuracy is unbiased.

## Cross-Subject Transfer Learning

Not every subject has enough data to train a good model on their own. `src/transfer.py` implements **Euclidean Alignment** to pool data across subjects:

1. Compute covariance matrices per subject (Ledoit-Wolf shrinkage)
2. Re-center each subject's data to a common reference (removes headset placement differences)
3. Train on everyone else, test on the held-out subject (leave-one-subject-out CV)

## Quick Start

**Requirements:** Python 3.11+ and [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo-url>
cd BCIS
uv sync
```

Place EEG recordings in `Data/unicorn-data/subject{NNNN}/session{NNN}/recording_*.csv`.

```bash
# Run the full pipeline
uv run python -m src.pipeline

# Run cross-subject transfer evaluation only
uv run python -m src.transfer

# Run tests
uv run pytest tests/ -v
```

### Loading a Trained Model

```python
import joblib
from src.train import predict, predict_riemann

# For FBCSP+LDA or SVM models
model = joblib.load("models/subject0006_fbcsp_lda.joblib")
predictions = predict(model, X_features, X_multichannel)  # 0=left, 1=right

# For Riemannian models
model = joblib.load("models/subject0011_riemann.joblib")
predictions = predict_riemann(model, X_multichannel)
```

## Project Structure

```
BCIS/
├── src/
│   ├── pipeline.py      # Main entry point — runs everything
│   ├── config.py        # Experiment settings
│   ├── data_loader.py   # Reads CSVs, parses event markers
│   ├── preprocess.py    # Bandpass + notch filtering
│   ├── epochs.py        # Cuts continuous EEG into trials
│   ├── features.py      # Feature extraction (handcrafted + CSP)
│   ├── train.py         # Classifier training, nested CV, model selection
│   ├── transfer.py      # Cross-subject transfer learning
│   └── validation.py    # CV split policies
├── tests/               # 60 tests covering all pipeline stages
├── models/              # Saved models (.joblib) + results JSON
├── Data/
│   ├── unicorn-data/    # EEG recordings
│   └── MI_DATA_NEW/     # Additional MI recordings
└── pyproject.toml       # Dependencies and project config
```

## Data Format

Each CSV contains columns: `timestamp`, `Fz`, `C3`, `Cz`, `C4`, `Pz`, `PO7`, `Oz`, `PO8`, `stim`.

The `stim` column encodes events as `phase * 10 + movement` (phase 3 = motor imagery; movement 1 = left, 2 = right). A typical recording has 100 trials (50 left, 50 right).

## Hardware

**g.tec Unicorn Hybrid Black** — 8 dry electrodes at positions Fz, C3, Cz, C4, Pz, PO7, Oz, PO8. 250 Hz sampling rate, 24-bit ADC.

## Design Decisions

- **In-fold artifact rejection** — Thresholds computed from training data only, preventing data leakage
- **Nested CV** — Separates model selection from evaluation so accuracy numbers are honest
- **Subject-level parallelism** — Each subject processed independently via joblib
- **Bandpass caching** — Filtered signals precomputed once per subject, reused across folds

## Dependencies

MNE-Python, scikit-learn, pyriemann, NumPy, SciPy, pandas, joblib, threadpoolctl. Dev: pytest, ruff. See `pyproject.toml` for full list.
