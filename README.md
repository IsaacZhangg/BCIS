# BCIS - Brain-Computer Interface Classifier

A machine learning pipeline that reads EEG brain signals and classifies whether someone is imagining moving their **left hand** or **right hand**. This powers a robotic 6th finger: left imagery = finger down, right imagery = finger up.

## How It Works

1. **Record** — 8-channel EEG via a [g.tec Unicorn](https://www.unicorn-bi.com/) headset (dry electrodes, 250 Hz)
2. **Clean** — Bandpass filter (1–40 Hz), notch filter (60 Hz), surface Laplacian to sharpen spatial signals
3. **Extract** — 45 features per trial (ERD, Hjorth parameters, coherence, entropy, CSP spatial filters)
4. **Classify** — 4 classifiers compete; nested cross-validation picks the best one per subject (and the FBCSP band config)
5. **Deploy** — Saved models output `0` (left) or `1` (right) in real time

## Results

**56.5% nested selection accuracy** (unbiased) across 15 subjects, up from 53.4% after combining both data folders with content-hash dedup and in-fold rejection (no test-set leakage). 4 of 15 above chance (>=60%): subject0006, subject0007, subject0008, subject0010. EA-gated donor augmentation for weak subjects: **58.7%**. Original 10-subject nested mean is unchanged at 59.7%. LOSO transfer mean: 51.3%.

The 15-subject mean is still pulled down by several BCI-illiterate recordings; the gain vs the previous 15-subject 53.4% comes from pooling extra sessions in `MI_DATA_NEW` without duplicating files or leaking artifact thresholds.

Top performers:

| Subject | Accuracy | Best Classifier |
|---------|----------|-----------------|
| subject0006 | **85.0%** | FBCSP+LDA |
| subject0010 | **81.1%** | FBCSP+LDA |
| subject0008 | **66.8%** | FBCSP+LDA |
| subject0007 | **64.0%** | FBCSP+LDA |
| subject0104 | **58.5%** | Ensemble |

<details>
<summary>Full results table</summary>

| Subject | FBCSP+LDA | Riemann | SVM | Ensemble | Nested (unbiased) | Selected |
|---------|-----------|---------|-----|----------|--------------------|----------|
| subject0001 | 48.5% | 39.1% | 49.3% | 45.7% | **46.0%** | FBCSP |
| subject0002 | 43.9% | 41.7% | 41.6% | 34.2% | **37.7%** | SVM |
| subject0004 | 42.8% | 30.9% | 62.9% | 47.2% | **55.7%** | SVM |
| subject0005 | 56.8% | 54.6% | 55.6% | 55.5% | **50.6%** | SVM |
| subject0006 | 90.0% | 39.0% | 82.0% | 85.0% | **85.0%** | FBCSP |
| subject0007 | 65.0% | 44.0% | 50.0% | 60.0% | **64.0%** | FBCSP |
| subject0008 | 69.8% | 52.8% | 57.2% | 63.4% | **66.8%** | FBCSP |
| subject0009 | 49.1% | 57.7% | 54.9% | 47.6% | **52.2%** | Riemann |
| subject0010 | 78.0% | 60.3% | 85.3% | 83.1% | **81.1%** | FBCSP |
| subject0011 | 53.6% | 60.4% | 49.9% | 49.6% | **57.9%** | Riemann |
| subject0100 | 39.6% | 45.7% | 47.6% | 44.9% | **45.2%** | Ensemble |
| subject0101 | 58.0% | 47.4% | 56.6% | 61.7% | **54.9%** | SVM |
| subject0102 | 58.3% | 49.2% | 48.1% | 51.1% | **47.5%** | SVM |
| subject0104 | 57.4% | 35.6% | 49.6% | 58.2% | **58.5%** | Ensemble |
| subject0106 | 58.9% | 60.6% | 59.4% | 58.9% | **43.9%** | SVM |

</details>

Signal quality with the consumer-grade 8-channel headset remains the primary bottleneck.

## The Four Classifiers

| Classifier | What It Does |
|------------|--------------|
| **FBCSP+LDA** | Applies filter-bank Common Spatial Patterns to isolate motor-related brain activity, then classifies with Linear Discriminant Analysis. Inner CV also selects among `standard` / `high_mu` / `wide_mu` band configs |
| **Riemannian** | Works directly with covariance matrices on a curved (Riemannian) manifold — no hand-crafted features needed |
| **FBCSP+SVM** | Same spatial filtering as FBCSP+LDA, but uses a Support Vector Machine (RBF kernel) for classification |
| **Ensemble** | Averages the confidence scores of LDA and SVM for a combined vote |

A **nested cross-validation** scheme (inner 7-fold selects the best classifier, outer 10-fold evaluates) ensures the reported accuracy is unbiased. Stacking (logistic regression on out-of-fold base probabilities) is reported as an extra metric and does not participate in model selection. EEGNet is optional and skipped unless PyTorch is installed.

## Cross-Subject Transfer Learning

Not every subject has enough data to train a good model on their own. `src/transfer.py` implements **Euclidean Alignment** (`src/alignment.py`) to pool data across subjects:

1. Compute covariance matrices per subject (Ledoit-Wolf shrinkage)
2. Re-center each subject's data to a common reference (removes headset placement differences)
3. Train on everyone else, test on the held-out subject (leave-one-subject-out CV)

Weak subjects can also be evaluated with leakage-free donor augmentation (nearest Riemannian neighbor selected from the training fold only).

## Quick Start

**Requirements:** Python 3.11+ and [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo-url>
cd BCIS
uv sync
```

Recordings are loaded from `data/unicorn-data/` and, when present, `data/MI_DATA_NEW/`. Layout: `subject{NNNN}/session{NNN}/recording_*.csv`.

```bash
# Run the full pipeline
uv run python -m src.pipeline

# Useful flags
uv run python -m src.pipeline --subjects 0006,0010 --verbose
uv run python -m src.pipeline --holdout-fraction 0.2 --no-cache

# Run cross-subject transfer evaluation only
uv run python -m src.transfer

# Run tests
uv run pytest tests/ -v
```

Trained models and `models/training_results.json` are committed so a clone can load them without re-running the pipeline. Legacy LightGBM artifacts (`models/*_lgbm.joblib`) are gitignored.

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
│   ├── cli.py           # Command-line flags (RunConfig)
│   ├── config.py        # Experiment settings (TrainingConfig)
│   ├── data_loader.py   # Reads CSVs, parses event markers
│   ├── preprocess.py    # Bandpass + notch filtering
│   ├── epochs.py        # Cuts continuous EEG into trials
│   ├── features.py      # Feature extraction (handcrafted + CSP)
│   ├── train.py         # Classifier training, nested CV, model selection
│   ├── transfer.py      # Cross-subject transfer learning
│   ├── alignment.py     # Shared Euclidean Alignment utility
│   ├── validation.py    # CV split policies
│   ├── experiments.py   # Optional experiment JSON logging
│   └── eegnet.py        # Optional EEGNet (requires torch)
├── tests/               # 168 tests covering pipeline stages and new modules
├── models/              # Saved models (.joblib) + training_results.json (tracked)
├── data/
│   ├── unicorn-data/    # Original EEG recordings
│   └── MI_DATA_NEW/     # Additional MI recordings (incl. subjects 0104–0106)
├── scripts/             # Smoke tests and analysis helpers
└── pyproject.toml       # Dependencies and project config
```

## Data Format

Each CSV contains columns: `timestamp`, `Fz`, `C3`, `Cz`, `C4`, `Pz`, `PO7`, `Oz`, `PO8`, `stim`.

The `stim` column encodes events as `phase * 10 + movement` (phase 3 = motor imagery; movement 1 = left, 2 = right). Original unicorn recordings typically have 100 trials (50 left, 50 right); `MI_DATA_NEW` recordings may have fewer and are still included when they meet `min_evaluation_trials` (default 30).

## Hardware

**g.tec Unicorn Hybrid Black** — 8 dry electrodes at positions Fz, C3, Cz, C4, Pz, PO7, Oz, PO8. 250 Hz sampling rate, 24-bit ADC.

## Design Decisions

- **In-fold artifact rejection** — Thresholds computed from training data only, preventing data leakage
- **Nested CV** — Separates model selection from evaluation so accuracy numbers are honest
- **Content-hash dedup** — Identical recordings copied into both data folders are loaded once
- **In-fold donor selection + Euclidean Alignment** — Cross-subject augmentation never sees the test fold; EA is fit on target training trials only; inner CV must beat target-only by 2pp before donors are used
- **Subject-level parallelism** — Each subject processed independently via joblib
- **Bandpass caching** — Filtered signals precomputed once per subject, reused across folds

## Dependencies

MNE-Python, scikit-learn, pyriemann, NumPy, SciPy, pandas, joblib, threadpoolctl. Dev: pytest, ruff. Optional: torch (EEGNet). See `pyproject.toml` for full list.
