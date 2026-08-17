# BCIS - Brain-Computer Interface Classifier

A machine learning pipeline that reads EEG brain signals and classifies whether someone is imagining moving their **left hand** or **right hand**. This powers a robotic 6th finger: left imagery = finger down, right imagery = finger up.

## How It Works

1. **Record** — 8-channel EEG via a [g.tec Unicorn](https://www.unicorn-bi.com/) headset (dry electrodes, 250 Hz)
2. **Clean** — Bandpass filter (1–40 Hz) and notch filter (60 Hz)
3. **Epoch** — 1.0 s baseline, 0.25 s skip, 3.0 s task window; in-fold artifact rejection
4. **Extract** — 45 handcrafted features per trial (surface Laplacian on C3/C4, ERD, Hjorth parameters, coherence, entropy) plus filter-bank CSP spatial filters
5. **Classify** — Nested cross-validation picks among FBCSP+LDA, Riemannian, FBCSP+SVM, Ensemble, and Composite CSP (and the FBCSP band config) per subject
6. **Deploy** — Saved models output `0` (left) or `1` (right)

## Results

**59.3% nested selection accuracy** (unbiased) across 15 subjects after Composite CSP plus per-session Euclidean Alignment on multi-session recordings. 6 of 15 above chance (>=60%): subject0006, subject0007, subject0008, subject0010, subject0101, subject0104. EA-gated donor augmentation for weak subjects (nested < 52%): **61.3%**. Original 10-subject nested mean is **60.5%**. LOSO transfer mean: 51.3%.

The 15-subject mean is still pulled down by several BCI-illiterate recordings. Composite CSP mixes other subjects' class covariances into the target CSP filters (λ=0.3) without training the classifier on foreign trials. Session-level EA (train-fold only) removes headset/session drift before pooling extra recordings for 0100/0101/0104.

Top performers:

| Subject | Accuracy | Best Classifier |
|---------|----------|-----------------|
| subject0006 | **85.0%** | FBCSP+LDA |
| subject0010 | **81.1%** | Ensemble |
| subject0101 | **71.0%** | SVM |
| subject0008 | **70.0%** | Composite CSP |
| subject0104 | **69.6%** | FBCSP+LDA |
| subject0007 | **64.0%** | FBCSP+LDA |

<details>
<summary>Full results table</summary>

| Subject | FBCSP+LDA | Riemann | SVM | Ensemble | Nested (unbiased) | Selected |
|---------|-----------|---------|-----|----------|--------------------|----------|
| subject0001 | 48.5% | 39.1% | 49.3% | 45.7% | **47.2%** | SVM |
| subject0002 | 43.9% | 41.7% | 41.6% | 34.2% | **37.7%** | SVM |
| subject0004 | 42.8% | 30.9% | 62.9% | 47.2% | **55.7%** | SVM |
| subject0005 | 56.8% | 54.6% | 55.6% | 55.5% | **54.3%** | SVM |
| subject0006 | 90.0% | 39.0% | 82.0% | 85.0% | **85.0%** | FBCSP |
| subject0007 | 65.0% | 44.0% | 50.0% | 60.0% | **64.0%** | FBCSP |
| subject0008 | 69.8% | 52.8% | 57.2% | 63.4% | **70.0%** | CCSP |
| subject0009 | 49.1% | 57.7% | 54.9% | 47.6% | **52.2%** | Riemann |
| subject0010 | 78.0% | 60.3% | 85.3% | 83.1% | **81.1%** | Ensemble |
| subject0011 | 53.6% | 60.4% | 49.9% | 49.6% | **57.9%** | Riemann |
| subject0100 | 39.6% | 45.7% | 47.6% | 44.9% | **43.7%** | Ensemble |
| subject0101 | 58.0% | 47.4% | 56.6% | 61.7% | **71.0%** | SVM |
| subject0102 | 58.3% | 49.2% | 48.1% | 51.1% | **50.3%** | SVM |
| subject0104 | 57.4% | 35.6% | 49.6% | 58.2% | **69.6%** | FBCSP |
| subject0106 | 58.9% | 60.6% | 59.4% | 58.9% | **49.4%** | CCSP |

</details>

Signal quality with the consumer-grade 8-channel headset remains the primary bottleneck.

## The Classifiers

| Classifier | What It Does |
|------------|--------------|
| **FBCSP+LDA** | Applies filter-bank Common Spatial Patterns to isolate motor-related brain activity, then classifies with Linear Discriminant Analysis. Inner CV also selects among `standard` / `high_mu` / `wide_mu` band configs |
| **Riemannian** | LWF covariances mapped through TangentSpace into logistic regression (C=0.1) — no hand-crafted features needed |
| **FBCSP+SVM** | Same spatial filtering as FBCSP+LDA, but uses a Support Vector Machine (RBF kernel, C=20) for classification |
| **Ensemble** | Averages the confidence scores of LDA and SVM for a combined vote |
| **Composite CSP** | Mixes other subjects' class covariances into the target CSP filters (Lotte & Guan, λ=0.3). Filters are applied to the target subject only |

A **nested cross-validation** scheme (inner 7-fold selects the best classifier, outer 10-fold evaluates) ensures the reported accuracy is unbiased. Stacking (logistic regression on out-of-fold base probabilities) is reported as an extra metric and does not participate in model selection. EEGNet is implemented in `src/eegnet.py` but skipped unless PyTorch is installed (it is not a default dependency).

## Cross-Subject Transfer Learning

Not every subject has enough data to train a good model on their own. `src/transfer.py` implements **Euclidean Alignment** (`src/alignment.py`) to pool data across subjects:

1. Compute covariance matrices per subject (Ledoit-Wolf shrinkage)
2. Re-center each subject's data to a common reference (removes headset placement differences)
3. Train on everyone else, test on the held-out subject (leave-one-subject-out CV)

Weak subjects can also be evaluated with leakage-free donor augmentation (nearest Riemannian neighbor selected from the training fold only). The full pipeline runs this transfer evaluation automatically; `uv run python -m src.transfer` runs it on its own.

## Quick Start

**Requirements:** Python 3.11+ and [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo-url>
cd BCIS
uv sync
```

Recordings are loaded from `data/unicorn-data/` and, when present, `data/MI_DATA_NEW/`. Layout: `subject{NNNN}/session{NNN}/recording_*.csv`. Folder aliases (`subject0100_2` → `subject0100`) are applied after grouping.

```bash
# Run the full pipeline (nested CV, donor augmentation, transfer, model export)
uv run python -m src.pipeline

# Useful flags (wired into the pipeline)
uv run python -m src.pipeline --verbose
uv run python -m src.pipeline --holdout-fraction 0.2
uv run python -m src.pipeline --data-dir data/unicorn-data --output-dir models

# Run cross-subject transfer evaluation only
uv run python -m src.transfer

# Nested-only experiment without overwriting models/training_results.json
uv run python scripts/run_nested_experiment.py --name my_run --composite-csp --session-ea

# Run tests
uv run pytest tests/ -v
```

Trained models and `models/training_results.json` are committed so a clone can load them without re-running the pipeline. Legacy LightGBM artifacts (`models/*_lgbm.joblib`) are gitignored.

### Loading a Trained Model

FBCSP+LDA, FBCSP+SVM, Ensemble, and Composite CSP artifacts all share the same dict layout (`csp_models`, `selector`, `scaler`, `classifier`), so they use `predict()`. Riemannian models use `predict_riemann()`.

```python
import joblib
from src.train import predict, predict_riemann

# FBCSP+LDA, SVM, Ensemble, or CCSP
model = joblib.load("models/subject0006_fbcsp_lda.joblib")
predictions = predict(model, X_features, X_multichannel)  # 0=left, 1=right

# Riemannian
model = joblib.load("models/subject0011_riemann.joblib")
predictions = predict_riemann(model, X_multichannel)
```

Selected deployment models are listed in `models/training_results.json` under `model_paths`.

## Project Structure

```
BCIS/
├── src/
│   ├── pipeline.py      # Main entry point — load, CV, transfer, export
│   ├── cli.py           # Command-line flags (RunConfig)
│   ├── config.py        # Experiment settings (TrainingConfig)
│   ├── data_loader.py   # Reads CSVs, parses event markers, dedups recordings
│   ├── preprocess.py    # Bandpass + notch (ASR helper exists, not default)
│   ├── epochs.py        # Cuts continuous EEG into trials
│   ├── features.py      # Handcrafted features (Laplacian, ERD, Hjorth, …)
│   ├── train.py         # Classifier training, nested CV, model selection
│   ├── classifiers.py   # Adapter registry (parallel to train.py orchestration)
│   ├── composite_csp.py # Lotte & Guan Composite CSP
│   ├── transfer.py      # Cross-subject transfer learning
│   ├── alignment.py     # Euclidean Alignment (subject- and session-level)
│   ├── validation.py    # CV split policies
│   ├── rejection.py     # In-fold peak-to-peak artifact rejection
│   ├── band_cache.py    # Per-band filtered-trial cache for FBCSP
│   ├── stage_cache.py   # On-disk NPZ cache keyed by config + content hash
│   ├── runtime_output.py# Quiet MNE / warning defaults
│   ├── experiments.py   # Optional experiment JSON logging
│   └── eegnet.py        # Optional EEGNet (requires torch)
├── tests/               # 184 tests covering pipeline stages and modules
├── models/              # Saved models (.joblib) + training_results.json (tracked)
├── data/
│   ├── unicorn-data/    # Original EEG recordings
│   └── MI_DATA_NEW/     # Additional MI recordings (incl. subjects 0100–0106)
├── scripts/             # Nested-experiment runner, stacking/temporal-aug checks, analysis helpers
├── research/            # Experiment log (program.md, results.tsv)
├── previouslytried.md   # Parameter and structural experiment history
└── pyproject.toml       # Dependencies and project config
```

## Data Format

Each CSV contains columns: `timestamps`, `Fz`, `C3`, `Cz`, `C4`, `Pz`, `PO7`, `Oz`, `PO8`, `stim`.

The `stim` column encodes events as `phase * 10 + movement` (phase 3 = motor imagery; movement 1 = left, 2 = right). Original unicorn recordings typically have 100 trials (50 left, 50 right). `MI_DATA_NEW` recordings may have fewer and are still included when they meet `min_evaluation_trials` (default 30): 0100/0101/0104 have 64 trials after session pooling; 0102/0106 have 32. Recordings below that threshold (subject0000, subject0003, subject0105) are skipped.

## Hardware

**g.tec Unicorn Hybrid Black** — 8 dry electrodes at positions Fz, C3, Cz, C4, Pz, PO7, Oz, PO8. 250 Hz sampling rate, 24-bit ADC.

## Design Decisions

- **In-fold artifact rejection** — Thresholds computed from training data only, preventing data leakage
- **Nested CV** — Separates model selection from evaluation so accuracy numbers are honest
- **Grouped stratified splits** — Default `stratified_group` CV keeps nearby trials together while preserving class balance
- **Content-hash dedup** — Identical recordings copied into both data folders are loaded once
- **Subject aliases** — `subject0100_2` and similar folder names are merged into the canonical subject id
- **In-fold donor selection + Euclidean Alignment** — Cross-subject augmentation never sees the test fold; EA is fit on target training trials only; inner CV must beat target-only by 2pp before donors are used
- **Session-level EA** — Multi-session subjects (0100/0101/0104) are aligned with a train-fold-only transform before pooling
- **Subject-level parallelism** — Each subject processed independently via joblib; BLAS threads capped per worker
- **Bandpass caching** — Filtered FBCSP bands precomputed once per subject, reused across folds
- **ASR is not default** — Artifact Subspace Reconstruction is implemented (`asrpy`) but was reverted from the main path because it removed discriminative motor-imagery signal

## Dependencies

MNE-Python, scikit-learn, pyriemann, NumPy, SciPy, pandas, joblib, asrpy. Dev: pytest, pytest-cov, ruff. Optional: torch (EEGNet). See `pyproject.toml` for the full list.
