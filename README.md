# BCIS - Brain-Computer Interface Classifier

A machine learning pipeline that reads EEG brain signals and classifies whether someone is imagining moving their **left hand** or **right hand**. This powers a robotic 6th finger: left imagery = finger down, right imagery = finger up.

This repository contains source code, synthetic tests, and aggregate research results. Participant recordings, trained models, and generated per-subject results are kept locally and excluded from version control. Running the training pipeline requires your own authorized recordings.

## How It Works

1. **Record** — 8-channel EEG via a [g.tec Unicorn](https://www.unicorn-bi.com/) headset (dry electrodes, 250 Hz)
2. **Clean** — Bandpass filter (1–40 Hz) and notch filter (60 Hz)
3. **Epoch** — 1.0 s baseline, 0.25 s skip, 3.0 s task window; in-fold artifact rejection
4. **Extract** — 45 handcrafted features per trial (surface Laplacian on C3/C4, ERD, Hjorth parameters, coherence, entropy) plus filter-bank CSP spatial filters
5. **Classify** — Nested cross-validation picks among FBCSP+LDA, Riemannian, FBCSP+SVM, Ensemble, and Composite CSP (and the FBCSP band config) per subject
6. **Deploy** — Saved models output `0` (left) or `1` (right)

## Results

The recorded 15-subject run reached **59.3% nested selection accuracy** after Composite CSP and session-level Euclidean Alignment, and **61.3%** with EA-gated donor augmentation. The original 10-subject subset reached **60.5%** nested accuracy. Six of 15 subjects reached at least 60% nested accuracy.

These aggregate results describe the local research dataset. The recordings and participant-level measurements are not distributed with this repository. Reproducing these exact numbers requires access to that dataset.

Composite CSP mixes other subjects' class covariances into the target CSP filters with λ=0.3 without training the classifier on foreign trials. Session-level EA fits on training folds before pooling multiple recordings.

Signal quality with the consumer-grade 8-channel headset remains the primary bottleneck.

## The Classifiers

| Classifier | What It Does |
|------------|--------------|
| **FBCSP+LDA** | Applies filter-bank Common Spatial Patterns to isolate motor-related brain activity, then classifies with Linear Discriminant Analysis. Inner CV also selects among `standard` / `high_mu` / `wide_mu` band configs |
| **Riemannian** | LWF covariances mapped through TangentSpace into logistic regression (C=0.1) — no hand-crafted features needed |
| **FBCSP+SVM** | Same spatial filtering as FBCSP+LDA, but uses a Support Vector Machine (RBF kernel, C=20) for classification |
| **Ensemble** | Averages the confidence scores of LDA and SVM for a combined vote |
| **Composite CSP** | Mixes other subjects' class covariances into the target CSP filters (Lotte & Guan, λ=0.3). Filters are applied to the target subject only |

A **nested cross-validation** scheme (inner 7-fold selects the best classifier, outer 10-fold evaluates) ensures the reported accuracy is unbiased.

## Donor Augmentation for Weak Subjects

Not every subject has enough data to train a good model on their own. For subjects below 52% nested accuracy, `src/transfer.py` re-runs nested CV with trials borrowed from other subjects. Both sides are re-centered with **Euclidean Alignment** (`src/alignment.py`) fit on the target's training fold only, and donors are used only when inner CV beats target-only by 2pp. This is an evaluation metric; the saved models are trained on each subject's own data.

## Quick Start

**Requirements:** Python 3.11+ and [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo-url>
cd BCIS
uv sync
```

Place authorized recordings under `data/unicorn-data/` and, optionally, `data/MI_DATA_NEW/`. These directories are gitignored and are not included in a fresh clone. Layout: `subject{NNNN}/session{NNN}/recording_*.csv`. Folder aliases configured in `src/config.py` are applied after grouping; review those mappings for your dataset.

The test suite generates synthetic recordings in temporary directories and runs without participant data or saved models.

```bash
# Run the full pipeline (nested CV, donor augmentation, model export)
uv run python -m src.pipeline

# Useful flags (wired into the pipeline)
uv run python -m src.pipeline --verbose
uv run python -m src.pipeline --holdout-fraction 0.2
uv run python -m src.pipeline --data-dir data/unicorn-data --output-dir models

# Nested-only experiment without overwriting models/training_results.json
uv run python scripts/run_nested_experiment.py --name my_run --composite-csp --session-ea

# Run tests
uv run pytest tests/ -v
```

The pipeline creates trained models and `models/training_results.json` in the gitignored `models/` directory. They are not included in a clone. Keep recordings, generated results, and models out of commits.

### Loading a Trained Model

FBCSP+LDA, FBCSP+SVM, Ensemble, and Composite CSP artifacts all share the same dict layout (`csp_models`, `selector`, `scaler`, `classifier`), so they use `predict()`. Riemannian models use `predict_riemann()`.

```python
import joblib
from src.train import predict, predict_riemann

# FBCSP+LDA, SVM, Ensemble, or CCSP
model = joblib.load("models/subjectNNNN_fbcsp_lda.joblib")
predictions = predict(model, X_features, X_multichannel)  # 0=left, 1=right

# Riemannian
model = joblib.load("models/subjectNNNN_riemann.joblib")
predictions = predict_riemann(model, X_multichannel)
```

Replace `subjectNNNN` with a locally trained subject ID. Selected deployment models are listed in the local `models/training_results.json` under `model_paths`.

## Project Structure

```
BCIS/
├── src/
│   ├── pipeline.py      # Main entry point — load, CV, donor augmentation, export
│   ├── config.py        # Experiment settings (TrainingConfig)
│   ├── data_loader.py   # Reads CSVs, parses event markers, dedups recordings
│   ├── preprocess.py    # Bandpass + notch
│   ├── epochs.py        # Cuts continuous EEG into trials, peak-to-peak rejection
│   ├── features.py      # Handcrafted features (Laplacian, ERD, Hjorth, …)
│   ├── train.py         # Classifier training, nested CV, model selection
│   ├── composite_csp.py # Lotte & Guan Composite CSP
│   ├── transfer.py      # EA-gated donor augmentation for weak subjects
│   ├── alignment.py     # Euclidean Alignment (subject- and session-level)
│   ├── validation.py    # CV split policies
│   └── band_cache.py    # Per-band filtered-trial cache for FBCSP
├── tests/               # Synthetic tests covering pipeline stages and modules
├── models/              # Local saved models + training_results.json (gitignored)
├── data/                # Local participant recordings (gitignored)
│   ├── unicorn-data/    # Original EEG recordings
│   └── MI_DATA_NEW/     # Optional additional MI recordings
├── scripts/             # Nested-only experiment runner
├── research/            # Experiment log (program.md, results.tsv)
├── previouslytried.md   # Parameter and structural experiment history
└── pyproject.toml       # Dependencies and project config
```

## Data Format

Each CSV contains columns: `timestamps`, `Fz`, `C3`, `Cz`, `C4`, `Pz`, `PO7`, `Oz`, `PO8`, `stim`.

The `stim` column encodes events as `phase * 10 + movement`: phase 3 is motor imagery, movement 1 is left, and movement 2 is right. The original recording protocol uses 100 trials, split evenly between movements. Shorter recordings can be pooled across sessions; subjects must meet `min_evaluation_trials`, which defaults to 30, to be evaluated.

## Hardware

**g.tec Unicorn Hybrid Black** — 8 dry electrodes at positions Fz, C3, Cz, C4, Pz, PO7, Oz, PO8. 250 Hz sampling rate, 24-bit ADC.

## Design Decisions

- **In-fold artifact rejection** — Thresholds computed from training data only, preventing data leakage
- **Nested CV** — Separates model selection from evaluation so accuracy numbers are honest
- **Stratified splits** — Default `stratified_group` CV preserves class balance; with `trial_group_size=1` every trial is its own group
- **Content-hash dedup** — Identical recordings copied into both data folders are loaded once
- **Subject aliases** — Configured folder names are merged into the canonical subject ID
- **In-fold donor selection + Euclidean Alignment** — Cross-subject augmentation never sees the test fold; EA is fit on target training trials only; inner CV must beat target-only by 2pp before donors are used
- **Session-level EA** — Multi-session subjects are aligned with a train-fold-only transform before pooling
- **Subject-level parallelism** — Each subject processed independently via joblib; BLAS threads capped per worker
- **Bandpass caching** — Filtered FBCSP bands precomputed once per subject, reused across folds
- **No ASR** — Artifact Subspace Reconstruction was tried and removed because it stripped discriminative motor-imagery signal (see `previouslytried.md`)

## Dependencies

MNE-Python, scikit-learn, pyriemann, NumPy, SciPy, pandas, joblib. Dev: pytest, pytest-cov, ruff. See `pyproject.toml` for the full list.
