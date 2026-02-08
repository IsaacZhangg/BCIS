# BCIS - Left/Right Motor Imagery Classifier

A brain-computer interface system that classifies left vs right hand motor imagery from EEG data. Designed for real-time control of a robotic 6th finger, where left imagery = finger down and right imagery = finger up.

## Results

**87.6% mean accuracy** (within-subject 10-fold CV across 10 subjects, target: 90%)

| Subject | Accuracy |
|---------|----------|
| subject0001 | 89% |
| subject0002 | 83% |
| subject0004 | 83% |
| subject0005 | 84% |
| subject0006 | **98%** |
| subject0007 | 86% |
| subject0008 | 86% |
| subject0009 | 86% |
| subject0010 | **95%** |
| subject0011 | 86% |

## How It Works

1. **Data loading** - Parses Unicorn EEG headset recordings (8 channels, 250 Hz) from CSV files
2. **Preprocessing** - Bandpass filter (1-40 Hz) and 60 Hz notch filter
3. **Epoch extraction** - Extracts paired baseline (1.0s) and task (1.5s) windows around motor imagery events
4. **Feature extraction** - Computes lateralization indices (C3 vs C4 power asymmetry) across mu (8-12 Hz) and beta (13-30 Hz) bands, plus CSP spatial filters, Hjorth parameters, and frontal theta
5. **Classification** - Ensemble of 40+ classifiers (Filter-Bank CSP, Riemannian geometry, LDA, Random Forest) with multiple aggregation strategies; best ensemble selected per subject

## Project Structure

```
src/
  pipeline.py      # Main pipeline orchestrator
  train.py         # Training & cross-validation
  data_loader.py   # EEG CSV loading & event parsing
  preprocess.py    # Signal filtering
  epochs.py        # Epoch extraction
  features.py      # Feature computation
tests/             # pytest test suite
models/            # Trained model & scaler (joblib)
unicorn-data/      # EEG recordings (not in repo)
docs/plans/        # Design & implementation docs
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

Run the full pipeline:

```bash
uv run python -m src.pipeline
```

Run tests:

```bash
uv run pytest tests/ -v
```

## Key Dependencies

- **MNE-Python** - EEG signal processing
- **pyRiemann** - Riemannian geometry classifiers
- **scikit-learn** - ML classifiers, cross-validation, scaling
- **NumPy / SciPy / pandas** - Numerical computing and data handling
