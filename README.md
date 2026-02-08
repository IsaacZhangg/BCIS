# BCIS - Left/Right Motor Imagery Classifier

A brain-computer interface system that classifies left vs right hand motor imagery from EEG data. Designed for real-time control of a robotic 6th finger, where left imagery = finger down and right imagery = finger up.

## Results

**57.2% mean accuracy** (within-subject 10-fold CV across 10 subjects using FBCSP + LDA)

Only 2 of 10 subjects show above-chance signal — this is expected given consumer-grade EEG hardware and varying subject aptitude for motor imagery.

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

## How It Works

1. **Data loading** - Parses Unicorn EEG headset recordings (8 channels, 250 Hz) from CSV files
2. **Preprocessing** - Bandpass filter (1-40 Hz) and 60 Hz notch filter
3. **Epoch extraction** - Extracts paired baseline (1.0s) and task (1.8s) windows around motor imagery events
4. **Feature extraction** - FBCSP (Filter-Bank CSP across 10 frequency bands, 4 components each = 40 features) + 39 handcrafted features (lateralization indices, ERD asymmetry, Hjorth parameters, frontal theta)
5. **Feature selection** - SelectKBest (f_classif, k=20) picks the most discriminative features
6. **Classification** - LDA with automatic shrinkage (per-subject models)

## Project Structure

```
src/
  pipeline.py      # Main pipeline orchestrator
  train.py         # FBCSP + LDA training & cross-validation
  data_loader.py   # EEG CSV loading & event parsing
  preprocess.py    # Signal filtering
  epochs.py        # Epoch extraction
  features.py      # Feature computation
tests/             # pytest test suite
models/            # Per-subject trained models (joblib)
unicorn-data/      # EEG recordings (not in repo)
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

- **MNE-Python** - EEG signal processing and CSP implementation
- **scikit-learn** - LDA classifier, feature selection, cross-validation, scaling
- **NumPy / SciPy / pandas** - Numerical computing and data handling
