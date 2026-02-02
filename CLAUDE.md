# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EEG-based fatigue/engagement detection system for BCI applications. Classifies Phase 3 (motor imagery = engaged) vs Phase 5 (rest = disengaged) from Unicorn EEG recordings.

**Current Status:** 92.8% mean accuracy across 10 subjects using within-subject cross-validation.

## Commands

```bash
# Install dependencies
uv sync

# Run full training pipeline
uv run python -m src.pipeline

# Run real-time simulation
uv run python -m src.simulate_realtime unicorn-data/<recording>.csv

# Run tests
uv run pytest tests/ -v
uv run pytest tests/test_pipeline.py -v  # Single file

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Architecture

### Processing Pipeline (`src/pipeline.py`)

```
Raw CSV → Load → Preprocess → Extract Epochs → Extract Features → Train → Save Models
```

1. **data_loader.py** - Load Unicorn CSV with 8 EEG channels + stim markers
2. **preprocess.py** - Bandpass (1-40 Hz), notch (60 Hz), CAR spatial filter
3. **epochs.py** - Extract baseline+task epoch pairs for Phase 3 and Phase 5
4. **features.py** - Three feature types:
   - ERD features (event-related desynchronization)
   - Riemannian features (covariance matrices)
   - Realtime features (absolute only, no baseline needed)
5. **train.py** - Within-subject CV with 20+ classifiers, weighted ensemble voting

### Dual Model Design

- **Full-feature model** (`engagement_classifier_model.joblib`): Uses ERD features, requires baseline
- **Realtime model** (`engagement_realtime_model.joblib`): Uses absolute features only, suitable for streaming

### Real-Time Simulation (`src/simulate_realtime.py`)

5-second sliding windows, extracts realtime features, outputs engagement score (0-100). Alert triggered after 3 consecutive windows below threshold (30).

## EEG Data Format

- **Channels:** Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 (8 channels)
- **Sample Rate:** 250 Hz
- **Location:** `unicorn-data/` directory

### Stim Column Encoding

Each non-zero value in the stim column is a numeric marker sent at the start of each phase of a trial:

```
stim = trial_number · phase · movement
```

| Digit Position | Meaning |
|----------------|---------|
| Leading digits | Trial index (1, 2, 3, ...) |
| Second-to-last | Phase code |
| Last digit | Movement class |

**Phase codes:** 1=Video, 2=Instruction, 3=Imagery (engaged), 4=Movement, 5=Rest (disengaged)

**Movement codes:** 1=Left, 2=Right

**Examples:**
- `131` → Trial 1, Phase 3 (imagery), Movement 1 (left)
- `252` → Trial 2, Phase 5 (rest), Movement 2 (right)

**Notes:**
- `stim = 0` means no event (most rows)
- Each phase marker sent once at phase onset
- Each session contains 100 imagery markers (50 per hand)

## Key Dependencies

- **pyriemann** - Riemannian geometry for BCI classification
- **MNE** - EEG analysis toolkit
- **scikit-learn** - Traditional ML classifiers
- **XGBoost** - Gradient boosting
