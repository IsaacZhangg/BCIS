# Theta-Based Focus Classifier Design

**Date:** 2026-01-31
**Goal:** Decode theta waves from EEG to classify focused vs not-focused states with >90% accuracy

## Overview

Build a classifier that uses frontal midline theta power (Fz electrode, 4-8 Hz) to determine whether a subject is focused or not. Train on existing Unicorn headset data, then adapt for real-time inference if accuracy exceeds 90%.

## Data Summary

| Property | Value |
|----------|-------|
| Headset | Unicorn Hybrid Black (8 channels, 250 Hz, 24-bit) |
| Subjects | 10 complete recordings |
| Trials per subject | 100 imagery + ~140 rest epochs |
| Key electrode | Fz (frontal midline) |
| Labels | Phase 3 (imagery) = Focused, Phase 5 (rest) = Not Focused |

### Phase Durations
- Phase 3 (imagery/focused): ~2.1 seconds average
- Phase 5 (rest/not focused): ~3.1 seconds average

## Architecture

```
OFFLINE TRAINING PIPELINE
─────────────────────────
Raw CSV → Preprocessing → Epoch Extraction → Feature Extraction
                                ↓
                    Theta Power (Fz, 4-8 Hz)
                                ↓
                    Train Classifier (LogReg/SVM)
                                ↓
                    Save Model + Threshold

REAL-TIME INFERENCE PIPELINE (Future)
─────────────────────────────────────
Live EEG Stream → Sliding Window → Same Feature Extraction
                                ↓
                    Load Trained Model
                                ↓
                Output: Focused / Not Focused
```

## Preprocessing Pipeline

1. Load CSV data
2. Apply bandpass filter (1-40 Hz) - remove DC drift and high-frequency noise
3. Apply notch filter (60 Hz) - remove power line interference
4. Extract Fz channel for theta analysis

## Epoch Extraction

| Parameter | Value |
|-----------|-------|
| Duration | 2.0 seconds (500 samples) |
| Start | At phase marker onset |
| Focused epochs | Phase 3 onset, 100 per subject |
| Not-focused epochs | Phase 5 onset, 100 per subject (subsampled for balance) |

**Rationale for 2.0 seconds:**
- Enough cycles for theta estimation (4 Hz = 8 cycles)
- Stays within shortest phase duration (~2.1 sec)
- Matches real-time sliding window size

## Feature Extraction

For each 2-second Fz epoch:

1. Compute PSD using Welch's method
   - Window: 1 sec (250 samples)
   - Overlap: 50%
2. Extract theta band power (4-8 Hz)
   - Average power across frequency bins in range
3. Log transform (optional, helps with skewed power distributions)

**Output:** Single feature per epoch: `log(theta_power)`

## Classification

### Primary Model: Logistic Regression
- Simple, fast, interpretable
- Outputs probability for confidence thresholds
- Single feature eliminates overfitting risk

### Fallback: SVM with RBF kernel
- If logistic regression < 90%, try SVM
- Can capture non-linear decision boundaries

## Evaluation Strategy

### Leave-One-Subject-Out (LOSO) Cross-Validation

```
For each of 10 subjects:
    Train on 9 subjects (~1800 epochs)
    Test on 1 held-out subject (~200 epochs)
    Record accuracy

Final: Mean accuracy across 10 folds
```

**Why LOSO:**
- Tests generalization to new subjects (realistic for deployment)
- More rigorous than random split
- Standard in BCI research

### Metrics
- Accuracy (primary, target >90%)
- Confusion matrix
- Per-subject accuracy

## Project Structure

```
BCIS Spring 2026/
├── unicorn-data/              # Existing data
├── src/
│   ├── preprocess.py          # Filtering, epoch extraction
│   ├── features.py            # Theta power calculation
│   ├── train.py               # Model training with LOSO CV
│   └── predict.py             # Single-epoch inference
├── models/
│   └── theta_classifier.pkl   # Saved trained model
├── notebooks/
│   └── analysis.ipynb         # Exploration, visualization
└── docs/
    └── plans/                 # Design documents
```

## Implementation Phases

1. **Data exploration** - Visualize raw signals, verify epoch extraction
2. **Feature pipeline** - Implement preprocessing + theta extraction
3. **Training pipeline** - LOSO cross-validation, evaluate accuracy
4. **Analysis** - If <90%, add features (theta/beta ratio, more channels)
5. **Real-time prep** - Package for live inference

## Success Criteria

| Metric | Target |
|--------|--------|
| LOSO CV Accuracy | >90% |
| Per-subject min accuracy | >80% |
| Inference latency | <100ms per epoch |

## Tech Stack

- Python (uv package manager)
- MNE - EEG filtering
- scipy - Spectral analysis (Welch's method)
- scikit-learn - Classification
- ruff - Linting

## Fallback Plan

If theta power alone doesn't achieve 90%:
1. Add theta/beta ratio feature
2. Include additional electrodes (Cz, Pz)
3. Try more complex classifiers (Random Forest, Neural Network)
