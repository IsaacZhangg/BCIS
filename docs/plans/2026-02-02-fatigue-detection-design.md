# Fatigue Detection Classifier Design

## Overview

Pivot the EEG classifier from Phase 3 vs Phase 4 (imagery vs actual movement) to Phase 3 vs Phase 5 (imagery vs rest) to detect patient fatigue during data collection. The goal is proactive fatigue detection—identifying when a patient is becoming disengaged before they notice—to ensure high-quality EEG data.

## Requirements

- **Detection target**: Fatigue/disengagement, using Phase 5 (rest) as proxy for fatigued brain state
- **Output**: Continuous engagement score (0-100), where 100 = fully engaged, 0 = fatigued
- **Update frequency**: Every 5 seconds via sliding window
- **Alerts**: Visual indicator + threshold alert when engagement drops below 30 for 15+ seconds
- **Deliverable**: Offline simulation processing existing unicorn-data recordings

## Phase Codes Reference

| Phase | Meaning | Role in Classifier |
|-------|---------|-------------------|
| 1 | Video | — |
| 2 | Instruction | — |
| 3 | Imagery-perform | Engaged class (label=1) |
| 4 | Actual movement | — (no longer used) |
| 5 | Rest | Disengaged class (label=0) |

## Architecture

### Files to Remove

Delete all Phase 3 vs 4 artifacts:

```
models/theta_classifier_model.joblib
models/theta_classifier_scaler.joblib
models/training_results.json
docs/plans/2026-01-31-theta-focus-classifier-design.md
docs/plans/2026-01-31-theta-focus-classifier-implementation.md
```

### Files to Modify

**`src/epochs.py`**
- Change `class2_phase` default from 4 to 5
- Update docstrings to reference rest (Phase 5) instead of movement (Phase 4)
- Labels: 1 = engaged (Phase 3), 0 = disengaged (Phase 5)

**`src/features.py`**
- Add `extract_realtime_features()` function for absolute features that don't require baseline
- Realtime features include:
  - Log band powers (theta, alpha, beta, gamma)
  - Relative band powers (% of total)
  - Theta/Alpha ratio (drowsiness indicator)
  - Theta/Beta ratio (attention indicator)
  - Hjorth parameters (activity, mobility, complexity)
  - Envelope features (mean, std, max)

**`src/train.py`**
- Ensure models output probability scores for engagement scoring
- Train two model variants:
  - Full-feature model (for validation)
  - Realtime-feature-only model (for simulation)

**`src/pipeline.py`**
- Update `class2_phase=5`
- Rename variables: "focused/rest" → "engaged/disengaged"

**`tests/test_epochs.py`**
- Update Phase 4 → Phase 5 in test expectations

**`tests/test_features.py`**
- Add tests for `extract_realtime_features()`

### Files to Create

**`src/simulate_realtime.py`**

Sliding window simulation that processes recordings as if live:

```
Input: Recording CSV path, trained model + scaler

Processing loop (every 5 seconds):
  1. Extract 5s window (8 channels × 1250 samples)
  2. Preprocess (bandpass 1-40 Hz, notch 60 Hz)
  3. Extract realtime features
  4. Classify → probability of engaged class
  5. Scale to 0-100 engagement score

Output:
  - CSV: timestamp, engagement_score, alert_triggered
  - Console summary: min/max/mean, alert count, flagged periods
  - Optional matplotlib plot with threshold line
```

Alert logic:
- Threshold: 30 (configurable)
- Trigger: Score below threshold for 3 consecutive windows (15 seconds)
- Prevents single-window false alarms

**`tests/test_simulate.py`**
- Test window extraction
- Test alert logic
- Test end-to-end simulation on sample data

### New Model Artifacts

After training:

```
models/
├── engagement_classifier_model.joblib      # full-feature model
├── engagement_classifier_scaler.joblib
├── engagement_realtime_model.joblib        # realtime-features-only
├── engagement_realtime_scaler.joblib
└── training_results.json                   # Phase 3 vs 5 metrics
```

## Feature Rationale

**Why different features for Phase 3 vs 5 (vs the old Phase 3 vs 4)?**

Phase 3 vs 4 (imagery vs movement) relied on:
- Motor cortex lateralization (C3/C4 asymmetry)
- Mu rhythm desynchronization
- Movement-specific patterns

Phase 3 vs 5 (imagery vs rest) relies on:
- Alpha power increase during rest (8-13 Hz)
- Beta suppression during rest
- Theta changes with drowsiness
- Overall arousal level differences

The existing feature set includes these, but the model will weight them differently after retraining.

## Window-Based Inference

**Challenge**: Training uses baseline+task epoch pairs, but realtime uses single 5-second windows without baseline reference.

**Solution**:
- Training: Continue using ERD (baseline vs task) for maximum discrimination
- Realtime: Use absolute features only (log powers, ratios, Hjorth)
- Train separate realtime model on absolute features
- Accept potentially lower accuracy for streaming capability

## Evaluation

**Success criteria:**
- Cross-validation accuracy ≥ 85% on Phase 3 vs Phase 5
- Realtime model accuracy within 10% of full-feature model

**Validation approach:**

1. **Offline accuracy**: 10-fold within-subject CV, per-subject and mean scores

2. **Simulation sanity checks**:
   - High engagement during Phase 3 segments
   - Low engagement during Phase 5 segments
   - Transitions detected within 10-15 seconds

3. **Alert validation**:
   - False positive rate (alerts during engaged periods)
   - False negative rate (missed alerts during rest)
   - Tune threshold based on results

## Usage

```bash
# Retrain on Phase 3 vs 5
python -m src.pipeline

# Run simulation on a recording
python -m src.simulate_realtime unicorn-data/subject0001/session000/recording.csv

# Output example:
# Engagement Score Summary
# ------------------------
# Mean: 72.3
# Min: 18.4 at 04:32
# Max: 94.1 at 01:15
# Alerts triggered: 3
# Total time below threshold: 45 seconds
```

## Future Work (Out of Scope)

- Live Unicorn headset integration via streaming API
- Real-time UI/dashboard
- Session-level trend analysis
- Adaptive thresholds per subject
