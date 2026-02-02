# Left/Right Motor Imagery Classification Design

## Overview

Update the BCIS project to classify left vs right hand motor imagery from EEG data, enabling control of a robotic 6th finger (left imagery = down, right imagery = up).

## Goals

- **Primary**: Cross-subject generalization with >90% mean accuracy
- **Fallback**: Hybrid approach (pre-trained model + short calibration) if cross-subject doesn't hit target
- **End state**: Real-time classification for robotic finger control

## Data Structure

- **Source**: Phase 3 (imagery-perform) epochs only
- **Labels**: movement=1 (left hand) → Label 0, movement=2 (right hand) → Label 1
- **Per session**: 100 trials (50 left, 50 right)
- **Total**: ~1000 trials across ~10 subjects

## Implementation Plan

### 1. Epoch Extraction (`src/epochs.py`)

Modify `extract_erd_epochs` to:
- Filter events to phase 3 only
- Split by movement code (1 vs 2) instead of phase code
- Keep baseline/task structure (baseline before event, task from 0.5s after onset)

New function signature:
```python
def extract_left_right_epochs(
    signal: np.ndarray,
    events: list[tuple[int, int, int]],
    sfreq: float,
    task_duration: float = 1.5,
    baseline_duration: float = 1.0,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """Extract left (movement=1) and right (movement=2) imagery epochs from phase 3."""
```

### 2. Lateralization-Focused Features (`src/features.py`)

**Primary Features (C3-C4 Lateralization)**
- Lateralization Index: `(C4_power - C3_power) / (C4_power + C3_power)` for mu and beta
- Asymmetry ERD: Difference in ERD percentage between C3 and C4
- Contralateral/Ipsilateral ratio

**Key Frequency Bands**
- Mu rhythm (8-12 Hz): Primary motor imagery signature
- Beta (13-30 Hz): Secondary motor signature
- Sub-bands: Low-beta (13-20 Hz), high-beta (20-30 Hz)

**Supporting Features**
- ERD from Cz (supplementary motor area)
- Fz theta (attention/effort marker)
- Time-course features: early vs late ERD

New function:
```python
def extract_lateralization_features(
    epoch_pairs_by_channel: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    sfreq: float,
) -> np.ndarray:
    """Extract features optimized for left/right motor imagery discrimination."""
```

### 3. Training Pipeline (`src/train.py`)

**Evaluation**: Leave-One-Subject-Out (LOSO) cross-validation
- Train on N-1 subjects, test on held-out subject
- Report mean accuracy across all folds

**Classifiers**:
1. CSP + LDA (classic motor imagery approach)
2. Riemannian MDM/Tangent Space
3. Feature-based: LDA, SVM, Random Forest on lateralization features

**Preprocessing**:
- Filter to mu+beta bands (8-30 Hz) before CSP
- Z-score standardization across subjects

### 4. Pipeline Updates (`src/pipeline.py`)

- Update to call new epoch extraction function
- Use lateralization features instead of generic ERD features
- Report LOSO cross-validation results
- Save model for deployment

## Fallback Strategies (If <90%)

### Tier 1: Feature Engineering
- Filter-Bank CSP with feature selection
- Laplacian filtering around C3/C4
- Time-frequency decomposition (wavelets)

### Tier 2: Deep Learning
- EEGNet (compact CNN for EEG)
- ShallowConvNet (learnable FBCSP)

### Tier 3: Hybrid Approach
- Pre-train on all subjects
- Short calibration (~20 trials) for new users
- Fine-tune or adapt model

### Tier 4: Subject Selection
- Identify subjects with strong lateralization for demos

## Success Criteria

- Mean LOSO accuracy ≥ 90%
- Consistent performance across subjects (low variance)
- Model suitable for real-time inference

## Files to Modify

1. `src/epochs.py` - Add `extract_left_right_epochs`
2. `src/features.py` - Add `extract_lateralization_features`
3. `src/train.py` - Add LOSO CV, update classifiers
4. `src/pipeline.py` - Wire everything together
5. `tests/` - Update tests for new functions
