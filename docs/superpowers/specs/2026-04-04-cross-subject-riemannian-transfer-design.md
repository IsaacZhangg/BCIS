# Cross-Subject Riemannian Transfer Learning

## Goal

Add a cross-subject evaluation mode using Euclidean Alignment (EA) on Riemannian covariance matrices, evaluated with leave-one-subject-out CV (LOSO). This addresses accuracy for "BCI-illiterate" subjects by leveraging data from other subjects.

## Current State

- 10 subjects in `Data/unicorn-data/`, 3 new subjects in `Data/MI_DATA_NEW/` (subject0100, subject0101, subject0102), plus subject0100_2 (second recording of subject0100) and subject0000 (existing subject, additional sessions)
- Within-subject nested CV: ~59% mean accuracy (seed-averaged), parameter tuning ceiling reached
- Riemannian pipeline exists: `Covariances(estimator='lwf') → TangentSpace(metric='riemann') → LogisticRegression(C=0.1)`
- pyriemann 0.10 has transfer module: `TLCenter`, `encode_domains`, `TLClassifier`

## Design

### New module: `src/transfer.py`

Keeps transfer learning separate from the existing within-subject pipeline. No changes to existing code.

### Data Loading

Call `get_complete_recordings()` on both `Data/unicorn-data/` and `Data/MI_DATA_NEW/`, merge results. Subject0100 and subject0100_2 are the same person — combine as two sessions of one subject.

For subjects with multiple sessions, pool all sessions into a single trial set per subject (more data for alignment and classification).

### Pipeline Flow

```
1. Load & preprocess all subjects from both data directories
2. Extract epochs (same parameters: task=3.0s, baseline=1.0s, skip=0.25s)
3. Build multichannel epoch arrays per subject: (n_trials, 8, n_samples)
4. Compute covariance matrices: Covariances(estimator='lwf') → (n_trials, 8, 8) per subject
5. Euclidean Alignment: per subject, compute Riemannian mean → re-center to identity
6. LOSO CV: for each held-out subject, train on all others' aligned data, test on held-out
7. LOSO + fine-tuning: re-center alignment using held-out subject's own mean before testing
```

### Euclidean Alignment (EA) Implementation

EA removes inter-subject variability by centering each subject's covariance distribution around the identity matrix:

```python
# For each subject s with covariance matrices C_s:
#   1. Compute Riemannian mean M_s of C_s
#   2. Transform: C_s_aligned = M_s^{-1/2} @ C_s @ M_s^{-1/2}
# After alignment, all subjects share the same "center" (identity matrix)
```

Using pyriemann's `TLCenter(target_domain, metric='riemann')` which implements this. Domain labels are encoded via `encode_domains(X, y, domain)`.

### LOSO CV

For each held-out subject h:
1. **Train set**: aligned covariances from all subjects except h
2. **Test set**: aligned covariances from subject h
3. Classifier: `TangentSpace(metric='riemann') → LogisticRegression(C=0.1, solver='lbfgs')`
4. Record accuracy for subject h

### LOSO + Fine-tuning

Same as LOSO but before testing:
1. Compute the Riemannian mean of the held-out subject's aligned covariances
2. Re-center the classifier's reference point toward this mean
3. This adapts the pooled model to the target subject's specific geometry

### Artifact Rejection

Apply the same in-fold rejection approach as the existing pipeline:
- Compute per-trial max PTP from multichannel data
- Threshold: median + 3.5 × MAD (computed from training subjects only in LOSO)

### Output

The module provides a `run_transfer_evaluation()` function that:
1. Prints a comparison table: within-subject nested CV vs LOSO vs LOSO+fine-tuning
2. Returns a results dict with per-subject scores for all three methods
3. Can be called from `pipeline.py` as an optional step or run standalone

### Entry Point

Add to `pipeline.py`:
- After existing step 4 (cross-session eval), add optional step: "Cross-subject transfer evaluation"
- Also runnable standalone: `uv run python -m src.transfer`

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| LOSO accuracy per subject | Train on all others, test on held-out |
| LOSO + fine-tuning per subject | Same but re-centered on target subject |
| Mean LOSO | Average across all subjects |
| Comparison table | Within-subject nested CV vs LOSO vs LOSO+fine-tuning |

### What This Does NOT Change

- Existing within-subject pipeline is untouched
- Existing classifiers (FBCSP+LDA, SVM, Ensemble) are unaffected
- Config parameters remain the same
- Test suite for existing code is unaffected

### Dependencies

No new dependencies — uses `pyriemann.transfer` (already installed, v0.10) and existing `pyriemann.estimation.Covariances`.

### File Changes

| File | Change |
|------|--------|
| `src/transfer.py` | **New** — EA, LOSO CV, comparison logic |
| `src/pipeline.py` | Add optional call to `run_transfer_evaluation()` after cross-session eval |
| `tests/test_transfer.py` | **New** — tests for alignment, LOSO, data merging |
