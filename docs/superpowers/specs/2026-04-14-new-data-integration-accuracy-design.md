# New Data Integration & Accuracy Improvement Design

**Date**: 2026-04-14
**Status**: Draft
**Baseline**: 60.1% nested / 62.9% augmented nested (10 subjects, seed=42)

## Goal

Leverage new subjects (104, 106) and all MI_DATA_NEW subjects to improve classification accuracy through three complementary changes: flexible data loading, expanded donor augmentation with Euclidean Alignment, and subject-specific FBCSP band selection via an expanded feature pool.

**Constraints** (user-specified):
- No data leakage or inflated metrics
- No significant pipeline slowdown
- Keep codebase clean and maintainable

## 1. Flexible Data Loading

### Problem
`get_complete_recordings()` requires exactly 100 phase-3 trials, excluding the new subjects (104: 64 trials, 106: 32 trials).

### Design
- Lower the trial floor for main evaluation to **30 trials**.
- Subjects with fewer than 100 trials get an **adaptive fold count**: `min(n_trials // 5, n_folds)`. Subject 106 (32 trials) uses 6-fold CV; subject 104 (64 trials) uses 10-fold; originals stay at 10-fold.
- Add session merging for subject 104: `"subject0104_session002": "subject0104"` in `DEFAULT_SUBJECT_MERGE` (same pattern as subject0100_2).
- Subject 105 is empty (0 trials) and naturally excluded by the floor.
- The `get_complete_recordings()` function gains a `min_trials` parameter (default 100 for backward compatibility). A new call site in the pipeline uses `min_trials=30` for the expanded evaluation.

### Files touched
- `src/config.py`: Add subject0104 merge entry, add `min_evaluation_trials = 30`
- `src/data_loader.py`: Parameterize `get_complete_recordings()` with `min_trials`
- `src/pipeline.py`: Use flexible loader, compute adaptive fold count

## 2. Expanded Donor Pool with Euclidean Alignment

### Problem
Augmentation for weak subjects currently picks the closest donor from only the 10 main subjects. More donors improve the chance of finding a compatible match. Raw donor data concatenation ignores covariance differences between subjects.

### Design
- **Donor pool**: All subjects with >= 10 trials become donor candidates. This includes the 10 originals plus subjects 100, 101, 102, 104, 106 (~15 total).
- **Euclidean Alignment**: Before adding donor trials to a target's training fold, align the donor's trial covariances to the target subject's reference matrix. The reference matrix is the geometric mean of the target's training-fold covariances. EA is already implemented in `transfer.py` and will be extracted into a shared utility.
- **Donor selection**: Single closest donor by Riemannian distance (computed post-alignment). Multi-donor was previously tried and didn't help with 10 subjects; we keep single-donor for now.
- **Leakage protection**: Unchanged. Donors only enter training folds, never validation or test folds.

### Files touched
- `src/alignment.py` (new module): Extract EA logic from `transfer.py` into a reusable function
- `src/pipeline.py`: Load expanded donor pool, apply EA before augmentation
- `src/transfer.py`: Refactor to use shared EA utility

## 3. Subject-Specific FBCSP Band Selection

### Problem
All subjects use the same 8 fixed frequency bands (8-30 Hz). Motor imagery frequency patterns are highly individual -- the optimal bands for a strong subject (e.g., 0006 at 85%) likely differ from a weak subject (e.g., 0001 at 39%).

### Design
Rather than building custom band selection logic, expand the candidate band pool and let the existing SelectKBest + inner CV do subject-specific feature selection:

- **Expanded band pool** (14 bands):
  ```
  (6, 8), (7, 9), (8, 10), (9, 11), (10, 12), (11, 13), (12, 14),
  (14, 16), (16, 18), (18, 20), (20, 24), (24, 28), (28, 32), (30, 36)
  ```
  Adds finer mu resolution (7-9, 9-11, 11-13 Hz), a theta-mu border band (6-8 Hz), and upper beta bands (28-32, 30-36 Hz).

- **Feature space**: 14 bands x 3 CSP = 42 FBCSP features + 45 handcrafted = **87 total features**.

- **Extended K candidates**: `(3, 5, 8, 10, 15, 20, 25, 30, 40)` to accommodate the larger feature pool. For subjects with fewer trials, K is dynamically capped at `n_train_samples - 1` to prevent selecting more features than samples (subject 106 with 32 trials has ~22 inner-CV training samples, so K is capped at 21).

- **Band caching** handles the filtering cost. The only additional compute is more CSP fits per fold (~1.75x current cost, from 8 to 14 bands).

- SelectKBest with `f_classif` scoring in the inner CV naturally selects the best features per subject -- if subject 0006's signal is in 10-12 Hz, those CSP features score highest; if subject 0001 needs 8-10 Hz, different features win.

### Files touched
- `src/train.py`: Update `FBCSP_BANDS` to the expanded 14-band set, update `K_CANDIDATES`
- `src/config.py`: Update `k_candidates` in `TrainingConfig`

## 4. Evaluation & Reporting

### Primary metric
Augmented nested CV accuracy (mean across subjects).

### Dual reporting
- **All subjects** (12): Full picture including 104 and 106.
- **Original 10 subjects only**: Direct comparison against 60.1%/62.9% baseline.

### Per-subject breakdown
Same format as current `training_results.json`: per-classifier accuracy, nested selection method, augmented nested score. New subjects flagged with trial count.

### Multi-seed validation
Run with 3 seeds (42, 123, 7) to confirm gains exceed the ~3.7 pp noise floor. A gain is "real" if the mean improvement across seeds is > 0 and no seed shows regression on the original 10 subjects.

### Files touched
- `src/pipeline.py`: Dual reporting logic, adaptive fold count in results
- `models/training_results.json`: Extended with new subjects

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Overfitting with 87 features | SelectKBest + nested CV limits effective dimensionality; K is selected by inner CV |
| EA adds complexity | EA is a well-understood linear transformation; shared utility keeps code DRY |
| New subjects have noisy estimates | Report trial counts; dual reporting separates new from original |
| Compute increase from 14 bands | Band caching eliminates filtering cost; CSP is ~1.75x (acceptable) |
| Expanded K candidates could overfit K | Inner CV prevents this; K=40 is still < n_trials for all subjects |

## Non-goals

- Deep learning (EEGNet, ShallowConvNet) -- likely data-limited with 32-100 trials per subject
- Multi-donor augmentation -- previously tried and reverted; revisit only if single-donor + EA doesn't help
- Augmentation threshold tuning (0.50 -> 0.55) -- previously tried without improvement; can revisit after other changes land
