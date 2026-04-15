# New Data Integration & Accuracy Improvement Design

**Date**: 2026-04-14
**Status**: Draft (v2, post-review)
**Baseline**: 60.1% nested / 62.9% augmented nested (10 subjects, seed=42)

## Goal

Leverage new subjects (104, 106) and all MI_DATA_NEW subjects to improve classification accuracy through three complementary changes: flexible data loading, expanded donor augmentation with Euclidean Alignment, and subject-specific feature selection via an expanded FBCSP band pool.

**Constraints** (user-specified):
- No data leakage or inflated metrics
- No significant pipeline slowdown
- Keep codebase clean and maintainable

## 1. Flexible Data Loading

### Problem
`get_complete_recordings()` requires exactly 100 phase-3 trials, excluding the new subjects (104: 64 trials, 106: 32 trials). A separate `_get_recordings_with_trials()` in `transfer.py` does nearly the same thing with `min_trials=10`.

### Design
- **Consolidate loading**: Replace both `get_complete_recordings()` and `_get_recordings_with_trials()` with a single `get_recordings(min_trials=100)` function in `data_loader.py`. The pipeline calls it with `min_trials=30` for expanded evaluation; transfer calls it with `min_trials=10` for the donor/LOSO pool. Default stays 100 for backward compatibility.
- **Adaptive fold counts**: Subjects with fewer than 100 trials get reduced fold counts. The formula is `min(n_trials // 5, n_folds)`:

  | Subject | Trials | Outer folds | Inner folds | Outer train size | Inner train size |
  |---------|--------|-------------|-------------|------------------|------------------|
  | Originals | 100 | 10 | 7 | 90 | ~77 |
  | subject0104 | 64 | 10 | 7 | ~58 | ~49 |
  | subject0106 | 32 | 6 | 4 | ~27 | ~20 |

  Inner fold count also adapts: `min(n_outer_train // 5, n_inner_folds)`. Subject 106 uses 4 inner folds (not 7) to keep inner training sets at ~20 samples — small but workable for LDA/SVM.

- **Session merging**: Add `"subject0104_session002": "subject0104"` to `DEFAULT_SUBJECT_MERGE`.
- Subject 105 is empty (0 trials) and naturally excluded.

### Files touched
- `src/config.py`: Add subject0104 merge entry, add `min_evaluation_trials = 30`
- `src/data_loader.py`: Consolidate into single `get_recordings(min_trials)` function
- `src/pipeline.py`: Use flexible loader, compute adaptive outer + inner fold counts
- `src/transfer.py`: Switch to shared `get_recordings(min_trials=10)`

## 2. Expanded Donor Pool with Euclidean Alignment

### Problem
Augmentation for weak subjects currently picks the closest donor from only the 10 main subjects. More donors improve the chance of finding a compatible match. Raw donor data concatenation ignores covariance differences between subjects.

### Design

**Donor pool**: All subjects with >= 10 trials become donor candidates (~15 total: 10 originals + subjects 100, 101, 102, 104, 106).

**Euclidean Alignment (standard, independent)**:
Apply standard EA to each subject independently — each subject's trial covariances are re-centered to identity using their own reference matrix. This is a well-understood normalization that makes cross-subject data more compatible without complex donor-to-target transforms.

Concretely:
1. Compute target subject's reference matrix `R_target` = geometric mean of target's **training-fold** covariances (not full dataset — see leakage fix below).
2. Compute donor's reference matrix `R_donor` = geometric mean of all donor's covariances.
3. Align both independently: `C_aligned = R^{-1/2} C R^{-1/2}` using each subject's own R.
4. Concatenate aligned donor trials into target's aligned training fold.

**Donor selection**: Single closest donor by Riemannian distance computed on **raw (pre-EA) covariance means**. Pre-EA distance preserves meaningful inter-subject geometry; post-EA distance would collapse to near-zero since EA re-centers everything to identity.

**Leakage fix (pre-existing issue)**: The current `_find_closest` computes Riemannian distance using the target's **full-dataset** covariance mean, which includes test-fold trials. Fix: compute the target's covariance mean from **training-fold trials only** within each outer CV fold. This means donor selection happens per outer fold (slightly more compute, but correct).

**Leakage protection**: Donors only enter training folds, never validation or test folds. With the fix above, donor *selection* is also training-fold-only.

**Shared utility API** (`src/alignment.py`):
```python
def compute_ea_transform(covariances: ndarray) -> ndarray:
    """Compute R^{-1/2} from geometric mean of trial covariances."""

def apply_ea_transform(covariances: ndarray, ref_inv_sqrt: ndarray) -> ndarray:
    """Apply precomputed alignment: ref_inv_sqrt @ C @ ref_inv_sqrt.T"""

def euclidean_align(covariances: ndarray) -> tuple[ndarray, ndarray]:
    """Convenience: compute transform + apply it. Returns (aligned, transform)."""
```

`transfer.py` refactors its inline `ref_inv_sqrt @ covs @ ref_inv_sqrt.T` patterns to call these functions.

### Files touched
- `src/alignment.py` (new): Shared EA utility with the 3 functions above
- `src/pipeline.py`: Load expanded donor pool, apply EA, fix donor selection leakage
- `src/transfer.py`: Refactor `align_subjects` and `regularized_within_subject_cv` to use shared utility

## 3. Subject-Specific Feature Selection via Expanded Band Pool

### Problem
All subjects use the same 8 fixed frequency bands (8-30 Hz). Motor imagery frequency patterns are highly individual — the optimal bands for a strong subject (e.g., 0006 at 85%) likely differ from a weak subject (e.g., 0001 at 39%).

### Design

**Important framing**: This is **subject-specific feature selection**, not band selection. SelectKBest with `f_classif` scores individual CSP features, not entire bands. It may pick partial bands (e.g., CSP component 1 from one band and component 3 from another). This is a weaker form of band selection but still useful — it lets different subjects weight different frequency regions without requiring group-selection infrastructure. True band-level selection (group lasso, per-band CV scoring) is a possible follow-up if feature-level selection shows promise.

**Expanded band pool** (10 non-overlapping bands):
```
(6, 8),                    # theta-mu border (new)
(8, 10), (10, 12),         # mu (keep)
(12, 14), (14, 16),        # low beta (keep)
(16, 18), (18, 20),        # mid beta (keep)
(20, 24),                  # high beta (keep)
(24, 28), (28, 34)         # upper beta (new, split old 24-30)
```

Non-overlapping bands avoid correlated CSP features that waste K capacity. The original 7 overlapping mu bands (6-14 Hz sliding window) were replaced with this cleaner set after review identified that `f_classif` doesn't account for feature correlation and would redundantly select from overlapping bands.

**Feature space**: 10 bands x 3 CSP = 30 FBCSP features + 45 handcrafted = **75 total features**.

**K candidates**: `(3, 5, 8, 10, 15, 20, 25, 30)`. Dynamically capped per subject at `n_inner_train - 1`:
- Originals (inner train ~77): all K values available
- Subject 104 (inner train ~49): all K values available
- Subject 106 (inner train ~20): K capped at 19, effective candidates = (3, 5, 8, 10, 15, 19)

**Compute estimate**: ~2-2.5x current cost. Breakdown: 10/8 = 1.25x for CSP fits, 1.25x for band caching, plus 8 K candidates (up from 7) adds ~15% to inner CV. Band caching mitigates filtering cost.

### Files touched
- `src/train.py`: Update `FBCSP_BANDS` to the 10-band set, add dynamic K capping
- `src/config.py`: Update `k_candidates` in `TrainingConfig`

## 4. Evaluation & Reporting

### Primary metric
Augmented nested CV accuracy (mean across subjects).

### Dual reporting
- **All subjects** (12): Full picture including 104 and 106.
- **Original 10 subjects only**: Direct comparison against 60.1%/62.9% baseline.

### Per-subject breakdown
Same format as current `training_results.json`: per-classifier accuracy, nested selection method, augmented nested score. New subjects flagged with trial count and adapted fold counts.

### Multi-seed validation
Run with 3 seeds (42, 123, 7) to confirm gains are real. Acceptance criteria:
- Mean improvement across all 3 seeds is positive on the original 10 subjects.
- No individual seed shows regression exceeding **2 pp** on the original 10 subjects (relaxed from "no regression" — with 3 seeds and high per-subject variance, a strict zero-regression criterion would reject real improvements).

Note: with 3 seeds, statistical power is limited. Improvements below ~3 pp may not be reliably distinguishable from noise. This is a practical screen, not a formal hypothesis test.

### Files touched
- `src/pipeline.py`: Dual reporting logic, adaptive fold count in results
- `models/training_results.json`: Extended with new subjects

## 5. Implementation Ordering

The three changes interact and should be **implemented and validated sequentially**:

1. **Change 1 (Flexible Loading)** first — it's purely structural, does not affect accuracy of existing subjects. Validate: run pipeline with `min_trials=100` and confirm 60.1%/62.9% baseline is exactly reproduced. Then run with `min_trials=30` and confirm new subjects load correctly.

2. **Change 3 (Expanded Bands)** second — this is the highest-potential lever and is independent of the donor pool changes. Validate: run the original 10 subjects with expanded bands and compare against baseline. If accuracy drops, revert to original 8 bands before proceeding.

3. **Change 2 (EA + Expanded Donors)** third — this changes augmentation behavior and benefits from the flexible loader already being in place. Validate: compare augmented nested accuracy for weak subjects against baseline.

Each change gets its own commit. If a change hurts accuracy, it is reverted before the next change is attempted.

## 6. Rollback Criteria

| Change | Revert if... |
|--------|-------------|
| Expanded bands | Mean nested accuracy on original 10 subjects drops > 1 pp across 3 seeds |
| EA + expanded donors | Mean augmented nested accuracy on original 10 subjects drops > 1 pp across 3 seeds |
| Flexible loading | N/A (structural only, cannot affect accuracy of existing subjects) |

If all three changes land but the combined result is worse than baseline, revert in reverse order (change 2, then change 3) and keep whichever subset improves or matches baseline.

## 7. Integration Testing

Before any accuracy-affecting changes:
1. Run full pipeline with current code and seed=42. Confirm `training_results.json` matches 60.1% nested / 62.9% augmented.
2. After each change, re-run with seed=42 on original 10 subjects. Results must match baseline (for structural changes) or show documented improvement (for accuracy-affecting changes).
3. After all changes, run 3-seed validation.

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Overfitting with 75 features | SelectKBest + nested CV limits effective dimensionality; K capped at n_train - 1 |
| EA adds complexity | Well-understood linear transform; 3-function shared utility keeps code DRY |
| New subjects have noisy estimates (32-64 trials) | Dual reporting separates them; adapted fold counts acknowledged in output |
| Compute increase (~2-2.5x) | Band caching mitigates filtering; acceptable for research pipeline |
| Inner CV unreliable for subject 106 (20 inner-train samples) | Adapted inner folds (4 instead of 7); K capped; results flagged as lower-confidence |
| Feature selection picks partial bands, not full bands | Explicitly framed as feature selection; group-level selection noted as follow-up |
| Donor selection leakage (pre-existing) | Fixed: distance computed from training-fold covariance only |

## Non-goals

- Deep learning (EEGNet, ShallowConvNet) — likely data-limited with 32-100 trials per subject
- Multi-donor augmentation — previously tried and reverted; revisit only if single-donor + EA doesn't help
- Augmentation threshold tuning (0.50 -> 0.55) — previously tried without improvement; can revisit after other changes land
- Group-level band selection (mRMR, group lasso) — possible follow-up if feature-level selection shows promise
