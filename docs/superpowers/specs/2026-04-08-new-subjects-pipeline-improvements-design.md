# New Subjects Integration & Pipeline Improvements

**Date**: 2026-04-08
**Branch**: P3LR
**Scope**: Integrate subjects 104-106, enhance augmentation, add subject-specific FBCSP bands

## Context

Three new subjects (104, 105, 106) have been recorded. The current pipeline achieves 60.1% nested CV mean across 10 subjects (unicorn-data) with 62.9% augmented. The accuracy ceiling from 31+ experiments was ~62% — adding more data and the one untried classical approach (subject-specific FBCSP bands) are the remaining levers within the current stack.

### New Subject Data

| Subject | Sessions | Trials/Session | Total Trials | Status |
|---------|----------|----------------|--------------|--------|
| subject0104 | 2 (session001, session002) | 32 each (16L/16R) | 64 | Merge sessions |
| subject0105 | 1 (session001) | 0 | 0 | **Empty — unusable** |
| subject0106 | 1 (session001) | 32 (16L/16R) | 32 | OK |

Stim encoding uses multi-digit values (e.g., 131 = trial 1/phase 3/left, 232 = trial 2/phase 3/right). The existing `load_recording` logic `(val // 10) % 10` for phase and `val % 10` for movement handles this correctly.

## Design

### 1. Data Integration

**Goal**: Include MI_DATA_NEW subjects in the main 4-classifier evaluation pipeline, not just as transfer donors.

**Changes to `data_loader.py`**:
- Add `get_recordings_flexible(data_dir, min_trials=20)` function that accepts any recording with >= `min_trials` phase-3 events (replaces the hard-coded `== 100` check for MI_DATA_NEW subjects).
- The existing `get_complete_recordings` stays unchanged for backward compatibility.

**Changes to `config.py`**:
- No merge mapping needed for subject0104 — both sessions live under `subject0104/` and will be auto-discovered and concatenated by the flexible loader (same pattern as `load_all_subjects` in transfer.py).

**Changes to `pipeline.py`**:
- After loading unicorn-data subjects, also scan `MI_DATA_NEW` for additional subjects using the flexible loader.
- Skip subjects already loaded from unicorn-data (e.g., subject0000).
- Skip subject0105 (empty recording — no stim events).
- Merge multi-session subjects (subject0104's two sessions concatenated).
- All discovered subjects run through the same preprocessing → epoching → feature extraction → 4-classifier nested CV path.
- Subjects with fewer trials naturally produce smaller CV folds but the nested CV handles this.

**Expected outcome**: ~12-13 subjects in the main evaluation (10 unicorn-data + subject0100, subject0101, subject0102, subject0104, subject0106 from MI_DATA_NEW, minus duplicates like subject0000).

### 2. Enhanced Augmented Nested CV

**Goal**: Better donor matching and augmentation with the larger subject pool.

**Multi-donor weighted augmentation**:
- Instead of the single closest donor, use top-K donors (K=3).
- Weight each donor's contribution by inverse Riemannian distance: `w_i = 1/d_i`, normalized so `sum(w_i) = 1`.
- Sample trials from each donor proportionally to their weight, keeping total augmentation size similar to before (matched to the size of the closest single donor's dataset).
- Leakage-free structure unchanged: donor data only in training folds.

**Raise weakness threshold**:
- Change `augmentation_weakness_threshold` from 0.50 to 0.55 in `TrainingConfig`.
- More borderline subjects get augmentation, and the larger pool provides better donors.

**Changes to `pipeline.py`**:
- Modify `_augmented_nested_cv_subject` call site to pass multi-donor data.
- New helper `_select_weighted_donors(target_mean, donor_means, donor_subjects, k=3)` that returns concatenated (feat, mc, y) arrays with distance-weighted sampling.

### 3. Subject-Specific FBCSP Band Optimization

**Goal**: Optimize the filter-bank frequency bands per subject within the inner CV, capturing individual variations in mu/beta rhythm frequencies.

**Candidate band pools**:
- Current fixed bands: `(8,10), (10,12), (12,16), (16,20), (20,24), (24,28), (28,32), (32,36)`.
- Extended candidate set with finer mu granularity:
  - Mu region: `(6,8), (7,9), (8,10), (9,11), (10,12), (11,13), (12,14)`
  - Beta region: `(13,17), (16,20), (20,24), (24,28), (28,32), (32,36), (36,40)`
- Define 3-4 pre-built band configurations (e.g., "standard", "low-mu", "high-mu", "wide-beta") as candidates.
- Inner CV tests each configuration alongside the existing band set and selects the one with best inner accuracy.
- This avoids combinatorial explosion — we're comparing a small number of curated band sets, not searching all combinations.

**Changes to `train.py`**:
- Define `FBCSP_BAND_CANDIDATES` as a list of band configurations.
- In the inner CV loop (within `train_nested_model_selection_cv`), for FBCSP-based classifiers (LDA, SVM), also vary the band configuration alongside `k_best`.
- The band configuration that yields the best inner CV accuracy is used for the outer fold.

**Changes to `config.py`**:
- Add `fbcsp_band_candidates` to `TrainingConfig` with the default candidate set.

### 4. Improved LOSO Transfer

No architectural changes. The larger subject pool (16 subjects in the donor pool) naturally improves:
- More training data per LOSO fold
- Better diversity for Euclidean Alignment
- Improved donor matching for adaptive augmented CV

The existing `run_transfer_evaluation` and `load_all_subjects` already scan both data directories and will automatically pick up the new subjects.

## Files Changed

| File | Changes |
|------|---------|
| `src/config.py` | Add `fbcsp_band_candidates`, update `augmentation_weakness_threshold` to 0.55 |
| `src/data_loader.py` | Add flexible recording discovery for MI_DATA_NEW |
| `src/pipeline.py` | Load MI_DATA_NEW subjects into main pipeline, multi-donor augmentation |
| `src/train.py` | Subject-specific FBCSP band selection in inner CV |
| `tests/test_data_loader.py` | Tests for flexible loader |
| `tests/test_train.py` | Tests for band selection logic |

## Success Criteria

- All valid new subjects (0104, 0106) appear in the main pipeline results table.
- Augmented nested CV uses multi-donor weighted augmentation.
- Inner CV selects best FBCSP band configuration per subject.
- No data leakage — verified by existing test suite + leakage reviewer.
- Pipeline runs without errors and produces `training_results.json` with updated results.

## Non-Goals

- Deep learning / EEGNet (deferred — user chose conservative approach)
- Changing the 4-classifier architecture
- New feature types beyond band configuration changes
