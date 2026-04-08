# New Subjects Integration & Pipeline Improvements (Revised)

**Date**: 2026-04-08 (revised after Codex review)
**Branch**: P3LR
**Scope**: Integrate subjects 104/106 into main pipeline, add subject-specific FBCSP band optimization

## Context

Three new subjects (104, 105, 106) have been recorded. The current pipeline achieves 62.0% nested CV mean (seed=42) across 10 subjects (unicorn-data) with 64.6% augmented. The multi-seed mean is 58.9% +/- 1.4%. After 31+ experiments, this is the established algorithmic ceiling.

Subject-specific FBCSP bands are the only remaining untried classical approach (`previouslytried.md` "Things Still Not Tried" #2). Adding more subjects expands the benchmark and improves the donor/transfer pool.

### What was dropped from v1 (per Codex review)

- **Multi-donor weighted augmentation**: Experiment 6 in `previouslytried.md` showed multi-donor (top-2) was "worse" overall — subject0002 dropped badly with the 2nd donor.
- **Weakness threshold 0.50 → 0.55**: Experiment 3a showed "no change" — only subject0009 was newly eligible, and it didn't benefit.
- **Extended frequency bands (6-8 Hz, 30-36+ Hz)**: Prior experiments showed these ranges hurt globally ("6-8Hz worse", "30-36 band is mostly noise").

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

**Key refactoring note**: The current main pipeline is per-recording (`run_pipeline` loops over `rec_path` and appends one subject entry per recording — `pipeline.py:326`). MI_DATA_NEW subjects have multiple recordings per subject that need merging. The transfer path already handles this via `load_all_subjects` (`transfer.py:50`), which does subject-level aggregation with multi-recording concatenation and per-recording artifact rejection. The main pipeline needs a similar subject-level loading step for MI_DATA_NEW data.

**Changes to `data_loader.py`**:
- Add `get_recordings_flexible(data_dir, min_trials=20)` — accepts recordings with >= `min_trials` phase-3 events. Groups by subject ID, returns `dict[str, list[Path]]`.
- The existing `get_complete_recordings` stays unchanged for unicorn-data backward compatibility.

**Changes to `pipeline.py`**:
- After processing unicorn-data recordings (existing per-recording loop), load MI_DATA_NEW subjects via a subject-level loader that:
  - Discovers all recordings per subject using `get_recordings_flexible`
  - Skips subjects already loaded from unicorn-data (e.g., subject0000)
  - Applies `DEFAULT_SUBJECT_MERGE` (subject0100_2 → subject0100, already configured)
  - For each subject: loads all recordings, preprocesses, extracts epochs per recording, applies per-recording artifact rejection, then concatenates across recordings
  - Extracts features from the concatenated epoch set
- All MI_DATA_NEW subjects then run through the same 4-classifier nested CV path.
- Report low-trial subjects separately in output (flag subjects with < 60 clean trials as "low-trial" in results).

**Statistical stability caveat**: Subjects with ~30 trials will have very small CV folds (3 trials/fold in 10-fold). Inner CV model selection will be noisy. These subjects' individual accuracy estimates should be interpreted cautiously, but they still contribute to the donor pool and transfer evaluation.

**Expected outcome**: ~15 subjects in the main evaluation (10 unicorn-data + subject0100, subject0101, subject0102, subject0104, subject0106 from MI_DATA_NEW, minus subject0000 which is a duplicate).

### 2. Subject-Specific FBCSP Band Optimization

**Goal**: Allow the inner CV to select the best FBCSP filter-bank configuration per subject, capturing individual mu/beta rhythm frequency variations.

This is the only genuinely untried classical lever. The current fixed bands are:
```
(8,10), (10,12), (12,14), (14,16), (16,18), (18,20), (20,24), (24,30)
```

**Design constraints** (informed by previous experiments):
- Do NOT extend below 8 Hz — theta/low-mu bands hurt globally (previouslytried.md Idea 8)
- Do NOT extend above 30 Hz — "(30,36) band is mostly noise" (previouslytried.md)
- Keep candidate set small (2-3 alternatives) to avoid overfitting on inner CV
- Candidates must be close to the proven current band set

**Candidate band configurations**:

| Name | Bands | Rationale |
|------|-------|-----------|
| `standard` | `(8,10),(10,12),(12,14),(14,16),(16,18),(18,20),(20,24),(24,30)` | Current proven set |
| `high_mu` | `(9,11),(11,13),(13,15),(15,18),(18,22),(22,26),(26,30)` | Shifts mu bands up 1 Hz for subjects with higher mu peaks |
| `wide_mu` | `(8,12),(10,14),(12,16),(16,20),(20,24),(24,30)` | Wider mu bands, fewer total bands — may help with low trial counts |

**Implementation**:
- Define `FBCSP_BAND_CANDIDATES` in `train.py` as the 3 configurations above.
- **Selection is per-subject, per-outer-fold**: In the inner CV of `train_nested_model_selection_cv`, for FBCSP-based classifiers (LDA, SVM), evaluate each band configuration. The winning (classifier, band_config) pair from inner CV is used for the outer fold.
- **Band cache**: Precompute bandpassed signals for the **union** of all candidate bands once per subject. `_precompute_bandpassed` (`train.py:42`) takes a `bands` parameter — call it once with the union set, then index into the cache for each candidate configuration.
- **Final model training**: The selected band configuration must propagate to `train_final_model` / `train_final_model_svm`. Store the winning band config per subject alongside the winning classifier method.
- **Tie-breaking**: If two band configs tie on inner CV accuracy, prefer `standard` (the proven default).

**Changes to `train.py`**:
- Add `FBCSP_BAND_CANDIDATES` constant
- Modify `_evaluate_classifiers_batch` to accept a `bands` parameter
- In nested CV inner loop, iterate over band candidates for FBCSP classifiers
- Propagate winning band config to outer fold evaluation and final model

**Changes to `config.py`**:
- Add `fbcsp_band_candidates` to `TrainingConfig` (default: the 3 configs above)

**Changes to `pipeline.py`**:
- Pass band candidates into nested CV
- Store selected band config per subject in results
- Use selected band config when training deployment models

### 3. Improved LOSO Transfer (Automatic)

No code changes needed. The existing `run_transfer_evaluation` and `load_all_subjects` already scan both data directories (`pipeline.py:699`). The new subjects will automatically appear in the donor pool (~15 subjects), providing more training data per LOSO fold.

**Clarification**: The spec v1 said more subjects help "Euclidean Alignment." This is imprecise — EA in `loso_cv` aligns each subject to its own covariance reference independently (`transfer.py:404`). More donors help the downstream classifier's training set, not EA itself.

**Known transductive element**: `loso_cv` uses all of a held-out subject's unlabeled test trials for its EA reference computation (`transfer.py:430`). This is not label leakage but is more generous than strict online transfer. Documenting, not changing.

### 4. Known Leakage Risk in Augmented CV (Document Only)

The current augmented nested CV selects the closest donor using the full target subject's covariance mean computed before the outer CV split (`pipeline.py:557-577`). This means donor identity is informed by test-fold data. This is a pre-existing issue, not introduced by this spec.

**Not fixing now** because: (a) the donor pool is fixed per subject regardless of fold, (b) moving donor selection inside each outer fold would require recomputing Riemannian means per fold, which is expensive and the augmented path is already the slowest step, (c) the selection uses only second-order statistics (covariance mean), not labels.

Flagging for future consideration.

## Files Changed

| File | Changes |
|------|---------|
| `src/config.py` | Add `fbcsp_band_candidates` to `TrainingConfig` |
| `src/data_loader.py` | Add `get_recordings_flexible` for MI_DATA_NEW discovery |
| `src/pipeline.py` | Subject-level MI_DATA_NEW loading, pass band candidates to nested CV, store per-subject band config |
| `src/train.py` | `FBCSP_BAND_CANDIDATES`, band selection in inner CV, propagate to final model |
| `tests/test_data_loader.py` | Tests for flexible loader |
| `tests/test_train.py` | Tests for band selection logic |

## Success Criteria

- All valid new subjects (0104, 0106 + existing 0100-0102) appear in the main pipeline results table with per-subject accuracy
- Low-trial subjects flagged in output
- Inner CV selects best FBCSP band configuration per subject; selected config recorded in results
- Selected band config used for deployment model training
- No new data leakage introduced — verified by existing test suite + leakage reviewer
- Pipeline runs without errors and produces updated `training_results.json`

## Non-Goals

- Deep learning / EEGNet (deferred — user chose conservative approach)
- Multi-donor augmentation or threshold changes (already tried, failed)
- Changing the 4-classifier architecture
- New feature types (current 45-feature set is balanced — adding/removing features hurts)
- Fixing the pre-existing donor-selection leakage in augmented CV
