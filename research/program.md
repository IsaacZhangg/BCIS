# BCIS Auto-Research Program

## Objective
Maximize **augmented nested CV** accuracy for left/right motor imagery classification using 8-channel consumer EEG (g.tec Unicorn).

## Primary Metric
**Augmented nested CV mean accuracy** — unbiased nested selection (inner 7-fold picks the classifier, outer 10-fold evaluates), with leakage-free Euclidean-Alignment donor pooling for weak subjects (nested < 52%). Nested mean is reported alongside it; donor pooling must not erase a better within-subject model (`max(nested, gated)`).

## Current Result (seed=42, 15 subjects)

| Metric | Value |
|--------|-------|
| Nested CV | **59.3%** |
| Augmented nested CV | **61.3%** |
| Original 10-subject nested | **60.5%** |
| LOSO transfer | **51.3%** |
| Subjects reaching ≥60% | 6 of 15 |

Historical 10-subject parameter-tuning ceiling (before MI_DATA_NEW + Composite CSP + session EA): nested 62.0% / augmented 64.6% at seed=42, true mean 58.9% ± 1.4% across 10 seeds. That 10-subject 62.0% is **not** the current headline number.

Full per-subject scores are generated locally in the gitignored `models/training_results.json`. This public log retains aggregate results only. Experiment narrative lives in `previouslytried.md`.

## Constraints
1. **Fixed evaluation**: nested CV structure (10 outer, 7 inner folds) must not change
2. **No data leakage**: thresholds, EA transforms, donor choice, and models fitted on training folds only
3. **Fixed recordings**: use the same authorized local files under the gitignored `data/unicorn-data/` and `data/MI_DATA_NEW/` directories when comparing experiments
4. **Dependencies**: packages in `pyproject.toml` (new ones only if justified)
5. **Reproducibility**: seed=42 for reported runs; document every experiment

## Current Configuration (kept)

- Epochs: 1.0 s baseline, 0.25 s skip, 3.0 s task; in-fold PTP rejection (`n_mad=3.5`)
- Classifiers in nested selection: FBCSP+LDA, Riemannian (LWF + TangentSpace + LR), FBCSP+SVM (C=20), Ensemble, Composite CSP (λ=0.3)
- Inner CV also picks FBCSP band config: `standard` / `high_mu` / `wide_mu`
- Session-level EA on multi-session subjects, train-fold only
- Donor augmentation: weakness threshold **0.52**, 2 pp inner-CV margin, nearest Riemannian neighbor / pool_ea
- `min_evaluation_trials=30` excludes subjects with too few trials

## Modifiable Files
- `src/train.py` — classifiers, nested CV, model selection
- `src/features.py` — handcrafted feature extraction
- `src/config.py` — hyperparameters (`TrainingConfig`)
- `src/epochs.py` / `src/rejection.py` — windowing and in-fold rejection
- `src/preprocess.py` — filtering (ASR exists, not default)
- `src/composite_csp.py` / `src/alignment.py` / `src/transfer.py` — CCSP, EA, donor pooling
- `src/pipeline.py` — orchestration

## Fixed Files
- `src/data_loader.py` — raw loading, content-hash dedup, subject aliases
- `src/validation.py` — CV split mechanics
- `tests/` — must still pass (`uv run pytest tests/`)

## Experiments (summary)

| Round | What | Nested | Augmented | Status |
|-------|------|--------|-----------|--------|
| 1–3 | Parameter tuning (10 subjects) | 62.0% | 64.6% | Kept as 10-subject config |
| 4 | Structural (LightGBM, ASR, PLV, …) | ≤62.3% | ≤64.6% | All reverted |
| 5 | Add MI_DATA_NEW + FBCSP band search | 53.4% | — | Kept (15-subject set) |
| 6 | Unified load, dedup, EA-gated donors | 56.5% | 58.7% | Kept |
| 7 | Composite CSP (λ=0.3) | 57.4% | 58.7% | Kept |
| 8 | Session-level EA | **59.3%** | 60.3% | Kept |
| 9 | Temporal aug / CCSP λ=0.1, 0.5 / group-mean shrink | ≤58.6% | — | Reverted |
| 10 | Weakness threshold 0.50 → **0.52** | 59.3% | **61.3%** | Kept |
| 11 | Threshold 0.55; CCSP sources ≥80 trials | 58.0% / same aug | 61.3% | Reverted |

Details and failed parameter sweeps: `previouslytried.md`. Machine-readable log: `research/results.tsv`.

## Conclusion

The current algorithmic ceiling on this dataset is **59.3% nested / 61.3% augmented** (15 subjects, seed=42). Composite CSP, session EA, and the 0.52 donor cutoff are the methods that moved the 15-subject mean after the original 10-subject parameter search.

The bottleneck is still hardware and subject-level signal quality:
- 9 of 15 subjects remain at chance on nested CV
- 8-channel dry electrodes limit spatial resolution
- Several MI_DATA_NEW recordings have only 32–64 trials

Further improvement needs more data, better hardware, or subject training protocols — not another small hyperparameter tweak.
