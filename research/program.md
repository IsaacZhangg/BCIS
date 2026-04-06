# BCIS Auto-Research Program

## Objective
Maximize augmented nested CV accuracy for left/right motor imagery classification using 8-channel consumer EEG (g.tec Unicorn).

## Primary Metric
**Augmented nested CV mean accuracy** — the unbiased estimate of pick-best-classifier accuracy, with cross-subject donor augmentation for weak subjects.

## Baseline
- Nested CV: 62.0% (seed=42), true mean 58.9% ± 1.4%
- Augmented nested CV: 64.6% (seed=42)
- Parameter tuning ceiling reached (25+ experiments, all neutral or negative)

## Constraints
1. **Fixed evaluation**: nested CV structure (10 outer, 7 inner folds) must not change
2. **No data leakage**: all thresholds/models fitted on training folds only
3. **Fixed data**: same recordings, same subjects — cannot collect more data
4. **Dependencies**: only packages in pyproject.toml (may add new ones if justified)
5. **Reproducibility**: seed=42 for all experiments, document everything

## Modifiable Files
- `src/train.py` — classifiers, model selection, feature pipelines
- `src/features.py` — handcrafted feature extraction
- `src/config.py` — hyperparameters
- `src/epochs.py` — windowing, artifact rejection
- `src/preprocess.py` — preprocessing pipeline
- `src/pipeline.py` — orchestration (to connect new components)

## Fixed Files
- `src/data_loader.py` — raw data loading
- `src/validation.py` — CV split mechanics
- `tests/` — must still pass

## Experiments Completed

| # | Experiment | Nested | Augmented | Status |
|---|-----------|--------|-----------|--------|
| 1 | LightGBM 5th classifier | 62.0% | 63.8% | Reverted |
| 2 | ASR preprocessing | 60.4% | 62.0% | Reverted |
| 3a | Augmentation threshold 0.55 | 62.0% | 64.6% | Reverted (no change) |
| 3b | Augmentation threshold 0.60 | 62.0% | 64.2% | Reverted |
| 5 | Early/late ERD temporal features | 60.2% | 59.5% | Reverted |
| 6 | Multi-donor augmentation (k=2) | 62.0% | 63.5% | Reverted |

## Conclusion

After 6 structural experiments (on top of 25+ parameter experiments in previouslytried.md),
the **64.6% augmented nested CV** is confirmed as the algorithmic ceiling for this data.

The bottleneck is hardware and subject-level signal quality, not the algorithm:
- 4 of 10 subjects are effectively BCI-illiterate (~50% accuracy)
- 8-channel dry electrodes limit spatial resolution
- ~100 trials per subject limit covariance estimation quality

Further improvement requires: more data, better hardware, or subject training protocols.
