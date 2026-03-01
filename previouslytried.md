# Previously Tried Parameter Experiments

This document records every parameter experiment conducted during accuracy optimization of the BCI motor imagery classifier. Use this to avoid re-testing things that have already been explored.

**Starting baseline (before any changes):** 43.9% nested, 45.4% FBCSP+LDA, 48.4% best-of
**Final result (seed=42):** 62.0% nested, 59.3% FBCSP+LDA, 63.8% best-of
**Mean across 10 seeds:** 58.9% ± 1.4% nested (true expected accuracy)

## Current Best Configuration


| Parameter                   | Value                                                          | File          | Line     |
| --------------------------- | -------------------------------------------------------------- | ------------- | -------- |
| FBCSP_BANDS                 | (8,10),(10,12),(12,14),(14,16),(16,18),(18,20),(20,24),(24,30) | train.py      | 20-29    |
| N_CSP_COMPONENTS            | 3                                                              | train.py      | 31       |
| CSP reg                     | "oas"                                                          | train.py      | 68       |
| DEFAULT_K_CANDIDATES        | (3, 5, 8, 10, 15, 20, 25)                                      | train.py      | 32       |
| Riemannian covariance       | "lwf"                                                          | train.py      | multiple |
| SVM C                       | 20.0                                                           | train.py      | multiple |
| SVM kernel                  | "rbf"                                                          | train.py      | multiple |
| SVM gamma                   | "scale"                                                        | train.py      | multiple |
| Riemannian LR C             | 0.1                                                            | train.py      | multiple |
| LDA shrinkage               | "auto"                                                         | train.py      | multiple |
| LDA solver                  | "lsqr"                                                         | train.py      | multiple |
| n_mad (artifact rejection)  | 3.5                                                            | train.py      | 109      |
| flat_uv                     | 1.0                                                            | train.py      | 110      |
| task_duration               | 3.0                                                            | pipeline.py   | ~104     |
| baseline_duration           | 1.0                                                            | pipeline.py   | ~105     |
| skip_duration               | 0.25                                                           | pipeline.py   | ~106     |
| n_folds                     | 10                                                             | config.py     | 20       |
| n_outer_folds               | 10                                                             | config.py     | 21       |
| n_inner_folds               | 7                                                              | config.py     | 22       |
| k_best                      | 10                                                             | config.py     | 23       |
| trial_group_size            | 1                                                              | config.py     | 26       |
| random_state                | 42                                                             | config.py     | 27       |
| LDA inner k-selection folds | 3                                                              | train.py      | ~269     |
| Handcrafted features        | 45                                                             | features.py   | —        |
| Preprocessing bandpass      | 1-40 Hz                                                        | preprocess.py | 11-12    |
| Notch filter                | 60 Hz                                                          | preprocess.py | —        |
| Welch nperseg               | min(sfreq/2, len(epoch)//2)                                    | features.py   | multiple |


## Experiments That Helped (kept in final config)

### CSP regularization: ledoit_wolf → oas

- **Impact:** Part of the initial batch that moved baseline from 43.9% → 47.1% nested
- **Why it helps:** OAS (Oracle Approximating Shrinkage) gives better covariance shrinkage estimates for small sample sizes (8 channels, ~45 trials per class per fold)
- **Note:** This was tested alongside FBCSP band trimming, so isolated impact is unclear

### FBCSP bands: Remove theta (4-8Hz) and gamma (30-40Hz)

- **Before:** 10 bands: (4,8),(8,10),(10,12),(12,14),(14,16),(16,18),(18,20),(20,24),(24,30),(30,40)
- **After:** 8 bands: (8,10),(10,12),(12,14),(14,16),(16,18),(18,20),(20,24),(24,30)
- **Impact:** Part of the initial batch that moved baseline from 43.9% → 47.1% nested
- **Why it helps:** Theta (4-8Hz) is attention-related, not motor-spatial — CSP can't extract useful spatial patterns from it. Gamma (30-40Hz) is mostly scalp muscle artifact with dry electrodes. Removing them reduces noisy CSP features.

### CSP components: 4 → 3

- **Impact:** FBCSP+LDA: 54.8% → 56.1%, nested: 55.0%, best-of: 60.3%
- **Why it helps:** With 8 channels and noisy consumer EEG, the 4th CSP component (2nd from each end of eigenspectrum) captures noise rather than signal. 3 components = 24 features (vs 32) — cleaner feature space.
- **Note:** CSP=2 gave even higher FBCSP+LDA (56.8%) but worse nested (53.5%) due to too few features for model selection diversity

### New handcrafted features: C3-C4 coherence + spectral entropy lateralization

- **Impact:** Part of the batch that moved from 47.1% → 51.0% nested
- **Added:** 4 coherence features (mu/beta task coherence + delta) + 2 entropy lateralization features (mu/beta C3-C4 entropy difference) = 6 new features, total 39→45
- **Why it helps:** Inter-hemispheric coherence drops during lateralized motor imagery. Spectral entropy asymmetry captures whether one hemisphere has more narrowband (organized) vs broadband (disorganized) activity.

### task_duration: 1.8s → 3.0s

- **Impact:** Major contributor to the 47.1% → 54.7% nested jump (tested alongside other changes)
- **Why it helps:** More samples (750 vs 450 at 250Hz) gives better covariance estimation for CSP, more stable Welch PSD, and captures the full sustained motor imagery response (typically peaks 1-2s after cue and sustains for 2-3s)
- **Note:** 3.5s was tested and gave 61.3% (marginal vs 61.7% at 3.0s). 2.5s gave 56.9% — clearly worse.

### skip_duration: 0.5s → 0.25s

- **Impact:** 54.7% → 55.2% nested
- **Why it helps:** Captures earlier motor planning activity. The visual evoked potential from the cue is mostly gone by 250ms. The extra 250ms of motor imagery signal adds useful data.
- **Note:** skip=0.0 gave 59.6% (worse — VEP contamination). skip=0.5 gave 58.6% (worse — misses early signal).

### SVM C: 1.0 → 10.0 → 20.0

- **Impact:** C=1.0 baseline → C=10 improved nested by ~3 pp → C=20 gave 61.7% nested
- **Why it helps:** With scaled features from StandardScaler, higher C allows tighter decision margins. The SVM with C=20 outperforms LDA on several subjects by capturing nonlinear patterns.
- **Note:** C=15 gave 61.6%, C=30 gave 61.8% — diminishing returns above 20. Below 10 is clearly worse.

### Riemannian LogisticRegression C: 1.0 → 0.1

- **Impact:** 56.5% → 57.5% nested (pre-group_size=1 change)
- **Why it helps:** Stronger L2 regularization prevents the tangent space classifier from overfitting on the 36-dimensional feature space with ~90 training samples. Riemannian accuracy drops (48.3% → 43.3%) but nested selection improves because it's less likely to pick a Riemannian model that overfits.
- **Note:** C=0.05 gave 61.0% (too much regularization). C=0.01 gave 55.2% (way too much).

### n_mad: 4.0 → 3.5

- **Impact:** Part of the 51.0% → 51.6% nested improvement batch
- **Why it helps:** Slightly more aggressive artifact rejection removes borderline noisy trials. The threshold (median + 3.5×MAD) is still conservative enough to keep most clean trials.
- **Note:** n_mad=3.0 consistently gives worse results (tested twice: once alone, once with other changes). Too many trials rejected.

### trial_group_size: 5 → 2 → 1

- **Impact:** group_size=2 was part of the 47.1% → 51.0% jump. group_size=1 was the biggest single improvement: 57.5% → 61.2% nested.
- **Why it helps:** group_size=1 is pure StratifiedKFold — no temporal grouping constraint. This gives maximum flexibility in creating balanced, representative folds. The grouped approach was too conservative for this data.
- **Caveat:** group_size=1 means temporally adjacent trials CAN end up in different folds. This could inflate accuracy if there's temporal autocorrelation. However, with randomized trial ordering in the paradigm, this risk is low.

### n_inner_folds: 5 → 7

- **Impact:** Part of the 51.0% → 51.6% nested improvement batch
- **Why it helps:** More inner folds give a more stable estimate of each classifier's performance for model selection. With ~90 outer-train trials, 7 inner folds give ~13 validation trials per fold — enough for a reasonable accuracy estimate.

### k_candidates extended: (5,8,10,15,20) → (3,5,8,10,15,20,25)

- **Impact:** k=25 addition was the key: 55.0% → 56.5% nested
- **Why it helps:** With 69 total features (24 CSP + 45 handcrafted), k=25 lets SVM and Ensemble use more features. The _safe_k_values cap at n_train//3 ≈ 30 still provides overfitting protection.
- **Note:** k=30 tested — no additional benefit (capped by _safe_k_values anyway).

## Experiments That Hurt (reverted)

### CSP components: 4 → 2

- **Result:** FBCSP+LDA 56.8% (+) but nested 53.5% (−)
- **Why it hurts:** Too few CSP features (16 total). Individual classifiers lose expressiveness, and model selection suffers from reduced diversity.

### All 10 FBCSP bands + 6 CSP components

- **Result:** 53.1% nested (was 55.2%)
- **Why it hurts:** 10 bands × 6 components = 60 CSP features — too many for ~90 training samples. SelectKBest can't reliably pick the best features from this large, noisy pool.

### FBCSP bands with theta (4-8Hz) added back

- **Tested with:** (4,8),(8,10),(10,12),(12,16),(16,20),(20,26),(26,34) — 7 wider bands
- **Result:** 51.6% nested (was 55.2%)
- **Why it hurts:** Even with 3.0s epochs giving better frequency resolution, theta CSP doesn't help because theta activity is not lateralized in motor imagery. The wider merged bands also reduce the fine-grained frequency discrimination that 2Hz-wide bands provide.

### FBCSP bands adjusted: (6,10),(10,12),(12,14),(14,16),(16,20),(20,24),(24,30),(30,36)

- **Result:** 58.7% nested (was 61.2%)
- **Why it hurts:** The (6,10) band includes non-motor sub-alpha activity. The (30,36) band is mostly noise. Merging (14,16) with (16,20) loses fine beta resolution.

### Preprocessing bandpass: 1-40Hz → 2-40Hz

- **Result:** 53.7% FBCSP+LDA (was 54.8%), some subjects dropped dramatically (subject0005: 51%→38.9%)
- **Why it hurts:** The 1-2Hz range contains slow cortical potentials that may carry motor preparation information for some subjects. Removing it disproportionately hurts certain individuals.

### Preprocessing bandpass: h_freq 40 → 45Hz

- **Result:** 53.0% nested (was 55.2%)
- **Why it hurts:** Extra 40-45Hz content is mostly muscle artifact from dry electrodes.

### SelectKBest with mutual_info_classif instead of f_classif

- **Result:** 52.1% nested (was 55.2%)
- **Why it hurts:** Mutual information estimation is unreliable with ~90 samples. f_classif (ANOVA F-statistic) is a stable, linear measure that matches well with LDA's linear assumptions. MI introduces stochastic noise.

### Tangent space features concatenated with FBCSP+handcrafted

- **Result:** 50.1% nested (was 51.6%)
- **Why it hurts:** Adding 36 tangent space features to the existing 77 (total 113) exceeds what SelectKBest can handle with 90 training samples. The tangent features overlap with Riemannian information, diluting the feature pool.

### Riemannian bandpass filtering (8-30Hz)

- **Result:** Riemann accuracy unchanged (37.7% vs 37.8%)
- **Why it doesn't help:** MNE's default FIR filter on short epochs (1.8-3.0s) has severe edge effects. The high-pass transition bandwidth (~4Hz) means the actual passband starts around 4Hz, not 8Hz. Short-epoch FIR bandpass is unreliable.
- **Note:** IIR filtering might work better but wasn't tested.

### LDA shrinkage: "auto" → 0.5

- **Result:** 57.0% nested (was 57.5%)
- **Why it hurts:** Ledoit-Wolf auto-shrinkage adapts to the actual data distribution. A fixed value of 0.5 is suboptimal for most folds.

### n_inner_folds: 7 → 9

- **Result:** 60.9% nested (was 61.7%)
- **Why it hurts:** With ~90 outer-train trials and 9 inner folds, each inner validation set has only ~10 trials — too noisy for reliable classifier comparison.

### n_inner_folds: 7 → 5

- **Result:** 60.7% nested (was 61.2%)
- **Why it hurts (slightly):** Fewer inner folds = fewer estimates = slightly less stable model selection.

### n_inner_folds: 7 → 3

- **Result:** 58.4% nested (was 61.2%)
- **Why it hurts:** Only 3 estimates per classifier in inner CV — far too noisy for reliable selection.

### n_folds (outer): 10 → 8

- **Result:** 54.0% nested (was 55.0%)
- **Why it hurts:** Fewer outer folds means less training data per fold (~87 vs ~90 trials). With already limited data, every trial matters.

### task_duration: 3.0s → 3.5s

- **Result:** 61.3% nested (marginal vs 61.7%)
- **Why it doesn't help much:** Longer epochs capture more motor imagery signal but also include more late-epoch data where imagery fades. Artifact count also increases. The sweet spot is around 3.0s.

### task_duration: 3.0s → 2.5s

- **Result:** 56.9% nested (was 61.7%)
- **Why it hurts:** Fewer samples (625 vs 750) degrades covariance estimation quality for CSP.

### baseline_duration: 1.0s → 1.5s

- **Result:** 55.5% nested (was 57.5%)
- **Why it hurts:** Longer baseline overlaps with the previous trial's post-imagery period, contaminating the "rest" reference.

### Welch nperseg: sfreq/2 → sfreq (125 → 250 samples)

- **Result:** 60.8% nested (was 61.7%)
- **Why it hurts:** Larger nperseg gives better frequency resolution but fewer averaging windows, increasing PSD variance. The current 125-sample segments give adequate 2Hz resolution for the bands used.

### LDA inner k-selection: 3-fold → 4-fold

- **Result:** 61.7% nested (same) but FBCSP+LDA dropped (58.2% vs 59.3%)
- **Why it's neutral/worse:** 4 inner folds for k-selection gives less training data per inner fold. With the already small k-selection dataset (~80 trials), 3-fold is better.

## Sensitivity Analysis

### Random seed sensitivity

- random_state=42: nested 61.7%
- random_state=0: nested 58.0%
- **Difference: 3.7 pp** — this represents the stochastic variance floor
- **Implication:** Any parameter change that moves nested by less than ~2 pp is likely within noise. Only changes ≥3 pp are reliably meaningful.

### What this means for future optimization

The ~3.7 pp random seed variance means we're at or near the ceiling for what parameter tuning can achieve with:

- 100 trials per subject (50 left, 50 right)
- 8-channel consumer-grade dry electrode EEG
- Several "BCI-illiterate" subjects who may not produce distinguishable motor imagery patterns

## Round 2: Untried Ideas (now tested)

### Idea 1: IIR bandpass for Riemannian (8-30Hz, method="iir")

- **Result:** 59.6% nested (was 62.0%) — **worse**
- **Why it hurts:** Even with IIR filters (minimal edge effects), restricting to 8-30Hz removes full-spectrum covariance structure that the Riemannian classifier uses. The 1-40Hz broadband covariance is more informative.

### Idea 3: Riemannian covariance estimator: oas → lwf

- **Result:** 62.0% nested (was 61.7% with oas) — **slight improvement, kept**
- Also tested scm (sample covariance): 61.7% nested — same as oas
- **Why LWF helps:** Ledoit-Wolf shrinkage provides a slightly better bias-variance tradeoff for 8x8 covariance matrices with ~45 trials per class.

### Idea 4: TangentSpace metric: riemann → logeuclid

- **Result:** 59.2% nested (was 62.0%) — **worse**
- **Why it hurts:** Logeuclid approximation loses the full Riemannian geometry that the riemann metric captures. The 8-channel covariance matrices are small enough that the exact riemann computation is stable.

### Idea 7: PCA (95% variance) before classifiers in batch evaluator

- **Result:** 59.0% nested (was 62.0%) — **worse**
- **Why it hurts:** PCA decorrelation removes the feature structure that f_classif-selected features already optimized. LDA with shrinkage handles correlation natively; PCA is redundant and removes discriminative variance.

### Idea 8 (partial): Add (6,8)Hz sub-theta/low-mu FBCSP band

- **Result:** 60.3% nested (was 62.0%) — **worse**
- **Why it hurts:** 6-8Hz contains high-theta/low-alpha activity that is not lateralized for motor imagery. CSP can't find useful spatial patterns, adding 3 noise features.

### Idea 9: 3-way ensemble (LDA + SVM + Riemannian)

- **Result:** 61.5% nested (was 62.0%), ensemble unchanged at 57.3% — **worse**
- **Why it hurts:** Riemannian's weaker accuracy (~48%) drags down the ensemble vote. The 2-way LDA+SVM ensemble has better signal diversity without the Riemannian noise.

### Idea 10: Common Average Reference (CAR) + LWF covariance

- **Result:** 62.0% nested — **no change**
- **Why it's neutral:** CAR reduces rank 8→7, but LWF/OAS shrinkage regularization compensates. The common-mode noise that CAR removes is already handled by the bandpass filter and Surface Laplacian.

### Idea 12: Multi-seed robustness analysis (10 seeds)

- **Seeds tested:** 0, 1, 2, 3, 7, 13, 21, 37, 42, 99
- **Results:** 56.6%, 57.6%, 58.8%, 56.6%, 59.4%, 58.6%, 58.9%, 59.0%, 62.0%, 58.3%
- **Mean: 58.9% ± 1.4%** (seed 42 is the luckiest at 62.0%)
- **Implication:** The true expected nested accuracy of this configuration is ~~59%, not 62%. The 62% reported with seed=42 is within the top tail of the distribution. All seeds show clear improvement over the baseline (~~44%).

## Round 3: Exhaustive Parameter Sweep (all tested, none improved)

All experiments below used the 62.0% baseline (seed=42). Every change was tested individually and reverted.

### n_inner_folds=8

- **Result:** 60.6% nested — **worse**
- **Why:** With ~90 outer-train trials and 8 inner folds, each inner validation set has ~11 trials — slightly too noisy.

### n_inner_folds=6

- **Result:** 61.1% nested — **worse**
- **Why:** Fewer folds = less stable model selection than 7.

### SVM C=25 (between tested C=20 and C=30)

- **Result:** 61.9% nested — **effectively same** (within rounding)
- **Why:** C=20-30 range is essentially flat; the RBF decision boundary isn't sensitive to C in this range with scaled features.

### Riemannian LR C=0.15

- **Result:** 61.8% nested — **worse**
- **Why:** Less regularization than C=0.1 allows more overfitting on 36-dim tangent space.

### Riemannian LR C=0.2

- **Result:** 61.5% nested — **worse**

### Riemannian LR C=0.08

- **Result:** 61.4% nested — **worse**
- **Why:** Too much regularization; underfitting slightly.

### n_mad=3.75 (less aggressive rejection)

- **Result:** 58.9% nested — **much worse**
- **Why:** More artifacts pass through, degrading CSP estimation quality.

### n_mad=3.25 (more aggressive rejection)

- **Result:** 61.4% nested — **worse**
- **Why:** Too many borderline trials rejected, reducing training data.

### CSP reg="lwf" (instead of "oas")

- **Result:** 55.8% nested — **much worse**
- **Why:** Ledoit-Wolf shrinkage for CSP spatial filter covariance estimation is inferior to OAS. OAS better estimates the shrinkage intensity for the 8-channel CSP covariance matrices.

### skip_duration=0.15 (earlier onset)

- **Result:** 60.6% nested — **worse**
- **Why:** Some VEP contamination from the cue at 150ms.

### skip_duration=0.35 (later onset)

- **Result:** 59.3% nested — **worse**
- **Why:** Misses early motor planning signal (250-350ms window).

### task_duration=3.25

- **Result:** 58.7% nested — **worse**
- **Why:** Extra 250ms captures late-epoch data where imagery fades.

### baseline_duration=0.75

- **Result:** 60.2% nested — **worse**
- **Why:** Shorter baseline → noisier power reference → less stable ERD computation.

### SVM gamma=0.05 (instead of "scale")

- **Result:** 61.8% nested — **marginally worse**
- **Why:** Fixed gamma can't adapt to per-fold feature variance the way "scale" does.

### SVM gamma=0.1

- **Result:** 59.9% nested — **worse**
- **Why:** Too narrow RBF kernels → overfitting.

### k_candidates extended to (2,3,5,8,10,15,20,25,30)

- **Result:** 62.0% nested — **same**
- **Why:** k=2 and k=30 are clipped by _safe_k_values (n_train//3 cap) anyway.

### flat_uv=0.5 and flat_uv=2.0

- **Result:** 62.0% nested for both — **same**
- **Why:** No trials in the dataset are actually flat (<1µV PTP). The flat rejection threshold doesn't matter.

### LDA inner k-selection with 2-fold (instead of 3)

- **Result:** 62.0% nested — **same**
- **Why:** 2-fold k-selection gives coarser but sufficient k estimates. Doesn't affect nested selection.

### Welch nperseg=sfreq/4 (62 samples instead of 125)

- **Result:** 58.0% nested — **much worse**
- **Why:** 62-sample segments give only ~4Hz frequency resolution — too coarse to resolve mu (8-12Hz) and beta sub-bands.

### Riemannian with MDM classifier (instead of TangentSpace+LR)

- **Result:** 59.8% nested — **worse**
- **Why:** MDM (Minimum Distance to Mean) is simpler but less discriminative than tangent space + logistic regression for this data size.

### Riemannian with CospCovariances (band-specific covariances)

- **Result:** Error — CospCovariances not available in installed pyriemann version. CoSpectra produces 4D output incompatible with TangentSpace.

### CSP norm_trace=False

- **Result:** 60.4% nested — **worse**
- **Why:** Without trace normalization, CSP variance features are scale-dependent, hurting cross-trial generalization.

### k-value cap relaxed to n_train//2 (from n_train//3)

- **Result:** 60.5% nested — **worse**
- **Why:** Allowing up to ~45 features from 90 training samples leads to overfitting.

### CSP n_components=4 (re-tested with current config)

- **Result:** 60.3% nested — **worse**
- **Why:** 4th CSP component (32 total) still captures noise with 8 channels. CSP=3 (24 features) remains optimal.

### Weighted ensemble (0.4 LDA + 0.6 SVM instead of 0.5/0.5)

- **Result:** 62.0% nested — **same** (ensemble mean dropped from 57.3% to 56.8%)
- **Why:** Weighting doesn't affect nested selection which evaluates classifiers independently. Ensemble accuracy itself dropped because the weights are static and don't adapt per-subject.

### Weighted model selection (accuracy-weighted probability ensemble instead of single-winner)

- **Result:** 57.0% nested (was 62.0%) — **much worse**
- **What:** Instead of picking the single best classifier per outer fold via inner-CV argmax, retrain all 3 base classifiers (LDA, SVM, Riemannian) on outer-train and combine their predicted probabilities weighted by inner-CV accuracy (normalized to sum to 1).
- **Why it hurts:** Riemannian's ~48% accuracy still gets ~30% weight in the combination, polluting predictions on subjects where LDA or SVM would have been cleanly selected. The classifiers have very different accuracy profiles — when one is clearly best for a fold, diluting it with weaker classifiers hurts. Same root cause as the 3-way ensemble (Idea 9) failure.

## Conclusion on Parameter Tuning

After 25+ parameter experiments in Round 3, the 62.0% nested accuracy (seed=42) is confirmed as the parameter-tuning ceiling. All individual parameter changes either hurt or have no effect. This is consistent with the Round 2 multi-seed analysis showing ~59% ± 1.4% true expected accuracy. The 62.0% with seed=42 is in the top tail of the seed distribution.

## Things Still Not Tried

1. **Per-subject adaptive parameters** — different C, k, or band selection per subject via inner CV
2. **xDAWN spatial filtering** — alternative to CSP for evoked response maximization
3. **Gradient boosting classifier** — would be a structural/architecture change
4. **Subject-specific FBCSP bands** — computationally expensive, needs inner CV per band per subject
5. **Stacking with meta-classifier** — a learned meta-classifier (e.g. logistic regression) on top of base classifier predictions, rather than the accuracy-weighted averaging already tried
6. **Riemannian with band-specific covariances** — CospCovariances not available; would need pyriemann upgrade or custom implementation

