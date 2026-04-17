"""Training pipeline: FBCSP + LDA, FBCSP + SVM, and Riemannian classifiers for left/right MI."""

import logging
from collections import Counter
from dataclasses import dataclass

import mne
import numpy as np
from joblib import Parallel, delayed
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    CacheScope,
    ParallelBackend,
    RiemannianClassifier,
    SplitStrategy,
)
from src.eegnet import TORCH_AVAILABLE as EEGNET_TORCH_AVAILABLE
from src.epochs import adaptive_threshold
from src.validation import build_classwise_trial_groups, make_cv_splits


@dataclass(frozen=True)
class AugmentationContext:
    """Training-time temporal augmentation data for a single subject.

    Each field indexed by ``origins`` refers back to the original (unaugmented)
    trial index.  ``X_features_canonical`` / ``X_multichannel_canonical`` are
    the centered-crop, one-row-per-trial tensors used at validation/test time
    so inference stays deterministic and matches the trained window length.

    Attributes:
        X_features_pool: Features computed from every augmented window across
            all original trials, shape ``(n_trials * n_windows, n_features)``.
        X_multichannel_pool: Augmented multichannel windows, shape
            ``(n_trials * n_windows, n_channels, window_samples)``.
        origins: Source trial index for each pool row, shape
            ``(n_trials * n_windows,)``.
        X_features_canonical: One row per original trial (centered crop).
        X_multichannel_canonical: One windowed trial per original trial
            (centered crop), shape ``(n_trials, n_channels, window_samples)``.
    """

    X_features_pool: np.ndarray
    X_multichannel_pool: np.ndarray
    origins: np.ndarray
    X_features_canonical: np.ndarray
    X_multichannel_canonical: np.ndarray

    def expand_train(
        self, train_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return augmented ``(X_feat, X_mc, origin_idx)`` for training trials."""
        train_set = set(int(i) for i in train_idx)
        mask = np.fromiter(
            (int(o) in train_set for o in self.origins),
            dtype=bool,
            count=len(self.origins),
        )
        return (
            self.X_features_pool[mask],
            self.X_multichannel_pool[mask],
            self.origins[mask],
        )


logger = logging.getLogger(__name__)

FBCSP_BANDS = [
    (8, 10),
    (10, 12),
    (12, 14),
    (14, 16),
    (16, 18),
    (18, 20),
    (20, 24),
    (24, 30),
]

N_CSP_COMPONENTS = 3
DEFAULT_K_CANDIDATES = (3, 5, 8, 10, 15, 20, 25)
ALL_CLASSIFIERS = ("lda", "riemann", "svm", "ensemble", "stacking")
OPTIONAL_CLASSIFIERS = ("eegnet",)


def available_classifiers() -> tuple[str, ...]:
    """Return the classifiers that are usable in the current environment.

    Optional classifiers (e.g. ``eegnet``) are included only when their
    backing libraries are importable; otherwise they are skipped with a log
    warning so pipeline runs degrade gracefully instead of crashing.
    """
    classifiers = list(ALL_CLASSIFIERS)
    if EEGNET_TORCH_AVAILABLE:
        classifiers.append("eegnet")
    else:
        logger.warning(
            "eegnet classifier unavailable: torch is not installed — skipping."
        )
    return tuple(classifiers)


BandCache = dict[tuple[float, float], np.ndarray]


class _BandpassTrials(BaseEstimator, TransformerMixin):
    """Sklearn transformer that bandpasses each trial along time.

    Used only when ``riemannian_band`` is set so the Riemannian covariance input
    can be narrowband (e.g. 8-30 Hz) while other classifiers keep broadband.
    """

    def __init__(self, low: float, high: float, sfreq: float):
        self.low = low
        self.high = high
        self.sfreq = sfreq

    def fit(self, X, y=None):  # noqa: D401 - sklearn signature
        return self

    def transform(self, X):
        return mne.filter.filter_data(
            X.astype(np.float64, copy=False),
            self.sfreq,
            self.low,
            self.high,
            verbose=False,
        )


def _make_riemann_pipeline(
    sfreq: float = 250.0,
    riemannian_band: tuple[float, float] | None = None,
    riemannian_classifier: RiemannianClassifier = "tangent_lr",
):
    """Create a fresh Riemannian classification pipeline.

    Defaults reproduce the baseline (lwf → TangentSpace → LR, broadband input).
    Optional knobs:

    - ``riemannian_band``: if set, prepend a per-trial bandpass so the
      covariance input is narrowband; leaves the broadband input used elsewhere
      (FBCSP/LDA/SVM) untouched.
    - ``riemannian_classifier``: ``"mdm"`` swaps TangentSpace+LR for
      ``pyriemann.classification.MDM(metric="riemann")``.
    """
    if riemannian_classifier == "mdm":
        from pyriemann.classification import MDM

        classifier_steps: tuple = (Covariances(estimator="lwf"), MDM(metric="riemann"))
    else:
        classifier_steps = (
            Covariances(estimator="lwf"),
            TangentSpace(metric="riemann"),
            LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000),
        )

    if riemannian_band is not None:
        low, high = riemannian_band
        return make_pipeline(
            _BandpassTrials(low=low, high=high, sfreq=sfreq), *classifier_steps
        )
    return make_pipeline(*classifier_steps)


def _precompute_bandpassed(
    X: np.ndarray,
    sfreq: float,
    bands: list[tuple[float, float]] = FBCSP_BANDS,
) -> BandCache:
    """Bandpass every trial for each FBCSP band once.

    Filtering is trial-wise along the time axis, so cached per-band tensors can be
    safely indexed for CV train/test splits without changing leakage boundaries.
    """
    filtered_by_band: BandCache = {}
    for low, high in bands:
        try:
            filtered_by_band[(low, high)] = mne.filter.filter_data(
                X, sfreq, low, high, verbose=False
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
    return filtered_by_band


def _subset_band_cache(cache: BandCache, indices: np.ndarray) -> BandCache:
    """Slice a precomputed band cache by trial indices."""
    return {band: X_band[indices] for band, X_band in cache.items()}


def _extract_fbcsp_features_prefiltered(
    X_train_by_band: BandCache,
    X_test_by_band: BandCache,
    y_train: np.ndarray,
    bands: list[tuple[float, float]] = FBCSP_BANDS,
    n_components: int = N_CSP_COMPONENTS,
    n_train_trials: int | None = None,
    n_test_trials: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[CSP, tuple[float, float]]]]:
    """Extract FBCSP features from pre-filtered tensors."""
    csp_models: list[tuple[CSP, tuple[float, float]]] = []
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for low, high in bands:
        band = (low, high)
        if band not in X_train_by_band or band not in X_test_by_band:
            continue
        try:
            csp = CSP(
                n_components=n_components,
                reg="oas",
                log=True,
                norm_trace=True,
            )
            train_parts.append(csp.fit_transform(X_train_by_band[band], y_train))
            test_parts.append(csp.transform(X_test_by_band[band]))
            csp_models.append((csp, band))
        except (ValueError, np.linalg.LinAlgError):
            continue

    if not train_parts:
        n_train = n_train_trials if n_train_trials is not None else len(y_train)
        n_test = n_test_trials
        if n_test is None:
            n_test = (
                next(iter(X_test_by_band.values())).shape[0] if X_test_by_band else 0
            )
        return np.empty((n_train, 0)), np.empty((n_test, 0)), []

    return np.hstack(train_parts), np.hstack(test_parts), csp_models


def _extract_fbcsp_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    sfreq: float,
    bands: list[tuple[float, float]] = FBCSP_BANDS,
    n_components: int = N_CSP_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, list[tuple[CSP, tuple[float, float]]]]:
    """Extract filter-bank CSP features for train/test splits.

    For each frequency band, bandpass-filters the data and fits CSP spatial
    filters on the training set, then transforms both train and test.

    Returns:
        Tuple of (train_features, test_features, list of (fitted_csp, band) pairs).
    """
    # Final-model training passes the same tensor as train/test; avoid filtering it twice.
    if X_train is X_test:
        shared_filtered = _precompute_bandpassed(X_train, sfreq, bands=bands)
        return _extract_fbcsp_features_prefiltered(
            shared_filtered,
            shared_filtered,
            y_train,
            bands=bands,
            n_components=n_components,
            n_train_trials=X_train.shape[0],
            n_test_trials=X_test.shape[0],
        )

    train_filtered = _precompute_bandpassed(X_train, sfreq, bands=bands)
    test_filtered = _precompute_bandpassed(X_test, sfreq, bands=bands)
    return _extract_fbcsp_features_prefiltered(
        train_filtered,
        test_filtered,
        y_train,
        bands=bands,
        n_components=n_components,
        n_train_trials=X_train.shape[0],
        n_test_trials=X_test.shape[0],
    )


def _extract_fbcsp_features_pretrained(
    X: np.ndarray,
    sfreq: float,
    csp_models: list[tuple[CSP, tuple[float, float]]],
) -> np.ndarray:
    """Apply pre-fitted CSP models to new data at inference time.

    Returns:
        Feature array of shape (n_trials, n_csp_components * n_bands).
    """
    parts = []
    for csp, (low, high) in csp_models:
        X_filt = mne.filter.filter_data(X, sfreq, low, high, verbose=False)
        parts.append(csp.transform(X_filt))

    if not parts:
        return np.empty((X.shape[0], 0))

    return np.hstack(parts)


def _reject_in_fold(
    trial_ptps: np.ndarray | None,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_mad: float = 3.5,
    flat_uv: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter train/test indices using an adaptive amplitude threshold.

    The threshold is computed from training indices only, then applied to both
    splits.  When *trial_ptps* is ``None``, indices are returned unchanged
    (backward compatibility).

    Args:
        trial_ptps: Max peak-to-peak amplitude per trial, or ``None``.
        train_idx: Training fold indices.
        test_idx: Test fold indices.
        n_mad: Number of MADs above median for the threshold.
        flat_uv: Minimum PTP below which a trial is considered flat.

    Returns:
        (clean_train_idx, clean_test_idx).
    """
    if trial_ptps is None:
        return train_idx, test_idx

    train_ptps = trial_ptps[train_idx]
    threshold = adaptive_threshold(train_ptps, n_mad)

    def _keep(idx: np.ndarray) -> np.ndarray:
        ptps = trial_ptps[idx]
        mask = (ptps >= flat_uv) & (ptps <= threshold)
        return idx[mask]

    return _keep(train_idx), _keep(test_idx)


def _resolve_subject_groups(
    y: np.ndarray,
    provided_groups: list[np.ndarray] | None,
    subject_index: int,
    split_strategy: SplitStrategy,
    trial_group_size: int,
) -> np.ndarray | None:
    """Return per-trial groups for a subject when grouped CV is enabled."""
    if split_strategy != "stratified_group":
        return None
    if provided_groups is not None:
        return provided_groups[subject_index]
    return build_classwise_trial_groups(y, group_size=trial_group_size)


def _safe_k_values(
    n_features: int,
    n_train_trials: int,
    k_best: int,
    k_candidates: tuple[int, ...],
) -> list[int]:
    """Return conservative K values to reduce overfitting on small folds."""
    if n_features <= 0:
        return []

    k_cap = min(n_features, max(2, n_train_trials // 3))
    candidates = sorted(k for k in k_candidates if 1 <= k <= k_cap)
    return candidates or [min(k_best, k_cap)]


def _evaluate_subject_all_models(
    X_features: np.ndarray,
    X_multichannel: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    n_folds: int,
    k_best: int,
    trial_ptps: np.ndarray | None,
    split_strategy: SplitStrategy,
    groups: np.ndarray | None,
    random_state: int,
    k_candidates: tuple[int, ...],
    enable_band_cache: bool,
    max_blas_threads_per_worker: int,
    riemannian_band: tuple[float, float] | None = None,
    riemannian_classifier: RiemannianClassifier = "tangent_lr",
    augmentation: AugmentationContext | None = None,
) -> dict[str, float]:
    """Evaluate all classifiers for a single subject."""
    with threadpool_limits(limits=max_blas_threads_per_worker or None):
        splits = make_cv_splits(
            y,
            n_splits=n_folds,
            strategy=split_strategy,
            groups=groups,
            random_state=random_state,
        )
        fold_scores: dict[str, list[float]] = {name: [] for name in ALL_CLASSIFIERS}
        # Band cache is keyed by original trial index — skip it when augmentation
        # changes the per-trial sample layout, since per-band slicing by train/test
        # indices would no longer match the augmented tensors.
        subject_band_cache = (
            _precompute_bandpassed(X_multichannel, sfreq)
            if enable_band_cache and augmentation is None
            else None
        )

        for train_idx, test_idx in splits:
            train_idx, test_idx = _reject_in_fold(trial_ptps, train_idx, test_idx)
            if len(train_idx) < 2 or len(test_idx) < 1:
                continue

            if augmentation is not None:
                X_feat_train, X_mc_train, train_origins = augmentation.expand_train(
                    train_idx
                )
                y_train = y[train_origins]
                X_feat_test = augmentation.X_features_canonical[test_idx]
                X_mc_test = augmentation.X_multichannel_canonical[test_idx]
                y_test = y[test_idx]
            else:
                X_feat_train = X_features[train_idx]
                X_feat_test = X_features[test_idx]
                X_mc_train = X_multichannel[train_idx]
                X_mc_test = X_multichannel[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

            riemann_pipe = _make_riemann_pipeline(
                sfreq=sfreq,
                riemannian_band=riemannian_band,
                riemannian_classifier=riemannian_classifier,
            )
            riemann_pipe.fit(X_mc_train, y_train)
            fold_scores["riemann"].append(float(riemann_pipe.score(X_mc_test, y_test)))

            train_band_cache: BandCache | None = None
            test_band_cache: BandCache | None = None
            if subject_band_cache is not None:
                train_band_cache = _subset_band_cache(subject_band_cache, train_idx)
                test_band_cache = _subset_band_cache(subject_band_cache, test_idx)
                fbcsp_train, fbcsp_test, _ = _extract_fbcsp_features_prefiltered(
                    train_band_cache,
                    test_band_cache,
                    y_train,
                    n_train_trials=len(train_idx),
                    n_test_trials=len(test_idx),
                )
            else:
                fbcsp_train, fbcsp_test, _ = _extract_fbcsp_features(
                    X_mc_train, X_mc_test, y_train, sfreq
                )

            X_train_combined = np.hstack([fbcsp_train, X_feat_train])
            X_test_combined = np.hstack([fbcsp_test, X_feat_test])

            k_values = _safe_k_values(
                n_features=X_train_combined.shape[1],
                n_train_trials=len(y_train),
                k_best=k_best,
                k_candidates=k_candidates,
            )
            if not k_values:
                continue

            _, outer_counts = np.unique(y_train, return_counts=True)
            outer_priors = outer_counts / outer_counts.sum()

            best_k = k_values[0]
            best_inner_score = -1.0
            if augmentation is not None:
                # Keep all windows from the same original trial in the same
                # inner-CV fold by grouping on source-trial index.
                inner_groups = train_origins
                inner_strategy: SplitStrategy = "stratified_group"
            else:
                inner_groups = groups[train_idx] if groups is not None else None
                inner_strategy = split_strategy
            inner_splits = make_cv_splits(
                y_train,
                n_splits=3,
                strategy=inner_strategy,
                groups=inner_groups,
                random_state=random_state,
            )
            if inner_splits:
                for k_actual in k_values:
                    inner_scores = []
                    for inner_train, inner_val in inner_splits:
                        sel = SelectKBest(f_classif, k=k_actual)
                        X_it = sel.fit_transform(
                            X_train_combined[inner_train], y_train[inner_train]
                        )
                        X_iv = sel.transform(X_train_combined[inner_val])
                        sc = StandardScaler()
                        X_it = sc.fit_transform(X_it)
                        X_iv = sc.transform(X_iv)
                        _, inner_counts = np.unique(
                            y_train[inner_train], return_counts=True
                        )
                        inner_priors = inner_counts / inner_counts.sum()
                        clf = LinearDiscriminantAnalysis(
                            solver="lsqr", shrinkage="auto", priors=inner_priors
                        )
                        clf.fit(X_it, y_train[inner_train])
                        inner_scores.append(float(clf.score(X_iv, y_train[inner_val])))
                    mean_inner = float(np.mean(inner_scores))
                    if mean_inner > best_inner_score:
                        best_inner_score = mean_inner
                        best_k = k_actual

            lda_selector = SelectKBest(f_classif, k=best_k)
            X_train_lda = lda_selector.fit_transform(X_train_combined, y_train)
            X_test_lda = lda_selector.transform(X_test_combined)
            lda_scaler = StandardScaler()
            X_train_lda = lda_scaler.fit_transform(X_train_lda)
            X_test_lda = lda_scaler.transform(X_test_lda)
            lda = LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto", priors=outer_priors
            )
            lda.fit(X_train_lda, y_train)
            fold_scores["lda"].append(float(lda.score(X_test_lda, y_test)))

            k_fixed = k_values[-1]
            selector = SelectKBest(f_classif, k=k_fixed)
            X_train_sel = selector.fit_transform(X_train_combined, y_train)
            X_test_sel = selector.transform(X_test_combined)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_test_scaled = scaler.transform(X_test_sel)

            svm = SVC(
                kernel="rbf",
                C=20.0,
                gamma="scale",
                probability=True,
                random_state=42,
            )
            svm.fit(X_train_scaled, y_train)
            fold_scores["svm"].append(float(svm.score(X_test_scaled, y_test)))

            lda_ens = LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto", priors=outer_priors
            )
            lda_ens.fit(X_train_scaled, y_train)
            proba_lda = lda_ens.predict_proba(X_test_scaled)

            proba_svm = svm.predict_proba(X_test_scaled)

            avg_proba = (proba_lda + proba_svm) / 2
            preds = lda_ens.classes_[np.argmax(avg_proba, axis=1)]
            fold_scores["ensemble"].append(float(np.mean(preds == y_test)))

            # Stacking: reuse the inner_splits already generated above for k
            # tuning to also build OOF meta features from LDA/Riemann/SVM, then
            # refit bases on full outer-train for outer-test probabilities.
            # Skipped under augmentation (inner_splits index augmented rows,
            # not the canonical-crop tensors used at outer test).
            if augmentation is None and inner_splits:
                stacking_acc = _stacking_score(
                    X_feat_train,
                    X_feat_test,
                    y_train,
                    y_test,
                    X_mc_train,
                    X_mc_test,
                    sfreq,
                    k_best,
                    inner_splits=inner_splits,
                    prefiltered_train=train_band_cache,
                    prefiltered_test=test_band_cache,
                    riemannian_band=riemannian_band,
                    riemannian_classifier=riemannian_classifier,
                )
                fold_scores["stacking"].append(stacking_acc)

        return {
            name: float(np.mean(fold_scores[name])) if fold_scores[name] else 0.5
            for name in ALL_CLASSIFIERS
        }


def train_within_subject_cv_all_models(
    X_by_subject: list[tuple[np.ndarray, np.ndarray]],
    y_by_subject: list[np.ndarray],
    sfreq: float = 250.0,
    n_folds: int = 10,
    k_best: int = 10,
    trial_ptps_by_subject: list[np.ndarray] | None = None,
    split_strategy: SplitStrategy = "stratified_group",
    trial_groups_by_subject: list[np.ndarray] | None = None,
    trial_group_size: int = 5,
    random_state: int = 42,
    k_candidates: tuple[int, ...] = DEFAULT_K_CANDIDATES,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "loky",
    max_blas_threads_per_worker: int = 1,
    enable_band_cache: bool = True,
    riemannian_band: tuple[float, float] | None = None,
    riemannian_classifier: RiemannianClassifier = "tangent_lr",
    augmentation_by_subject: list[AugmentationContext | None] | None = None,
) -> dict[str, tuple[list[float], float, float]]:
    """Within-subject CV for all four classifiers in a single shared pass.

    This preserves the original train-only fitting boundaries and in-fold
    artifact rejection while avoiding repeated split generation and repeated
    FBCSP extraction across separate classifier entrypoints.

    Returns:
        Dict mapping classifier name ('lda', 'riemann', 'svm', 'ensemble') to
        (per_subject_scores, mean_accuracy, std_accuracy).

    Extra controls:
        n_jobs: Subject-level parallel jobs (1 disables parallelism).
        parallel_backend: joblib backend for subject-level parallelism.
        max_blas_threads_per_worker: BLAS/OpenMP thread cap per worker.
        enable_band_cache: Reuse precomputed per-band filtered trials.
    """
    payloads: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            np.ndarray | None,
            AugmentationContext | None,
        ]
    ] = []
    for subj_idx, ((X_features, X_multichannel), y) in enumerate(
        zip(X_by_subject, y_by_subject)
    ):
        groups = _resolve_subject_groups(
            y,
            provided_groups=trial_groups_by_subject,
            subject_index=subj_idx,
            split_strategy=split_strategy,
            trial_group_size=trial_group_size,
        )
        trial_ptps = trial_ptps_by_subject[subj_idx] if trial_ptps_by_subject else None
        aug = (
            augmentation_by_subject[subj_idx]
            if augmentation_by_subject is not None
            else None
        )
        payloads.append((X_features, X_multichannel, y, groups, trial_ptps, aug))

    if n_jobs == 1:
        per_subject_results = [
            _evaluate_subject_all_models(
                X_features,
                X_multichannel,
                y,
                sfreq=sfreq,
                n_folds=n_folds,
                k_best=k_best,
                trial_ptps=trial_ptps,
                split_strategy=split_strategy,
                groups=groups,
                random_state=random_state,
                k_candidates=k_candidates,
                enable_band_cache=enable_band_cache,
                max_blas_threads_per_worker=max_blas_threads_per_worker,
                riemannian_band=riemannian_band,
                riemannian_classifier=riemannian_classifier,
                augmentation=aug,
            )
            for X_features, X_multichannel, y, groups, trial_ptps, aug in payloads
        ]
    else:
        per_subject_results = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
            delayed(_evaluate_subject_all_models)(
                X_features,
                X_multichannel,
                y,
                sfreq=sfreq,
                n_folds=n_folds,
                k_best=k_best,
                trial_ptps=trial_ptps,
                split_strategy=split_strategy,
                groups=groups,
                random_state=random_state,
                k_candidates=k_candidates,
                enable_band_cache=enable_band_cache,
                max_blas_threads_per_worker=max_blas_threads_per_worker,
                riemannian_band=riemannian_band,
                riemannian_classifier=riemannian_classifier,
                augmentation=aug,
            )
            for X_features, X_multichannel, y, groups, trial_ptps, aug in payloads
        )

    scores_by_model: dict[str, list[float]] = {
        name: [subject_scores[name] for subject_scores in per_subject_results]
        for name in ALL_CLASSIFIERS
    }
    return {
        name: (
            subject_scores,
            float(np.mean(subject_scores)),
            float(np.std(subject_scores)),
        )
        for name, subject_scores in scores_by_model.items()
    }


def train_final_model(
    X_features: np.ndarray,
    X_multichannel: np.ndarray,
    y: np.ndarray,
    sfreq: float = 250.0,
    k_best: int = 10,
) -> dict:
    """Train a deployable FBCSP + LDA model on all data for a single subject.

    Returns:
        Dict with keys: csp_models, selector, scaler, classifier, sfreq, k_best.
    """
    fbcsp_features, _, csp_models = _extract_fbcsp_features(
        X_multichannel, X_multichannel, y, sfreq
    )

    X_combined = np.hstack([fbcsp_features, X_features])

    k = min(k_best, X_combined.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X_combined, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    classifier.fit(X_scaled, y)

    return {
        "csp_models": csp_models,
        "selector": selector,
        "scaler": scaler,
        "classifier": classifier,
        "sfreq": sfreq,
        "k_best": k,
    }


def predict(
    model: dict,
    X_features: np.ndarray,
    X_multichannel: np.ndarray,
) -> np.ndarray:
    """Run inference with a saved FBCSP + LDA model.

    Args:
        model: Dict returned by train_final_model.
        X_features: Handcrafted features of shape (n_trials, n_features).
        X_multichannel: Multichannel EEG of shape (n_trials, n_channels, n_samples).

    Returns:
        Predicted class labels.
    """
    fbcsp = _extract_fbcsp_features_pretrained(
        X_multichannel, model["sfreq"], model["csp_models"]
    )
    X_combined = np.hstack([fbcsp, X_features])
    X_selected = model["selector"].transform(X_combined)
    X_scaled = model["scaler"].transform(X_selected)
    return model["classifier"].predict(X_scaled)


# ---------------------------------------------------------------------------
# Riemannian geometry classifier (Covariances → TangentSpace → LogisticRegression)
# ---------------------------------------------------------------------------


def train_within_subject_cv_riemann(
    X_by_subject: list[np.ndarray],
    y_by_subject: list[np.ndarray],
    sfreq: float = 250.0,
    n_folds: int = 10,
    trial_ptps_by_subject: list[np.ndarray] | None = None,
    split_strategy: SplitStrategy = "stratified_group",
    trial_groups_by_subject: list[np.ndarray] | None = None,
    trial_group_size: int = 5,
    random_state: int = 42,
    riemannian_band: tuple[float, float] | None = None,
    riemannian_classifier: RiemannianClassifier = "tangent_lr",
) -> tuple[list[float], float, float]:
    """Within-subject CV using Riemannian tangent-space classifier.

    Per fold:
    1. Bandpass filter 8-30Hz (mu/beta bands)
    2. Covariances(estimator='oas') — shrinkage covariance estimation
    3. TangentSpace(metric='riemann') — project SPD matrices to tangent space
    4. LogisticRegression(C=1.0, solver='lbfgs') — classify

    Args:
        X_by_subject: List of multichannel EEG arrays (n_trials, n_channels, n_samples).
        y_by_subject: List of label arrays per subject.
        sfreq: Sampling frequency in Hz.
        n_folds: Requested number of CV folds.
        trial_ptps_by_subject: Per-trial max PTP amplitudes per subject for
            in-fold artifact rejection.  ``None`` disables per-fold rejection.
        split_strategy: ``'stratified'`` or ``'stratified_group'``.
        trial_groups_by_subject: Optional explicit CV groups per subject.
        trial_group_size: Group size when groups are auto-generated.
        random_state: Random seed for deterministic splits.

    Returns:
        Tuple of (per_subject_scores, mean_accuracy, std_accuracy).
    """
    scores = []

    for subj_idx, (X_multichannel, y) in enumerate(zip(X_by_subject, y_by_subject)):
        groups = _resolve_subject_groups(
            y,
            provided_groups=trial_groups_by_subject,
            subject_index=subj_idx,
            split_strategy=split_strategy,
            trial_group_size=trial_group_size,
        )
        splits = make_cv_splits(
            y,
            n_splits=n_folds,
            strategy=split_strategy,
            groups=groups,
            random_state=random_state,
        )
        fold_scores = []

        for train_idx, test_idx in splits:
            trial_ptps = (
                trial_ptps_by_subject[subj_idx] if trial_ptps_by_subject else None
            )
            train_idx, test_idx = _reject_in_fold(trial_ptps, train_idx, test_idx)
            if len(train_idx) < 2 or len(test_idx) < 1:
                continue
            pipe = _make_riemann_pipeline(
                sfreq=sfreq,
                riemannian_band=riemannian_band,
                riemannian_classifier=riemannian_classifier,
            )
            pipe.fit(X_multichannel[train_idx], y[train_idx])
            fold_scores.append(pipe.score(X_multichannel[test_idx], y[test_idx]))

        scores.append(float(np.mean(fold_scores)) if fold_scores else 0.5)

    mean_acc = float(np.mean(scores))
    std_acc = float(np.std(scores))
    return scores, mean_acc, std_acc


def train_final_model_riemann(
    X_multichannel: np.ndarray,
    y: np.ndarray,
    sfreq: float = 250.0,
    riemannian_band: tuple[float, float] | None = None,
    riemannian_classifier: RiemannianClassifier = "tangent_lr",
) -> dict:
    """Train a deployable Riemannian model on all data for a single subject.

    Returns:
        Dict with keys 'pipeline', 'sfreq'.
    """
    pipe = _make_riemann_pipeline(
        sfreq=sfreq,
        riemannian_band=riemannian_band,
        riemannian_classifier=riemannian_classifier,
    )
    pipe.fit(X_multichannel, y)
    return {"pipeline": pipe, "sfreq": sfreq}


# ---------------------------------------------------------------------------
# Nested model selection CV (unbiased best-of-4 estimate)
# ---------------------------------------------------------------------------


def _evaluate_classifiers_batch(
    names: list[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_mc_train: np.ndarray,
    X_mc_test: np.ndarray,
    sfreq: float,
    k_best: int,
    prefiltered_train: BandCache | None = None,
    prefiltered_test: BandCache | None = None,
    riemannian_band: tuple[float, float] | None = None,
    riemannian_classifier: RiemannianClassifier = "tangent_lr",
) -> dict[str, float]:
    """Train and score one or more classifiers on a fixed split.

    The helper reuses FBCSP extraction for the FBCSP-based heads ('lda', 'svm',
    'ensemble') so nested inner loops can evaluate all candidates without
    recomputing identical transforms.
    """
    # Stacking is handled outside this per-fold batch because it needs the full
    # inner CV to generate OOF meta features. Silently drop it here so callers
    # can pass ALL_CLASSIFIERS without extra bookkeeping.
    requested = set(names) - {"stacking"}
    unknown = requested.difference(ALL_CLASSIFIERS)
    if unknown:
        unknown_txt = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown classifier(s): {unknown_txt}")

    scores: dict[str, float] = {}

    if "riemann" in requested:
        pipe = _make_riemann_pipeline(
            sfreq=sfreq,
            riemannian_band=riemannian_band,
            riemannian_classifier=riemannian_classifier,
        )
        pipe.fit(X_mc_train, y_train)
        scores["riemann"] = float(pipe.score(X_mc_test, y_test))

    fbcsp_names = requested.intersection({"lda", "svm", "ensemble"})
    if not fbcsp_names:
        return scores

    if prefiltered_train is not None and prefiltered_test is not None:
        fbcsp_train, fbcsp_test, _ = _extract_fbcsp_features_prefiltered(
            prefiltered_train,
            prefiltered_test,
            y_train,
            n_train_trials=X_mc_train.shape[0],
            n_test_trials=X_mc_test.shape[0],
        )
    else:
        fbcsp_train, fbcsp_test, _ = _extract_fbcsp_features(
            X_mc_train, X_mc_test, y_train, sfreq
        )
    X_train_combined = np.hstack([fbcsp_train, X_train])
    X_test_combined = np.hstack([fbcsp_test, X_test])

    k_values = _safe_k_values(
        n_features=X_train_combined.shape[1],
        n_train_trials=len(y_train),
        k_best=k_best,
        k_candidates=DEFAULT_K_CANDIDATES,
    )
    if not k_values:
        for name in fbcsp_names:
            scores[name] = 0.5
        return scores

    k = k_values[-1]
    selector = SelectKBest(f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train_combined, y_train)
    X_test_sel = selector.transform(X_test_combined)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    _, counts = np.unique(y_train, return_counts=True)
    priors = counts / counts.sum()

    lda_model: LinearDiscriminantAnalysis | None = None
    if "lda" in fbcsp_names or "ensemble" in fbcsp_names:
        lda_model = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=priors
        )
        lda_model.fit(X_train_scaled, y_train)

    if "lda" in fbcsp_names and lda_model is not None:
        scores["lda"] = float(lda_model.score(X_test_scaled, y_test))

    svm_model: SVC | None = None
    if "svm" in fbcsp_names or "ensemble" in fbcsp_names:
        svm_model = SVC(
            kernel="rbf",
            C=20.0,
            gamma="scale",
            probability=True,
            random_state=42,
        )
        svm_model.fit(X_train_scaled, y_train)

    if "svm" in fbcsp_names and svm_model is not None:
        scores["svm"] = float(svm_model.score(X_test_scaled, y_test))

    if "ensemble" in fbcsp_names:
        if lda_model is None:
            raise RuntimeError("Internal error: LDA model missing for ensemble")
        proba_lda = lda_model.predict_proba(X_test_scaled)

        if svm_model is None:
            raise RuntimeError("Internal error: SVM model missing for ensemble")
        proba_svm = svm_model.predict_proba(X_test_scaled)

        avg_proba = (proba_lda + proba_svm) / 2
        preds = lda_model.classes_[np.argmax(avg_proba, axis=1)]
        scores["ensemble"] = float(np.mean(preds == y_test))

    return scores


def _fit_base_classifiers_and_proba(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    X_mc_train: np.ndarray,
    X_mc_test: np.ndarray,
    sfreq: float,
    k_best: int,
    prefiltered_train: BandCache | None = None,
    prefiltered_test: BandCache | None = None,
    riemannian_band: tuple[float, float] | None = None,
    riemannian_classifier: RiemannianClassifier = "tangent_lr",
) -> tuple[np.ndarray, np.ndarray]:
    """Fit LDA/Riemann/SVM on a train split and return positive-class probs on the test split.

    Stacking uses probability of class 1 from each base classifier so the meta
    feature matrix is (n_trials, 3). Storing only the positive-class column keeps
    the downstream logistic-regression minimal and avoids redundant columns that
    sum to 1.
    """
    riemann_pipe = _make_riemann_pipeline(
        sfreq=sfreq,
        riemannian_band=riemannian_band,
        riemannian_classifier=riemannian_classifier,
    )
    riemann_pipe.fit(X_mc_train, y_train)

    if prefiltered_train is not None and prefiltered_test is not None:
        fbcsp_train, fbcsp_test, _ = _extract_fbcsp_features_prefiltered(
            prefiltered_train,
            prefiltered_test,
            y_train,
            n_train_trials=X_mc_train.shape[0],
            n_test_trials=X_mc_test.shape[0],
        )
    else:
        fbcsp_train, fbcsp_test, _ = _extract_fbcsp_features(
            X_mc_train, X_mc_test, y_train, sfreq
        )
    X_train_combined = np.hstack([fbcsp_train, X_train])
    X_test_combined = np.hstack([fbcsp_test, X_test])

    k_values = _safe_k_values(
        n_features=X_train_combined.shape[1],
        n_train_trials=len(y_train),
        k_best=k_best,
        k_candidates=DEFAULT_K_CANDIDATES,
    )
    k = k_values[-1] if k_values else min(k_best, X_train_combined.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train_combined, y_train)
    X_test_sel = selector.transform(X_test_combined)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    _, counts = np.unique(y_train, return_counts=True)
    priors = counts / counts.sum()

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors)
    lda.fit(X_train_scaled, y_train)
    svm = SVC(kernel="rbf", C=20.0, gamma="scale", probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)

    def _pos_proba(model, X, classes) -> np.ndarray:
        proba = model.predict_proba(X)
        # Column for class label 1. Falls back to last column when classes differ.
        if 1 in classes:
            col = int(np.where(classes == 1)[0][0])
        else:
            col = proba.shape[1] - 1
        return proba[:, col]

    # Columns: [lda, riemann, svm] — fixed ordering so the meta-learner sees a
    # consistent feature layout across folds.
    lda_test = _pos_proba(lda, X_test_scaled, lda.classes_)
    svm_test = _pos_proba(svm, X_test_scaled, svm.classes_)
    riemann_test = _pos_proba(riemann_pipe, X_mc_test, riemann_pipe.classes_)
    return np.vstack([lda_test, riemann_test, svm_test]).T, np.array([])


def _stacking_score(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_mc_train: np.ndarray,
    X_mc_test: np.ndarray,
    sfreq: float,
    k_best: int,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    prefiltered_train: BandCache | None = None,
    prefiltered_test: BandCache | None = None,
    riemannian_band: tuple[float, float] | None = None,
    riemannian_classifier: RiemannianClassifier = "tangent_lr",
) -> float:
    """Compute stacking accuracy on (outer) test using leak-free OOF meta-training.

    Pipeline (no leakage):
      1. For each inner fold split of the outer-train set:
         - Fit LDA/Riemann/SVM on inner-train.
         - Record their positive-class probabilities on inner-val as OOF features.
      2. Train LogisticRegression meta-learner on the OOF matrix + y_train[covered].
      3. Refit LDA/Riemann/SVM on the FULL outer-train, predict probabilities on
         outer-test → X_meta_test. Meta-learner predicts on X_meta_test.

    The meta-learner NEVER sees outer-test predictions during its fitting. The
    base classifiers used at test time are trained on full outer-train, matching
    standard stacking.
    """
    n_train = len(y_train)
    # Columns are [lda, riemann, svm] positive-class probability.
    X_meta_oof = np.full((n_train, 3), np.nan, dtype=float)

    for inner_train_idx, inner_val_idx in inner_splits:
        if len(inner_train_idx) < 2 or len(inner_val_idx) < 1:
            continue
        # Inner-fold train/test share the outer-train tensors only; no outer-test
        # data is visible inside the inner loop.
        inner_prefilt_train: BandCache | None = None
        inner_prefilt_val: BandCache | None = None
        if prefiltered_train is not None:
            inner_prefilt_train = _subset_band_cache(prefiltered_train, inner_train_idx)
            inner_prefilt_val = _subset_band_cache(prefiltered_train, inner_val_idx)
        proba_block, _ = _fit_base_classifiers_and_proba(
            X_train[inner_train_idx],
            X_train[inner_val_idx],
            y_train[inner_train_idx],
            X_mc_train[inner_train_idx],
            X_mc_train[inner_val_idx],
            sfreq,
            k_best,
            prefiltered_train=inner_prefilt_train,
            prefiltered_test=inner_prefilt_val,
            riemannian_band=riemannian_band,
            riemannian_classifier=riemannian_classifier,
        )
        X_meta_oof[inner_val_idx] = proba_block

    # Some trials may miss from every inner-val split (rare but possible with
    # grouped folds). Drop rows the meta-learner has no coverage for.
    covered = ~np.isnan(X_meta_oof).any(axis=1)
    if covered.sum() < 2 or len(np.unique(y_train[covered])) < 2:
        return 0.5

    meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    meta.fit(X_meta_oof[covered], y_train[covered])

    # Refit bases on full outer-train and transform outer-test.
    X_meta_test, _ = _fit_base_classifiers_and_proba(
        X_train,
        X_test,
        y_train,
        X_mc_train,
        X_mc_test,
        sfreq,
        k_best,
        prefiltered_train=prefiltered_train,
        prefiltered_test=prefiltered_test,
        riemannian_band=riemannian_band,
        riemannian_classifier=riemannian_classifier,
    )
    preds = meta.predict(X_meta_test)
    return float(np.mean(preds == y_test))


def _evaluate_classifier(
    name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_mc_train: np.ndarray,
    X_mc_test: np.ndarray,
    sfreq: float,
    k_best: int,
    prefiltered_train: BandCache | None = None,
    prefiltered_test: BandCache | None = None,
) -> float:
    """Train and score a single classifier on pre-split data."""
    scores = _evaluate_classifiers_batch(
        [name],
        X_train,
        X_test,
        y_train,
        y_test,
        X_mc_train,
        X_mc_test,
        sfreq,
        k_best,
        prefiltered_train=prefiltered_train,
        prefiltered_test=prefiltered_test,
    )
    return scores[name]


def _evaluate_subject_nested_model_selection(
    X_features: np.ndarray,
    X_multichannel: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    n_outer_folds: int,
    n_inner_folds: int,
    k_best: int,
    trial_ptps: np.ndarray | None,
    split_strategy: SplitStrategy,
    groups: np.ndarray | None,
    random_state: int,
    enable_band_cache: bool,
    cache_scope: CacheScope,
    max_blas_threads_per_worker: int,
    augmentation: AugmentationContext | None = None,
) -> tuple[float, str, float]:
    """Nested model-selection CV for a single subject.

    Returns (selected_best_score, best_method, stacking_score). The stacking
    score is computed on each outer fold by training a LogisticRegression
    meta-learner on OOF predictions from LDA/Riemann/SVM collected across the
    inner CV folds, then predicting on outer-test with bases refit on full
    outer-train. Stacking does not participate in the inner-CV model selection;
    it is an independent per-fold evaluation reported alongside the selector.
    """
    # Inner-CV selection candidates exclude stacking because stacking needs the
    # full inner loop to generate OOF meta features, not a per-fold score.
    classifier_names = [c for c in ALL_CLASSIFIERS if c != "stacking"]
    aug_enabled = augmentation is not None
    with threadpool_limits(limits=max_blas_threads_per_worker or None):
        # Band cache requires stable per-trial sample shapes; disable when
        # augmentation changes the time axis length for training windows.
        subject_band_cache = (
            _precompute_bandpassed(X_multichannel, sfreq)
            if enable_band_cache and cache_scope == "subject" and not aug_enabled
            else None
        )
        outer_splits = make_cv_splits(
            y,
            n_splits=n_outer_folds,
            strategy=split_strategy,
            groups=groups,
            random_state=random_state,
        )
        fold_scores: list[float] = []
        fold_winners: list[str] = []
        stacking_fold_scores: list[float] = []

        for outer_train_idx, outer_test_idx in outer_splits:
            outer_train_idx, outer_test_idx = _reject_in_fold(
                trial_ptps, outer_train_idx, outer_test_idx
            )
            if len(outer_train_idx) < 2 or len(outer_test_idx) < 1:
                continue

            if aug_enabled:
                assert augmentation is not None
                # Outer train → augmented pool rows whose origin lies in
                # outer_train_idx. Outer test → canonical (centered crop).
                X_feat_otrain, X_mc_otrain, otrain_origins = augmentation.expand_train(
                    outer_train_idx
                )
                y_otrain = y[otrain_origins]
                X_feat_otest = augmentation.X_features_canonical[outer_test_idx]
                X_mc_otest = augmentation.X_multichannel_canonical[outer_test_idx]
                y_otest = y[outer_test_idx]
            else:
                X_feat_otrain = X_features[outer_train_idx]
                X_feat_otest = X_features[outer_test_idx]
                X_mc_otrain = X_multichannel[outer_train_idx]
                X_mc_otest = X_multichannel[outer_test_idx]
                y_otrain = y[outer_train_idx]
                y_otest = y[outer_test_idx]

            outer_train_cache: BandCache | None = None
            outer_test_cache: BandCache | None = None
            if not aug_enabled and enable_band_cache and subject_band_cache is not None:
                outer_train_cache = _subset_band_cache(
                    subject_band_cache, outer_train_idx
                )
                outer_test_cache = _subset_band_cache(
                    subject_band_cache, outer_test_idx
                )
            elif not aug_enabled and enable_band_cache:
                outer_train_cache = _precompute_bandpassed(X_mc_otrain, sfreq)
                outer_test_cache = _precompute_bandpassed(X_mc_otest, sfreq)

            if aug_enabled:
                # Split the ORIGINAL outer-train trial indices so each inner
                # fold has a clean split of source trials.  Then each inner
                # fold rebuilds its own augmented train and canonical val.
                inner_splits_orig = make_cv_splits(
                    y[outer_train_idx],
                    n_splits=n_inner_folds,
                    strategy=split_strategy,
                    groups=(groups[outer_train_idx] if groups is not None else None),
                    random_state=random_state,
                )
                inner_splits = []
                for it_orig, iv_orig in inner_splits_orig:
                    it_global = outer_train_idx[it_orig]
                    iv_global = outer_train_idx[iv_orig]
                    inner_splits.append((it_global, iv_global))
            else:
                inner_groups = groups[outer_train_idx] if groups is not None else None
                inner_splits = make_cv_splits(
                    y_otrain,
                    n_splits=n_inner_folds,
                    strategy=split_strategy,
                    groups=inner_groups,
                    random_state=random_state,
                )
            inner_scores_by_clf = {name: [] for name in classifier_names}
            for inner_train_idx, inner_val_idx in inner_splits:
                if aug_enabled:
                    assert augmentation is not None
                    inner_feat_train, inner_mc_train, inner_train_origins = (
                        augmentation.expand_train(inner_train_idx)
                    )
                    inner_y_train = y[inner_train_origins]
                    inner_feat_val = augmentation.X_features_canonical[inner_val_idx]
                    inner_mc_val = augmentation.X_multichannel_canonical[inner_val_idx]
                    inner_y_val = y[inner_val_idx]
                    split_scores = _evaluate_classifiers_batch(
                        classifier_names,
                        inner_feat_train,
                        inner_feat_val,
                        inner_y_train,
                        inner_y_val,
                        inner_mc_train,
                        inner_mc_val,
                        sfreq,
                        k_best,
                    )
                else:
                    inner_train_cache: BandCache | None = None
                    inner_val_cache: BandCache | None = None
                    if outer_train_cache is not None:
                        inner_train_cache = _subset_band_cache(
                            outer_train_cache, inner_train_idx
                        )
                        inner_val_cache = _subset_band_cache(
                            outer_train_cache, inner_val_idx
                        )
                    split_scores = _evaluate_classifiers_batch(
                        classifier_names,
                        X_feat_otrain[inner_train_idx],
                        X_feat_otrain[inner_val_idx],
                        y_otrain[inner_train_idx],
                        y_otrain[inner_val_idx],
                        X_mc_otrain[inner_train_idx],
                        X_mc_otrain[inner_val_idx],
                        sfreq,
                        k_best,
                        prefiltered_train=inner_train_cache,
                        prefiltered_test=inner_val_cache,
                    )
                for clf_name, score in split_scores.items():
                    inner_scores_by_clf[clf_name].append(score)
            inner_means = {
                name: float(np.mean(scores)) if scores else 0.5
                for name, scores in inner_scores_by_clf.items()
            }

            best_clf = max(inner_means, key=inner_means.get)  # type: ignore[arg-type]

            outer_score = _evaluate_classifier(
                best_clf,
                X_feat_otrain,
                X_feat_otest,
                y_otrain,
                y_otest,
                X_mc_otrain,
                X_mc_otest,
                sfreq,
                k_best,
                prefiltered_train=outer_train_cache,
                prefiltered_test=outer_test_cache,
            )
            fold_scores.append(outer_score)
            fold_winners.append(best_clf)

            # Stacking per outer fold: reuse the same inner_splits that drove
            # model selection. OOF predictions from LDA/Riemann/SVM on inner-val
            # become the meta training matrix; bases are then refit on full
            # outer-train to transform outer-test. Meta-learner never sees
            # outer-test at training time — standard stacking discipline.
            # Skipped under augmentation because inner-fold indices are global
            # and expanding augmented windows inside the stacking helper would
            # need its own plumbing; planned as a follow-up.
            if not aug_enabled:
                stacking_outer = _stacking_score(
                    X_feat_otrain,
                    X_feat_otest,
                    y_otrain,
                    y_otest,
                    X_mc_otrain,
                    X_mc_otest,
                    sfreq,
                    k_best,
                    inner_splits=inner_splits,
                    prefiltered_train=outer_train_cache,
                    prefiltered_test=outer_test_cache,
                )
                stacking_fold_scores.append(stacking_outer)

        score = float(np.mean(fold_scores)) if fold_scores else 0.5
        winner_counts = Counter(fold_winners)
        best_method = winner_counts.most_common(1)[0][0] if winner_counts else "lda"
        stacking_score = (
            float(np.mean(stacking_fold_scores)) if stacking_fold_scores else 0.5
        )
        return score, best_method, stacking_score


def train_nested_model_selection_cv(
    X_by_subject: list[tuple[np.ndarray, np.ndarray]],
    y_by_subject: list[np.ndarray],
    sfreq: float = 250.0,
    n_outer_folds: int = 10,
    n_inner_folds: int = 5,
    k_best: int = 10,
    trial_ptps_by_subject: list[np.ndarray] | None = None,
    split_strategy: SplitStrategy = "stratified_group",
    trial_groups_by_subject: list[np.ndarray] | None = None,
    trial_group_size: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "loky",
    max_blas_threads_per_worker: int = 1,
    enable_band_cache: bool = True,
    cache_scope: CacheScope = "subject",
    augmentation_by_subject: list[AugmentationContext | None] | None = None,
) -> tuple[list[float], float, float, list[str], list[float]]:
    """Nested model-selection CV that picks the best classifier per outer fold.

    Outer loop: StratifiedKFold(n_outer_folds) per subject.
    Inner loop: for each outer fold, evaluate all 4 classifiers via inner CV
    on the outer-train split, pick the best, retrain on full outer-train, and
    evaluate on outer-test.

    This gives an unbiased estimate of the "pick the best classifier" strategy
    because the classifier selection is done inside the CV loop.

    Stacking is reported as an additional, independent per-subject metric: for
    each outer fold we train a LogisticRegression meta-learner on OOF base
    classifier (LDA/Riemann/SVM) probabilities gathered across the inner CV
    folds, refit the bases on full outer-train, and evaluate on outer-test.
    Stacking does not participate in the inner-CV selection (no per-inner-fold
    score exists for it by construction); it's a parallel evaluation path.

    Args:
        X_by_subject: List of (handcrafted_features, multichannel_eeg) per subject.
        y_by_subject: List of label arrays per subject.
        sfreq: Sampling frequency in Hz.
        n_outer_folds: Number of outer CV folds.
        n_inner_folds: Number of inner CV folds for model selection.
        k_best: Number of features for FBCSP-based classifiers.
        trial_ptps_by_subject: Per-trial max PTP amplitudes per subject for
            in-fold artifact rejection at the outer level.  ``None`` disables
            per-fold rejection.
        split_strategy: ``'stratified'`` or ``'stratified_group'``.
        trial_groups_by_subject: Optional explicit CV groups per subject.
        trial_group_size: Group size when groups are auto-generated.
        random_state: Random seed for deterministic splits.

    Returns:
        Tuple of (per_subject_scores, mean, std, per_subject_best_methods,
        per_subject_stacking_scores).
        per_subject_best_methods is the most frequent inner-CV winner per subject.
        per_subject_stacking_scores is the averaged outer-fold stacking accuracy.

    Extra controls:
        n_jobs: Subject-level parallel jobs (1 disables parallelism).
        parallel_backend: joblib backend for subject-level parallelism.
        max_blas_threads_per_worker: BLAS/OpenMP thread cap per worker.
        enable_band_cache: Reuse precomputed per-band filtered trials.
        cache_scope: Cache precompute scope ('subject' or 'outer_fold').
    """
    payloads: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            np.ndarray | None,
            AugmentationContext | None,
        ]
    ] = []
    for subj_idx, ((X_features, X_multichannel), y) in enumerate(
        zip(X_by_subject, y_by_subject)
    ):
        groups = _resolve_subject_groups(
            y,
            provided_groups=trial_groups_by_subject,
            subject_index=subj_idx,
            split_strategy=split_strategy,
            trial_group_size=trial_group_size,
        )
        trial_ptps = trial_ptps_by_subject[subj_idx] if trial_ptps_by_subject else None
        aug = (
            augmentation_by_subject[subj_idx]
            if augmentation_by_subject is not None
            else None
        )
        payloads.append((X_features, X_multichannel, y, groups, trial_ptps, aug))

    if n_jobs == 1:
        per_subject_results = [
            _evaluate_subject_nested_model_selection(
                X_features,
                X_multichannel,
                y,
                sfreq=sfreq,
                n_outer_folds=n_outer_folds,
                n_inner_folds=n_inner_folds,
                k_best=k_best,
                trial_ptps=trial_ptps,
                split_strategy=split_strategy,
                groups=groups,
                random_state=random_state,
                enable_band_cache=enable_band_cache,
                cache_scope=cache_scope,
                max_blas_threads_per_worker=max_blas_threads_per_worker,
                augmentation=aug,
            )
            for X_features, X_multichannel, y, groups, trial_ptps, aug in payloads
        ]
    else:
        per_subject_results = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
            delayed(_evaluate_subject_nested_model_selection)(
                X_features,
                X_multichannel,
                y,
                sfreq=sfreq,
                n_outer_folds=n_outer_folds,
                n_inner_folds=n_inner_folds,
                k_best=k_best,
                trial_ptps=trial_ptps,
                split_strategy=split_strategy,
                groups=groups,
                random_state=random_state,
                enable_band_cache=enable_band_cache,
                cache_scope=cache_scope,
                max_blas_threads_per_worker=max_blas_threads_per_worker,
                augmentation=aug,
            )
            for X_features, X_multichannel, y, groups, trial_ptps, aug in payloads
        )

    scores = [result[0] for result in per_subject_results]
    best_methods = [result[1] for result in per_subject_results]
    stacking_scores = [result[2] for result in per_subject_results]

    mean_acc = float(np.mean(scores))
    std_acc = float(np.std(scores))
    return scores, mean_acc, std_acc, best_methods, stacking_scores


# ---------------------------------------------------------------------------
# Cross-session evaluation
# ---------------------------------------------------------------------------


def cross_session_evaluate(
    train_data: tuple[np.ndarray, np.ndarray],
    train_labels: np.ndarray,
    test_data: tuple[np.ndarray, np.ndarray],
    test_labels: np.ndarray,
    sfreq: float = 250.0,
    k_best: int = 10,
) -> dict[str, float]:
    """Train on one session, evaluate on another (no CV — independent test set).

    Args:
        train_data: (handcrafted_features, multichannel_eeg) from session A.
        train_labels: Labels from session A.
        test_data: (handcrafted_features, multichannel_eeg) from session B.
        test_labels: Labels from session B.
        sfreq: Sampling frequency in Hz.
        k_best: Number of features for FBCSP-based classifiers.

    Returns:
        Dict mapping classifier name to accuracy on session B.
    """
    X_feat_train, X_mc_train = train_data
    X_feat_test, X_mc_test = test_data

    return _evaluate_classifiers_batch(
        list(ALL_CLASSIFIERS),
        X_feat_train,
        X_feat_test,
        train_labels,
        test_labels,
        X_mc_train,
        X_mc_test,
        sfreq,
        k_best,
    )


def train_final_model_svm(
    X_features: np.ndarray,
    X_multichannel: np.ndarray,
    y: np.ndarray,
    sfreq: float = 250.0,
    k_best: int = 10,
) -> dict:
    """Train a deployable FBCSP + SVM model on all data for a single subject.

    Returns:
        Dict with keys: csp_models, selector, scaler, classifier, sfreq, k_best.
    """
    fbcsp_features, _, csp_models = _extract_fbcsp_features(
        X_multichannel, X_multichannel, y, sfreq
    )

    X_combined = np.hstack([fbcsp_features, X_features])

    k = min(k_best, X_combined.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X_combined, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    classifier = SVC(kernel="rbf", C=20.0, gamma="scale")
    classifier.fit(X_scaled, y)

    return {
        "csp_models": csp_models,
        "selector": selector,
        "scaler": scaler,
        "classifier": classifier,
        "sfreq": sfreq,
        "k_best": k,
    }


def predict_riemann(
    model: dict,
    X_multichannel: np.ndarray,
) -> np.ndarray:
    """Run inference with a saved Riemannian model.

    Args:
        model: Dict returned by train_final_model_riemann.
        X_multichannel: Multichannel EEG of shape (n_trials, n_channels, n_samples).

    Returns:
        Predicted class labels.
    """
    return model["pipeline"].predict(X_multichannel)
