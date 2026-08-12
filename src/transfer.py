"""Cross-subject Riemannian transfer learning with Euclidean Alignment."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.utils.mean import mean_covariance
from sklearn.linear_model import LogisticRegression

from src.alignment import euclidean_align

# pyriemann.transfer availability probe (pyriemann 0.10 on 2026-04-17):
# TLCenter and TLScale are available; TLStretch was renamed to TLScale in
# recent releases. We import defensively so a future removal/rename logs a
# warning and falls back to the manual EA path below.
try:
    from pyriemann.transfer import (  # type: ignore[attr-defined]
        TLCenter,
        TLScale,
        encode_domains,
    )

    _PYRIEMANN_TRANSFER_AVAILABLE = True
    _PYRIEMANN_TRANSFER_ERROR: str | None = None
except ImportError as _exc:  # pragma: no cover
    TLCenter = None  # type: ignore[assignment]
    TLScale = None  # type: ignore[assignment]
    encode_domains = None  # type: ignore[assignment]
    _PYRIEMANN_TRANSFER_AVAILABLE = False
    _PYRIEMANN_TRANSFER_ERROR = str(_exc)
from src.config import DEFAULT_SUBJECT_MERGE, MI_DATA_NEW_DIR
from src.data_loader import CHANNELS, get_recordings_by_subject, load_recording
from src.epochs import (
    compute_rejection_threshold,
    extract_left_right_epochs,
    reject_bad_epochs,
    task_epochs,
)
from src.features import extract_lateralization_features
from src.preprocess import preprocess_multichannel_eeg

logger = logging.getLogger(__name__)


def load_all_subjects(
    data_dirs: list[Path],
    sfreq: float = 250.0,
    subject_merge: dict[str, str] | None = None,
    min_trials: int = 10,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load and preprocess data from multiple directories, merging by subject.

    Args:
        data_dirs: List of directories to scan for recordings.
        sfreq: Sampling frequency.
        subject_merge: Optional mapping of subject_id -> canonical_id for merging
            (e.g., {"subject0100_2": "subject0100"}).
        min_trials: Minimum phase-3 trials per recording to include.

    Returns:
        Dict mapping subject_id to (X_features, X_multichannel, y) where
        X_features is (n_trials, 45) handcrafted features,
        X_multichannel is (n_trials, 8, n_samples), and y is (n_trials,).
    """
    if subject_merge is None:
        subject_merge = {}

    all_recordings: dict[str, list[Path]] = {}
    for data_dir in data_dirs:
        for sid, paths in get_recordings_by_subject(
            data_dir, min_trials=min_trials
        ).items():
            canonical = subject_merge.get(sid, sid)
            all_recordings.setdefault(canonical, []).extend(paths)

    subjects: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for sid, rec_paths in sorted(all_recordings.items()):
        all_left: dict[str, list] = {ch: [] for ch in CHANNELS}
        all_right: dict[str, list] = {ch: [] for ch in CHANNELS}

        for rec_path in rec_paths:
            data, events, rec_sfreq = load_recording(rec_path)
            if abs(rec_sfreq - sfreq) > 1e-6:
                continue
            preprocessed = preprocess_multichannel_eeg(data, sfreq)

            left_pairs: dict[str, list] = {}
            right_pairs: dict[str, list] = {}
            for ch_idx, ch_name in enumerate(CHANNELS):
                signal = preprocessed[ch_idx]
                left, right = extract_left_right_epochs(
                    signal,
                    events,
                    sfreq,
                    task_duration=3.0,
                    baseline_duration=1.0,
                    skip_duration=0.25,
                )
                left_pairs[ch_name] = left
                right_pairs[ch_name] = right

            threshold = compute_rejection_threshold(left_pairs, right_pairs)
            left_pairs, right_pairs, _, _ = reject_bad_epochs(
                left_pairs,
                right_pairs,
                threshold_uv=threshold,
            )

            for ch in CHANNELS:
                all_left[ch].extend(left_pairs[ch])
                all_right[ch].extend(right_pairs[ch])

        n_left = len(all_left[CHANNELS[0]])
        n_right = len(all_right[CHANNELS[0]])
        if n_left == 0 or n_right == 0:
            continue

        left_features = extract_lateralization_features(all_left, sfreq)
        right_features = extract_lateralization_features(all_right, sfreq)
        X_features = np.vstack([left_features, right_features])
        X_mc = np.vstack(
            [task_epochs(all_left, CHANNELS), task_epochs(all_right, CHANNELS)]
        )
        y = np.array([0] * n_left + [1] * n_right)
        subjects[sid] = (X_features, X_mc, y)

    return subjects


def _pyriemann_transfer_align_per_subject(
    subjects: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    cov_estimator: Covariances,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Align per-subject covariances with pyriemann.transfer.{TLCenter, TLScale}.

    TLCenter re-centers each subject's covariances to the identity (analogous to
    manual EA) while TLScale additionally normalizes dispersion. Domains are
    keyed by subject ID via ``encode_domains``-style (sample_idx, domain) pairs
    supplied to fit/transform.

    Callers must verify availability via ``_PYRIEMANN_TRANSFER_AVAILABLE``
    before invoking.
    """
    assert TLCenter is not None and TLScale is not None and encode_domains is not None
    aligned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sid in sorted(subjects.keys()):
        _, X_mc, y = subjects[sid]
        covs = cov_estimator.fit_transform(X_mc)
        domains = np.array([sid] * len(covs))
        # TLCenter/TLScale consume composite 'domain/label' strings produced by
        # encode_domains; fitting per-subject with that subject as the target
        # domain re-centers its covariances to identity.
        _, y_enc = encode_domains(covs, y, domains)
        center = TLCenter(target_domain=sid, metric="riemann")
        scale = TLScale(target_domain=sid, metric="riemann")
        covs_c = center.fit_transform(covs, y_enc)
        covs_cs = scale.fit_transform(covs_c, y_enc)
        aligned[sid] = (covs_cs, y)
    return aligned


def align_subjects(
    subjects: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    sfreq: float = 250.0,
    use_pyriemann_transfer: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute covariance matrices and apply Euclidean Alignment across subjects.

    EA re-centers each subject's covariance distribution to the identity matrix,
    removing inter-subject variability from electrode impedance and placement.

    Args:
        subjects: Dict mapping subject_id to (X_multichannel, y).
        sfreq: Sampling frequency (unused, kept for API consistency).
        use_pyriemann_transfer: If True and ``pyriemann.transfer`` is importable,
            use ``TLCenter`` + ``TLScale`` per subject instead of manual EA.
            Falls back to manual EA with a warning if the import probe failed.

    Returns:
        Tuple of:
        - aligned_covs: (total_trials, n_channels, n_channels) aligned SPD matrices
        - labels: (total_trials,) class labels
        - subject_ids: list of subject_id per trial (for LOSO indexing)
    """
    cov_estimator = Covariances(estimator="lwf")

    if use_pyriemann_transfer and not _PYRIEMANN_TRANSFER_AVAILABLE:
        logger.warning(
            "use_pyriemann_transfer=True but pyriemann.transfer import failed "
            "(%s); falling back to manual Euclidean Alignment.",
            _PYRIEMANN_TRANSFER_ERROR,
        )
        use_pyriemann_transfer = False

    if use_pyriemann_transfer:
        aligned_map = _pyriemann_transfer_align_per_subject(subjects, cov_estimator)
        all_covs = []
        all_labels = []
        all_subject_ids: list[str] = []
        for sid in sorted(subjects.keys()):
            covs_aligned, y = aligned_map[sid]
            all_covs.append(covs_aligned)
            all_labels.append(y)
            all_subject_ids.extend([sid] * len(y))
        return np.vstack(all_covs), np.concatenate(all_labels), all_subject_ids

    all_covs = []
    all_labels = []
    all_subject_ids = []

    for sid in sorted(subjects.keys()):
        _, X_mc, y = subjects[sid]
        covs = cov_estimator.fit_transform(X_mc)
        covs_aligned, _ = euclidean_align(covs)

        all_covs.append(covs_aligned)
        all_labels.append(y)
        all_subject_ids.extend([sid] * len(y))

    return np.vstack(all_covs), np.concatenate(all_labels), all_subject_ids


def shrink_covariances(
    covs: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Shrink covariance matrices toward a target along the Riemannian geodesic.

    Uses affine-invariant Riemannian geodesic interpolation:
    C_shrunk = geodesic(C, target, alpha)

    Args:
        covs: (n_trials, n, n) SPD matrices.
        target: (n, n) SPD matrix to shrink toward (e.g., group Riemannian mean).
        alpha: Shrinkage strength in [0, 1]. 0 = no shrinkage, 1 = full shrinkage.

    Returns:
        (n_trials, n, n) shrunk SPD matrices.
    """
    if alpha <= 0.0:
        return covs.copy()
    if alpha >= 1.0:
        return np.broadcast_to(target, covs.shape).copy()
    # geodesic_riemann(A, B, alpha): moves from A toward B by alpha
    return geodesic_riemann(covs, np.broadcast_to(target, covs.shape), alpha)


def regularized_within_subject_cv(
    subjects: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    sfreq: float = 250.0,
    shrinkage_k: float = 20.0,
    n_folds: int = 10,
    random_state: int = 42,
) -> dict[str, float]:
    """Within-subject CV with covariance shrinkage toward cross-subject group mean.

    For each subject:
    1. Estimate covariances with LWF, align via EA
    2. Shrink toward group Riemannian mean (alpha = k / (k + n_trials))
    3. Run stratified K-fold CV: TangentSpace + LogisticRegression

    Args:
        subjects: Dict mapping subject_id to (X_multichannel, y).
        sfreq: Sampling frequency.
        shrinkage_k: Controls shrinkage strength. Higher = more shrinkage.
            alpha = shrinkage_k / (shrinkage_k + n_trials).
        n_folds: Number of CV folds.
        random_state: Random seed.

    Returns:
        Dict mapping subject_id to regularized within-subject accuracy.
    """
    from sklearn.model_selection import StratifiedKFold

    cov_estimator = Covariances(estimator="lwf")

    aligned_per_subject: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_aligned_covs = []
    for sid in sorted(subjects.keys()):
        _, X_mc, y = subjects[sid]
        covs = cov_estimator.fit_transform(X_mc)
        covs_aligned, _ = euclidean_align(covs)
        aligned_per_subject[sid] = (covs_aligned, y)
        all_aligned_covs.append(covs_aligned)

    all_covs = np.vstack(all_aligned_covs)
    group_mean = mean_covariance(all_covs, metric="riemann")

    scores: dict[str, float] = {}
    for sid in sorted(subjects.keys()):
        covs, y = aligned_per_subject[sid]
        n_trials = len(y)
        alpha = shrinkage_k / (shrinkage_k + n_trials)

        # Shrink toward group mean
        covs_shrunk = shrink_covariances(covs, group_mean, alpha)

        # Stratified K-fold CV
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_scores = []
        for train_idx, test_idx in skf.split(covs_shrunk, y):
            ts = TangentSpace(metric="riemann")
            X_train = ts.fit_transform(covs_shrunk[train_idx])
            X_test = ts.transform(covs_shrunk[test_idx])

            clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000)
            clf.fit(X_train, y[train_idx])
            fold_scores.append(float(clf.score(X_test, y[test_idx])))

        scores[sid] = float(np.mean(fold_scores))

    return scores


def adaptive_augmented_cv(
    subjects: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    sfreq: float = 250.0,
    weakness_threshold: float = 0.55,
    n_folds: int = 10,
    random_state: int = 42,
) -> tuple[dict[str, float], dict[str, float]]:
    """Within-subject FBCSP+LDA CV, augmenting weak subjects with closest neighbor's data.

    For each subject:
    1. Run baseline FBCSP+LDA within-subject CV
    2. If baseline < weakness_threshold, re-run with closest subject's data added
       to training folds (CSP refitted on pooled data)
    3. Strong subjects keep their baseline score

    Closest subject is determined by Riemannian distance between raw covariance means.

    Args:
        subjects: Dict mapping subject_id to (X_multichannel, y).
        sfreq: Sampling frequency.
        weakness_threshold: Subjects below this accuracy get augmented.
        n_folds: Number of CV folds.
        random_state: Random seed.

    Returns:
        Tuple of (baseline_scores, adaptive_scores) dicts mapping subject_id to accuracy.
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    from src.train import (
        DEFAULT_K_CANDIDATES,
        _extract_fbcsp_features,
        _safe_k_values,
    )

    # Compute raw covariance distances for nearest-neighbor lookup
    cov_estimator = Covariances(estimator="lwf")
    raw_means: dict[str, np.ndarray] = {}
    for sid in sorted(subjects.keys()):
        _, X_mc, y = subjects[sid]
        covs = cov_estimator.fit_transform(X_mc)
        raw_means[sid] = mean_covariance(covs, metric="riemann")

    def _find_closest(target_sid: str) -> str:
        from pyriemann.utils.distance import distance_riemann

        dists = {
            s: float(distance_riemann(raw_means[target_sid], raw_means[s]))
            for s in subjects
            if s != target_sid
        }
        return min(dists, key=dists.get)

    def _run_fbcsp_cv(target_sid: str, aug_sid: str | None = None) -> float:
        _, X_mc_target, y_target = subjects[target_sid]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_scores: list[float] = []

        for tr_idx, te_idx in skf.split(X_mc_target, y_target):
            X_mc_tr = X_mc_target[tr_idx]
            y_tr = y_target[tr_idx]

            if aug_sid is not None:
                _, X_mc_aug, y_aug = subjects[aug_sid]
                X_mc_tr = np.vstack([X_mc_tr, X_mc_aug])
                y_tr = np.concatenate([y_tr, y_aug])

            fbcsp_tr, fbcsp_te, _ = _extract_fbcsp_features(
                X_mc_tr, X_mc_target[te_idx], y_tr, sfreq
            )
            if fbcsp_tr.shape[1] == 0:
                continue
            k_vals = _safe_k_values(
                fbcsp_tr.shape[1], len(y_tr), 10, DEFAULT_K_CANDIDATES
            )
            if not k_vals:
                continue

            sel = SelectKBest(f_classif, k=k_vals[-1])
            X_tr = sel.fit_transform(fbcsp_tr, y_tr)
            X_te = sel.transform(fbcsp_te)
            sc = StandardScaler()
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            lda.fit(sc.fit_transform(X_tr), y_tr)
            fold_scores.append(float(lda.score(sc.transform(X_te), y_target[te_idx])))

        return float(np.mean(fold_scores)) if fold_scores else 0.5

    baseline_scores: dict[str, float] = {}
    adaptive_scores: dict[str, float] = {}

    for sid in sorted(subjects.keys()):
        baseline = _run_fbcsp_cv(sid)
        baseline_scores[sid] = baseline

        if baseline < weakness_threshold:
            closest = _find_closest(sid)
            augmented = _run_fbcsp_cv(sid, aug_sid=closest)
            adaptive_scores[sid] = augmented
        else:
            adaptive_scores[sid] = baseline

    return baseline_scores, adaptive_scores


def loso_cv(
    subjects: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    sfreq: float = 250.0,
    fine_tune: bool = False,
) -> dict[str, float]:
    """Leave-one-subject-out CV with Euclidean Alignment.

    For each held-out subject:
    1. Align all subjects independently (EA per subject)
    2. Train TangentSpace + LR on aligned data from all other subjects
    3. Test on held-out subject's aligned data
    4. (If fine_tune) Re-center tangent space reference on held-out subject's mean

    Args:
        subjects: Dict mapping subject_id to (X_multichannel, y).
        sfreq: Sampling frequency.
        fine_tune: If True, re-center the tangent space reference point
            to the held-out subject's Riemannian mean before testing.

    Returns:
        Dict mapping subject_id to LOSO accuracy.
    """
    cov_estimator = Covariances(estimator="lwf")

    aligned_per_subject: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sid in sorted(subjects.keys()):
        _, X_mc, y = subjects[sid]
        covs = cov_estimator.fit_transform(X_mc)
        covs_aligned, _ = euclidean_align(covs)
        aligned_per_subject[sid] = (covs_aligned, y)

    subject_ids = sorted(subjects.keys())
    scores: dict[str, float] = {}

    for held_out in subject_ids:
        train_covs = []
        train_labels = []
        for sid in subject_ids:
            if sid == held_out:
                continue
            covs, y = aligned_per_subject[sid]
            train_covs.append(covs)
            train_labels.append(y)

        X_train = np.vstack(train_covs)
        y_train = np.concatenate(train_labels)

        X_test, y_test = aligned_per_subject[held_out]

        if fine_tune:
            # Re-center tangent space on held-out subject's mean
            test_ref = mean_covariance(X_test, metric="riemann")
            # Train projection uses pooled training data reference
            train_ts = TangentSpace(metric="riemann")
            X_train_ts = train_ts.fit_transform(X_train)
            # Test projection uses held-out subject's own reference
            test_ts = TangentSpace(metric="riemann")
            test_ts.fit(X_train)  # fit to get consistent dimensionality
            test_ts.reference_ = test_ref  # override reference point
            X_test_ts = test_ts.transform(X_test)
        else:
            pipe_ts = TangentSpace(metric="riemann")
            X_train_ts = pipe_ts.fit_transform(X_train)
            X_test_ts = pipe_ts.transform(X_test)

        clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000)
        clf.fit(X_train_ts, y_train)
        scores[held_out] = float(clf.score(X_test_ts, y_test))

    return scores


def run_transfer_evaluation(
    data_dirs: list[Path] | None = None,
    sfreq: float = 250.0,
    subject_merge: dict[str, str] | None = None,
    within_subject_scores: dict[str, float] | None = None,
) -> dict:
    """Run cross-subject transfer learning evaluation and print comparison.

    Args:
        data_dirs: Directories to scan. Defaults to data/unicorn-data + data/MI_DATA_NEW.
        sfreq: Sampling frequency.
        subject_merge: Subject ID merging map.
        within_subject_scores: Optional dict of subject_id -> nested CV accuracy
            for comparison table.

    Returns:
        Results dict with LOSO and LOSO+FT scores.
    """
    if data_dirs is None:
        data_dirs = [Path("data/unicorn-data"), MI_DATA_NEW_DIR]
    if subject_merge is None:
        subject_merge = DEFAULT_SUBJECT_MERGE

    print("=" * 68)
    print("Cross-Subject Transfer Learning (Euclidean Alignment + LOSO)")
    print("=" * 68)

    print("\nLoading subjects from all data directories...")
    subjects = load_all_subjects(data_dirs, sfreq=sfreq, subject_merge=subject_merge)
    print(f"Loaded {len(subjects)} subjects: {', '.join(sorted(subjects.keys()))}")
    for sid in sorted(subjects.keys()):
        _, X_mc, y = subjects[sid]
        n_left = int(np.sum(y == 0))
        n_right = int(np.sum(y == 1))
        print(f"  {sid}: {len(y)} trials ({n_left}L/{n_right}R)")

    print("\nRunning adaptive augmented FBCSP+LDA CV...")
    fbcsp_base, fbcsp_adaptive = adaptive_augmented_cv(subjects, sfreq=sfreq)
    fbcsp_base_mean = float(np.mean(list(fbcsp_base.values())))
    fbcsp_adp_mean = float(np.mean(list(fbcsp_adaptive.values())))

    print("Running LOSO CV...")
    loso_scores = loso_cv(subjects, sfreq=sfreq, fine_tune=False)
    loso_mean = float(np.mean(list(loso_scores.values())))

    # Print comparison table
    header = f"\n{'Subject':<16}{'FBCSP':>8}{'Adaptive':>10}{'Delta':>8}"
    if within_subject_scores:
        header += f"  {'Nested':>8}"
    header += f"{'LOSO':>8}"
    print(header)
    print("-" * len(header))

    for sid in sorted(fbcsp_base.keys()):
        base = fbcsp_base[sid]
        adp = fbcsp_adaptive[sid]
        delta = adp - base
        sign = "+" if delta >= 0 else ""
        weak = " *" if base < 0.55 else ""
        row = f"  {sid:<14}{base:>7.1%}{adp:>9.1%}{sign}{delta:>6.1%}"
        if within_subject_scores and sid in within_subject_scores:
            row += f"  {within_subject_scores[sid]:>7.1%}"
        elif within_subject_scores:
            row += f"  {'n/a':>8}"
        row += f"{loso_scores[sid]:>7.1%}{weak}"
        print(row)

    if within_subject_scores:
        ws_mean = float(np.mean(list(within_subject_scores.values())))
        print(f"\n  Nested (best-of-4) mean:  {ws_mean:.1%}")
    print(f"  FBCSP+LDA baseline mean:  {fbcsp_base_mean:.1%}")
    print(f"  Adaptive augmented mean:  {fbcsp_adp_mean:.1%}")
    delta_mean = fbcsp_adp_mean - fbcsp_base_mean
    sign = "+" if delta_mean >= 0 else ""
    print(f"  Augmentation delta:       {sign}{delta_mean:.1%}")
    print(f"  LOSO mean:                {loso_mean:.1%}")
    print("  (* = weak subject, augmented with closest neighbor)")

    return {
        "fbcsp_base_scores": fbcsp_base,
        "fbcsp_base_mean": fbcsp_base_mean,
        "fbcsp_adaptive_scores": fbcsp_adaptive,
        "fbcsp_adaptive_mean": fbcsp_adp_mean,
        "loso_scores": loso_scores,
        "loso_mean": loso_mean,
        "n_subjects": len(subjects),
    }


def select_nearest_donor(
    target_id: str,
    target_mc: np.ndarray,
    donor_subjects: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[str, float]:
    """Select the nearest donor by Riemannian distance between covariance means.

    Args:
        target_id: Subject ID to exclude from donors.
        target_mc: Target multichannel EEG (n_trials, n_channels, n_samples).
        donor_subjects: Dict mapping subject_id -> (features, multichannel, labels).

    Returns:
        (best_donor_id, distance).
    """
    from pyriemann.estimation import Covariances
    from pyriemann.utils.distance import distance_riemann
    from pyriemann.utils.mean import mean_covariance

    cov_est = Covariances(estimator="lwf")
    target_mean = mean_covariance(cov_est.fit_transform(target_mc), metric="riemann")

    best_donor = None
    best_dist = float("inf")

    for sid, (_, mc, _) in donor_subjects.items():
        if sid == target_id:
            continue
        donor_mean = mean_covariance(cov_est.fit_transform(mc), metric="riemann")
        dist = float(distance_riemann(target_mean, donor_mean))
        if dist < best_dist:
            best_dist = dist
            best_donor = sid

    if best_donor is None:
        raise ValueError("No donors available (all excluded)")

    return best_donor, best_dist


if __name__ == "__main__":
    results = run_transfer_evaluation()
