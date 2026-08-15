"""Tests for cross-subject Riemannian transfer learning."""

import numpy as np
import pandas as pd

from src.transfer import (
    align_subjects,
    gated_ea_augmented_nested_cv,
    load_all_subjects,
    loso_cv,
    regularized_within_subject_cv,
    run_transfer_evaluation,
    shrink_covariances,
)


def _make_recording_csv(
    path,
    sfreq: float = 250.0,
    n_left: int = 50,
    n_right: int = 50,
    seed: int = 42,
) -> None:
    """Create a synthetic EEG recording CSV with phase-3 trials.

    Generates exactly n_left + n_right phase-3 trials so the file passes
    ``get_complete_recordings``'s 100-trial filter when n_left=50, n_right=50.
    """
    n_trials = n_left + n_right
    n_samples_per_trial = int(sfreq * 5)
    # Enough padding before first event and after last event
    n_total = 2000 + n_samples_per_trial * n_trials

    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_total, 8)) * 50

    stim = np.zeros(n_total)
    for i in range(n_trials):
        event_idx = 1000 + i * n_samples_per_trial
        # stim 31 = phase 3, movement 1 (left); 32 = phase 3, movement 2 (right)
        stim[event_idx] = 31 if i < n_left else 32

    df = pd.DataFrame(data, columns=["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"])
    df["stim"] = stim
    df["timestamps"] = np.arange(n_total) / sfreq
    df = df[["timestamps", "Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8", "stim"]]
    df.to_csv(path, index=False)


def test_load_all_subjects_returns_dict(tmp_path):
    """load_all_subjects returns a dict mapping subject_id to (X_mc, y) tuples."""
    sfreq = 250.0

    for i, subj_name in enumerate(["subject0001", "subject0002"]):
        subj_dir = tmp_path / subj_name / "session001"
        subj_dir.mkdir(parents=True)
        _make_recording_csv(subj_dir / "recording_test.csv", sfreq=sfreq, seed=10 + i)

    result = load_all_subjects([tmp_path], sfreq=sfreq)

    assert isinstance(result, dict)
    assert len(result) == 2
    for sid, (X_feat, X_mc, y) in result.items():
        assert X_feat.ndim == 2
        assert X_feat.shape[0] == len(y)
        assert X_mc.ndim == 3
        assert X_mc.shape[1] == 8
        assert len(y) == X_mc.shape[0]
        assert set(np.unique(y)) == {0, 1}


def test_load_all_subjects_merges_sessions(tmp_path):
    """Sessions for the same subject are pooled into one entry."""
    sfreq = 250.0

    subj_dir_s1 = tmp_path / "subject0001" / "session001"
    subj_dir_s2 = tmp_path / "subject0001" / "session002"
    subj_dir_s1.mkdir(parents=True)
    subj_dir_s2.mkdir(parents=True)

    _make_recording_csv(subj_dir_s1 / "recording_test.csv", sfreq=sfreq, seed=42)
    _make_recording_csv(subj_dir_s2 / "recording_test.csv", sfreq=sfreq, seed=99)

    result = load_all_subjects([tmp_path], sfreq=sfreq)

    assert "subject0001" in result
    X_feat, X_mc, y = result["subject0001"]
    assert X_mc.shape[0] == len(y)
    assert X_feat.shape[0] == len(y)
    # Two sessions x 100 trials each; most should survive artifact rejection
    assert X_mc.shape[0] >= 150


def test_align_subjects_centers_covariances():
    """After EA, each subject's mean covariance should be close to identity."""
    rng = np.random.default_rng(42)
    n_channels = 8

    # Create 3 subjects with different covariance structures
    subjects = {}
    for i in range(3):
        A = rng.standard_normal((n_channels, n_channels))
        cov_mean = A @ A.T + np.eye(n_channels)
        X = rng.standard_normal((20, n_channels, 375))
        X = np.einsum("ij,njt->nit", np.linalg.cholesky(cov_mean), X)
        y = np.array([0] * 10 + [1] * 10)
        subjects[f"subj{i:02d}"] = (np.zeros((len(y), 45)), X, y)

    aligned_covs, aligned_labels, subject_ids = align_subjects(subjects, sfreq=250.0)

    assert aligned_covs.ndim == 3  # (total_trials, 8, 8)
    assert aligned_covs.shape[0] == 60  # 3 subjects * 20 trials
    assert aligned_covs.shape[1] == n_channels
    assert aligned_covs.shape[2] == n_channels
    assert len(aligned_labels) == 60
    assert len(subject_ids) == 60

    # Check each subject's aligned covariances are centered near identity
    for i in range(3):
        mask = np.array(subject_ids) == f"subj{i:02d}"
        subj_covs = aligned_covs[mask]
        mean_cov = np.mean(subj_covs, axis=0)
        np.testing.assert_allclose(mean_cov, np.eye(n_channels), atol=0.5)


def test_loso_cv_returns_per_subject_scores():
    """LOSO CV returns a score for each subject."""
    rng = np.random.default_rng(42)
    n_channels = 8

    subjects = {}
    for i in range(4):
        X = rng.standard_normal((20, n_channels, 375))
        y = np.array([0] * 10 + [1] * 10)
        subjects[f"subj{i:02d}"] = (np.zeros((len(y), 45)), X, y)

    scores = loso_cv(subjects, sfreq=250.0)

    assert isinstance(scores, dict)
    assert len(scores) == 4
    for sid, score in scores.items():
        assert 0.0 <= score <= 1.0
        assert sid.startswith("subj")


def test_loso_cv_on_random_data_near_chance():
    """On random data, LOSO accuracy should be near chance (~50%)."""
    rng = np.random.default_rng(99)
    n_channels = 8

    subjects = {}
    for i in range(5):
        X = rng.standard_normal((30, n_channels, 375))
        y = np.array([0] * 15 + [1] * 15)
        subjects[f"subj{i:02d}"] = (np.zeros((len(y), 45)), X, y)

    scores = loso_cv(subjects, sfreq=250.0)

    mean_acc = np.mean(list(scores.values()))
    assert 0.3 <= mean_acc <= 0.7


def test_loso_cv_fine_tune_returns_scores():
    """LOSO CV with fine-tuning returns a score for each subject."""
    rng = np.random.default_rng(42)
    n_channels = 8

    subjects = {}
    for i in range(4):
        X = rng.standard_normal((20, n_channels, 375))
        y = np.array([0] * 10 + [1] * 10)
        subjects[f"subj{i:02d}"] = (np.zeros((len(y), 45)), X, y)

    scores = loso_cv(subjects, sfreq=250.0, fine_tune=True)

    assert isinstance(scores, dict)
    assert len(scores) == 4
    for sid, score in scores.items():
        assert 0.0 <= score <= 1.0


def test_run_transfer_evaluation_returns_results(tmp_path):
    """run_transfer_evaluation returns a results dict with expected keys."""
    for subj_name in ["subject0001", "subject0002", "subject0003"]:
        subj_dir = tmp_path / subj_name / "session001"
        subj_dir.mkdir(parents=True)
        _make_recording_csv(
            subj_dir / "recording_test.csv",
            sfreq=250.0,
            seed=hash(subj_name) % 2**31,
        )

    results = run_transfer_evaluation(data_dirs=[tmp_path])

    assert "fbcsp_base_scores" in results
    assert "fbcsp_adaptive_scores" in results
    assert "loso_scores" in results
    assert "loso_mean" in results
    assert len(results["loso_scores"]) == 3


def test_shrink_covariances_alpha_zero():
    """alpha=0 returns original covariances unchanged."""
    rng = np.random.default_rng(42)
    n = 8
    covs = np.array([rng.standard_normal((n, n)) for _ in range(5)])
    covs = covs @ covs.transpose(0, 2, 1) + np.eye(n)  # make SPD
    target = np.eye(n) * 2

    result = shrink_covariances(covs, target, alpha=0.0)
    np.testing.assert_allclose(result, covs, atol=1e-10)


def test_shrink_covariances_alpha_one():
    """alpha=1 returns the target for every trial."""
    rng = np.random.default_rng(42)
    n = 8
    covs = np.array([rng.standard_normal((n, n)) for _ in range(5)])
    covs = covs @ covs.transpose(0, 2, 1) + np.eye(n)
    target = np.eye(n) * 2

    result = shrink_covariances(covs, target, alpha=1.0)
    for i in range(5):
        np.testing.assert_allclose(result[i], target, atol=1e-10)


def test_shrink_covariances_intermediate():
    """0 < alpha < 1 moves covariances toward target."""
    rng = np.random.default_rng(42)
    n = 8
    covs = np.array([rng.standard_normal((n, n)) for _ in range(5)])
    covs = covs @ covs.transpose(0, 2, 1) + np.eye(n)
    target = np.eye(n)

    result = shrink_covariances(covs, target, alpha=0.5)

    # Each shrunk covariance should be closer to target than the original
    from pyriemann.utils.distance import distance_riemann

    for i in range(5):
        d_original = distance_riemann(covs[i], target)
        d_shrunk = distance_riemann(result[i], target)
        assert d_shrunk < d_original


def test_gated_ea_nested_cv_random_data_near_chance():
    """Inner-CV gate plus EA must not leak: random EEG stays near chance."""
    n_channels, n_samples, n_feat = 8, 200, 10
    n_trials = 40
    y = np.array([0] * 20 + [1] * 20)

    def _subject(seed: int):
        r = np.random.default_rng(seed)
        feat = r.standard_normal((n_trials, n_feat))
        mc = r.standard_normal((n_trials, n_channels, n_samples))
        return feat, mc, y.copy()

    target_feat, target_mc, target_y = _subject(1)
    donors = {
        "donor_a": _subject(2),
        "donor_b": _subject(3),
        "donor_c": _subject(4),
    }
    from pyriemann.estimation import Covariances
    from pyriemann.utils.mean import mean_covariance

    cov_est = Covariances(estimator="lwf")
    donor_means = {
        sid: mean_covariance(cov_est.fit_transform(mc), metric="riemann")
        for sid, (_, mc, _) in donors.items()
    }
    donors["target"] = (target_feat, target_mc, target_y)
    donor_means["target"] = mean_covariance(
        cov_est.fit_transform(target_mc), metric="riemann"
    )

    acc, strategy, _ = gated_ea_augmented_nested_cv(
        target_feat,
        target_mc,
        target_y,
        donor_subjects=donors,
        donor_means=donor_means,
        target_sid="target",
        sfreq=250.0,
        n_outer_folds=4,
        n_inner_folds=3,
        k_best=5,
        trial_ptps=None,
        split_strategy="stratified",
        groups=None,
        random_state=42,
        signal_donor_ids={"donor_a", "donor_b"},
        pool_k=2,
    )
    assert 0.0 <= acc <= 1.0
    assert acc < 0.75
    assert strategy in {"none", "nearest_ea", "pool_ea"}


def test_regularized_within_subject_cv_returns_scores():
    """Regularized within-subject CV returns per-subject scores."""
    rng = np.random.default_rng(42)
    n_channels = 8

    subjects = {}
    for i in range(3):
        X = rng.standard_normal((30, n_channels, 375))
        y = np.array([0] * 15 + [1] * 15)
        subjects[f"subj{i:02d}"] = (np.zeros((len(y), 45)), X, y)

    scores = regularized_within_subject_cv(subjects, sfreq=250.0, n_folds=5)

    assert isinstance(scores, dict)
    assert len(scores) == 3
    for sid, score in scores.items():
        assert 0.0 <= score <= 1.0
