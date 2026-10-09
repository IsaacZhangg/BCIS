"""Tests for EA-gated donor augmentation."""

import numpy as np

from src.transfer import gated_ea_augmented_nested_cv


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
