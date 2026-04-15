"""Tests for FBCSP + LDA, FBCSP + SVM, and Riemannian training pipelines."""

import numpy as np

from src.band_cache import precompute_bandpassed
from src.epochs import compute_trial_max_ptp
from src.rejection import reject_in_fold
from src.train import (
    _evaluate_classifiers_batch,
    cross_session_evaluate,
    predict,
    predict_riemann,
    train_final_model,
    train_final_model_riemann,
    train_final_model_svm,
    train_nested_model_selection_cv,
    train_within_subject_cv_all_models,
    train_within_subject_cv,
    train_within_subject_cv_ensemble,
    train_within_subject_cv_riemann,
    train_within_subject_cv_svm,
)

N_EPOCHS = 40
N_FEATURES = 10
N_CHANNELS = 8
N_SAMPLES = 375  # 1.5s at 250Hz
LABELS = np.array([0] * 20 + [1] * 20)


def _make_subject_data(rng):
    X_features = rng.standard_normal((N_EPOCHS, N_FEATURES))
    X_multichannel = rng.standard_normal((N_EPOCHS, N_CHANNELS, N_SAMPLES))
    return X_features, X_multichannel


def test_train_within_subject_cv_returns_scores():
    """Within-subject CV returns per-subject scores with correct shape and range."""
    n_subjects = 3
    X_by_subject = [
        _make_subject_data(np.random.default_rng(42 + i)) for i in range(n_subjects)
    ]
    y_by_subject = [LABELS for _ in range(n_subjects)]

    scores, mean_acc, std_acc = train_within_subject_cv(
        X_by_subject, y_by_subject, n_folds=5
    )

    assert len(scores) == n_subjects
    assert 0 <= mean_acc <= 1
    assert std_acc >= 0
    for s in scores:
        assert 0 <= s <= 1


def test_fbcsp_no_leakage_on_random_data():
    """On pure random data, FBCSP + LDA accuracy should be near chance (~50%)."""
    n_subjects = 2
    rng = np.random.default_rng(123)
    X_by_subject = [_make_subject_data(rng) for _ in range(n_subjects)]
    y_by_subject = [LABELS for _ in range(n_subjects)]

    _, mean_acc, _ = train_within_subject_cv(X_by_subject, y_by_subject, n_folds=5)

    assert mean_acc < 0.70, (
        f"Random data accuracy {mean_acc:.1%} is suspiciously high — possible data leakage"
    )


def test_train_final_model_returns_complete_model():
    """train_final_model returns a dict with all required keys."""
    rng = np.random.default_rng(42)
    X_features, X_multichannel = _make_subject_data(rng)

    model = train_final_model(X_features, X_multichannel, LABELS)

    assert set(model.keys()) == {
        "csp_models",
        "selector",
        "scaler",
        "classifier",
        "sfreq",
        "k_best",
    }
    assert model["sfreq"] == 250.0
    assert isinstance(model["k_best"], int)
    assert model["k_best"] > 0


def test_predict_with_saved_model():
    """Round-trip: train a model, then predict with it."""
    rng = np.random.default_rng(42)
    X_features, X_multichannel = _make_subject_data(rng)

    model = train_final_model(X_features, X_multichannel, LABELS)
    preds = predict(model, X_features, X_multichannel)

    assert preds.shape == (N_EPOCHS,)
    assert set(np.unique(preds)).issubset({0, 1})


# ---------- Riemannian classifier tests ----------


def test_riemann_cv_returns_scores():
    """Riemannian CV returns per-subject scores with correct shape and range."""
    n_subjects = 3
    rng = np.random.default_rng(42)
    X_by_subject = [
        rng.standard_normal((N_EPOCHS, N_CHANNELS, N_SAMPLES))
        for _ in range(n_subjects)
    ]
    y_by_subject = [LABELS for _ in range(n_subjects)]

    scores, mean_acc, std_acc = train_within_subject_cv_riemann(
        X_by_subject, y_by_subject, n_folds=5
    )

    assert len(scores) == n_subjects
    assert 0 <= mean_acc <= 1
    assert std_acc >= 0
    for s in scores:
        assert 0 <= s <= 1


def test_riemann_no_leakage_on_random_data():
    """On pure random data, Riemannian accuracy should be near chance (~50%)."""
    n_subjects = 2
    rng = np.random.default_rng(123)
    X_by_subject = [
        rng.standard_normal((N_EPOCHS, N_CHANNELS, N_SAMPLES))
        for _ in range(n_subjects)
    ]
    y_by_subject = [LABELS for _ in range(n_subjects)]

    _, mean_acc, _ = train_within_subject_cv_riemann(
        X_by_subject, y_by_subject, n_folds=5
    )

    assert mean_acc < 0.70, (
        f"Random data accuracy {mean_acc:.1%} is suspiciously high — possible data leakage"
    )


def test_riemann_final_model_and_predict():
    """Round-trip: train a Riemannian model, then predict with it."""
    rng = np.random.default_rng(42)
    X_multichannel = rng.standard_normal((N_EPOCHS, N_CHANNELS, N_SAMPLES))

    model = train_final_model_riemann(X_multichannel, LABELS)
    assert "pipeline" in model
    assert "sfreq" in model
    assert model["sfreq"] == 250.0

    preds = predict_riemann(model, X_multichannel)
    assert preds.shape == (N_EPOCHS,)
    assert set(np.unique(preds)).issubset({0, 1})


# ---------- FBCSP + SVM classifier tests ----------


def test_svm_cv_returns_scores():
    """SVM CV returns per-subject scores with correct shape and range."""
    n_subjects = 3
    X_by_subject = [
        _make_subject_data(np.random.default_rng(42 + i)) for i in range(n_subjects)
    ]
    y_by_subject = [LABELS for _ in range(n_subjects)]

    scores, mean_acc, std_acc = train_within_subject_cv_svm(
        X_by_subject, y_by_subject, n_folds=5
    )

    assert len(scores) == n_subjects
    assert 0 <= mean_acc <= 1
    assert std_acc >= 0
    for s in scores:
        assert 0 <= s <= 1


def test_svm_no_leakage_on_random_data():
    """On pure random data, FBCSP + SVM accuracy should be near chance (~50%)."""
    n_subjects = 2
    rng = np.random.default_rng(123)
    X_by_subject = [_make_subject_data(rng) for _ in range(n_subjects)]
    y_by_subject = [LABELS for _ in range(n_subjects)]

    _, mean_acc, _ = train_within_subject_cv_svm(X_by_subject, y_by_subject, n_folds=5)

    assert mean_acc < 0.70, (
        f"Random data accuracy {mean_acc:.1%} is suspiciously high — possible data leakage"
    )


def test_svm_final_model_and_predict():
    """Round-trip: train an SVM model, then predict with it."""
    rng = np.random.default_rng(42)
    X_features, X_multichannel = _make_subject_data(rng)

    model = train_final_model_svm(X_features, X_multichannel, LABELS)

    assert set(model.keys()) == {
        "csp_models",
        "selector",
        "scaler",
        "classifier",
        "sfreq",
        "k_best",
    }
    assert model["sfreq"] == 250.0
    assert isinstance(model["k_best"], int)
    assert model["k_best"] > 0

    # predict() works with SVM models since SVC has .predict()
    preds = predict(model, X_features, X_multichannel)
    assert preds.shape == (N_EPOCHS,)
    assert set(np.unique(preds)).issubset({0, 1})


# ---------- Ensemble classifier tests ----------


def test_ensemble_cv_returns_scores():
    """Ensemble CV returns per-subject scores with correct shape and range."""
    n_subjects = 3
    X_by_subject = [
        _make_subject_data(np.random.default_rng(42 + i)) for i in range(n_subjects)
    ]
    y_by_subject = [LABELS for _ in range(n_subjects)]

    scores, mean_acc, std_acc = train_within_subject_cv_ensemble(
        X_by_subject, y_by_subject, n_folds=5
    )

    assert len(scores) == n_subjects
    assert 0 <= mean_acc <= 1
    assert std_acc >= 0
    for s in scores:
        assert 0 <= s <= 1


def test_ensemble_no_leakage_on_random_data():
    """On pure random data, ensemble accuracy should be near chance (~50%)."""
    n_subjects = 2
    rng = np.random.default_rng(123)
    X_by_subject = [_make_subject_data(rng) for _ in range(n_subjects)]
    y_by_subject = [LABELS for _ in range(n_subjects)]

    _, mean_acc, _ = train_within_subject_cv_ensemble(
        X_by_subject, y_by_subject, n_folds=5
    )

    assert mean_acc < 0.70, (
        f"Random data accuracy {mean_acc:.1%} is suspiciously high — possible data leakage"
    )


def test_all_models_cv_matches_individual_entrypoints():
    """Shared all-model CV path matches the legacy per-model entrypoints."""
    X_by_subject = [_make_subject_data(np.random.default_rng(42))]
    y_by_subject = [LABELS]
    trial_ptps = [compute_trial_max_ptp(X_by_subject[0][1])]
    cv_kwargs = dict(
        n_folds=5,
        split_strategy="stratified_group",
        trial_group_size=5,
        random_state=7,
        trial_ptps_by_subject=trial_ptps,
    )

    all_results = train_within_subject_cv_all_models(
        X_by_subject, y_by_subject, **cv_kwargs
    )

    lda_scores, lda_mean, lda_std = train_within_subject_cv(
        X_by_subject, y_by_subject, **cv_kwargs
    )
    np.testing.assert_allclose(all_results["lda"][0], lda_scores)
    np.testing.assert_allclose(all_results["lda"][1:], (lda_mean, lda_std))

    riemann_scores, riemann_mean, riemann_std = train_within_subject_cv_riemann(
        [X_mc for _, X_mc in X_by_subject], y_by_subject, **cv_kwargs
    )
    np.testing.assert_allclose(all_results["riemann"][0], riemann_scores)
    np.testing.assert_allclose(all_results["riemann"][1:], (riemann_mean, riemann_std))

    svm_scores, svm_mean, svm_std = train_within_subject_cv_svm(
        X_by_subject, y_by_subject, **cv_kwargs
    )
    np.testing.assert_allclose(all_results["svm"][0], svm_scores)
    np.testing.assert_allclose(all_results["svm"][1:], (svm_mean, svm_std))

    ens_scores, ens_mean, ens_std = train_within_subject_cv_ensemble(
        X_by_subject, y_by_subject, **cv_kwargs
    )
    np.testing.assert_allclose(all_results["ensemble"][0], ens_scores)
    np.testing.assert_allclose(all_results["ensemble"][1:], (ens_mean, ens_std))


def test_batched_classifier_eval_matches_single_requests():
    """Batch split evaluation returns the same score as per-classifier requests."""
    rng = np.random.default_rng(123)
    X_features, X_multichannel = _make_subject_data(rng)
    train_idx = np.arange(0, 30)
    test_idx = np.arange(30, 40)

    batch_scores = _evaluate_classifiers_batch(
        ["lda", "riemann", "svm", "ensemble"],
        X_features[train_idx],
        X_features[test_idx],
        LABELS[train_idx],
        LABELS[test_idx],
        X_multichannel[train_idx],
        X_multichannel[test_idx],
        sfreq=250.0,
        k_best=10,
    )

    for name in ("lda", "riemann", "svm", "ensemble"):
        single = _evaluate_classifiers_batch(
            [name],
            X_features[train_idx],
            X_features[test_idx],
            LABELS[train_idx],
            LABELS[test_idx],
            X_multichannel[train_idx],
            X_multichannel[test_idx],
            sfreq=250.0,
            k_best=10,
        )[name]
        np.testing.assert_allclose(batch_scores[name], single)


def test_batched_classifier_eval_prefiltered_matches_uncached():
    """Pre-filtered evaluation path matches uncached evaluation."""
    rng = np.random.default_rng(123)
    X_features, X_multichannel = _make_subject_data(rng)
    train_idx = np.arange(0, 30)
    test_idx = np.arange(30, 40)

    uncached_scores = _evaluate_classifiers_batch(
        ["lda", "riemann", "svm", "ensemble"],
        X_features[train_idx],
        X_features[test_idx],
        LABELS[train_idx],
        LABELS[test_idx],
        X_multichannel[train_idx],
        X_multichannel[test_idx],
        sfreq=250.0,
        k_best=10,
    )

    prefiltered_scores = _evaluate_classifiers_batch(
        ["lda", "riemann", "svm", "ensemble"],
        X_features[train_idx],
        X_features[test_idx],
        LABELS[train_idx],
        LABELS[test_idx],
        X_multichannel[train_idx],
        X_multichannel[test_idx],
        sfreq=250.0,
        k_best=10,
        prefiltered_train=precompute_bandpassed(X_multichannel[train_idx], 250.0),
        prefiltered_test=precompute_bandpassed(X_multichannel[test_idx], 250.0),
    )

    for name in ("lda", "riemann", "svm", "ensemble"):
        np.testing.assert_allclose(uncached_scores[name], prefiltered_scores[name])


def test_all_models_cached_parallel_path_keeps_statistical_parity():
    """Cached+parallel all-model CV should stay statistically aligned."""
    n_subjects = 2
    X_by_subject = [
        _make_subject_data(np.random.default_rng(100 + i)) for i in range(n_subjects)
    ]
    y_by_subject = [LABELS for _ in range(n_subjects)]
    trial_ptps = [compute_trial_max_ptp(X_mc) for _, X_mc in X_by_subject]
    shared_kwargs = dict(
        n_folds=3,
        trial_ptps_by_subject=trial_ptps,
        split_strategy="stratified_group",
        trial_group_size=5,
        random_state=21,
        k_best=10,
    )

    baseline = train_within_subject_cv_all_models(
        X_by_subject, y_by_subject, n_jobs=1, enable_band_cache=False, **shared_kwargs
    )
    optimized = train_within_subject_cv_all_models(
        X_by_subject,
        y_by_subject,
        n_jobs=2,
        parallel_backend="threading",
        max_blas_threads_per_worker=1,
        enable_band_cache=True,
        **shared_kwargs,
    )

    for name in ("lda", "riemann", "svm", "ensemble"):
        b_scores, b_mean, b_std = baseline[name]
        o_scores, o_mean, o_std = optimized[name]
        np.testing.assert_allclose(b_scores, o_scores, atol=5e-3, rtol=0)
        np.testing.assert_allclose(b_mean, o_mean, atol=2e-3, rtol=0)
        np.testing.assert_allclose(b_std, o_std, atol=2e-3, rtol=0)


# ---------- Nested model selection CV tests ----------


def test_nested_model_selection_cv_returns_scores():
    """Nested CV returns per-subject scores, mean, std, and method names."""
    n_subjects = 2
    X_by_subject = [
        _make_subject_data(np.random.default_rng(42 + i)) for i in range(n_subjects)
    ]
    y_by_subject = [LABELS for _ in range(n_subjects)]

    scores, mean_acc, std_acc, methods = train_nested_model_selection_cv(
        X_by_subject, y_by_subject, n_outer_folds=5, n_inner_folds=3
    )

    assert len(scores) == n_subjects
    assert 0 <= mean_acc <= 1
    assert std_acc >= 0
    for s in scores:
        assert 0 <= s <= 1
    assert len(methods) == n_subjects
    for m in methods:
        assert m in {"lda", "riemann", "svm", "ensemble"}


def test_nested_model_selection_no_leakage():
    """On random data, nested model selection CV should stay near chance."""
    n_subjects = 2
    rng = np.random.default_rng(123)
    X_by_subject = [_make_subject_data(rng) for _ in range(n_subjects)]
    y_by_subject = [LABELS for _ in range(n_subjects)]

    _, mean_acc, _, _ = train_nested_model_selection_cv(
        X_by_subject, y_by_subject, n_outer_folds=5, n_inner_folds=3
    )

    assert mean_acc < 0.70, (
        f"Random data accuracy {mean_acc:.1%} is suspiciously high — possible data leakage"
    )


def test_nested_cached_parallel_path_keeps_statistical_parity():
    """Nested cached+parallel path should stay statistically aligned."""
    n_subjects = 2
    X_by_subject = [
        _make_subject_data(np.random.default_rng(200 + i)) for i in range(n_subjects)
    ]
    y_by_subject = [LABELS for _ in range(n_subjects)]
    trial_ptps = [compute_trial_max_ptp(X_mc) for _, X_mc in X_by_subject]

    shared_kwargs = dict(
        n_outer_folds=3,
        n_inner_folds=3,
        trial_ptps_by_subject=trial_ptps,
        split_strategy="stratified_group",
        trial_group_size=5,
        random_state=17,
    )

    b_scores, b_mean, b_std, _ = train_nested_model_selection_cv(
        X_by_subject, y_by_subject, n_jobs=1, enable_band_cache=False, **shared_kwargs
    )
    o_scores, o_mean, o_std, _ = train_nested_model_selection_cv(
        X_by_subject,
        y_by_subject,
        n_jobs=2,
        parallel_backend="threading",
        max_blas_threads_per_worker=1,
        enable_band_cache=True,
        cache_scope="subject",
        **shared_kwargs,
    )

    np.testing.assert_allclose(b_scores, o_scores, atol=5e-3, rtol=0)
    np.testing.assert_allclose(b_mean, o_mean, atol=2e-3, rtol=0)
    np.testing.assert_allclose(b_std, o_std, atol=2e-3, rtol=0)


# ---------- Cross-session evaluation tests ----------


def test_cross_session_evaluate_returns_all_methods():
    """cross_session_evaluate returns accuracy for all 4 classifiers."""
    rng = np.random.default_rng(42)
    X_feat_a, X_mc_a = _make_subject_data(rng)
    X_feat_b, X_mc_b = _make_subject_data(rng)

    results = cross_session_evaluate(
        (X_feat_a, X_mc_a), LABELS, (X_feat_b, X_mc_b), LABELS
    )

    assert set(results.keys()) == {"lda", "riemann", "svm", "ensemble"}
    for acc in results.values():
        assert 0 <= acc <= 1


def test_cross_session_no_leakage():
    """On random data, cross-session accuracy should stay near chance."""
    rng = np.random.default_rng(123)
    X_feat_a, X_mc_a = _make_subject_data(rng)
    X_feat_b, X_mc_b = _make_subject_data(rng)

    results = cross_session_evaluate(
        (X_feat_a, X_mc_a), LABELS, (X_feat_b, X_mc_b), LABELS
    )

    for method, acc in results.items():
        assert acc < 0.70, (
            f"Cross-session {method} accuracy {acc:.1%} is suspiciously high"
        )


# ---------- Per-fold artifact rejection tests ----------


def test_reject_in_fold_removes_outliers():
    """reject_in_fold drops trials with extreme PTP amplitudes."""
    ptps = np.array([10.0, 12.0, 11.0, 500.0, 9.0, 0.5, 13.0, 11.5])
    train_idx = np.array([0, 1, 2, 3, 4])
    test_idx = np.array([5, 6, 7])

    clean_train, clean_test = reject_in_fold(ptps, train_idx, test_idx)

    # Trial 3 (ptp=500) should be removed from train; trial 5 (ptp=0.5 < 1.0) from test
    assert 3 not in clean_train
    assert 5 not in clean_test
    # Normal trials should survive
    assert 0 in clean_train
    assert 6 in clean_test


def test_reject_in_fold_none_is_noop():
    """When trial_ptps is None, indices are returned unchanged."""
    train_idx = np.array([0, 1, 2])
    test_idx = np.array([3, 4])

    clean_train, clean_test = reject_in_fold(None, train_idx, test_idx)

    np.testing.assert_array_equal(clean_train, train_idx)
    np.testing.assert_array_equal(clean_test, test_idx)


def test_per_fold_rejection_excludes_artifacts():
    """CV with trial_ptps_by_subject runs without error and excludes artifact trials."""
    rng = np.random.default_rng(42)
    X_features, X_multichannel = _make_subject_data(rng)

    # Inject 2 extreme outlier trials (amplitude × 1000)
    X_multichannel[0] *= 1000
    X_multichannel[1] *= 1000

    trial_ptps = compute_trial_max_ptp(X_multichannel)

    scores, mean_acc, std_acc = train_within_subject_cv(
        [(X_features, X_multichannel)],
        [LABELS],
        n_folds=5,
        trial_ptps_by_subject=[trial_ptps],
    )

    assert len(scores) == 1
    assert 0 <= mean_acc <= 1
    assert std_acc >= 0
