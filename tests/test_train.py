"""Tests for FBCSP + LDA, FBCSP + SVM, and Riemannian training pipelines."""

import numpy as np

from src.train import (
    cross_session_evaluate,
    predict,
    predict_riemann,
    train_final_model,
    train_final_model_riemann,
    train_final_model_svm,
    train_nested_model_selection_cv,
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
