"""Tests for FBCSP + LDA training pipeline."""

import numpy as np

from src.train import (
    predict,
    predict_riemann,
    predict_transfer,
    train_final_model,
    train_final_model_riemann,
    train_final_model_transfer,
    train_transfer_cv,
    train_within_subject_cv,
    train_within_subject_cv_riemann,
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

    preds = predict_riemann(model, X_multichannel)
    assert preds.shape == (N_EPOCHS,)
    assert set(np.unique(preds)).issubset({0, 1})


# ---------- Transfer learning classifier tests ----------


def test_transfer_cv_returns_scores():
    """Transfer CV returns per-subject scores with correct shape and range."""
    n_subjects = 3
    rng = np.random.default_rng(42)
    X_by_subject = [
        rng.standard_normal((N_EPOCHS, N_CHANNELS, N_SAMPLES))
        for _ in range(n_subjects)
    ]
    y_by_subject = [LABELS for _ in range(n_subjects)]
    subject_ids = [f"sub{i:04d}" for i in range(n_subjects)]

    scores, mean_acc, std_acc = train_transfer_cv(
        X_by_subject, y_by_subject, subject_ids, n_folds=5
    )

    assert len(scores) == n_subjects
    assert 0 <= mean_acc <= 1
    assert std_acc >= 0
    for s in scores:
        assert 0 <= s <= 1


def test_transfer_no_leakage():
    """On pure random data, transfer accuracy should be near chance (~50%)."""
    n_subjects = 3
    rng = np.random.default_rng(123)
    X_by_subject = [
        rng.standard_normal((N_EPOCHS, N_CHANNELS, N_SAMPLES))
        for _ in range(n_subjects)
    ]
    y_by_subject = [LABELS for _ in range(n_subjects)]
    subject_ids = [f"sub{i:04d}" for i in range(n_subjects)]

    _, mean_acc, _ = train_transfer_cv(
        X_by_subject, y_by_subject, subject_ids, n_folds=5
    )

    assert mean_acc < 0.70, (
        f"Random data accuracy {mean_acc:.1%} is suspiciously high — possible data leakage"
    )


def test_transfer_final_model_predict():
    """Round-trip: train a transfer model, then predict with it."""
    n_subjects = 3
    rng = np.random.default_rng(42)
    X_all = [
        rng.standard_normal((N_EPOCHS, N_CHANNELS, N_SAMPLES))
        for _ in range(n_subjects)
    ]
    y_all = [LABELS for _ in range(n_subjects)]
    subject_ids = [f"sub{i:04d}" for i in range(n_subjects)]

    model = train_final_model_transfer(X_all, y_all, subject_ids, target_idx=0)
    assert "cov_estimator" in model
    assert "tlc" in model
    assert "ts" in model
    assert "classifier" in model

    preds = predict_transfer(model, X_all[0])
    assert preds.shape == (N_EPOCHS,)
    assert set(np.unique(preds)).issubset({0, 1})
