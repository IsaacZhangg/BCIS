"""Tests for FBCSP + LDA training pipeline."""

import numpy as np

from src.train import predict, train_final_model, train_within_subject_cv


def test_train_within_subject_cv_returns_scores():
    """Within-subject CV returns per-subject scores with correct shape and range."""
    n_subjects = 3
    n_epochs = 40
    n_features = 10
    n_channels = 8
    n_samples = 375  # 1.5s at 250Hz

    X_by_subject = [
        (
            np.random.default_rng(42 + i).standard_normal((n_epochs, n_features)),
            np.random.default_rng(42 + i).standard_normal(
                (n_epochs, n_channels, n_samples)
            ),
        )
        for i in range(n_subjects)
    ]
    y_by_subject = [np.array([0] * 20 + [1] * 20) for _ in range(n_subjects)]

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
    n_epochs = 40
    n_features = 10
    n_channels = 8
    n_samples = 375

    rng = np.random.default_rng(123)
    X_by_subject = [
        (
            rng.standard_normal((n_epochs, n_features)),
            rng.standard_normal((n_epochs, n_channels, n_samples)),
        )
        for _ in range(n_subjects)
    ]
    y_by_subject = [np.array([0] * 20 + [1] * 20) for _ in range(n_subjects)]

    _, mean_acc, _ = train_within_subject_cv(X_by_subject, y_by_subject, n_folds=5)

    assert mean_acc < 0.70, (
        f"Random data accuracy {mean_acc:.1%} is suspiciously high — possible data leakage"
    )


def test_train_final_model_returns_complete_model():
    """train_final_model returns a dict with all required keys."""
    n_epochs = 40
    n_features = 10
    n_channels = 8
    n_samples = 375

    rng = np.random.default_rng(42)
    X_features = rng.standard_normal((n_epochs, n_features))
    X_multichannel = rng.standard_normal((n_epochs, n_channels, n_samples))
    y = np.array([0] * 20 + [1] * 20)

    model = train_final_model(X_features, X_multichannel, y)

    required_keys = {
        "csp_models",
        "selector",
        "scaler",
        "classifier",
        "sfreq",
        "k_best",
    }
    assert set(model.keys()) == required_keys
    assert model["sfreq"] == 250.0
    assert isinstance(model["k_best"], int)
    assert model["k_best"] > 0


def test_predict_with_saved_model():
    """Round-trip: train a model, then predict with it."""
    n_epochs = 40
    n_features = 10
    n_channels = 8
    n_samples = 375

    rng = np.random.default_rng(42)
    X_features = rng.standard_normal((n_epochs, n_features))
    X_multichannel = rng.standard_normal((n_epochs, n_channels, n_samples))
    y = np.array([0] * 20 + [1] * 20)

    model = train_final_model(X_features, X_multichannel, y)
    preds = predict(model, X_features, X_multichannel)

    assert preds.shape == (n_epochs,)
    assert set(np.unique(preds)).issubset({0, 1})
