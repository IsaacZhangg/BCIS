"""Tests for training pipeline."""

import inspect

import numpy as np
import pytest

from src.train import (
    _aggregate_ensemble,
    train_final_model,
    train_final_model_cv,
    train_left_right_within_subject,
    train_loso_cv,
    train_within_subject_cv_riemannian,
)


def test_train_loso_cv_returns_scores():
    """Test that LOSO CV returns accuracy scores for each subject."""
    np.random.seed(42)
    n_subjects = 5
    n_per_class = 10

    X_by_subject = [
        np.vstack(
            [np.random.randn(n_per_class, 1) + 2, np.random.randn(n_per_class, 1) - 2]
        )
        for _ in range(n_subjects)
    ]
    y_by_subject = [
        np.array([1] * n_per_class + [0] * n_per_class) for _ in range(n_subjects)
    ]

    scores, mean_acc, std_acc = train_loso_cv(X_by_subject, y_by_subject)

    # Should return one score per subject
    assert len(scores) == n_subjects
    # With separable data, accuracy should be high
    assert mean_acc > 0.8
    assert 0 <= std_acc <= 1


def test_train_final_model_returns_model():
    """Test that final model training returns a fitted model."""
    np.random.seed(42)

    # Simple separable data
    X = np.vstack(
        [
            np.random.randn(50, 1) + 2,  # focused
            np.random.randn(50, 1) - 2,  # not focused
        ]
    )
    y = np.array([1] * 50 + [0] * 50)

    model, scaler = train_final_model(X, y)

    # Model should be able to predict
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    assert len(predictions) == len(y)

    # Should have high accuracy on training data
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.9


def test_train_left_right_within_subject_returns_scores():
    """Test within-subject CV returns per-subject scores."""
    n_subjects = 3
    n_epochs_per_subject = 40  # Need more for 10-fold CV
    n_features = 10
    n_channels = 8
    n_samples = 625  # 2.5 seconds at 250Hz

    X_by_subject = [
        (
            np.random.randn(n_epochs_per_subject, n_features),
            np.random.randn(n_epochs_per_subject, n_channels, n_samples),
        )
        for _ in range(n_subjects)
    ]
    y_by_subject = [np.array([0] * 20 + [1] * 20) for _ in range(n_subjects)]

    scores, mean_acc, std_acc = train_left_right_within_subject(
        X_by_subject,
        y_by_subject,
        n_folds=5,  # Use fewer folds for speed
    )

    assert len(scores) == n_subjects
    assert 0 <= mean_acc <= 1
    assert std_acc >= 0


# --- Anti-leakage tests ---


def test_aggregate_ensemble_signature_no_y_test():
    """_aggregate_ensemble must not accept y_test or any test labels."""
    sig = inspect.signature(_aggregate_ensemble)
    param_names = list(sig.parameters.keys())
    assert param_names == ["all_probs"], (
        f"Expected only 'all_probs' parameter, got {param_names}"
    )


def test_aggregate_ensemble_returns_predictions():
    """_aggregate_ensemble returns correct class predictions from probabilities."""
    probs = [
        np.array([0.9, 0.2, 0.8, 0.3]),
        np.array([0.7, 0.4, 0.6, 0.1]),
        np.array([0.8, 0.3, 0.9, 0.2]),
    ]
    preds = _aggregate_ensemble(probs)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (4,)
    assert set(np.unique(preds)).issubset({0, 1})
    # Mean probs: [0.8, 0.3, 0.767, 0.2] → [1, 0, 1, 0]
    np.testing.assert_array_equal(preds, np.array([1, 0, 1, 0]))


def test_ensemble_no_leakage_on_random_data():
    """On pure random data, ensemble accuracy should be near chance (~50%), not 70%+."""
    np.random.seed(123)
    n_subjects = 2
    n_epochs = 40
    n_features = 10
    n_channels = 8
    n_samples = 375  # 1.5s at 250Hz

    X_by_subject = [
        (
            np.random.randn(n_epochs, n_features),
            np.random.randn(n_epochs, n_channels, n_samples),
        )
        for _ in range(n_subjects)
    ]
    y_by_subject = [np.array([0] * 20 + [1] * 20) for _ in range(n_subjects)]

    scores, mean_acc, _ = train_left_right_within_subject(
        X_by_subject, y_by_subject, n_folds=5
    )

    # With random data, accuracy should be near 50%. Allow generous margin for
    # variance in small samples, but it must NOT be above 70% (which would
    # indicate oracle selection / data leakage).
    assert mean_acc < 0.70, (
        f"Random data accuracy {mean_acc:.1%} is suspiciously high — possible data leakage"
    )


def test_riemannian_no_leakage_on_random_data():
    """On pure random data, riemannian CV should be near chance, not 70%+."""
    np.random.seed(456)
    n_subjects = 2
    n_epochs = 40
    n_features = 10
    n_channels = 8
    n_samples = 375

    X_by_subject = [
        (
            np.random.randn(n_epochs, n_features),
            np.random.randn(n_epochs, n_channels, n_samples),
        )
        for _ in range(n_subjects)
    ]
    y_by_subject = [np.array([0] * 20 + [1] * 20) for _ in range(n_subjects)]

    scores, mean_acc, _ = train_within_subject_cv_riemannian(
        X_by_subject, y_by_subject, n_folds=5
    )

    assert mean_acc < 0.70, (
        f"Random data accuracy {mean_acc:.1%} is suspiciously high — possible data leakage"
    )


def test_train_final_model_cv_returns_valid_scores():
    """train_final_model_cv returns per-subject scores in [0, 1]."""
    np.random.seed(789)
    n_subjects = 3
    n_epochs = 40
    n_features = 10
    n_channels = 8
    n_samples = 375

    X_by_subject = [
        (
            np.random.randn(n_epochs, n_features),
            np.random.randn(n_epochs, n_channels, n_samples),
        )
        for _ in range(n_subjects)
    ]
    y_by_subject = [np.array([0] * 20 + [1] * 20) for _ in range(n_subjects)]

    scores, mean_acc, std_acc = train_final_model_cv(
        X_by_subject, y_by_subject, n_folds=5
    )

    assert len(scores) == n_subjects
    assert 0 <= mean_acc <= 1
    assert std_acc >= 0
    for s in scores:
        assert 0 <= s <= 1


@pytest.mark.parametrize("seed", [0, 42, 99])
def test_final_model_cv_no_leakage_on_random_data(seed):
    """On random data, final model CV should be near chance."""
    np.random.seed(seed)
    n_subjects = 2
    n_epochs = 40
    n_features = 10

    X_by_subject = [
        (
            np.random.randn(n_epochs, n_features),
            np.empty((n_epochs, 8, 375)),  # multichannel unused by this function
        )
        for _ in range(n_subjects)
    ]
    y_by_subject = [np.array([0] * 20 + [1] * 20) for _ in range(n_subjects)]

    _, mean_acc, _ = train_final_model_cv(X_by_subject, y_by_subject, n_folds=5)

    assert mean_acc < 0.70, (
        f"Random data accuracy {mean_acc:.1%} is suspiciously high — possible data leakage"
    )
