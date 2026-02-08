"""Tests for training pipeline."""

import numpy as np

from src.train import train_loso_cv, train_final_model, train_left_right_within_subject


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
