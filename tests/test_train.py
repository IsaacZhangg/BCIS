"""Tests for training pipeline."""

import numpy as np

from src.train import train_loso_cv, train_final_model


def test_train_loso_cv_returns_scores():
    """Test that LOSO CV returns accuracy scores for each subject."""
    # Simulated data: 5 subjects, 20 epochs each
    np.random.seed(42)
    n_subjects = 5
    n_epochs_per_class = 10

    X_by_subject = []
    y_by_subject = []

    for i in range(n_subjects):
        # Create separable data: focused has higher feature values
        focused = np.random.randn(n_epochs_per_class, 1) + 2
        not_focused = np.random.randn(n_epochs_per_class, 1) - 2

        X = np.vstack([focused, not_focused])
        y = np.array([1] * n_epochs_per_class + [0] * n_epochs_per_class)

        X_by_subject.append(X)
        y_by_subject.append(y)

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
