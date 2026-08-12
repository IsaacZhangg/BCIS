"""Smoke-test the stacking meta-learner on synthetic data.

Run: ``uv run python scripts/verify_stacking.py``

Builds a small synthetic subject and runs nested model-selection CV end-to-end.
Asserts that the 5-tuple return now includes per-subject stacking scores in
[0, 1] and that the within-subject CV dict also reports a ``stacking`` entry.
"""

from __future__ import annotations

import numpy as np

from src.train import (
    ALL_CLASSIFIERS,
    train_nested_model_selection_cv,
    train_within_subject_cv_all_models,
)


def _make_subject(seed: int, n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_features = rng.standard_normal((n, 10))
    X_mc = rng.standard_normal((n, 8, 375))
    return X_features, X_mc


def main() -> None:
    assert "stacking" in ALL_CLASSIFIERS, ALL_CLASSIFIERS
    print(f"ALL_CLASSIFIERS = {ALL_CLASSIFIERS}")

    y = np.array([0] * 20 + [1] * 20)
    X_by_subject = [_make_subject(42 + i) for i in range(2)]
    y_by_subject = [y, y]

    scores, mean_acc, std_acc, methods, _band_configs, stacking_scores = (
        train_nested_model_selection_cv(
            X_by_subject,
            y_by_subject,
            n_outer_folds=5,
            n_inner_folds=3,
        )
    )
    print(
        f"nested: mean={mean_acc:.3f} std={std_acc:.3f} "
        f"methods={methods} stacking_scores={stacking_scores}"
    )
    assert len(stacking_scores) == 2
    for s in stacking_scores:
        assert 0.0 <= s <= 1.0, s

    results = train_within_subject_cv_all_models(X_by_subject, y_by_subject, n_folds=5)
    print("within-subject results (mean per classifier):")
    for name in ALL_CLASSIFIERS:
        _, m, s = results[name]
        print(f"  {name:>10} = {m:.3f} (+/- {s:.3f})")
    assert "stacking" in results, list(results.keys())

    print("OK: stacking integrated into both CV paths.")


if __name__ == "__main__":
    main()
