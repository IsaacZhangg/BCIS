"""Run a nested-only experiment without overwriting models/training_results.json.

Run from the repo root: ``PYTHONPATH=. uv run python scripts/run_nested_experiment.py ...``
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.config import TrainingConfig
from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", required=True, help="Experiment name (results JSON stem)"
    )
    parser.add_argument("--composite-csp", action="store_true")
    parser.add_argument("--session-ea", action="store_true")
    parser.add_argument("--temporal-aug", action="store_true")
    parser.add_argument("--lam", type=float, default=0.3)
    args = parser.parse_args()

    cfg = replace(
        TrainingConfig(),
        use_composite_csp=args.composite_csp,
        composite_csp_lam=args.lam,
        session_level_ea=args.session_ea,
        temporal_augmentation=args.temporal_aug,
        ea_donor_augmentation=False,
    )
    out_dir = Path("experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{args.name}.json"
    print(f"Writing nested-only results to {results_path}")
    run_pipeline(
        Path("data/unicorn-data"),
        out_dir,
        training_config=cfg,
        skip_all_models_cv=True,
        skip_aug=True,
        skip_transfer=True,
        skip_cross_session=True,
        skip_save=True,
        results_path=results_path,
        quiet_output=True,
    )


if __name__ == "__main__":
    main()
