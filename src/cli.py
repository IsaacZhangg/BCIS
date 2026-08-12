"""CLI argument parsing and execution configuration.

RunConfig controls WHAT to run (stages, subjects, classifiers).
TrainingConfig controls HOW to run (ML hyperparameters).
They are separate so TrainingConfig stays hashable for cache invalidation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    """Execution parameters for a pipeline run."""

    stages: list[str] = field(default_factory=lambda: ["all"])
    subjects: list[str] | None = None
    classifiers: list[str] | None = None
    experiment_name: str | None = None
    cache_dir: Path | None = Path("cache")
    export_models: bool = True
    data_dir: Path = Path("data/unicorn-data")
    output_dir: Path = Path("models")
    holdout_fraction: float = 0.0
    quiet: bool = True


def parse_args(argv: list[str] | None = None) -> RunConfig:
    """Parse CLI arguments into a RunConfig."""
    parser = argparse.ArgumentParser(description="BCI Motor Imagery Training Pipeline")
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated stages to run (e.g., 'cv', 'features,cv')",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject IDs to include (e.g., '0001,0003')",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=None,
        help="Comma-separated classifiers to evaluate (e.g., 'lda,svm')",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment run",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Directory for stage caches",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip model export step",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/unicorn-data"),
        help="Directory with subject/session recordings (default: data/unicorn-data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save models and training_results.json",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help="Fraction of trials held out per subject (0.0 disables)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose third-party logs (MNE, warnings)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable on-disk stage cache (always recompute)",
    )

    args = parser.parse_args(argv)

    return RunConfig(
        stages=args.stages.split(","),
        subjects=args.subjects.split(",") if args.subjects else None,
        classifiers=args.classifier.split(",") if args.classifier else None,
        experiment_name=args.experiment_name,
        cache_dir=None if args.no_cache else args.cache_dir,
        export_models=not args.no_export,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        holdout_fraction=args.holdout_fraction,
        quiet=not args.verbose,
    )
