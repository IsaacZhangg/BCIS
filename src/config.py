"""Typed configuration objects for reproducible EEG training/evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

MI_DATA_NEW_DIR = Path("Data/MI_DATA_NEW")
DEFAULT_SUBJECT_MERGE: dict[str, str] = {
    "subject0100_2": "subject0100",
    "subject0104_session002": "subject0104",
}

SplitStrategy = Literal["stratified", "stratified_group"]
ParallelBackend = Literal["loky", "threading"]
CacheScope = Literal["subject", "outer_fold"]


@dataclass(frozen=True)
class TrainingConfig:
    """Global training/evaluation settings.

    The defaults are conservative for leakage control: grouped stratified CV keeps
    nearby trials together while preserving class balance.
    """

    sfreq: float = 250.0
    n_folds: int = 10
    n_outer_folds: int = 10
    n_inner_folds: int = 7
    k_best: int = 10
    k_candidates: tuple[int, ...] = (3, 5, 8, 10, 15, 20, 25, 30)
    split_strategy: SplitStrategy = "stratified_group"
    trial_group_size: int = 1
    random_state: int = 42
    n_jobs: int = -1
    parallel_backend: ParallelBackend = "loky"
    max_blas_threads_per_worker: int = 1
    enable_band_cache: bool = True
    cache_scope: CacheScope = "subject"
    augmentation_weakness_threshold: float = 0.50
    fbcsp_band_candidates: tuple[str, ...] = ("standard", "high_mu", "wide_mu")
    min_evaluation_trials: int = 30

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)
