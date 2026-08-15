"""Typed configuration objects for reproducible EEG training/evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

MI_DATA_NEW_DIR = Path("data/MI_DATA_NEW")
DEFAULT_SUBJECT_MERGE: dict[str, str] = {
    "subject0100_2": "subject0100",
    "subject0104_session002": "subject0104",
}

SplitStrategy = Literal["stratified", "stratified_group"]
ParallelBackend = Literal["loky", "threading"]
CacheScope = Literal["subject", "outer_fold"]
RiemannianClassifier = Literal["tangent_lr", "mdm"]


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
    ea_donor_augmentation: bool = True
    fbcsp_band_candidates: tuple[str, ...] = ("standard", "high_mu", "wide_mu")
    min_evaluation_trials: int = 30
    # Optimized Riemannian flags (defaults preserve the 60.1% baseline).
    riemannian_band: tuple[float, float] | None = None
    riemannian_classifier: RiemannianClassifier = "tangent_lr"
    use_pyriemann_transfer: bool = False
    # Temporal augmentation (sliding window applied to training data only).
    temporal_augmentation: bool = False
    aug_window_sec: float = 2.5
    aug_stride_sec: float = 0.25

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)

    def config_hash(self) -> str:
        """Return a 16-char hex hash for cache invalidation."""
        import hashlib
        import json

        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
