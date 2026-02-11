"""Typed configuration objects for reproducible EEG training/evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

SplitStrategy = Literal["stratified", "stratified_group"]


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
    k_candidates: tuple[int, ...] = (3, 5, 8, 10, 15, 20)
    split_strategy: SplitStrategy = "stratified_group"
    trial_group_size: int = 2
    random_state: int = 42

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)
