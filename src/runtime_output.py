"""Console output configuration for cleaner BCIS runs."""

from __future__ import annotations

import os
import warnings

import mne

_FILTER_LENGTH_WARNING = r"filter_length .* is longer than the signal.*"
_VERBOSE_VALUES = {"1", "true", "yes", "on"}


def configure_console_output(quiet: bool | None = None) -> None:
    """Configure third-party log/warning verbosity.

    Args:
        quiet: If ``True``, suppress noisy library output. When ``None``,
            defaults to quiet unless ``BCIS_VERBOSE_LOGS=1``.
    """
    if quiet is None:
        quiet = os.getenv("BCIS_VERBOSE_LOGS", "0").lower() not in _VERBOSE_VALUES

    if quiet:
        mne.set_log_level("ERROR")
        warnings.filterwarnings(
            "ignore",
            message=_FILTER_LENGTH_WARNING,
            category=RuntimeWarning,
        )
    else:
        mne.set_log_level("INFO")
