"""Console output configuration for cleaner BCIS runs."""

from __future__ import annotations

import os
import warnings

import mne

_FILTER_LENGTH_WARNING = r"filter_length .* is longer than the signal.*"


def should_use_quiet_output() -> bool:
    """Return whether console output should default to quiet mode.

    Set `BCIS_VERBOSE_LOGS=1` to keep third-party logs/warnings visible.
    """
    return os.getenv("BCIS_VERBOSE_LOGS", "0").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def configure_console_output(quiet: bool | None = None) -> None:
    """Configure third-party log/warning verbosity.

    Args:
        quiet: If ``True``, suppress noisy library output. When ``None``,
            defaults from :func:`should_use_quiet_output`.
    """
    use_quiet = should_use_quiet_output() if quiet is None else quiet

    if use_quiet:
        mne.set_log_level("ERROR")
        warnings.filterwarnings(
            "ignore",
            message=_FILTER_LENGTH_WARNING,
            category=RuntimeWarning,
        )
    else:
        mne.set_log_level("INFO")
