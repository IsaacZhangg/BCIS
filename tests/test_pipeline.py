"""Tests for pipeline configuration."""

from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path


def test_pipeline_uses_phase_5_for_disengaged():
    """Test that pipeline extracts Phase 5 (rest) as disengaged class."""
    from src.pipeline import run_pipeline

    mock_rec_path = MagicMock()
    mock_rec_path.parent.parent.name = "test_subject"

    with (
        patch("src.pipeline.extract_augmented_epochs") as mock_extract,
        patch("src.pipeline.get_complete_recordings") as mock_get,
        patch("src.pipeline.load_recording") as mock_load,
        patch("src.pipeline.preprocess_eeg_multichannel") as mock_preprocess,
    ):
        mock_extract.return_value = ([], [])
        mock_get.return_value = [mock_rec_path]
        mock_load.return_value = (
            np.zeros((8, 1000)),
            {
                "phase_1": 0,
                "phase_2": 100,
                "phase_3": 200,
                "phase_4": 300,
                "phase_5": 400,
            },
            250,
        )
        mock_preprocess.return_value = np.zeros((8, 1000))

        try:
            run_pipeline(Path("unicorn-data"), Path("models"))
        except Exception:
            pass

        assert mock_extract.called, "extract_augmented_epochs should have been called"

        call_kwargs = mock_extract.call_args[1]
        assert call_kwargs.get("class2_phase") == 5, (
            f"Expected class2_phase=5, got {call_kwargs.get('class2_phase')}"
        )
