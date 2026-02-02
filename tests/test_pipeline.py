"""Tests for pipeline configuration."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path


def test_pipeline_uses_phase_5_for_disengaged():
    """Test that pipeline extracts Phase 5 (rest) as disengaged class."""
    from src.pipeline import run_pipeline

    # Create mock recording path
    mock_rec_path = MagicMock()
    mock_rec_path.parent.parent.name = "test_subject"

    # We'll verify by checking the extract_erd_epochs call
    with patch('src.pipeline.extract_erd_epochs') as mock_extract:
        # Return empty pairs to trigger early exit
        mock_extract.return_value = ([], [])

        with patch('src.pipeline.get_complete_recordings') as mock_get:
            # Return one mock recording so extract_erd_epochs gets called
            mock_get.return_value = [mock_rec_path]

            with patch('src.pipeline.load_recording') as mock_load:
                # Return minimal mock data
                mock_load.return_value = (
                    np.zeros((8, 1000)),  # 8 channels, 1000 samples
                    {'phase_1': 0, 'phase_2': 100, 'phase_3': 200, 'phase_4': 300, 'phase_5': 400},
                    250  # sample rate
                )

                with patch('src.pipeline.preprocess_eeg') as mock_preprocess:
                    mock_preprocess.return_value = np.zeros(1000)

                    try:
                        run_pipeline(Path("unicorn-data"), Path("models"))
                    except Exception:
                        pass

                    # Assert that extract_erd_epochs was called
                    assert mock_extract.called, "extract_erd_epochs should have been called"

                    # Check the class2_phase parameter
                    call_kwargs = mock_extract.call_args[1]
                    assert call_kwargs.get('class2_phase') == 5, \
                        f"Expected class2_phase=5, got {call_kwargs.get('class2_phase')}"
