"""Tests for EEG streaming interface."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import CHANNELS, SFREQ
from src.eeg_stream import FileEEGStream, RandomEEGStream


class TestRandomEEGStream:
    def test_start_stop(self):
        stream = RandomEEGStream()
        assert not stream.is_running
        stream.start()
        assert stream.is_running
        stream.stop()
        assert not stream.is_running

    def test_read_samples_returns_correct_count(self):
        stream = RandomEEGStream(seed=42)
        stream.start()
        samples = stream.read_samples(10)
        assert len(samples) == 10
        stream.stop()

    def test_sample_shape_and_type(self):
        stream = RandomEEGStream(seed=42)
        stream.start()
        samples = stream.read_samples(5)
        for timestamp, channels in samples:
            assert isinstance(timestamp, float)
            assert isinstance(channels, np.ndarray)
            assert channels.shape == (8,)
        stream.stop()

    def test_timestamps_are_sequential(self):
        stream = RandomEEGStream(seed=42)
        stream.start()
        samples = stream.read_samples(100)
        timestamps = [t for t, _ in samples]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]
        # Check spacing is 1/250 = 0.004s
        dt = timestamps[1] - timestamps[0]
        assert abs(dt - 1.0 / SFREQ) < 1e-10
        stream.stop()

    def test_reproducible_with_seed(self):
        stream1 = RandomEEGStream(seed=123)
        stream1.start()
        s1 = stream1.read_samples(50)
        stream1.stop()

        stream2 = RandomEEGStream(seed=123)
        stream2.start()
        s2 = stream2.read_samples(50)
        stream2.stop()

        for (t1, c1), (t2, c2) in zip(s1, s2):
            assert t1 == t2
            np.testing.assert_array_equal(c1, c2)

    def test_read_before_start_raises(self):
        stream = RandomEEGStream()
        with pytest.raises(RuntimeError, match="not started"):
            stream.read_samples(10)

    def test_channel_names(self):
        stream = RandomEEGStream()
        assert stream.channel_names == list(CHANNELS)

    def test_sfreq_default(self):
        stream = RandomEEGStream()
        assert stream.sfreq == SFREQ


class TestFileEEGStream:
    @pytest.fixture
    def csv_file(self, tmp_path):
        """Create a minimal Unicorn-format CSV for testing."""
        n_samples = 2500  # 10 seconds at 250 Hz
        rng = np.random.default_rng(42)
        data = {ch: rng.normal(0, 20, n_samples) for ch in CHANNELS}
        data["stim"] = np.zeros(n_samples)
        df = pd.DataFrame(data)
        path = tmp_path / "test_recording.csv"
        df.to_csv(path, index=False)
        return path

    def test_start_stop(self, csv_file):
        stream = FileEEGStream(csv_file)
        stream.start()
        assert stream.is_running
        stream.stop()
        assert not stream.is_running

    def test_reads_correct_number_of_samples(self, csv_file):
        stream = FileEEGStream(csv_file)
        stream.start()
        samples = stream.read_samples(100)
        assert len(samples) == 100
        stream.stop()

    def test_reads_all_samples(self, csv_file):
        stream = FileEEGStream(csv_file)
        stream.start()
        # Read more than available
        samples = stream.read_samples(5000)
        assert len(samples) == 2500
        stream.stop()

    def test_stops_at_end_without_loop(self, csv_file):
        stream = FileEEGStream(csv_file, loop=False)
        stream.start()
        # Read all
        samples = stream.read_samples(2500)
        assert len(samples) == 2500
        # Next read should return empty
        samples = stream.read_samples(10)
        assert len(samples) == 0
        stream.stop()

    def test_loops_back_with_loop(self, csv_file):
        stream = FileEEGStream(csv_file, loop=True)
        stream.start()
        # Read past the end
        samples = stream.read_samples(2600)
        assert len(samples) == 2600
        # The last 100 samples should be from the beginning again
        stream.stop()

    def test_sample_values_match_csv(self, csv_file):
        df = pd.read_csv(csv_file)
        stream = FileEEGStream(csv_file)
        stream.start()
        samples = stream.read_samples(10)
        for i, (timestamp, channels) in enumerate(samples):
            expected = df[CHANNELS].values[i]
            np.testing.assert_array_almost_equal(channels, expected)
        stream.stop()

    def test_timestamps_are_correct(self, csv_file):
        stream = FileEEGStream(csv_file)
        stream.start()
        samples = stream.read_samples(5)
        for i, (timestamp, _) in enumerate(samples):
            assert abs(timestamp - i / SFREQ) < 1e-10
        stream.stop()

    def test_read_before_start_raises(self, csv_file):
        stream = FileEEGStream(csv_file)
        with pytest.raises(RuntimeError, match="not started"):
            stream.read_samples(10)
