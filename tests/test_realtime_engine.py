"""Tests for the real-time streaming engine."""

import numpy as np

from src.eeg_stream import RandomEEGStream
from src.realtime_engine import (
    EngineConfig,
    IncrementalFilter,
    RealtimeEngine,
    RingBuffer,
    WindowResult,
    _make_bandpass_sos,
    _make_notch_sos,
)


class TestRingBuffer:
    def test_basic_append_and_read(self):
        buf = RingBuffer(max_samples=10, n_channels=2)
        buf.append(np.array([1.0, 2.0]))
        buf.append(np.array([3.0, 4.0]))
        data = buf.get_latest(2)
        assert data.shape == (2, 2)
        np.testing.assert_array_equal(data[:, 0], [1.0, 2.0])
        np.testing.assert_array_equal(data[:, 1], [3.0, 4.0])

    def test_wrapping(self):
        buf = RingBuffer(max_samples=3, n_channels=1)
        for i in range(5):
            buf.append(np.array([float(i)]))
        # Should contain [2, 3, 4] in order
        data = buf.get_latest()
        assert data.shape == (1, 3)
        np.testing.assert_array_equal(data[0], [2.0, 3.0, 4.0])

    def test_is_full(self):
        buf = RingBuffer(max_samples=3, n_channels=1)
        assert not buf.is_full
        buf.append(np.array([1.0]))
        buf.append(np.array([2.0]))
        assert not buf.is_full
        buf.append(np.array([3.0]))
        assert buf.is_full

    def test_count(self):
        buf = RingBuffer(max_samples=5, n_channels=1)
        assert buf.count == 0
        buf.append(np.array([1.0]))
        assert buf.count == 1
        for i in range(10):
            buf.append(np.array([float(i)]))
        assert buf.count == 5  # capped at max

    def test_get_latest_partial(self):
        buf = RingBuffer(max_samples=10, n_channels=1)
        for i in range(10):
            buf.append(np.array([float(i)]))
        data = buf.get_latest(3)
        assert data.shape == (1, 3)
        np.testing.assert_array_equal(data[0], [7.0, 8.0, 9.0])

    def test_append_many(self):
        buf = RingBuffer(max_samples=5, n_channels=2)
        samples = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        buf.append_many(samples)
        assert buf.count == 3
        data = buf.get_latest()
        assert data.shape == (2, 3)

    def test_clear(self):
        buf = RingBuffer(max_samples=5, n_channels=1)
        buf.append(np.array([1.0]))
        buf.clear()
        assert buf.count == 0
        assert not buf.is_full

    def test_get_latest_empty(self):
        buf = RingBuffer(max_samples=5, n_channels=2)
        data = buf.get_latest()
        assert data.shape == (2, 0)


class TestIncrementalFilter:
    def test_filter_shape_preserved(self):
        sos = _make_bandpass_sos(1.0, 40.0, 250.0)
        filt = IncrementalFilter(sos, n_channels=8)
        data = np.random.randn(8, 100)
        out = filt.filter(data)
        assert out.shape == data.shape

    def test_incremental_vs_batch(self):
        """Incremental filtering should match batch filtering."""
        from scipy.signal import sosfilt

        sos = _make_bandpass_sos(1.0, 40.0, 250.0)
        rng = np.random.default_rng(42)
        data = rng.normal(0, 10, (2, 1000))

        # Batch filter
        batch_out = np.zeros_like(data)
        for ch in range(2):
            batch_out[ch] = sosfilt(sos, data[ch])

        # Incremental filter (100-sample chunks)
        inc_out = self._filter_incrementally(data, sos, chunk_size=100)

        np.testing.assert_array_almost_equal(batch_out, inc_out, decimal=10)

    def _filter_incrementally(
        self, data: np.ndarray, sos: np.ndarray, chunk_size: int
    ) -> np.ndarray:
        """Filter data incrementally in chunks."""
        inc_filt = IncrementalFilter(sos, n_channels=data.shape[0])
        inc_out = np.zeros_like(data)

        for start in range(0, data.shape[1], chunk_size):
            end = min(start + chunk_size, data.shape[1])
            chunk = data[:, start:end]
            inc_out[:, start:end] = inc_filt.filter(chunk)

        return inc_out

    def test_reset(self):
        sos = _make_bandpass_sos(1.0, 40.0, 250.0)
        filt = IncrementalFilter(sos, n_channels=2)
        data = np.random.randn(2, 100)
        filt.filter(data)
        filt.reset()
        # After reset, filtering should produce same result as fresh filter
        fresh = IncrementalFilter(sos, n_channels=2)
        out1 = filt.filter(data)
        out2 = fresh.filter(data)
        np.testing.assert_array_almost_equal(out1, out2)


class TestNotchSOS:
    def test_notch_filter_shape(self):
        sos = _make_notch_sos(60.0, 250.0)
        assert sos.shape == (1, 6)

    def test_bandpass_filter_shape(self):
        sos = _make_bandpass_sos(1.0, 40.0, 250.0)
        assert sos.ndim == 2
        assert sos.shape[1] == 6


class MockModel:
    """Mock classifier that always returns 50/50 probability."""

    def predict_proba(self, X):
        n = X.shape[0]
        return np.full((n, 2), 0.5)


class MockScaler:
    """Mock scaler that passes through data unchanged."""

    def transform(self, X):
        return X


class TestRealtimeEngine:
    def _make_engine(self, window_sec=1.0, slide_sec=0.5, seed=42):
        stream = RandomEEGStream(seed=seed)
        config = EngineConfig(
            window_sec=window_sec,
            slide_sec=slide_sec,
            alert_threshold=30.0,
            alert_consecutive=3,
        )
        return RealtimeEngine(
            stream=stream,
            model=MockModel(),
            scaler=MockScaler(),
            config=config,
        )

    def test_start_stop(self):
        engine = self._make_engine()
        engine.start()
        assert engine.stream.is_running
        engine.stop()
        assert not engine.stream.is_running

    def test_produces_results_after_enough_samples(self):
        engine = self._make_engine(window_sec=1.0, slide_sec=1.0)
        engine.start()
        # 1 second = 250 samples. Need at least 250 to fill buffer.
        # Read 500 to get at least one window.
        results = engine.run(max_samples=500)
        engine.stop()
        assert len(results) >= 1

    def test_no_results_before_window_fills(self):
        engine = self._make_engine(window_sec=2.0, slide_sec=1.0)
        engine.start()
        # Read only 100 samples (0.4s, not enough for 2s window)
        results = engine.run(max_samples=100)
        engine.stop()
        assert len(results) == 0

    def test_result_fields(self):
        engine = self._make_engine(window_sec=1.0, slide_sec=1.0)
        engine.start()
        results = engine.run(max_samples=500)
        engine.stop()
        for r in results:
            assert isinstance(r, WindowResult)
            assert isinstance(r.timestamp_sec, float)
            assert 0 <= r.engagement_score <= 100
            assert isinstance(r.alert_triggered, bool)

    def test_callback_is_called(self):
        received = []

        def on_result(result):
            received.append(result)

        stream = RandomEEGStream(seed=42)
        config = EngineConfig(window_sec=1.0, slide_sec=1.0)
        engine = RealtimeEngine(
            stream=stream,
            model=MockModel(),
            scaler=MockScaler(),
            config=config,
            on_result=on_result,
        )
        engine.start()
        engine.run(max_samples=500)
        engine.stop()
        assert len(received) >= 1

    def test_score_history_accumulated(self):
        engine = self._make_engine(window_sec=1.0, slide_sec=1.0)
        engine.start()
        engine.run(max_samples=1000)
        engine.stop()
        # score_history is bounded to alert_consecutive most recent entries
        assert len(engine.score_history) <= engine.config.alert_consecutive
        assert len(engine.score_history) == min(
            len(engine.results), engine.config.alert_consecutive
        )

    def test_process_samples_incremental(self):
        engine = self._make_engine(window_sec=1.0, slide_sec=1.0)
        engine.start()
        stream = engine.stream

        # Feed samples in small batches
        all_results = []
        for _ in range(10):
            samples = stream.read_samples(50)
            results = engine.process_samples(samples)
            all_results.extend(results)

        engine.stop()
        # 500 samples total = 2 seconds, should have at least 1 result
        assert len(all_results) >= 1
