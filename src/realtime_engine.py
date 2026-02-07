"""Streaming processing engine for real-time EEG engagement detection."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi

from src.data_loader import CHANNELS, SFREQ
from src.eeg_stream import EEGStream
from src.features import extract_realtime_features
from src.preprocess import apply_car
from src.simulate_realtime import check_alert, compute_engagement_score


@dataclass
class EngineConfig:
    sfreq: float = SFREQ
    window_sec: float = 5.0
    slide_sec: float = 1.0
    alert_threshold: float = 30.0
    alert_consecutive: int = 3
    bandpass_low: float = 1.0
    bandpass_high: float = 40.0
    notch_freq: float = 60.0
    max_results: int | None = (
        None  # None = unbounded (for finite runs), set for streaming
    )


@dataclass
class WindowResult:
    timestamp_sec: float
    engagement_score: float
    alert_triggered: bool
    features: np.ndarray | None = None


class RingBuffer:
    """Fixed-size ring buffer for accumulating multi-channel EEG samples.

    Stores the most recent `max_samples` worth of data across all channels.
    Supports efficient append and bulk read operations.
    """

    def __init__(self, max_samples: int, n_channels: int):
        self._buf = np.zeros((n_channels, max_samples), dtype=np.float64)
        self._max = max_samples
        self._n_channels = n_channels
        self._write_pos = 0
        self._count = 0

    @property
    def is_full(self) -> bool:
        return self._count >= self._max

    @property
    def count(self) -> int:
        return min(self._count, self._max)

    def append(self, sample: np.ndarray) -> None:
        self._buf[:, self._write_pos] = sample
        self._write_pos = (self._write_pos + 1) % self._max
        self._count += 1

    def append_many(self, samples: np.ndarray) -> None:
        """Append multiple samples. samples shape: (n_samples, n_channels)."""
        n = samples.shape[0]
        for i in range(n):
            self.append(samples[i])

    def get_latest(self, n: int | None = None) -> np.ndarray:
        """Get the latest n samples as (n_channels, n_samples).

        Returns data in chronological order (oldest first).
        """
        available = self.count
        if n is None:
            n = available
        n = min(n, available)

        if n == 0:
            return np.zeros((self._n_channels, 0), dtype=np.float64)

        if self._count >= self._max:
            # Buffer has wrapped - reconstruct chronological order
            indices = (
                np.arange(self._write_pos, self._write_pos + self._max) % self._max
            )
            return self._buf[:, indices][:, -n:]

        # Buffer hasn't wrapped yet - data is already in order
        return self._buf[:, self._count - n : self._count]

    def clear(self) -> None:
        self._buf[:] = 0
        self._write_pos = 0
        self._count = 0


class IncrementalFilter:
    """Incremental IIR filter using sosfilt with zi state.

    Maintains filter state across calls so that streaming data can be
    filtered continuously without edge artifacts between chunks.
    """

    def __init__(self, sos: np.ndarray, n_channels: int):
        self._sos = sos
        self._n_channels = n_channels
        self.reset()

    def filter(self, data: np.ndarray) -> np.ndarray:
        """Filter multi-channel data incrementally.

        Args:
            data: shape (n_channels, n_samples)

        Returns:
            Filtered data, same shape as input.
        """
        out = np.zeros_like(data)
        for ch in range(self._n_channels):
            out[ch], self._zi[ch] = sosfilt(self._sos, data[ch], zi=self._zi[ch])
        return out

    def reset(self) -> None:
        zi_shape = sosfilt_zi(self._sos).shape  # (n_sections, 2)
        self._zi = np.zeros((self._n_channels, *zi_shape), dtype=np.float64)


def _make_bandpass_sos(
    low: float, high: float, sfreq: float, order: int = 4
) -> np.ndarray:
    nyq = sfreq / 2.0
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def _make_notch_sos(freq: float, sfreq: float, Q: float = 30.0) -> np.ndarray:
    b, a = iirnotch(freq, Q, sfreq)
    # Convert to SOS via a single biquad section
    sos = np.zeros((1, 6))
    sos[0, :3] = b
    sos[0, 3:] = a
    return sos


class RealtimeEngine:
    """Streaming EEG processing engine.

    Accumulates samples from an EEGStream into a ring buffer, applies
    incremental bandpass and notch filtering, and at each window boundary
    extracts features and runs classification.

    Usage:
        engine = RealtimeEngine(stream, model, scaler, config, on_result=callback)
        engine.start()
        engine.run()  # blocking loop
        engine.stop()
    """

    def __init__(
        self,
        stream: EEGStream,
        model: object,
        scaler: object,
        config: EngineConfig | None = None,
        on_result: Callable[[WindowResult], None] | None = None,
    ):
        self.stream = stream
        self.model = model
        self.scaler = scaler
        self.config = config or EngineConfig(sfreq=stream.sfreq)
        self.on_result = on_result

        self._validate_config()
        self._buffer = RingBuffer(self._window_samples, self._n_channels)
        self._filtered_buffer = RingBuffer(self._window_samples, self._n_channels)

        bp_sos = _make_bandpass_sos(
            self.config.bandpass_low, self.config.bandpass_high, self.config.sfreq
        )
        notch_sos = _make_notch_sos(self.config.notch_freq, self.config.sfreq)
        self._bp_filter = IncrementalFilter(bp_sos, self._n_channels)
        self._notch_filter = IncrementalFilter(notch_sos, self._n_channels)

        self._score_history: deque[float] = deque(maxlen=self.config.alert_consecutive)
        self._results: deque[WindowResult] = deque(maxlen=self.config.max_results)
        self._samples_since_last_window = 0
        self._total_samples = 0
        self._first_window_emitted = False
        self._running = False

    def _validate_config(self) -> None:
        if self.config.sfreq != self.stream.sfreq:
            raise ValueError(
                f"Config sfreq ({self.config.sfreq}) does not match "
                f"stream sfreq ({self.stream.sfreq}). Pass matching values or "
                f"omit config.sfreq to use the stream's rate."
            )

        self._n_channels = len(CHANNELS)
        self._window_samples = int(self.config.sfreq * self.config.window_sec)
        self._slide_samples = int(self.config.sfreq * self.config.slide_sec)

        if self._window_samples < 1:
            raise ValueError(
                f"window_sec={self.config.window_sec} is too small for "
                f"sfreq={self.config.sfreq} (results in 0 samples per window). "
                f"Minimum window_sec is {1.0 / self.config.sfreq:.4f}."
            )

        if self._slide_samples < 1:
            raise ValueError(
                f"slide_sec={self.config.slide_sec} is too small for "
                f"sfreq={self.config.sfreq} (results in 0 samples per slide). "
                f"Minimum slide_sec is {1.0 / self.config.sfreq:.4f}."
            )

        if self.config.alert_consecutive < 1:
            raise ValueError(
                f"alert_consecutive must be >= 1, got {self.config.alert_consecutive}."
            )

    def start(self) -> None:
        self._buffer.clear()
        self._filtered_buffer.clear()
        self._bp_filter.reset()
        self._notch_filter.reset()
        self._score_history = deque(maxlen=self.config.alert_consecutive)
        self._results = deque(maxlen=self.config.max_results)
        self._samples_since_last_window = 0
        self._total_samples = 0
        self._first_window_emitted = False
        self.stream.start()
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.stream.stop()

    def process_samples(
        self, samples: list[tuple[float, np.ndarray]]
    ) -> list[WindowResult]:
        """Process a batch of incoming samples.

        Adds samples to the buffer, applies incremental filtering, and
        triggers window processing when enough new samples have arrived.

        Args:
            samples: List of (timestamp, channels) from an EEGStream.

        Returns:
            List of WindowResult objects produced during this batch.
        """
        results: list[WindowResult] = []

        for _timestamp, channels in samples:
            raw_2d = channels.reshape(self._n_channels, 1)
            bp_out = self._bp_filter.filter(raw_2d)
            filtered = self._notch_filter.filter(bp_out)

            self._buffer.append(channels)
            self._filtered_buffer.append(filtered[:, 0])

            self._total_samples += 1
            self._samples_since_last_window += 1

            if self._should_emit_window():
                result = self._process_window()
                if result is not None:
                    results.append(result)
                    self._results.append(result)
                    self._first_window_emitted = True
                    if self.on_result is not None:
                        self.on_result(result)
                self._samples_since_last_window = 0

        return results

    def _should_emit_window(self) -> bool:
        if not self._filtered_buffer.is_full:
            return False

        if not self._first_window_emitted:
            return True

        return self._samples_since_last_window >= self._slide_samples

    def _process_window(self) -> WindowResult | None:
        window_data = self._filtered_buffer.get_latest(self._window_samples)

        if window_data.shape[1] < self._window_samples:
            return None

        window_data = apply_car(window_data)

        window_by_channel: dict[str, np.ndarray] = {
            ch_name: window_data[ch_idx] for ch_idx, ch_name in enumerate(CHANNELS)
        }

        features = extract_realtime_features(window_by_channel, self.config.sfreq)
        scaled = self.scaler.transform(features.reshape(1, -1))

        prob_engaged = self.model.predict_proba(scaled)[0, 1]
        score = compute_engagement_score(prob_engaged)

        self._score_history.append(score)
        alert = check_alert(
            list(self._score_history),
            self.config.alert_threshold,
            self.config.alert_consecutive,
        )

        # Timestamp marks the start of the window, not the end
        window_start_sample = self._total_samples - self._window_samples
        timestamp = window_start_sample / self.config.sfreq

        return WindowResult(
            timestamp_sec=timestamp,
            engagement_score=round(score, 1),
            alert_triggered=alert,
            features=features,
        )

    def run(self, max_samples: int | None = None) -> list[WindowResult]:
        """Run the engine in a blocking loop.

        Reads from the stream continuously until the stream ends, the
        engine is stopped, or max_samples have been read.

        Args:
            max_samples: Stop after this many total samples (None = run until
                         stream ends or stop() is called).

        Returns:
            All WindowResult objects produced during the run.
        """
        total_read = 0

        while self._running:
            remaining = max_samples - total_read if max_samples is not None else None
            if remaining is not None and remaining <= 0:
                break

            chunk_size = self._slide_samples
            if remaining is not None:
                chunk_size = min(chunk_size, remaining)

            samples = self.stream.read_samples(chunk_size)
            if not samples:
                break

            self.process_samples(samples)
            total_read += len(samples)

        return list(self._results)

    @property
    def results(self) -> list[WindowResult]:
        return list(self._results)

    @property
    def score_history(self) -> list[float]:
        return list(self._score_history)
