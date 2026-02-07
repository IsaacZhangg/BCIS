"""Abstract streaming interface for EEG data sources."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import CHANNELS, SFREQ

N_EEG_CHANNELS = len(CHANNELS)


class EEGStream(ABC):
    """Base class for EEG data streams.

    All streams yield (timestamp, 8-channel ndarray) tuples representing
    individual samples at 250 Hz from the Unicorn Hybrid Black montage.
    """

    def __init__(self, sfreq: float = SFREQ):
        self.sfreq = sfreq
        self._running = False
        self._sample_index = 0

    @abstractmethod
    def start(self) -> None:
        """Initialize and begin the data stream."""
        self._running = True
        self._sample_index = 0

    @abstractmethod
    def stop(self) -> None:
        """Stop the data stream and release resources."""
        self._running = False

    @abstractmethod
    def read_samples(self, n: int) -> list[tuple[float, np.ndarray]]:
        """Read n samples from the stream.

        Args:
            n: Number of samples to read.

        Returns:
            List of (timestamp_sec, channels) tuples where channels is
            a 1D ndarray of shape (8,) in the order defined by CHANNELS.
        """

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def channel_names(self) -> list[str]:
        return list(CHANNELS)


class FileEEGStream(EEGStream):
    """Replay a Unicorn CSV recording sample-by-sample.

    Reads the full CSV into memory on start() and serves samples
    sequentially from the array. Optionally loops back to the beginning
    when the recording ends.
    """

    def __init__(
        self,
        csv_path: Path | str,
        sfreq: float = SFREQ,
        loop: bool = False,
    ):
        super().__init__(sfreq)
        self.csv_path = Path(csv_path)
        self.loop = loop
        self._data: np.ndarray | None = None  # (n_samples, n_channels)

    def start(self) -> None:
        df = pd.read_csv(self.csv_path)
        self._data = df[CHANNELS].values  # (n_samples, 8)
        # Only mark as running after CSV load succeeds
        super().start()

    def stop(self) -> None:
        super().stop()
        self._data = None

    def read_samples(self, n: int) -> list[tuple[float, np.ndarray]]:
        if self._data is None:
            raise RuntimeError("Stream not started. Call start() first.")

        total = len(self._data)
        if total == 0:
            return []

        samples: list[tuple[float, np.ndarray]] = []
        for _ in range(n):
            if self._sample_index >= total and not self.loop:
                break

            timestamp = self._sample_index / self.sfreq
            channels = self._data[self._sample_index % total].copy()
            samples.append((timestamp, channels))
            self._sample_index += 1

        return samples


class RandomEEGStream(EEGStream):
    """Generate synthetic 8-channel EEG for testing without hardware.

    Produces realistic-looking EEG by summing sinusoids in standard
    frequency bands (alpha ~10 Hz, beta ~20 Hz) with pink noise.
    """

    def __init__(
        self,
        sfreq: float = SFREQ,
        seed: int | None = None,
    ):
        super().__init__(sfreq)
        self._rng: np.random.Generator | None = None
        self._seed = seed

    def start(self) -> None:
        super().start()
        self._rng = np.random.default_rng(self._seed)

    def stop(self) -> None:
        super().stop()
        self._rng = None

    def read_samples(self, n: int) -> list[tuple[float, np.ndarray]]:
        if self._rng is None:
            raise RuntimeError("Stream not started. Call start() first.")

        samples: list[tuple[float, np.ndarray]] = []
        ch_offsets = np.arange(N_EEG_CHANNELS, dtype=np.float64)

        for _ in range(n):
            t = self._sample_index / self.sfreq

            alpha = 15.0 * np.sin(2 * np.pi * 10 * t + ch_offsets * 0.5)
            beta = 5.0 * np.sin(2 * np.pi * 20 * t + ch_offsets * 0.3)
            theta = 8.0 * np.sin(2 * np.pi * 6 * t + ch_offsets * 0.7)
            noise = self._rng.normal(0, 10.0, size=N_EEG_CHANNELS)
            channels = alpha + beta + theta + noise

            samples.append((t, channels))
            self._sample_index += 1

        return samples


class LiveEEGStream(EEGStream):
    """Stream from a Unicorn Hybrid Black headset via UnicornPy.

    Requires the UnicornPy SDK to be installed (provided with the
    g.tec Unicorn Suite). Falls back with a clear error if unavailable.

    The Unicorn Hybrid Black outputs 17 channels per sample:
      8 EEG + 3 accelerometer + 3 gyroscope + battery + counter + validation.
    This stream extracts only the 8 EEG channels.
    """

    def __init__(
        self,
        serial: str | None = None,
        sfreq: float = SFREQ,
        frame_length: int = 1,
    ):
        super().__init__(sfreq)
        self._serial = serial
        self._frame_length = frame_length
        self._device = None
        self._n_acquired_channels: int = 0
        self._eeg_indices: list[int] = []
        self._receive_buffer: bytearray | None = None
        self._leftover: list[tuple[float, np.ndarray]] = []

    def start(self) -> None:
        try:
            import UnicornPy as _UnicornPy
        except ImportError as e:
            raise ImportError(
                "UnicornPy is not installed. Install the Unicorn Suite from "
                "g.tec to use LiveEEGStream. For testing, use RandomEEGStream "
                "or FileEEGStream instead."
            ) from e

        if self._serial is None:
            device_list = _UnicornPy.GetAvailableDevices(True)
            if not device_list:
                raise RuntimeError(
                    "No Unicorn devices found. Ensure the headset is paired "
                    "via Bluetooth."
                )
            self._serial = device_list[0]

        self._device = _UnicornPy.Unicorn(self._serial)

        self._n_acquired_channels = self._device.GetNumberOfAcquiredChannels()
        self._eeg_indices = [
            self._device.GetChannelIndex(f"EEG {i}") for i in range(1, 9)
        ]

        buf_len = self._frame_length * self._n_acquired_channels * 4  # float32
        self._receive_buffer = bytearray(buf_len)

        self._device.StartAcquisition(False)
        self._leftover = []

        # Only mark as running after all setup succeeds
        super().start()

    def stop(self) -> None:
        if self._device is not None:
            try:
                self._device.StopAcquisition()
            except Exception:
                pass
            self._device = None
        self._receive_buffer = None
        self._leftover = []
        super().stop()

    def read_samples(self, n: int) -> list[tuple[float, np.ndarray]]:
        if self._device is None or self._receive_buffer is None:
            raise RuntimeError("Stream not started. Call start() first.")

        samples: list[tuple[float, np.ndarray]] = []
        buf = self._receive_buffer
        buf_len = len(buf)

        # Drain any leftover samples from a previous partial frame read
        while self._leftover and len(samples) < n:
            samples.append(self._leftover.pop(0))

        # Read from device in full frames until we have enough
        while len(samples) < n:
            self._device.GetData(self._frame_length, buf, buf_len)

            all_values = np.frombuffer(
                buf,
                dtype=np.float32,
                count=self._n_acquired_channels * self._frame_length,
            ).reshape(self._frame_length, self._n_acquired_channels)

            for row_idx in range(self._frame_length):
                timestamp = self._sample_index / self.sfreq
                channels = all_values[row_idx, self._eeg_indices]
                self._sample_index += 1
                if len(samples) < n:
                    samples.append((timestamp, channels))
                else:
                    # Buffer excess samples for the next read_samples call
                    self._leftover.append((timestamp, channels))

        return samples
