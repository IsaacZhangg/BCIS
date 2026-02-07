"""Real-time EEG engagement simulation.

Processes recordings with sliding windows to simulate real-time
fatigue/engagement detection. Supports both the legacy batch approach
and the new streaming architecture (FileEEGStream + RealtimeEngine).
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_recording, CHANNELS
from src.preprocess import preprocess_eeg
from src.features import extract_realtime_features


def compute_engagement_score(probability: float) -> float:
    """
    Convert model probability to 0-100 engagement score.

    Args:
        probability: Probability of engaged class (0-1)

    Returns:
        Engagement score (0-100)
    """
    return probability * 100


def check_alert(
    score_history: list[float],
    threshold: float = 30.0,
    consecutive: int = 3,
) -> bool:
    """
    Check if alert should be triggered based on score history.

    Args:
        score_history: List of recent engagement scores
        threshold: Score below which is considered fatigued
        consecutive: Number of consecutive low scores to trigger alert

    Returns:
        True if alert should trigger
    """
    if len(score_history) < consecutive:
        return False

    recent_scores = score_history[-consecutive:]
    return all(score < threshold for score in recent_scores)


def extract_sliding_windows(
    data: dict[str, np.ndarray],
    sfreq: float,
    window_sec: float = 5.0,
    step_sec: float = 5.0,
) -> list[dict[str, np.ndarray]]:
    """
    Extract sliding windows from multi-channel data.

    Args:
        data: Dict mapping channel names to signal arrays
        sfreq: Sampling frequency in Hz
        window_sec: Window duration in seconds
        step_sec: Step between windows in seconds

    Returns:
        List of window dicts, each mapping channel names to signal arrays
    """
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)

    # Get total length from first channel
    first_channel = list(data.keys())[0]
    total_samples = len(data[first_channel])

    windows = []
    start = 0

    while start + window_samples <= total_samples:
        window = {
            channel_name: signal[start : start + window_samples]
            for channel_name, signal in data.items()
        }
        windows.append(window)
        start += step_samples

    return windows


def simulate_recording(
    recording_path: Path,
    model_path: Path,
    scaler_path: Path,
    window_sec: float = 5.0,
    threshold: float = 30.0,
    consecutive: int = 3,
) -> pd.DataFrame:
    """
    Simulate real-time engagement detection on a recording.

    Args:
        recording_path: Path to recording CSV
        model_path: Path to trained model
        scaler_path: Path to fitted scaler
        window_sec: Window size in seconds
        threshold: Alert threshold (0-100)
        consecutive: Consecutive windows below threshold to trigger alert

    Returns:
        DataFrame with timestamp, engagement_score, alert_triggered columns
    """
    # Load resources
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    data, events, sample_rate = load_recording(recording_path)

    # Preprocess all channels
    processed_channels = preprocess_recording(data, sample_rate, CHANNELS)

    # Extract and process sliding windows
    windows = extract_sliding_windows(
        processed_channels, sample_rate, window_sec, window_sec
    )
    results = process_windows(
        windows, model, scaler, sample_rate, window_sec, threshold, consecutive
    )

    return pd.DataFrame(results)


def simulate_recording_streaming(
    recording_path: Path,
    model_path: Path,
    scaler_path: Path,
    window_sec: float = 5.0,
    slide_sec: float = 1.0,
    threshold: float = 30.0,
    consecutive: int = 3,
) -> pd.DataFrame:
    """
    Simulate real-time engagement detection using the streaming architecture.

    Uses FileEEGStream + RealtimeEngine for sample-by-sample processing
    with incremental filtering, matching how a live headset would operate.

    Args:
        recording_path: Path to recording CSV
        model_path: Path to trained model
        scaler_path: Path to fitted scaler
        window_sec: Window size in seconds
        slide_sec: Slide between windows in seconds
        threshold: Alert threshold (0-100)
        consecutive: Consecutive windows below threshold to trigger alert

    Returns:
        DataFrame with timestamp_sec, engagement_score, alert_triggered columns
    """
    from src.eeg_stream import FileEEGStream
    from src.realtime_engine import EngineConfig, RealtimeEngine

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    stream = FileEEGStream(recording_path)
    config = EngineConfig(
        window_sec=window_sec,
        slide_sec=slide_sec,
        alert_threshold=threshold,
        alert_consecutive=consecutive,
    )
    engine = RealtimeEngine(stream, model, scaler, config)
    engine.start()
    results = engine.run()
    engine.stop()

    rows = [
        {
            "timestamp_sec": r.timestamp_sec,
            "engagement_score": r.engagement_score,
            "alert_triggered": r.alert_triggered,
        }
        for r in results
    ]
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(
            columns=["timestamp_sec", "engagement_score", "alert_triggered"]
        )
    )


def preprocess_recording(
    data: np.ndarray, sample_rate: float, channels: list[str]
) -> dict[str, np.ndarray]:
    """Preprocess all channels in a recording."""
    return {
        channel_name: preprocess_eeg(data[channel_idx], sample_rate)
        for channel_idx, channel_name in enumerate(channels)
    }


def process_windows(
    windows: list[dict[str, np.ndarray]],
    model: object,
    scaler: object,
    sample_rate: float,
    window_sec: float,
    threshold: float,
    consecutive: int,
) -> list[dict]:
    """Process all windows and compute engagement scores."""
    results = []
    score_history = []

    for window_idx, window in enumerate(windows):
        # Extract and scale features
        features = extract_realtime_features(window, sample_rate)
        scaled_features = scaler.transform(features.reshape(1, -1))

        # Compute engagement score
        engaged_probability = model.predict_proba(scaled_features)[0, 1]
        score = compute_engagement_score(engaged_probability)

        # Check for alert
        score_history.append(score)
        alert = check_alert(score_history, threshold, consecutive)

        # Store result
        results.append(
            {
                "timestamp_sec": window_idx * window_sec,
                "engagement_score": round(score, 1),
                "alert_triggered": alert,
            }
        )

    return results


def _compute_time_coverage(timestamps: np.ndarray, window_sec: float) -> float:
    """Compute total time covered by merging overlapping window intervals.

    Each window at timestamp t covers [t, t + window_sec]. Overlapping
    intervals are merged so time is not double-counted.
    """
    if len(timestamps) == 0:
        return 0.0

    starts = np.sort(timestamps)
    total = 0.0
    current_start = starts[0]
    current_end = starts[0] + window_sec

    for timestamp in starts[1:]:
        window_end = timestamp + window_sec

        if timestamp <= current_end:
            # Overlapping or adjacent - extend the current interval
            current_end = max(current_end, window_end)
        else:
            # Gap - finalize previous interval and start a new one
            total += current_end - current_start
            current_start = timestamp
            current_end = window_end

    # Add the final interval
    total += current_end - current_start
    return total


def print_summary(
    df: pd.DataFrame,
    threshold: float,
    window_sec: float = 5.0,
) -> None:
    """Print summary statistics from simulation results."""
    print("\n" + "=" * 50)
    print("Engagement Score Summary")
    print("=" * 50)

    scores = df["engagement_score"]
    print(f"Mean:  {scores.mean():.1f}")
    print(f"Std:   {scores.std():.1f}")
    print(
        f"Min:   {scores.min():.1f} at {df.loc[scores.idxmin(), 'timestamp_sec']:.0f}s"
    )
    print(
        f"Max:   {scores.max():.1f} at {df.loc[scores.idxmax(), 'timestamp_sec']:.0f}s"
    )

    alerts = df["alert_triggered"].sum()
    print(f"\nAlerts triggered: {alerts}")

    below_threshold = (scores < threshold).sum()
    total_windows = len(df)
    print(f"Windows below threshold ({threshold}): {below_threshold}/{total_windows}")

    if below_threshold > 0:
        # Compute actual time coverage by merging overlapping window intervals
        below_mask = scores < threshold
        timestamps = df.loc[below_mask, "timestamp_sec"].values
        time_below = _compute_time_coverage(timestamps, window_sec)
        print(f"Total time below threshold: {time_below:.0f} seconds")


def main():
    args = parse_arguments()

    # Validate paths
    model_path = args.model_dir / "engagement_realtime_model.joblib"
    scaler_path = args.model_dir / "engagement_realtime_scaler.joblib"

    validation_error = validate_paths(args.recording, model_path, scaler_path)
    if validation_error:
        print(validation_error)
        return 1

    # Display configuration
    print(f"Processing: {args.recording}")
    print(f"Model: {model_path}")
    print(
        f"Window: {args.window}s, Threshold: {args.threshold}, Consecutive: {args.consecutive}"
    )

    # Run simulation based on mode
    results_df = run_simulation(args, model_path, scaler_path)

    if results_df.empty:
        print("No results produced (recording may be too short for window size).")
        return 0

    # Display and save results
    print_summary(results_df, args.threshold, args.window)

    output_path = args.output or Path("simulation_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return 0


def run_simulation(
    args: argparse.Namespace, model_path: Path, scaler_path: Path
) -> pd.DataFrame:
    """Run simulation in the appropriate mode."""
    if args.streaming:
        print("Mode: streaming (FileEEGStream + RealtimeEngine)")
        return simulate_recording_streaming(
            args.recording,
            model_path,
            scaler_path,
            window_sec=args.window,
            slide_sec=args.slide,
            threshold=args.threshold,
            consecutive=args.consecutive,
        )

    print("Mode: batch (legacy)")
    return simulate_recording(
        args.recording,
        model_path,
        scaler_path,
        window_sec=args.window,
        threshold=args.threshold,
        consecutive=args.consecutive,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simulate real-time engagement detection on EEG recording"
    )
    parser.add_argument(
        "recording",
        type=Path,
        help="Path to recording CSV file",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing trained model (default: models/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: simulation_results.csv)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Alert threshold (0-100, default: 30)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=5.0,
        help="Window size in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--slide",
        type=float,
        default=1.0,
        help="Slide between windows in seconds (default: 1.0, streaming mode only)",
    )
    parser.add_argument(
        "--consecutive",
        type=int,
        default=3,
        help="Consecutive windows below threshold to trigger alert (default: 3)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming architecture (FileEEGStream + RealtimeEngine)",
    )

    return parser.parse_args()


def validate_paths(
    recording_path: Path, model_path: Path, scaler_path: Path
) -> str | None:
    """Validate that all required paths exist."""
    if not model_path.exists():
        return (
            f"Error: Model not found at {model_path}\n"
            "Run the training pipeline first: python -m src.pipeline"
        )

    if not scaler_path.exists():
        return (
            f"Error: Scaler not found at {scaler_path}\n"
            "Run the training pipeline first: python -m src.pipeline"
        )

    if not recording_path.exists():
        return f"Error: Recording not found at {recording_path}"

    return None


if __name__ == "__main__":
    exit(main())
