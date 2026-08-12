"""Structured experiment tracking.

Each experiment run produces a JSON file in the experiments/ directory.
Provides save, load, list, and compare operations.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_EXPERIMENTS_DIR = Path("experiments")


def save_experiment(
    name: str,
    results: dict,
    config_diff: dict,
    verdict: str,
    seed: int,
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR,
    baseline_ref: str | None = None,
    seed_type: str = "single",
) -> Path:
    """Save an experiment result as structured JSON."""
    experiments_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filename = f"{date_prefix}_{name}.json"
    path = experiments_dir / filename

    data = {
        "name": name,
        "timestamp": timestamp,
        "seed": seed,
        "seed_type": seed_type,
        "config_diff": config_diff,
        "results": results,
        "verdict": verdict,
    }
    if baseline_ref is not None:
        data["baseline_ref"] = baseline_ref

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def load_experiment(
    name: str,
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR,
) -> dict:
    """Load an experiment by name (matches any file containing the name)."""
    if not experiments_dir.exists():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")

    matches = list(experiments_dir.glob(f"*{name}*.json"))
    if not matches:
        raise FileNotFoundError(f"No experiment matching {name!r} in {experiments_dir}")

    path = sorted(matches)[-1]
    with open(path) as f:
        return json.load(f)


def list_experiments(
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR,
) -> list[dict]:
    """List all experiments as summary dicts."""
    if not experiments_dir.exists():
        return []

    summaries = []
    for path in sorted(experiments_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        summaries.append(
            {
                "name": data["name"],
                "timestamp": data.get("timestamp", ""),
                "verdict": data.get("verdict", ""),
                "nested_mean": data.get("results", {}).get("nested_mean"),
                "file": path.name,
            }
        )
    return summaries


def compare_experiments(
    name_a: str,
    name_b: str,
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR,
) -> dict:
    """Compare two experiments, returning per-metric deltas."""
    exp_a = load_experiment(name_a, experiments_dir)
    exp_b = load_experiment(name_b, experiments_dir)

    comparison = {}
    results_a = exp_a.get("results", {})
    results_b = exp_b.get("results", {})

    all_keys = set(results_a) | set(results_b)
    for key in sorted(all_keys):
        val_a = results_a.get(key)
        val_b = results_b.get(key)
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            comparison[key] = {
                "a": val_a,
                "b": val_b,
                "delta": val_b - val_a,
            }

    return comparison


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.experiments [list|compare <a> <b>]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "list":
        for exp in list_experiments():
            nested = exp.get("nested_mean")
            nested_str = f"{nested:.1%}" if nested is not None else "N/A"
            print(f"  {exp['name']:<30} {nested_str:>8}  [{exp['verdict']}]")
    elif command == "compare" and len(sys.argv) >= 4:
        result = compare_experiments(sys.argv[2], sys.argv[3])
        for key, vals in result.items():
            print(
                f"  {key:<20} {vals['a']:>8.3f} -> {vals['b']:>8.3f}  ({vals['delta']:+.3f})"
            )
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
