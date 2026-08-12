"""Probe availability of pyriemann.transfer symbols used by the optimized pipeline.

Run: ``uv run python scripts/probe_pyriemann_transfer.py``

Result from 2026-04-17 probe (pyriemann 0.10 installed in .venv):
    TLCenter: available
    TLScale:  available  (replaces legacy TLStretch name from older docs)
    TLStretch: NOT available -- import fails with ImportError
The optimized transfer path therefore uses ``TLCenter`` + ``TLScale``. If a
future pyriemann release renames/removes either symbol, the fallback in
``src/transfer.py`` will log a warning and use manual EA.
"""

from __future__ import annotations

import importlib
import sys


def probe() -> dict[str, bool]:
    try:
        mod = importlib.import_module("pyriemann.transfer")
    except ImportError as exc:  # pragma: no cover
        print(f"pyriemann.transfer not importable: {exc}")
        return {}

    targets = ["TLCenter", "TLScale", "TLStretch", "TLRotate", "TLDummy"]
    results = {name: hasattr(mod, name) for name in targets}
    for name, ok in results.items():
        print(f"  {name}: {'available' if ok else 'MISSING'}")
    return results


if __name__ == "__main__":
    results = probe()
    if not results.get("TLCenter") or not results.get("TLScale"):
        sys.exit(1)
