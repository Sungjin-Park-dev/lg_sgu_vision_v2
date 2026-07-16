#!/usr/bin/env python3
"""Command-line entry point for the DP trajectory pipeline."""

from pathlib import Path
import sys

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_ROOT))

from core.trajectory.pipeline import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
