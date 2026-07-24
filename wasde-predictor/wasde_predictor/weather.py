"""Load the weather clue: U.S. Drought Monitor weekly % of corn area in D2+ drought.

More of the Corn Belt in serious drought (D2 = "severe" or worse) tends to lead
USDA to lower its yield estimate. Thin loader over `series`.
"""
from __future__ import annotations

from pathlib import Path

from .series import WeeklyReading, load_weekly

CORN_BELT = "US CORN BELT"


def load_drought(path: str | Path) -> list[WeeklyReading]:
    """Read a Drought-Monitor-style CSV (week_ending, region, metric, value)."""
    return load_weekly(path, key_column="region")
