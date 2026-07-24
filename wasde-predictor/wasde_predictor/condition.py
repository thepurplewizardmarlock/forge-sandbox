"""Load the crop-condition clue: USDA NASS weekly "% Good/Excellent" ratings.

A healthier-looking corn crop tends to lead USDA to raise its yield estimate.
This is a thin loader over `series`; the point-in-time accessors live there.
"""
from __future__ import annotations

from pathlib import Path

from .series import WeeklyReading, load_weekly

US_TOTAL = "US TOTAL"


def load_condition(path: str | Path) -> list[WeeklyReading]:
    """Read a NASS-style crop-condition CSV (week_ending, state, metric, value)."""
    return load_weekly(path, key_column="state")
