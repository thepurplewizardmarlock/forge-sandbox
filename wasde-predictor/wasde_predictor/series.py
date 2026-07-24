"""A generic weekly time series and the point-in-time accessors used everywhere.

Both of our clues -- crop condition and drought -- are weekly readings keyed by
a place label (a state for condition, a region for drought). Rather than repeat
the loader/accessor logic, they share this module.

`latest_before` is the anti-leakage workhorse: it only ever returns a reading
whose week ended *strictly before* a cutoff date, so no future information can
sneak into a feature that feeds a report published on that date.
"""
from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WeeklyReading:
    week_ending: dt.date
    key: str        # state (condition) or region (drought)
    value: float


def load_weekly(path: str | Path, key_column: str) -> list[WeeklyReading]:
    """Read a CSV with columns week_ending, <key_column>, value; date-sorted."""
    path = Path(path)
    readings: list[WeeklyReading] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            readings.append(
                WeeklyReading(
                    week_ending=dt.date.fromisoformat(row["week_ending"].strip()),
                    key=row[key_column].strip(),
                    value=float(row["value"]),
                )
            )
    readings.sort(key=lambda r: r.week_ending)
    return readings


def latest_before(
    readings: list[WeeklyReading], cutoff: dt.date, key: str
) -> WeeklyReading | None:
    """Most recent reading for `key` whose week ended strictly before `cutoff`."""
    best: WeeklyReading | None = None
    for r in readings:
        if r.key == key or r.key.lower() == key.lower():
            if r.week_ending < cutoff and (best is None or r.week_ending > best.week_ending):
                best = r
    return best


def first_in_year(
    readings: list[WeeklyReading], year: int, key: str, before: dt.date | None = None
) -> WeeklyReading | None:
    """Earliest reading for `key` in `year` (optionally before a cutoff)."""
    best: WeeklyReading | None = None
    for r in readings:
        if r.week_ending.year != year:
            continue
        if r.key != key and r.key.lower() != key.lower():
            continue
        if before is not None and r.week_ending >= before:
            continue
        if best is None or r.week_ending < best.week_ending:
            best = r
    return best
