"""Load the clue (feature): USDA's weekly crop-condition ratings.

Real source: USDA NASS "Quick Stats" -- the weekly "% Good/Excellent" rating of
how the corn crop looks in the fields. A healthier-looking crop tends to lead
USDA to raise its yield estimate.

The one rule we never break: when a report comes out at noon on a given day, we
may only use a condition reading whose week ended *strictly before* that day.
`latest_before` enforces this point-in-time cutoff, so no future information can
leak into a feature.
"""
from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConditionReading:
    week_ending: dt.date
    state: str
    value: float  # percent good+excellent


def load_condition(path: str | Path) -> list[ConditionReading]:
    """Read a NASS-style crop-condition CSV, date-sorted."""
    path = Path(path)
    readings: list[ConditionReading] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            readings.append(
                ConditionReading(
                    week_ending=dt.date.fromisoformat(row["week_ending"].strip()),
                    state=row["state"].strip(),
                    value=float(row["value"]),
                )
            )
    readings.sort(key=lambda r: r.week_ending)
    return readings


def latest_before(
    readings: list[ConditionReading],
    cutoff: dt.date,
    state: str = "US TOTAL",
) -> ConditionReading | None:
    """Most recent reading for `state` whose week ended strictly before `cutoff`.

    This is the leakage-safe accessor: the returned reading is guaranteed to be
    knowable before a report published on `cutoff`.
    """
    best: ConditionReading | None = None
    for r in readings:
        if r.state != state:
            continue
        if r.week_ending < cutoff:
            if best is None or r.week_ending > best.week_ending:
                best = r
    return best
