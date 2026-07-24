"""Load the demand-side clues used for the ENDING STOCKS target.

Ending stocks = supply minus demand, so besides the supply clues (condition,
drought) the model gets two demand clues:

  * export pace surprise  -- how far ahead of / behind the normal pace corn export
    commitments are running (+ = selling faster than normal -> fewer ending stocks)
  * ethanol pace surprise -- how far ahead of / behind normal the ethanol grind is
    (+ = grinding faster than needed -> fewer ending stocks)

Both are stationary "vs. normal pace" measures (the kind analysts actually build
from USDA FAS export sales and EIA weekly ethanol data). Thin loaders over
`series`; the point-in-time accessors live there.
"""
from __future__ import annotations

from pathlib import Path

from .series import WeeklyReading, load_weekly

CORN = "CORN"
US = "US"


def load_exports(path: str | Path) -> list[WeeklyReading]:
    """Read an export-pace CSV (week_ending, commodity, metric, value)."""
    return load_weekly(path, key_column="commodity")


def load_ethanol(path: str | Path) -> list[WeeklyReading]:
    """Read an ethanol-pace CSV (week_ending, region, metric, value)."""
    return load_weekly(path, key_column="region")
