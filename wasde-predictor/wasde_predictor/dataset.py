"""Join the target and the clues into model-ready observations.

Each observation is one report in the target window:
  * label    -> did USDA raise the number that month? (1 up, 0 down/flat)
  * change   -> the actual change (units depend on the target)
  * features -> the clue vector known *before* that report's noon release
  * meta     -> marketing year, report date, and every source week used

The target attribute (yield / ending stocks), the demand clues, and the report
window are all passed in (see commodities.py). `feature_weeks` is carried through
so a test can assert every one is strictly before the report date -- the
anti-leakage guarantee.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from . import condition as condition_mod
from . import features as features_mod
from . import series
from . import weather as weather_mod
from . import wasde as wasde_mod

DEFAULT_MONTHS = (8, 9, 10, 11)


@dataclass(frozen=True)
class Observation:
    market_year: str
    report_date: dt.date
    month: int
    label: int
    change: float
    features: dict[str, float] = field(default_factory=dict)
    feature_weeks: tuple[dt.date, ...] = ()
    prev_direction: int | None = None
    prev_change: float | None = None


def build_dataset(
    wasde_path: str | Path,
    condition_path: str | Path,
    drought_path: str | Path,
    *,
    demand: list[tuple] | None = None,   # list of (DemandClue, path)
    commodity: str = "Corn",
    attribute: str = "Yield per Harvested Acre",
    target_months: tuple[int, ...] = DEFAULT_MONTHS,
) -> list[Observation]:
    reports = wasde_mod.load_reports(wasde_path, commodity=commodity, attribute=attribute)
    revs = wasde_mod.revisions(reports, target_months=target_months)
    conditions = condition_mod.load_condition(condition_path)
    droughts = weather_mod.load_drought(drought_path)

    demand_loaded = None
    if demand:
        demand_loaded = [(clue, series.load_weekly(path, clue.key_column)) for clue, path in demand]

    by_year: dict[str, list] = {}
    for rev in revs:
        by_year.setdefault(rev.market_year, []).append(rev)

    observations: list[Observation] = []
    for market_year in sorted(by_year):
        prev_direction: int | None = None
        prev_change: float | None = None
        for rev in sorted(by_year[market_year], key=lambda r: r.report_date):
            feats, weeks = features_mod.build_all_features(
                rev.report_date, rev.prev_report_date, conditions, droughts, demand_loaded
            )
            if feats is not None:
                observations.append(
                    Observation(
                        market_year=rev.market_year,
                        report_date=rev.report_date,
                        month=rev.month,
                        label=rev.direction,
                        change=rev.change,
                        features=feats,
                        feature_weeks=tuple(weeks),
                        prev_direction=prev_direction,
                        prev_change=prev_change,
                    )
                )
            prev_direction = rev.direction
            prev_change = rev.change
    observations.sort(key=lambda o: o.report_date)
    return observations


def class_balance(observations: list[Observation]) -> dict:
    n = len(observations)
    ups = sum(o.label for o in observations)
    return {
        "n": n,
        "up": ups,
        "down_or_flat": n - ups,
        "up_rate": round(ups / n, 3) if n else 0.0,
    }
