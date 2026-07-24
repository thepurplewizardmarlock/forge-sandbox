"""Join the target and the clues into model-ready observations.

Each observation is one August-November corn report:
  * label    -> did USDA raise the yield that month? (1 up, 0 down/flat)
  * features -> the clue vector known *before* that report's noon release
  * meta     -> marketing year, report date, and every source week used

`feature_weeks` is carried through on purpose so a test can assert every one is
strictly before the report date -- our anti-leakage guarantee.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from . import condition as condition_mod
from . import features as features_mod
from . import weather as weather_mod
from . import wasde as wasde_mod


@dataclass(frozen=True)
class Observation:
    market_year: str
    report_date: dt.date
    month: int
    label: int                       # 1 = USDA raised yield, 0 = lowered/unchanged
    change: float                    # the actual yield change (for reference only)
    features: dict[str, float] = field(default_factory=dict)
    feature_weeks: tuple[dt.date, ...] = ()
    prev_direction: int | None = None  # direction of the previous report this season (for persistence)
    prev_change: float | None = None   # magnitude of the previous report's change (for persistence regression)


def build_dataset(
    yield_path: str | Path,
    condition_path: str | Path,
    drought_path: str | Path,
) -> list[Observation]:
    reports = wasde_mod.load_yield_reports(yield_path)
    revs = wasde_mod.revisions(reports)
    conditions = condition_mod.load_condition(condition_path)
    droughts = weather_mod.load_drought(drought_path)

    by_year: dict[str, list] = {}
    for rev in revs:
        by_year.setdefault(rev.market_year, []).append(rev)

    observations: list[Observation] = []
    for market_year in sorted(by_year):
        prev_direction: int | None = None  # no prior target report at the season's start
        prev_change: float | None = None
        for rev in sorted(by_year[market_year], key=lambda r: r.report_date):
            feats, weeks = features_mod.build_features(
                rev.report_date, rev.prev_report_date, conditions, droughts
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
            # The previous report actually happened even if we couldn't build features.
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
