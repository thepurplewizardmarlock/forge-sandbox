"""Join the target and the clue into model-ready observations.

Each observation is one August-November corn report:
  * label    -> did USDA raise the yield that month? (1 up, 0 down/flat)
  * features -> the crop-condition clue known *before* that report's noon release
  * meta     -> marketing year, report date, the condition week actually used

The `condition_week` is carried through on purpose so a test can assert it is
always strictly before the report date -- our anti-leakage guarantee.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from . import condition as condition_mod
from . import wasde as wasde_mod


@dataclass(frozen=True)
class Observation:
    market_year: str
    report_date: dt.date
    month: int
    label: int                       # 1 = USDA raised yield, 0 = lowered/unchanged
    change: float                    # the actual yield change (for reference only)
    features: dict[str, float] = field(default_factory=dict)
    condition_week: dt.date | None = None  # week_ending of the clue used (for leak checks)


def build_dataset(
    yield_path: str | Path,
    condition_path: str | Path,
) -> list[Observation]:
    reports = wasde_mod.load_yield_reports(yield_path)
    revs = wasde_mod.revisions(reports)
    readings = condition_mod.load_condition(condition_path)

    observations: list[Observation] = []
    for rev in revs:
        clue = condition_mod.latest_before(readings, rev.report_date)
        if clue is None:
            # No condition known before this report -> can't build a feature; skip.
            continue
        observations.append(
            Observation(
                market_year=rev.market_year,
                report_date=rev.report_date,
                month=rev.month,
                label=rev.direction,
                change=rev.change,
                features={"condition_ge": clue.value},
                condition_week=clue.week_ending,
            )
        )
    return observations


def class_balance(observations: list[Observation]) -> dict[str, float]:
    n = len(observations)
    ups = sum(o.label for o in observations)
    return {
        "n": n,
        "up": ups,
        "down_or_flat": n - ups,
        "up_rate": round(ups / n, 3) if n else 0.0,
    }
