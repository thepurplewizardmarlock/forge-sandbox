"""Turn the raw clues into the feature vector the model sees.

All features are built only from readings dated strictly before the report, so
the whole vector is knowable at prediction time. `build_features` returns the
features plus the list of source weeks it used, so a test can assert none of
them leak past the report date.

The five v1 features:
  condition_ge              - latest % good/excellent before the report (level)
  condition_change          - that vs. the reading used at the previous report (momentum)
  condition_vs_season_start - that vs. this season's first reading (trajectory)
  drought_d2plus            - latest % corn area in D2+ drought before the report
  drought_change            - that vs. the reading used at the previous report
"""
from __future__ import annotations

import datetime as dt

from . import condition as condition_mod
from . import series
from . import weather as weather_mod

FEATURE_NAMES = [
    "condition_ge",
    "condition_change",
    "condition_vs_season_start",
    "drought_d2plus",
    "drought_change",
]

# Demand feature names are per-commodity now (see commodities.py DemandClue.feature).


def build_features(
    report_date: dt.date,
    prev_report_date: dt.date | None,
    conditions: list[series.WeeklyReading],
    droughts: list[series.WeeklyReading],
) -> tuple[dict[str, float] | None, list[dt.date]]:
    c_now = series.latest_before(conditions, report_date, condition_mod.US_TOTAL)
    d_now = series.latest_before(droughts, report_date, weather_mod.CORN_BELT)
    if c_now is None or d_now is None:
        return None, []  # not enough known before the report to build a vector

    weeks: list[dt.date] = [c_now.week_ending, d_now.week_ending]

    season_start = series.first_in_year(
        conditions, report_date.year, condition_mod.US_TOTAL, before=report_date
    )

    c_prev = (
        series.latest_before(conditions, prev_report_date, condition_mod.US_TOTAL)
        if prev_report_date else None
    )
    d_prev = (
        series.latest_before(droughts, prev_report_date, weather_mod.CORN_BELT)
        if prev_report_date else None
    )
    for r in (season_start, c_prev, d_prev):
        if r is not None:
            weeks.append(r.week_ending)

    features = {
        "condition_ge": c_now.value,
        "condition_change": c_now.value - c_prev.value if c_prev else 0.0,
        "condition_vs_season_start": c_now.value - season_start.value if season_start else 0.0,
        "drought_d2plus": d_now.value,
        "drought_change": d_now.value - d_prev.value if d_prev else 0.0,
    }
    return features, weeks


def build_demand_features(
    report_date: dt.date,
    demand_loaded: list[tuple],
) -> tuple[dict[str, float] | None, list[dt.date]]:
    """Build the demand feature vector from a list of (DemandClue, readings).

    Each clue contributes one feature: its latest point-in-time reading before the
    report. Returns (None, []) if any clue is missing before the report.
    """
    out: dict[str, float] = {}
    weeks: list[dt.date] = []
    for clue, readings in demand_loaded:
        r = series.latest_before(readings, report_date, clue.key)
        if r is None:
            return None, []
        out[clue.feature] = r.value
        weeks.append(r.week_ending)
    return out, weeks


def build_all_features(
    report_date: dt.date,
    prev_report_date: dt.date | None,
    conditions: list[series.WeeklyReading],
    droughts: list[series.WeeklyReading],
    demand_loaded: list[tuple] | None = None,
) -> tuple[dict[str, float] | None, list[dt.date]]:
    """Supply features, plus demand features when demand_loaded is provided.

    `demand_loaded` is a list of (DemandClue, readings). Returns (None, []) if any
    required clue is missing before the report, so the caller can skip that report.
    """
    feats, weeks = build_features(report_date, prev_report_date, conditions, droughts)
    if feats is None:
        return None, []
    if demand_loaded:
        dfeats, dweeks = build_demand_features(report_date, demand_loaded)
        if dfeats is None:
            return None, []
        feats = {**feats, **dfeats}
        weeks = weeks + dweeks
    return feats, weeks
