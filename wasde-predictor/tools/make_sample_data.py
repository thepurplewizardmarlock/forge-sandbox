"""Generate SYNTHETIC sample data for the wasde-predictor walking skeleton.

THIS IS NOT REAL USDA DATA. It is deterministic fake data whose only purpose is
to let the whole pipeline run and the tests pass without a network connection.
Replace the files in data/sample/ with a real USDA download to get real results
(see README.md -> "Getting real data").

The generated CSVs deliberately mimic the *shape* of the real sources:
  * wasde_corn_yield_sample.csv  -> like USDA's Consolidated Historical WASDE CSV
    (columns Commodity, Region, Attribute, MarketYear, ReportDate, Value, Unit)
  * crop_condition_sample.csv    -> like NASS "% Good/Excellent" weekly readings
    (columns week_ending, state, metric, value)

A latent "season quality" per year drives BOTH the crop-condition path and the
month-to-month yield revisions, so there is a real (but noisy) relationship for
the model to find -- just as, in reality, a healthy-looking crop tends to lead
USDA to revise its yield estimate upward.

Run:  python3 tools/make_sample_data.py
"""
from __future__ import annotations

import csv
import datetime as dt
import random
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"

# Corn's new-crop marketing years we fabricate (report calendar year == start year).
START_YEARS = list(range(2013, 2025))  # 2013/14 .. 2024/25  -> 12 seasons
NORMAL_GE = 64.0  # long-run "normal" % good/excellent we center the fake crop on
# Plausible mid-month WASDE report dates per (month) -> we only need Jul..Nov.
REPORT_DAYS = {7: 12, 8: 12, 9: 12, 10: 11, 11: 9}
REPORT_MONTHS = [7, 8, 9, 10, 11]  # Jul is the pre-survey trend baseline; Aug-Nov are targets


def _mondays(start: dt.date, end: dt.date):
    d = start
    while d.weekday() != 0:  # 0 == Monday
        d += dt.timedelta(days=1)
    while d <= end:
        yield d
        d += dt.timedelta(days=7)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def build_rows():
    rng = random.Random(20260724)  # fixed seed -> reproducible fixtures
    yield_rows = []
    condition_rows = []

    for y in START_YEARS:
        market_year = f"{y}/{str(y + 1)[-2:]}"
        season = rng.gauss(0.0, 6.0)  # + good growing year, - bad year (latent)

        # Weekly crop-condition path, June 1 -> Nov 30, centered on normal + season.
        weekly = {}
        cur = NORMAL_GE + season + rng.gauss(0.0, 2.0)
        for monday in _mondays(dt.date(y, 6, 1), dt.date(y, 11, 30)):
            cur += rng.gauss(0.0, 1.5)  # random walk drift over the season
            cur = _clamp(cur, 30.0, 92.0)
            weekly[monday] = round(cur, 0)
            condition_rows.append(
                {
                    "week_ending": monday.isoformat(),
                    "state": "US TOTAL",
                    "metric": "PCT GOOD_EXCELLENT",
                    "value": f"{weekly[monday]:.0f}",
                }
            )

        # Yields: July is the trend baseline; each later report is revised from the
        # prior one by an amount tied to how the crop looks vs. normal, plus noise.
        trend = 158.0 + 1.4 * (y - 2013) + rng.gauss(0.0, 2.0)
        prev_val = trend
        for m in REPORT_MONTHS:
            report_date = dt.date(y, m, REPORT_DAYS[m])
            if m == 7:
                val = trend  # pre-survey trend guess, no condition input yet
            else:
                ge_at = _condition_before(weekly, report_date)
                signal = ge_at - NORMAL_GE
                revision = 0.16 * signal + rng.gauss(0.0, 0.9)
                val = prev_val + revision
            yield_rows.append(
                {
                    "Commodity": "Corn",
                    "Region": "United States",
                    "Attribute": "Yield per Harvested Acre",
                    "MarketYear": market_year,
                    "ReportDate": report_date.isoformat(),
                    "Value": f"{val:.1f}",
                    "Unit": "bu/acre",
                }
            )
            prev_val = val

    return yield_rows, condition_rows


def _condition_before(weekly: dict, report_date: dt.date) -> float:
    """Latest weekly reading strictly before the report date (point-in-time)."""
    candidates = [v for wk, v in weekly.items() if wk < report_date]
    return candidates[-1] if candidates else NORMAL_GE


def _write(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows):4d} rows -> {path.relative_to(path.parent.parent.parent)}")


def main():
    yield_rows, condition_rows = build_rows()
    _write(
        SAMPLE_DIR / "wasde_corn_yield_sample.csv",
        yield_rows,
        ["Commodity", "Region", "Attribute", "MarketYear", "ReportDate", "Value", "Unit"],
    )
    _write(
        SAMPLE_DIR / "crop_condition_sample.csv",
        condition_rows,
        ["week_ending", "state", "metric", "value"],
    )


if __name__ == "__main__":
    main()
