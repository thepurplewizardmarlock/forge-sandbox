"""Generate SYNTHETIC sample data for the wasde-predictor project.

THIS IS NOT REAL USDA DATA. It is deterministic fake data whose only purpose is
to let the whole pipeline run and the tests pass without a network connection.
Replace the files in data/sample/ with real USDA downloads to get real results
(see README.md -> "Getting real data").

The generated CSVs deliberately mimic the *shape* of the real sources:
  * wasde_corn_yield_sample.csv  -> like USDA's Consolidated Historical WASDE CSV
    (Commodity, Region, Attribute, MarketYear, ReportDate, Value, Unit)
  * crop_condition_sample.csv    -> like NASS "% Good/Excellent" weekly readings
    (week_ending, state, metric, value)
  * drought_sample.csv           -> like the U.S. Drought Monitor weekly % area
    (week_ending, region, metric, value)   [value = % of corn area in D2+ drought]

A latent "season quality" per year drives the crop-condition path, the drought
path, AND the month-to-month yield revisions -- so healthier crop + less drought
tends to lead USDA to revise yield UP, just as in reality. Condition and drought
each carry independent noise, so both are individually useful clues.

Also emits a "current season" (no yield yet) so `cli.py predict-next` has an
upcoming report to forecast:
  * current_condition_sample.csv / current_drought_sample.csv

Run:  python3 tools/make_sample_data.py
"""
from __future__ import annotations

import csv
import datetime as dt
import random
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"

START_YEARS = list(range(2013, 2025))  # 2013/14 .. 2024/25 -> 12 historical seasons
CURRENT_YEAR = 2025                     # 2025/26 "in progress" season for predict-next
CURRENT_ASOF = dt.date(2025, 9, 5)      # pretend today is early September

NORMAL_GE = 64.0        # long-run "normal" % good/excellent
NORMAL_DROUGHT = 12.0   # long-run "normal" % of corn area in D2+ drought
REPORT_DAYS = {7: 12, 8: 12, 9: 12, 10: 11, 11: 9}
REPORT_MONTHS = [7, 8, 9, 10, 11]  # July = pre-survey trend baseline; Aug-Nov = targets


def _mondays(start: dt.date, end: dt.date):
    d = start
    while d.weekday() != 0:
        d += dt.timedelta(days=1)
    while d <= end:
        yield d
        d += dt.timedelta(days=7)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _season_paths(rng: random.Random, year: int, end: dt.date):
    """Return {monday: ge}, {monday: drought} for a season, sharing a latent."""
    season = rng.gauss(0.0, 6.0)  # + good year, - bad year
    ge, drought = {}, {}
    cur_ge = NORMAL_GE + season + rng.gauss(0.0, 2.0)
    cur_dr = NORMAL_DROUGHT - 0.7 * season + rng.gauss(0.0, 2.0)
    for monday in _mondays(dt.date(year, 6, 1), end):
        cur_ge = _clamp(cur_ge + rng.gauss(0.0, 1.5), 30.0, 92.0)
        cur_dr = _clamp(cur_dr + rng.gauss(0.0, 1.2), 0.0, 70.0)
        ge[monday] = round(cur_ge, 0)
        drought[monday] = round(cur_dr, 0)
    return season, ge, drought


def _before(series: dict, when: dt.date, default: float) -> float:
    vals = [v for wk, v in series.items() if wk < when]
    return vals[-1] if vals else default


def build():
    rng = random.Random(20260724)
    yield_rows, condition_rows, drought_rows = [], [], []

    def emit_condition(monday, value, state="US TOTAL"):
        condition_rows.append({"week_ending": monday.isoformat(), "state": state,
                               "metric": "PCT GOOD_EXCELLENT", "value": f"{value:.0f}"})

    def emit_drought(monday, value, region="US CORN BELT"):
        drought_rows.append({"week_ending": monday.isoformat(), "region": region,
                             "metric": "PCT_AREA_D2PLUS", "value": f"{value:.0f}"})

    for y in START_YEARS:
        market_year = f"{y}/{str(y + 1)[-2:]}"
        _, ge, drought = _season_paths(rng, y, dt.date(y, 11, 30))
        for monday in sorted(ge):
            emit_condition(monday, ge[monday])
            emit_drought(monday, drought[monday])

        trend = 158.0 + 1.4 * (y - 2013) + rng.gauss(0.0, 2.0)
        prev = trend
        for m in REPORT_MONTHS:
            report_date = dt.date(y, m, REPORT_DAYS[m])
            if m == 7:
                val = trend
            else:
                ge_at = _before(ge, report_date, NORMAL_GE)
                dr_at = _before(drought, report_date, NORMAL_DROUGHT)
                revision = 0.12 * (ge_at - NORMAL_GE) - 0.07 * (dr_at - NORMAL_DROUGHT) + rng.gauss(0.0, 0.7)
                val = prev + revision
            yield_rows.append({"Commodity": "Corn", "Region": "United States",
                               "Attribute": "Yield per Harvested Acre", "MarketYear": market_year,
                               "ReportDate": report_date.isoformat(), "Value": f"{val:.1f}",
                               "Unit": "bu/acre"})
            prev = val

    # Current in-progress season (condition + drought only, no yield yet).
    _, ge_cur, dr_cur = _season_paths(rng, CURRENT_YEAR, CURRENT_ASOF)
    cur_condition = [{"week_ending": wk.isoformat(), "state": "US TOTAL",
                      "metric": "PCT GOOD_EXCELLENT", "value": f"{v:.0f}"} for wk, v in sorted(ge_cur.items())]
    cur_drought = [{"week_ending": wk.isoformat(), "region": "US CORN BELT",
                    "metric": "PCT_AREA_D2PLUS", "value": f"{v:.0f}"} for wk, v in sorted(dr_cur.items())]

    return yield_rows, condition_rows, drought_rows, cur_condition, cur_drought


def _write(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows):4d} rows -> {path.name}")


def main():
    yr, cond, dr, cur_c, cur_d = build()
    _write(SAMPLE_DIR / "wasde_corn_yield_sample.csv", yr,
           ["Commodity", "Region", "Attribute", "MarketYear", "ReportDate", "Value", "Unit"])
    _write(SAMPLE_DIR / "crop_condition_sample.csv", cond, ["week_ending", "state", "metric", "value"])
    _write(SAMPLE_DIR / "drought_sample.csv", dr, ["week_ending", "region", "metric", "value"])
    _write(SAMPLE_DIR / "current_condition_sample.csv", cur_c, ["week_ending", "state", "metric", "value"])
    _write(SAMPLE_DIR / "current_drought_sample.csv", cur_d, ["week_ending", "region", "metric", "value"])


if __name__ == "__main__":
    main()
