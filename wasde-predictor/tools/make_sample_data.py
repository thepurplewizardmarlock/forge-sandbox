"""Generate SYNTHETIC sample data for the wasde-predictor project.

THIS IS NOT REAL USDA DATA. It is deterministic fake data whose only purpose is
to let the whole pipeline run and the tests pass without a network connection.
Replace the files in data/sample/ (or point --data-dir at real downloads with the
same names) to get real results -- see README.md -> "Getting real data".

Per commodity {corn, soybeans, wheat} it writes (basenames, no "_sample" suffix,
so the same names work for a real data dir):
  * wasde_<c>.csv         -> USDA Consolidated Historical WASDE CSV shape
                             (Yield per Harvested Acre + Ending Stocks)
  * condition_<c>.csv     -> NASS weekly "% good/excellent"
  * drought_<c>.csv       -> Drought Monitor weekly "% area in D2+"
  * <demand clue>.csv     -> per commodity (exports for all; ethanol for corn;
                             crush for soybeans) as a "vs. normal pace" surprise
  * current_*.csv         -> an in-progress 2025/26 season (no yield yet) so
                             `cli.py predict-next` has an upcoming report.

Latent drivers: a per-year "season quality" moves condition, drought AND yield;
independent per-year strengths move each demand clue. Ending stocks are built from
the yield revision (supply) minus the demand surprises, so both clue groups carry
real signal. The commodity/demand-clue STRUCTURE comes from wasde_predictor.commodities;
only the generative parameters live here.

Run:  python3 tools/make_sample_data.py
"""
from __future__ import annotations

import csv
import datetime as dt
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wasde_predictor import commodities  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
START_YEARS = list(range(2013, 2025))
CURRENT_YEAR = 2025
CURRENT_ASOF = dt.date(2025, 9, 5)
NORMAL_DROUGHT = 12.0
REPORT_DAYS = {5: 12, 6: 11, 7: 12, 8: 12, 9: 12, 10: 11, 11: 9}

# Generative parameters (synthetic only). Structure (which clues) is in commodities.py.
COMMODITY_GEN = {
    "corn":     {"base_yield": 158.0, "yslope": 1.4, "ynoise": 0.7, "ge_normal": 64.0, "es_base": 1800.0, "es_ycoef": 55.0},
    "soybeans": {"base_yield": 46.0,  "yslope": 0.4, "ynoise": 0.4, "ge_normal": 60.0, "es_base": 350.0,  "es_ycoef": 15.0},
    "wheat":    {"base_yield": 47.0,  "yslope": 0.3, "ynoise": 0.5, "ge_normal": 54.0, "es_base": 700.0,  "es_ycoef": 20.0},
}
DEMAND_GEN = {  # keyed by clue.feature
    "export_pace_surprise":  {"sigma": 6.0, "walk": 1.0, "lo": -25.0, "hi": 25.0, "es_coef": 4.0},
    "ethanol_pace_surprise": {"sigma": 4.0, "walk": 0.8, "lo": -20.0, "hi": 20.0, "es_coef": 3.0},
    "crush_pace_surprise":   {"sigma": 4.0, "walk": 0.8, "lo": -20.0, "hi": 20.0, "es_coef": 3.5},
}


def _mondays(start, end):
    d = start
    while d.weekday() != 0:
        d += dt.timedelta(days=1)
    while d <= end:
        yield d
        d += dt.timedelta(days=7)


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _before(series, when, default):
    vals = [v for wk, v in series.items() if wk < when]
    return vals[-1] if vals else default


def _condition_drought(rng, year, end, ge_normal):
    season = rng.gauss(0.0, 6.0)
    ge, dr = {}, {}
    cur_ge = ge_normal + season + rng.gauss(0.0, 2.0)
    cur_dr = NORMAL_DROUGHT - 0.7 * season + rng.gauss(0.0, 2.0)
    for m in _mondays(dt.date(year, 6, 1), end):
        cur_ge = _clamp(cur_ge + rng.gauss(0.0, 1.5), 30.0, 92.0)
        cur_dr = _clamp(cur_dr + rng.gauss(0.0, 1.2), 0.0, 70.0)
        ge[m] = round(cur_ge, 0)
        dr[m] = round(cur_dr, 0)
    return ge, dr


def _demand_series(rng, year, end, feature):
    p = DEMAND_GEN[feature]
    cur = rng.gauss(0.0, p["sigma"])
    s = {}
    for m in _mondays(dt.date(year, 6, 1), end):
        cur = _clamp(cur + rng.gauss(0.0, p["walk"]), p["lo"], p["hi"])
        s[m] = round(cur, 1)
    return s


def _write(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows):4d} rows -> {path.name}")


def _weekly_rows(series_by_week, key_column, key, metric):
    return [{"week_ending": wk.isoformat(), key_column: key, "metric": metric, "value": f"{v}"}
            for wk, v in sorted(series_by_week.items())]


def generate_commodity(rng, c: commodities.Commodity):
    gen = COMMODITY_GEN[c.slug]
    ge_normal = gen["ge_normal"]
    months = sorted(set([min(c.report_months) - 1, *c.report_months]))  # + a baseline report
    wasde_rows = []

    def wrow(attr, my, date, val, unit):
        wasde_rows.append({"Commodity": c.label, "Region": "United States", "Attribute": attr,
                           "MarketYear": my, "ReportDate": date.isoformat(), "Value": val, "Unit": unit})

    cond_rows, dr_rows = [], []
    demand_rows = {clue.basename: [] for clue in c.demand_clues}

    for y in START_YEARS:
        my = f"{y}/{str(y + 1)[-2:]}"
        ge, dr = _condition_drought(rng, y, dt.date(y, 11, 30), ge_normal)
        demand = {clue: _demand_series(rng, y, dt.date(y, 11, 30), clue.feature) for clue in c.demand_clues}
        cond_rows += _weekly_rows(ge, "state", "US TOTAL", "PCT GOOD_EXCELLENT")
        dr_rows += _weekly_rows(dr, "region", "US CORN BELT", "PCT_AREA_D2PLUS")
        for clue in c.demand_clues:
            demand_rows[clue.basename] += _weekly_rows(demand[clue], clue.key_column, clue.key, clue.feature.upper())

        trend = gen["base_yield"] + gen["yslope"] * (y - 2013) + rng.gauss(0.0, 2.0)
        prev_yield, prev_es = trend, gen["es_base"] + rng.gauss(0.0, 0.08 * gen["es_base"])
        for m in months:
            date = dt.date(y, m, REPORT_DAYS[m])
            if m == months[0]:
                yld, es = trend, prev_es
            else:
                ge_at, dr_at = _before(ge, date, ge_normal), _before(dr, date, NORMAL_DROUGHT)
                yld = prev_yield + 0.12 * (ge_at - ge_normal) - 0.07 * (dr_at - NORMAL_DROUGHT) + rng.gauss(0.0, gen["ynoise"])
                es = prev_es + gen["es_ycoef"] * (yld - prev_yield)
                for clue in c.demand_clues:
                    es -= DEMAND_GEN[clue.feature]["es_coef"] * _before(demand[clue], date, 0.0)
                es += rng.gauss(0.0, 20.0)
            wrow("Yield per Harvested Acre", my, date, f"{yld:.1f}", "bu/acre")
            wrow("Ending Stocks", my, date, f"{es:.0f}", "million bushels")
            prev_yield, prev_es = yld, es

    _write(SAMPLE_DIR / f"wasde_{c.slug}.csv", wasde_rows,
           ["Commodity", "Region", "Attribute", "MarketYear", "ReportDate", "Value", "Unit"])
    _write(SAMPLE_DIR / f"condition_{c.slug}.csv", cond_rows, ["week_ending", "state", "metric", "value"])
    _write(SAMPLE_DIR / f"drought_{c.slug}.csv", dr_rows, ["week_ending", "region", "metric", "value"])
    for clue in c.demand_clues:
        _write(SAMPLE_DIR / f"{clue.basename}.csv", demand_rows[clue.basename],
               ["week_ending", clue.key_column, "metric", "value"])

    # Current in-progress season (clues only).
    ge_c, dr_c = _condition_drought(rng, CURRENT_YEAR, CURRENT_ASOF, ge_normal)
    _write(SAMPLE_DIR / f"current_condition_{c.slug}.csv",
           _weekly_rows(ge_c, "state", "US TOTAL", "PCT GOOD_EXCELLENT"),
           ["week_ending", "state", "metric", "value"])
    _write(SAMPLE_DIR / f"current_drought_{c.slug}.csv",
           _weekly_rows(dr_c, "region", "US CORN BELT", "PCT_AREA_D2PLUS"),
           ["week_ending", "region", "metric", "value"])
    for clue in c.demand_clues:
        s = _demand_series(rng, CURRENT_YEAR, CURRENT_ASOF, clue.feature)
        _write(SAMPLE_DIR / f"current_{clue.basename}.csv",
               _weekly_rows(s, clue.key_column, clue.key, clue.feature.upper()),
               ["week_ending", clue.key_column, "metric", "value"])


def main():
    rng = random.Random(20260724)
    for c in commodities.ALL.values():
        generate_commodity(rng, c)


if __name__ == "__main__":
    main()
