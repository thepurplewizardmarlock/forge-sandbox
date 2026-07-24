"""Generate SYNTHETIC sample data for the wasde-predictor project.

THIS IS NOT REAL USDA DATA. It is deterministic fake data whose only purpose is
to let the whole pipeline run and the tests pass without a network connection.
Replace the files in data/sample/ with real USDA downloads to get real results
(see README.md -> "Getting real data").

Per commodity {corn, soybeans, wheat} it writes:
  * wasde_<c>_sample.csv        -> USDA Consolidated Historical WASDE CSV shape
  * condition_<c>_sample.csv    -> NASS weekly "% good/excellent"
  * drought_<c>_sample.csv      -> Drought Monitor weekly "% area in D2+"
  * current_condition_<c>_sample.csv / current_drought_<c>_sample.csv  (in-progress season)

Corn additionally gets the demand clues used for the ending-stocks target:
  * exports_corn_sample.csv / ethanol_corn_sample.csv (+ current_* versions)
  and an "Ending Stocks" attribute inside wasde_corn_sample.csv.

Wheat is INCLUDED BUT APPROXIMATE: real wheat yields are largely settled by the
late-September Small Grains Summary, so the Aug-Nov row-crop window used here is
a simplification (see README). Corn and soybeans fit this window well.

Run:  python3 tools/make_sample_data.py
"""
from __future__ import annotations

import csv
import datetime as dt
import random
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
START_YEARS = list(range(2013, 2025))
CURRENT_YEAR = 2025
CURRENT_ASOF = dt.date(2025, 9, 5)
NORMAL_DROUGHT = 12.0
BASE_ENDING_STOCKS = 1800.0
REPORT_DAYS = {7: 12, 8: 12, 9: 12, 10: 11, 11: 9}
REPORT_MONTHS = [7, 8, 9, 10, 11]

# Per-commodity parameters. `demand` (ending stocks + exports/ethanol) is corn-only.
COMMODITIES = {
    "corn":     {"base_yield": 158.0, "yslope": 1.4, "ynoise": 0.7, "ge_normal": 64.0, "demand": True},
    "soybeans": {"base_yield": 46.0,  "yslope": 0.4, "ynoise": 0.4, "ge_normal": 60.0, "demand": False},
    "wheat":    {"base_yield": 47.0,  "yslope": 0.3, "ynoise": 0.5, "ge_normal": 54.0, "demand": False},
}
COMMODITY_LABEL = {"corn": "Corn", "soybeans": "Soybeans", "wheat": "Wheat"}


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


def _paths(rng, year, end, ge_normal, demand):
    season = rng.gauss(0.0, 6.0)
    ge, dr = {}, {}
    exp, eth = {}, {}
    cur_ge = ge_normal + season + rng.gauss(0.0, 2.0)
    cur_dr = NORMAL_DROUGHT - 0.7 * season + rng.gauss(0.0, 2.0)
    cur_exp = rng.gauss(0.0, 6.0)
    cur_eth = rng.gauss(0.0, 4.0)
    for monday in _mondays(dt.date(year, 6, 1), end):
        cur_ge = _clamp(cur_ge + rng.gauss(0.0, 1.5), 30.0, 92.0)
        cur_dr = _clamp(cur_dr + rng.gauss(0.0, 1.2), 0.0, 70.0)
        ge[monday] = round(cur_ge, 0)
        dr[monday] = round(cur_dr, 0)
        if demand:
            cur_exp = _clamp(cur_exp + rng.gauss(0.0, 1.0), -25.0, 25.0)
            cur_eth = _clamp(cur_eth + rng.gauss(0.0, 0.8), -20.0, 20.0)
            exp[monday] = round(cur_exp, 1)
            eth[monday] = round(cur_eth, 1)
    return ge, dr, exp, eth


def _write(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows):4d} rows -> {path.name}")


def generate_commodity(rng, key):
    cfg = COMMODITIES[key]
    label = COMMODITY_LABEL[key]
    ge_normal = cfg["ge_normal"]
    wasde_rows, cond_rows, dr_rows, exp_rows, eth_rows = [], [], [], [], []

    def wrow(attr, my, date, val, unit):
        wasde_rows.append({"Commodity": label, "Region": "United States", "Attribute": attr,
                           "MarketYear": my, "ReportDate": date.isoformat(), "Value": val, "Unit": unit})

    for y in START_YEARS:
        my = f"{y}/{str(y + 1)[-2:]}"
        ge, dr, exp, eth = _paths(rng, y, dt.date(y, 11, 30), ge_normal, cfg["demand"])
        for wk in sorted(ge):
            cond_rows.append({"week_ending": wk.isoformat(), "state": "US TOTAL",
                              "metric": "PCT GOOD_EXCELLENT", "value": f"{ge[wk]:.0f}"})
            dr_rows.append({"week_ending": wk.isoformat(), "region": "US CORN BELT",
                            "metric": "PCT_AREA_D2PLUS", "value": f"{dr[wk]:.0f}"})
            if cfg["demand"]:
                exp_rows.append({"week_ending": wk.isoformat(), "commodity": "CORN",
                                 "metric": "EXPORT_PACE_SURPRISE", "value": f"{exp[wk]:.1f}"})
                eth_rows.append({"week_ending": wk.isoformat(), "region": "US",
                                 "metric": "ETHANOL_PACE_SURPRISE", "value": f"{eth[wk]:.1f}"})

        trend = cfg["base_yield"] + cfg["yslope"] * (y - 2013) + rng.gauss(0.0, 2.0)
        prev_yield, prev_es = trend, BASE_ENDING_STOCKS + rng.gauss(0.0, 150.0)
        for m in REPORT_MONTHS:
            date = dt.date(y, m, REPORT_DAYS[m])
            if m == 7:
                yld, es = trend, prev_es
            else:
                ge_at, dr_at = _before(ge, date, ge_normal), _before(dr, date, NORMAL_DROUGHT)
                yld = prev_yield + 0.12 * (ge_at - ge_normal) - 0.07 * (dr_at - NORMAL_DROUGHT) + rng.gauss(0.0, cfg["ynoise"])
                if cfg["demand"]:
                    exp_at, eth_at = _before(exp, date, 0.0), _before(eth, date, 0.0)
                    es = prev_es + 55.0 * (yld - prev_yield) - 4.0 * exp_at - 3.0 * eth_at + rng.gauss(0.0, 20.0)
                else:
                    es = prev_es
            wrow("Yield per Harvested Acre", my, date, f"{yld:.1f}", "bu/acre")
            if cfg["demand"]:
                wrow("Ending Stocks", my, date, f"{es:.0f}", "million bushels")
            prev_yield, prev_es = yld, es

    _write(SAMPLE_DIR / f"wasde_{key}_sample.csv", wasde_rows,
           ["Commodity", "Region", "Attribute", "MarketYear", "ReportDate", "Value", "Unit"])
    _write(SAMPLE_DIR / f"condition_{key}_sample.csv", cond_rows, ["week_ending", "state", "metric", "value"])
    _write(SAMPLE_DIR / f"drought_{key}_sample.csv", dr_rows, ["week_ending", "region", "metric", "value"])
    if cfg["demand"]:
        _write(SAMPLE_DIR / "exports_corn_sample.csv", exp_rows, ["week_ending", "commodity", "metric", "value"])
        _write(SAMPLE_DIR / "ethanol_corn_sample.csv", eth_rows, ["week_ending", "region", "metric", "value"])

    # Current in-progress season (clues only).
    ge_c, dr_c, exp_c, eth_c = _paths(rng, CURRENT_YEAR, CURRENT_ASOF, ge_normal, cfg["demand"])
    _write(SAMPLE_DIR / f"current_condition_{key}_sample.csv",
           [{"week_ending": wk.isoformat(), "state": "US TOTAL", "metric": "PCT GOOD_EXCELLENT",
             "value": f"{v:.0f}"} for wk, v in sorted(ge_c.items())],
           ["week_ending", "state", "metric", "value"])
    _write(SAMPLE_DIR / f"current_drought_{key}_sample.csv",
           [{"week_ending": wk.isoformat(), "region": "US CORN BELT", "metric": "PCT_AREA_D2PLUS",
             "value": f"{v:.0f}"} for wk, v in sorted(dr_c.items())],
           ["week_ending", "region", "metric", "value"])
    if cfg["demand"]:
        _write(SAMPLE_DIR / "current_exports_corn_sample.csv",
               [{"week_ending": wk.isoformat(), "commodity": "CORN", "metric": "EXPORT_PACE_SURPRISE",
                 "value": f"{v:.1f}"} for wk, v in sorted(exp_c.items())],
               ["week_ending", "commodity", "metric", "value"])
        _write(SAMPLE_DIR / "current_ethanol_corn_sample.csv",
               [{"week_ending": wk.isoformat(), "region": "US", "metric": "ETHANOL_PACE_SURPRISE",
                 "value": f"{v:.1f}"} for wk, v in sorted(eth_c.items())],
               ["week_ending", "region", "metric", "value"])


def main():
    rng = random.Random(20260724)
    for key in COMMODITIES:
        generate_commodity(rng, key)


if __name__ == "__main__":
    main()
