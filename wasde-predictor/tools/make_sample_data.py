"""Generate SYNTHETIC sample data for the wasde-predictor project.

THIS IS NOT REAL USDA DATA. It is deterministic fake data whose only purpose is
to let the whole pipeline run and the tests pass without a network connection.
Replace the files in data/sample/ with real USDA downloads to get real results
(see README.md -> "Getting real data").

Files it writes (all mimic the SHAPE of the real sources):
  * wasde_corn_sample.csv       -> USDA Consolidated Historical WASDE CSV. Now
    carries TWO attributes: "Yield per Harvested Acre" and "Ending Stocks".
  * crop_condition_sample.csv   -> NASS weekly "% good/excellent"
  * drought_sample.csv          -> Drought Monitor weekly "% corn area in D2+"
  * exports_sample.csv          -> weekly export-commitment pace surprise (pct pts
    vs. the normal pace; + = selling faster than normal -> fewer ending stocks)
  * ethanol_sample.csv          -> weekly ethanol-grind pace surprise (+ = grinding
    faster than needed -> fewer ending stocks)
  * current_*_sample.csv        -> an in-progress 2025/26 season (no yield yet) so
    `cli.py predict-next` has an upcoming report to forecast.

Latent drivers: a per-year "season quality" moves condition, drought AND yield;
independent per-year export/ethanol strengths move the demand clues. Ending
stocks are then built from the yield revision (supply) minus the demand
surprises -- so BOTH the supply clues and the demand clues carry real signal for
the ending-stocks target, just like reality.

Run:  python3 tools/make_sample_data.py
"""
from __future__ import annotations

import csv
import datetime as dt
import random
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"

START_YEARS = list(range(2013, 2025))  # 12 historical seasons
CURRENT_YEAR = 2025
CURRENT_ASOF = dt.date(2025, 9, 5)

NORMAL_GE = 64.0
NORMAL_DROUGHT = 12.0
BASE_ENDING_STOCKS = 1800.0  # million bushels, a plausible corn carryout
REPORT_DAYS = {7: 12, 8: 12, 9: 12, 10: 11, 11: 9}
REPORT_MONTHS = [7, 8, 9, 10, 11]


def _mondays(start: dt.date, end: dt.date):
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


def _paths(rng, year, end):
    """Weekly condition, drought, export-surprise, ethanol-surprise for a season."""
    season = rng.gauss(0.0, 6.0)
    export_strength = rng.gauss(0.0, 6.0)
    ethanol_strength = rng.gauss(0.0, 4.0)
    ge, dr, exp, eth = {}, {}, {}, {}
    cur_ge = NORMAL_GE + season + rng.gauss(0.0, 2.0)
    cur_dr = NORMAL_DROUGHT - 0.7 * season + rng.gauss(0.0, 2.0)
    cur_exp = export_strength
    cur_eth = ethanol_strength
    for monday in _mondays(dt.date(year, 6, 1), end):
        cur_ge = _clamp(cur_ge + rng.gauss(0.0, 1.5), 30.0, 92.0)
        cur_dr = _clamp(cur_dr + rng.gauss(0.0, 1.2), 0.0, 70.0)
        cur_exp = _clamp(cur_exp + rng.gauss(0.0, 1.0), -25.0, 25.0)
        cur_eth = _clamp(cur_eth + rng.gauss(0.0, 0.8), -20.0, 20.0)
        ge[monday] = round(cur_ge, 0)
        dr[monday] = round(cur_dr, 0)
        exp[monday] = round(cur_exp, 1)
        eth[monday] = round(cur_eth, 1)
    return ge, dr, exp, eth


def build():
    rng = random.Random(20260724)
    wasde_rows, cond_rows, dr_rows, exp_rows, eth_rows = [], [], [], [], []

    def wasde_row(attr, my, date, val, unit):
        wasde_rows.append({"Commodity": "Corn", "Region": "United States", "Attribute": attr,
                           "MarketYear": my, "ReportDate": date.isoformat(),
                           "Value": val, "Unit": unit})

    for y in START_YEARS:
        my = f"{y}/{str(y + 1)[-2:]}"
        ge, dr, exp, eth = _paths(rng, y, dt.date(y, 11, 30))
        for wk in sorted(ge):
            cond_rows.append({"week_ending": wk.isoformat(), "state": "US TOTAL",
                              "metric": "PCT GOOD_EXCELLENT", "value": f"{ge[wk]:.0f}"})
            dr_rows.append({"week_ending": wk.isoformat(), "region": "US CORN BELT",
                            "metric": "PCT_AREA_D2PLUS", "value": f"{dr[wk]:.0f}"})
            exp_rows.append({"week_ending": wk.isoformat(), "commodity": "CORN",
                             "metric": "EXPORT_PACE_SURPRISE", "value": f"{exp[wk]:.1f}"})
            eth_rows.append({"week_ending": wk.isoformat(), "region": "US",
                             "metric": "ETHANOL_PACE_SURPRISE", "value": f"{eth[wk]:.1f}"})

        trend = 158.0 + 1.4 * (y - 2013) + rng.gauss(0.0, 2.0)
        prev_yield = trend
        prev_es = BASE_ENDING_STOCKS + rng.gauss(0.0, 150.0)
        for m in REPORT_MONTHS:
            date = dt.date(y, m, REPORT_DAYS[m])
            if m == 7:
                yld = trend
                es = prev_es
            else:
                ge_at = _before(ge, date, NORMAL_GE)
                dr_at = _before(dr, date, NORMAL_DROUGHT)
                exp_at = _before(exp, date, 0.0)
                eth_at = _before(eth, date, 0.0)
                yld = prev_yield + 0.12 * (ge_at - NORMAL_GE) - 0.07 * (dr_at - NORMAL_DROUGHT) + rng.gauss(0.0, 0.7)
                yield_rev = yld - prev_yield
                es = prev_es + 55.0 * yield_rev - 4.0 * exp_at - 3.0 * eth_at + rng.gauss(0.0, 20.0)
            wasde_row("Yield per Harvested Acre", my, date, f"{yld:.1f}", "bu/acre")
            wasde_row("Ending Stocks", my, date, f"{es:.0f}", "million bushels")
            prev_yield, prev_es = yld, es

    # Current in-progress season (clues only, no yield/ending stocks yet).
    ge_c, dr_c, exp_c, eth_c = _paths(rng, CURRENT_YEAR, CURRENT_ASOF)
    cur = {
        "current_condition_sample.csv": ([{"week_ending": wk.isoformat(), "state": "US TOTAL",
            "metric": "PCT GOOD_EXCELLENT", "value": f"{v:.0f}"} for wk, v in sorted(ge_c.items())],
            ["week_ending", "state", "metric", "value"]),
        "current_drought_sample.csv": ([{"week_ending": wk.isoformat(), "region": "US CORN BELT",
            "metric": "PCT_AREA_D2PLUS", "value": f"{v:.0f}"} for wk, v in sorted(dr_c.items())],
            ["week_ending", "region", "metric", "value"]),
        "current_exports_sample.csv": ([{"week_ending": wk.isoformat(), "commodity": "CORN",
            "metric": "EXPORT_PACE_SURPRISE", "value": f"{v:.1f}"} for wk, v in sorted(exp_c.items())],
            ["week_ending", "commodity", "metric", "value"]),
        "current_ethanol_sample.csv": ([{"week_ending": wk.isoformat(), "region": "US",
            "metric": "ETHANOL_PACE_SURPRISE", "value": f"{v:.1f}"} for wk, v in sorted(eth_c.items())],
            ["week_ending", "region", "metric", "value"]),
    }
    return wasde_rows, cond_rows, dr_rows, exp_rows, eth_rows, cur


def _write(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows):4d} rows -> {path.name}")


def main():
    wasde_rows, cond, dr, exp, eth, cur = build()
    _write(SAMPLE_DIR / "wasde_corn_sample.csv", wasde_rows,
           ["Commodity", "Region", "Attribute", "MarketYear", "ReportDate", "Value", "Unit"])
    _write(SAMPLE_DIR / "crop_condition_sample.csv", cond, ["week_ending", "state", "metric", "value"])
    _write(SAMPLE_DIR / "drought_sample.csv", dr, ["week_ending", "region", "metric", "value"])
    _write(SAMPLE_DIR / "exports_sample.csv", exp, ["week_ending", "commodity", "metric", "value"])
    _write(SAMPLE_DIR / "ethanol_sample.csv", eth, ["week_ending", "region", "metric", "value"])
    for name, (rows, fields) in cur.items():
        _write(SAMPLE_DIR / name, rows, fields)


if __name__ == "__main__":
    main()
