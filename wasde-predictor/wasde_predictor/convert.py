"""Turn real public downloads into the project's simple CSV schema.

USDA blocks automated downloads, so you fetch the files by hand and run these
converters. Each function is pure (list-of-dicts in, list-of-dicts out) so it's
easy to test; the thin CLI in tools/convert.py just does the file I/O.

Covered:
  * nass_condition_to_rows -- NASS Quick Stats crop-condition export -> our
    condition CSV (sums "% GOOD" + "% EXCELLENT" for the national total).
  * drought_to_rows        -- U.S. Drought Monitor export -> our drought CSV
    (% area in D2+).
  * pace_surprise_rows     -- a generic weekly level series -> a "vs. week-of-year
    normal" surprise (for the export/ethanol/crush demand clues).

The WASDE consolidated historical CSV needs NO conversion: point --wasde-file
straight at it (wasde.load_reports filters by commodity/attribute and tolerates
the real header names).
"""
from __future__ import annotations

import csv
import datetime as dt
from collections import defaultdict
from pathlib import Path


def _num(x) -> float:
    return float(str(x).replace(",", "").strip())


def _iso(x: str) -> str:
    """Normalize a date string (ISO, or YYYYMMDD, or MM/DD/YYYY) to ISO."""
    s = str(x).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    # already ISO-ish or unknown -> return as-is (fromisoformat will validate later)
    return s


def _is_national(r: dict) -> bool:
    if (r.get("agg_level_desc") or "").upper() == "NATIONAL":
        return True
    for k in ("state_name", "state_alpha", "state"):
        if (r.get(k) or "").upper() in ("US TOTAL", "UNITED STATES", "US"):
            return True
    return False


def nass_condition_to_rows(records: list[dict]) -> list[dict]:
    """NASS Quick Stats condition export -> [{week_ending, state, metric, value}].

    Keeps the national total and sums the PCT GOOD + PCT EXCELLENT categories for
    each week. Category is read from `unit_desc` (falls back to `short_desc`).
    """
    per_week: dict[str, dict[str, float]] = defaultdict(dict)
    for r in records:
        if not _is_national(r):
            continue
        unit = (r.get("unit_desc") or r.get("short_desc") or "").upper()
        wk = r.get("week_ending") or r.get("reference_period_desc")
        if not wk or "Value" not in r:
            continue
        try:
            val = _num(r["Value"])
        except ValueError:
            continue  # skip "(D)"/"(NA)" suppressed values
        if "EXCELLENT" in unit:
            per_week[_iso(wk)]["excellent"] = val
        elif "GOOD" in unit:
            per_week[_iso(wk)]["good"] = val

    out = []
    for wk in sorted(per_week):
        d = per_week[wk]
        if "good" in d and "excellent" in d:
            out.append({"week_ending": wk, "state": "US TOTAL",
                        "metric": "PCT GOOD_EXCELLENT", "value": f"{d['good'] + d['excellent']:.0f}"})
    return out


def drought_to_rows(records: list[dict], region: str = "US CORN BELT",
                    cumulative: bool = True) -> list[dict]:
    """U.S. Drought Monitor export -> [{week_ending, region, metric, value}].

    Value is % area in D2 or worse. USDM's standard "percent area" export is
    cumulative (the D2 column already means "D2 or worse"), so cumulative=True
    uses the D2 column; set cumulative=False to sum D2+D3+D4 for a non-overlapping
    export.
    """
    out = []
    for r in records:
        wk = r.get("ValidEnd") or r.get("MapDate") or r.get("week_ending")
        if not wk:
            continue
        d2 = _num(r.get("D2", 0) or 0)
        if cumulative:
            val = d2
        else:
            val = d2 + _num(r.get("D3", 0) or 0) + _num(r.get("D4", 0) or 0)
        out.append({"week_ending": _iso(wk), "region": region,
                    "metric": "PCT_AREA_D2PLUS", "value": f"{val:.0f}"})
    return out


def pace_surprise_rows(records: list[dict], date_col: str, value_col: str,
                       key_column: str, key: str, metric: str) -> list[dict]:
    """Weekly level series -> deviation from the week-of-year mean (a "surprise").

    Used to turn a raw weekly demand series (FAS export commitments, EIA ethanol
    grind, NOPA crush, ...) into the stationary "vs. normal pace" clue the model
    expects. The normal is the mean value for that ISO week across all rows.
    """
    parsed = []
    for r in records:
        try:
            d = _iso(r[date_col])
            v = _num(r[value_col])
            parsed.append((d, v))
        except (KeyError, ValueError):
            continue
    by_woy: dict[int, list[float]] = defaultdict(list)
    for d, v in parsed:
        by_woy[dt.date.fromisoformat(d).isocalendar()[1]].append(v)
    means = {w: sum(vs) / len(vs) for w, vs in by_woy.items()}
    out = []
    for d, v in sorted(parsed):
        woy = dt.date.fromisoformat(d).isocalendar()[1]
        out.append({"week_ending": d, key_column: key, "metric": metric,
                    "value": f"{v - means[woy]:.1f}"})
    return out


# --- file I/O helpers (used by tools/convert.py) ---------------------------

def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: str | Path, rows: list[dict], fieldnames: list[str]) -> None:
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
