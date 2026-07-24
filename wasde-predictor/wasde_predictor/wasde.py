"""Load the target: USDA's corn yield estimates, and the month-to-month revisions.

The real source is USDA's "Consolidated Historical WASDE Report Data" CSV, which
stores each value *as first published* (exactly the point-in-time property we
need). We read it defensively: the exact header spellings should be confirmed
once against a real download, so column lookup is case-insensitive and accepts a
few candidate names.

A "revision" is what we predict: for a given report, how did USDA's yield number
change versus the previous report in the same marketing year? We only keep the
August-November reports, because that is the window where USDA switches from a
fixed trend guess to survey-based yields (see README).
"""
from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

# Candidate header names, lowercased. Real USDA headers should be confirmed once.
_COLUMNS = {
    "commodity": ["commodity"],
    "region": ["region"],
    "attribute": ["attribute"],
    "market_year": ["marketyear", "market_year", "marketing_year"],
    "report_date": ["reportdate", "report_date", "date"],
    "value": ["value"],
}

# The August-November reports are our prediction targets.
TARGET_MONTHS = (8, 9, 10, 11)


@dataclass(frozen=True)
class YieldReport:
    market_year: str          # e.g. "2024/25"
    report_date: dt.date
    value: float              # bushels per harvested acre

    @property
    def month(self) -> int:
        return self.report_date.month


@dataclass(frozen=True)
class Revision:
    """One month's change in USDA's yield estimate for a marketing year."""
    market_year: str
    report_date: dt.date
    month: int
    prev_value: float
    value: float
    prev_report_date: dt.date | None = None

    @property
    def change(self) -> float:
        return round(self.value - self.prev_value, 4)

    @property
    def direction(self) -> int:
        """1 if USDA raised the yield, else 0 (lowered or unchanged)."""
        return 1 if self.change > 0 else 0

    @property
    def label(self) -> str:
        if self.change > 0:
            return "up"
        if self.change < 0:
            return "down"
        return "flat"


def _resolve_headers(fieldnames: list[str]) -> dict[str, str]:
    lower = {name.lower().strip(): name for name in fieldnames}
    resolved: dict[str, str] = {}
    for key, candidates in _COLUMNS.items():
        for cand in candidates:
            if cand in lower:
                resolved[key] = lower[cand]
                break
        else:
            raise ValueError(
                f"Could not find a column for '{key}' in {fieldnames!r}. "
                f"Tried {candidates!r}. Confirm the real USDA header names and "
                f"update wasde._COLUMNS."
            )
    return resolved


def load_reports(
    path: str | Path,
    commodity: str = "Corn",
    region: str = "United States",
    attribute: str = "Yield per Harvested Acre",
) -> list[YieldReport]:
    """Read a WASDE-style CSV and return one attribute's reports, date-sorted.

    `attribute` selects the balance-sheet line: "Yield per Harvested Acre" (the
    default) or "Ending Stocks", etc. Despite the class name YieldReport, this
    returns whatever attribute you ask for (value carries that attribute's unit).
    """
    path = Path(path)
    reports: list[YieldReport] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        cols = _resolve_headers(reader.fieldnames or [])
        for row in reader:
            if row[cols["commodity"]].strip().lower() != commodity.lower():
                continue
            if row[cols["region"]].strip().lower() != region.lower():
                continue
            if row[cols["attribute"]].strip().lower() != attribute.lower():
                continue
            reports.append(
                YieldReport(
                    market_year=row[cols["market_year"]].strip(),
                    report_date=dt.date.fromisoformat(row[cols["report_date"]].strip()),
                    value=float(row[cols["value"]]),
                )
            )
    reports.sort(key=lambda r: r.report_date)
    return reports


# Backward-compatible alias (v1 called this load_yield_reports).
load_yield_reports = load_reports


def revisions(
    reports: list[YieldReport], target_months: tuple[int, ...] = TARGET_MONTHS
) -> list[Revision]:
    """Compute month-to-month yield revisions, keeping only the target months.

    Within each marketing year the reports are ordered by date; each target
    report is compared with the immediately preceding report of the same year
    (so August is measured against July's trend guess).
    """
    by_year: dict[str, list[YieldReport]] = {}
    for r in reports:
        by_year.setdefault(r.market_year, []).append(r)

    out: list[Revision] = []
    for _, year_reports in by_year.items():
        year_reports = sorted(year_reports, key=lambda r: r.report_date)
        for prev, cur in zip(year_reports, year_reports[1:]):
            if cur.month in target_months:
                out.append(
                    Revision(
                        market_year=cur.market_year,
                        report_date=cur.report_date,
                        month=cur.month,
                        prev_value=prev.value,
                        value=cur.value,
                        prev_report_date=prev.report_date,
                    )
                )
    out.sort(key=lambda rev: rev.report_date)
    return out
