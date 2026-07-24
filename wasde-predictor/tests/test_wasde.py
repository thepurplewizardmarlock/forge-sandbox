"""Tests for the yield target loader and revision computation."""
import datetime as dt
import unittest
from pathlib import Path

from wasde_predictor import wasde

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_YIELD = PROJECT_ROOT / "data" / "sample" / "wasde_corn_sample.csv"


def _report(month, value, year=2024, my="2024/25"):
    return wasde.YieldReport(market_year=my, report_date=dt.date(year, month, 10), value=value)


class RevisionTests(unittest.TestCase):
    def test_only_target_months_emitted(self):
        reports = [
            _report(7, 180.0),  # July trend baseline (not a target)
            _report(8, 182.0),
            _report(9, 181.0),
            _report(10, 181.0),
            _report(11, 183.0),
        ]
        revs = wasde.revisions(reports)
        months = [r.month for r in revs]
        self.assertEqual(months, [8, 9, 10, 11])  # July excluded, Aug-Nov kept

    def test_august_is_measured_against_july(self):
        reports = [_report(7, 180.0), _report(8, 182.5)]
        (aug,) = wasde.revisions(reports)
        self.assertEqual(aug.prev_value, 180.0)
        self.assertAlmostEqual(aug.change, 2.5)
        self.assertEqual(aug.direction, 1)
        self.assertEqual(aug.label, "up")

    def test_down_and_flat_directions(self):
        down = wasde.revisions([_report(7, 180.0), _report(8, 179.0)])[0]
        self.assertEqual(down.direction, 0)
        self.assertEqual(down.label, "down")
        flat = wasde.revisions([_report(7, 180.0), _report(8, 180.0)])[0]
        self.assertEqual(flat.direction, 0)  # unchanged counts as "not up"
        self.assertEqual(flat.label, "flat")

    def test_years_do_not_bleed_into_each_other(self):
        reports = [
            _report(11, 200.0, year=2023, my="2023/24"),
            _report(8, 180.0, year=2024, my="2024/25"),
        ]
        # 2024 August must compare within 2024, not against 2023's November.
        revs = wasde.revisions(reports)
        self.assertEqual(revs, [])  # no prior 2024 report exists, so nothing to revise from


class LoaderTests(unittest.TestCase):
    def test_loads_corn_us_yield_from_sample(self):
        reports = wasde.load_yield_reports(SAMPLE_YIELD)
        self.assertTrue(reports)
        self.assertTrue(all(150 < r.value < 210 for r in reports))  # sane corn yields
        # sorted ascending by date
        self.assertEqual(reports, sorted(reports, key=lambda r: r.report_date))

    def test_missing_column_raises_clear_error(self):
        with self.assertRaises(ValueError):
            wasde._resolve_headers(["Commodity", "Region", "Value"])  # no attribute/date/year


if __name__ == "__main__":
    unittest.main()
