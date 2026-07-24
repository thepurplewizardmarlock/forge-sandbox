"""Tests for the generic weekly-series accessors (the anti-leakage core)."""
import datetime as dt
import unittest

from wasde_predictor import series


def _r(day, value, key="US TOTAL", month=8):
    return series.WeeklyReading(week_ending=dt.date(2024, month, day), key=key, value=value)


class LatestBeforeTests(unittest.TestCase):
    def setUp(self):
        self.readings = [_r(4, 60.0), _r(11, 62.0), _r(18, 65.0)]

    def test_most_recent_strictly_before(self):
        got = series.latest_before(self.readings, dt.date(2024, 8, 12), "US TOTAL")
        self.assertEqual(got.week_ending, dt.date(2024, 8, 11))
        self.assertEqual(got.value, 62.0)

    def test_reading_on_cutoff_day_excluded(self):
        got = series.latest_before(self.readings, dt.date(2024, 8, 18), "US TOTAL")
        self.assertEqual(got.week_ending, dt.date(2024, 8, 11))  # the 18th itself is excluded

    def test_none_when_nothing_precedes(self):
        self.assertIsNone(series.latest_before(self.readings, dt.date(2024, 8, 1), "US TOTAL"))

    def test_other_keys_ignored(self):
        mixed = self.readings + [_r(10, 99.0, key="IOWA")]
        got = series.latest_before(mixed, dt.date(2024, 8, 12), "US TOTAL")
        self.assertEqual(got.value, 62.0)


class FirstInYearTests(unittest.TestCase):
    def test_returns_earliest_in_year_before_cutoff(self):
        readings = [_r(4, 60, month=6), _r(11, 61, month=6), _r(2, 64, month=8)]
        got = series.first_in_year(readings, 2024, "US TOTAL", before=dt.date(2024, 8, 12))
        self.assertEqual(got.week_ending, dt.date(2024, 6, 4))

    def test_respects_year_filter(self):
        readings = [series.WeeklyReading(dt.date(2023, 6, 5), "US TOTAL", 70), _r(4, 60, month=6)]
        got = series.first_in_year(readings, 2024, "US TOTAL")
        self.assertEqual(got.week_ending.year, 2024)


if __name__ == "__main__":
    unittest.main()
