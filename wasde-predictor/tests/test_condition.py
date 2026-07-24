"""Tests for the crop-condition feature loader and the point-in-time accessor."""
import datetime as dt
import unittest

from wasde_predictor import condition


def _reading(day, value, state="US TOTAL"):
    return condition.ConditionReading(week_ending=dt.date(2024, 8, day), state=state, value=value)


class LatestBeforeTests(unittest.TestCase):
    def setUp(self):
        self.readings = [
            _reading(4, 60.0),
            _reading(11, 62.0),
            _reading(18, 65.0),
        ]

    def test_returns_most_recent_strictly_before_cutoff(self):
        got = condition.latest_before(self.readings, dt.date(2024, 8, 12))
        self.assertIsNotNone(got)
        self.assertEqual(got.week_ending, dt.date(2024, 8, 11))
        self.assertEqual(got.value, 62.0)

    def test_reading_on_the_cutoff_day_is_excluded(self):
        # A reading whose week ends exactly on report day must NOT be used (leak).
        got = condition.latest_before(self.readings, dt.date(2024, 8, 18))
        self.assertEqual(got.week_ending, dt.date(2024, 8, 11))

    def test_returns_none_when_nothing_precedes(self):
        self.assertIsNone(condition.latest_before(self.readings, dt.date(2024, 8, 1)))

    def test_other_states_are_ignored(self):
        mixed = self.readings + [_reading(10, 99.0, state="IOWA")]
        got = condition.latest_before(mixed, dt.date(2024, 8, 12), state="US TOTAL")
        self.assertEqual(got.value, 62.0)  # ignores the Iowa reading


if __name__ == "__main__":
    unittest.main()
