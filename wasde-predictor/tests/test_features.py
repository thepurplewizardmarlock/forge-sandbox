"""Tests for feature construction, including the no-leakage guarantee."""
import datetime as dt
import unittest

from wasde_predictor import features as F
from wasde_predictor.series import WeeklyReading


def _cond(month, day, value):
    return WeeklyReading(dt.date(2024, month, day), "US TOTAL", value)


def _dro(month, day, value):
    return WeeklyReading(dt.date(2024, month, day), "US CORN BELT", value)


class BuildFeaturesTests(unittest.TestCase):
    def setUp(self):
        # Weekly condition & drought June -> mid-September.
        self.conditions = [_cond(6, 3, 60), _cond(8, 5, 66), _cond(9, 9, 70)]
        self.droughts = [_dro(6, 3, 20), _dro(8, 5, 15), _dro(9, 9, 10)]
        self.report = dt.date(2024, 9, 12)      # September report
        self.prev = dt.date(2024, 8, 12)        # August report

    def test_all_features_present(self):
        feats, weeks = F.build_features(self.report, self.prev, self.conditions, self.droughts)
        self.assertIsNotNone(feats)
        self.assertEqual(set(feats), set(F.FEATURE_NAMES))

    def test_no_week_leaks_past_report(self):
        feats, weeks = F.build_features(self.report, self.prev, self.conditions, self.droughts)
        self.assertTrue(weeks)
        for w in weeks:
            self.assertLess(w, self.report)

    def test_values_are_correct(self):
        feats, _ = F.build_features(self.report, self.prev, self.conditions, self.droughts)
        # latest condition before Sep 12 is Sep 9 = 70; before Aug 12 is Aug 5 = 66.
        self.assertEqual(feats["condition_ge"], 70)
        self.assertEqual(feats["condition_change"], 70 - 66)
        self.assertEqual(feats["condition_vs_season_start"], 70 - 60)  # season start = Jun 3
        self.assertEqual(feats["drought_d2plus"], 10)
        self.assertEqual(feats["drought_change"], 10 - 15)

    def test_returns_none_when_no_condition_before_report(self):
        feats, weeks = F.build_features(dt.date(2024, 6, 1), None, self.conditions, self.droughts)
        self.assertIsNone(feats)
        self.assertEqual(weeks, [])


class DemandFeaturesTests(unittest.TestCase):
    def setUp(self):
        from wasde_predictor import commodities
        self.exp_clue = commodities.DemandClue("export_pace_surprise", "exports", "CORN", "commodity")
        self.eth_clue = commodities.DemandClue("ethanol_pace_surprise", "ethanol", "US", "region")
        exports = [WeeklyReading(dt.date(2024, 8, 5), "CORN", 3.0),
                   WeeklyReading(dt.date(2024, 9, 9), "CORN", 6.0)]
        ethanol = [WeeklyReading(dt.date(2024, 8, 5), "US", -1.0),
                   WeeklyReading(dt.date(2024, 9, 9), "US", 2.0)]
        self.demand_loaded = [(self.exp_clue, exports), (self.eth_clue, ethanol)]
        self.report = dt.date(2024, 9, 12)

    def test_demand_features_point_in_time(self):
        feats, weeks = F.build_demand_features(self.report, self.demand_loaded)
        self.assertEqual(set(feats), {"export_pace_surprise", "ethanol_pace_surprise"})
        self.assertEqual(feats["export_pace_surprise"], 6.0)   # Sep 9 value
        self.assertEqual(feats["ethanol_pace_surprise"], 2.0)
        for w in weeks:
            self.assertLess(w, self.report)

    def test_build_all_merges_supply_and_demand(self):
        conditions = [WeeklyReading(dt.date(2024, 6, 3), "US TOTAL", 60),
                      WeeklyReading(dt.date(2024, 9, 9), "US TOTAL", 68)]
        droughts = [WeeklyReading(dt.date(2024, 6, 3), "US CORN BELT", 15),
                    WeeklyReading(dt.date(2024, 9, 9), "US CORN BELT", 10)]
        feats, weeks = F.build_all_features(
            self.report, dt.date(2024, 8, 12), conditions, droughts, self.demand_loaded)
        self.assertEqual(set(feats), set(F.FEATURE_NAMES) | {"export_pace_surprise", "ethanol_pace_surprise"})


if __name__ == "__main__":
    unittest.main()
