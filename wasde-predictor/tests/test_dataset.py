"""Tests for building observations -- including the no-leakage guarantee."""
import unittest
from pathlib import Path

from wasde_predictor import commodities
from wasde_predictor import features as F
from wasde_predictor.dataset import build_dataset, class_balance

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


def _demand_paths(c):
    return [(clue, SAMPLE / f"{clue.basename}.csv") for clue in c.demand_clues]


class YieldDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs = build_dataset(SAMPLE / "wasde_corn.csv", SAMPLE / "condition_corn.csv",
                                SAMPLE / "drought_corn.csv", commodity="Corn")

    def test_dataset_is_non_empty(self):
        self.assertTrue(self.obs)

    def test_no_leakage_every_feature_week_before_report(self):
        for o in self.obs:
            self.assertTrue(o.feature_weeks)
            for w in o.feature_weeks:
                self.assertLess(w, o.report_date, msg=f"leak: {w} >= {o.report_date}")

    def test_labels_are_binary(self):
        self.assertTrue(all(o.label in (0, 1) for o in self.obs))

    def test_supply_feature_vector(self):
        for o in self.obs:
            self.assertEqual(set(o.features), set(F.FEATURE_NAMES))

    def test_only_target_months(self):
        self.assertTrue(all(o.month in (8, 9, 10, 11) for o in self.obs))

    def test_class_balance_sums(self):
        bal = class_balance(self.obs)
        self.assertEqual(bal["up"] + bal["down_or_flat"], bal["n"])


class CornEndingStocksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c = commodities.CORN
        cls.obs = build_dataset(SAMPLE / "wasde_corn.csv", SAMPLE / "condition_corn.csv",
                                SAMPLE / "drought_corn.csv", demand=_demand_paths(c),
                                commodity="Corn", attribute="Ending Stocks")

    def test_includes_corn_demand_features(self):
        expected = set(F.FEATURE_NAMES) | {"export_pace_surprise", "ethanol_pace_surprise"}
        for o in self.obs:
            self.assertEqual(set(o.features), expected)

    def test_no_leakage_with_demand_clues(self):
        for o in self.obs:
            for w in o.feature_weeks:
                self.assertLess(w, o.report_date)

    def test_changes_are_million_bushel_scale(self):
        self.assertTrue(any(abs(o.change) > 10 for o in self.obs))


class SoybeansEndingStocksTests(unittest.TestCase):
    def test_soybeans_ending_stocks_uses_crush_and_exports(self):
        c = commodities.SOYBEANS
        obs = build_dataset(SAMPLE / "wasde_soybeans.csv", SAMPLE / "condition_soybeans.csv",
                            SAMPLE / "drought_soybeans.csv", demand=_demand_paths(c),
                            commodity="Soybeans", attribute="Ending Stocks")
        self.assertTrue(obs)
        expected = set(F.FEATURE_NAMES) | {"export_pace_surprise", "crush_pace_surprise"}
        self.assertEqual(set(obs[0].features), expected)


class WheatDatasetTests(unittest.TestCase):
    def test_wheat_yield_dataset_builds(self):
        obs = build_dataset(SAMPLE / "wasde_wheat.csv", SAMPLE / "condition_wheat.csv",
                            SAMPLE / "drought_wheat.csv", commodity="Wheat",
                            target_months=commodities.WHEAT.report_months)
        self.assertTrue(obs)


if __name__ == "__main__":
    unittest.main()
