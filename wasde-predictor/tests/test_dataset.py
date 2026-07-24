"""Tests for building observations -- including the no-leakage guarantee."""
import unittest
from pathlib import Path

from wasde_predictor import features as F
from wasde_predictor.dataset import build_dataset, class_balance

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"
WASDE = SAMPLE / "wasde_corn_sample.csv"
COND = SAMPLE / "condition_corn_sample.csv"
DROUGHT = SAMPLE / "drought_corn_sample.csv"
EXP = SAMPLE / "exports_corn_sample.csv"
ETH = SAMPLE / "ethanol_corn_sample.csv"


class YieldDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs = build_dataset(WASDE, COND, DROUGHT)  # default target = yield

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


class EndingStocksDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs = build_dataset(WASDE, COND, DROUGHT, exports_path=EXP, ethanol_path=ETH,
                                attribute="Ending Stocks")

    def test_dataset_is_non_empty(self):
        self.assertTrue(self.obs)

    def test_includes_demand_features(self):
        expected = set(F.FEATURE_NAMES) | set(F.DEMAND_FEATURE_NAMES)
        for o in self.obs:
            self.assertEqual(set(o.features), expected)

    def test_no_leakage_with_demand_clues(self):
        for o in self.obs:
            for w in o.feature_weeks:
                self.assertLess(w, o.report_date)

    def test_changes_are_in_million_bushels_scale(self):
        # ending-stocks changes are much bigger than yield changes (bushels/acre)
        self.assertTrue(any(abs(o.change) > 10 for o in self.obs))


class SoybeansDatasetTests(unittest.TestCase):
    def test_soybeans_yield_dataset_builds(self):
        obs = build_dataset(
            SAMPLE / "wasde_soybeans_sample.csv",
            SAMPLE / "condition_soybeans_sample.csv",
            SAMPLE / "drought_soybeans_sample.csv",
            commodity="Soybeans",
        )
        self.assertTrue(obs)
        self.assertEqual(set(obs[0].features), set(F.FEATURE_NAMES))
        for o in obs:  # leak guard holds for the other commodity too
            for w in o.feature_weeks:
                self.assertLess(w, o.report_date)


if __name__ == "__main__":
    unittest.main()
