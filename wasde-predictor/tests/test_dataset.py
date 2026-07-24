"""Tests for building observations -- including the no-leakage guarantee."""
import unittest
from pathlib import Path

from wasde_predictor.dataset import build_dataset, class_balance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_YIELD = PROJECT_ROOT / "data" / "sample" / "wasde_corn_yield_sample.csv"
SAMPLE_CONDITION = PROJECT_ROOT / "data" / "sample" / "crop_condition_sample.csv"


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs = build_dataset(SAMPLE_YIELD, SAMPLE_CONDITION)

    def test_dataset_is_non_empty(self):
        self.assertTrue(self.obs)

    def test_no_leakage_condition_week_before_report(self):
        # THE guarantee: every clue used must pre-date the report it feeds.
        for o in self.obs:
            self.assertIsNotNone(o.condition_week)
            self.assertLess(
                o.condition_week, o.report_date,
                msg=f"leak: condition week {o.condition_week} >= report {o.report_date}",
            )

    def test_labels_are_binary(self):
        self.assertTrue(all(o.label in (0, 1) for o in self.obs))

    def test_feature_present(self):
        for o in self.obs:
            self.assertIn("condition_ge", o.features)
            self.assertIsInstance(o.features["condition_ge"], float)

    def test_only_target_months(self):
        self.assertTrue(all(o.month in (8, 9, 10, 11) for o in self.obs))

    def test_class_balance_sums(self):
        bal = class_balance(self.obs)
        self.assertEqual(bal["up"] + bal["down_or_flat"], bal["n"])


if __name__ == "__main__":
    unittest.main()
