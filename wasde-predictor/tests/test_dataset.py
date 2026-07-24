"""Tests for building observations -- including the no-leakage guarantee."""
import unittest
from pathlib import Path

from wasde_predictor import features as F
from wasde_predictor.dataset import build_dataset, class_balance

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs = build_dataset(
            SAMPLE / "wasde_corn_yield_sample.csv",
            SAMPLE / "crop_condition_sample.csv",
            SAMPLE / "drought_sample.csv",
        )

    def test_dataset_is_non_empty(self):
        self.assertTrue(self.obs)

    def test_no_leakage_every_feature_week_before_report(self):
        # THE guarantee: every source week used must pre-date the report it feeds.
        for o in self.obs:
            self.assertTrue(o.feature_weeks)
            for w in o.feature_weeks:
                self.assertLess(w, o.report_date, msg=f"leak: {w} >= {o.report_date}")

    def test_labels_are_binary(self):
        self.assertTrue(all(o.label in (0, 1) for o in self.obs))

    def test_full_feature_vector_present(self):
        for o in self.obs:
            self.assertEqual(set(o.features), set(F.FEATURE_NAMES))

    def test_only_target_months(self):
        self.assertTrue(all(o.month in (8, 9, 10, 11) for o in self.obs))

    def test_class_balance_sums(self):
        bal = class_balance(self.obs)
        self.assertEqual(bal["up"] + bal["down_or_flat"], bal["n"])


if __name__ == "__main__":
    unittest.main()
