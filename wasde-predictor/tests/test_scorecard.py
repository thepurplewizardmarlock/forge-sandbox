"""Tests for the all-combos scorecard."""
import unittest
from pathlib import Path

from wasde_predictor import commodities, scorecard

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


class ScorecardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = scorecard.evaluate_all(SAMPLE)

    def test_one_row_per_commodity_and_target(self):
        self.assertEqual(len(self.rows), len(commodities.ALL) * 2)
        combos = {(r["commodity"], r["target"]) for r in self.rows}
        for slug in commodities.ALL:
            self.assertIn((slug, "yield"), combos)
            self.assertIn((slug, "ending-stocks"), combos)

    def test_rows_have_sane_metrics(self):
        for r in self.rows:
            self.assertGreater(r["n"], 0)
            self.assertGreaterEqual(r["logistic"], 0.0)
            self.assertLessEqual(r["logistic"], 1.0)
            self.assertAlmostEqual(r["lift"], round(r["logistic"] - r["best_baseline"], 3))


if __name__ == "__main__":
    unittest.main()
