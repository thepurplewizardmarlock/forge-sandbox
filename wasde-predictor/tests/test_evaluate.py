"""Tests for accuracy, classification metrics, and leave-one-year-out CV."""
import datetime as dt
import unittest

from wasde_predictor import evaluate
from wasde_predictor.dataset import Observation
from wasde_predictor.models import ConditionThresholdModel, LogisticRegression, MajorityBaseline


def _obs(label, condition_ge, year, month=8):
    return Observation(
        market_year=year,
        report_date=dt.date(int(year[:4]), month, 12),
        month=month,
        label=label,
        change=0.0,
        features={"condition_ge": condition_ge, "drought_d2plus": 40.0 - condition_ge / 2},
        feature_weeks=(dt.date(int(year[:4]), month, 11),),
    )


class MetricTests(unittest.TestCase):
    def test_accuracy(self):
        self.assertEqual(evaluate.accuracy([1, 0, 1], [1, 0, 1]), 1.0)
        self.assertEqual(evaluate.accuracy([1, 0], [1, 1]), 0.5)
        self.assertEqual(evaluate.accuracy([], []), 0.0)

    def test_confusion_and_prf(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 0, 1]
        c = evaluate.confusion(y_true, y_pred)
        self.assertEqual(c, {"tp": 1, "fp": 1, "tn": 1, "fn": 1})
        prf = evaluate.precision_recall_f1(y_true, y_pred)
        self.assertAlmostEqual(prf["precision"], 0.5)
        self.assertAlmostEqual(prf["recall"], 0.5)


class LeaveOneYearOutTests(unittest.TestCase):
    def _sep(self):
        obs = []
        for yr in ("2021/22", "2022/23", "2023/24", "2024/25"):
            obs.append(_obs(1, 75, yr))
            obs.append(_obs(0, 45, yr))
        return obs

    def test_covers_every_observation_once(self):
        res = evaluate.leave_one_year_out(self._sep(), ConditionThresholdModel)
        self.assertEqual(res["n"], 8)
        self.assertIn("per_month", res)
        self.assertIn("precision", res)
        self.assertIn("confusion", res)

    def test_models_beat_majority_on_separable_data(self):
        res = evaluate.compare(self._sep(), {
            "maj": MajorityBaseline,
            "cond": ConditionThresholdModel,
            "log": LogisticRegression,
        })
        self.assertGreater(res["cond"]["accuracy"], res["maj"]["accuracy"])
        self.assertGreater(res["log"]["accuracy"], res["maj"]["accuracy"])


if __name__ == "__main__":
    unittest.main()
