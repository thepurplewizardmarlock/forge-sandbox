"""Tests for accuracy and leave-one-year-out cross-validation."""
import datetime as dt
import unittest

from wasde_predictor import evaluate
from wasde_predictor.dataset import Observation
from wasde_predictor.models import ConditionThresholdModel, MajorityBaseline


def _obs(label, condition_ge, year, month=8):
    return Observation(
        market_year=year,
        report_date=dt.date(int(year[:4]), month, 12),
        month=month,
        label=label,
        change=0.0,
        features={"condition_ge": condition_ge},
        condition_week=dt.date(int(year[:4]), month, 11),
    )


class AccuracyTests(unittest.TestCase):
    def test_perfect_and_zero(self):
        self.assertEqual(evaluate.accuracy([1, 0, 1], [1, 0, 1]), 1.0)
        self.assertEqual(evaluate.accuracy([1, 1], [0, 0]), 0.0)

    def test_half(self):
        self.assertEqual(evaluate.accuracy([1, 0], [1, 1]), 0.5)

    def test_empty_is_zero(self):
        self.assertEqual(evaluate.accuracy([], []), 0.0)


class LeaveOneYearOutTests(unittest.TestCase):
    def test_covers_every_observation_once(self):
        obs = [
            _obs(1, 70, "2022/23"), _obs(0, 50, "2022/23"),
            _obs(1, 72, "2023/24"), _obs(0, 48, "2023/24"),
            _obs(1, 71, "2024/25"), _obs(0, 49, "2024/25"),
        ]
        res = evaluate.leave_one_year_out(obs, ConditionThresholdModel)
        self.assertEqual(res["n"], len(obs))  # every obs predicted exactly once
        self.assertIn("per_month", res)
        self.assertGreaterEqual(res["accuracy"], 0.0)
        self.assertLessEqual(res["accuracy"], 1.0)

    def test_condition_model_beats_majority_on_separable_data(self):
        # Cleanly separable: high condition -> up, low -> down, across 3 years.
        obs = []
        for yr in ("2022/23", "2023/24", "2024/25"):
            obs.append(_obs(1, 75, yr))
            obs.append(_obs(0, 45, yr))
        res = evaluate.compare(
            obs, {"maj": MajorityBaseline, "cond": ConditionThresholdModel}
        )
        self.assertGreater(res["cond"]["accuracy"], res["maj"]["accuracy"])


if __name__ == "__main__":
    unittest.main()
