"""Tests for the baseline and the one-feature condition model."""
import datetime as dt
import unittest

from wasde_predictor.dataset import Observation
from wasde_predictor.models import ConditionThresholdModel, MajorityBaseline


def _obs(label, condition_ge, year="2024/25"):
    return Observation(
        market_year=year,
        report_date=dt.date(2024, 8, 12),
        month=8,
        label=label,
        change=0.0,
        features={"condition_ge": condition_ge},
        condition_week=dt.date(2024, 8, 11),
    )


class MajorityBaselineTests(unittest.TestCase):
    def test_predicts_the_majority_class(self):
        train = [_obs(1, 70), _obs(1, 71), _obs(0, 50)]  # majority = up (1)
        m = MajorityBaseline().fit(train)
        self.assertEqual(m.predict(_obs(0, 40)), 1)

    def test_predicts_zero_when_down_is_majority(self):
        train = [_obs(0, 50), _obs(0, 51), _obs(1, 70)]
        m = MajorityBaseline().fit(train)
        self.assertEqual(m.predict(_obs(1, 90)), 0)


class ConditionThresholdModelTests(unittest.TestCase):
    def test_threshold_is_training_mean(self):
        train = [_obs(0, 60), _obs(1, 70)]
        m = ConditionThresholdModel().fit(train)
        self.assertAlmostEqual(m.threshold, 65.0)

    def test_predicts_up_at_or_above_threshold(self):
        m = ConditionThresholdModel().fit([_obs(0, 60), _obs(1, 70)])  # threshold 65
        self.assertEqual(m.predict(_obs(0, 65)), 1)   # at threshold -> up
        self.assertEqual(m.predict(_obs(0, 66)), 1)
        self.assertEqual(m.predict(_obs(0, 64)), 0)   # below -> down


if __name__ == "__main__":
    unittest.main()
