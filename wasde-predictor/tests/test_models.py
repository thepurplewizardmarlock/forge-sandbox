"""Tests for the baselines and the logistic-regression model."""
import datetime as dt
import unittest

from wasde_predictor.dataset import Observation
from wasde_predictor.models import (
    ConditionThresholdModel,
    LogisticRegression,
    MajorityBaseline,
    PersistenceBaseline,
)


def _obs(label, feats, year="2024/25", month=8, prev_direction=None):
    return Observation(
        market_year=year,
        report_date=dt.date(2024, month, 12),
        month=month,
        label=label,
        change=0.0,
        features=feats,
        feature_weeks=(dt.date(2024, month, 11),),
        prev_direction=prev_direction,
    )


class PersistenceBaselineTests(unittest.TestCase):
    def test_predicts_previous_direction(self):
        m = PersistenceBaseline().fit([_obs(1, {"condition_ge": 70})])
        self.assertEqual(m.predict(_obs(0, {"condition_ge": 50}, month=9, prev_direction=1)), 1)
        self.assertEqual(m.predict(_obs(1, {"condition_ge": 70}, month=9, prev_direction=0)), 0)

    def test_falls_back_to_majority_when_no_prior(self):
        train = [_obs(1, {"condition_ge": 70}), _obs(1, {"condition_ge": 71}), _obs(0, {"condition_ge": 50})]
        m = PersistenceBaseline().fit(train)  # majority = up
        self.assertEqual(m.predict(_obs(0, {"condition_ge": 40}, prev_direction=None)), 1)


class MajorityBaselineTests(unittest.TestCase):
    def test_predicts_majority(self):
        train = [_obs(1, {"condition_ge": 70}), _obs(1, {"condition_ge": 71}), _obs(0, {"condition_ge": 50})]
        m = MajorityBaseline().fit(train)
        self.assertEqual(m.predict(_obs(0, {"condition_ge": 40})), 1)


class ConditionThresholdTests(unittest.TestCase):
    def test_threshold_is_training_mean(self):
        m = ConditionThresholdModel().fit([_obs(0, {"condition_ge": 60}), _obs(1, {"condition_ge": 70})])
        self.assertAlmostEqual(m.threshold, 65.0)
        self.assertEqual(m.predict(_obs(0, {"condition_ge": 66})), 1)
        self.assertEqual(m.predict(_obs(0, {"condition_ge": 64})), 0)


class LogisticRegressionTests(unittest.TestCase):
    def _separable(self):
        obs = []
        for yr in ("2021/22", "2022/23", "2023/24", "2024/25"):
            obs.append(_obs(1, {"condition_ge": 75.0, "drought_d2plus": 5.0}, year=yr))
            obs.append(_obs(0, {"condition_ge": 45.0, "drought_d2plus": 35.0}, year=yr))
        return obs

    def test_learns_separable_data(self):
        m = LogisticRegression(n_iter=2000).fit(self._separable())
        self.assertEqual(m.predict(_obs(0, {"condition_ge": 76.0, "drought_d2plus": 4.0})), 1)
        self.assertEqual(m.predict(_obs(0, {"condition_ge": 44.0, "drought_d2plus": 36.0})), 0)

    def test_proba_in_unit_interval(self):
        m = LogisticRegression(n_iter=500).fit(self._separable())
        p = m.predict_proba(_obs(0, {"condition_ge": 60.0, "drought_d2plus": 20.0}))
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_coefficient_signs_make_sense(self):
        m = LogisticRegression(n_iter=3000).fit(self._separable())
        coefs = m.coefficients()
        self.assertGreater(coefs["condition_ge"], 0)     # better crop -> more likely up
        self.assertLess(coefs["drought_d2plus"], 0)      # more drought -> less likely up

    def test_empty_fit_raises(self):
        with self.assertRaises(ValueError):
            LogisticRegression().fit([])


if __name__ == "__main__":
    unittest.main()
