"""Tests for the magnitude (regression) models and metrics."""
import datetime as dt
import unittest

from wasde_predictor import evaluate
from wasde_predictor.dataset import Observation
from wasde_predictor.regression import MeanBaseline, PersistenceRegressor, RidgeRegression, ZeroBaseline


def _obs(change, feats, year="2024/25", month=8, prev_change=None):
    return Observation(
        market_year=year,
        report_date=dt.date(2024, month, 12),
        month=month,
        label=1 if change > 0 else 0,
        change=change,
        features=feats,
        feature_weeks=(dt.date(2024, month, 11),),
        prev_change=prev_change,
    )


class BaselineTests(unittest.TestCase):
    def test_zero_predicts_zero(self):
        self.assertEqual(ZeroBaseline().fit([]).predict(_obs(3.0, {"x": 1.0})), 0.0)

    def test_mean_predicts_training_mean(self):
        m = MeanBaseline().fit([_obs(2.0, {"x": 1.0}), _obs(4.0, {"x": 1.0})])
        self.assertAlmostEqual(m.predict(_obs(0.0, {"x": 1.0})), 3.0)

    def test_persistence_uses_prev_change(self):
        m = PersistenceRegressor().fit([_obs(1.0, {"x": 1.0})])
        self.assertEqual(m.predict(_obs(0.0, {"x": 1.0}, prev_change=2.5)), 2.5)


class RidgeTests(unittest.TestCase):
    def test_learns_a_linear_signal(self):
        # change ~ 0.5 * condition, so predictions should track it.
        obs = [_obs(0.5 * c, {"condition_ge": float(c)}) for c in range(-10, 11)]
        m = RidgeRegression(n_iter=4000).fit(obs)
        pred_hi = m.predict(_obs(0.0, {"condition_ge": 10.0}))
        pred_lo = m.predict(_obs(0.0, {"condition_ge": -10.0}))
        self.assertGreater(pred_hi, pred_lo)
        self.assertGreater(m.coefficients()["condition_ge"], 0)


class MetricTests(unittest.TestCase):
    def test_mae_rmse(self):
        self.assertAlmostEqual(evaluate.mae([1.0, 3.0], [1.0, 1.0]), 1.0)
        self.assertAlmostEqual(evaluate.rmse([0.0, 0.0], [3.0, 4.0]), 3.5355, places=3)

    def test_regression_harness_structure(self):
        obs = []
        for yr in ("2022/23", "2023/24", "2024/25"):
            obs += [_obs(2.0, {"condition_ge": 70.0}, year=yr),
                    _obs(-2.0, {"condition_ge": 40.0}, year=yr)]
        res = evaluate.leave_one_year_out_regression(obs, RidgeRegression)
        self.assertEqual(res["n"], 6)
        self.assertIn("mae", res)
        self.assertIn("direction_accuracy", res)


if __name__ == "__main__":
    unittest.main()
