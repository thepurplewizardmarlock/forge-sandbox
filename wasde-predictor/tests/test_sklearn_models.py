"""Tests for the OPTIONAL scikit-learn models. Skipped when sklearn is absent."""
import datetime as dt
import unittest

from wasde_predictor import sklearn_models
from wasde_predictor.dataset import Observation


def _obs(label, change, feats, year="2024/25", month=8):
    return Observation(
        market_year=year, report_date=dt.date(2024, month, 12), month=month,
        label=label, change=change, features=feats,
        feature_weeks=(dt.date(2024, month, 11),),
    )


def _separable():
    obs = []
    for yr in ("2021/22", "2022/23", "2023/24", "2024/25"):
        obs.append(_obs(1, 2.0, {"condition_ge": 75.0, "drought_d2plus": 5.0}, year=yr))
        obs.append(_obs(0, -2.0, {"condition_ge": 45.0, "drought_d2plus": 35.0}, year=yr))
    return obs


@unittest.skipUnless(sklearn_models.HAS_SKLEARN, "scikit-learn not installed")
class SklearnModelTests(unittest.TestCase):
    def test_classifier_learns_separable(self):
        m = sklearn_models.GBClassifier().fit(_separable())
        self.assertEqual(m.predict(_obs(0, 0.0, {"condition_ge": 78.0, "drought_d2plus": 3.0})), 1)
        self.assertEqual(m.predict(_obs(0, 0.0, {"condition_ge": 42.0, "drought_d2plus": 38.0})), 0)

    def test_classifier_handles_single_class(self):
        obs = [_obs(1, 1.0, {"condition_ge": 70.0, "drought_d2plus": 5.0})]
        m = sklearn_models.GBClassifier().fit(obs)  # only one class present
        self.assertEqual(m.predict(_obs(0, 0.0, {"condition_ge": 20.0, "drought_d2plus": 60.0})), 1)

    def test_regressor_tracks_signal(self):
        m = sklearn_models.GBRegressor().fit(_separable())
        hi = m.predict(_obs(0, 0.0, {"condition_ge": 75.0, "drought_d2plus": 5.0}))
        lo = m.predict(_obs(0, 0.0, {"condition_ge": 45.0, "drought_d2plus": 35.0}))
        self.assertGreater(hi, lo)


if __name__ == "__main__":
    unittest.main()
