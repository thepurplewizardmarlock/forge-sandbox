"""Optional scikit-learn models (gradient-boosted trees).

These are *optional*: the rest of the project runs on the standard library alone.
If scikit-learn isn't installed, `HAS_SKLEARN` is False and the CLI simply skips
these models. Install with:  pip install scikit-learn

Honest expectation: on a sample this small (a couple hundred real observations at
most), gradient boosting tends to OVERFIT and rarely beats the plain logistic /
ridge models. It's included so you can see that for yourself -- a useful lesson,
not a magic upgrade.

Both wrappers adapt our Observation objects to scikit-learn's X/y arrays and keep
the same .fit()/.predict() interface as the stdlib models.
"""
from __future__ import annotations

from .dataset import Observation

try:  # optional dependency
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover - exercised only when sklearn is absent
    HAS_SKLEARN = False


def _matrix(observations: list[Observation], feature_names: list[str]) -> list[list[float]]:
    return [[o.features[f] for f in feature_names] for o in observations]


class GBClassifier:
    """Gradient-boosted trees for the up/down direction target."""

    def __init__(self, **kwargs) -> None:
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn is not installed")
        params = dict(n_estimators=100, max_depth=2, learning_rate=0.05)
        params.update(kwargs)
        self._model = GradientBoostingClassifier(**params)
        self.feature_names: list[str] = []

    def fit(self, observations: list[Observation]) -> "GBClassifier":
        self.feature_names = sorted(observations[0].features)
        X = _matrix(observations, self.feature_names)
        y = [o.label for o in observations]
        # GradientBoosting needs at least two classes to fit.
        if len(set(y)) < 2:
            self._model = None
            self._constant = y[0]
            return self
        self._constant = None
        self._model.fit(X, y)
        return self

    def predict(self, observation: Observation) -> int:
        if self._model is None:
            return int(self._constant)
        x = [observation.features[f] for f in self.feature_names]
        return int(self._model.predict([x])[0])

    def predict_proba(self, observation: Observation) -> float:
        if self._model is None:
            return float(self._constant)
        x = [observation.features[f] for f in self.feature_names]
        return float(self._model.predict_proba([x])[0][1])


class GBRegressor:
    """Gradient-boosted trees for the magnitude (bushels/acre) target."""

    def __init__(self, **kwargs) -> None:
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn is not installed")
        params = dict(n_estimators=100, max_depth=2, learning_rate=0.05)
        params.update(kwargs)
        self._model = GradientBoostingRegressor(**params)
        self.feature_names: list[str] = []

    def fit(self, observations: list[Observation]) -> "GBRegressor":
        self.feature_names = sorted(observations[0].features)
        X = _matrix(observations, self.feature_names)
        y = [o.change for o in observations]
        self._model.fit(X, y)
        return self

    def predict(self, observation: Observation) -> float:
        x = [observation.features[f] for f in self.feature_names]
        return float(self._model.predict([x])[0])
