"""Baselines and models. Every model exposes .fit(observations) then
.predict(observation) -> int; models that can also give a probability expose
.predict_proba(observation) -> float in [0, 1].

  * MajorityBaseline    -- always predict the most common training outcome. The
                           "no skill" bar every real model must beat.
  * ConditionThreshold  -- predict "up" when condition is above the training mean.
                           One feature, one learned threshold; the v1 skeleton model.
  * LogisticRegression  -- multi-feature logistic regression, trained by gradient
                           descent on standardized features (pure standard library,
                           no numpy). Coefficients double as feature importance.
"""
from __future__ import annotations

import math
from statistics import mean, pstdev

from .dataset import Observation


class MajorityBaseline:
    def __init__(self) -> None:
        self._prediction = 0

    def fit(self, observations: list[Observation]) -> "MajorityBaseline":
        ups = sum(o.label for o in observations)
        self._prediction = 1 if ups * 2 > len(observations) else 0
        return self

    def predict(self, observation: Observation) -> int:
        return self._prediction


class PersistenceBaseline:
    """Predict the same direction USDA revised at the previous report this season.

    This is the honest bar for an autocorrelated target: if a crop looked good
    enough for an August upgrade, it often stays good enough for a September one.
    For the season's first target (August, no prior report), fall back to the
    training-majority direction. A real model has to beat *this*, not just a coin
    flip.
    """

    def __init__(self) -> None:
        self._fallback = 0

    def fit(self, observations: list[Observation]) -> "PersistenceBaseline":
        ups = sum(o.label for o in observations)
        self._fallback = 1 if ups * 2 > len(observations) else 0
        return self

    def predict(self, observation: Observation) -> int:
        if observation.prev_direction is None:
            return self._fallback
        return observation.prev_direction


class ConditionThresholdModel:
    """Predict 'up' when condition is at/above the average condition seen in training."""

    feature = "condition_ge"

    def __init__(self) -> None:
        self._threshold = 0.0

    def fit(self, observations: list[Observation]) -> "ConditionThresholdModel":
        values = [o.features[self.feature] for o in observations]
        self._threshold = mean(values) if values else 0.0
        return self

    @property
    def threshold(self) -> float:
        return self._threshold

    def predict(self, observation: Observation) -> int:
        return 1 if observation.features[self.feature] >= self._threshold else 0


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


class LogisticRegression:
    """Batch gradient descent logistic regression with L2 regularization."""

    def __init__(self, lr: float = 0.3, n_iter: int = 3000, l2: float = 0.02) -> None:
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.feature_names: list[str] = []
        self._mean: list[float] = []
        self._std: list[float] = []
        self._w: list[float] = []
        self._b: float = 0.0

    def fit(self, observations: list[Observation]) -> "LogisticRegression":
        if not observations:
            raise ValueError("cannot fit on an empty dataset")
        self.feature_names = sorted(observations[0].features)
        X = [[o.features[f] for f in self.feature_names] for o in observations]
        y = [float(o.label) for o in observations]

        cols = list(zip(*X))
        self._mean = [mean(c) for c in cols]
        self._std = [pstdev(c) or 1.0 for c in cols]
        Z = [self._standardize_row(row) for row in X]

        n = len(Z)
        k = len(self.feature_names)
        w = [0.0] * k
        b = 0.0
        for _ in range(self.n_iter):
            gw = [0.0] * k
            gb = 0.0
            for zi, yi in zip(Z, y):
                p = _sigmoid(sum(w[j] * zi[j] for j in range(k)) + b)
                err = p - yi
                for j in range(k):
                    gw[j] += err * zi[j]
                gb += err
            for j in range(k):
                w[j] -= self.lr * (gw[j] / n + self.l2 * w[j])
            b -= self.lr * (gb / n)
        self._w = w
        self._b = b
        return self

    def _standardize_row(self, row: list[float]) -> list[float]:
        return [(x - m) / s for x, m, s in zip(row, self._mean, self._std)]

    def predict_proba(self, observation: Observation) -> float:
        row = [observation.features[f] for f in self.feature_names]
        z = self._standardize_row(row)
        return _sigmoid(sum(self._w[j] * z[j] for j in range(len(self._w))) + self._b)

    def predict(self, observation: Observation) -> int:
        return 1 if self.predict_proba(observation) >= 0.5 else 0

    def coefficients(self) -> dict[str, float]:
        """Standardized weights -> directly comparable as feature importance."""
        return {name: round(w, 4) for name, w in zip(self.feature_names, self._w)}
