"""Magnitude models: predict *how much* USDA changes the yield (bushels/acre),
not just the direction.

Same tiny interface as the classifiers: .fit(observations) then
.predict(observation) -> float. Baselines first, then a real (stdlib) linear
model:

  * ZeroBaseline        -- predict 0.0 ("USDA won't change it"). The MAE of this
                           is the number to beat.
  * MeanBaseline        -- predict the average training change.
  * PersistenceRegressor-- predict the previous report's change this season.
  * RidgeRegression     -- gradient-descent linear regression on standardized
                           features with L2 regularization (pure standard library).
"""
from __future__ import annotations

from statistics import mean, pstdev

from .dataset import Observation


class ZeroBaseline:
    def fit(self, observations: list[Observation]) -> "ZeroBaseline":
        return self

    def predict(self, observation: Observation) -> float:
        return 0.0


class MeanBaseline:
    def __init__(self) -> None:
        self._mean = 0.0

    def fit(self, observations: list[Observation]) -> "MeanBaseline":
        self._mean = mean([o.change for o in observations]) if observations else 0.0
        return self

    def predict(self, observation: Observation) -> float:
        return self._mean

    @property
    def value(self) -> float:
        return self._mean


class PersistenceRegressor:
    def __init__(self) -> None:
        self._fallback = 0.0

    def fit(self, observations: list[Observation]) -> "PersistenceRegressor":
        self._fallback = mean([o.change for o in observations]) if observations else 0.0
        return self

    def predict(self, observation: Observation) -> float:
        return observation.prev_change if observation.prev_change is not None else self._fallback


class RidgeRegression:
    """L2-regularized linear regression via batch gradient descent."""

    def __init__(self, lr: float = 0.2, n_iter: int = 3000, l2: float = 0.05) -> None:
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.feature_names: list[str] = []
        self._mean: list[float] = []
        self._std: list[float] = []
        self._w: list[float] = []
        self._b: float = 0.0

    def fit(self, observations: list[Observation]) -> "RidgeRegression":
        if not observations:
            raise ValueError("cannot fit on an empty dataset")
        self.feature_names = sorted(observations[0].features)
        X = [[o.features[f] for f in self.feature_names] for o in observations]
        y = [o.change for o in observations]

        cols = list(zip(*X))
        self._mean = [mean(c) for c in cols]
        self._std = [pstdev(c) or 1.0 for c in cols]
        Z = [self._standardize_row(row) for row in X]

        n, k = len(Z), len(self.feature_names)
        w = [0.0] * k
        b = mean(y)
        for _ in range(self.n_iter):
            gw = [0.0] * k
            gb = 0.0
            for zi, yi in zip(Z, y):
                pred = sum(w[j] * zi[j] for j in range(k)) + b
                err = pred - yi
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

    def predict(self, observation: Observation) -> float:
        z = self._standardize_row([observation.features[f] for f in self.feature_names])
        return sum(self._w[j] * z[j] for j in range(len(self._w))) + self._b

    def coefficients(self) -> dict[str, float]:
        return {name: round(w, 4) for name, w in zip(self.feature_names, self._w)}
