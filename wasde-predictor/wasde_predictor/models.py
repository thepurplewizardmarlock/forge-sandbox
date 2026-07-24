"""Baselines and the (deliberately dumb) v1 model.

The point of the walking skeleton is to establish the *bar* honestly:

  * MajorityBaseline   -- always predict the most common outcome in the training
                          data. This is the "no skill" bar every real model must
                          beat. (A coin flip is 0.50; the majority class is often
                          higher, so this is the stricter, fairer baseline.)
  * ConditionThreshold -- the actual v1 model: predict "up" when the crop-
                          condition clue is above the training-set average. One
                          feature, one learned threshold. Easy to reason about;
                          later phases upgrade this to logistic regression / GBM.

Every model exposes the same tiny interface: `.fit(observations)` then
`.predict(observation) -> int`.
"""
from __future__ import annotations

from statistics import mean

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
