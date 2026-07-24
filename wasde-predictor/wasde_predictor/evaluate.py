"""Honest evaluation: leave-one-marketing-year-out cross-validation.

We never test on a year we trained on. For each marketing year we hold it out,
train on all the *other* years, then predict the held-out year. This mimics real
life (you only ever have the past to learn from) and, with a small sample, is far
more honest than a single random split.

Metrics are plain accuracy overall and per report-month, so we can see where the
model helps (and where -- August especially -- it struggles).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

from .dataset import Observation

ModelFactory = Callable[[], object]  # returns an object with .fit()/.predict()


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def _years(observations: list[Observation]) -> list[str]:
    return sorted({o.market_year for o in observations})


def leave_one_year_out(
    observations: list[Observation], model_factory: ModelFactory
) -> dict:
    """Train on all-but-one marketing year; predict the held-out year. Repeat."""
    years = _years(observations)
    y_true: list[int] = []
    y_pred: list[int] = []
    per_month_true: dict[int, list[int]] = defaultdict(list)
    per_month_pred: dict[int, list[int]] = defaultdict(list)

    for held_out in years:
        train = [o for o in observations if o.market_year != held_out]
        test = [o for o in observations if o.market_year == held_out]
        if not train or not test:
            continue
        model = model_factory()
        model.fit(train)
        for o in test:
            p = model.predict(o)
            y_true.append(o.label)
            y_pred.append(p)
            per_month_true[o.month].append(o.label)
            per_month_pred[o.month].append(p)

    per_month = {
        m: {
            "n": len(per_month_true[m]),
            "accuracy": round(accuracy(per_month_true[m], per_month_pred[m]), 3),
        }
        for m in sorted(per_month_true)
    }
    return {
        "n": len(y_true),
        "accuracy": round(accuracy(y_true, y_pred), 3),
        "per_month": per_month,
    }


COIN_FLIP_ACCURACY = 0.5


def compare(observations: list[Observation], models: dict[str, ModelFactory]) -> dict:
    """Run leave-one-year-out for each named model and collect the results."""
    return {name: leave_one_year_out(observations, factory) for name, factory in models.items()}
