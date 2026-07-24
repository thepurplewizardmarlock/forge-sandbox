"""Honest evaluation: leave-one-marketing-year-out cross-validation.

We never test on a year we trained on. For each marketing year we hold it out,
train on all the *other* years, then predict the held-out year. This mimics real
life (you only ever have the past to learn from) and, with a small sample, is far
more honest than a single random split.

Reported metrics: overall accuracy, precision/recall/F1 for the "up" class, a
confusion matrix, and per-report-month accuracy so we can see where the model
helps (and where -- August especially -- it struggles).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

from .dataset import Observation

ModelFactory = Callable[[], object]  # returns an object with .fit()/.predict()
COIN_FLIP_ACCURACY = 0.5


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)


def confusion(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_f1(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    c = confusion(y_true, y_pred)
    prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def _years(observations: list[Observation]) -> list[str]:
    return sorted({o.market_year for o in observations})


def leave_one_year_out(observations: list[Observation], model_factory: ModelFactory) -> dict:
    """Train on all-but-one marketing year; predict the held-out year. Repeat."""
    y_true: list[int] = []
    y_pred: list[int] = []
    per_month_true: dict[int, list[int]] = defaultdict(list)
    per_month_pred: dict[int, list[int]] = defaultdict(list)

    for held_out in _years(observations):
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
        m: {"n": len(per_month_true[m]),
            "accuracy": round(accuracy(per_month_true[m], per_month_pred[m]), 3)}
        for m in sorted(per_month_true)
    }
    result = {
        "n": len(y_true),
        "accuracy": round(accuracy(y_true, y_pred), 3),
        "confusion": confusion(y_true, y_pred),
        "per_month": per_month,
    }
    result.update(precision_recall_f1(y_true, y_pred))
    return result


def compare(observations: list[Observation], models: dict[str, ModelFactory]) -> dict:
    return {name: leave_one_year_out(observations, factory) for name, factory in models.items()}


# --- magnitude (regression) metrics -----------------------------------------

def mae(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true:
        return 0.0
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true:
        return 0.0
    return (sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)) ** 0.5


def leave_one_year_out_regression(
    observations: list[Observation], model_factory: ModelFactory
) -> dict:
    """Leave-one-year-out for a magnitude model; reports MAE, RMSE, and the
    directional accuracy implied by the sign of the prediction."""
    y_true: list[float] = []
    y_pred: list[float] = []
    dir_true: list[int] = []
    dir_pred: list[int] = []
    for held_out in _years(observations):
        train = [o for o in observations if o.market_year != held_out]
        test = [o for o in observations if o.market_year == held_out]
        if not train or not test:
            continue
        model = model_factory()
        model.fit(train)
        for o in test:
            p = model.predict(o)
            y_true.append(o.change)
            y_pred.append(p)
            dir_true.append(o.label)
            dir_pred.append(1 if p > 0 else 0)
    return {
        "n": len(y_true),
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "direction_accuracy": round(accuracy(dir_true, dir_pred), 3),
    }


def compare_regression(observations: list[Observation], models: dict[str, ModelFactory]) -> dict:
    return {name: leave_one_year_out_regression(observations, f) for name, f in models.items()}
