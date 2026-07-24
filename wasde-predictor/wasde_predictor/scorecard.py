"""Evaluate every commodity x target at once -- the one-glance summary.

For each (commodity, target) it builds the dataset from a data directory, runs the
logistic model and the baselines under leave-one-year-out, and reports the model
accuracy, the strongest baseline, and the lift. Pure/importable so it's testable;
the CLI `scorecard` command just prints the returned rows as a table.
"""
from __future__ import annotations

from pathlib import Path

from . import commodities, evaluate
from .dataset import build_dataset
from .models import LogisticRegression, MajorityBaseline, PersistenceBaseline

TARGETS = {
    "yield": ("Yield per Harvested Acre", False),
    "ending-stocks": ("Ending Stocks", True),
}


def _build(data_dir: Path, slug: str, target_key: str):
    c = commodities.get(slug)
    attribute, demand = TARGETS[target_key]
    kw = dict(commodity=c.label, attribute=attribute, target_months=c.report_months)
    if demand:
        kw["demand"] = [(clue, data_dir / f"{clue.basename}.csv") for clue in c.demand_clues]
    return build_dataset(
        data_dir / f"wasde_{slug}.csv",
        data_dir / f"condition_{slug}.csv",
        data_dir / f"drought_{slug}.csv",
        **kw,
    )


def evaluate_all(data_dir: str | Path) -> list[dict]:
    data_dir = Path(data_dir)
    rows: list[dict] = []
    for slug in commodities.ALL:
        for target_key in ("yield", "ending-stocks"):
            obs = _build(data_dir, slug, target_key)
            res = evaluate.compare(obs, {
                "logistic": LogisticRegression,
                "majority": MajorityBaseline,
                "persistence": PersistenceBaseline,
            })
            best = max(res["majority"]["accuracy"], res["persistence"]["accuracy"],
                       evaluate.COIN_FLIP_ACCURACY)
            rows.append({
                "commodity": slug,
                "target": target_key,
                "n": res["logistic"]["n"],
                "logistic": res["logistic"]["accuracy"],
                "best_baseline": round(best, 3),
                "lift": round(res["logistic"]["accuracy"] - best, 3),
            })
    return rows
