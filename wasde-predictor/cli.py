#!/usr/bin/env python3
"""wasde-predictor command line.

  python3 cli.py run                       # direction: baselines vs. models
  python3 cli.py regress                   # magnitude (units of the target)
  python3 cli.py predict-next              # forecast the next upcoming report

  # pick the target with --target (default: yield):
  python3 cli.py run --target ending-stocks

Everything defaults to the bundled SYNTHETIC sample data. Point the --*-file
options at real USDA downloads for real results (see README -> "Getting real data").
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from wasde_predictor import condition as condition_mod
from wasde_predictor import demand as demand_mod
from wasde_predictor import evaluate, features as features_mod
from wasde_predictor import sklearn_models
from wasde_predictor import weather as weather_mod
from wasde_predictor.dataset import Observation, build_dataset, class_balance
from wasde_predictor.models import (
    ConditionThresholdModel, LogisticRegression, MajorityBaseline, PersistenceBaseline,
)
from wasde_predictor.regression import MeanBaseline, PersistenceRegressor, RidgeRegression, ZeroBaseline

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "data" / "sample"
S_WASDE = SAMPLE / "wasde_corn_sample.csv"
S_COND = SAMPLE / "crop_condition_sample.csv"
S_DROUGHT = SAMPLE / "drought_sample.csv"
S_EXP = SAMPLE / "exports_sample.csv"
S_ETH = SAMPLE / "ethanol_sample.csv"
S_CUR_COND = SAMPLE / "current_condition_sample.csv"
S_CUR_DROUGHT = SAMPLE / "current_drought_sample.csv"
S_CUR_EXP = SAMPLE / "current_exports_sample.csv"
S_CUR_ETH = SAMPLE / "current_ethanol_sample.csv"

TARGETS = {
    "yield": {"attribute": "Yield per Harvested Acre", "unit": "bu/acre",
              "noun": "yield", "demand": False},
    "ending-stocks": {"attribute": "Ending Stocks", "unit": "million bu",
                      "noun": "ending stocks", "demand": True},
}

MONTH = {8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov"}
BAR = "=" * 68
DASH = "-" * 68


def _target(args) -> dict:
    return TARGETS[args.target]


def _dataset(args):
    t = _target(args)
    kw = dict(attribute=t["attribute"])
    if t["demand"]:
        kw["exports_path"] = args.exports_file
        kw["ethanol_path"] = args.ethanol_file
    return build_dataset(args.wasde_file, args.condition_file, args.drought_file, **kw)


def _sample_note(is_sample: bool) -> None:
    if is_sample:
        print("  !! SYNTHETIC SAMPLE DATA -- results are illustrative only.")
        print("     On real USDA data expect scores much closer to the baseline;")
        print("     these revisions are genuinely hard (see README).")
        print(DASH)


def cmd_run(args: argparse.Namespace) -> None:
    t = _target(args)
    obs = _dataset(args)
    bal = class_balance(obs)

    print(BAR)
    print(f"  WASDE corn {t['noun']}-revision DIRECTION predictor  --  v1")
    print(BAR)
    _sample_note(args.wasde_file == S_WASDE)
    print(f"  Observations (Aug-Nov reports): {bal['n']}")
    print(f"  Marketing years              : {len({o.market_year for o in obs})}")
    print(f"  Class balance                : {bal['up']} up / {bal['down_or_flat']} down-or-flat"
          f"  (up-rate {bal['up_rate']})")
    print(DASH)

    models = {
        "majority baseline": MajorityBaseline,
        "persistence baseline": PersistenceBaseline,
        "condition threshold": ConditionThresholdModel,
        "logistic regression": LogisticRegression,
    }
    if sklearn_models.HAS_SKLEARN:
        models["gradient boosting"] = sklearn_models.GBClassifier
    results = evaluate.compare(obs, models)

    print("  Leave-one-year-out accuracy:")
    print(f"    {'coin flip':22s}: {evaluate.COIN_FLIP_ACCURACY:.3f}")
    for name, res in results.items():
        print(f"    {name:22s}: {res['accuracy']:.3f}  (n={res['n']})")
    if not sklearn_models.HAS_SKLEARN:
        print(f"    {'gradient boosting':22s}: (skipped -- pip install scikit-learn)")
    print(DASH)

    log = results["logistic regression"]
    print("  Logistic regression, 'up' class:")
    print(f"    precision {log['precision']:.3f}   recall {log['recall']:.3f}   F1 {log['f1']:.3f}")
    c = log["confusion"]
    print(f"    confusion: tp={c['tp']} fp={c['fp']} tn={c['tn']} fn={c['fn']}")
    print("  Accuracy by report month:")
    for m, s in log["per_month"].items():
        print(f"    {MONTH.get(m, m):>3s}: {s['accuracy']:.3f}  (n={s['n']})")
    print(DASH)

    importance = LogisticRegression().fit(obs).coefficients()
    print("  Feature importance (standardized logistic weights; + => higher = more 'up'):")
    for name, w in sorted(importance.items(), key=lambda kv: -abs(kv[1])):
        print(f"    {name:26s}: {w:+.3f}")
    print(DASH)

    baselines = {
        "coin flip": evaluate.COIN_FLIP_ACCURACY,
        "majority baseline": results["majority baseline"]["accuracy"],
        "persistence baseline": results["persistence baseline"]["accuracy"],
    }
    best = max(baselines, key=baselines.get)
    lift = log["accuracy"] - baselines[best]
    verdict = "beats" if lift > 0 else ("ties" if lift == 0 else "loses to")
    print(f"  Strongest baseline: {best} ({baselines[best]:.3f}).")
    print(f"  Logistic regression {verdict} it by {lift:+.3f}.")
    print(BAR)


def cmd_regress(args: argparse.Namespace) -> None:
    t = _target(args)
    obs = _dataset(args)

    print(BAR)
    print(f"  WASDE corn {t['noun']}-revision MAGNITUDE ({t['unit']})  --  v1")
    print(BAR)
    _sample_note(args.wasde_file == S_WASDE)

    reg_models = {
        "zero baseline": ZeroBaseline,
        "mean baseline": MeanBaseline,
        "persistence": PersistenceRegressor,
        "ridge regression": RidgeRegression,
    }
    if sklearn_models.HAS_SKLEARN:
        reg_models["gradient boosting"] = sklearn_models.GBRegressor
    results = evaluate.compare_regression(obs, reg_models)

    print("  Leave-one-year-out error (lower MAE/RMSE is better):")
    print(f"    {'model':22s}  {'MAE':>8s}  {'RMSE':>8s}  {'dir.acc':>7s}")
    for name, r in results.items():
        print(f"    {name:22s}  {r['mae']:8.3f}  {r['rmse']:8.3f}  {r['direction_accuracy']:7.3f}")
    print(DASH)

    best_naive = min(results["zero baseline"]["mae"], results["persistence"]["mae"])
    ridge_mae = results["ridge regression"]["mae"]
    impr = (best_naive - ridge_mae) / best_naive * 100 if best_naive else 0.0
    verb = "reduces" if impr > 0 else "increases"
    print(f"  Ridge {verb} MAE vs. the best naive baseline by {abs(impr):.1f}%.")
    print(BAR)


def cmd_predict_next(args: argparse.Namespace) -> None:
    t = _target(args)
    obs = _dataset(args)
    model = LogisticRegression().fit(obs)

    conditions = condition_mod.load_condition(args.current_condition_file)
    droughts = weather_mod.load_drought(args.current_drought_file)
    exports = demand_mod.load_exports(args.current_exports_file) if t["demand"] else None
    ethanol = demand_mod.load_ethanol(args.current_ethanol_file) if t["demand"] else None

    report_date = dt.date.fromisoformat(args.report_date)
    prev_report_date = dt.date.fromisoformat(args.prev_report_date) if args.prev_report_date else None
    feats, weeks = features_mod.build_all_features(
        report_date, prev_report_date, conditions, droughts,
        exports=exports, ethanol=ethanol, include_demand=t["demand"],
    )
    if feats is None:
        raise SystemExit("Not enough current-season data before the report date to build features.")

    probe = Observation(market_year="pending", report_date=report_date, month=report_date.month,
                        label=0, change=0.0, features=feats, feature_weeks=tuple(weeks))
    proba = model.predict_proba(probe)
    direction = "UP" if proba >= 0.5 else "DOWN"
    confidence = proba if proba >= 0.5 else 1 - proba

    print(BAR)
    print(f"  Forecast: {MONTH.get(report_date.month, report_date.month)} {report_date.year} "
          f"WASDE corn {t['noun']} revision")
    print(BAR)
    _sample_note(args.wasde_file == S_WASDE)
    print(f"  Clues known as of {max(weeks).isoformat()} (all before {report_date.isoformat()}):")
    for k in sorted(feats):
        print(f"    {k:26s}: {feats[k]:+.2f}")
    print(DASH)
    print(f"  Prediction: USDA will revise corn {t['noun']}  >>> {direction} <<<")
    print(f"  P(up) = {proba:.3f}   (confidence {confidence:.0%})")
    print(BAR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    def add_data_args(p):
        p.add_argument("--target", choices=sorted(TARGETS), default="yield")
        p.add_argument("--wasde-file", type=Path, default=S_WASDE)
        p.add_argument("--condition-file", type=Path, default=S_COND)
        p.add_argument("--drought-file", type=Path, default=S_DROUGHT)
        p.add_argument("--exports-file", type=Path, default=S_EXP)
        p.add_argument("--ethanol-file", type=Path, default=S_ETH)

    p_run = sub.add_parser("run", help="score direction baselines vs. models")
    add_data_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_reg = sub.add_parser("regress", help="score magnitude models")
    add_data_args(p_reg)
    p_reg.set_defaults(func=cmd_regress)

    p_next = sub.add_parser("predict-next", help="forecast the next upcoming report")
    add_data_args(p_next)
    p_next.add_argument("--current-condition-file", type=Path, default=S_CUR_COND)
    p_next.add_argument("--current-drought-file", type=Path, default=S_CUR_DROUGHT)
    p_next.add_argument("--current-exports-file", type=Path, default=S_CUR_EXP)
    p_next.add_argument("--current-ethanol-file", type=Path, default=S_CUR_ETH)
    p_next.add_argument("--report-date", default="2025-09-12")
    p_next.add_argument("--prev-report-date", default="2025-08-12")
    p_next.set_defaults(func=cmd_predict_next)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        args = parser.parse_args(["run"])
    args.func(args)


if __name__ == "__main__":
    main()
