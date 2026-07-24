#!/usr/bin/env python3
"""wasde-predictor command line.

  python3 cli.py run            # score baselines vs. models (default; sample data)
  python3 cli.py predict-next   # forecast the next upcoming report from current clues

Both default to the bundled SYNTHETIC sample data. Point the --*-file options at
real USDA downloads for real results (see README -> "Getting real data").
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from wasde_predictor import evaluate, features as features_mod
from wasde_predictor import condition as condition_mod
from wasde_predictor import weather as weather_mod
from wasde_predictor.dataset import Observation, build_dataset, class_balance
from wasde_predictor.models import (
    ConditionThresholdModel,
    LogisticRegression,
    MajorityBaseline,
    PersistenceBaseline,
)
from wasde_predictor.regression import (
    MeanBaseline,
    PersistenceRegressor,
    RidgeRegression,
    ZeroBaseline,
)

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "data" / "sample"
S_YIELD = SAMPLE / "wasde_corn_yield_sample.csv"
S_COND = SAMPLE / "crop_condition_sample.csv"
S_DROUGHT = SAMPLE / "drought_sample.csv"
S_CUR_COND = SAMPLE / "current_condition_sample.csv"
S_CUR_DROUGHT = SAMPLE / "current_drought_sample.csv"

MONTH = {8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov"}
BAR = "=" * 68
DASH = "-" * 68


def _sample_note(is_sample: bool) -> None:
    if is_sample:
        print("  !! SYNTHETIC SAMPLE DATA -- results are illustrative only.")
        print("     On real USDA data expect scores much closer to the baseline;")
        print("     these revisions are genuinely hard (see README).")
        print(DASH)


def cmd_run(args: argparse.Namespace) -> None:
    is_sample = args.yield_file == S_YIELD
    obs = build_dataset(args.yield_file, args.condition_file, args.drought_file)
    bal = class_balance(obs)

    print(BAR)
    print("  WASDE corn yield-revision predictor  --  v1")
    print(BAR)
    _sample_note(is_sample)
    print(f"  Observations (Aug-Nov reports): {bal['n']}")
    print(f"  Marketing years              : {len({o.market_year for o in obs})}")
    print(f"  Class balance                : {bal['up']} up / {bal['down_or_flat']} down-or-flat"
          f"  (up-rate {bal['up_rate']})")
    print(DASH)

    results = evaluate.compare(obs, {
        "majority baseline": MajorityBaseline,
        "persistence baseline": PersistenceBaseline,
        "condition threshold": ConditionThresholdModel,
        "logistic regression": LogisticRegression,
    })
    print("  Leave-one-year-out accuracy:")
    print(f"    {'coin flip':22s}: {evaluate.COIN_FLIP_ACCURACY:.3f}")
    for name, res in results.items():
        print(f"    {name:22s}: {res['accuracy']:.3f}  (n={res['n']})")
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

    # The fair bar is the STRONGEST baseline (persistence usually wins here).
    baselines = {
        "coin flip": evaluate.COIN_FLIP_ACCURACY,
        "majority baseline": results["majority baseline"]["accuracy"],
        "persistence baseline": results["persistence baseline"]["accuracy"],
    }
    best_name = max(baselines, key=baselines.get)
    lift = log["accuracy"] - baselines[best_name]
    verdict = "beats" if lift > 0 else ("ties" if lift == 0 else "loses to")
    print(f"  Strongest baseline: {best_name} ({baselines[best_name]:.3f}).")
    print(f"  Logistic regression {verdict} it by {lift:+.3f}.")
    print(BAR)


def cmd_regress(args: argparse.Namespace) -> None:
    is_sample = args.yield_file == S_YIELD
    obs = build_dataset(args.yield_file, args.condition_file, args.drought_file)

    print(BAR)
    print("  WASDE corn yield-revision MAGNITUDE (bushels/acre)  --  v1")
    print(BAR)
    _sample_note(is_sample)

    results = evaluate.compare_regression(obs, {
        "zero baseline": ZeroBaseline,
        "mean baseline": MeanBaseline,
        "persistence": PersistenceRegressor,
        "ridge regression": RidgeRegression,
    })
    print("  Leave-one-year-out error (lower MAE/RMSE is better):")
    print(f"    {'model':22s}  {'MAE':>7s}  {'RMSE':>7s}  {'dir.acc':>7s}")
    for name, r in results.items():
        print(f"    {name:22s}  {r['mae']:7.3f}  {r['rmse']:7.3f}  {r['direction_accuracy']:7.3f}")
    print(DASH)

    best_naive = min(results["zero baseline"]["mae"], results["persistence"]["mae"])
    ridge_mae = results["ridge regression"]["mae"]
    impr = (best_naive - ridge_mae) / best_naive * 100 if best_naive else 0.0
    verb = "reduces" if impr > 0 else "increases"
    print(f"  Ridge {verb} MAE vs. the best naive baseline by {abs(impr):.1f}%.")
    print(BAR)


def cmd_predict_next(args: argparse.Namespace) -> None:
    is_sample = args.yield_file == S_YIELD
    obs = build_dataset(args.yield_file, args.condition_file, args.drought_file)
    model = LogisticRegression().fit(obs)

    conditions = condition_mod.load_condition(args.current_condition_file)
    droughts = weather_mod.load_drought(args.current_drought_file)
    report_date = dt.date.fromisoformat(args.report_date)
    prev_report_date = dt.date.fromisoformat(args.prev_report_date) if args.prev_report_date else None

    feats, weeks = features_mod.build_features(report_date, prev_report_date, conditions, droughts)
    if feats is None:
        raise SystemExit("Not enough current-season data before the report date to build features.")

    probe = Observation(market_year="pending", report_date=report_date, month=report_date.month,
                        label=0, change=0.0, features=feats, feature_weeks=tuple(weeks))
    proba = model.predict_proba(probe)
    direction = "UP" if proba >= 0.5 else "DOWN"
    confidence = proba if proba >= 0.5 else 1 - proba

    print(BAR)
    print(f"  Forecast for the {MONTH.get(report_date.month, report_date.month)} "
          f"{report_date.year} WASDE corn yield revision")
    print(BAR)
    _sample_note(is_sample)
    print(f"  Clues known as of {max(weeks).isoformat()} (all before {report_date.isoformat()}):")
    for k in features_mod.FEATURE_NAMES:
        print(f"    {k:26s}: {feats[k]:+.2f}")
    print(DASH)
    print(f"  Prediction: USDA will revise corn yield  >>> {direction} <<<")
    print(f"  P(up) = {proba:.3f}   (confidence {confidence:.0%})")
    print(BAR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    def add_data_args(p):
        p.add_argument("--yield-file", type=Path, default=S_YIELD)
        p.add_argument("--condition-file", type=Path, default=S_COND)
        p.add_argument("--drought-file", type=Path, default=S_DROUGHT)

    p_run = sub.add_parser("run", help="score baselines vs. models (direction)")
    add_data_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_reg = sub.add_parser("regress", help="score magnitude models (bushels/acre)")
    add_data_args(p_reg)
    p_reg.set_defaults(func=cmd_regress)

    p_next = sub.add_parser("predict-next", help="forecast the next upcoming report")
    add_data_args(p_next)
    p_next.add_argument("--current-condition-file", type=Path, default=S_CUR_COND)
    p_next.add_argument("--current-drought-file", type=Path, default=S_CUR_DROUGHT)
    p_next.add_argument("--report-date", default="2025-09-12", help="upcoming report date (YYYY-MM-DD)")
    p_next.add_argument("--prev-report-date", default="2025-08-12", help="previous report date (YYYY-MM-DD)")
    p_next.set_defaults(func=cmd_predict_next)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        # default to `run` on the sample data
        args = parser.parse_args(["run"])
    args.func(args)


if __name__ == "__main__":
    main()
