#!/usr/bin/env python3
"""wasde-predictor command line.

  python3 cli.py run                       # direction: baselines vs. models
  python3 cli.py regress                   # magnitude (units of the target)
  python3 cli.py predict-next              # forecast the next upcoming report

  # pick the commodity and target:
  python3 cli.py run --commodity soybeans
  python3 cli.py run --commodity corn --target ending-stocks

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

COMMODITIES = ["corn", "soybeans", "wheat"]
TARGETS = {
    "yield": {"attribute": "Yield per Harvested Acre", "unit": "bu/acre",
              "noun": "yield", "demand": False},
    "ending-stocks": {"attribute": "Ending Stocks", "unit": "million bu",
                      "noun": "ending stocks", "demand": True},
}
MONTH = {8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov"}
BAR = "=" * 68
DASH = "-" * 68


def _defaults(commodity: str) -> dict:
    return {
        "wasde": SAMPLE / f"wasde_{commodity}_sample.csv",
        "condition": SAMPLE / f"condition_{commodity}_sample.csv",
        "drought": SAMPLE / f"drought_{commodity}_sample.csv",
        "exports": SAMPLE / "exports_corn_sample.csv",
        "ethanol": SAMPLE / "ethanol_corn_sample.csv",
        "cur_condition": SAMPLE / f"current_condition_{commodity}_sample.csv",
        "cur_drought": SAMPLE / f"current_drought_{commodity}_sample.csv",
        "cur_exports": SAMPLE / "current_exports_corn_sample.csv",
        "cur_ethanol": SAMPLE / "current_ethanol_corn_sample.csv",
    }


def _resolve(args):
    """Resolve file paths (explicit overrides win over per-commodity defaults)."""
    d = _defaults(args.commodity)
    t = TARGETS[args.target]
    if t["demand"] and args.commodity != "corn":
        raise SystemExit(
            f"--target ending-stocks is only wired for corn (its demand clues); "
            f"'{args.commodity}' supports --target yield.")
    is_sample = getattr(args, "wasde_file", None) is None
    files = {
        "wasde": args.wasde_file or d["wasde"],
        "condition": args.condition_file or d["condition"],
        "drought": args.drought_file or d["drought"],
        "exports": getattr(args, "exports_file", None) or d["exports"],
        "ethanol": getattr(args, "ethanol_file", None) or d["ethanol"],
    }
    return files, t, is_sample, d


def _dataset(files, t, commodity):
    kw = dict(commodity=commodity.capitalize(), attribute=t["attribute"])
    if t["demand"]:
        kw["exports_path"] = files["exports"]
        kw["ethanol_path"] = files["ethanol"]
    return build_dataset(files["wasde"], files["condition"], files["drought"], **kw)


def _sample_note(is_sample: bool) -> None:
    if is_sample:
        print("  !! SYNTHETIC SAMPLE DATA -- results are illustrative only.")
        print("     On real USDA data expect scores much closer to the baseline;")
        print("     these revisions are genuinely hard (see README).")
        print(DASH)


def _title(commodity, t, kind):
    return f"  WASDE {commodity} {t['noun']}-revision {kind}  --  v1"


def cmd_run(args):
    files, t, is_sample, _ = _resolve(args)
    obs = _dataset(files, t, args.commodity)
    bal = class_balance(obs)

    print(BAR); print(_title(args.commodity, t, "DIRECTION")); print(BAR)
    _sample_note(is_sample)
    if args.commodity == "wheat":
        print("  NOTE: wheat is an APPROXIMATE extension (its real yield window differs).")
        print(DASH)
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


def cmd_regress(args):
    files, t, is_sample, _ = _resolve(args)
    obs = _dataset(files, t, args.commodity)

    print(BAR); print(_title(args.commodity, t, f"MAGNITUDE ({t['unit']})")); print(BAR)
    _sample_note(is_sample)

    reg_models = {
        "zero baseline": ZeroBaseline, "mean baseline": MeanBaseline,
        "persistence": PersistenceRegressor, "ridge regression": RidgeRegression,
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
    print(f"  Ridge {'reduces' if impr > 0 else 'increases'} MAE vs. the best naive baseline "
          f"by {abs(impr):.1f}%.")
    print(BAR)


def cmd_predict_next(args):
    files, t, is_sample, d = _resolve(args)
    obs = _dataset(files, t, args.commodity)
    model = LogisticRegression().fit(obs)

    conditions = condition_mod.load_condition(args.current_condition_file or d["cur_condition"])
    droughts = weather_mod.load_drought(args.current_drought_file or d["cur_drought"])
    exports = demand_mod.load_exports(args.current_exports_file or d["cur_exports"]) if t["demand"] else None
    ethanol = demand_mod.load_ethanol(args.current_ethanol_file or d["cur_ethanol"]) if t["demand"] else None

    report_date = dt.date.fromisoformat(args.report_date)
    prev_report_date = dt.date.fromisoformat(args.prev_report_date) if args.prev_report_date else None
    feats, weeks = features_mod.build_all_features(
        report_date, prev_report_date, conditions, droughts,
        exports=exports, ethanol=ethanol, include_demand=t["demand"])
    if feats is None:
        raise SystemExit("Not enough current-season data before the report date to build features.")

    probe = Observation(market_year="pending", report_date=report_date, month=report_date.month,
                        label=0, change=0.0, features=feats, feature_weeks=tuple(weeks))
    proba = model.predict_proba(probe)
    direction = "UP" if proba >= 0.5 else "DOWN"
    confidence = proba if proba >= 0.5 else 1 - proba

    print(BAR)
    print(f"  Forecast: {MONTH.get(report_date.month, report_date.month)} {report_date.year} "
          f"WASDE {args.commodity} {t['noun']} revision")
    print(BAR)
    _sample_note(is_sample)
    print(f"  Clues known as of {max(weeks).isoformat()} (all before {report_date.isoformat()}):")
    for k in sorted(feats):
        print(f"    {k:26s}: {feats[k]:+.2f}")
    print(DASH)
    print(f"  Prediction: USDA will revise {args.commodity} {t['noun']}  >>> {direction} <<<")
    print(f"  P(up) = {proba:.3f}   (confidence {confidence:.0%})")
    print(BAR)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    def add_data_args(p):
        p.add_argument("--commodity", choices=COMMODITIES, default="corn")
        p.add_argument("--target", choices=sorted(TARGETS), default="yield")
        p.add_argument("--wasde-file", type=Path, default=None)
        p.add_argument("--condition-file", type=Path, default=None)
        p.add_argument("--drought-file", type=Path, default=None)
        p.add_argument("--exports-file", type=Path, default=None)
        p.add_argument("--ethanol-file", type=Path, default=None)

    p_run = sub.add_parser("run", help="score direction baselines vs. models")
    add_data_args(p_run); p_run.set_defaults(func=cmd_run)

    p_reg = sub.add_parser("regress", help="score magnitude models")
    add_data_args(p_reg); p_reg.set_defaults(func=cmd_regress)

    p_next = sub.add_parser("predict-next", help="forecast the next upcoming report")
    add_data_args(p_next)
    p_next.add_argument("--current-condition-file", type=Path, default=None)
    p_next.add_argument("--current-drought-file", type=Path, default=None)
    p_next.add_argument("--current-exports-file", type=Path, default=None)
    p_next.add_argument("--current-ethanol-file", type=Path, default=None)
    p_next.add_argument("--report-date", default="2025-09-12")
    p_next.add_argument("--prev-report-date", default="2025-08-12")
    p_next.set_defaults(func=cmd_predict_next)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        args = parser.parse_args(["run"])
    args.func(args)


if __name__ == "__main__":
    main()
