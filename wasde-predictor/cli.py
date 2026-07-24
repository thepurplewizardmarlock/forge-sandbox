#!/usr/bin/env python3
"""wasde-predictor command line: run the whole walking-skeleton loop.

    python3 cli.py                       # runs on the bundled SYNTHETIC sample data
    python3 cli.py --yield-file X.csv --condition-file Y.csv   # your real data

It loads the data, builds observations, and prints an honest scoreboard: the
no-skill majority baseline vs. the one-feature condition model, using
leave-one-marketing-year-out testing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wasde_predictor import evaluate
from wasde_predictor.dataset import build_dataset, class_balance
from wasde_predictor.models import ConditionThresholdModel, MajorityBaseline

HERE = Path(__file__).resolve().parent
SAMPLE_YIELD = HERE / "data" / "sample" / "wasde_corn_yield_sample.csv"
SAMPLE_CONDITION = HERE / "data" / "sample" / "crop_condition_sample.csv"

MONTH_NAME = {8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov"}


def run(yield_file: Path, condition_file: Path, is_sample: bool) -> None:
    observations = build_dataset(yield_file, condition_file)
    bal = class_balance(observations)

    print("=" * 66)
    print("  WASDE corn yield-revision predictor  --  walking skeleton (v1)")
    print("=" * 66)
    if is_sample:
        print("  !! SYNTHETIC SAMPLE DATA -- results are illustrative only.")
        print("     Point --yield-file/--condition-file at real USDA downloads")
        print("     for real numbers (see README -> 'Getting real data').")
        print("-" * 66)

    print(f"  Observations (Aug-Nov reports): {bal['n']}")
    print(f"  Marketing years              : {len({o.market_year for o in observations})}")
    print(f"  Class balance                : {bal['up']} up / {bal['down_or_flat']} down-or-flat"
          f"  (up-rate {bal['up_rate']})")
    print("-" * 66)

    results = evaluate.compare(
        observations,
        {"majority baseline": MajorityBaseline, "condition model": ConditionThresholdModel},
    )

    print("  Leave-one-year-out accuracy:")
    print(f"    coin flip           : {evaluate.COIN_FLIP_ACCURACY:.3f}")
    for name, res in results.items():
        print(f"    {name:20s}: {res['accuracy']:.3f}  (n={res['n']})")
    print("-" * 66)

    print("  Condition model, accuracy by report month:")
    for month, stats in results["condition model"]["per_month"].items():
        print(f"    {MONTH_NAME.get(month, month):>3s}: {stats['accuracy']:.3f}  (n={stats['n']})")

    lift = results["condition model"]["accuracy"] - results["majority baseline"]["accuracy"]
    print("-" * 66)
    verdict = "beats" if lift > 0 else ("ties" if lift == 0 else "loses to")
    print(f"  Condition model {verdict} the majority baseline by {lift:+.3f}.")
    print("=" * 66)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yield-file", type=Path, default=SAMPLE_YIELD,
                        help="WASDE-style yield CSV (default: bundled sample)")
    parser.add_argument("--condition-file", type=Path, default=SAMPLE_CONDITION,
                        help="NASS-style crop-condition CSV (default: bundled sample)")
    args = parser.parse_args()

    is_sample = (args.yield_file == SAMPLE_YIELD and args.condition_file == SAMPLE_CONDITION)
    run(args.yield_file, args.condition_file, is_sample)


if __name__ == "__main__":
    main()
