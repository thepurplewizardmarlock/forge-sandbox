#!/usr/bin/env python3
"""Convert real public downloads into the project's CSV schema.

  # NASS Quick Stats crop-condition export -> condition_corn.csv
  python3 tools/convert.py condition nass_corn_condition.csv data/raw/condition_corn.csv

  # U.S. Drought Monitor export -> drought_corn.csv
  python3 tools/convert.py drought usdm_cornbelt.csv data/raw/drought_corn.csv --region "US CORN BELT"

  # A weekly demand level series -> a "vs. normal pace" surprise clue
  python3 tools/convert.py pace fas_corn_exports.csv data/raw/exports_corn.csv \
      --date-col week_ending --value-col commitments_pct --key CORN --key-column commodity \
      --metric EXPORT_PACE_SURPRISE

The WASDE consolidated historical CSV needs no conversion -- point --wasde-file at it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wasde_predictor import convert  # noqa: E402


def cmd_condition(args):
    rows = convert.nass_condition_to_rows(convert.read_csv(args.infile))
    convert.write_rows(args.outfile, rows, ["week_ending", "state", "metric", "value"])
    print(f"wrote {len(rows)} rows -> {args.outfile}")


def cmd_drought(args):
    rows = convert.drought_to_rows(convert.read_csv(args.infile), region=args.region,
                                   cumulative=not args.noncumulative)
    convert.write_rows(args.outfile, rows, ["week_ending", "region", "metric", "value"])
    print(f"wrote {len(rows)} rows -> {args.outfile}")


def cmd_pace(args):
    rows = convert.pace_surprise_rows(convert.read_csv(args.infile), date_col=args.date_col,
                                      value_col=args.value_col, key_column=args.key_column,
                                      key=args.key, metric=args.metric)
    convert.write_rows(args.outfile, rows, ["week_ending", args.key_column, "metric", "value"])
    print(f"wrote {len(rows)} rows -> {args.outfile}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("condition", help="NASS Quick Stats condition -> condition CSV")
    pc.add_argument("infile", type=Path); pc.add_argument("outfile", type=Path)
    pc.set_defaults(func=cmd_condition)

    pd = sub.add_parser("drought", help="U.S. Drought Monitor -> drought CSV")
    pd.add_argument("infile", type=Path); pd.add_argument("outfile", type=Path)
    pd.add_argument("--region", default="US CORN BELT")
    pd.add_argument("--noncumulative", action="store_true",
                    help="sum D2+D3+D4 instead of using the cumulative D2 column")
    pd.set_defaults(func=cmd_drought)

    pp = sub.add_parser("pace", help="weekly level series -> pace-surprise clue")
    pp.add_argument("infile", type=Path); pp.add_argument("outfile", type=Path)
    pp.add_argument("--date-col", required=True)
    pp.add_argument("--value-col", required=True)
    pp.add_argument("--key", required=True)
    pp.add_argument("--key-column", required=True)
    pp.add_argument("--metric", required=True)
    pp.set_defaults(func=cmd_pace)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
