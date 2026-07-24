# wasde-predictor

Predict the **direction of USDA's corn yield revision** — will next month's report
raise or lower the yield estimate? — using only free, public clues known *before*
the report comes out.

**v1**: a complete, honest end-to-end pipeline — two data clues (crop condition +
drought), a logistic-regression model with feature importance, careful
leave-one-year-out evaluation against sensible baselines, and a `predict-next`
command. It runs on **synthetic sample data** with **zero dependencies**, then
you point it at real USDA files.

## 60-second background (no ag knowledge needed)

- Think of all US corn as one **pantry**. Farmers fill it at harvest; the country
  empties it over the year (feed, food, ethanol, exports). How full it ends up —
  **ending stocks** — is the market's most-watched number.
- Monthly, USDA publishes **WASDE**, its official estimate of that pantry math, at
  exactly **noon ET**, and **revises** it as new info arrives. Traders bet on the
  *change*.
- The biggest driver of the change is **yield** (bushels per acre). We predict the
  direction USDA moves its yield estimate.
- **The catch:** USDA's new yield lands at the *same instant* as the report, so we
  can't see it early. We forecast from clues that *are* public beforehand — mainly
  **crop condition** (weekly "% good/excellent") and **drought** coverage.

We focus on the **August–November** reports, the only window where USDA uses real
survey yields (so the clues matter); earlier it just holds yield at a fixed trend.

## Quick start

```bash
cd wasde-predictor
python3 cli.py run                              # direction: baselines vs. models
python3 cli.py regress                          # magnitude (units of the target)
python3 cli.py predict-next                     # forecast the next upcoming report
python3 -m unittest discover -s tests -t .      # run the tests

# choose the commodity and target:
python3 cli.py run --commodity soybeans         # corn | soybeans | wheat
python3 cli.py run --commodity corn --target ending-stocks   # the headline number
```

All commands default to the bundled **synthetic** sample data.

### Three commodities

`--commodity corn|soybeans|wheat` (default corn). Corn and soybeans use the
August–November survey-yield window. **Wheat uses its own May–September window**
(winter-wheat harvest + the late-September Small Grains Summary); it still blends
winter and spring wheat into one series, so it's a simplification and the CLI
prints a note when you select it.

### Two targets

- **`--target yield`** (default) — predicts USDA's corn *yield* revision from the
  supply clues (condition, drought). Yield is the biggest driver of the pantry.
- **`--target ending-stocks`** — predicts the *ending stocks* revision (the number
  traders actually watch), for any commodity. Ending stocks = supply − demand, so
  this target adds **per-commodity demand clues**: corn → exports + ethanol grind;
  soybeans → exports + crush; wheat → exports.

## How to read the `run` output

```
  Leave-one-year-out accuracy:
    coin flip             : 0.500
    majority baseline     : 0.167   <- misleading here (see note); ignore
    persistence baseline  : 0.604   <- the honest bar to beat
    condition threshold   : 0.792
    logistic regression   : 0.833   <- our v1 model
```

- **Leave-one-year-out**: each marketing year is predicted by a model trained only
  on the *other* years — no peeking.
- **Why the majority baseline looks broken (0.167):** a single season's revisions
  are mostly the same direction (a good crop gets raised in Aug *and* Sep *and*
  Oct). So when you hold out a year, the training majority flips to the *opposite*
  of that year — the majority baseline ends up anti-correlated. That's why we use
  the **persistence baseline** (predict the previous report's direction) as the
  fair bar: it's the honest thing to beat for an autocorrelated target.
- **Feature importance** prints the standardized logistic weights: `condition_ge`
  positive (healthier crop → more likely "up"), `drought_d2plus` negative (more
  drought → less likely "up") — which is exactly the real-world direction.
- On this **synthetic** data the model beats persistence handily *because the fake
  data has a clean built-in signal*. **On real USDA data expect a much smaller
  edge** — these revisions are genuinely hard (the main driver, yield, is hidden
  until release). Beating persistence by a little, honestly, is the real goal.

## `predict-next`

Trains on all history, then forecasts a single upcoming report from the
current-season clues:

```
  Prediction: USDA will revise corn yield  >>> DOWN <<<
  P(up) = 0.025   (confidence 97%)
```

## Getting real data (replaces the synthetic sample)

USDA blocks automated downloads, so grab these by hand in a browser, then pass
them in:

1. **Yield history** — USDA *Consolidated Historical WASDE Report Data* CSV
   (stores values *as first published* — exactly what we need). Confirm the
   columns match `wasde._COLUMNS`.
2. **Crop condition** — NASS "Quick Stats": corn CONDITION, "% GOOD" + "% EXCELLENT"
   summed, US total, weekly → `week_ending, state, value`.
3. **Drought** — U.S. Drought Monitor, % of a corn-belt region in D2+ drought,
   weekly → `week_ending, region, value`.
4. **(ending-stocks target only) Exports** — USDA FAS weekly export sales, as a
   pace-vs-normal surprise → `week_ending, commodity, value`.
5. **(ending-stocks target only) Ethanol** — EIA weekly ethanol production, as a
   grind-vs-normal surprise → `week_ending, region, value`.

Save them into a folder (e.g. `data/raw/`) using the **same base names** as the
sample files — `wasde_corn.csv`, `condition_corn.csv`, `drought_corn.csv`,
`exports_corn.csv`, `ethanol_corn.csv`, etc. — then point `--data-dir` at it:

```bash
python3 cli.py run --commodity corn --target ending-stocks --data-dir data/raw
```

Individual files can still be overridden with `--wasde-file`/`--condition-file`/
`--drought-file`. Anything under `data/raw/` is git-ignored, so downloads stay local.

### Converters (`tools/convert.py`)

Real NASS / Drought-Monitor / demand exports don't match our simple schema, so
convert them first (the WASDE consolidated CSV needs no conversion — point
`--wasde-file` straight at it):

```bash
# NASS Quick Stats crop condition -> condition_corn.csv (sums % good + % excellent)
python3 tools/convert.py condition nass_corn_condition.csv data/raw/condition_corn.csv

# U.S. Drought Monitor -> drought_corn.csv (% area in D2+)
python3 tools/convert.py drought usdm_cornbelt.csv data/raw/drought_corn.csv --region "US CORN BELT"

# a weekly demand level (FAS exports, EIA ethanol, NOPA crush) -> a pace-surprise clue
python3 tools/convert.py pace fas_corn_exports.csv data/raw/exports_corn.csv \
    --date-col week_ending --value-col commitments_pct \
    --key CORN --key-column commodity --metric EXPORT_PACE_SURPRISE
```

## Glossary

| Term | Plain meaning |
|---|---|
| **WASDE** | USDA's monthly supply/demand report; the market-moving scorecard |
| **Yield** | Bushels of corn per acre — the biggest driver of supply |
| **Marketing year** | Corn's "fiscal year": Sept 1 → Aug 31 |
| **Crop condition** | USDA's weekly "% good/excellent" field rating |
| **D2+ drought** | Drought Monitor category "severe" or worse |
| **Revision** | Month-to-month change in a USDA estimate — what we predict |
| **Persistence baseline** | Predict that this report moves the same way the last one did |
| **Leave-one-year-out** | Test each year with a model trained only on the others |

## Project layout

```
wasde-predictor/
  cli.py                     # `run` (scoreboard) and `predict-next`
  wasde_predictor/
    commodities.py           # config: each commodity's demand clues + report window
    wasde.py                 # load any WASDE attribute (yield / ending stocks) + revisions
    series.py                # generic weekly series + point-in-time (leak-safe) accessors
    condition.py             # crop-condition loader
    weather.py               # drought loader
    features.py              # build the supply (+demand) clue vector (leak-safe)
    dataset.py               # join target + clues into observations (target-selectable)
    models.py                # majority + persistence baselines; threshold + logistic
    regression.py            # magnitude models (zero/mean/persistence + ridge)
    sklearn_models.py        # OPTIONAL gradient-boosted trees (classifier + regressor)
    convert.py               # turn real downloads into the project CSV schema
    evaluate.py              # accuracy, precision/recall/F1, confusion, MAE/RMSE, LOYO
  tools/convert.py           # CLI for the converters above
  tools/make_sample_data.py  # regenerate the synthetic sample (deterministic)
  data/sample/               # synthetic CSVs (committed, clearly fake)
  data/raw/                  # your real USDA downloads (git-ignored)
  tests/                     # 36 unittest cases, incl. the no-leakage guards
```

## Roadmap

1. ✅ Walking skeleton — end-to-end loop, one clue, honest baseline.
2. ✅ Real crop-condition features (level, momentum, season trajectory).
3. ✅ Second clue: Corn Belt drought.
4. ✅ Logistic-regression model + feature importance + persistence baseline + `predict-next`.
5. ✅ Predict *magnitude* (bushels/acre) via `regress`, not just direction.
6. ✅ Optional scikit-learn gradient boosting (`pip install scikit-learn`) — and
   the honest finding that on this small sample it does *not* beat the plain
   logistic/ridge models (it overfits).
7. ✅ Full **ending stocks** target (`--target ending-stocks`) with demand clues
   (export-pace + ethanol-pace surprises).
8. ✅ **Soybeans and wheat** via `--commodity`.
9. ✅ **Per-commodity demand clues** (corn ethanol, soybean crush, wheat exports)
   — ending stocks now works for every commodity, via a central `commodities.py`.
10. ✅ **Wheat-appropriate May–Sep window** (winter-wheat harvest + Small Grains Summary).
11. ✅ **Real-data converters** (`tools/convert.py`) for NASS condition, Drought
    Monitor, and generic weekly→pace-surprise demand series.

### Beyond v1 (natural next steps)

- Separate winter vs. spring wheat instead of one blended series.
- Scripted fetchers (keyed EIA / NASS / CFTC APIs) where a source's terms allow it.

> Note: the data in `data/sample/` is **synthetic and for testing only** — it is
> not real USDA data and must not be read as such.
