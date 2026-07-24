# wasde-predictor

Predict the **direction of USDA's corn yield revision** — will next month's report
raise or lower the yield estimate? — using only free, public clues known *before*
the report comes out.

This is **v1, a "walking skeleton"**: the whole pipeline exists end-to-end
(load data → build features → baseline → model → honest score), but it is
deliberately small. It runs on **synthetic sample data** with **zero
dependencies** so you can see the loop work immediately, then you point it at
real USDA files.

## 60-second background (no ag knowledge needed)

- Think of all US corn as one **pantry**. Farmers fill it at harvest; the country
  empties it over the year (animal feed, food, ethanol, exports). How full it
  ends up — **ending stocks** — is the market's most-watched number.
- Once a month, USDA publishes **WASDE**, its official estimate of that pantry
  math, at exactly **noon ET**. Every month it **revises** the estimate as new
  info arrives. Traders bet on the *change*.
- The biggest driver of the change is **yield** (bushels of corn per acre). We
  predict the direction USDA moves its yield estimate.
- **The catch:** USDA's new yield number is released at the *same instant* as the
  report, so we can't see it early. We forecast from clues that *are* public
  beforehand — mainly **crop condition** (USDA's weekly "% good/excellent" rating
  of how the crop looks).

We focus on the **August–November** reports, because that's the only window where
USDA uses real survey yields (so the clues actually matter); earlier in the year
it just holds yield at a fixed trend.

## Quick start

```bash
cd wasde-predictor
python3 cli.py                       # runs on bundled SYNTHETIC sample data
python3 -m unittest discover -s tests -t .   # run the tests
```

## How to read the output

```
  Class balance         : 27 up / 21 down-or-flat  (up-rate 0.562)
  Leave-one-year-out accuracy:
    coin flip           : 0.500
    majority baseline   : 0.562   <- the "no skill" bar to beat
    condition model     : 0.896   <- our one-feature model
```

- **coin flip (0.50)** and **majority baseline** are the bars. The majority
  baseline just always guesses the most common outcome — any real model must beat
  it to be worth anything.
- **Leave-one-year-out** means we test each marketing year using a model trained
  only on the *other* years — no peeking at the answer.
- On the **synthetic** data the model looks great (~0.90) *because the fake data
  has a clean built-in relationship*. **On real USDA data expect it to be much
  closer to the baseline** — these revisions are genuinely hard to predict (the
  main driver, yield, is hidden until release). Beating the baseline by even a
  little, honestly, is the real goal.

## Getting real data (replaces the synthetic sample)

USDA blocks automated downloads, so grab these by hand in a browser, then pass
them in:

1. **Yield history** — USDA's *Consolidated Historical WASDE Report Data* CSV
   (search "USDA Historical WASDE Report Data"). It stores values *as first
   published*, which is exactly what we need. Open it once and confirm the column
   names match `wasde._COLUMNS` (Commodity, Region, Attribute, MarketYear,
   ReportDate, Value); adjust if USDA's spellings differ.
2. **Crop condition** — USDA NASS "Quick Stats": corn, "CONDITION", "PCT
   GOOD" + "PCT EXCELLENT" (sum them), US total, weekly. Save as a CSV with
   columns `week_ending, state, value`.

```bash
python3 cli.py --yield-file data/raw/wasde_history.csv \
               --condition-file data/raw/corn_condition.csv
```

Anything under `data/raw/` is git-ignored, so real downloads won't be committed.

## Glossary

| Term | Plain meaning |
|---|---|
| **WASDE** | USDA's monthly supply/demand report; the market-moving scorecard |
| **Ending stocks** | Corn left over at the end of the marketing year (the buffer) |
| **Yield** | Bushels of corn harvested per acre — the biggest driver of supply |
| **Marketing year** | Corn's "fiscal year": Sept 1 → Aug 31 |
| **New crop** | The crop just planted, harvested in the fall |
| **Crop condition** | USDA's weekly "% good/excellent" rating of how the crop looks |
| **Revision** | The month-to-month change in a USDA estimate — what we predict |
| **Leave-one-year-out** | Test each year with a model trained only on the others |

## Project layout

```
wasde-predictor/
  cli.py                     # run the whole loop and print the scoreboard
  wasde_predictor/
    wasde.py                 # load yield history; compute Aug-Nov revisions + labels
    condition.py             # load crop condition; point-in-time (leak-safe) accessor
    dataset.py               # join target + clue into observations
    models.py                # majority baseline + one-feature condition model
    evaluate.py              # accuracy + leave-one-year-out cross-validation
  tools/make_sample_data.py  # regenerate the synthetic sample (deterministic)
  data/sample/               # synthetic CSVs (committed, clearly fake)
  data/raw/                  # your real USDA downloads (git-ignored)
  tests/                     # unittest suite, incl. the no-leakage guard
```

## Roadmap (each a small next step)

1. ✅ **Walking skeleton** — this: end-to-end loop, one clue, honest baseline.
2. Real crop-condition features (levels, deviation from normal, top states).
3. Add weather (Corn Belt drought) as a second clue.
4. Upgrade the model (logistic regression → gradient-boosted tree) once features earn it.
5. Predict *magnitude*, not just direction.
6. Move from yield to full **ending stocks**; add demand-side clues (exports, ethanol).
7. Extend to soybeans and wheat.

> Note: the sample data in `data/sample/` is **synthetic and for testing only** —
> it is not real USDA data and must not be read as such.
