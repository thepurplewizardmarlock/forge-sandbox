"""Per-commodity configuration -- the single source of truth for what clues and
report window each commodity uses.

Every commodity uses the same supply clues (crop condition + drought). They differ
in their DEMAND clues, which matter for the ending-stocks target:
  * corn      -> exports + ethanol grind
  * soybeans  -> exports + crush
  * wheat     -> exports (little domestic processing)

`report_months` is the set of monthly reports we treat as targets (the survey-yield
window). Corn and soybeans use Aug-Nov; wheat's window differs (set in a later phase).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemandClue:
    feature: str       # feature name the model sees, e.g. "export_pace_surprise"
    basename: str      # data-file basename (no extension), e.g. "exports_corn"
    key: str           # series key to match, e.g. "CORN" or "US"
    key_column: str    # CSV column that holds the key, e.g. "commodity" or "region"


@dataclass(frozen=True)
class Commodity:
    slug: str                        # "corn"
    label: str                       # "Corn" (matches the WASDE Commodity column)
    report_months: tuple[int, ...]
    demand_clues: tuple[DemandClue, ...]


CORN = Commodity("corn", "Corn", (8, 9, 10, 11), (
    DemandClue("export_pace_surprise", "exports_corn", "CORN", "commodity"),
    DemandClue("ethanol_pace_surprise", "ethanol_corn", "US", "region"),
))
SOYBEANS = Commodity("soybeans", "Soybeans", (8, 9, 10, 11), (
    DemandClue("export_pace_surprise", "exports_soybeans", "SOYBEANS", "commodity"),
    DemandClue("crush_pace_surprise", "crush_soybeans", "US", "region"),
))
# Wheat's yield firms up earlier than the row crops: winter-wheat harvest runs
# May-July and the Small Grains Summary (late Sept) finalizes production, so its
# survey window is ~May-September rather than Aug-Nov.
WHEAT = Commodity("wheat", "Wheat", (5, 6, 7, 8, 9), (
    DemandClue("export_pace_surprise", "exports_wheat", "WHEAT", "commodity"),
))

ALL = {c.slug: c for c in (CORN, SOYBEANS, WHEAT)}


def get(slug: str) -> Commodity:
    return ALL[slug]
