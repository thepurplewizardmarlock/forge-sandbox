"""Tanker AIS snapshot job.

Connects to aisstream.io for a fixed window, captures every PositionReport
emitted by known tanker MMSIs (ITU ship type 80-89) during that window, and
writes the messages as JSONL to snapshots/. Designed to be run from cron a
handful of times per day.

A persistent MMSI->ship-type cache (cache/mmsi_to_type.json) is updated from
ShipStaticData messages on every run, so successive runs recognize known
tankers immediately on connect.
"""
import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from sources.aisstream import stream

TANKER_TYPE_RANGE = range(80, 90)
CACHE_PATH = Path("cache/mmsi_to_type.json")
SNAPSHOT_DIR = Path("snapshots")


def is_tanker(ship_type: int | None) -> bool:
    return ship_type is not None and ship_type in TANKER_TYPE_RANGE


def load_cache() -> dict[int, int]:
    if not CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(CACHE_PATH.read_text())
        return {int(k): int(v) for k, v in raw.items()}
    except (json.JSONDecodeError, ValueError):
        return {}


def save_cache(cache: dict[int, int]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    tmp.replace(CACHE_PATH)


async def capture(window_seconds: int) -> None:
    mmsi_to_type = load_cache()
    tanker_mmsi = {m for m, t in mmsi_to_type.items() if is_tanker(t)}
    cached_tankers_at_start = len(tanker_mmsi)

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = SNAPSHOT_DIR / f"tanker-snapshot-{stamp}.jsonl"

    reports_written = 0
    new_tankers = 0
    unique_tankers_seen: set[int] = set()

    try:
        with out_path.open("w") as f:
            async with asyncio.timeout(window_seconds):
                async for msg in stream(
                    message_types=["PositionReport", "ShipStaticData"],
                ):
                    meta = msg.get("MetaData", {})
                    mmsi = meta.get("MMSI")
                    kind = msg.get("MessageType")
                    if mmsi is None:
                        continue

                    if kind == "ShipStaticData":
                        ship_type = (
                            msg.get("Message", {})
                            .get("ShipStaticData", {})
                            .get("Type")
                        )
                        if ship_type is None:
                            continue
                        if mmsi_to_type.get(mmsi) != ship_type:
                            mmsi_to_type[mmsi] = ship_type
                        if is_tanker(ship_type) and mmsi not in tanker_mmsi:
                            tanker_mmsi.add(mmsi)
                            new_tankers += 1
                    elif kind == "PositionReport" and mmsi in tanker_mmsi:
                        f.write(json.dumps(msg) + "\n")
                        reports_written += 1
                        unique_tankers_seen.add(mmsi)
    except asyncio.TimeoutError:
        pass
    finally:
        save_cache(mmsi_to_type)

    print(
        f"snapshot file: {out_path}\n"
        f"  position reports written:   {reports_written}\n"
        f"  unique tankers in snapshot: {len(unique_tankers_seen)}\n"
        f"  tanker MMSIs cached:        {len(tanker_mmsi)} "
        f"(+{new_tankers} new this run, {cached_tankers_at_start} carried in)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=15,
        help=(
            "How long to listen on the WebSocket before writing the snapshot. "
            "Default 15 min covers >2x the 6-min ShipStaticData interval and "
            ">5x the 3-min anchored-vessel position interval."
        ),
    )
    args = parser.parse_args()
    asyncio.run(capture(args.window_minutes * 60))


if __name__ == "__main__":
    main()
