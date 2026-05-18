"""Tanker AIS pipeline (experiment 1).

Reads live AIS from aisstream.io, filters for tanker vessels (ITU ship type
80-89), and prints position updates. The source module is intentionally
isolated so it can be swapped for a Kpler feed later.
"""
import asyncio

from sources.aisstream import stream

TANKER_TYPE_RANGE = range(80, 90)


def is_tanker(ship_type: int | None) -> bool:
    return ship_type is not None and ship_type in TANKER_TYPE_RANGE


async def run() -> None:
    mmsi_to_type: dict[int, int] = {}
    tanker_mmsi: set[int] = set()

    async for msg in stream(message_types=["PositionReport", "ShipStaticData"]):
        meta = msg.get("MetaData", {})
        mmsi = meta.get("MMSI")
        kind = msg.get("MessageType")

        if kind == "ShipStaticData":
            ship_type = msg.get("Message", {}).get("ShipStaticData", {}).get("Type")
            if mmsi is not None and ship_type is not None:
                mmsi_to_type[mmsi] = ship_type
                if is_tanker(ship_type):
                    tanker_mmsi.add(mmsi)
            continue

        if kind == "PositionReport" and mmsi in tanker_mmsi:
            report = msg.get("Message", {}).get("PositionReport", {})
            lat = report.get("Latitude")
            lon = report.get("Longitude")
            sog = report.get("Sog")
            print(f"TANKER {mmsi} lat={lat} lon={lon} sog={sog}")


if __name__ == "__main__":
    asyncio.run(run())
