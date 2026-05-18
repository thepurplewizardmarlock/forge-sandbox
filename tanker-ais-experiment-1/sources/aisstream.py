"""aisstream.io WebSocket source.

Yields parsed AIS messages from wss://stream.aisstream.io/v0/stream.
Placeholder source; will be replaced with Kpler once the pipeline is validated.
"""
import asyncio
import json
import os
from typing import AsyncIterator

import websockets

STREAM_URL = "wss://stream.aisstream.io/v0/stream"
GLOBAL_BBOX = [[[-90.0, -180.0], [90.0, 180.0]]]


async def stream(
    api_key: str | None = None,
    bounding_boxes: list | None = None,
    message_types: list[str] | None = None,
) -> AsyncIterator[dict]:
    key = api_key or os.environ.get("AISSTREAM_API_KEY")
    if not key:
        raise RuntimeError("AISSTREAM_API_KEY env var not set")

    subscribe = {
        "APIKey": key,
        "BoundingBoxes": bounding_boxes or GLOBAL_BBOX,
    }
    if message_types:
        subscribe["FilterMessageTypes"] = message_types

    async with websockets.connect(STREAM_URL) as ws:
        await ws.send(json.dumps(subscribe))
        async for raw in ws:
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


if __name__ == "__main__":
    async def _demo():
        async for msg in stream(message_types=["PositionReport", "ShipStaticData"]):
            print(msg.get("MessageType"), msg.get("MetaData", {}).get("MMSI"))

    asyncio.run(_demo())
