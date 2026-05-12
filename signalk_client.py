"""
Signal K WebSocket client.
Connects to the Signal K server (running on the Pi or locally),
subscribes to all vessel data, and calls on_update() with a flat
dict of Signal K path → raw value every time new data arrives.
Automatically reconnects on disconnect.
"""

import asyncio
import json
import math
import os

import websockets

SIGNALK_HOST = os.getenv("SIGNALK_HOST", "localhost")
SIGNALK_PORT = int(os.getenv("SIGNALK_PORT", "3000"))
SIGNALK_WS   = f"ws://{SIGNALK_HOST}:{SIGNALK_PORT}/signalk/v1/stream?subscribe=all"

# All Signal K paths we care about (vessel-relative, strip "vessels.self.")
WATCHED_PATHS = {
    "environment.wind.angleTrueNorth",
    "environment.wind.speedTrue",
    "environment.wind.angleApparent",
    "environment.wind.speedApparent",
    "navigation.headingMagnetic",
    "navigation.headingTrue",
    "navigation.speedThroughWater",
    "navigation.courseOverGroundTrue",
    "navigation.speedOverGround",
    "navigation.position",
    "navigation.attitude.roll",
    "navigation.leewayAngle",
    # B&G H5000 performance paths
    "performance.polarSpeed",
    "performance.polarSpeedRatio",
    "performance.beatAngle",
    "performance.gybeAngle",
    "performance.targetAngle",
    "performance.velocityMadeGood",
    "performance.velocityMadeGoodToWaypoint",
}

RECONNECT_DELAY = 5   # seconds between reconnect attempts


class SignalKClient:
    def __init__(self, on_update):
        self.on_update  = on_update
        self._state     = {}   # accumulated path → value dict
        self._connected = False

    async def connect_with_retry(self):
        """Keep trying to connect forever — handles Pi reboots, WiFi drops."""
        while True:
            try:
                print(f"[signalk] Connecting to {SIGNALK_WS}")
                async with websockets.connect(
                    SIGNALK_WS,
                    ping_interval=20,
                    ping_timeout=10,
                    open_timeout=10,
                ) as ws:
                    self._connected = True
                    print("[signalk] Connected")
                    await self._listen(ws)
            except (OSError, websockets.exceptions.WebSocketException) as exc:
                self._connected = False
                print(f"[signalk] Disconnected ({exc}) — retrying in {RECONNECT_DELAY}s")
                await asyncio.sleep(RECONNECT_DELAY)
            except asyncio.CancelledError:
                print("[signalk] Client cancelled")
                return

    async def _listen(self, ws):
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if "updates" not in msg:
                continue

            updated = False
            for update in msg["updates"]:
                for entry in update.get("values", []):
                    path  = entry.get("path", "")
                    value = entry.get("value")

                    # Strip vessel prefix if present
                    for prefix in ("vessels.self.", "vessels.urn:mrn:imo:mmsi:"):
                        if path.startswith(prefix):
                            # Remove prefix up to the second dot segment
                            path = path.split(".", 2)[-1] if path.count(".") >= 2 else path

                    if path not in WATCHED_PATHS:
                        continue

                    # Special case: position comes as {latitude, longitude}
                    if path == "navigation.position" and isinstance(value, dict):
                        self._state["_lat"] = value.get("latitude")
                        self._state["_lon"] = value.get("longitude")
                        updated = True
                        continue

                    self._state[path] = value
                    updated = True

            if updated and self._has_minimum_data():
                self.on_update(dict(self._state))

    def _has_minimum_data(self) -> bool:
        """Require at least wind + heading before firing updates."""
        required = {
            "environment.wind.angleTrueNorth",
            "environment.wind.speedTrue",
            "navigation.headingMagnetic",
        }
        return required.issubset(self._state.keys())

    @property
    def connected(self) -> bool:
        return self._connected
