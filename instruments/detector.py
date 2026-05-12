"""
Instrument brand detection via Signal K REST API.

Detection strategy (in order):
  1. Check for B&G H5000 performance paths  →  BG_H5000
  2. Check NMEA 2000 manufacturer codes in sources  →  BG_STANDARD / GARMIN
  3. Fall back to UNKNOWN (standard N2K paths used)
"""

import os
from enum import Enum

import httpx

SIGNALK_HOST = os.getenv("SIGNALK_HOST", "localhost")
SIGNALK_PORT = int(os.getenv("SIGNALK_PORT", "3000"))
SIGNALK_BASE = f"http://{SIGNALK_HOST}:{SIGNALK_PORT}/signalk/v1/api"

# Signal K performance paths that only a B&G H5000 exposes
BG_H5000_SIGNATURES = {
    "performance.polarSpeed",
    "performance.polarSpeedRatio",
    "performance.beatAngle",
    "performance.gybeAngle",
    "performance.targetAngle",
}

# NMEA 2000 manufacturer codes
# Navico (B&G parent): 69, 275, 381
# Garmin:              229
MANUFACTURER_MAP = {
    69:  "bg",
    275: "bg",
    381: "bg",
    229: "garmin",
}


class InstrumentBrand(Enum):
    BG_H5000   = "bg_h5000"      # Full B&G performance computer
    BG_STANDARD = "bg_standard"  # B&G Zeus / Triton (no H5000)
    GARMIN     = "garmin"
    UNKNOWN    = "unknown"


async def detect_brand() -> InstrumentBrand:
    async with httpx.AsyncClient(timeout=5.0) as client:

        # ── Step 1: check for H5000 performance paths ─────────────
        try:
            resp = await client.get(f"{SIGNALK_BASE}/vessels/self/performance")
            if resp.status_code == 200:
                available = set(resp.json().keys())
                if available & BG_H5000_SIGNATURES:
                    print("[detector] B&G H5000 detected (performance paths found)")
                    return InstrumentBrand.BG_H5000
        except Exception:
            pass

        # ── Step 2: check NMEA 2000 source manufacturer codes ─────
        try:
            resp = await client.get(f"{SIGNALK_BASE}/sources")
            if resp.status_code == 200:
                for _bus, devices in resp.json().items():
                    if not isinstance(devices, dict):
                        continue
                    for _addr, device in devices.items():
                        if not isinstance(device, dict):
                            continue
                        n2k  = device.get("n2k", {})
                        code = n2k.get("manufacturerCode") or n2k.get("mfr")
                        if code in MANUFACTURER_MAP:
                            brand_str = MANUFACTURER_MAP[code]
                            if brand_str == "bg":
                                print("[detector] B&G Standard detected (N2K manufacturer code)")
                                return InstrumentBrand.BG_STANDARD
                            elif brand_str == "garmin":
                                print("[detector] Garmin detected (N2K manufacturer code)")
                                return InstrumentBrand.GARMIN
        except Exception:
            pass

    print("[detector] Brand unknown — using standard N2K paths")
    return InstrumentBrand.UNKNOWN
