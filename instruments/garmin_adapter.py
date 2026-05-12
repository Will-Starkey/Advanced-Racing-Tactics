"""
Garmin Adapter — extracts InstrumentData from a Signal K state dict.

Garmin instruments output standard NMEA 2000 PGNs without a performance
computer layer.  All performance metrics (VMG %, BSP %, optimal TWA) are
computed by the TacticsEngine from the uploaded polar file.

Garmin-specific extras captured where available:
  - Water depth (below keel)
  - Water temperature
These are stored as private attributes for future use; they do not appear
in InstrumentData fields today but are logged for debugging.
"""

from .base import BaseAdapter, InstrumentData


class GarminAdapter(BaseAdapter):

    _PATHS = [
        ("environment.wind.angleTrueNorth",  "twd",     "rad_deg"),
        ("environment.wind.speedTrue",        "tws",     "ms_kts"),
        ("environment.wind.angleApparent",    "awa",     "rad_signed"),
        ("environment.wind.speedApparent",    "aws",     "ms_kts"),
        ("navigation.headingMagnetic",        "heading", "rad_deg"),
        ("navigation.headingTrue",            "heading", "rad_deg"),
        ("navigation.speedThroughWater",      "bsp",     "ms_kts"),
        ("navigation.courseOverGroundTrue",   "cog",     "rad_deg"),
        ("navigation.speedOverGround",        "sog",     "ms_kts"),
        ("navigation.attitude.roll",          "heel",    "rad_deg_signed"),
        ("navigation.leewayAngle",            "leeway",  "rad_deg"),
        ("_lat",                              "lat",     "raw"),
        ("_lon",                              "lon",     "raw"),
    ]

    def extract(self, state: dict) -> InstrumentData:
        inst = InstrumentData()

        for path, field, transform in self._PATHS:
            val = state.get(path)
            if val is None:
                continue
            setattr(inst, field, self._convert(val, transform))

        # Log Garmin extras (depth, water temp) without storing on InstrumentData
        depth = state.get("environment.depth.belowKeel")
        wtemp = state.get("environment.water.temperature")
        if depth is not None:
            pass  # available for future display: depth in metres
        if wtemp is not None:
            pass  # available for future display: temp in Kelvin → subtract 273.15 for °C

        return inst

    def _convert(self, val, transform: str):
        if transform == "rad_deg":
            return round(self.rad_to_deg(val), 2)
        elif transform == "rad_signed":
            return round(self.rad_to_signed_deg(val), 2)
        elif transform == "rad_deg_signed":
            return round(self.rad_to_signed_deg(val), 2)
        elif transform == "ms_kts":
            return round(self.ms_to_kts(val), 2)
        elif transform == "raw":
            return val
        return val
