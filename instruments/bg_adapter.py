"""
B&G Adapter — extracts InstrumentData from a Signal K state dict.

Handles both:
  - B&G Standard (Zeus / Triton):  standard N2K paths only
  - B&G H5000:                     standard paths + performance.* paths

Performance paths are populated when present and left as None when absent,
so the tactics engine can branch on performance_source automatically.
"""

from .base import BaseAdapter, InstrumentData


class BGAdapter(BaseAdapter):

    # (Signal K path, InstrumentData field, transform)
    _STANDARD_PATHS = [
        ("environment.wind.angleTrueNorth",  "twd",     "rad_deg"),
        ("environment.wind.speedTrue",        "tws",     "ms_kts"),
        ("environment.wind.angleApparent",    "awa",     "rad_signed"),
        ("environment.wind.speedApparent",    "aws",     "ms_kts"),
        ("navigation.headingMagnetic",        "heading", "rad_deg"),
        ("navigation.headingTrue",            "heading", "rad_deg"),   # prefer true if available
        ("navigation.speedThroughWater",      "bsp",     "ms_kts"),
        ("navigation.courseOverGroundTrue",   "cog",     "rad_deg"),
        ("navigation.speedOverGround",        "sog",     "ms_kts"),
        ("navigation.attitude.roll",          "heel",    "rad_deg_signed"),
        ("navigation.leewayAngle",            "leeway",  "rad_deg"),
        ("_lat",                              "lat",     "raw"),
        ("_lon",                              "lon",     "raw"),
    ]

    _PERFORMANCE_PATHS = [
        ("performance.polarSpeed",                  "polar_speed",       "ms_kts"),
        ("performance.polarSpeedRatio",             "polar_speed_ratio", "ratio_pct"),
        ("performance.beatAngle",                   "beat_angle",        "rad_deg"),
        ("performance.gybeAngle",                   "gybe_angle",        "rad_deg"),
        ("performance.targetAngle",                 "target_twa",        "rad_deg"),
        ("performance.velocityMadeGoodToWaypoint",  "vmg_performance",   "ms_kts"),
    ]

    def extract(self, state: dict) -> InstrumentData:
        inst = InstrumentData()

        for path, field, transform in self._STANDARD_PATHS:
            val = state.get(path)
            if val is None:
                continue
            setattr(inst, field, self._convert(val, transform))

        for path, field, transform in self._PERFORMANCE_PATHS:
            val = state.get(path)
            if val is None:
                continue
            setattr(inst, field, self._convert(val, transform))

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
        elif transform == "ratio_pct":
            return round(val * 100, 1)   # 0–1 fraction → percentage
        elif transform == "raw":
            return val
        return val
