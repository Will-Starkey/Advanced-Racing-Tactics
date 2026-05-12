"""
Base adapter interface and InstrumentData dataclass.
All brand-specific adapters extend BaseAdapter and return InstrumentData.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class InstrumentData:
    # ── Standard fields (all brands) ─────────────────────────────
    twd:     float = 0.0   # True Wind Direction (°)
    tws:     float = 0.0   # True Wind Speed (kts)
    awa:     float = 0.0   # Apparent Wind Angle (°, signed ±180)
    aws:     float = 0.0   # Apparent Wind Speed (kts)
    heading: float = 0.0   # Magnetic heading (°)
    bsp:     float = 0.0   # Boat Speed through water (kts)
    cog:     float = 0.0   # Course Over Ground (°)
    sog:     float = 0.0   # Speed Over Ground (kts)
    heel:    float = 0.0   # Heel / roll angle (°, + = stbd)
    leeway:  float = 0.0   # Leeway angle (°)

    # GPS position (may be None until first fix)
    lat: Optional[float] = None
    lon: Optional[float] = None

    # ── B&G H5000 performance fields (None if not available) ─────
    polar_speed:        Optional[float] = None   # target BSP from H5000
    polar_speed_ratio:  Optional[float] = None   # BSP / polar target (%)
    beat_angle:         Optional[float] = None   # H5000 optimal upwind TWA
    gybe_angle:         Optional[float] = None   # H5000 optimal downwind TWA
    target_twa:         Optional[float] = None   # H5000 current target TWA
    vmg_performance:    Optional[float] = None   # VMG vs target VMG (kts)


class BaseAdapter(ABC):
    """
    Subclasses implement extract() to map a raw Signal K state dict
    (path → value) into a clean InstrumentData instance.
    """

    @abstractmethod
    def extract(self, signalk_state: dict) -> InstrumentData:
        pass

    # ── Unit conversion helpers ───────────────────────────────────

    @staticmethod
    def rad_to_deg(val: float) -> float:
        """Radians → degrees, normalised to 0–360."""
        return (math.degrees(val) + 360) % 360

    @staticmethod
    def rad_to_signed_deg(val: float) -> float:
        """Radians → signed degrees (±180)."""
        deg = math.degrees(val)
        while deg >  180: deg -= 360
        while deg < -180: deg += 360
        return deg

    @staticmethod
    def ms_to_kts(val: float) -> float:
        """Metres per second → knots."""
        return round(val * 1.94384, 3)
