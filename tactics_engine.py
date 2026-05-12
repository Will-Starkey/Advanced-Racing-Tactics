"""
Tactics Engine
Consumes InstrumentData, produces TacticalState.

Key computations:
  - LIFT / HEADER  : degrees shifted since last tack or jibe
  - VMG            : velocity made good toward/away from mark
  - Laylines       : port & starboard bearings from mark at optimal TWA
  - TTM            : time-to-maneuver (distance to active layline / SOG)
  - Performance    : BSP and VMG vs polar targets
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from instruments.base import InstrumentData
from polar.models import PolarData


@dataclass
class TacticalState:
    # Wind shift (relative to TWD at last tack/jibe)
    shift_type:    str   = "neutral"   # "lift" | "header" | "neutral"
    shift_degrees: float = 0.0

    # Wind trend over last 5 min
    wind_trend: str = "insufficient data"   # "veering" | "backing" | "steady"

    # Angles
    twa:        float = 0.0   # absolute TWA (degrees, 0-180)
    on_starboard: bool = True

    # Speed / VMG
    vmg:                 float = 0.0
    target_vmg:          float = 0.0
    vmg_performance_pct: float = 0.0
    bsp_performance_pct: float = 0.0

    # Laylines (bearings FROM mark)
    port_layline: float = 0.0
    stbd_layline: float = 0.0
    optimal_twa:  float = 42.0

    # Time to maneuver (minutes)
    ttm_minutes: float = 0.0

    # Heel
    heel: float = 0.0

    # Source of performance data
    performance_source: str = "none"   # "polar" | "h5000" | "none"


class TacticsEngine:
    def __init__(self):
        self.wind_history: deque  = deque(maxlen=300)   # 5 min @ 1 Hz
        self.baseline_twd: Optional[float] = None        # rolling average for trend
        self.tack_baseline_twd: Optional[float] = None  # TWD at last tack/jibe
        self.last_on_starboard: Optional[bool] = None

        self.polar: Optional[PolarData] = None

        # Mark position (set by /mark endpoint)
        self.mark_lat: Optional[float] = None
        self.mark_lon: Optional[float] = None

    # ── Public API ────────────────────────────────────────────────

    def set_polar(self, polar: PolarData):
        self.polar = polar

    def set_mark(self, lat: float, lon: float):
        self.mark_lat = lat
        self.mark_lon = lon

    def update(self, inst: InstrumentData) -> TacticalState:
        # Accumulate wind history for trend analysis
        self.wind_history.append(inst.twd)
        self._update_baseline()

        twa          = self._signed_twa(inst.twd, inst.heading)
        abs_twa      = abs(twa)
        on_starboard = twa < 0   # negative signed TWA → starboard tack

        # Detect tack / jibe and reset shift baseline
        self._detect_tack(on_starboard, inst.twd)

        state = TacticalState(
            shift_type    = self._shift_type(inst.twd, on_starboard),
            shift_degrees = self._shift_degrees(inst.twd, on_starboard),
            wind_trend    = self._wind_trend(),
            twa           = abs_twa,
            on_starboard  = on_starboard,
            heel          = inst.heel,
        )

        # ── Performance from B&G H5000 (if available) ────────────
        if inst.polar_speed is not None:
            state.performance_source  = "h5000"
            state.bsp_performance_pct = inst.polar_speed_ratio or 0.0
            state.optimal_twa         = inst.beat_angle or 42.0
            state.vmg                 = self._vmg(inst.bsp, abs_twa)
            state.port_layline        = (inst.twd + (inst.beat_angle or 42)) % 360
            state.stbd_layline        = (inst.twd - (inst.beat_angle or 42) + 360) % 360

        # ── Performance from uploaded polar ───────────────────────
        elif self.polar is not None:
            state.performance_source = "polar"
            upwind       = abs_twa < 100
            opt_twa      = self.polar.optimal_twa(inst.tws, upwind=upwind)
            target_bsp   = self.polar.target_bsp(abs_twa, inst.tws)
            target_vmg   = self._vmg(target_bsp, opt_twa)
            actual_vmg   = self._vmg(inst.bsp, abs_twa)

            state.optimal_twa         = opt_twa
            state.vmg                 = actual_vmg
            state.target_vmg          = target_vmg
            state.vmg_performance_pct = round(actual_vmg / target_vmg * 100, 1) if target_vmg else 0.0
            state.bsp_performance_pct = self.polar.performance_ratio(inst.bsp, abs_twa, inst.tws) * 100
            state.port_layline        = (inst.twd + opt_twa) % 360
            state.stbd_layline        = (inst.twd - opt_twa + 360) % 360

        # ── Fallback: no polar, no H5000 ─────────────────────────
        else:
            state.performance_source = "none"
            state.vmg                = self._vmg(inst.bsp, abs_twa)
            state.optimal_twa        = 42.0
            state.port_layline       = (inst.twd + 42) % 360
            state.stbd_layline       = (inst.twd - 42 + 360) % 360

        # ── TTM ───────────────────────────────────────────────────
        if self.mark_lat is not None and inst.lat is not None and inst.sog > 0:
            state.ttm_minutes = self._calc_ttm(
                inst.lat, inst.lon, inst.sog, inst.heading,
                inst.twd, state.optimal_twa, on_starboard,
            )

        return state

    # ── Internal helpers ──────────────────────────────────────────

    def _signed_twa(self, twd: float, heading: float) -> float:
        """TWA in ±180 convention.  Negative = starboard tack."""
        angle = (twd - heading + 360) % 360
        return angle - 360 if angle > 180 else angle

    def _update_baseline(self):
        if len(self.wind_history) >= 20:
            sample = list(self.wind_history)[-60:]
            self.baseline_twd = sum(sample) / len(sample)

    def _detect_tack(self, on_starboard: bool, twd: float):
        if self.last_on_starboard is None:
            self.tack_baseline_twd = twd
        elif on_starboard != self.last_on_starboard:
            # Tack or jibe — reset shift baseline
            self.tack_baseline_twd = twd
        self.last_on_starboard = on_starboard

    def _shift_degrees(self, twd: float, on_starboard: bool) -> float:
        """
        Degrees the wind has shifted since last tack/jibe.
        Positive = lift on current tack, negative = header.
        """
        if self.tack_baseline_twd is None:
            return 0.0
        delta = twd - self.tack_baseline_twd
        while delta >  180: delta -= 360
        while delta < -180: delta += 360
        return round(delta if on_starboard else -delta, 1)

    def _shift_type(self, twd: float, on_starboard: bool) -> str:
        deg = self._shift_degrees(twd, on_starboard)
        if deg > 2:
            return "lift"
        elif deg < -2:
            return "header"
        return "neutral"

    def _wind_trend(self) -> str:
        """Veering / backing trend over the last 5 minutes."""
        data = list(self.wind_history)
        if len(data) < 60:
            return "insufficient data"
        mid    = len(data) // 2
        early  = sum(data[:mid]) / mid
        late   = sum(data[mid:]) / (len(data) - mid)
        diff   = late - early
        if diff > 3:
            return "veering"
        elif diff < -3:
            return "backing"
        return "steady"

    def _vmg(self, bsp: float, twa: float) -> float:
        return round(bsp * math.cos(math.radians(twa)), 2)

    def _calc_ttm(
        self,
        boat_lat: float, boat_lon: float,
        sog: float, heading: float,
        twd: float, opt_twa: float,
        on_starboard: bool,
    ) -> float:
        """
        Time to maneuver in minutes.
        = perpendicular distance from boat to the active layline / SOG
        The active layline is the one the boat is currently sailing toward.
        """
        # Active layline: the one the boat will cross next
        active_brng = (
            (twd - opt_twa + 360) % 360   # starboard layline
            if on_starboard
            else (twd + opt_twa) % 360    # port layline
        )

        # Two points on the layline (from mark, 20 nm extension)
        p1_lat, p1_lon = self.mark_lat, self.mark_lon
        p2 = self._dest_point(self.mark_lat, self.mark_lon, active_brng, 37.04)  # 20 nm in km

        # Closest point on that line segment to the boat
        x1, y1 = p1_lon, p1_lat
        x2, y2 = p2[1], p2[0]
        px, py  = boat_lon, boat_lat

        dx, dy = x2 - x1, y2 - y1
        denom  = dx * dx + dy * dy
        if denom == 0:
            return 0.0

        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
        cx = x1 + t * dx
        cy = y1 + t * dy

        # Haversine distance boat → closest point on layline (nm)
        dist_nm = self._haversine_nm(boat_lat, boat_lon, cy, cx)

        return round((dist_nm / sog) * 60, 1) if sog > 0 else 0.0

    @staticmethod
    def _dest_point(lat, lon, bearing_deg, dist_km):
        R    = 6371.0
        d    = dist_km / R
        brng = math.radians(bearing_deg)
        φ1   = math.radians(lat)
        λ1   = math.radians(lon)
        φ2   = math.asin(math.sin(φ1) * math.cos(d) +
                         math.cos(φ1) * math.sin(d) * math.cos(brng))
        λ2   = λ1 + math.atan2(math.sin(brng) * math.sin(d) * math.cos(φ1),
                                math.cos(d) - math.sin(φ1) * math.sin(φ2))
        return math.degrees(φ2), math.degrees(λ2)

    @staticmethod
    def _haversine_nm(lat1, lon1, lat2, lon2) -> float:
        R    = 6371.0
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a    = (math.sin(dLat / 2) ** 2 +
                math.cos(math.radians(lat1)) *
                math.cos(math.radians(lat2)) *
                math.sin(dLon / 2) ** 2)
        km   = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return km * 0.539957   # km → nm
