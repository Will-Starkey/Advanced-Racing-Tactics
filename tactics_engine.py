"""
Tactics Engine
Consumes InstrumentData, produces TacticalState.

Key computations:
  - LIFT / HEADER        : degrees shifted since last tack/jibe
  - SHIFT STATE          : none | transient | persistent (60 s threshold)
  - VMG                  : velocity made good toward mark on both tacks
  - Laylines             : port & starboard bearings + on/overstanding detection
  - TACK RECOMMENDATION  : hold | tack | sail_to_mark with reason
  - TTM                  : time-to-maneuver (distance to active layline / SOG)
  - Performance          : BSP and VMG vs polar targets
"""

import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from instruments.base import InstrumentData
from polar.models import PolarData


# ── Tuning constants ──────────────────────────────────────────────
SHIFT_THRESHOLD_DEG   = 5.0    # smaller shifts treated as noise
PERSISTENCE_WINDOW_S  = float(os.getenv("SIM_PERSISTENCE_WINDOW", "60.0"))
LAYLINE_TOLERANCE_DEG = 3.0    # degrees either side of exact layline
TACKING_PENALTY_S     = 18.0   # seconds of boatspeed lost per tack


@dataclass
class TacticalState:
    # ── Wind shift (relative to TWD at last tack/jibe) ────────────
    shift_type:    str   = "neutral"   # "lift" | "header" | "neutral"
    shift_degrees: float = 0.0

    # ── Persistent shift state ────────────────────────────────────
    shift_state:       str   = "none"   # "none" | "transient" | "persistent"
    shift_age_seconds: float = 0.0      # how long current shift has held

    # ── Wind trend over last 5 min ────────────────────────────────
    wind_trend: str = "insufficient data"   # "veering" | "backing" | "steady"

    # ── Angles ───────────────────────────────────────────────────
    twa:          float = 0.0
    on_starboard: bool  = True

    # ── Speed / VMG ───────────────────────────────────────────────
    vmg:                 float = 0.0
    target_vmg:          float = 0.0
    vmg_performance_pct: float = 0.0
    bsp_performance_pct: float = 0.0

    # ── VMG comparison (both tacks, toward mark) ──────────────────
    vmg_current_tack: float = 0.0   # polar VMG toward mark on current tack
    vmg_other_tack:   float = 0.0   # polar VMG toward mark after tacking
    vmg_gain_from_tack: float = 0.0  # net gain (other - current), after penalty

    # ── Laylines (bearings FROM mark) ─────────────────────────────
    port_layline: float = 0.0
    stbd_layline: float = 0.0
    optimal_twa:  float = 42.0

    # ── Layline status ────────────────────────────────────────────
    on_layline:   bool = False
    overstanding: bool = False

    # ── Mark geometry ─────────────────────────────────────────────
    bearing_to_mark:  Optional[float] = None   # degrees true
    dist_to_mark_nm:  Optional[float] = None

    # ── Tacking recommendation ────────────────────────────────────
    tack_recommendation:        str = "hold"   # "hold" | "tack" | "sail_to_mark"
    tack_recommendation_reason: str = ""

    # ── Time to maneuver (minutes) ────────────────────────────────
    ttm_minutes: float = 0.0

    # ── Heel ──────────────────────────────────────────────────────
    heel: float = 0.0

    # ── Source of performance data ────────────────────────────────
    performance_source: str = "none"   # "polar" | "h5000" | "none"


class TacticsEngine:
    def __init__(self):
        self.wind_history: deque         = deque(maxlen=300)   # 5 min @ 1 Hz
        self.baseline_twd: Optional[float] = None
        self.tack_baseline_twd: Optional[float] = None
        self.last_on_starboard: Optional[bool] = None

        # Persistent shift tracking
        self._shift_start_time:  Optional[float] = None   # unix timestamp
        self._last_shift_sign:   int = 0                  # +1 lift / -1 header / 0 none

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
        self.wind_history.append(inst.twd)
        self._update_baseline()

        twa          = self._signed_twa(inst.twd, inst.heading)
        abs_twa      = abs(twa)
        on_starboard = twa > 0

        self._detect_tack(on_starboard, inst.twd)

        shift_deg   = self._shift_degrees(inst.twd, on_starboard)
        shift_type  = self._shift_type(shift_deg)
        shift_state, shift_age = self._update_shift_state(shift_deg)

        state = TacticalState(
            shift_type         = shift_type,
            shift_degrees      = shift_deg,
            shift_state        = shift_state,
            shift_age_seconds  = shift_age,
            wind_trend         = self._wind_trend(),
            twa                = abs_twa,
            on_starboard       = on_starboard,
            heel               = inst.heel,
        )

        # ── Performance from B&G H5000 ────────────────────────────
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

        # ── Fallback ──────────────────────────────────────────────
        else:
            state.performance_source = "none"
            state.vmg                = self._vmg(inst.bsp, abs_twa)
            state.optimal_twa        = 42.0
            state.port_layline       = (inst.twd + 42) % 360
            state.stbd_layline       = (inst.twd - 42 + 360) % 360

        # ── Mark geometry & layline status ────────────────────────
        if self.mark_lat is not None and inst.lat is not None:
            bearing  = self._bearing(inst.lat, inst.lon, self.mark_lat, self.mark_lon)
            dist_nm  = self._haversine_nm(inst.lat, inst.lon, self.mark_lat, self.mark_lon)
            state.bearing_to_mark = round(bearing, 1)
            state.dist_to_mark_nm = round(dist_nm, 3)

            on_ll, over = self._check_layline(
                bearing, inst.twd, state.optimal_twa, on_starboard,
            )
            state.on_layline   = on_ll
            state.overstanding = over

            # ── VMG comparison on both tacks ──────────────────────
            vmg_cur, vmg_oth = self._compare_tack_vmg(
                inst.twd, inst.tws, state.optimal_twa, bearing, on_starboard,
            )
            state.vmg_current_tack = vmg_cur
            state.vmg_other_tack   = vmg_oth

            # Subtract tacking penalty from other-tack gain
            # penalty expressed as equivalent VMG loss over next 5 minutes
            dist_remaining = max(dist_nm, 0.1)
            penalty_vmg = (TACKING_PENALTY_S / 3600) * (vmg_cur / dist_remaining) if dist_remaining else 0
            state.vmg_gain_from_tack = round(vmg_oth - vmg_cur - penalty_vmg, 2)

        # ── Tacking recommendation ────────────────────────────────
        rec, reason = self._tacking_recommendation(state)
        state.tack_recommendation        = rec
        state.tack_recommendation_reason = reason

        # ── TTM ───────────────────────────────────────────────────
        if self.mark_lat is not None and inst.lat is not None and inst.sog > 0:
            state.ttm_minutes = self._calc_ttm(
                inst.lat, inst.lon, inst.sog, inst.heading,
                inst.twd, state.optimal_twa, on_starboard,
            )

        return state

    # ── Shift state ───────────────────────────────────────────────

    def _update_shift_state(self, shift_degrees: float):
        """
        Track how long a shift has been sustained.
        Returns (shift_state, age_seconds).
        """
        now = time.time()

        if abs(shift_degrees) < SHIFT_THRESHOLD_DEG:
            self._shift_start_time = None
            self._last_shift_sign  = 0
            return "none", 0.0

        current_sign = 1 if shift_degrees > 0 else -1

        if current_sign != self._last_shift_sign:
            # Shift changed direction — restart timer
            self._shift_start_time = now
            self._last_shift_sign  = current_sign
            return "transient", 0.0

        if self._shift_start_time is None:
            self._shift_start_time = now
            return "transient", 0.0

        age = now - self._shift_start_time
        if age >= PERSISTENCE_WINDOW_S:
            return "persistent", round(age, 1)
        return "transient", round(age, 1)

    # ── Layline check ─────────────────────────────────────────────

    def _check_layline(
        self,
        bearing_to_mark: float,
        twd: float,
        opt_twa: float,
        on_starboard: bool,
    ):
        """
        Returns (on_layline, overstanding).
        on_layline  — boat is within LAYLINE_TOLERANCE of the layline
        overstanding — boat has sailed past the layline (extra distance)
        """
        stbd_heading = (twd - opt_twa + 360) % 360
        port_heading = (twd + opt_twa) % 360

        def ang_diff(a, b):
            d = (a - b + 360) % 360
            return d if d <= 180 else 360 - d

        on_stbd = ang_diff(bearing_to_mark, stbd_heading) <= LAYLINE_TOLERANCE_DEG
        on_port = ang_diff(bearing_to_mark, port_heading) <= LAYLINE_TOLERANCE_DEG
        on_ll   = on_stbd or on_port

        # Overstanding: bearing_to_mark has left the upwind cone [stbd_heading → twd → port_heading].
        # The cone spans opt_twa*2 degrees clockwise from stbd_heading to port_heading.
        # If bearing is outside (beyond) the cone on either side, the boat is overstanding.
        cone_size = (port_heading - stbd_heading + 360) % 360   # always ~2*opt_twa
        d         = (bearing_to_mark - stbd_heading + 360) % 360
        over      = d > cone_size + LAYLINE_TOLERANCE_DEG

        return on_ll, over

    # ── VMG comparison ────────────────────────────────────────────

    def _compare_tack_vmg(
        self,
        twd: float,
        tws: float,
        opt_twa: float,
        bearing_to_mark: float,
        on_starboard: bool,
    ):
        """
        Compute polar VMG toward the mark on both tacks.
        Returns (vmg_current, vmg_other).
        """
        # Target BSP at optimal TWA
        if self.polar is not None:
            bsp = self.polar.target_bsp(opt_twa, tws)
        else:
            bsp = 6.0   # reasonable fallback

        def tack_vmg(stbd: bool) -> float:
            hdg = (twd - opt_twa + 360) % 360 if stbd else (twd + opt_twa) % 360
            # Angle between this heading and the bearing to mark
            diff = (bearing_to_mark - hdg + 360) % 360
            if diff > 180:
                diff = 360 - diff
            return round(bsp * math.cos(math.radians(diff)), 2)

        vmg_cur = tack_vmg(on_starboard)
        vmg_oth = tack_vmg(not on_starboard)
        return vmg_cur, vmg_oth

    # ── Tacking recommendation ────────────────────────────────────

    def _tacking_recommendation(self, state: TacticalState):
        """
        Returns (recommendation, reason).
        recommendation: "hold" | "tack" | "sail_to_mark"
        """
        # Overstanding always takes priority
        if state.overstanding:
            return "tack", "overstanding — tack immediately to avoid sailing extra distance"

        shift_state = state.shift_state
        shift_deg   = state.shift_degrees

        # Pure VMG geometry: if the other tack is significantly faster to the mark,
        # tack regardless of wind shift.  This catches "past the other layline" and
        # cases where one tack has a much better angle to the mark.
        # Threshold is intentionally high (0.75 kt net) to suppress oscillating-wind chatter.
        VMG_TACK_THRESHOLD = 0.75
        if (
            state.vmg_gain_from_tack > VMG_TACK_THRESHOLD
            and not state.on_layline
            and (state.dist_to_mark_nm or 1.0) > 0.12  # don't tack within ~730 ft of mark
        ):
            return (
                "tack",
                f"other tack is {state.vmg_gain_from_tack:+.2f} kts VMG faster toward mark "
                f"— geometry demands a tack (no wind shift needed)",
            )

        # Transient or no shift — hold
        if shift_state in ("none", "transient"):
            age_s = state.shift_age_seconds
            if shift_state == "transient" and abs(shift_deg) >= SHIFT_THRESHOLD_DEG:
                remaining = PERSISTENCE_WINDOW_S - age_s
                return (
                    "hold",
                    f"{'header' if shift_deg < 0 else 'lift'} of {abs(shift_deg):.1f}° detected "
                    f"— waiting {remaining:.0f}s to confirm persistence",
                )
            return "hold", "no significant shift"

        # Persistent shift
        if shift_deg > 0:
            # Lifted — stay on current tack
            return (
                "hold",
                f"persistent lift of {abs(shift_deg):.1f}° "
                f"({state.shift_age_seconds:.0f}s) — maximize gain on current tack",
            )

        # Persistent header
        if state.on_layline:
            return (
                "sail_to_mark",
                f"persistent header of {abs(shift_deg):.1f}° but on layline — "
                f"hold tack and sail directly to mark",
            )

        gain = state.vmg_gain_from_tack
        gain_str = f"{gain:+.2f} kts VMG" if gain != 0 else ""
        return (
            "tack",
            f"persistent header of {abs(shift_deg):.1f}° "
            f"({state.shift_age_seconds:.0f}s) — tack recommended"
            + (f", net gain {gain_str} after penalty" if gain_str else ""),
        )

    # ── Internal helpers ──────────────────────────────────────────

    def _signed_twa(self, twd: float, heading: float) -> float:
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
            self.tack_baseline_twd = twd
            # Reset persistent shift timer on tack
            self._shift_start_time = None
            self._last_shift_sign  = 0
        self.last_on_starboard = on_starboard

    def _shift_degrees(self, twd: float, on_starboard: bool) -> float:
        if self.tack_baseline_twd is None:
            return 0.0
        delta = twd - self.tack_baseline_twd
        while delta >  180: delta -= 360
        while delta < -180: delta += 360
        return round(delta if on_starboard else -delta, 1)

    def _shift_type(self, shift_deg: float) -> str:
        if shift_deg > 2:
            return "lift"
        elif shift_deg < -2:
            return "header"
        return "neutral"

    def _wind_trend(self) -> str:
        data = list(self.wind_history)
        if len(data) < 60:
            return "insufficient data"
        mid   = len(data) // 2
        early = sum(data[:mid]) / mid
        late  = sum(data[mid:]) / (len(data) - mid)
        diff  = late - early
        if diff > 3:
            return "veering"
        elif diff < -3:
            return "backing"
        return "steady"

    def _vmg(self, bsp: float, twa: float) -> float:
        return round(bsp * math.cos(math.radians(twa)), 2)

    def _bearing(self, lat1, lon1, lat2, lon2) -> float:
        """True bearing from point 1 to point 2 (degrees)."""
        φ1 = math.radians(lat1)
        φ2 = math.radians(lat2)
        Δλ = math.radians(lon2 - lon1)
        x  = math.sin(Δλ) * math.cos(φ2)
        y  = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    def _calc_ttm(
        self,
        boat_lat: float, boat_lon: float,
        sog: float, heading: float,
        twd: float, opt_twa: float,
        on_starboard: bool,
    ) -> float:
        active_brng = (
            (twd - opt_twa + 360) % 360
            if on_starboard
            else (twd + opt_twa) % 360
        )

        p1_lat, p1_lon = self.mark_lat, self.mark_lon
        p2 = self._dest_point(self.mark_lat, self.mark_lon, active_brng, 37.04)

        x1, y1 = p1_lon, p1_lat
        x2, y2 = p2[1], p2[0]
        px, py  = boat_lon, boat_lat

        dx, dy = x2 - x1, y2 - y1
        denom  = dx * dx + dy * dy
        if denom == 0:
            return 0.0

        t  = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
        cx = x1 + t * dx
        cy = y1 + t * dy

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
        return km * 0.539957
