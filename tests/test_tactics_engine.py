"""
Tests for TacticsEngine — shift detection, layline, VMG comparison,
and tacking recommendations.
"""
import time
import pytest
from unittest.mock import patch

from instruments.base import InstrumentData
from tactics_engine import TacticsEngine, TacticalState, PERSISTENCE_WINDOW_S, SHIFT_THRESHOLD_DEG
from polar.parser import PolarParser


# ── Helpers ───────────────────────────────────────────────────────

SIMPLE_POLAR_CSV = """\
TWA,6,8,10,12,16
42,5.40,6.51,7.04,7.30,7.52
52,5.70,6.80,7.28,7.51,7.73
60,6.11,7.17,7.58,7.78,8.01
90,7.07,7.92,8.52,9.02,9.71
120,6.98,8.06,8.71,9.16,10.00
150,5.13,6.58,7.62,8.35,9.73
"""


def make_engine(with_polar=False, with_mark=False):
    e = TacticsEngine()
    if with_polar:
        polar = PolarParser.parse("p.csv", SIMPLE_POLAR_CSV, "Test")
        e.set_polar(polar)
    if with_mark:
        e.set_mark(41.510, -71.310)
    return e


def inst(twd=260.0, tws=12.0, heading=218.0, bsp=7.0, sog=7.0,
         lat=41.500, lon=-71.320, heel=5.0):
    """Build a minimal InstrumentData for testing."""
    return InstrumentData(
        twd=twd, tws=tws, heading=heading, bsp=bsp,
        sog=sog, awa=twd - heading, heel=heel,
        lat=lat, lon=lon,
    )


# ── Tack detection & shift ────────────────────────────────────────

class TestShiftDetection:
    def test_starboard_tack_detected(self):
        # heading 218, twd 260 → signed twa = -42 → starboard
        e = make_engine()
        state = e.update(inst(twd=260, heading=218))
        assert state.on_starboard is True

    def test_port_tack_detected(self):
        # heading 302, twd 260 → signed twa = +42 → port
        e = make_engine()
        state = e.update(inst(twd=260, heading=302))
        assert state.on_starboard is False

    def test_lift_on_starboard(self):
        # Wind veers (increases) on starboard → lift
        e = make_engine()
        e.update(inst(twd=260, heading=218))   # establish baseline
        state = e.update(inst(twd=270, heading=218))
        assert state.shift_type == "lift"
        assert state.shift_degrees > 0

    def test_header_on_starboard(self):
        # Wind backs (decreases) on starboard → header
        e = make_engine()
        e.update(inst(twd=260, heading=218))
        state = e.update(inst(twd=250, heading=218))
        assert state.shift_type == "header"
        assert state.shift_degrees < 0

    def test_lift_on_port(self):
        # Wind backs on port → lift
        e = make_engine()
        e.update(inst(twd=260, heading=302))
        state = e.update(inst(twd=250, heading=302))
        assert state.shift_type == "lift"

    def test_header_on_port(self):
        # Wind veers on port → header
        e = make_engine()
        e.update(inst(twd=260, heading=302))
        state = e.update(inst(twd=270, heading=302))
        assert state.shift_type == "header"

    def test_baseline_resets_on_tack(self):
        e = make_engine()
        e.update(inst(twd=260, heading=218))   # starboard
        e.update(inst(twd=302, heading=218))   # tack to port
        state = e.update(inst(twd=302, heading=302))
        # immediately after tack, shift should be near zero
        assert abs(state.shift_degrees) < 3


# ── Persistent shift state ────────────────────────────────────────

class TestPersistentShift:
    def test_small_shift_is_none(self):
        e = make_engine()
        e.update(inst(twd=260, heading=218))
        state = e.update(inst(twd=262, heading=218))   # 2° < threshold
        assert state.shift_state == "none"

    def test_new_shift_is_transient(self):
        e = make_engine()
        e.update(inst(twd=260, heading=218))
        state = e.update(inst(twd=250, heading=218))   # 10° header
        assert state.shift_state == "transient"

    def test_shift_becomes_persistent_after_window(self):
        e = make_engine()
        e.update(inst(twd=260, heading=218))
        # Simulate time passing beyond the persistence window
        e._shift_start_time = time.time() - (PERSISTENCE_WINDOW_S + 1)
        e._last_shift_sign = -1  # header direction
        state = e.update(inst(twd=250, heading=218))
        assert state.shift_state == "persistent"

    def test_shift_direction_change_resets_timer(self):
        e = make_engine()
        e.update(inst(twd=260, heading=218))
        # Start a header
        e._shift_start_time = time.time() - 50
        e._last_shift_sign = -1
        # Now wind reverses to a lift
        state = e.update(inst(twd=270, heading=218))
        # Should reset to transient since direction changed
        assert state.shift_state == "transient"
        assert state.shift_age_seconds < 5   # timer restarted


# ── Layline detection ─────────────────────────────────────────────

class TestLaylineDetection:
    def setup_method(self):
        # Mark directly upwind at ~0.5nm
        # Boat at 41.500, mark at 41.510 (roughly north)
        # TWD=000 (wind from north), heading=318 (stbd tack at 42° TWA)
        self.e = make_engine(with_polar=True, with_mark=True)
        self.e.set_mark(41.510, -71.320)

    def test_not_on_layline_normally(self):
        # Boat well below layline
        state = self.e.update(inst(twd=260, heading=218, lat=41.490, lon=-71.350))
        assert state.on_layline is False
        assert state.overstanding is False

    def test_bearing_to_mark_computed(self):
        state = self.e.update(inst(twd=260, heading=218, lat=41.500, lon=-71.320))
        assert state.bearing_to_mark is not None
        assert 0 <= state.bearing_to_mark < 360

    def test_dist_to_mark_computed(self):
        state = self.e.update(inst(twd=260, heading=218, lat=41.500, lon=-71.320))
        assert state.dist_to_mark_nm is not None
        assert state.dist_to_mark_nm > 0


# ── VMG comparison ────────────────────────────────────────────────

class TestVMGComparison:
    def test_vmg_both_tacks_computed(self):
        e = make_engine(with_polar=True, with_mark=True)
        state = e.update(inst(twd=260, heading=218, lat=41.500, lon=-71.320))
        assert state.vmg_current_tack != 0.0
        assert state.vmg_other_tack != 0.0

    def test_correct_tack_has_higher_vmg_toward_mark(self):
        # Mark is to the north-east. Starboard tack heading NE should give better VMG.
        e = make_engine(with_polar=True)
        e.set_mark(41.520, -71.300)   # NE of boat
        # Starboard tack ~NE: twd=180 (S wind), heading=138 (stbd at 42° TWA)
        state = e.update(inst(twd=180, heading=138, lat=41.500, lon=-71.320))
        # Tacking (port tack, heading SW) would go away from mark → lower VMG
        assert state.vmg_current_tack > state.vmg_other_tack


# ── Tacking recommendations ───────────────────────────────────────

class TestTackingRecommendation:
    def _make_state(self, **kwargs):
        """Build a TacticalState with sensible defaults."""
        defaults = dict(
            shift_state="none", shift_degrees=0.0,
            on_layline=False, overstanding=False,
            on_starboard=True, vmg_current_tack=6.0,
            vmg_other_tack=6.0, vmg_gain_from_tack=0.0,
        )
        defaults.update(kwargs)
        return TacticalState(**defaults)

    def test_overstanding_always_tacks(self):
        e = make_engine()
        state = self._make_state(overstanding=True)
        rec, reason = e._tacking_recommendation(state)
        assert rec == "tack"
        assert "overstanding" in reason.lower()

    def test_transient_shift_holds(self):
        e = make_engine()
        state = self._make_state(shift_state="transient", shift_degrees=-8.0)
        rec, _ = e._tacking_recommendation(state)
        assert rec == "hold"

    def test_no_shift_holds(self):
        e = make_engine()
        state = self._make_state(shift_state="none")
        rec, _ = e._tacking_recommendation(state)
        assert rec == "hold"

    def test_persistent_lift_holds(self):
        e = make_engine()
        state = self._make_state(shift_state="persistent", shift_degrees=+8.0)
        rec, _ = e._tacking_recommendation(state)
        assert rec == "hold"

    def test_persistent_header_off_layline_tacks(self):
        e = make_engine()
        state = self._make_state(
            shift_state="persistent", shift_degrees=-8.0,
            on_layline=False,
        )
        rec, _ = e._tacking_recommendation(state)
        assert rec == "tack"

    def test_persistent_header_on_layline_holds(self):
        e = make_engine()
        state = self._make_state(
            shift_state="persistent", shift_degrees=-8.0,
            on_layline=True,
        )
        rec, reason = e._tacking_recommendation(state)
        assert rec == "sail_to_mark"
        assert "layline" in reason.lower()

    def test_overstanding_beats_layline_rule(self):
        # Even if on_layline, overstanding takes priority
        e = make_engine()
        state = self._make_state(
            overstanding=True, on_layline=True,
            shift_state="persistent", shift_degrees=-8.0,
        )
        rec, _ = e._tacking_recommendation(state)
        assert rec == "tack"


# ── TTM & geometry ────────────────────────────────────────────────

class TestGeometry:
    def test_ttm_zero_without_mark(self):
        e = make_engine()
        state = e.update(inst())
        assert state.ttm_minutes == 0.0

    def test_ttm_positive_with_mark(self):
        e = make_engine(with_mark=True)
        state = e.update(inst(sog=7.0, lat=41.490, lon=-71.320))
        assert state.ttm_minutes > 0

    def test_bearing_none_without_gps(self):
        e = make_engine(with_mark=True)
        state = e.update(InstrumentData(twd=260, tws=12, heading=218, bsp=7, sog=7))
        assert state.bearing_to_mark is None

    def test_wind_trend_needs_data(self):
        e = make_engine()
        state = e.update(inst())
        assert state.wind_trend == "insufficient data"
