"""
Sailing Tactics Simulator
Feeds synthetic InstrumentData through the tactics engine and LLM without
needing a Raspberry Pi or instruments.

Launch with:
    SIM_PERSISTENCE_WINDOW=8 SIM_ONLY=true python main.py

Then open:
    http://localhost:8000/sim.html   ← start / stop scenarios
    http://localhost:8000/           ← watch live updates
"""

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from instruments.base import InstrumentData

# ── Shared geo constants ──────────────────────────────────────────
BASE_LAT  = 37.8270    # San Francisco Bay starting position
BASE_LON  = -122.4350
MARK_LAT  = 37.8370    # upwind mark (~0.6 nm due north)
MARK_LON  = -122.4350
BASE_TWS  = 12.0       # knots
BASE_BSP  = 6.5        # knots


# ── Data structures ───────────────────────────────────────────────

StepFn = Callable[[int, float], InstrumentData]   # (step_index, elapsed_s) → data


@dataclass
class SimStep:
    duration_seconds: float
    data_fn: StepFn
    label: str = ""


@dataclass
class SimScenario:
    name: str
    description: str
    steps: List[SimStep]


class SimState:
    """Mutable simulation status — read by /sim/* endpoints."""
    def __init__(self):
        self.running:            bool  = False
        self.scenario_name:      str   = ""
        self.current_step_label: str   = ""
        self.elapsed_seconds:    float = 0.0
        self._task: Optional[asyncio.Task] = None


# ── Geo helper ────────────────────────────────────────────────────

def _dest_coords(lat: float, lon: float, bearing_deg: float, dist_nm: float):
    """Return (lat, lon) of a point dist_nm from (lat, lon) on bearing_deg."""
    R    = 6371.0
    d    = (dist_nm * 1.852) / R
    brng = math.radians(bearing_deg)
    φ1   = math.radians(lat)
    λ1   = math.radians(lon)
    φ2   = math.asin(math.sin(φ1) * math.cos(d) +
                     math.cos(φ1) * math.sin(d) * math.cos(brng))
    λ2   = λ1 + math.atan2(math.sin(brng) * math.sin(d) * math.cos(φ1),
                            math.cos(d) - math.sin(φ1) * math.sin(φ2))
    return math.degrees(φ2), math.degrees(λ2)


# ── Scenario 1: Oscillating shifts ───────────────────────────────
# Wind oscillates ±8° on a 12s cycle (6s half-period).
# With SIM_PERSISTENCE_WINDOW=8, each half-cycle is shorter than the
# window so the shift never goes persistent. LLM should say "hold".

def _oscillating_fn(step_idx: int, elapsed: float) -> InstrumentData:
    twd     = 260.0 + 8.0 * math.sin(2 * math.pi * elapsed / 12.0)
    heading = 220.0   # starboard tack, ~40° TWA
    return InstrumentData(
        twd=twd, tws=BASE_TWS, heading=heading,
        bsp=BASE_BSP, sog=BASE_BSP,
        awa=twd - heading, aws=BASE_TWS,
        lat=BASE_LAT, lon=BASE_LON,
    )

SCENARIO_OSCILLATING = SimScenario(
    name="oscillating_shifts",
    description=(
        "Wind oscillates ±8° every 12s. Shift never persists — "
        "LLM should consistently hold."
    ),
    steps=[SimStep(90.0, _oscillating_fn, "oscillating ±8° around 260°")],
)


# ── Scenario 2: Persistent header ────────────────────────────────
# Wind backs steadily 10° over 90s. After SIM_PERSISTENCE_WINDOW the
# engine marks it persistent and the LLM should call the tack.
# Boat is off layline.

def _persistent_header_fn(step_idx: int, elapsed: float) -> InstrumentData:
    rate    = 10.0 / 90.0
    twd     = 260.0 - rate * min(elapsed, 90.0)
    heading = 220.0   # starboard tack
    return InstrumentData(
        twd=twd, tws=BASE_TWS, heading=heading,
        bsp=BASE_BSP, sog=BASE_BSP,
        awa=twd - heading, aws=BASE_TWS,
        lat=BASE_LAT, lon=BASE_LON,
    )

SCENARIO_PERSISTENT_HEADER = SimScenario(
    name="persistent_header",
    description=(
        "Wind backs 10° over 90s. After the persistence window, "
        "LLM calls the tack (boat is off layline)."
    ),
    steps=[SimStep(90.0, _persistent_header_fn, "header building steadily")],
)


# ── Scenario 3: Approaching layline ──────────────────────────────
# Steady wind. Boat position moves toward the starboard layline over
# 3 min. TTM drops. LLM should warn then call sail_to_mark.
# Starboard layline bearing (from boat to mark) ≈ 218° with twd=260, opt_twa=42.

def _approaching_layline_fn(step_idx: int, elapsed: float) -> InstrumentData:
    twd     = 260.0
    heading = 220.0
    # Start ~0.08° west of the direct upwind line, converge to layline over 180s
    lon_offset = -0.04 + (0.04 * elapsed / 180.0)
    return InstrumentData(
        twd=twd, tws=BASE_TWS, heading=heading,
        bsp=BASE_BSP, sog=BASE_BSP,
        awa=twd - heading, aws=BASE_TWS,
        lat=BASE_LAT, lon=BASE_LON + lon_offset,
    )

SCENARIO_APPROACHING_LAYLINE = SimScenario(
    name="approaching_layline",
    description=(
        "Steady wind, boat converges on starboard layline over 3 min. "
        "TTM drops — LLM warns then calls sail to mark."
    ),
    steps=[SimStep(180.0, _approaching_layline_fn, "converging on stbd layline")],
)


# ── Scenario 4: Persistent header on layline ─────────────────────
# Same backing wind as scenario 2, but boat IS already on the layline.
# Engine should output sail_to_mark. LLM should hold.
# Place boat on stbd layline: bearing from boat to mark ≈ stbd_heading = 218°.

_LAYLINE_LAT, _LAYLINE_LON = _dest_coords(MARK_LAT, MARK_LON, (218 + 180) % 360, 0.30)

def _header_on_layline_fn(step_idx: int, elapsed: float) -> InstrumentData:
    rate    = 10.0 / 90.0
    twd     = 260.0 - rate * min(elapsed, 90.0)
    heading = 220.0
    return InstrumentData(
        twd=twd, tws=BASE_TWS, heading=heading,
        bsp=BASE_BSP, sog=BASE_BSP,
        awa=twd - heading, aws=BASE_TWS,
        lat=_LAYLINE_LAT, lon=_LAYLINE_LON,
    )

SCENARIO_HEADER_ON_LAYLINE = SimScenario(
    name="header_on_layline",
    description=(
        "Wind backs persistently but boat is already on the layline. "
        "LLM should hold and sail to mark."
    ),
    steps=[SimStep(90.0, _header_on_layline_fn, "persistent header — on layline")],
)


# ── Scenario 5: Overstanding ─────────────────────────────────────
# Boat has sailed 5° past the starboard layline from frame 1.
# Engine returns overstanding=True immediately.
# LLM should urgently call the tack.
# Stbd layline bearing = 218°. Overstand → bearing to mark ≈ 213°.
# Place boat 0.3 nm from mark along bearing 213°+180° = 33°.

_OVER_LAT, _OVER_LON = _dest_coords(MARK_LAT, MARK_LON, 33.0, 0.30)

def _overstanding_fn(step_idx: int, elapsed: float) -> InstrumentData:
    return InstrumentData(
        twd=260.0, tws=BASE_TWS, heading=220.0,
        bsp=BASE_BSP, sog=BASE_BSP,
        awa=40.0, aws=BASE_TWS,
        lat=_OVER_LAT, lon=_OVER_LON,
    )

SCENARIO_OVERSTANDING = SimScenario(
    name="overstanding",
    description=(
        "Boat is 5° past the starboard layline from the start. "
        "LLM should urgently call the tack."
    ),
    steps=[SimStep(60.0, _overstanding_fn, "overstanding by 5°")],
)


# ── Scenario registry ─────────────────────────────────────────────

ALL_SCENARIOS: dict = {s.name: s for s in [
    SCENARIO_OSCILLATING,
    SCENARIO_PERSISTENT_HEADER,
    SCENARIO_APPROACHING_LAYLINE,
    SCENARIO_HEADER_ON_LAYLINE,
    SCENARIO_OVERSTANDING,
]}


# ── Runner ────────────────────────────────────────────────────────

class ScenarioRunner:
    """
    Async runner that feeds SimScenario data through TacticsEngine and
    populates main.py's global state so the existing /tactics endpoint
    and frontend UI work without any changes.
    """

    def __init__(self, engine, sim_state: SimState, globals_ref):
        """
        engine       — TacticsEngine instance from main.py
        sim_state    — SimState instance (mutated by runner, read by endpoints)
        globals_ref  — sys.modules['main'] — provides latest_inst, latest_state,
                       latest_advice, advice_cooldown
        """
        self._engine    = engine
        self._sim_state = sim_state
        self._g         = globals_ref

    def start(self, scenario: SimScenario) -> asyncio.Task:
        self.stop()
        task = asyncio.create_task(self._run(scenario))
        self._sim_state._task = task
        return task

    def stop(self):
        if self._sim_state._task and not self._sim_state._task.done():
            self._sim_state._task.cancel()
        self._sim_state.running = False

    async def _run(self, scenario: SimScenario):
        self._sim_state.running       = True
        self._sim_state.scenario_name = scenario.name

        # Reset engine (clears wind history, shift state, baselines)
        self._engine.__init__()
        self._engine.set_mark(MARK_LAT, MARK_LON)

        # Wire in uploaded polar if one is active
        import main as _main
        if _main.polar_manager.active:
            self._engine.set_polar(_main.polar_manager.active)

        # Force LLM to fire on the very first tick
        self._g.advice_cooldown = 0

        try:
            for step_idx, step in enumerate(scenario.steps):
                self._sim_state.current_step_label = step.label
                step_start = time.monotonic()

                while True:
                    elapsed = time.monotonic() - step_start
                    if elapsed >= step.duration_seconds:
                        break

                    inst = step.data_fn(step_idx, elapsed)

                    # Push through the full pipeline
                    self._g.latest_inst  = inst
                    self._g.latest_state = self._engine.update(inst)

                    # LLM cooldown — same logic as on_instrument_update
                    self._g.advice_cooldown -= 1
                    significant = abs(self._g.latest_state.shift_degrees) > 5
                    if significant or self._g.advice_cooldown <= 0:
                        from llm_bridge import get_tactical_advice
                        self._g.latest_advice   = get_tactical_advice(self._g.latest_state)
                        self._g.advice_cooldown = 30

                    # Update elapsed for status endpoint
                    step_offset = sum(s.duration_seconds for s in scenario.steps[:step_idx])
                    self._sim_state.elapsed_seconds = step_offset + elapsed

                    await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            pass
        finally:
            self._sim_state.running = False
