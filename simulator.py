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
import concurrent.futures
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from instruments.base import InstrumentData

# Shared executor for off-thread LLM calls so they don't block the sim loop
_llm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm")

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


# ── Race simulation helpers ───────────────────────────────────────

def _advance_position(lat: float, lon: float, heading_deg: float, dist_nm: float):
    """Return (lat, lon) after sailing dist_nm on heading_deg."""
    return _dest_coords(lat, lon, heading_deg, dist_nm)


class RaceSimulation:
    """
    Stateful callable used as a SimStep.data_fn for the race-to-mark scenario.

    The autopilot reads `globals_ref.latest_state` each tick and follows the
    engine's tack_recommendation:
      - "tack"         → flip tack (mirror heading around the wind)
      - "sail_to_mark" → point at bearing_to_mark
      - anything else  → hold current heading

    Wind follows a gentle mean-reverting random walk (±0.3° per tick,
    restoring toward origin at 5% per tick).
    """

    TACK_COOLDOWN_S   = 6.0    # wall-clock seconds between tacks (= 15 s at 2.5× speed)
    ARRIVAL_NM        = 0.05   # stop when within 0.05 nm of mark
    BSP               = 6.5    # boat speed knots (fixed for now)
    TICK_S            = 0.5    # simulated seconds of sailing per tick

    def __init__(self, globals_ref, mark_lat: float, mark_lon: float,
                 tws: float, twd: float,
                 boat_lat: float = BASE_LAT, boat_lon: float = BASE_LON):
        self._g        = globals_ref
        self.mark_lat  = mark_lat
        self.mark_lon  = mark_lon
        self.tws       = tws
        self.twd_orig  = twd        # mean-reversion target
        self.twd       = twd        # current true wind direction

        # Boat starts at the given position on starboard tack.
        # Starboard tack: twa = twd - heading > 0  →  heading = twd - 42°
        self.heading   = (twd - 42.0 + 360.0) % 360.0
        self.on_stbd   = True
        self.lat       = boat_lat
        self.lon       = boat_lon

        self.last_tack_time: float = -999.0    # monotonic
        self.arrived   = False

    # ------------------------------------------------------------------ #
    def __call__(self, step_idx: int, elapsed: float) -> InstrumentData:
        """Called every 0.5s tick by ScenarioRunner._run()."""
        now = time.monotonic()

        # 1. Advance wind (mean-reverting random walk, ±0.3°/tick, 5% restore)
        delta      = random.gauss(0, 0.3)
        restore    = 0.05 * ((self.twd_orig - self.twd + 180) % 360 - 180)
        self.twd   = (self.twd + delta + restore) % 360.0

        # 2. Read latest engine state for autopilot decisions
        state = self._g.latest_state
        if state is not None and (now - self.last_tack_time) > self.TACK_COOLDOWN_S:
            rec = state.tack_recommendation

            if rec == "tack":
                # Mirror heading through the wind: new_heading = twd * 2 − old_heading (mod 360)
                self.heading     = (2 * self.twd - self.heading) % 360.0
                self.on_stbd     = not self.on_stbd
                self.last_tack_time = now

            elif rec == "sail_to_mark" and state.bearing_to_mark is not None:
                self.heading = state.bearing_to_mark

        # 3. Advance position
        dist_per_tick = self.BSP * (self.TICK_S / 3600.0)   # nm per tick
        self.lat, self.lon = _advance_position(
            self.lat, self.lon, self.heading, dist_per_tick
        )

        # 4. Check arrival (flag read by ScenarioRunner)
        if state is not None and state.dist_to_mark_nm is not None:
            if state.dist_to_mark_nm < self.ARRIVAL_NM:
                self.arrived = True

        # 5. Build InstrumentData
        # awa = twd - heading (positive = starboard, consistent with other scenarios)
        raw = (self.twd - self.heading + 360.0) % 360.0
        awa = raw - 360.0 if raw > 180.0 else raw
        return InstrumentData(
            twd=self.twd, tws=self.tws, heading=self.heading,
            bsp=self.BSP, sog=self.BSP,
            awa=awa, aws=self.tws,
            lat=self.lat, lon=self.lon,
        )


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

    def start_race(self) -> asyncio.Task:
        """
        Start a live race simulation with random wind.
        TWD is chosen randomly 0–360°; TWS in range 12–17 kts.
        Mark is placed ~0.6 nm directly upwind of the starting position.
        """
        self.stop()
        tws = random.uniform(12.0, 17.0)
        # Northerly wind (315°–45°) so the mark always appears above the
        # starting position on a north-up chart — visually intuitive.
        twd = random.uniform(315.0, 405.0) % 360.0

        # Mark is exactly 0.6 nm upwind of BASE; boat starts exactly at BASE.
        mark_lat, mark_lon = _dest_coords(BASE_LAT, BASE_LON, twd, 0.60)
        race = RaceSimulation(
            self._g, mark_lat, mark_lon, tws, twd,
            boat_lat=BASE_LAT, boat_lon=BASE_LON,
        )

        # Max 30 min of simulated time; at 2.5× wall-clock that's ~12 min real
        MAX_RACE_S = 1800.0
        step = SimStep(
            duration_seconds=MAX_RACE_S,
            data_fn=race,
            label=f"racing upwind  TWD={twd:.0f}°  TWS={tws:.1f}kt",
        )
        scenario = SimScenario(
            name="race_to_mark",
            description=(
                f"Autonomous race to upwind mark. "
                f"TWD {twd:.0f}°, TWS {tws:.1f} kts."
            ),
            steps=[step],
        )

        # tick_s=0.2 → 5 ticks/s wall-clock × 0.5 s simulated/tick = 2.5× speed
        task = asyncio.create_task(self._run(
            scenario, race_sim=race,
            mark_lat=mark_lat, mark_lon=mark_lon,
            tick_s=0.2,
        ))
        self._sim_state._task = task
        return task

    def stop(self):
        if self._sim_state._task and not self._sim_state._task.done():
            self._sim_state._task.cancel()
        self._sim_state.running = False

    async def _run(self, scenario: SimScenario, *,
                   race_sim: Optional[RaceSimulation] = None,
                   mark_lat: Optional[float] = None,
                   mark_lon: Optional[float] = None,
                   tick_s: float = 0.5):
        self._sim_state.running       = True
        self._sim_state.scenario_name = scenario.name

        # Reset engine (clears wind history, shift state, baselines)
        self._engine.__init__()

        # Set mark — race uses dynamic mark, scenarios use fixed mark
        m_lat = mark_lat if mark_lat is not None else MARK_LAT
        m_lon = mark_lon if mark_lon is not None else MARK_LON
        self._engine.set_mark(m_lat, m_lon)

        # Wire in uploaded polar if one is active
        import main as _main
        if _main.polar_manager.active:
            self._engine.set_polar(_main.polar_manager.active)

        # Reset live-mode LLM cooldown so next real update also fires promptly
        self._g.advice_cooldown = 0

        # LLM timing: fire immediately on start, then no more often than every
        # LLM_MIN_INTERVAL_S wall-clock seconds (avoids blocking at 5 Hz).
        LLM_MIN_INTERVAL_S = 8.0
        last_llm_wall: float = -999.0
        llm_in_flight: bool  = False

        async def _fire_llm(state) -> None:
            """Call the LLM in a thread so it doesn't block the sim loop."""
            nonlocal llm_in_flight, last_llm_wall
            if llm_in_flight:
                return
            llm_in_flight  = True
            last_llm_wall  = time.monotonic()
            try:
                from llm_bridge import get_tactical_advice
                loop   = asyncio.get_event_loop()
                advice = await loop.run_in_executor(_llm_executor, get_tactical_advice, state)
                self._g.latest_advice = advice
            finally:
                llm_in_flight = False

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

                    # LLM — fire async if enough wall-clock time has passed
                    now_wall = time.monotonic()
                    if (now_wall - last_llm_wall) >= LLM_MIN_INTERVAL_S:
                        asyncio.create_task(_fire_llm(self._g.latest_state))

                    # Update elapsed for status endpoint
                    step_offset = sum(s.duration_seconds for s in scenario.steps[:step_idx])
                    self._sim_state.elapsed_seconds = step_offset + elapsed

                    # Race arrival check
                    if race_sim is not None and race_sim.arrived:
                        self._sim_state.current_step_label = (
                            f"✓ Mark reached in {self._sim_state.elapsed_seconds:.0f}s!"
                        )
                        return

                    await asyncio.sleep(tick_s)

        except asyncio.CancelledError:
            pass
        finally:
            self._sim_state.running = False
