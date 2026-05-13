"""
Advanced Sailing Tactics System — FastAPI Server
Serves the tablet UI and exposes the /tactics API endpoint.
Detects B&G vs Garmin instruments on startup via Signal K.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from instruments.bg_adapter import BGAdapter
from instruments.detector import InstrumentBrand, detect_brand
from instruments.garmin_adapter import GarminAdapter
from llm_bridge import get_tactical_advice
from polar.manager import PolarManager
from signalk_client import SignalKClient
from simulator import ALL_SCENARIOS, ScenarioRunner, SimState
from tactics_engine import TacticsEngine

load_dotenv()

# ── Shared state ──────────────────────────────────────────────────
polar_manager   = PolarManager()
engine          = TacticsEngine()
latest_inst     = None   # most recent InstrumentData from adapter
latest_state    = None   # most recent TacticalState from engine
latest_advice   = ""
advice_cooldown = 0
brand           = InstrumentBrand.UNKNOWN
adapter         = None

# ── Simulation state ──────────────────────────────────────────────
sim_state:  SimState                    = SimState()
sim_runner: Optional[ScenarioRunner]   = None


def on_instrument_update(raw: dict):
    """Called by SignalKClient every time new data arrives."""
    global latest_inst, latest_state, latest_advice, advice_cooldown

    if adapter is None:
        return

    inst = adapter.extract(raw)
    latest_inst  = inst
    latest_state = engine.update(inst)

    advice_cooldown -= 1
    significant_shift = abs(latest_state.shift_degrees) > 5
    if significant_shift or advice_cooldown <= 0:
        latest_advice   = get_tactical_advice(latest_state)
        advice_cooldown = 30  # refresh every ~60 s at 2 s poll rate


@asynccontextmanager
async def lifespan(app: FastAPI):
    global brand, adapter, sim_runner

    # Detect instrument brand from Signal K network
    brand   = await detect_brand()
    adapter = (
        BGAdapter()
        if brand in (InstrumentBrand.BG_H5000, InstrumentBrand.BG_STANDARD)
        else GarminAdapter()
    )
    print(f"[tactics] Instrument brand detected: {brand.value}")

    # Create simulator runner (always available, even in live mode)
    sim_runner = ScenarioRunner(engine, sim_state, sys.modules[__name__])

    # Start Signal K only when not in SIM_ONLY mode
    sim_only = os.getenv("SIM_ONLY", "false").lower() == "true"
    if sim_only:
        print("[tactics] SIM_ONLY mode — SignalK client not started")
        task = None
    else:
        sk_client = SignalKClient(on_update=on_instrument_update)
        task = asyncio.create_task(sk_client.connect_with_retry())

    yield

    if task:
        task.cancel()
    sim_runner.stop()


app = FastAPI(title="Advanced Tactics Display", lifespan=lifespan)


# ── Tactics endpoint ──────────────────────────────────────────────
@app.get("/tactics")
async def get_tactics():
    if latest_state is None or latest_inst is None:
        return JSONResponse({"status": "waiting for instruments", "brand": brand.value})

    inst  = latest_inst
    state = latest_state

    return {
        "brand":              brand.value,
        "performance_source": state.performance_source,

        # Raw instrument values
        "tws":     round(inst.tws, 2),
        "twd":     round(inst.twd, 1),
        "awd":     round(inst.awa - inst.twd, 1),   # apparent wind offset from true
        "sog":     round(inst.sog, 2),
        "awa":     round(inst.awa, 2),
        "bsp":     round(inst.bsp, 2),
        "heading": round(inst.heading, 1),
        "heel":    round(inst.heel, 1),
        "lat":     inst.lat,
        "lon":     inst.lon,

        # Computed tactics
        "vmg":         state.vmg,
        "target_vmg":  state.target_vmg,
        "vmg_pct":     state.vmg_performance_pct,
        "bsp_pct":     state.bsp_performance_pct,
        "shift":       state.shift_type,
        "shift_deg":   state.shift_degrees,
        "wind_trend":  state.wind_trend,
        "optimal_twa": state.optimal_twa,
        "ttm":         state.ttm_minutes,
        "laylines": {
            "port":      state.port_layline,
            "starboard": state.stbd_layline,
        },

        # Persistent shift & tacking recommendation
        "shift_state":              state.shift_state,
        "shift_age_seconds":        state.shift_age_seconds,
        "on_layline":               state.on_layline,
        "overstanding":             state.overstanding,
        "vmg_current_tack":         state.vmg_current_tack,
        "vmg_other_tack":           state.vmg_other_tack,
        "vmg_gain_from_tack":       state.vmg_gain_from_tack,
        "tack_recommendation":      state.tack_recommendation,
        "tack_recommendation_reason": state.tack_recommendation_reason,
        "bearing_to_mark":          state.bearing_to_mark,
        "dist_to_mark_nm":          state.dist_to_mark_nm,

        # LLM advice
        "advice": latest_advice,
    }


# ── Polar endpoints ───────────────────────────────────────────────
@app.get("/polars")
async def list_polars():
    return {"boats": polar_manager.list_boats()}


@app.post("/polars/upload")
async def upload_polar(file: UploadFile, boat_name: str = Form(...)):
    content = (await file.read()).decode("utf-8")
    try:
        polar = polar_manager.upload(file.filename, content, boat_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "status":     "uploaded",
        "boat":       boat_name,
        "tws_range":  f"{min(polar.tws_values)}–{max(polar.tws_values)} kts",
        "twa_range":  f"{min(polar.twa_values)}–{max(polar.twa_values)}°",
        "upwind_angles": polar.upwind_vmg_angles,
    }


@app.post("/polars/activate")
async def activate_polar(boat_name: str = Form(...)):
    try:
        polar = polar_manager.set_active(boat_name)
        engine.set_polar(polar)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No polar found for '{boat_name}'")
    return {"status": "active", "boat": boat_name}


@app.get("/polars/active")
async def active_polar():
    p = polar_manager.active
    if p is None:
        return {"active": None}
    return {
        "active":         p.boat_name,
        "tws_range":      f"{min(p.tws_values)}–{max(p.tws_values)} kts",
        "upwind_angles":  p.upwind_vmg_angles,
        "downwind_angles": p.downwind_vmg_angles,
    }


# ── Mark endpoint (tablet sets mark position) ─────────────────────
@app.post("/mark")
async def set_mark(lat: float = Form(...), lon: float = Form(...)):
    engine.set_mark(lat, lon)
    return {"status": "ok", "lat": lat, "lon": lon}


@app.get("/mark")
async def get_mark():
    if engine.mark_lat is None:
        return {"mark": None}
    return {"lat": engine.mark_lat, "lon": engine.mark_lon}


# ── Status endpoint ───────────────────────────────────────────────
@app.get("/status")
async def status():
    return {
        "brand":        brand.value,
        "instruments":  latest_inst is not None,
        "polar_loaded": polar_manager.active is not None,
        "polar_boat":   polar_manager.active.boat_name if polar_manager.active else None,
        "mark_set":     engine.mark_lat is not None,
    }


# ── Simulation endpoints ──────────────────────────────────────────
@app.get("/sim/scenarios")
async def list_sim_scenarios():
    return {
        "scenarios": [
            {
                "name":          s.name,
                "description":   s.description,
                "step_count":    len(s.steps),
                "total_seconds": sum(st.duration_seconds for st in s.steps),
            }
            for s in ALL_SCENARIOS.values()
        ]
    }


@app.post("/sim/start/{scenario_name}")
async def start_sim(scenario_name: str):
    if scenario_name not in ALL_SCENARIOS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario '{scenario_name}'. "
                   f"Available: {list(ALL_SCENARIOS.keys())}",
        )
    sim_runner.start(ALL_SCENARIOS[scenario_name])
    return {"status": "started", "scenario": scenario_name}


@app.post("/sim/stop")
async def stop_sim():
    sim_runner.stop()
    return {"status": "stopped"}


@app.get("/sim/status")
async def get_sim_status():
    return {
        "running":            sim_state.running,
        "scenario_name":      sim_state.scenario_name,
        "current_step_label": sim_state.current_step_label,
        "elapsed_seconds":    round(sim_state.elapsed_seconds, 1),
    }


# ── Static files (tablet UI) ──────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
