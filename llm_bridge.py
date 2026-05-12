"""
LLM Bridge — Claude API integration.
Sends tactical state to Claude and returns concise tactical advice.
Uses prompt caching on the system prompt to minimise cost and latency.
"""

import os

import anthropic

from tactics_engine import TacticalState

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are an expert sailing tactician embedded aboard a racing yacht.
You receive real-time instrument data processed from B&G or Garmin sensors via NMEA 2000.

Your role:
- Give CONCISE, ACTIONABLE tactical advice — maximum 2 sentences.
- Use sailing shorthand where appropriate (e.g. "tack now", "hold", "approaching stbd layline").
- Consider shift persistence: don't overreact to small or temporary fluctuations.
- If the wind is veering, advise favouring starboard. If backing, favour port.
- Flag if close to layline (TTM < 3 min) with an explicit call.
- Flag if heel suggests overpowered (>28°) with a reef recommendation.
- Distinguish oscillating shifts (sail the headers) from persistent shifts (tack on it).
- When lifted, generally hold the tack. When headed, consider tacking.

Respond with tactical advice only — no explanations, no preamble."""


def get_tactical_advice(state: TacticalState) -> str:
    """
    Build a compact instrument snapshot and ask Claude for tactical advice.
    Returns advice string, or empty string on any error.
    """
    try:
        context = _build_context(state)

        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=120,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},   # cache system prompt
                }
            ],
            messages=[{"role": "user", "content": context}],
        )
        return response.content[0].text.strip()

    except anthropic.APIError as exc:
        print(f"[llm] API error: {exc}")
        return ""
    except Exception as exc:
        print(f"[llm] Unexpected error: {exc}")
        return ""


def _build_context(state: TacticalState) -> str:
    tack = "starboard" if state.on_starboard else "port"

    lines = [
        f"Current tack: {tack}",
        f"Wind shift since last tack/jibe: {state.shift_type} {abs(state.shift_degrees):.1f}°",
        f"Wind trend (5 min): {state.wind_trend}",
        f"TWA: {state.twa:.0f}°",
        f"VMG: {state.vmg:.1f} kts",
    ]

    if state.target_vmg > 0:
        lines.append(f"VMG performance: {state.vmg_performance_pct:.0f}% of polar target")

    if state.bsp_performance_pct > 0:
        lines.append(f"Boat speed performance: {state.bsp_performance_pct:.0f}% of polar")

    lines += [
        f"Heel: {state.heel:.0f}°",
        f"Port layline bearing: {state.port_layline:.0f}°T",
        f"Stbd layline bearing: {state.stbd_layline:.0f}°T",
        f"Time to maneuver: {state.ttm_minutes:.1f} min",
        f"Performance data source: {state.performance_source}",
    ]

    return "\n".join(lines)
