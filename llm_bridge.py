"""
LLM Bridge — Claude API integration.
Sends tactical state to Claude and returns concise tactical advice.
Uses prompt caching on the system prompt to minimise cost and latency.
"""

import os

import anthropic
from dotenv import load_dotenv

from tactics_engine import TacticalState

load_dotenv(override=True)

_client: anthropic.Anthropic | None = None

def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client

SYSTEM_PROMPT = """You are an expert sailing tactician embedded aboard a racing yacht.
You receive real-time instrument data and pre-computed tactical analysis.

The engine has already classified the wind shift and computed the recommended action.
Your job is to communicate it clearly and add any relevant nuance.

Rules you must follow:
- Maximum 2 sentences. Be concise and direct.
- Use sailing shorthand: "tack now", "hold", "approaching stbd layline", "you're lifted".
- The engine's tack_recommendation is authoritative — back it up with the reason given.
- If shift_state is "transient", always say to hold and how many seconds remain to confirm.
- If shift_state is "persistent" and headed and NOT on layline: call the tack clearly.
- If on layline with a persistent header: explicitly say to hold and sail to the mark.
- If overstanding: urgently call the tack.
- If lifted: tell them to hold and enjoy it.
- Flag heel > 28° with a reef suggestion.
- Flag TTM < 2 min with an explicit layline call.
- Starboard-tack approach preference: a final starboard tack to the mark gives right-of-way
  over port-tack boats at the rounding — treat this as worth roughly 1 minute of extra distance.
  When the engine recommends tacking to set up a starboard layline approach, mention this advantage.
- Never explain the algorithm. Just give the call.

Respond with tactical advice only — no preamble, no explanations."""


def get_tactical_advice(state: TacticalState) -> str:
    """
    Build a compact instrument snapshot and ask Claude for tactical advice.
    Returns advice string, or empty string on any error.
    """
    try:
        context = _build_context(state)

        response = _get_client().messages.create(
            model="claude-opus-4-5",
            max_tokens=120,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
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
        f"TWA: {state.twa:.0f}°  |  Optimal TWA: {state.optimal_twa:.0f}°",
        f"Wind shift since last tack: {state.shift_type} {abs(state.shift_degrees):.1f}°",
        f"Shift state: {state.shift_state} (held for {state.shift_age_seconds:.0f}s)",
        f"Wind trend (5 min): {state.wind_trend}",
        f"VMG current tack toward mark: {state.vmg_current_tack:.2f} kts",
        f"VMG other tack toward mark:   {state.vmg_other_tack:.2f} kts",
        f"Net VMG gain from tacking (after penalty): {state.vmg_gain_from_tack:+.2f} kts",
        f"On layline: {state.on_layline}  |  Overstanding: {state.overstanding}",
        f"Time to maneuver: {state.ttm_minutes:.1f} min",
        f"Heel: {state.heel:.0f}°",
    ]

    if state.dist_to_mark_nm is not None:
        lines.append(f"Distance to mark: {state.dist_to_mark_nm:.2f} nm  |  Bearing: {state.bearing_to_mark:.0f}°T")

    if state.target_vmg > 0:
        lines.append(f"VMG performance: {state.vmg_performance_pct:.0f}% of polar target")

    if state.bsp_performance_pct > 0:
        lines.append(f"Boat speed performance: {state.bsp_performance_pct:.0f}% of polar")

    lines += [
        f"Performance data source: {state.performance_source}",
        f"",
        f"ENGINE RECOMMENDATION: {state.tack_recommendation.upper()}",
        f"Reason: {state.tack_recommendation_reason}",
    ]

    return "\n".join(lines)
