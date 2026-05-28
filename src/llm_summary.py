import os
from datetime import date
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

try:
    from google import genai as _genai
    from google.genai import types as _genai_types
except ImportError:
    _genai = None
    _genai_types = None


def _build_prompt(station, planned_date, risk, prob, disrupted, delay,
                  nearby_closures, mean_delay_history, late_share_history) -> str:
    closure_lines = []
    for cl in nearby_closures[:5]:
        road  = cl.get("road_name") or cl.get("road", "unknown road")
        cause = cl.get("cause_type", "unknown cause")
        dur   = cl.get("duration_hours", 0)
        dist  = cl.get("distance_km", "?")
        ctype = cl.get("closure_type", "unplanned")
        closure_lines.append(f"  - {ctype.title()} closure on {road} ({cause}), {dist} km away, lasting {dur:.1f} hours")

    closures_text = "\n".join(closure_lines) if closure_lines else "  - No nearby road closures found"
    hist_text = (
        f"Historical context: mean arrival delay {mean_delay_history:.1f} min, "
        f"{(late_share_history or 0) * 100:.0f}% of services run late on average.\n"
        if mean_delay_history is not None else ""
    )
    direction = "late" if delay > 0.5 else "early" if delay < -0.5 else "on time"

    return (
        f"You are an operational transport analyst writing a concise briefing for a station manager.\n"
        f"Write 3-4 sentences in plain English (no bullet points, no headers) explaining the disruption\n"
        f"prediction below. Be specific about the road closures and their likely impact. Use UK English.\n"
        f"Do not use em dashes. Do not begin with 'This station' or 'The model'. Avoid vague phrases\n"
        f"like 'significant impact' or 'please note'. Be direct and human.\n\n"
        f"Station: {station}\n"
        f"Date: {planned_date.strftime('%A %d %B %Y')}\n"
        f"Risk band: {risk.upper()}\n"
        f"Disruption probability: {prob:.0%}\n"
        f"Disruption predicted: {'Yes' if disrupted else 'No'}\n"
        f"Predicted mean delay: {delay:+.1f} minutes ({direction})\n"
        f"{hist_text}"
        f"Nearby road closures:\n{closures_text}\n\nWrite the briefing now:"
    )


def _summary_html(text: str, risk: str) -> str:
    accent = {"high": "#d4351c", "medium": "#f47738", "low": "#00703c", "critical": "#912b11"}.get(risk.lower(), "#1d70b8")
    return (
        f"<div class='rr-llm-summary' style='border-left-color:{accent}'>"
        f"<div class='rr-llm-label'>Operational briefing</div>"
        f"<p class='rr-llm-text'>{text}</p>"
        f"</div>"
    )


def _error_html(msg: str) -> str:
    return f"<div class='rr-llm-summary rr-llm-error'>Briefing unavailable: {msg}</div>"


def generate_briefing(station: str, planned_date: date, risk: str, prob: float,
                      disrupted: bool, delay: float, nearby_closures: list[dict],
                      mean_delay_history: Optional[float] = None,
                      late_share_history: Optional[float] = None) -> str:
    if _genai is None:
        return _error_html("google-genai not installed.")
    if not GEMINI_API_KEY:
        return _error_html("GEMINI_API_KEY not set.")

    prompt = _build_prompt(station, planned_date, risk, prob, disrupted, delay,
                           nearby_closures, mean_delay_history, late_share_history)
    try:
        client   = _genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=_genai_types.GenerateContentConfig(max_output_tokens=300),
        )
        return _summary_html(response.text.strip(), risk)
    except Exception as exc:
        err = str(exc)
        if "api_key" in err.lower() or "401" in err or "permission" in err.lower():
            return _error_html("API key not configured.")
        return _error_html(err[:120])