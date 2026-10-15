# jarvis-install: tools
"""
tools/briefing.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
The welcome briefing. It asks every loaded plugin that has a briefing()
function for a sentence or two (weather, tasks, desktop, and later the
calendar) and puts them together, in the order plugins are listed in
config.toml. A plugin that fails is reported in one line and skipped, so one
broken service never spoils the whole briefing.
"""
import sys
from datetime import datetime

import tools
from tools import tool

PROMPT = """Briefings: when Tony greets you to start his day or after time away ("good morning", "I'm back", "what did I miss", "welcome me"), or asks for a briefing, status or rundown, call get_briefing.
Turn the result into a short spoken welcome: greet him, give the weather, then anything urgent (weather alerts, overdue or due tasks), then the rest. A briefing may use up to five short sentences.
For a plain hello, just greet him and offer a briefing."""


def _greeting(hour):
    if hour < 5:
        return "Working late, sir"
    if hour < 12:
        return "Good morning, sir"
    if hour < 17:
        return "Good afternoon, sir"
    return "Good evening, sir"


@tool
def get_briefing() -> str:
    """Get a welcome briefing: the time, weather, tasks that are due, open apps and what's playing."""
    now = datetime.now()
    clock = f"{now.hour % 12 or 12}:{now.minute:02d} {'AM' if now.hour < 12 else 'PM'}"
    parts = [f"{_greeting(now.hour)}. It is {now:%A, %B} {now.day}, {clock}."]
    this_module = sys.modules[__name__]
    for name, module in tools.LOADED.items():
        section = getattr(module, "briefing", None)
        if module is this_module or not callable(section):
            continue
        try:
            text = section()
        except Exception as e:  # ValueError messages are already readable sentences
            text = f"{name.capitalize()}: unavailable ({str(e).rstrip('.')})."
        if text:
            parts.append(" ".join(str(text).split()))
    return " ".join(parts)
