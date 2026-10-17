# jarvis-install: tools
"""
tools/actions.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
F-32 Action log: "what did you do today?"

The Toolbox writes every tool call to ~/.local/state/jarvis/actions/<date>.jsonl
(time, your request, tool, arguments, outcome). This plugin reads it back.
Old days are deleted together with the conversation history ([history] keep_days).
To read the raw log yourself:  less ~/.local/state/jarvis/actions/$(date +%F).jsonl
"""
import json
from collections import Counter
from datetime import date, datetime, timedelta

import tools
from tools import tool

PROMPT = ("recent_actions lists what J.A.R.V.I.S. actually did (tools that ran, what changed, what was blocked). "
          "Use it for questions like \"what did you do today\"; summarize it in a sentence or two, most important first.")
EXAMPLES = ["what did you do today", "what have you done so far", "what did you change yesterday",
            "did you open anything", "what actions did you take", "show me your action log"]

MAX_LISTED = 20


def _day(text):
    text = str(text or "today").strip().lower()
    if text in ("today", ""):
        return date.today()
    if text == "yesterday":
        return date.today() - timedelta(days=1)
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise ValueError("day must be today, yesterday or a date like 2026-10-16")


def _brief_args(args):
    parts = [f"{k}={v}" for k, v in (args or {}).items()]
    text = ", ".join(parts)
    return text if len(text) <= 80 else text[:79] + "…"


def read_actions(day):
    path = tools.ACTIONS_DIR / f"{day:%Y-%m-%d}.jsonl"
    found = []
    try:
        for line in path.read_text().splitlines():
            try:
                found.append(json.loads(line))
            except ValueError:
                continue  # a half-written line from a crash
    except OSError:
        pass
    return found


@tool(untrusted=True)
def recent_actions(day: str = "today", changes_only: bool = False) -> str:
    """List what J.A.R.V.I.S. did (tool calls and their outcomes) on a day.

    Args:
        day: today, yesterday, or a date like 2026-10-16
        changes_only: true to list only actions that change something
    """
    when = _day(day)
    actions = read_actions(when)
    if changes_only:
        actions = [a for a in actions if a.get("writes")]
    label = "today" if when == date.today() else ("yesterday" if when == date.today() - timedelta(days=1) else f"on {when:%A, %B %d}")
    if not actions:
        return f"No {'changes' if changes_only else 'actions'} logged {label}."
    counts = Counter(a.get("outcome", "ok") for a in actions)
    head = f"{len(actions)} action{'s' if len(actions) != 1 else ''} {label}"
    extra = [f"{counts[k]} {k}" for k in ("error", "blocked", "unknown") if counts.get(k)]
    if extra:
        head += f" ({', '.join(extra)})"
    lines = [head + ". Most recent last:"]
    shown = actions[-MAX_LISTED:]
    if len(actions) > MAX_LISTED:
        older = Counter(a["tool"] for a in actions[:-MAX_LISTED])
        lines.append("Earlier: " + ", ".join(f"{tool_name} ×{n}" for tool_name, n in older.most_common(8)) + ".")
    for a in shown:
        stamp = datetime.fromisoformat(a["time"]).strftime("%I:%M %p").lstrip("0")
        outcome = "" if a.get("outcome") == "ok" else f" [{a.get('outcome')}]"
        lines.append(f"{stamp} {a['tool']}({_brief_args(a.get('args'))}){outcome}")
    return "\n".join(lines)
