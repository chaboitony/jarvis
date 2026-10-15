# jarvis-install: tools
"""
tools/timers.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
Timers ("10 minutes for the pizza") and reminders ("at 3 PM, call the lab").

How it works: each one is a temporary systemd timer (systemd-run --user), so it
goes off even if J.A.R.V.I.S. restarts. When it fires, systemd runs
jarvis-alert, which shows a notification that stays until dismissed and plays
an alarm sound. They're lost on reboot or logout (fine for timers; say so for
reminders). A small list of what was set lives in the runtime folder, which is
also cleared at reboot.

Safety: labels are cut to plain letters, digits and basic punctuation before they
reach systemd or the notification, so they can't carry commands or markup.
"""
import json
import os
import re
import secrets
import subprocess
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from tools import tool

PROMPT = """Timers and reminders: use set_timer for a countdown ("in 10 minutes"), set_reminder for a time of day ("at 3 PM"), list_timers to see them and cancel_timer to cancel one. Never guess a timer number; call list_timers first.
For durations, pass minutes (90 seconds = 1.5). Reminders don't survive a reboot; mention that only if asked."""

EXAMPLES = [
    "set a timer for 10 minutes", "remind me at 3 pm to call the lab", "timer for the pizza, 12 minutes",
    "what timers do I have", "cancel the pizza timer", "wake me up in 20 minutes", "remind me tomorrow at 9 to submit homework",
]

ALERT = Path.home() / ".local" / "bin" / "jarvis-alert"
STATE = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "jarvis-timers.json"
MAX_MINUTES = 24 * 60
MAX_DAYS_AHEAD = 7
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _safe_label(text, default):
    """Letters, digits and basic punctuation only; no $, %, <, & or quotes systemd or dunst might act on."""
    text = re.sub(r"[^\w .,!?'-]", " ", str(text or ""))
    text = " ".join(text.split())[:60]
    return text or default


def _load():
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return []


def _save(items):
    temp = STATE.with_name(STATE.name + ".tmp")
    temp.write_text(json.dumps(items))
    temp.replace(STATE)


def _active(items):
    """Only timers still in the future, soonest first."""
    now = time.time()
    return sorted((t for t in items if t["due"] > now - 1), key=lambda t: t["due"])


def _clock(dt):
    return f"{dt.hour % 12 or 12}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def _when_text(due):
    dt = datetime.fromtimestamp(due)
    day = dt.date()
    today = date.today()
    if day == today:
        label = _clock(dt)
    elif day == today + timedelta(days=1):
        label = f"tomorrow at {_clock(dt)}"
    else:
        label = f"{dt:%A} at {_clock(dt)}"
    left = int(due - time.time())
    if left < 3600:
        mins, secs = divmod(max(left, 0), 60)
        label += f" (in {mins} min {secs} s)" if mins < 5 else f" (in {mins} minutes)"
    return label


def _schedule(kind, label, message, *, seconds=None, at=None):
    """Create the systemd timer. Exactly one of seconds (countdown) or at (datetime)."""
    if not ALERT.exists():
        raise ValueError("The jarvis-alert script is missing from ~/.local/bin. Run jarvis-update.")
    unit = f"jarvis-timer-{int(time.time())}-{secrets.token_hex(2)}"
    when = f"--on-active={int(round(seconds))}s" if seconds is not None else f"--on-calendar={at:%Y-%m-%d %H:%M:%S}"
    cmd = ["systemd-run", "--user", f"--unit={unit}", when,
           "--timer-property=AccuracySec=1s",  # the default accuracy is a whole minute
           "--collect", f"--description=J.A.R.V.I.S. {kind}: {label}",
           str(ALERT), f"J.A.R.V.I.S. {kind}", message]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        raise ValueError("systemd-run isn't available, so I can't set timers.")
    if result.returncode != 0:
        raise ValueError(f"systemd couldn't create the timer: {result.stderr.strip()[:150]}")
    due = time.time() + seconds if seconds is not None else at.timestamp()
    items = _active(_load())
    items.append({"unit": unit, "kind": kind, "label": label, "due": due})
    _save(items)
    return due




def _parse_time(text):
    """'15:30', '3:30 pm', '3 pm', '3pm', 'noon', 'midnight' -> (hour, minute)."""
    t = str(text or "").strip().lower().replace(".", "")
    if t in ("noon", "midday"):
        return 12, 0
    if t == "midnight":
        return 0, 0
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", t)
    if not m:
        raise ValueError(f"I can't read the time '{text}'. Use something like 15:30 or 3:30 pm.")
    hour, minute, half = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if half:
        if not 1 <= hour <= 12:
            raise ValueError(f"'{text}' isn't a valid time.")
        hour = hour % 12 + (12 if half == "pm" else 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"'{text}' isn't a valid time.")
    return hour, minute


def _parse_day(text):
    d = " ".join(str(text or "today").lower().split())
    today = date.today()
    if d in ("", "today", "tonight"):
        return today
    if d == "tomorrow":
        return today + timedelta(days=1)
    if d in WEEKDAYS:
        ahead = (WEEKDAYS.index(d) - today.weekday()) % 7 or 7
        return today + timedelta(days=ahead)
    try:
        day = date.fromisoformat(d)
    except ValueError:
        raise ValueError(f"I can't read the day '{text}'. Use today, tomorrow, a weekday or YYYY-MM-DD.")
    if not today <= day <= today + timedelta(days=MAX_DAYS_AHEAD):
        raise ValueError(f"Reminders can be set up to {MAX_DAYS_AHEAD} days ahead.")
    return day


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
@tool(writes=True)
def set_timer(minutes: float, label: str = "Timer") -> str:
    """Start a countdown timer that alerts Tony when time is up.

    Args:
        minutes: how long, in minutes (use decimals for seconds: 1.5 = 90 seconds)
        label: what it's for, e.g. "pizza"
    """
    try:
        minutes = float(minutes)
    except (TypeError, ValueError):
        raise ValueError(f"'{minutes}' isn't a number of minutes.")
    if not 0.1 <= minutes <= MAX_MINUTES:
        raise ValueError("Timers can be from 6 seconds to 24 hours.")
    label = _safe_label(label, "Timer")
    due = _schedule("timer", label, f"Time's up: {label}", seconds=minutes * 60)
    return f"Timer '{label}' set; it goes off at {_when_text(due)}."


@tool(writes=True)
def set_reminder(time: str, message: str, day: str = "today") -> str:
    """Set a reminder for a time of day; Tony gets an alert with the message.

    Args:
        time: the time of day, e.g. "15:30" or "3:30 pm"
        message: what to remind him about
        day: today, tomorrow, a weekday name, or YYYY-MM-DD (up to 7 days ahead)
    """
    hour, minute = _parse_time(time)
    at = datetime.combine(_parse_day(day), datetime.min.time()).replace(hour=hour, minute=minute)
    if at <= datetime.now():
        raise ValueError(f"{_clock(at)} {day} has already passed. Choose a later time or day.")
    message = _safe_label(message, "Reminder")
    due = _schedule("reminder", message, message, at=at)
    return f"Reminder set for {_when_text(due)}: {message}."


@tool
def list_timers() -> str:
    """List the timers and reminders that are still waiting, numbered soonest first."""
    items = _active(_load())
    if not items:
        return "No timers or reminders are set."
    return " ".join(f"{i}. {t['kind'].capitalize()} '{t['label']}': {_when_text(t['due'])}."
                    for i, t in enumerate(items, 1))


@tool(writes=True)
def cancel_timer(number: int) -> str:
    """Cancel a timer or reminder by its number from list_timers.

    Args:
        number: the number shown by list_timers
    """
    items = _active(_load())
    try:
        number = int(number)
    except (TypeError, ValueError):
        raise ValueError(f"'{number}' isn't a timer number.")
    if not 1 <= number <= len(items):
        raise ValueError(f"There is no timer number {number}. Call list_timers to see them.")
    item = items.pop(number - 1)
    subprocess.run(["systemctl", "--user", "stop", f"{item['unit']}.timer"],
                   capture_output=True, text=True, timeout=10)
    _save(items)
    return f"Cancelled {item['kind']} '{item['label']}'."


def briefing():
    """Waiting timers and reminders, if any (nothing when there are none)."""
    items = _active(_load())
    if not items:
        return None
    first = items[0]
    more = f", plus {len(items) - 1} more" if len(items) > 1 else ""
    return f"Timers: next is {first['kind']} '{first['label']}', {_when_text(first['due'])}{more}."
