# jarvis-install: tools
"""
tools/power.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
F-04 Power and session: lock, suspend, log out, reboot, shut down, reboot into Windows.

Logging out, rebooting and shutting down close every app, which the "no close
tools" rule forbids. Tony approved an exception on 2026-10-16 with these safeguards:
  * the tool counts as a write, so it's blocked right after outside text;
  * two steps: the first request only lists the open apps and asks; it happens
    only after a "yes" in a later request;
  * then a countdown (10 s) with a notification every second; the talk key
    (or Ctrl+C in typed mode) cancels it.
Lock happens at once. Suspend has a short countdown so the reply can finish.
Reboot into Windows is staged in a terminal (F-33): you press Enter and type
your sudo password; it sets Windows for the next boot only (efibootmgr --bootnext)
and doesn't touch GRUB or the boot order.
"""
import re
import subprocess
import time

import tools
from tools import tool

PROMPT = """power locks, suspends, logs out, reboots or shuts down the PC, or reboots into Windows.
- Lock and suspend happen right away.
- Log out, reboot and shut down close every app: the first call only lists open apps. Tell Tony what's open and ask him to confirm. Only if he then says yes, call power again with the same action; a countdown starts that the talk key cancels.
- Windows: a terminal opens with the command typed in; he presses Enter and types his password."""

EXAMPLES = ["lock the screen", "lock my computer", "put the computer to sleep", "suspend",
            "log me out", "restart the computer", "reboot", "shut down the PC", "turn off the computer",
            "reboot into Windows", "game protocol, boot Windows"]

ACTIONS = ("lock", "suspend", "logout", "reboot", "shutdown", "windows")
CLOSES_APPS = ("logout", "reboot", "shutdown")
ALIASES = {"sleep": "suspend", "log out": "logout", "log_out": "logout", "sign out": "logout",
           "restart": "reboot", "shut down": "shutdown", "shut_down": "shutdown", "poweroff": "shutdown",
           "power off": "shutdown", "turn off": "shutdown", "windows 11": "windows", "boot windows": "windows",
           "reboot to windows": "windows", "reboot into windows": "windows", "lock screen": "lock"}
LABELS = {"suspend": "Suspending", "logout": "Logging out", "reboot": "Rebooting", "shutdown": "Shutting down"}
CANCELLED = {"suspend": "Suspend", "logout": "Log out", "reboot": "Reboot", "shutdown": "Shutdown"}

SETTINGS = {
    "countdown": 10,
    "suspend_countdown": 3,
    "lock": ["loginctl", "lock-session"],
    "suspend": ["systemctl", "suspend"],
    "logout": ["i3-msg", "exit"],
    "reboot": ["systemctl", "reboot"],
    "shutdown": ["systemctl", "poweroff"],
    "windows_entry": "Windows Boot Manager",
}
_pending = None   # {"action", "turn"} after the first step of a closing action


def setup(settings):
    global _pending
    _pending = None
    for key in ("lock", "suspend", "logout", "reboot", "shutdown"):
        if key in settings:
            value = settings[key]
            if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
                raise ValueError(f"[tools.power] {key} must be a command as a list, e.g. [\"systemctl\", \"{key}\"]")
            SETTINGS[key] = value
    SETTINGS["countdown"] = max(3, int(settings.get("countdown", 10)))
    SETTINGS["suspend_countdown"] = max(0, int(settings.get("suspend_countdown", 3)))
    SETTINGS["windows_entry"] = str(settings.get("windows_entry", "Windows Boot Manager"))


def _notify(title, body, seconds=1.5, urgency="critical"):
    try:
        subprocess.run(["notify-send", "-a", "J.A.R.V.I.S.", "-u", urgency, "-t", str(int(seconds * 1000)),
                        "-h", "string:x-dunst-stack-tag:jarvis-power", title, body], timeout=2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _open_apps():
    """The desktop plugin's list of open apps (never window titles), or a note why not."""
    try:
        from tools.desktop import whats_open
        return whats_open()
    except Exception as e:  # noqa: BLE001 (i3 unreachable etc.: still allow the action)
        return f"(couldn't check open apps: {e})"


def _countdown(action, seconds):
    """Count down with a notification each second. Returns False if Tony cancelled."""
    label = LABELS[action]
    sir = tools.PERSONA.get("address", "sir")
    if seconds >= 5:
        tools.announce(f"{label} in {seconds} seconds, {sir}. Press the talk key to cancel.")
    for remaining in range(seconds, 0, -1):
        _notify(f"{label} in {remaining} s", "Press the talk key to cancel." if seconds >= 5 else "")
        for _ in range(10):
            if tools.cancel_requested():
                _notify(f"{CANCELLED[action]} cancelled", "Nothing was done.", seconds=4, urgency="normal")
                return False
            time.sleep(0.1)
    return True


def _run(action):
    """Run the configured command for an action."""
    # Log first: after a reboot there's no chance to log anything
    tools.log_action("power", {"action": action}, "ok", f"running {' '.join(SETTINGS[action])}",
                     {"group": "power", "writes": True})
    try:
        done = subprocess.run(SETTINGS[action], capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return  # logout/reboot may never return; that's success
    if done.returncode != 0:
        error = " ".join((done.stderr or done.stdout).split())[:200]
        raise ValueError(f"{action} failed: {error or 'exit code ' + str(done.returncode)}")


def _lock():
    if SETTINGS["lock"] == ["loginctl", "lock-session"]:
        running = subprocess.run(["pgrep", "-x", "xss-lock"], capture_output=True).returncode == 0
        if not running:
            raise ValueError("locking needs a screen locker running: install i3lock and enable the xss-lock "
                             "line in the i3 config (see [tools.power] in config.toml)")
    _run("lock")
    return "Screen locked."


def _windows_entry():
    """Find the firmware boot entry for Windows in `efibootmgr` (reading needs no root)."""
    try:
        listing = subprocess.run(["efibootmgr"], capture_output=True, text=True, timeout=5).stdout
    except FileNotFoundError:
        raise ValueError("efibootmgr isn't installed")
    for line in listing.splitlines():
        match = re.match(r"Boot([0-9A-Fa-f]{4})\*?\s+(.*)", line)
        if match and match.group(2).startswith(SETTINGS["windows_entry"]):
            return match.group(1).upper()
    raise ValueError(f"no boot entry named '{SETTINGS['windows_entry']}' was found in efibootmgr")


def _windows():
    entry = _windows_entry()
    from tools.staged import stage

    apps = _open_apps()
    stage(f"sudo efibootmgr --bootnext {entry} && systemctl reboot",
          f"Boots Windows next (this boot only; GRUB and the boot order stay as they are), then reboots now. Open: {apps}")
    return ("A terminal is open with the Windows reboot command typed in. Tony presses Enter and types his "
            "password; nothing happens until then.")


@tool(writes=True, schema={"type": "function", "function": {
    "name": "power",
    "description": "Lock, suspend, log out, reboot or shut down the PC, or reboot into Windows. "
                   "Log out, reboot and shut down need Tony's yes in a later request.",
    "parameters": {"type": "object", "required": ["action"], "properties": {
        "action": {"type": "string", "enum": list(ACTIONS),
                   "description": "lock, suspend, logout, reboot, shutdown or windows"}}}}})
def power(action: str) -> str:
    global _pending
    action = " ".join(str(action or "").lower().replace("-", " ").split())
    action = ALIASES.get(action, action).replace(" ", "")
    if action not in ACTIONS:
        raise ValueError(f"action must be one of: {', '.join(ACTIONS)}")

    if action == "lock":
        return _lock()
    if action == "windows":
        return _windows()
    if action == "suspend":
        if not _countdown("suspend", SETTINGS["suspend_countdown"]):
            return "Cancelled; the computer stays on."
        _run("suspend")
        return "Suspending."

    # Closing actions: two steps
    confirmed = (_pending is not None and _pending["action"] == action
                 and 1 <= tools.current_turn() - _pending["turn"] <= 2 and tools.is_yes())
    if not confirmed:
        _pending = {"action": action, "turn": tools.current_turn()}
        return (f"Not yet: {LABELS[action].lower()} closes every app. Open now: {_open_apps()} "
                f"Tell Tony what's open and ask him to confirm. Only if he says yes, call power({action}) again.")
    _pending = None
    if not _countdown(action, SETTINGS["countdown"]):
        return "Cancelled; nothing was closed."
    _run(action)
    return f"{LABELS[action]} now."
