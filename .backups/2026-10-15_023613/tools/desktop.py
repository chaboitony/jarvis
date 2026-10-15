# jarvis-install: tools
"""
tools/desktop.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
What's on the desktop: open apps per i3 workspace, and what's playing.
Read-only: it looks, it never changes anything.

Privacy and safety:
  * Only app names are reported, never window titles. Titles can contain
    private text (email subjects, documents) or text a web page chose.
  * Song titles come from music players such as Spotify. Browsers also report
    media, but their "titles" are web page text, so for browsers only the fact
    that something is playing is reported.
"""
import glob
import json
import os
import re
import subprocess

from tools import tool

PROMPT = """Desktop: use whats_open for questions about open apps, windows or workspaces, and now_playing for questions about music or media."""

# Window class -> the name people use. Unknown classes are tidied up automatically.
APP_NAMES = {
    "firefox": "Firefox", "google-chrome": "Chrome", "chromium": "Chromium",
    "konsole": "Konsole", "xterm": "XTerm", "dolphin": "Dolphin", "filelight": "Filelight",
    "discord": "Discord", "spotify": "Spotify", "obsidian": "Obsidian", "geany": "Geany",
    "pavucontrol": "Volume Control", "arandr": "ARandR", "zenity": "J.A.R.V.I.S. history",
    "spectacle": "Spectacle", "nitrogen": "Nitrogen", "system-config-printer": "Printer Settings",
    "libreoffice-writer": "LibreOffice Writer", "libreoffice-calc": "LibreOffice Calc",
    "libreoffice-impress": "LibreOffice Impress", "libreoffice-startcenter": "LibreOffice",
    "soffice": "LibreOffice", "org.gnome.nautilus": "Files", "steam": "Steam",
}
BROWSERS = ("firefox", "chrome", "chromium", "brave", "vivaldi", "opera", "plasma-browser-integration")
PLAYER_NAMES = {"spotify": "Spotify", "firefox": "Firefox", "chromium": "Chromium", "chrome": "Chrome"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _clean(text, limit=60):
    """Strip control characters and cut long text, so outside text stays short and plain."""
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", str(text or ""))  # terminal color codes
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _app_name(window_class):
    raw = _clean(window_class, 40)
    if not raw:
        return None
    known = APP_NAMES.get(raw.lower())
    if known:
        return known
    return raw if any(c.isupper() for c in raw) else raw.replace("-", " ").title()


def _i3_tree():
    """
    Ask i3 for its window tree. i3-msg finds i3 through the display, which a
    background service may not have, so fall back to i3's socket in the runtime folder.
    """
    commands = [["i3-msg", "-t", "get_tree"]]
    run_dir = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    sockets = sorted(glob.glob(f"{run_dir}/i3/ipc-socket.*"), key=os.path.getmtime, reverse=True)
    commands += [["i3-msg", "-s", sock, "-t", "get_tree"] for sock in sockets]
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        except FileNotFoundError:
            raise ValueError("i3-msg isn't installed, so I can't see the desktop.")
        except subprocess.TimeoutExpired:
            continue
        if result.returncode == 0 and result.stdout.lstrip().startswith("{"):
            return json.loads(result.stdout)
    raise ValueError("I can't reach the i3 window manager, so I can't see what's open.")


def _workspaces():
    """[(workspace name, monitor, {app: window count}, focused app or None)] in i3's order."""
    found, focused = [], {"app": None}

    def walk(node, workspace, monitor):
        kind = node.get("type")
        if kind == "output":
            monitor = node.get("name")
        elif kind == "workspace":
            workspace = node.get("name")
            if workspace and not workspace.startswith("__"):  # skip i3's hidden scratchpad
                found.append((workspace, monitor, {}, None))
        if node.get("window") and workspace and not workspace.startswith("__"):
            props = node.get("window_properties") or {}
            app = _app_name(props.get("class") or props.get("instance"))
            if app:
                apps = next(w[2] for w in found if w[0] == workspace)
                apps[app] = apps.get(app, 0) + 1
                if node.get("focused"):
                    focused["app"] = app
        for child in (node.get("nodes") or []) + (node.get("floating_nodes") or []):
            walk(child, workspace, monitor)

    walk(_i3_tree(), None, None)
    return [ws for ws in found if ws[2]], focused["app"]


def _players():
    """[(player, status, artist, title)] for every media player playerctl can see."""
    fmt = "{{playerName}}\t{{status}}\t{{artist}}\t{{title}}"
    try:
        result = subprocess.run(["playerctl", "-a", "metadata", "--format", fmt],
                                capture_output=True, text=True, timeout=5)
    except FileNotFoundError:
        raise ValueError("playerctl isn't installed, so I can't see what's playing.")
    except subprocess.TimeoutExpired:
        return []
    if result.returncode != 0:  # "No players found"
        return []
    players = []
    for line in result.stdout.splitlines():
        parts = (line.split("\t") + ["", "", "", ""])[:4]
        players.append(tuple(_clean(p) for p in parts))
    return players


def _describe_player(player, status, artist, title):
    name = PLAYER_NAMES.get(player.lower(), player.title() or "A player")
    state = {"Playing": "is playing", "Paused": "is paused on", "Stopped": "is stopped"}.get(status, "has")
    if any(player.lower().startswith(b) for b in BROWSERS):
        return f"{name} {'is playing media' if status == 'Playing' else 'has paused media'}."
    if not title:
        return f"{name} {state.replace(' on', '')} something."
    song = f'"{title}"' + (f" by {artist}" if artist else "")
    return f"{name} {state} {song}." if status != "Stopped" else f"{name} is stopped."


def _list(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _apps_text(apps):
    return _list(f"{app} ({count} windows)" if count > 1 else app for app, count in apps.items())


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
@tool
def whats_open() -> str:
    """List the apps open on each i3 workspace and which one is focused."""
    workspaces, focused = _workspaces()
    if not workspaces:
        return "No windows are open."
    lines = [f"Workspace {name} ({monitor}): {_apps_text(apps)}." for name, monitor, apps, _ in workspaces]
    if focused:
        lines.append(f"Focused: {focused}.")
    return " ".join(lines)


@tool
def now_playing() -> str:
    """Say what music or media is playing or paused, and in which app."""
    players = _players()
    if not players:
        return "Nothing is playing."
    # Playing first, then paused
    players.sort(key=lambda p: {"Playing": 0, "Paused": 1}.get(p[1], 2))
    return " ".join(_describe_player(*p) for p in players)


def briefing():
    """One or two sentences for the welcome briefing."""
    parts = []
    try:
        workspaces, focused = _workspaces()
        if workspaces:
            parts.append("Open: " + "; ".join(f"{_apps_text(apps)} on workspace {name}"
                                              for name, _, apps, _ in workspaces) + ".")
        else:
            parts.append("No windows are open.")
    except ValueError as e:
        parts.append(f"Desktop: {e}")
    try:
        playing = [p for p in _players() if p[1] in ("Playing", "Paused")]
        if playing:
            playing.sort(key=lambda p: p[1] != "Playing")
            parts.append("Media: " + _describe_player(*playing[0]))
    except ValueError:
        pass
    return " ".join(parts)
