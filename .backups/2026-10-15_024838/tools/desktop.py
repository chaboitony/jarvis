# jarvis-install: tools
"""
tools/desktop.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
The desktop: see what's open and playing, and control it.
  Look:     whats_open, now_playing                      (read-only)
  Control:  media_control (music + volume), open_app, switch_workspace

Safety:
  * Apps can only be opened from the approved list in config.toml
    ([tools.desktop.apps]). The model picks a name from that list; it never
    supplies a command.
  * Apps are started by i3, not by J.A.R.V.I.S. The background service's own
    processes are stopped when it restarts, so an app it started directly would
    close too.
  * There is no tool to close apps or windows, so unsaved work can't be lost to
    a misheard command. Volume never goes above 100%.

Privacy:
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
import time

from tools import tool

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

# Used when config.toml has no [tools.desktop.apps] section: spoken name -> command
DEFAULT_APPS = {
    "firefox": "firefox", "terminal": "konsole", "files": "dolphin",
    "obsidian": "obsidian", "spotify": "spotify-launcher", "discord": "discord",
}
# Everyday words for apps -> the approved name they mean (used only if that app is approved)
ALIASES = {
    "browser": "firefox", "web browser": "firefox", "internet": "firefox",
    "google chrome": "chrome", "konsole": "terminal", "console": "terminal", "shell": "terminal",
    "dolphin": "files", "file manager": "files", "file explorer": "files",
    "notes": "obsidian", "vault": "obsidian", "music": "spotify",
    "text editor": "geany", "editor": "geany", "office": "libreoffice",
    "volume settings": "sound settings", "pavucontrol": "sound settings",
}
MEDIA_ACTIONS = ["play", "pause", "next", "previous", "volume_up", "volume_down",
                 "set_volume", "mute", "unmute"]

APPS = dict(DEFAULT_APPS)
VOLUME_STEP = 10
MAX_WORKSPACE = 10
PROMPT = ""


def setup(settings):
    """Read [tools.desktop] from config.toml and write the instructions the model sees."""
    global APPS, VOLUME_STEP, PROMPT
    apps = settings.get("apps", DEFAULT_APPS)
    if not isinstance(apps, dict) or not apps:
        raise ValueError("[tools.desktop.apps] must list at least one app, like firefox = \"firefox\"")
    checked = {}
    for name, command in apps.items():
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"app '{name}' needs a command, like {name} = \"{name}\"")
        if any(ch in command for ch in ";\n,"):  # i3 treats these as command separators
            raise ValueError(f"the command for '{name}' can't contain ; , or a line break")
        checked[" ".join(str(name).lower().split())] = command.strip()
    APPS = checked
    VOLUME_STEP = max(1, min(25, int(settings.get("volume_step", 10))))
    PROMPT = f"""Desktop: use whats_open for questions about open apps or workspaces, now_playing for questions about music or media,
media_control to play, pause or skip music and to change or mute the volume, open_app to open an app, and switch_workspace to change workspaces.
Apps you can open: {", ".join(APPS)}. If Tony asks for an app that isn't on this list, say it isn't approved yet.
You cannot close apps or windows; if asked, say so."""


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


def _i3(*args):
    """
    Run i3-msg and return its JSON answer. i3-msg finds i3 through the display,
    which a background service may not have, so fall back to i3's socket file.
    """
    run_dir = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    sockets = sorted(glob.glob(f"{run_dir}/i3/ipc-socket.*"), key=os.path.getmtime, reverse=True)
    for prefix in [[]] + [["-s", sock] for sock in sockets]:
        try:
            result = subprocess.run(["i3-msg", *prefix, *args], capture_output=True, text=True, timeout=5)
        except FileNotFoundError:
            raise ValueError("i3-msg isn't installed, so I can't reach the desktop.")
        except subprocess.TimeoutExpired:
            continue
        out = result.stdout.lstrip()
        if out.startswith(("{", "[")):
            return json.loads(out)
    raise ValueError("I can't reach the i3 window manager.")


def _i3_command(command):
    """Run an i3 command (like 'workspace number 3') and raise if i3 refuses it."""
    replies = _i3(command)
    for reply in replies if isinstance(replies, list) else [replies]:
        if not reply.get("success", False):
            raise ValueError(f"i3 refused that: {_clean(reply.get('error', 'unknown error'), 100)}")


def _workspaces():
    """[(workspace name, monitor, {app: window count}, None)] in i3's order, plus the focused app."""
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

    tree = _i3("-t", "get_tree")
    if not isinstance(tree, dict):
        raise ValueError("i3 sent an unexpected answer.")
    walk(tree, None, None)
    return [ws for ws in found if ws[2]], focused["app"]


def _players():
    """[(player, status, artist, title, instance)] for every media player playerctl can see."""
    fmt = "{{playerName}}\t{{status}}\t{{artist}}\t{{title}}\t{{playerInstance}}"
    try:
        result = subprocess.run(["playerctl", "-a", "metadata", "--format", fmt],
                                capture_output=True, text=True, timeout=5)
    except FileNotFoundError:
        raise ValueError("playerctl isn't installed, so I can't reach the media players.")
    except subprocess.TimeoutExpired:
        return []
    if result.returncode != 0:  # "No players found"
        return []
    players = []
    for line in result.stdout.splitlines():
        parts = (line.split("\t") + [""] * 5)[:5]
        players.append(tuple(_clean(p) for p in parts[:4]) + (parts[4].strip(),))
    return players


def _is_browser(player):
    return any(player.lower().startswith(b) for b in BROWSERS)


def _describe_player(player, status, artist, title, *_):
    name = PLAYER_NAMES.get(player.lower(), player.title() or "A player")
    if _is_browser(player):
        return f"{name} {'is playing media' if status == 'Playing' else 'has paused media'}."
    if status == "Stopped":
        return f"{name} is stopped."
    state = "is playing" if status == "Playing" else "is paused on"
    if not title:
        return f"{name} {state.replace(' on', '')} something."
    song = f'"{title}"' + (f" by {artist}" if artist else "")
    return f"{name} {state} {song}."


def _list(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _apps_text(apps):
    return _list(f"{app} ({count} windows)" if count > 1 else app for app, count in apps.items())


def _pactl(*args):
    try:
        result = subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=5)
    except FileNotFoundError:
        raise ValueError("pactl isn't installed, so I can't change the volume.")
    except subprocess.TimeoutExpired:
        raise ValueError("The sound system took too long to respond.")
    if result.returncode != 0:
        raise ValueError("The sound system refused that request.")
    return result.stdout


def _volume():
    """(volume percent, muted?) of the default output."""
    match = re.search(r"(\d+)%", _pactl("get-sink-volume", "@DEFAULT_SINK@"))
    if not match:
        raise ValueError("I couldn't read the current volume.")
    muted = "yes" in _pactl("get-sink-mute", "@DEFAULT_SINK@").lower()
    return int(match.group(1)), muted


def _volume_text():
    level, muted = _volume()
    return f"Volume is {level}%" + (" but muted." if muted else ".")


# --------------------------------------------------------------------------
# Tools: looking
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
    players.sort(key=lambda p: {"Playing": 0, "Paused": 1}.get(p[1], 2))  # playing first
    return " ".join(_describe_player(*p) for p in players)


# --------------------------------------------------------------------------
# Tools: controlling
# --------------------------------------------------------------------------
@tool(writes=True)
def media_control(action: str, level: int | None = None) -> str:
    """Control music and sound: play, pause or skip tracks, and change or mute the volume.

    Args:
        action: play, pause, next, previous, volume_up, volume_down, set_volume, mute or unmute
        level: only for set_volume: the new volume from 0 to 100
    """
    action = str(action or "").strip().lower().replace(" ", "_").replace("-", "_")
    action = {"skip": "next", "next_track": "next", "back": "previous", "previous_track": "previous",
              "resume": "play", "stop": "pause", "louder": "volume_up", "quieter": "volume_down",
              "turn_up": "volume_up", "turn_down": "volume_down", "volume": "set_volume"}.get(action, action)
    if action not in MEDIA_ACTIONS:
        raise ValueError(f"Unknown action '{action}'. Use one of: {', '.join(MEDIA_ACTIONS)}.")

    # ---- volume ----
    if action in ("volume_up", "volume_down", "set_volume"):
        if action == "set_volume":
            try:
                target = int(level)
            except (TypeError, ValueError):
                raise ValueError("set_volume needs a level from 0 to 100.")
            if not 0 <= target <= 100:
                raise ValueError("The volume level must be from 0 to 100.")
        else:
            current, _ = _volume()
            step = VOLUME_STEP if action == "volume_up" else -VOLUME_STEP
            target = max(0, min(100, current + step))  # never above 100%
        _pactl("set-sink-volume", "@DEFAULT_SINK@", f"{target}%")
        return _volume_text()
    if action in ("mute", "unmute"):
        _pactl("set-sink-mute", "@DEFAULT_SINK@", "1" if action == "mute" else "0")
        return "Sound muted." if action == "mute" else "Sound unmuted. " + _volume_text()

    # ---- music ----
    players = _players()
    if not players:
        return "No music or media player is running. Spotify can be opened with open_app."
    playing = [p for p in players if p[1] == "Playing"]
    if action == "play":
        paused = [p for p in players if p[1] == "Paused"]
        # prefer a paused music player over a paused browser tab
        choices = sorted(paused, key=lambda p: _is_browser(p[0])) or players
        target = choices[0]
    else:
        target = playing[0] if playing else sorted(players, key=lambda p: _is_browser(p[0]))[0]
    instance = target[4] or target[0]
    try:
        subprocess.run(["playerctl", "-p", instance, action], capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        raise ValueError("The media player took too long to respond.")
    time.sleep(0.4)  # give the player a moment to switch tracks
    after = next((p for p in _players() if (p[4] or p[0]) == instance), None)
    done = {"play": "Playing.", "pause": "Paused.", "next": "Skipped.", "previous": "Went back."}[action]
    return f"{done} {_describe_player(*after)}" if after else done


@tool(writes=True)
def open_app(app: str, workspace: int | None = None) -> str:
    """Open an app from Tony's approved list, optionally on a chosen workspace.

    Args:
        app: the app's name from the approved list in your instructions
        workspace: optional workspace number to open it on (1 to 10)
    """
    name = " ".join(str(app or "").lower().split())
    name = name if name in APPS else ALIASES.get(name, name)
    if name not in APPS:
        raise ValueError(f"'{_clean(app, 40)}' isn't on the approved app list. Approved: {', '.join(APPS)}. "
                         "Tony can add apps under [tools.desktop.apps] in config.toml.")
    command = f"exec --no-startup-id {APPS[name]}"
    if workspace is not None:
        command = f"workspace number {_check_workspace(workspace)}; {command}"
    _i3_command(command)
    where = f" on workspace {int(workspace)}" if workspace is not None else ""
    return f"Opening {name}{where}."


def _check_workspace(number):
    try:
        number = int(number)
    except (TypeError, ValueError):
        raise ValueError(f"'{number}' isn't a workspace number.")
    if not 1 <= number <= MAX_WORKSPACE:
        raise ValueError(f"Workspaces are numbered 1 to {MAX_WORKSPACE}.")
    return number


@tool(writes=True)
def switch_workspace(number: int) -> str:
    """Switch the screen to another i3 workspace.

    Args:
        number: the workspace number, 1 to 10
    """
    number = _check_workspace(number)
    _i3_command(f"workspace number {number}")
    return f"Switched to workspace {number}."


# --------------------------------------------------------------------------
# Welcome briefing section
# --------------------------------------------------------------------------
def briefing():
    """One or two sentences for the welcome briefing."""
    parts = []
    try:
        workspaces, _ = _workspaces()
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
