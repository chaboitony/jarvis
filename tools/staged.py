# jarvis-install: tools
"""
tools/staged.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
F-33 Staged commands: J.A.R.V.I.S. never runs shell commands himself. For
anything that needs a terminal (sudo, updates, fixes), he opens Konsole with the
command typed in but not run, plus a one-line explanation. Tony reads it, edits
it if he likes, and presses Enter, or Ctrl+C to cancel. (bin/jarvis-stage)

Safety:
  * Marked as a write, so it's blocked right after outside text (a web page
    can't put a command in front of Tony).
  * One line only, at most 500 characters, no control characters.
  * Commands that delete or overwrite data are refused outright (rm, dd, mkfs,
    find -delete, git reset --hard, package removal, curl | sh, ...). If Tony
    wants one of those, he types it himself.
  * The command travels in a private file (mode 600) that jarvis-stage deletes
    as soon as it's read; nothing goes through a shell on the way.

Other plugins can stage their own fixed commands:  from tools.staged import stage
"""
import os
import re
import shlex
import time
import uuid
from pathlib import Path

from tools import tool

PROMPT = """stage_command opens a terminal window with a shell command typed in but NOT run; Tony presses Enter to run it himself.
- Use it when he asks you to run, install, update, restart or fix something that needs a terminal, especially with sudo (for "update my system": sudo pacman -Syu).
- Give a one-sentence explanation of what the command does.
- Afterwards say it's ready in the terminal and waiting for him; never claim it ran.
- It refuses commands that delete or overwrite data; tell him to type those himself."""

EXAMPLES = ["update my system", "put the command to restart pipewire in a terminal",
            "stage the command to install htop", "open a terminal with the command to check disk usage",
            "run sudo pacman -Syu", "give me the command to see failed services"]

RUN_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "jarvis-stage"
SCRIPT = Path.home() / ".local" / "bin" / "jarvis-stage"
SETTINGS = {"terminal": ["konsole", "--hide-menubar", "-e"]}
MAX_CHARS = 500

# (pattern, why) -- checked on the whole command line
REFUSED = [
    (r"(^|[;&|(`]|\$\(|\s)(sudo\s+(-\S+\s+)*)?(rm|rmdir|shred|unlink|srm|wipefs|mkfs(\.\w+)?|mkswap|fdisk|sfdisk|cfdisk|gdisk|sgdisk|parted|dd|truncate|blkdiscard)(\s|$)",
     "deletes or overwrites data"),
    (r"\bfind\b.*\s-(delete|exec\s+rm)\b", "deletes files"),
    (r"\b(curl|wget|fetch)\b[^|]*\|\s*(sudo\s+)?(ba|z|da|fi|k)?sh\b", "pipes a download straight into a shell"),
    (r":\s*\(\s*\)\s*\{", "is a fork bomb"),
    (r">\s*/dev/(sd|nvme|hd|vd|mmcblk|disk)", "writes straight onto a disk"),
    (r"(^|[^>])>\s*(/etc/|/boot/|/usr/|~/\.|\$HOME/\.)", "overwrites a system or config file"),
    (r"\b(chmod|chown|chgrp)\s+(\S+\s+)*-\w*R", "changes permissions recursively"),
    (r"\bgit\s+(reset\s+--hard|clean\s+-\w*f|push\s+(\S+\s+)*(--force\b|-f\b)|checkout\s+--\s|restore\s)", "throws away git work"),
    (r"\b(pacman|yay|paru)\s+(\S+\s+)*-R\w*", "removes packages"),
    (r"\bmv\s+(\S+\s+)*\S+\s+/dev/null\b", "throws files away"),
]


def setup(settings):
    terminal = settings.get("terminal", SETTINGS["terminal"])
    if isinstance(terminal, str):
        terminal = shlex.split(terminal)
    if not terminal:
        raise ValueError("[tools.staged] terminal can't be empty")
    SETTINGS["terminal"] = [str(t) for t in terminal]


def check(command, explanation=""):
    """Validate a command before staging it. Returns (command, explanation) cleaned up."""
    command = str(command or "").strip()
    explanation = " ".join(str(explanation or "").split())[:200]
    if not command:
        raise ValueError("there's no command to stage")
    if "\n" in command or "\r" in command:
        raise ValueError("only a single command line can be staged")
    if any(ord(c) < 32 and c != "\t" for c in command):
        raise ValueError("the command contains control characters")
    if len(command) > MAX_CHARS:
        raise ValueError(f"the command is longer than {MAX_CHARS} characters")
    for pattern, why in REFUSED:
        if re.search(pattern, command):  # shell commands are case-sensitive, so the checks are too
            raise ValueError(f"not staged: this command {why}. J.A.R.V.I.S. doesn't prepare commands "
                             "that delete or overwrite things; if Tony really wants it, he types it himself")
    return command, explanation


def _clear_old():
    """Remove staged files nobody opened within an hour."""
    cutoff = time.time() - 3600
    for f in RUN_DIR.glob("*.cmd"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def stage(command, explanation=""):
    """Open a terminal with this command typed in (not run). Returns the path of the hand-over file."""
    command, explanation = check(command, explanation)
    if not SCRIPT.exists():
        raise ValueError(f"{SCRIPT} is missing; run jarvis-update to install it")
    RUN_DIR.mkdir(mode=0o700, exist_ok=True)
    os.chmod(RUN_DIR, 0o700)
    _clear_old()
    path = RUN_DIR / f"{uuid.uuid4().hex}.cmd"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"{explanation}\n{command}\n")
    from tools.desktop import _i3_command  # launch through i3, so restarting J.A.R.V.I.S. doesn't close it

    parts = [*SETTINGS["terminal"], str(SCRIPT), str(path)]
    try:
        _i3_command("exec --no-startup-id " + " ".join(shlex.quote(p) for p in parts))
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


@tool(writes=True)
def stage_command(command: str, explanation: str) -> str:
    """Open a terminal with a shell command typed in but NOT run; Tony reads it and presses Enter himself.

    Args:
        command: one shell command line, e.g. sudo pacman -Syu
        explanation: one short sentence saying what the command does
    """
    stage(command, explanation)
    return "The command is typed into a terminal window and waiting; nothing has run. Tony presses Enter to run it or Ctrl+C to cancel."
