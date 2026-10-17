# jarvis-install: tools
"""
tools/personality.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
F-27 Personality: change J.A.R.V.I.S.'s humor by voice ("humor off", "less jokes",
"be a bit funnier"). The change lasts until J.A.R.V.I.S. restarts or reloads; to
keep it, set humor under [personality] in config.toml.
The persona itself (manner, address, humor rules) is in jarvis.py's core prompt.
"""
import tools
from tools import tool

PROMPT = ("set_humor changes how much dry humor J.A.R.V.I.S. uses: off, slight or more "
          "(or less / more relative to now). Confirm the new level in one short sentence.")
EXAMPLES = ["humor off", "less jokes please", "be a bit funnier", "turn your humor up",
            "no more jokes", "back to normal humor", "how funny are you set to"]

ALIASES = {"none": "off", "no": "off", "zero": "off", "normal": "slight", "default": "slight",
           "some": "slight", "a little": "slight", "light": "slight", "lots": "more", "high": "more", "max": "more"}


@tool
def set_humor(level: str) -> str:
    """Set J.A.R.V.I.S.'s humor until restart: off, slight or more (or less / more than now), or check it with "current".

    Args:
        level: off, slight, more, less, or current
    """
    levels = tools.HUMOR_LEVELS
    now = tools.PERSONA["humor"]
    wanted = " ".join(str(level or "").lower().split())
    if wanted in ("current", "now", "what", ""):
        return f"Humor is set to {now}."
    if wanted in ("less", "lower", "down"):
        wanted = levels[max(0, levels.index(now) - 1)]
    elif wanted in ("more", "higher", "up") and now == "more":
        return "Humor is already at its highest setting, more."
    elif wanted in ("higher", "up"):
        wanted = levels[min(len(levels) - 1, levels.index(now) + 1)]
    wanted = ALIASES.get(wanted, wanted)
    if wanted not in levels:
        raise ValueError(f"humor can be {', '.join(levels)}")
    tools.PERSONA["humor"] = wanted
    return (f"Humor set to {wanted} until restart. To keep it, set humor = \"{wanted}\" "
            "under [personality] in config.toml.")
