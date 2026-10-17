# jarvis-install: tools
"""
tools/ask_claude.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
Hands a question to Claude (Anthropic's API) when the local model isn't enough:
current information from the web, deep technical answers, real code. Claude can
search the web and run code in Anthropic's sandbox (never on this PC).

Rules (agreed with Tony, 2026-10-16):
  * Sent only when Tony says "Claude" in the request, or says yes after
    J.A.R.V.I.S. offers ("Shall I ask Claude?"). Otherwise nothing leaves the PC.
  * Only the question goes out (plus the short about_me line in config.toml),
    never the conversation, tasks, notes or history.
  * Counts as a write: it sends data off the machine, so it's blocked after
    untrusted text in the same request (a web page can't make it send things).
  * A local monthly budget stops it before Anthropic's own spend limit does.
  * The key lives in ~/.config/rima/anthropic-key (mode 600), never in the repo.
  * Claude's answer is outside text (untrusted). The full answer is saved as a
    Markdown file; J.A.R.V.I.S. speaks a two-sentence summary.

Check the setup:  cd ~/rima/jarvis && uv run python -m tools.ask_claude test
"""
import json
import os
import re
import shlex
import stat
import threading
import time
import tomllib
from datetime import datetime
from pathlib import Path

import tools
from tools import tool

PROMPT = """ask_claude sends a question to Claude, a much more capable AI on the internet. It costs money and takes 5 to 60 seconds.
- Use it when Tony asks you to ask Claude, or when he says yes after you offered.
- If he needs current information (news, prices, recent versions), detailed technical answers or real code, and you aren't sure, offer in one sentence: "Shall I ask Claude?"
- Put only his question in `question`, with the details needed to answer it. Don't add his tasks, notes or other personal details unless he asks.
- When the answer comes back, tell him its summary in your own short words. show_claude_answer opens the full answer on screen."""

EXAMPLES = [
    "ask Claude how a buck converter works",
    "have Claude explain the Fourier transform",
    "ask Claude to write a Python script that renames my files",
    "check with Claude what the latest Linux kernel is",
    "look this up online",
    "ask Claude to think hard about this beam deflection problem",
    "show me Claude's full answer",
    "how much have I spent on Claude this month",
]

# Dollars per million tokens (input, output), from Anthropic's pricing page, October 2026.
# [tools.ask_claude.prices] in config.toml overrides these.
DEFAULT_PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
SEARCH_PRICE = 10.0 / 1000      # dollars per web search
FALLBACK_PRICE = (4.00, 20.00)  # unknown model: assume an expensive one, so the budget errs safe

DEEP_WORDS = re.compile(
    r"\b(think (hard|carefully|it through)|in depth|in detail|detailed|deep dive|thorough(ly)?|"
    r"step by step|write (me )?(a |the |some )?(code|script|program|function|class))\b", re.I)
YES = re.compile(r"^(yes|yeah|yep|yup|sure|ok|okay|please|go ahead|do it|do that|send it|go for it|"
                 r"ask (him|her|it|them)|affirmative|absolutely|definitely)\b", re.I)

SETTINGS = {}
_pending = None           # {"question", "deep", "turn"} after J.A.R.V.I.S. offered to ask
_web_direct = False       # set if the API refuses the code-filtered web search for this model
_usage_lock = threading.Lock()
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "jarvis"
USAGE_FILE = STATE_DIR / "claude-usage.json"

SYSTEM = """You are answering a question relayed by J.A.R.V.I.S., a local voice assistant. {about}
Today is {today}.
Start your reply with one line that begins with "SUMMARY:" and gives the answer in at most two short sentences. That line is read aloud, so use plain words: no markdown, symbols, code or web addresses.
Then a blank line, then the full answer in Markdown. It is saved to a file for reading on screen, so it can include code blocks, tables and detail, but stay focused.
If you search the web, prefer primary sources. If you are unsure, say so."""


# --------------------------------------------------------------------------
# Settings, key and budget
# --------------------------------------------------------------------------
def _path(value):
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def setup(settings):
    global _web_direct
    prices = dict(DEFAULT_PRICES)
    for model, pair in (settings.get("prices") or {}).items():
        prices[model] = (float(pair[0]), float(pair[1]))
    SETTINGS.clear()
    SETTINGS.update({
        "model": str(settings.get("model", "claude-haiku-5-5")),
        "deep_model": str(settings.get("deep_model", "claude-sonnet-5-5")),
        "max_tokens": int(settings.get("max_tokens", 2000)),
        "deep_max_tokens": int(settings.get("deep_max_tokens", 8000)),
        "monthly_budget": float(settings.get("monthly_budget", 10.0)),
        "web_search": bool(settings.get("web_search", True)),
        "max_searches": max(1, int(settings.get("max_searches", 3))),
        "code_execution": bool(settings.get("code_execution", True)),
        "trigger_words": [str(w).lower() for w in settings.get("trigger_words", ["claude"])],
        "about_me": str(settings.get("about_me", "")).strip(),
        "key_file": _path(settings.get("key_file", "~/.config/rima/anthropic-key")),
        "save_dir": _path(settings.get("save_dir", "~/.local/share/jarvis/claude")),
        "timeout": float(settings.get("timeout", 120)),
        "base_url": settings.get("base_url") or None,   # only for testing with a stand-in server
        "prices": prices,
    })
    if not SETTINGS["trigger_words"]:
        raise ValueError("trigger_words can't be empty: Claude must only be asked on request")
    _web_direct = False


def _read_key():
    """The API key from its file. Refuses a key file other users could read."""
    path = SETTINGS["key_file"]
    if not path.exists():
        raise ValueError(f"there's no Claude API key yet. Save it to {path} (see the [tools.ask_claude] notes in config.toml)")
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ValueError(f"the key file {path} can be read by other users. Fix it with: chmod 600 {path}")
    key = path.read_text().strip()
    if not key:
        raise ValueError(f"the key file {path} is empty")
    return key


def _month():
    return datetime.now().strftime("%Y-%m")


def _load_usage():
    try:
        return json.loads(USAGE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def month_spend():
    entry = _load_usage().get(_month(), {})
    return float(entry.get("cost", 0.0)), int(entry.get("questions", 0))


def _cost(model, usage):
    """Dollars for one API response, from its token counts and web searches."""
    price_in, price_out = SETTINGS["prices"].get(model, FALLBACK_PRICE)
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    tokens_in = (usage.input_tokens or 0) + 1.25 * cache_write + 0.1 * cache_read
    server = getattr(usage, "server_tool_use", None)
    searches = (getattr(server, "web_search_requests", 0) or 0) if server else 0
    dollars = tokens_in * price_in / 1e6 + (usage.output_tokens or 0) * price_out / 1e6 + searches * SEARCH_PRICE
    return dollars, searches


def _record(dollars, question=False):
    """Add to this month's total (also for answers that arrive after Tony interrupted)."""
    with _usage_lock:
        data = _load_usage()
        entry = data.setdefault(_month(), {"cost": 0.0, "questions": 0})
        entry["cost"] = round(entry.get("cost", 0.0) + dollars, 6)
        entry["questions"] = entry.get("questions", 0) + (1 if question else 0)
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            temp = USAGE_FILE.with_suffix(".tmp")
            temp.write_text(json.dumps(data, indent=1))
            temp.replace(USAGE_FILE)
        except OSError:
            pass


# --------------------------------------------------------------------------
# Talking to the API
# --------------------------------------------------------------------------
def _server_tools():
    found = []
    if SETTINGS["web_search"]:
        search = {"type": "web_search_20260318", "name": "web_search", "max_uses": SETTINGS["max_searches"]}
        if _web_direct:
            search["allowed_callers"] = ["direct"]
        found.append(search)
    if SETTINGS["code_execution"]:
        found.append({"type": "code_execution_20260521", "name": "code_execution"})
    return found


def _friendly(error):
    """API problems -> a plain sentence."""
    import anthropic

    text = str(getattr(error, "message", "") or error)
    if isinstance(error, anthropic.AuthenticationError):
        return f"Claude rejected the API key. Check the key in {SETTINGS['key_file']}."
    if isinstance(error, anthropic.PermissionDeniedError):
        return "the Claude API refused this request (permission denied)."
    if isinstance(error, anthropic.RateLimitError):
        return "Claude's rate limit was reached; try again in a minute."
    if isinstance(error, anthropic.BadRequestError) and re.search(r"credit|billing|spend|limit", text, re.I):
        return "the Claude API account is out of credit or has hit its spend limit (see the Claude Console)."
    if isinstance(error, anthropic.APITimeoutError):
        return f"Claude didn't answer within {SETTINGS['timeout']:g} seconds."
    if isinstance(error, anthropic.APIConnectionError):
        return "I can't reach Claude. The internet connection may be down."
    if isinstance(error, (anthropic.InternalServerError, anthropic.OverloadedError)):
        return "Claude's servers are busy or having trouble; try again shortly."
    return f"Claude returned an error: {text[:200]}"


def _send(question, deep, use_tools=True):
    """
    One question to Claude. Returns {"text", "sources", "model", "cost", "searches"}.
    Usage is recorded as each response arrives, so the monthly total stays right
    even if Tony interrupts.
    """
    global _web_direct
    import anthropic

    model = SETTINGS["deep_model"] if deep else SETTINGS["model"]
    client = anthropic.Anthropic(api_key=_read_key(), base_url=SETTINGS["base_url"],
                                 timeout=SETTINGS["timeout"], max_retries=2)
    about = SETTINGS["about_me"] or ""
    system = SYSTEM.format(about=about, today=datetime.now().strftime("%A, %B %d, %Y"))
    messages = [{"role": "user", "content": question}]
    texts, sources, total, searches = [], {}, 0.0, 0
    for _ in range(5):  # a long search turn can pause; it's continued by sending it back
        kwargs = dict(model=model, max_tokens=SETTINGS["deep_max_tokens"] if deep else SETTINGS["max_tokens"],
                      system=system, messages=messages)
        server_tools = _server_tools() if use_tools else []
        if server_tools:
            kwargs["tools"] = server_tools
        try:
            response = client.messages.create(**kwargs)
        except anthropic.BadRequestError as e:
            if "allowed_callers" in str(e) and not _web_direct and SETTINGS["web_search"]:
                _web_direct = True  # this model can't filter searches with code; search directly
                continue
            raise
        dollars, n = _cost(model, response.usage)
        total += dollars
        searches += n
        _record(dollars)
        for block in response.content:
            if getattr(block, "type", None) != "text":
                # A search or code run between pieces of text: the next text is a
                # new paragraph. (Pieces split only by a citation stay joined.)
                if texts and not texts[-1].endswith("\n\n"):
                    texts.append("\n\n")
                continue
            if texts and texts[-1] == "\n\n":
                texts.append(block.text.lstrip())
            else:
                texts.append(block.text)
            for cite in getattr(block, "citations", None) or []:
                url = getattr(cite, "url", None)
                if url:
                    sources.setdefault(url, getattr(cite, "title", None) or url)
        if response.stop_reason != "pause_turn":
            break
        messages = [messages[0], {"role": "assistant", "content": response.content}]
    _record(0.0, question=True)
    return {"text": "".join(texts).strip(), "sources": sources, "model": model,
            "cost": total, "searches": searches, "stop": response.stop_reason}


def _split(text):
    """Claude's reply -> (spoken summary, full answer)."""
    match = re.search(r"^\s*\**SUMMARY:?\**\s*(.+?)(?:\n\s*\n|\Z)", text, re.S | re.M | re.I)
    if match:
        summary = match.group(1)
        full = (text[:match.start()] + text[match.end():]).strip()
    else:
        full = text
        prose = re.sub(r"```.*?(```|\Z)", " ", text, flags=re.S)        # drop code blocks
        prose = re.sub(r"^\s*(#.*|[|>].*|[-*] .*)$", " ", prose, flags=re.M)  # headings, tables, quotes, lists
        sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"[*_`]", "", " ".join(prose.split())))
        summary = " ".join(sentences[:2])
    summary = " ".join(re.sub(r"[*_`#]", "", summary).split())
    if len(summary) > 400:
        summary = summary[:399].rsplit(" ", 1)[0] + "…"
    return summary, full or text


def _save(question, result, full):
    """Write the full answer to a Markdown file only Tony can read. Returns its path."""
    folder = SETTINGS["save_dir"]
    folder.mkdir(parents=True, exist_ok=True)
    os.chmod(folder, 0o700)
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower())[:40].strip("-") or "answer"
    path = folder / f"{datetime.now():%Y-%m-%d_%H%M%S}-{slug}.md"
    lines = [f"# {question}", "",
             f"*Asked {datetime.now():%Y-%m-%d %I:%M %p} · {result['model']} · "
             f"{result['searches']} web searches · about ${result['cost']:.4f}*", "", full]
    if result["sources"]:
        lines += ["", "## Sources", ""] + [f"- [{title}]({url})" for url, title in result["sources"].items()]
    if result.get("stop") == "max_tokens":
        lines += ["", "*(Cut short: the answer reached the length limit.)*"]
    path.write_text("\n".join(lines) + "\n")
    os.chmod(path, 0o600)
    return path


def _last_answer():
    folder = SETTINGS.get("save_dir")
    files = sorted(folder.glob("*.md")) if folder and folder.exists() else []
    return files[-1] if files else None


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
@tool(writes=True, untrusted=True)
def ask_claude(question: str) -> str:
    """Send Tony's question to Claude, a more capable online AI that can search the web and run code. Only when he asks for Claude or agreed to it.

    Args:
        question: the question in full, with the details needed to answer it
    """
    global _pending
    question = " ".join(str(question or "").split())
    request = tools.current_request()
    asked_by_name = any(word in request.lower() for word in SETTINGS["trigger_words"])
    agreed = (_pending is not None and 1 <= tools.current_turn() - _pending["turn"] <= 2
              and YES.match(request.strip().lower()))
    if agreed:
        question, deep = _pending["question"], _pending["deep"]   # exactly what he agreed to
    elif asked_by_name:
        if not question:
            raise ValueError("there's no question to send")
        deep = bool(DEEP_WORDS.search(request))
    else:
        _pending = {"question": question or request, "deep": bool(DEEP_WORDS.search(request)),
                    "turn": tools.current_turn()}
        return ("Not sent: Claude is a paid online service, used only when Tony asks for it. "
                "Ask him in one short sentence whether to ask Claude. If he says yes, call ask_claude again.")
    _pending = None

    spent, _ = month_spend()
    if spent >= SETTINGS["monthly_budget"]:
        raise ValueError(f"this month's Claude budget (${SETTINGS['monthly_budget']:.2f}) is used up. "
                         "Tony can raise monthly_budget in config.toml.")
    _read_key()  # fail now, before announcing, if the key is missing

    tools.announce("One moment, sir. Asking Claude." if not deep else "One moment, sir. Claude is thinking it over.")
    outcome = {}

    def work():
        try:
            outcome["result"] = _send(question, deep)
        except Exception as e:  # noqa: BLE001 (reported below)
            outcome["error"] = e

    worker = threading.Thread(target=work, name="ask-claude", daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(0.1)
        if tools.cancel_requested():
            return "Stopped: Tony interrupted, so Claude's answer will be ignored."
    if "error" in outcome:
        error = outcome["error"]
        if isinstance(error, ValueError):
            raise error
        raise ValueError(_friendly(error))

    result = outcome["result"]
    summary, full = _split(result["text"])
    if not full:
        raise ValueError("Claude sent back an empty answer")
    path = _save(question, result, full)
    print(f"  [claude] {result['model']}, {result['searches']} searches, ${result['cost']:.4f}, saved {path}")
    return f"Claude's answer: {summary} (The full answer is saved; show_claude_answer opens it.)"


@tool
def show_claude_answer() -> str:
    """Open Claude's most recent full answer in a window on screen."""
    path = _last_answer()
    if path is None:
        raise ValueError("there's no saved answer from Claude yet")
    from tools.desktop import _i3_command  # launch through i3, so restarting J.A.R.V.I.S. doesn't close it

    title = shlex.quote("Claude's answer")
    _i3_command(f"exec --no-startup-id zenity --text-info --title={title} --width=900 --height=750 "
                f"--filename={shlex.quote(str(path))}")
    return "Opening Claude's last answer."


@tool
def claude_usage() -> str:
    """How much the Claude API has cost this month, against the monthly budget."""
    spent, questions = month_spend()
    budget = SETTINGS["monthly_budget"]
    return (f"This month: {questions} question{'s' if questions != 1 else ''} to Claude, "
            f"about ${spent:.2f} of the ${budget:.2f} budget.")


def briefing():
    spent, _ = month_spend()
    budget = SETTINGS.get("monthly_budget", 10.0)
    if spent >= 0.8 * budget:
        return f"Claude budget: ${spent:.2f} of ${budget:.2f} used this month."
    return None


# --------------------------------------------------------------------------
# Command line check:  uv run python -m tools.ask_claude test
# --------------------------------------------------------------------------
def _main(argv):
    config_file = Path(__file__).resolve().parent.parent / "config.toml"
    with config_file.open("rb") as f:
        setup(tomllib.load(f).get("tools", {}).get("ask_claude", {}))
    command = argv[1] if len(argv) > 1 else "test"
    if command == "usage":
        print(claude_usage())
        return 0
    if command != "test":
        print("usage: python -m tools.ask_claude [test|usage]")
        return 2
    try:
        _read_key()
        print(f"Key file OK: {SETTINGS['key_file']}")
        started = time.monotonic()
        result = _send("Reply with just the word OK.", deep=False, use_tools=False)
    except ValueError as e:
        print(f"Problem: {e}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"Problem: {_friendly(e)}")
        return 1
    print(f"Claude ({result['model']}) answered {result['text']!r} in {time.monotonic() - started:.1f} s, "
          f"cost about ${result['cost']:.5f}.")
    print(claude_usage())
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
