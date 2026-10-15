# jarvis-install: .
"""
jarvis.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
Local model (Ollama) + plugin tools from tools/, by keyboard or by voice.

  uv run jarvis.py           typed commands
  uv run jarvis.py --voice   push-to-talk: press the i3 talk key, speak, hear the reply
                             (normally run in the background by jarvis-voice.service)

Conversations are saved to ~/.local/state/jarvis/history/ (see jarvis-history).

Typed commands while running:
  reset    clear memory
  reload   re-read config.toml and the tool files
  tools    list the loaded tools
  route X  show which plugins request X would be offered (routing check)
  quit     exit

Settings live in config.toml. Abilities live in tools/ (see tools/__init__.py).
"""
import argparse
import os
import re
import signal
import threading
import tomllib
from datetime import datetime
from pathlib import Path

import ollama

from tools import Toolbox

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "config.toml"
RUN_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
PID_FILE = RUN_DIR / "jarvis.pid"      # tells jarvis-ptt which process to signal
STATE_FILE = RUN_DIR / "jarvis.state"  # what voice mode is doing, for the Polybar icon
# Conversation history, one file per day (shown by jarvis-history). Kept out of the
# git repo because it contains what you said.
HISTORY_DIR = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "jarvis" / "history"
RESET_PHRASES = {"reset", "clear memory", "forget everything"}
RELOAD_PHRASES = {"reload tools", "reload your tools"}


def load_config():
    """Read config.toml. Raises a readable error if it's missing or has a typo."""
    try:
        with CONFIG_FILE.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise SystemExit(f"Settings file not found: {CONFIG_FILE}")
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"config.toml has a mistake: {e}")


# --------------------------------------------------------------------------
# Conversation history
# --------------------------------------------------------------------------
def log_exchange(speaker, text):
    """Append one line to today's history file, e.g. '02:06 AM  You (voice): ...'."""
    try:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        text = " ".join(str(text).split())  # one line per message
        with (HISTORY_DIR / f"{now:%Y-%m-%d}.txt").open("a") as f:
            f.write(f"{now:%I:%M %p}  {speaker}: {text}\n")
    except OSError:
        pass  # history is a convenience; never let it stop J.A.R.V.I.S.


def prune_history(config):
    """Delete history files older than [history] keep_days (default 30)."""
    try:
        keep_days = int(config.get("history", {}).get("keep_days", 30))
        cutoff = datetime.now().timestamp() - keep_days * 86400
        for f in HISTORY_DIR.glob("*.txt"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
    except (OSError, ValueError, TypeError):
        pass


CORE_PROMPT = """You are J.A.R.V.I.S., the personal assistant of Anthony Rima, who goes by Tony.
Address him as "Mr. Rima", "Tony" or "sir". He is not Tony Stark; never call him Mr. Stark.
PROJECT RIMA is his personal engineering program, named after him.
Today is {now:%A, %B %d, %Y}, and the time is {now:%I:%M %p}.

{tool_prompts}

General rules:
- If a tool returns an error, explain it simply or fix your request and try again.
- Requests may come from speech recognition, so expect small transcription mistakes.
- Reply in one to three short plain sentences with no markdown, because replies may be read aloud."""


# --------------------------------------------------------------------------
# The brain: shared by typed and voice mode
# --------------------------------------------------------------------------
class Brain:
    def __init__(self, config):
        self.client = ollama.Client()  # talks to http://localhost:11434
        self.toolbox = Toolbox()
        self.last_plugins = []  # plugins used in the last request (kept for follow-ups)
        self.apply(config)
        self.messages = [self._system_message(self.toolbox.loaded)]

    def apply(self, config):
        """Take settings from config.toml and (re)load the tool plugins."""
        self.config = config
        brain = config.get("brain", {})
        self.model = brain.get("model", "qwen3.5:4b")
        self.think = brain.get("think")  # None = don't send the setting at all
        self.keep_alive = brain.get("keep_alive", "30m")
        self.max_tool_rounds = int(brain.get("max_tool_rounds", 5))
        self.keep_turns = int(brain.get("keep_turns", 6))
        self.routing = bool(brain.get("routing", True))
        self.routing_plugins = max(1, int(brain.get("routing_plugins", 3)))
        self.embed_model = brain.get("embed_model", "nomic-embed-text")
        self.toolbox.load(config)
        self.last_plugins = [p for p in self.last_plugins if p in self.toolbox.loaded]

    def reload(self):
        """Re-read config.toml and the tool files. Returns a report to show Tony."""
        try:
            config = load_config()
        except SystemExit as e:  # keep running on the old settings
            return f"Reload failed, still using the old settings. {e}"
        self.apply(config)
        return "Tools reloaded:\n" + self.toolbox.summary()

    def _system_message(self, plugins):
        text = CORE_PROMPT.format(now=datetime.now(), tool_prompts=self.toolbox.prompt_for(plugins))
        return {"role": "system", "content": text}

    def _embed(self, texts):
        """Turn texts into embeddings with the small embedding model (used for routing)."""
        return self.client.embed(model=self.embed_model, input=texts, keep_alive=self.keep_alive).embeddings

    def route(self, text):
        """Which plugins to offer for this request: ([plugins], {plugin: score})."""
        if not self.routing:
            return list(self.toolbox.loaded), {}
        return self.toolbox.select(text, self._embed, sticky=self.last_plugins, top=self.routing_plugins)

    def warm_up(self):
        """Load the models into memory now so the first answer isn't slow."""
        self.client.generate(model=self.model, prompt="", keep_alive=self.keep_alive)
        if self.routing:
            try:
                self.route("warm up")  # also builds the plugin descriptions for routing
            except Exception:
                pass

    def reset(self):
        self.messages = self.messages[:1]
        self.last_plugins = []

    def ask(self, text):
        plugins, scores = self.route(text)
        if self.toolbox.route_error:
            print(f"  [route] routing unavailable, offering every tool ({self.toolbox.route_error})")
        elif scores:
            print(f"  [route] {', '.join(plugins)}")
        # refresh date/time, and include only the chosen plugins' instructions
        self.messages[0] = self._system_message(plugins)
        self.messages.append({"role": "user", "content": text})
        self.toolbox.start_turn()
        try:
            reply, used = self._run_turn(self.toolbox.schemas_for(plugins))
        except Exception:
            self.messages.pop()
            raise
        if used:
            self.last_plugins = sorted(used)
        self._trim()
        return reply

    def _run_turn(self, tools):
        """Let the model call tools until it has a final answer. Returns (reply, plugins used)."""
        used = set()
        for _ in range(self.max_tool_rounds):
            response = self.client.chat(
                model=self.model, messages=self.messages, tools=tools,
                think=self.think, keep_alive=self.keep_alive,
            )
            msg = response.message
            self.messages.append(msg)

            if not msg.tool_calls:
                return (msg.content or "").strip() or "Done.", used

            for call in msg.tool_calls:
                name = call.function.name
                args = call.function.arguments or {}
                result = self.toolbox.call(name, args)
                if name in self.toolbox.entries:
                    used.add(self.toolbox.entries[name]["plugin"])
                print(f"  [tool] {name}({args}) -> {result[:200]}")
                self.messages.append({"role": "tool", "content": result, "tool_name": name})

        return "I couldn't finish that request. Try rephrasing it.", used

    def _trim(self):
        """Keep the system prompt plus the last keep_turns requests and their replies."""
        user_positions = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
        if len(user_positions) > self.keep_turns:
            cut = user_positions[-self.keep_turns]
            self.messages = [self.messages[0], *self.messages[cut:]]


def _startup_report(brain):
    return f"J.A.R.V.I.S. online ({brain.model}). Tools:\n{brain.toolbox.summary()}"


# --------------------------------------------------------------------------
# Typed mode
# --------------------------------------------------------------------------
def typed_mode():
    config = load_config()
    prune_history(config)
    brain = Brain(config)
    print(_startup_report(brain))
    print("Commands: reset, reload, tools, route <request>, quit.")

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        command = text.lower()
        if command in {"quit", "exit"}:
            break
        if command == "reset":
            brain.reset()
            print("Memory cleared.")
            continue
        if command == "reload":
            print(brain.reload())
            continue
        if command == "tools":
            print(brain.toolbox.summary())
            continue
        if command.startswith("route "):  # show which plugins a request would get
            plugins, scores = brain.route(text[6:])
            if brain.toolbox.route_error:
                print(f"Routing unavailable ({brain.toolbox.route_error}); every plugin would be offered.")
            elif not scores:
                print("Routing is off or there are too few plugins to choose from; all are offered.")
            else:
                for name in sorted(scores, key=scores.get, reverse=True):
                    print(f"  {'*' if name in plugins else ' '} {scores[name]:.3f}  {name}")
                print("  (* = offered to the model)")
            continue
        log_exchange("You (typed)", text)
        try:
            reply = brain.ask(text)
        except Exception as e:  # Ollama not running, model missing, etc.
            print(f"  [error] {e}")
            log_exchange("Error", e)
            continue
        print(f"J.A.R.V.I.S.: {reply}")
        log_exchange("J.A.R.V.I.S.", reply)


# --------------------------------------------------------------------------
# Voice mode
# --------------------------------------------------------------------------
def set_state(state):
    """
    Record what voice mode is doing: starting, ready, listening, thinking or speaking.
    Written to a temporary name and then renamed, so a reader never sees half a word.
    """
    try:
        temp = STATE_FILE.with_name(STATE_FILE.name + ".tmp")
        temp.write_text(state)
        temp.replace(STATE_FILE)
    except OSError:
        pass  # the icon is a nicety; never let it stop J.A.R.V.I.S.


def _stop(*_):
    """SIGTERM (from `systemctl --user stop`) -> exit normally so cleanup runs."""
    raise SystemExit(0)


def voice_mode():
    signal.signal(signal.SIGTERM, _stop)
    set_state("starting")
    try:
        _voice_loop()
    finally:
        PID_FILE.unlink(missing_ok=True)
        STATE_FILE.unlink(missing_ok=True)


def _voice_loop():
    # only load audio libraries in voice mode
    from voice import Ears, Mouth, beep, configure_notifications, notify

    config = load_config()
    prune_history(config)
    configure_notifications(config.get("notifications", {}))
    voice_cfg = config.get("voice", {})
    voice_file = HERE / voice_cfg.get("voice_file", "voices/en_GB-alan-medium.onnx")
    if not voice_file.exists():
        raise SystemExit(f"Voice file not found: {voice_file}\nDownload it first (see setup steps).")

    print("Loading speech recognition, voice and language model...")
    ears = Ears(voice_cfg.get("whisper_model", "small.en"))
    mouth = Mouth(voice_file)
    brain = Brain(config)
    brain.warm_up()

    # Signals from other programs (the PID file tells them which process to signal):
    #   USR1  the talk key was pressed (sent by jarvis-ptt)
    #   USR2  tool files changed, reload them (sent by jarvis-update)
    # The handlers only set flags; the main loop does the work between requests.
    talk = threading.Event()
    reload_requested = threading.Event()
    signal.signal(signal.SIGUSR1, lambda *_: talk.set())
    signal.signal(signal.SIGUSR2, lambda *_: reload_requested.set())
    PID_FILE.write_text(str(os.getpid()))
    set_state("ready")

    print(_startup_report(brain))
    print("Voice mode: press the talk key to speak. Stop with Ctrl+C, or "
          "`systemctl --user stop jarvis-voice` when running as a service.")
    if voice_cfg.get("announce_startup", True):
        set_state("speaking")
        mouth.say("J.A.R.V.I.S. online.")
        set_state("ready")

    try:
        while True:
            if reload_requested.is_set():
                reload_requested.clear()
                report = brain.reload()
                print(report)
                configure_notifications(brain.config.get("notifications", {}))
                notify("J.A.R.V.I.S.", report)

            if not talk.wait(timeout=0.5):  # short timeout keeps Ctrl+C responsive
                continue
            talk.clear()

            set_state("listening")
            beep(880)
            notify("J.A.R.V.I.S.", "Listening…")
            audio = ears.record()
            beep(660)
            if audio is None:
                notify("J.A.R.V.I.S.", "Didn't hear anything.", seconds=4)
                set_state("ready")
                continue

            set_state("thinking")
            text = ears.transcribe(audio)
            if not text:
                notify("J.A.R.V.I.S.", "Couldn't make that out.", seconds=4)
                set_state("ready")
                continue
            print(f"\nYou (voice): {text}")
            log_exchange("You (voice)", text)
            # One notification shows your words, then your words plus the reply
            notify("J.A.R.V.I.S.", f"You: {text}\n\nThinking…")

            command = re.sub(r"[^a-z ]", "", text.lower()).strip()
            if command in RESET_PHRASES:
                brain.reset()
                reply = "Memory cleared."
            elif command in RELOAD_PHRASES:
                report = brain.reload()
                print(report)
                reply = "Reload failed. Check the log." if report.startswith("Reload failed") \
                    else f"Tools reloaded. {len(brain.toolbox.entries)} tools ready."
                configure_notifications(brain.config.get("notifications", {}))
            else:
                try:
                    reply = brain.ask(text)
                except Exception as e:
                    print(f"  [error] {e}")
                    reply = "Sorry, I couldn't reach the language model."

            print(f"J.A.R.V.I.S.: {reply}")
            log_exchange("J.A.R.V.I.S.", reply)
            notify("J.A.R.V.I.S.", f"You: {text}\n\n{reply}")
            set_state("speaking")
            mouth.say(reply)
            set_state("ready")
            talk.clear()  # ignore presses that happened while busy
    except KeyboardInterrupt:
        print()


def main():
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. | PROJECT RIMA")
    parser.add_argument("--voice", action="store_true", help="push-to-talk voice mode")
    args = parser.parse_args()
    voice_mode() if args.voice else typed_mode()


if __name__ == "__main__":
    main()
