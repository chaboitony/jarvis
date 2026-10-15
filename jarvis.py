# jarvis-install: .
"""
jarvis.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
Local model (Ollama) + plugin tools from tools/, by keyboard or by voice.

  uv run jarvis.py           typed commands
  uv run jarvis.py --voice   push-to-talk: press the i3 talk key, speak, hear the reply
                             (normally run in the background by jarvis-voice.service)

Typed commands while running:
  reset    clear memory
  reload   re-read config.toml and the tool files
  tools    list the loaded tools
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
        self.apply(config)
        self.messages = [self._system_message()]

    def apply(self, config):
        """Take settings from config.toml and (re)load the tool plugins."""
        brain = config.get("brain", {})
        self.model = brain.get("model", "qwen3.5:4b")
        self.think = brain.get("think")  # None = don't send the setting at all
        self.keep_alive = brain.get("keep_alive", "30m")
        self.max_tool_rounds = int(brain.get("max_tool_rounds", 5))
        self.keep_turns = int(brain.get("keep_turns", 6))
        self.toolbox.load(config)

    def reload(self):
        """Re-read config.toml and the tool files. Returns a report to show Tony."""
        try:
            config = load_config()
        except SystemExit as e:  # keep running on the old settings
            return f"Reload failed, still using the old settings. {e}"
        self.apply(config)
        return "Tools reloaded:\n" + self.toolbox.summary()

    def _system_message(self):
        text = CORE_PROMPT.format(now=datetime.now(), tool_prompts=self.toolbox.prompt)
        return {"role": "system", "content": text}

    def warm_up(self):
        """Load the model into VRAM now so the first answer isn't slow."""
        self.client.generate(model=self.model, prompt="", keep_alive=self.keep_alive)

    def reset(self):
        self.messages = self.messages[:1]

    def ask(self, text):
        self.messages[0] = self._system_message()  # refresh date/time and tool instructions
        self.messages.append({"role": "user", "content": text})
        self.toolbox.start_turn()
        try:
            reply = self._run_turn()
        except Exception:
            self.messages.pop()
            raise
        self._trim()
        return reply

    def _run_turn(self):
        """Let the model call tools until it has a final answer."""
        for _ in range(self.max_tool_rounds):
            response = self.client.chat(
                model=self.model, messages=self.messages, tools=self.toolbox.schemas,
                think=self.think, keep_alive=self.keep_alive,
            )
            msg = response.message
            self.messages.append(msg)

            if not msg.tool_calls:
                return (msg.content or "").strip() or "Done."

            for call in msg.tool_calls:
                name = call.function.name
                args = call.function.arguments or {}
                result = self.toolbox.call(name, args)
                print(f"  [tool] {name}({args}) -> {result[:200]}")
                self.messages.append({"role": "tool", "content": result, "tool_name": name})

        return "I couldn't finish that request. Try rephrasing it."

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
    brain = Brain(load_config())
    print(_startup_report(brain))
    print("Commands: reset, reload, tools, quit.")

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
        try:
            reply = brain.ask(text)
        except Exception as e:  # Ollama not running, model missing, etc.
            print(f"  [error] {e}")
            continue
        print(f"J.A.R.V.I.S.: {reply}")


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
    from voice import Ears, Mouth, beep, notify  # only load audio libraries in voice mode

    config = load_config()
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
                notify("J.A.R.V.I.S.", report)

            if not talk.wait(timeout=0.5):  # short timeout keeps Ctrl+C responsive
                continue
            talk.clear()

            set_state("listening")
            beep(880)
            notify("Listening…")
            audio = ears.record()
            beep(660)
            if audio is None:
                notify("Didn't hear anything")
                set_state("ready")
                continue

            set_state("thinking")
            text = ears.transcribe(audio)
            if not text:
                notify("Couldn't make that out")
                set_state("ready")
                continue
            print(f"\nYou (voice): {text}")
            notify("You said", text)

            command = re.sub(r"[^a-z ]", "", text.lower()).strip()
            if command in RESET_PHRASES:
                brain.reset()
                reply = "Memory cleared."
            elif command in RELOAD_PHRASES:
                report = brain.reload()
                print(report)
                reply = "Reload failed. Check the log." if report.startswith("Reload failed") \
                    else f"Tools reloaded. {len(brain.toolbox.entries)} tools ready."
            else:
                try:
                    reply = brain.ask(text)
                except Exception as e:
                    print(f"  [error] {e}")
                    reply = "Sorry, I couldn't reach the language model."

            print(f"J.A.R.V.I.S.: {reply}")
            notify("J.A.R.V.I.S.", reply)
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
