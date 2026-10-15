"""
jarvis.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
Local model (Ollama) + Taskwarrior tools, by keyboard or by voice.

  uv run jarvis.py           typed commands
  uv run jarvis.py --voice   push-to-talk: press the i3 talk key, speak, hear the reply

Typed commands while running:  reset (clear memory)   quit (exit)
"""
import argparse
import os
import re
import signal
import threading
from datetime import datetime
from pathlib import Path

import ollama

from task_tools import TOOLS, call_tool

MODEL = "qwen3.5:4b"
# Skip the model's hidden "thinking" step so replies come back quickly.
# If you switch to a model without thinking support and get an error, set to None.
THINK = False
KEEP_ALIVE = "30m"    # keep the model in VRAM this long after the last request
MAX_TOOL_ROUNDS = 5   # tool calls allowed per request before giving up
KEEP_TURNS = 6        # past requests remembered (keeps the context window small)

HERE = Path(__file__).resolve().parent
VOICE_FILE = HERE / "voices" / "en_GB-alan-medium.onnx"
WHISPER_MODEL = "small.en"
PID_FILE = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "jarvis.pid"
RESET_PHRASES = {"reset", "clear memory", "forget everything"}


def system_prompt():
    now = datetime.now()
    return f"""You are J.A.R.V.I.S., the personal assistant of Anthony Rima, who goes by Tony.
Address him as "Mr. Rima", "Tony" or "sir". He is not Tony Stark; never call him Mr. Stark.
PROJECT RIMA is his personal engineering program, named after him.
Today is {now:%A, %B %d, %Y}, and the time is {now:%I:%M %p}.

You manage Tony's Taskwarrior tasks with the tools provided.
Projects match his Obsidian branches:
- RDC: research and technical theory
- RAIL: prototypes and engineering builds
- AOC: coursework and school
- KIA: archives and reference
- SOC: planning and strategy
- POD: project management

Rules:
- Use a tool whenever a request involves tasks. Never guess a task ID; call list_tasks first.
- Choose the project that fits best. If none clearly fits, leave it out.
- For dates use YYYY-MM-DD, or today, tomorrow, a weekday name, eow, eom or eoy.
- You cannot delete tasks. If asked, say so and offer to mark the task done instead.
- If a tool returns an error, explain it simply or fix your request and try again.
- Requests may come from speech recognition, so expect small transcription mistakes.
- Reply in one to three short plain sentences with no markdown, because replies may be read aloud."""


# --------------------------------------------------------------------------
# The brain: shared by typed and voice mode
# --------------------------------------------------------------------------
class Brain:
    def __init__(self):
        self.client = ollama.Client()  # talks to http://localhost:11434
        self.messages = [{"role": "system", "content": system_prompt()}]

    def warm_up(self):
        """Load the model into VRAM now so the first answer isn't slow."""
        self.client.generate(model=MODEL, prompt="", keep_alive=KEEP_ALIVE)

    def reset(self):
        self.messages = self.messages[:1]

    def ask(self, text):
        self.messages[0] = {"role": "system", "content": system_prompt()}  # refresh date/time
        self.messages.append({"role": "user", "content": text})
        try:
            reply = self._run_turn()
        except Exception:
            self.messages.pop()
            raise
        self._trim()
        return reply

    def _run_turn(self):
        """Let the model call tools until it has a final answer."""
        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.chat(
                model=MODEL, messages=self.messages, tools=TOOLS,
                think=THINK, keep_alive=KEEP_ALIVE,
            )
            msg = response.message
            self.messages.append(msg)

            if not msg.tool_calls:
                return (msg.content or "").strip() or "Done."

            for call in msg.tool_calls:
                name = call.function.name
                args = call.function.arguments or {}
                result = call_tool(name, args)
                print(f"  [tool] {name}({args}) -> {result[:200]}")
                self.messages.append({"role": "tool", "content": result, "tool_name": name})

        return "I couldn't finish that request. Try rephrasing it."

    def _trim(self):
        """Keep the system prompt plus the last KEEP_TURNS requests and their replies."""
        user_positions = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
        if len(user_positions) > KEEP_TURNS:
            cut = user_positions[-KEEP_TURNS]
            self.messages = [self.messages[0], *self.messages[cut:]]


# --------------------------------------------------------------------------
# Typed mode
# --------------------------------------------------------------------------
def typed_mode():
    brain = Brain()
    print(f"J.A.R.V.I.S. online ({MODEL}). Type 'reset' to clear memory or 'quit' to exit.")

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            break
        if text.lower() == "reset":
            brain.reset()
            print("Memory cleared.")
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
def voice_mode():
    from voice import Ears, Mouth, beep, notify  # only load audio libraries in voice mode

    if not VOICE_FILE.exists():
        raise SystemExit(f"Voice file not found: {VOICE_FILE}\nDownload it first (see setup steps).")

    print("Loading speech recognition, voice and language model...")
    ears = Ears(WHISPER_MODEL)
    mouth = Mouth(VOICE_FILE)
    brain = Brain()
    brain.warm_up()

    # The talk key sends this process the USR1 signal (see jarvis-ptt).
    # The PID file tells jarvis-ptt which process to signal.
    talk = threading.Event()
    signal.signal(signal.SIGUSR1, lambda *_: talk.set())
    PID_FILE.write_text(str(os.getpid()))

    print(f"J.A.R.V.I.S. voice mode online ({MODEL}). Press the talk key to speak; Ctrl+C to exit.")
    mouth.say("J.A.R.V.I.S. online.")

    try:
        while True:
            if not talk.wait(timeout=0.5):  # short timeout keeps Ctrl+C responsive
                continue
            talk.clear()

            beep(880)
            notify("Listening…")
            audio = ears.record()
            beep(660)
            if audio is None:
                notify("Didn't hear anything")
                continue

            text = ears.transcribe(audio)
            if not text:
                notify("Couldn't make that out")
                continue
            print(f"\nYou (voice): {text}")
            notify("You said", text)

            if re.sub(r"[^a-z ]", "", text.lower()).strip() in RESET_PHRASES:
                brain.reset()
                reply = "Memory cleared."
            else:
                try:
                    reply = brain.ask(text)
                except Exception as e:
                    print(f"  [error] {e}")
                    reply = "Sorry, I couldn't reach the language model."

            print(f"J.A.R.V.I.S.: {reply}")
            notify("J.A.R.V.I.S.", reply)
            mouth.say(reply)
            talk.clear()  # ignore presses that happened while busy
    except KeyboardInterrupt:
        print()
    finally:
        PID_FILE.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. | PROJECT RIMA")
    parser.add_argument("--voice", action="store_true", help="push-to-talk voice mode")
    args = parser.parse_args()
    voice_mode() if args.voice else typed_mode()


if __name__ == "__main__":
    main()
