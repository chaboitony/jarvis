# jarvis-install: .
"""
jarvis.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
Local model (Ollama) + plugin tools from tools/, by keyboard or by voice.

  uv run jarvis.py             typed commands
  uv run jarvis.py --voice     push-to-talk: press the i3 talk key, speak, hear the reply
                               (normally run in the background by jarvis-voice.service)
  uv run jarvis.py --mic-test  check the microphone and speech recognition
                               (--model and --device try other Whisper settings)
  uv run jarvis.py --voice-test  hear J.A.R.V.I.S.'s voice options and pick one
                               (--voices and --speed try specific settings)

Conversations are saved to ~/.local/state/jarvis/history/ (see jarvis-history).

Typed commands while running:
  reset    clear memory
  reload   re-read config.toml and the tool files
  tools    list the loaded tools
  route X  show which plugins request X would be offered (routing check)
  quit     exit

Voice mode streams: J.A.R.V.I.S. starts speaking the first sentence while the
model is still writing the rest. Press the talk key while it's thinking or
speaking to cut it off and ask something else.

Settings live in config.toml. Abilities live in tools/ (see tools/__init__.py).
"""
import argparse
import os
import re
import signal
import threading
import time
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
- Reply in one to three short plain sentences with no markdown, because replies may be read aloud.
- Write the way you would say it: words instead of symbols ("72 degrees", "40 percent"), no lists or parentheses."""


# --------------------------------------------------------------------------
# Cutting a streamed reply into sentences
# --------------------------------------------------------------------------
class SentenceSplitter:
    """
    The model writes a few characters at a time. This collects them and hands
    out whole sentences, so speech can start after the first one instead of
    waiting for the full reply. "Mr. Rima" and "3.5" don't count as sentence ends.
    """

    END = re.compile(r"[.!?…]+[\"')\]]*(?=\s)|\n+")
    ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "e.g.", "i.e.", "etc.", "approx.", "no.", "jr.", "sr."}
    LONG = 220  # no sentence end after this many characters: break at a comma instead

    def __init__(self, on_sentence=None):
        self.on_sentence = on_sentence
        self.buffer = ""
        self.count = 0  # sentences handed out so far

    def feed(self, piece):
        self.buffer += piece
        while True:
            cut = self._next_cut()
            if cut is None:
                return
            self.say(self.buffer[:cut])
            self.buffer = self.buffer[cut:]

    def _not_an_end(self, word, mark):
        """True if this '.' probably isn't the end of a sentence: "Mr.", "J.A.R.V.I.S.", "3." (of 3.5)."""
        return mark == "." and (word[-1:].isdigit() or (word + ".").lower() in self.ABBREVIATIONS
                                or re.fullmatch(r"([A-Za-z]\.)*[A-Za-z]", word) is not None)

    def _next_cut(self):
        for match in self.END.finditer(self.buffer):
            if match.group().startswith("\n"):
                return match.end()
            before = self.buffer[:match.start()].split()
            if not before or not self._not_an_end(before[-1], match.group()[0]):
                return match.end()
        # A full stop at the very end of what's arrived so far: cut now rather
        # than wait for the next piece, unless it could still become "3.5"
        tail = re.search(r"(\S+?)([.!?…]+)[\"')\]]*$", self.buffer)
        if tail and not self._not_an_end(tail.group(1), tail.group(2)[0]):
            return len(self.buffer)
        if len(self.buffer) > self.LONG:
            comma = max(self.buffer.rfind(", ", 0, self.LONG), self.buffer.rfind("; ", 0, self.LONG))
            if comma > 0:
                return comma + 1
        return None

    def flush(self):
        self.say(self.buffer)
        self.buffer = ""

    @classmethod
    def split(cls, text):
        """A whole text -> its sentences."""
        out = []
        splitter = cls(out.append)
        splitter.feed(text)
        splitter.flush()
        return out

    def say(self, text):
        text = text.strip()
        if text and self.on_sentence:
            self.count += 1
            self.on_sentence(text)


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

    def ask(self, text, on_text=None, on_sentence=None, cancel=None):
        """
        Answer one request, letting the model use tools.
          on_text(piece)       called with each piece of the reply as the model writes it
          on_sentence(text)    called with each finished sentence (voice mode speaks these)
          cancel               a threading.Event; if it gets set, stop and keep what was said
        Returns the reply. Afterwards self.interrupted says whether it was cut off,
        and self.last_tools lists the tools that ran.
        """
        plugins, scores = self.route(text)
        if self.toolbox.route_error:
            print(f"  [route] routing unavailable, offering every tool ({self.toolbox.route_error})")
        elif scores:
            print(f"  [route] {', '.join(plugins)}")
        # refresh date/time, and include only the chosen plugins' instructions
        self.messages[0] = self._system_message(plugins)
        start = len(self.messages)
        self.messages.append({"role": "user", "content": text})
        self.toolbox.start_turn()
        self.interrupted, self.last_tools = False, []
        sentences = SentenceSplitter(on_sentence)
        try:
            reply, used = self._run_turn(self.toolbox.schemas_for(plugins), on_text, sentences, cancel)
        except BaseException:  # an error, or Ctrl+C in typed mode
            del self.messages[start:]  # forget the failed request completely
            raise
        if sentences.count == 0 and not self.interrupted:
            sentences.say(reply)  # e.g. "Done." when the model wrote nothing
        if used:
            self.last_plugins = sorted(used)
        self._trim()
        return reply

    def note_interrupted(self, heard):
        """
        The reply was cut off while being spoken. Keep only what Tony actually
        heard in memory, so the model doesn't think it said the rest.
        """
        if len(self.messages) > 1 and self.messages[-1]["role"] == "assistant" \
                and not self.messages[-1].get("tool_calls"):
            self.messages[-1] = ollama.Message(
                role="assistant", content=(heard + " …" if heard else "") + " [Tony interrupted]")

    def _stream_reply(self, tools, on_text, sentences, cancel):
        """
        One round of the model writing. Pieces arrive as it writes them (streaming),
        so the first sentence can be spoken before the reply is finished.
        Returns (text, tool calls, cancelled).
        """
        pieces, calls, cancelled = [], [], False
        stream = self.client.chat(
            model=self.model, messages=self.messages, tools=tools,
            think=self.think, keep_alive=self.keep_alive, stream=True,
        )
        try:
            for chunk in stream:
                msg = chunk.message
                if msg.content:
                    pieces.append(msg.content)
                    if on_text:
                        on_text(msg.content)
                    sentences.feed(msg.content)
                if msg.tool_calls:
                    calls.extend(msg.tool_calls)
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    break
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()  # stops Ollama generating the rest
        return "".join(pieces), calls, cancelled

    def _run_turn(self, tools, on_text, sentences, cancel):
        """Let the model call tools until it has a final answer. Returns (reply, plugins used)."""
        used = set()
        for _ in range(self.max_tool_rounds):
            text, calls, cancelled = self._stream_reply(tools, on_text, sentences, cancel)
            if cancelled:
                self.interrupted = True
                partial = text.strip()
                # Remember that the answer was cut off, so a follow-up makes sense
                self.messages.append(ollama.Message(
                    role="assistant", content=(partial + " …" if partial else "") + " [Tony interrupted]"))
                return partial, used

            sentences.flush()
            self.messages.append(ollama.Message(role="assistant", content=text, tool_calls=calls or None))
            if not calls:
                return text.strip() or "Done.", used

            for call in calls:
                name = call.function.name
                args = call.function.arguments or {}
                result = self.toolbox.call(name, args)
                if name in self.toolbox.entries:
                    used.add(self.toolbox.entries[name]["plugin"])
                self.last_tools.append(name)
                print(f"  [tool] {name}({args}) -> {result[:200]}")
                self.messages.append({"role": "tool", "content": result, "tool_name": name})
            if cancel is not None and cancel.is_set():
                self.interrupted = True
                self.messages.append(ollama.Message(role="assistant", content="[Tony interrupted]"))
                return "", used

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
        shown = []

        def show(piece):  # print the reply as the model writes it
            if not shown:
                print("J.A.R.V.I.S.: ", end="")
            shown.append(piece)
            print(piece, end="", flush=True)

        try:
            reply = brain.ask(text, on_text=show)
        except KeyboardInterrupt:  # Ctrl+C while it's answering: stop that answer only
            print("\n  (stopped)")
            continue
        except Exception as e:  # Ollama not running, model missing, etc.
            print(f"\n  [error] {e}" if shown else f"  [error] {e}")
            log_exchange("Error", e)
            continue
        if shown:
            print()
        if not "".join(shown).strip():
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
    from voice import Ears, MediaPause, Mouth, beep, configure_notifications, notify

    config = load_config()
    prune_history(config)
    configure_notifications(config.get("notifications", {}))
    voice_cfg = config.get("voice", {})

    print("Loading speech recognition, voice and language model...")
    mouth = Mouth(voice_cfg, HERE)
    print(f"  [voice] speaking with {mouth.engine.name}")
    if mouth.warning:
        notify("J.A.R.V.I.S.", mouth.warning, seconds=10)
    ears = Ears(voice_cfg)
    print(f"  [voice] Whisper {ears.model_name} on the {ears.device.upper()}")
    if ears.warning:
        notify("J.A.R.V.I.S.", ears.warning, seconds=10)
    media = MediaPause(bool(voice_cfg.get("pause_media", True)))
    brain = Brain(config)
    brain.warm_up()

    # Signals from other programs (the PID file tells them which process to signal):
    #   USR1  the talk key was pressed (sent by jarvis-ptt)
    #   USR2  tool files changed, reload them (sent by jarvis-update)
    # The handlers only set flags; the main loop does the work between requests.
    # A talk press while J.A.R.V.I.S. is thinking or speaking interrupts it and
    # starts listening again.
    talk = threading.Event()
    reload_requested = threading.Event()
    signal.signal(signal.SIGUSR1, lambda *_: talk.set())
    signal.signal(signal.SIGUSR2, lambda *_: reload_requested.set())
    PID_FILE.write_text(str(os.getpid()))
    set_state("ready")

    print(_startup_report(brain))
    print("Voice mode: press the talk key to speak (again to interrupt). Stop with Ctrl+C, or "
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

            # ---- listen -------------------------------------------------
            media.pause()
            set_state("listening")
            beep(880)
            notify("J.A.R.V.I.S.", "Listening…")
            audio = ears.record()
            beep(660)
            talk.clear()  # a press while listening doesn't count as an interruption
            if audio is None:
                notify("J.A.R.V.I.S.", "Didn't hear anything.", seconds=4)
                media.resume()
                set_state("ready")
                continue

            set_state("thinking")
            text = ears.transcribe(audio)
            if not text:
                notify("J.A.R.V.I.S.", "Couldn't make that out.", seconds=4)
                media.resume()
                set_state("ready")
                continue
            print(f"\nYou (voice): {text}")
            log_exchange("You (voice)", text)
            notify("J.A.R.V.I.S.", f"You: {text}\n\nThinking…")

            # ---- answer, speaking each sentence as soon as it's written ----
            spoken = []
            mouth.heard.clear()

            def speak(sentence):
                if not spoken:
                    set_state("speaking")
                spoken.append(sentence)
                mouth.speak(sentence)
                notify("J.A.R.V.I.S.", f"You: {text}\n\n{' '.join(spoken)}")

            command = re.sub(r"[^a-z ]", "", text.lower()).strip()
            interrupted = False
            if command in RESET_PHRASES:
                brain.reset()
                reply = "Memory cleared."
                speak(reply)
            elif command in RELOAD_PHRASES:
                report = brain.reload()
                print(report)
                reply = "Reload failed. Check the log." if report.startswith("Reload failed") \
                    else f"Tools reloaded. {len(brain.toolbox.entries)} tools ready."
                configure_notifications(brain.config.get("notifications", {}))
                speak(reply)
            else:
                try:
                    reply = brain.ask(text, on_sentence=speak, cancel=talk)
                    interrupted = brain.interrupted
                except Exception as e:
                    print(f"  [error] {e}")
                    reply = "Sorry, I couldn't reach the language model."
                    speak(reply)
                if "media_control" in getattr(brain, "last_tools", []):
                    media.forget()  # you asked for a media change; don't undo it

            if not mouth.wait(cancel=talk):
                interrupted = True
            if interrupted:
                mouth.stop()
                heard = " ".join(mouth.heard)
                brain.note_interrupted(heard)
                reply = (heard + " …" if heard else "") + " (interrupted)"

            print(f"J.A.R.V.I.S.: {reply}")
            log_exchange("J.A.R.V.I.S.", reply)
            notify("J.A.R.V.I.S.", f"You: {text}\n\n{reply}")
            if not talk.is_set():  # if it was interrupted, the loop listens again right away
                media.resume()
            set_state("ready")
    except KeyboardInterrupt:
        print()
    finally:
        media.resume()
        mouth.close()


# --------------------------------------------------------------------------
# Microphone test
# --------------------------------------------------------------------------
def mic_test(model=None, device=None):
    """
    Record and transcribe, showing what J.A.R.V.I.S. heard and why.
    Nothing is sent to the language model. Try other Whisper models with
    --model (e.g. distil-large-v3.5) and --device cuda to compare.
    """
    from voice import LAST_RECORDING, Ears, beep, list_input_devices, resolve_input_device

    voice_cfg = dict(load_config().get("voice", {}))
    if model:
        voice_cfg["whisper_model"] = model
    if device:
        voice_cfg["whisper_device"] = device

    print("Microphones (inputs):")
    chosen = resolve_input_device(voice_cfg.get("input_device"))
    for index, name, rate in list_input_devices():
        print(f"  {'*' if index == chosen else ' '} {index:2d}  {name}  ({rate} Hz)")
    if chosen is None:
        print("  Using the system default input (set [voice] input_device to pick another).")

    started = time.monotonic()
    print(f"\nLoading Whisper {voice_cfg.get('whisper_model', 'small.en')}...")
    ears = Ears(voice_cfg)
    print(f"Loaded on the {ears.device.upper()} in {time.monotonic() - started:.1f} s.")
    if ears.warning:
        print(f"Note: {ears.warning}")

    while True:
        try:
            input("\nPress Enter, wait for the beep, then speak (Ctrl+C to quit). ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        beep(880)
        audio = ears.record()
        beep(660)
        info = ears.last
        if audio is None:
            print(f"  Didn't hear speech. Background noise {info['noise_db']:.0f} dB.")
            continue
        text = ears.transcribe(audio)
        info = ears.last
        print(f"  Heard:       \"{text}\"")
        print(f"  Confidence:  {info['confidence']:.0%}   transcribing took {info['transcribe_seconds']:.2f} s")
        print(f"  Recording:   {info['seconds']:.1f} s ({info['speech_seconds']:.1f} s of speech), {info['result']}")
        print(f"  Levels:      voice {info['speech_db']:.0f} dB, peak {info['peak_db']:.0f} dB, "
              f"background {info['noise_db']:.0f} dB (raised {info['gain_db']:.0f} dB for Whisper)")
        if info["peak_db"] < -30:
            print("  Tip: your mic is very quiet. Raise its input volume in pavucontrol (Input Devices).")
        print(f"  Listen to it: pw-play {LAST_RECORDING}")


# --------------------------------------------------------------------------
# Voice test
# --------------------------------------------------------------------------
VOICE_SAMPLE = ("Good evening, sir. All systems are online. Tomorrow looks sunny with a high of "
                "seventy two degrees, and you have two tasks due on Friday. Shall I start the briefing?")
VOICE_CANDIDATES = ["bm_george", "bm_lewis", "bm_daniel", "bm_fable", "bm_george:0.6,bm_fable:0.4"]


def voice_test(voices=None, speed=None):
    """
    Speak the same sample in several voices so you can pick one by ear.
    Uses your [voice] settings; --voices and --speed override them for the test.
    """
    from voice import Mouth, load_voice_engine

    voice_cfg = dict(load_config().get("voice", {}))
    if speed:
        voice_cfg["speed"] = speed
    engine, warning = load_voice_engine(voice_cfg, HERE)
    if warning:
        print(f"Note: {warning}")
    mouth = Mouth(voice_cfg, HERE, engine=engine)
    if engine.name != "Kokoro":
        print("Speaking with Piper (set tts_engine = \"kokoro\" to compare Kokoro voices).")
        mouth.say(VOICE_SAMPLE)
        mouth.close()
        return

    choices = [v for v in re.split(r"[|\s]+", voices or "") if v] or \
        [voice_cfg.get("kokoro_voice", "bm_george")] + VOICE_CANDIDATES
    seen = []
    for v in choices:
        if v not in seen:
            seen.append(v)
    print(f"Speed {engine.speed}, pause {mouth.pause} s between sentences.")
    try:
        for number, v in enumerate(seen, 1):
            try:
                engine.set_voice(v)
            except ValueError as e:
                print(f"  {number}. {v}: {e}")
                continue
            print(f"  {number}. {v}")
            for sentence in SentenceSplitter.split(VOICE_SAMPLE):
                mouth.speak(sentence)  # sentence by sentence, the way replies arrive
            mouth.wait()
            time.sleep(0.6)
    except KeyboardInterrupt:
        print()
    mouth.close()
    print('\nTo use one, set kokoro_voice = "<name>" under [voice] in config.toml, then:\n'
          "  systemctl --user restart jarvis-voice")



def main():
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. | PROJECT RIMA")
    parser.add_argument("--voice", action="store_true", help="push-to-talk voice mode")
    parser.add_argument("--mic-test", action="store_true",
                        help="check the microphone and speech recognition (nothing goes to the model)")
    parser.add_argument("--model", help="with --mic-test: Whisper model to try, e.g. distil-large-v3.5")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="with --mic-test: where Whisper runs")
    parser.add_argument("--voice-test", action="store_true", help="hear the voice options")
    parser.add_argument("--voices", help="with --voice-test: voices to compare, separated by '|'; "
                                         "a blend uses commas, e.g. \"bm_george|bm_george:0.6,bm_fable:0.4\"")
    parser.add_argument("--speed", type=float, help="with --voice-test: 1.0 = normal, 1.1 = 10%% faster")
    args = parser.parse_args()
    if args.mic_test:
        mic_test(args.model, args.device)
    elif args.voice_test:
        voice_test(args.voices, args.speed)
    elif args.voice:
        voice_mode()
    else:
        typed_mode()


if __name__ == "__main__":
    main()
