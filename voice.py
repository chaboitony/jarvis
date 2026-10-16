# jarvis-install: .
"""
voice.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
Ears and mouth:
  Ears    microphone -> text. Silero VAD (a small speech-detection network that
          ships with faster-whisper) decides when you start and stop talking;
          faster-whisper turns the recording into text.
  Mouth   text -> speech on background threads, so it can start talking while
          the model is still writing, flows without gaps, and can be
          interrupted. Engine: Kokoro (natural) or Piper (fallback).
  MediaPause  pauses music/videos while you talk, then resumes them.
Also: beep() and notify() (desktop notifications through dunst).

Everything here runs on the CPU unless [voice] whisper_device = "cuda".
"""
import html
import os
import queue
import re
import subprocess
import threading
import time
import traceback
import wave
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000                    # Whisper and Silero both expect 16 kHz mono
WINDOW = 512                           # 32 ms: the block size Silero VAD works on
WINDOWS_PER_SEC = SAMPLE_RATE / WINDOW
RUN_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
LAST_RECORDING = RUN_DIR / "jarvis-last.wav"   # in RAM, readable only by you

# Words Whisper might not know. Keep this list SHORT: every word here nudges
# Whisper toward hearing it, so a long list of acronyms makes it "hear" them in
# ordinary speech. Override with [voice] hotwords in config.toml.
DEFAULT_HOTWORDS = "J.A.R.V.I.S., Taskwarrior, Obsidian, Ollama"

# Speech detection thresholds (Silero gives a speech probability 0..1 per window)
START_PROB = 0.5   # this sure, for START_WINDOWS windows in a row, = you started talking
START_WINDOWS = 2  # 64 ms; stops a click or a cough from starting a recording
KEEP_PROB = 0.35   # once talking, anything above this still counts as speech

# How to say words the voice gets wrong: written form -> spelling that sounds right.
# Only affects what is spoken; the text on screen stays the same.
PRONUNCIATIONS = {
    "RIMA": "Ryma",   # RYE-ma, not REE-ma
    "Rima": "Ryma",
    "J.A.R.V.I.S.": "Jarvis",
    "J.A.R.V.I.S": "Jarvis",
}


def _for_speech(text):
    """Swap in pronunciation spellings and drop markdown symbols."""
    for written, spoken in PRONUNCIATIONS.items():
        text = re.sub(rf"(?<!\w){re.escape(written)}(?!\w)", spoken, text)
    return re.sub(r"[*_#`]", "", text).strip()


def _db(value):
    """Loudness in dBFS: 0 = the loudest possible, -20 = a tenth of that, -60 = near silence."""
    return 20 * np.log10(max(float(value), 1e-6))


def _rms(block):
    return float(np.sqrt(np.mean(np.square(block))))


# --------------------------------------------------------------------------
# Choosing audio devices
# --------------------------------------------------------------------------
def list_input_devices():
    """[(index, name, default sample rate)] for every device with a microphone input."""
    found = []
    for index, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            found.append((index, dev["name"], int(dev.get("default_samplerate", 0))))
    return found


def resolve_input_device(setting):
    """
    [voice] input_device -> what sounddevice wants.
    Empty = the system default (the same microphone Discord uses unless you
    picked another one there). A number = that device index. Text = the first
    input device whose name contains it, ignoring case.
    """
    if setting in (None, "", "default"):
        return None
    if isinstance(setting, int):
        return setting
    wanted = str(setting).lower()
    for index, name, _ in list_input_devices():
        if wanted in name.lower():
            return index
    raise ValueError(f"No microphone matches input_device = \"{setting}\". "
                     "Run `jarvis --mic-test` to see the names.")


# --------------------------------------------------------------------------
# Speech detection
# --------------------------------------------------------------------------
class SpeechDetector:
    """
    Silero VAD, fed one 32 ms window at a time.
    It's a small recurrent network: h and c are its memory of the audio so far,
    carried from one window to the next. It answers "how likely is this speech?"
    and, unlike a loudness threshold, doesn't care how loud your mic is, and
    ignores fans, beeps and keyboard clicks fairly well.
    """

    def __init__(self):
        from faster_whisper.vad import get_vad_model  # ships with faster-whisper

        self.session = get_vad_model().session
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.context = np.zeros(64, np.float32)  # the end of the previous window

    def __call__(self, window):
        window = np.asarray(window, np.float32)
        x = np.concatenate([self.context, window])[None, :]
        prob, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.context = window[-64:]
        return float(np.asarray(prob).reshape(-1)[0])


# --------------------------------------------------------------------------
# Ears
# --------------------------------------------------------------------------
def _load_whisper(model, device, compute_type, threads):
    """
    Load faster-whisper. If the GPU was asked for but can't be used (CUDA 12's
    libcublas missing, no GPU), fall back to the CPU and say why. A tiny test
    transcription runs here, because CUDA problems only show up on first use.
    Returns (model, device actually used, warning or None).
    """
    from faster_whisper import WhisperModel  # slow import, so only load when needed

    if device == "cuda":
        try:
            m = WhisperModel(model, device="cuda", compute_type=compute_type or "int8_float16")
            list(m.transcribe(np.zeros(SAMPLE_RATE, np.float32), language="en")[0])
            return m, "cuda", None
        except Exception as e:
            warning = (f"Whisper couldn't use the GPU ({type(e).__name__}: {e}); using the CPU. "
                       "See the [voice] notes in config.toml.")
            print(f"  [voice] {warning}")
    else:
        warning = None
    cpu_type = compute_type if device != "cuda" and compute_type else "int8"
    m = WhisperModel(model, device="cpu", compute_type=cpu_type, cpu_threads=threads)
    list(m.transcribe(np.zeros(SAMPLE_RATE, np.float32), language="en")[0])  # warm up
    return m, "cpu", warning


class Ears:
    def __init__(self, settings=None):
        s = settings or {}
        self.model_name = str(s.get("whisper_model", "small.en"))
        self.hotwords = str(s.get("hotwords", DEFAULT_HOTWORDS)).strip() or None
        self.beam_size = max(1, int(s.get("beam_size", 5)))
        self.input_device = resolve_input_device(s.get("input_device"))
        self.end_silence = float(s.get("end_silence", 1.0))
        self.short_request_pause = float(s.get("short_request_pause", 1.5))
        self.start_timeout = float(s.get("start_timeout", 6.0))
        self.max_seconds = float(s.get("max_seconds", 20.0))
        self.save_last = bool(s.get("save_last_recording", True))
        self.detector = SpeechDetector()
        self.model, self.device, self.warning = _load_whisper(
            self.model_name, str(s.get("whisper_device", "cpu")).lower(),
            s.get("whisper_compute"), int(s.get("cpu_threads", 8)))
        self.last = {}  # numbers about the last recording, for jarvis --mic-test

    def record(self):
        """
        Record one request. Starts when you start talking, ends after a pause.
        Returns the audio (cleaned up for Whisper), or None if you didn't start
        talking within start_timeout seconds.

        Pause length: end_silence seconds normally, but short_request_pause when
        you've said less than 2 seconds so far, so "set a timer for… twenty
        minutes" isn't cut off after "for".
        """
        preroll = deque(maxlen=int(0.4 * WINDOWS_PER_SEC))  # keeps your first syllable
        windows, noise, speech_levels = [], [], []
        started, run, waited, silent, speech_windows, last_speech = False, 0, 0, 0, 0, 0
        reason = "pause"
        self.detector.reset()
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=WINDOW, device=self.input_device) as stream:
            while True:
                block = stream.read(WINDOW)[0][:, 0].copy()
                prob = self.detector(block)
                if not started:
                    preroll.append(block)
                    if prob < 0.2:
                        noise.append(_rms(block))
                    run = run + 1 if prob >= START_PROB else 0
                    if run >= START_WINDOWS:
                        started, windows, speech_windows = True, list(preroll), run
                        last_speech = len(windows) - 1
                    else:
                        waited += 1
                        if waited > self.start_timeout * WINDOWS_PER_SEC:
                            self.last = {"result": "no speech heard", "noise_db": _db(np.median(noise or [0]))}
                            return None
                    continue

                windows.append(block)
                if prob >= KEEP_PROB:
                    silent, last_speech = 0, len(windows) - 1
                    if prob >= START_PROB:
                        speech_windows += 1
                        speech_levels.append(_rms(block))
                else:
                    silent += 1
                spoken = speech_windows / WINDOWS_PER_SEC
                limit = self.short_request_pause if spoken < 2.0 else self.end_silence
                if silent >= limit * WINDOWS_PER_SEC:
                    break
                if len(windows) >= self.max_seconds * WINDOWS_PER_SEC:
                    reason = "time limit"
                    break

        # Keep 0.3 s after the last speech; the rest of the pause is just silence
        keep = min(len(windows), last_speech + 1 + int(0.3 * WINDOWS_PER_SEC))
        audio = np.concatenate(windows[:keep])
        peak = float(np.abs(audio).max()) or 1e-6
        # Even out the volume: a quiet mic is raised so its loudest point is at
        # -1 dB, but never by more than 30 dB (that would just amplify hiss).
        gain = min(0.9 / peak, 31.6)
        self.last = {
            "result": f"stopped at a {reason}",
            "seconds": len(audio) / SAMPLE_RATE,
            "speech_seconds": speech_windows / WINDOWS_PER_SEC,
            "peak_db": _db(peak),
            "speech_db": _db(np.median(speech_levels or [0])),
            "noise_db": _db(np.median(noise or [0])),
            "gain_db": _db(gain),
        }
        audio = (audio * gain).astype(np.float32)
        if self.save_last:
            save_wav(LAST_RECORDING, audio)
        return audio

    def transcribe(self, audio):
        """Audio -> text. Also fills self.last with timing and confidence."""
        started = time.monotonic()
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=self.beam_size, hotwords=self.hotwords,
            vad_filter=False,                  # record() already kept only the speech
            condition_on_previous_text=False,  # each request stands alone
            without_timestamps=True,
        )
        kept, scores = [], []
        for seg in segments:
            # Skip what Whisper itself thinks is not speech (it can invent
            # "Thank you." from noise)
            if seg.no_speech_prob > 0.8 and seg.avg_logprob < -1.0:
                continue
            kept.append(seg.text.strip())
            scores.append(seg.avg_logprob)
        self.last["transcribe_seconds"] = time.monotonic() - started
        # avg_logprob -> a rough 0..1 confidence (1 = Whisper was sure of every word)
        self.last["confidence"] = float(np.exp(np.mean(scores))) if scores else 0.0
        return " ".join(kept).strip()


def save_wav(path, audio, rate=SAMPLE_RATE):
    """Save float audio (-1..1) as a 16-bit WAV file only you can read."""
    try:
        data = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        temp = Path(str(path) + ".tmp")
        with wave.open(str(temp), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(data)
        os.chmod(temp, 0o600)
        temp.replace(path)
    except OSError:
        pass


# --------------------------------------------------------------------------
# Mouth
# --------------------------------------------------------------------------
def _trim_silence(audio, rate, threshold_db=-45, keep=0.02):
    """Cut quiet audio off both ends (keeping 20 ms), so pauses between sentences are even."""
    if len(audio) == 0:
        return audio
    frame = max(1, int(rate * 0.01))
    usable = len(audio) // frame * frame
    if usable == 0:
        return audio
    loud = np.sqrt((audio[:usable].reshape(-1, frame) ** 2).mean(1))
    voiced = np.nonzero(loud > max(float(loud.max()), 1e-6) * 10 ** (threshold_db / 20))[0]
    if len(voiced) == 0:
        return audio[:0]
    start = max(0, voiced[0] * frame - int(keep * rate))
    end = min(len(audio), (voiced[-1] + 1) * frame + int(keep * rate))
    return audio[start:end]


def _fade(audio, rate, seconds=0.005):
    """5 ms fade in and out: stops clicks where audio pieces meet."""
    n = min(len(audio) // 2, int(rate * seconds))
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        audio = audio.copy()
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio


class _Piper:
    name = "Piper"

    def __init__(self, voice_path, speed):
        from piper import PiperVoice, SynthesisConfig

        self.voice = PiperVoice.load(str(voice_path))
        # length_scale: 1.0 = normal pace, higher = slower
        self.config = SynthesisConfig(length_scale=1.0 / speed)

    def render(self, text):
        chunks = list(self.voice.synthesize(text, syn_config=self.config))
        if not chunks:
            return np.zeros(0, np.float32), 22050
        audio = np.concatenate([c.audio_float_array.astype(np.float32) for c in chunks])
        return audio, chunks[0].sample_rate


class _Kokoro:
    """
    Kokoro: a small (82M-parameter) speech model that sounds much more natural
    than Piper. Needs `uv add kokoro-onnx` and its two model files.
    A voice can be one name ("bm_george") or a blend ("bm_george:0.7,bm_lewis:0.3"):
    each voice is a list of numbers describing how it sounds, and a blend mixes them.
    """
    name = "Kokoro"

    def __init__(self, model_path, voices_path, voice, speed, pause):
        try:
            from kokoro_onnx import Kokoro
        except ImportError:
            raise RuntimeError("the kokoro-onnx package isn't installed (cd ~/rima/jarvis && uv add kokoro-onnx)")
        for path in (model_path, voices_path):
            if not Path(path).exists():
                raise RuntimeError(f"Kokoro file not found: {path} (see the [voice] notes in config.toml)")
        self.kokoro = Kokoro(str(model_path), str(voices_path))
        self.speed, self.pause = speed, pause
        self.set_voice(voice)

    def set_voice(self, voice):
        """Choose a voice or a blend. Raises ValueError with the available names if one is wrong."""
        known = set(self.kokoro.get_voices())
        parts = []
        for item in str(voice).split(","):
            name, _, weight = item.strip().partition(":")
            if name not in known:
                english = sorted(v for v in known if v[:2] in ("am", "af", "bm", "bf"))
                raise ValueError(f"Unknown Kokoro voice '{name}'. English voices: {', '.join(english)}")
            parts.append((name, float(weight or 1.0)))
        total = sum(w for _, w in parts)
        self.style = sum(self.kokoro.get_voice_style(n) * (w / total) for n, w in parts)
        self.voice = voice
        # bf_/bm_ voices are British; the accent comes from the pronunciation rules
        self.lang = "en-gb" if parts[0][0].startswith("b") else "en-us"

    def render(self, text):
        audio, rate = self.kokoro.create(text, voice=self.style, speed=self.speed, lang=self.lang,
                                         sentence_pause=self.pause, clause_pause=0.1)
        return np.asarray(audio, np.float32), rate


def load_voice_engine(settings, base_dir):
    """
    The engine named by [voice] tts_engine. If Kokoro can't load, falls back to
    Piper so J.A.R.V.I.S. still talks. Returns (engine, warning or None).
    """
    s = settings or {}
    speed = float(s.get("speed", 1.0))
    warning = None
    if str(s.get("tts_engine", "piper")).lower() == "kokoro":
        try:
            return _Kokoro(base_dir / s.get("kokoro_model", "voices/kokoro-v1.0.onnx"),
                           base_dir / s.get("kokoro_voices", "voices/voices-v1.0.bin"),
                           str(s.get("kokoro_voice", "bm_george")), speed,
                           float(s.get("sentence_pause", 0.25))), None
        except (RuntimeError, ValueError) as e:
            warning = f"Kokoro voice unavailable ({e}); using Piper."
            print(f"  [voice] {warning}")
    voice_file = base_dir / s.get("voice_file", "voices/en_GB-alan-medium.onnx")
    if not voice_file.exists():
        raise SystemExit(f"Voice file not found: {voice_file}\nDownload it first (see setup steps).")
    return _Piper(voice_file, speed), warning


class Mouth:
    """
    Speaks with two background threads, so speech flows without gaps:
      writer  turns text into audio. If several sentences are waiting, it does
              them together, so the intonation carries across them.
      player  plays the audio, with an even pause between pieces, while the
              writer is already preparing the next piece.
    speak() queues a sentence and returns at once; stop() cuts it off within ~50 ms.
    """

    SLICE_SECONDS = 0.05   # how often playback checks whether it was told to stop
    BATCH_CHARS = 400      # most text the writer does in one go

    def __init__(self, settings, base_dir, engine=None):
        s = settings or {}
        if engine is None:
            engine, self.warning = load_voice_engine(s, base_dir)
        else:
            self.warning = None
        self.engine = engine
        self.pause = float(s.get("sentence_pause", 0.25))   # seconds between sentences
        self.volume = float(s.get("volume", 1.0))
        self._text = queue.Queue()
        self._audio = queue.Queue()
        self._lock = threading.Lock()
        self._pending = 0              # sentences queued but not yet played (or dropped)
        self._generation = 0           # bumped by stop(); older work is dropped
        self._idle = threading.Event()
        self._idle.set()
        self._playing = threading.Event()  # set while sound is being sent to the speakers
        self.heard = []                # text of pieces that actually started playing
        self._last_end = 0.0           # when the previous piece finished playing
        self._busy_until = 0.0         # when the piece playing now will end
        self._queued_seconds = 0.0     # audio made but not yet playing
        self._render_speed = 0.3       # seconds of work per second of speech (measured)
        self._writer = threading.Thread(target=self._write_loop, name="mouth-writer", daemon=True)
        self._player = threading.Thread(target=self._play_loop, name="mouth-player", daemon=True)
        self._writer.start()
        self._player.start()

    # ---- what the rest of J.A.R.V.I.S. calls ------------------------------
    def speak(self, text):
        """Queue one sentence (or more) to be spoken. Returns immediately."""
        text = _for_speech(text)
        if not text:
            return
        with self._lock:
            self._pending += 1
            self._idle.clear()
            self._text.put((self._generation, text))

    def wait(self, cancel=None):
        """Block until everything queued has been spoken. Returns False if `cancel` was set first."""
        while not self._idle.wait(0.05):
            if cancel is not None and cancel.is_set():
                return False
        return True

    def say(self, text):
        """Speak and wait (used for "J.A.R.V.I.S. online")."""
        self.speak(text)
        self.wait()

    def stop(self):
        """
        Stop talking now and forget anything still queued. Returns once the
        speakers are quiet (~50 ms); a piece the writer is still making is
        thrown away when it's done, without making stop() wait for it.
        """
        with self._lock:
            self._generation += 1
            self._pending = 0
            self._idle.set()
        deadline = time.monotonic() + 1.0
        while self._playing.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)

    def close(self):
        self.stop()
        self._text.put(None)

    # ---- shared bookkeeping ------------------------------------------------
    def _stale(self, generation):
        return generation != self._generation

    def _done(self, count, generation):
        with self._lock:
            if generation != self._generation:
                return  # stop() already wrote these off
            self._pending = max(0, self._pending - count)
            if self._pending == 0:
                self._idle.set()

    def _estimate(self, text):
        """Roughly how long making speech for this text will take (about 14 characters per spoken second)."""
        return self._render_speed * len(text) / (14.0 * getattr(self.engine, "speed", 1.0))

    # ---- writer thread: text -> audio --------------------------------------
    def _write_loop(self):
        held = None  # a sentence taken from the queue that belongs to the next batch
        while True:
            item, held = (held, None) if held is not None else (self._text.get(), None)
            if item is None:
                self._audio.put(None)
                return
            generation, text = item
            texts = [text]
            # Sentences said in one go get smoother intonation, but take longer
            # to make. Take more of the waiting sentences only while they'd still
            # be ready before the player runs out of audio; otherwise send this
            # one alone so there's no gap.
            while sum(len(t) for t in texts) < self.BATCH_CHARS:
                ahead = max(0.0, self._busy_until - time.monotonic()) + self._queued_seconds
                try:
                    peek = self._text.queue[0]
                except IndexError:
                    break
                if peek is None or peek[0] != generation:
                    pass  # handled below
                elif self._estimate(" ".join(texts + [peek[1]])) > 0.8 * ahead:
                    break
                try:
                    nxt = self._text.get_nowait()
                except queue.Empty:
                    break
                if nxt is None or nxt[0] != generation:
                    held = nxt if nxt is not None else held
                    if nxt is None:
                        self._text.put(None)  # closing: handle after this batch
                    break
                texts.append(nxt[1])
            if self._stale(generation):
                continue
            joined = " ".join(texts)
            try:
                started = time.monotonic()
                audio, rate = self.engine.render(joined)
                took = time.monotonic() - started
                if len(audio):
                    self._render_speed = 0.7 * self._render_speed + 0.3 * (took / (len(audio) / rate))
                audio = _fade(_trim_silence(audio, rate), rate) * self.volume
            except Exception:
                traceback.print_exc()  # a voice problem shouldn't stop J.A.R.V.I.S.
                self._done(len(texts), generation)
                continue
            with self._lock:
                self._queued_seconds += len(audio) / rate
            self._audio.put((generation, joined, audio, rate, len(texts)))

    # ---- player thread: audio -> speakers ----------------------------------
    def _play_loop(self):
        stream, rate = None, None
        while True:
            try:
                item = self._audio.get(timeout=1.0)
            except queue.Empty:
                if stream is not None:  # idle for a second: release the audio device
                    stream.close()
                    stream, rate = None, None
                continue
            if item is None:
                break
            generation, text, audio, sample_rate, count = item
            with self._lock:
                self._queued_seconds = max(0.0, self._queued_seconds - len(audio) / sample_rate)
            try:
                if self._stale(generation) or len(audio) == 0:
                    continue
                if stream is None or rate != sample_rate:
                    if stream is not None:
                        stream.close()
                    stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16")
                    stream.start()
                    rate = sample_rate
                # An even pause after the previous piece; less if that much time already passed
                gap = self.pause - (time.monotonic() - self._last_end)
                if 0 < gap <= self.pause:
                    stream.write(np.zeros(int(gap * sample_rate), np.int16))
                pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
                self._busy_until = time.monotonic() + len(pcm) / sample_rate
                self._playing.set()
                self.heard.append(text)
                step = int(sample_rate * self.SLICE_SECONDS)
                for i in range(0, len(pcm), step):
                    if self._stale(generation):
                        break
                    stream.write(pcm[i:i + step])
                self._last_end = time.monotonic()
                if self._stale(generation):
                    stream.abort()  # drop what's still in the sound card's buffer
                    stream.close()
                    stream, rate = None, None
            except Exception:
                traceback.print_exc()
                stream, rate = None, None
            finally:
                self._playing.clear()
                self._done(count, generation)
        if stream is not None:
            stream.close()


# --------------------------------------------------------------------------
# Pausing media while you talk
# --------------------------------------------------------------------------
def _playerctl(*args):
    try:
        done = subprocess.run(["playerctl", *args], capture_output=True, text=True, timeout=2)
        return done.stdout.strip() if done.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


class MediaPause:
    """Pauses whatever is playing (Spotify, browser videos…) and later resumes only those."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.paused = []

    def pause(self):
        if not self.enabled or self.paused:
            return
        for player in _playerctl("--list-all").split():
            if _playerctl("--player", player, "status") == "Playing":
                _playerctl("--player", player, "pause")
                self.paused.append(player)

    def resume(self):
        for player in self.paused:
            if _playerctl("--player", player, "status") == "Paused":  # you didn't change it meanwhile
                _playerctl("--player", player, "play")
        self.paused = []

    def forget(self):
        """You asked J.A.R.V.I.S. to control the media itself, so don't undo that."""
        self.paused = []


# --------------------------------------------------------------------------
# Beeps and notifications
# --------------------------------------------------------------------------
def beep(freq=880, ms=120, volume=0.2):
    """Short tone so you know when J.A.R.V.I.S. starts and stops listening."""
    t = np.arange(int(SAMPLE_RATE * ms / 1000)) / SAMPLE_RATE
    tone = volume * np.sin(2 * np.pi * freq * t)
    fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.01)  # avoid clicks
    sd.play((tone * fade).astype(np.float32), SAMPLE_RATE)
    sd.wait()


# Notification settings; [notifications] in config.toml overrides these
NOTIFY = {"seconds": 15, "text_size": "large"}
TEXT_SIZES = {"small", "medium", "large", "x-large", "xx-large"}


def configure_notifications(settings):
    """Apply [notifications] from config.toml, ignoring values that would break the markup."""
    try:
        NOTIFY["seconds"] = max(1.0, float(settings.get("seconds", NOTIFY["seconds"])))
    except (TypeError, ValueError):
        pass
    size = str(settings.get("text_size", NOTIFY["text_size"]))
    if size in TEXT_SIZES:
        NOTIFY["text_size"] = size


def notify(title, body="", seconds=None):
    """
    Desktop notification through dunst. Each one replaces the last.
    The body is shown larger using Pango markup (dunst's default markup = full).
    Text is escaped first, so characters like < and & appear as typed instead of
    being read as markup. dunst shows title markup literally, so the title stays plain.
    """
    body_markup = ""
    if body:
        body_markup = f"<span size='{NOTIFY['text_size']}'>{html.escape(body, quote=False)}</span>"
    milliseconds = int((seconds or NOTIFY["seconds"]) * 1000)
    try:
        subprocess.run(
            ["notify-send", "-a", "J.A.R.V.I.S.", "-t", str(milliseconds),
             "-h", "string:x-dunst-stack-tag:jarvis", title, body_markup],
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
