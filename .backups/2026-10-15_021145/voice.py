"""
voice.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
Ears and mouth:
  Ears   microphone -> text   (faster-whisper)
  Mouth  text -> speech       (Piper)
Both run on the CPU so the GPU stays free for the language model.
"""
import re
import subprocess
from collections import deque

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono audio
BLOCK = 480          # 30 ms of audio per read
BLOCKS_PER_SEC = SAMPLE_RATE / BLOCK

# Words Whisper might not know. Listing them makes it spell them correctly.
VOCAB = (
    "J.A.R.V.I.S., Jarvis, Taskwarrior, RIMA, RDC, RAIL, AOC, KIA, SOC, POD, "
    "Obsidian, Ollama, Datsun 280Z"
)


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
    return re.sub(r"[*_#`]", "", text)


def _rms(block):
    """Loudness of one block of audio."""
    return float(np.sqrt(np.mean(np.square(block))))


class Ears:
    def __init__(self, model="small.en", threads=8):
        from faster_whisper import WhisperModel  # slow import, so only load when needed

        # int8 = smaller and faster on CPU with almost no accuracy loss
        self.model = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)

    def record(self, start_timeout=5.0, end_silence=1.0, max_seconds=15.0):
        """
        Record until you stop talking.
        Returns the audio, or None if you didn't start talking within start_timeout.
        """
        preroll = deque(maxlen=int(0.3 * BLOCKS_PER_SEC))  # keeps your first syllable
        audio = []
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK) as stream:
            # Listen to the room for 0.3 s. Speech must be clearly louder than this.
            noise = [_rms(stream.read(BLOCK)[0]) for _ in range(int(0.3 * BLOCKS_PER_SEC))]
            threshold = max(0.01, 3.0 * float(np.median(noise)))

            waited = 0
            while not audio:  # wait for speech to start
                block = stream.read(BLOCK)[0].copy()
                if _rms(block) > threshold:
                    audio = [*preroll, block]
                else:
                    preroll.append(block)
                    waited += 1
                    if waited > start_timeout * BLOCKS_PER_SEC:
                        return None

            silent = 0
            while True:  # record until a pause or the time limit
                block = stream.read(BLOCK)[0].copy()
                audio.append(block)
                silent = silent + 1 if _rms(block) < threshold else 0
                if silent > end_silence * BLOCKS_PER_SEC or len(audio) > max_seconds * BLOCKS_PER_SEC:
                    break
        return np.concatenate(audio).flatten()

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio, language="en", vad_filter=True, initial_prompt=VOCAB
        )
        return " ".join(s.text.strip() for s in segments).strip()


class Mouth:
    def __init__(self, voice_path, length_scale=1.0):
        from piper import PiperVoice, SynthesisConfig

        self.voice = PiperVoice.load(str(voice_path))
        # length_scale: 1.0 = normal pace, higher = slower, lower = faster
        self.config = SynthesisConfig(length_scale=length_scale)

    def say(self, text):
        chunks = list(self.voice.synthesize(_for_speech(text), syn_config=self.config))
        if not chunks:
            return
        audio = np.frombuffer(b"".join(c.audio_int16_bytes for c in chunks), dtype=np.int16)
        sd.play(audio, chunks[0].sample_rate)
        sd.wait()


def beep(freq=880, ms=120, volume=0.2):
    """Short tone so you know when J.A.R.V.I.S. starts and stops listening."""
    t = np.arange(int(SAMPLE_RATE * ms / 1000)) / SAMPLE_RATE
    tone = volume * np.sin(2 * np.pi * freq * t)
    fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.01)  # avoid clicks
    sd.play((tone * fade).astype(np.float32), SAMPLE_RATE)
    sd.wait()


def notify(title, body=""):
    """Desktop notification through dunst. Each one replaces the last."""
    try:
        subprocess.run(
            ["notify-send", "-a", "J.A.R.V.I.S.", "-t", "5000",
             "-h", "string:x-dunst-stack-tag:jarvis", title, body],
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
