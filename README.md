# J.A.R.V.I.S. (PROJECT RIMA, RIMA-01)

A local voice assistant for Tony's Arch Linux workstation: speech in, a local
language model with plugin tools, speech out. Everything runs on this machine
unless a setting below says otherwise.

| Command | What it does |
|---|---|
| `jarvis` | Typed conversation in the terminal |
| `jarvis --voice` | Voice mode (normally runs as `jarvis-voice.service`) |
| `jarvis --mic-test` | Check the microphone and speech recognition; nothing goes to the model |
| `jarvis --voice-test` | Hear the voice options (`--voices "a|b"`, `--speed 1.1`) |
| `jarvis-update` | Install downloaded J.A.R.V.I.S. files from ~/Downloads |

Talk key: `$mod+t` or left-click the Polybar icon. Press it again while
J.A.R.V.I.S. is thinking or speaking to interrupt.

Settings: `config.toml`. Tools: `tools/` (see `tools/__init__.py`).

## How a request flows

1. **Listening:** Silero VAD (speech detection) decides when you start and stop talking.
2. **Transcribing:** faster-whisper turns the recording into text.
3. **Routing:** nomic-embed-text picks the plugins that match the request.
4. **Thinking:** the model (Ollama) answers, calling tools as needed. The reply is streamed.
5. **Speaking:** Kokoro (or Piper) speaks each sentence as soon as it's written. A writer thread makes the next piece while a player thread plays the current one, so speech flows without gaps.

## Outside code and models

Each integration records what it is, its version, why it was chosen and how to remove it.

| What | Version | Why | Remove |
|---|---|---|---|
| faster-whisper (PyPI) | 1.2.1 (uv.lock) | Speech-to-text; also ships the Silero VAD model used for listening | Core; not removable |
| Silero VAD v6 (inside faster-whisper) | bundled | Detects speech regardless of mic volume; replaced the loudness threshold in 2026-10 | Core |
| piper-tts (PyPI, GPL-3.0, now from OHF-Voice/piper1-gpl) | 1.8.0 | Fallback voice, fast on CPU | Core |
| kokoro-onnx (PyPI, MIT; model Apache-2.0; its phonemizer dependency is GPL-3.0) | 0.6.1, model files `model-files-v1.0` (full-precision `kokoro-v1.0.onnx`) | Default voice since 2026-10-16: much more natural than Piper, and the full model is as fast as the int8 one on this CPU. Falls back to Piper automatically if it can't load | `uv remove kokoro-onnx`, delete `voices/kokoro-v1.0.onnx` and `voices/voices-v1.0.bin`, set `tts_engine = "piper"` |
| nvidia-cublas-cu12 (PyPI) | 12.9 (uv.lock), installed 2026-10-16 | CUDA 12 math library so Whisper can run on the GPU; CTranslate2 4.8 needs CUDA 12, not Arch's CUDA 13 | `uv remove nvidia-cublas-cu12`, set `whisper_device = "cpu"` |

## Not in git

`.venv/`, `voices/`, `.backups/`, history (`~/.local/state/jarvis/history/`),
Google credentials (`~/.config/rima/`), the last recording (`$XDG_RUNTIME_DIR/jarvis-last.wav`).
