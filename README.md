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
| `jarvis-stage` | Opened by J.A.R.V.I.S. (not by you): a terminal with a prepared command typed in but not run. Enter runs it, Ctrl+C cancels |
| `uv run python -m tools.ask_claude test` | Check the Claude API key and connection (costs a fraction of a cent); `usage` shows this month's spend |

Talk key: `$mod+t` or left-click the Polybar icon. Press it again while
J.A.R.V.I.S. is thinking or speaking to interrupt.

J.A.R.V.I.S. never runs shell commands himself: he prepares them in a terminal
(`stage_command`) and you press Enter. Every tool call is written to an action
log ("what did you do today?"). His manner and humor are set under `[personality]`. Power actions that close apps
(log out, reboot, shut down) need your "yes" in a second request plus a countdown
the talk key cancels; "reboot into Windows" is a staged command (next boot only).

Settings: `config.toml`. Tools: `tools/` (see `tools/__init__.py`). Outside tool
servers (MCP): `tools/mcp.py` and `[tools.mcp.servers.*]` in `config.toml`.

## How a request flows

1. **Listening:** Silero VAD (speech detection) decides when you start and stop talking.
2. **Transcribing:** faster-whisper turns the recording into text.
3. **Routing:** nomic-embed-text picks the tool groups that match the request (one group per plugin, and one per MCP server).
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
| mcp (PyPI, MIT; official MCP Python SDK) | >=2.3.0 (uv.lock) | Lets J.A.R.V.I.S. use tools from MCP servers (`tools/mcp.py`) | `uv remove mcp`, take `"mcp"` out of `[tools] enabled` |
| mcp-server-git (PyPI, MIT; official reference server) | 2026.10.10, installed with `uv tool install` | Read-only git tools for repos in ~/rima (status, log, diffs, show, branches) | `uv tool uninstall mcp-server-git`, delete `[tools.mcp.servers.git]` |
| mcp-server-fetch (PyPI, MIT; official reference server) | 2026.10.10, installed with `uv tool install` | Reads public web pages as text (refuses local/private addresses; respects robots.txt) | `uv tool uninstall mcp-server-fetch`, delete `[tools.mcp.servers.fetch]` |
| anthropic (PyPI, MIT; official Claude SDK) | >=1.15.0 (uv.lock) | `ask_claude`: questions to Claude only when Tony says "Claude" or agrees; web search and sandbox code run at Anthropic. Key in `~/.config/rima/anthropic-key` (600), local monthly budget + Console spend limit | `uv remove anthropic`, take `"ask_claude"` out of `[tools] enabled`, delete the key file |
| nvidia-cublas-cu12 (PyPI) | 12.9 (uv.lock), installed 2026-10-16 | CUDA 12 math library so Whisper can run on the GPU; CTranslate2 4.8 needs CUDA 12, not Arch's CUDA 13 | `uv remove nvidia-cublas-cu12`, set `whisper_device = "cpu"` |

## Not in git

`.venv/`, `voices/`, `.backups/`, history (`~/.local/state/jarvis/history/`),
the action log (`~/.local/state/jarvis/actions/`, pruned with the history), API keys and credentials (`~/.config/rima/`), Claude's saved answers (`~/.local/share/jarvis/claude/`),
Claude spend tracking (`~/.local/state/jarvis/claude-usage.json`), the last recording (`$XDG_RUNTIME_DIR/jarvis-last.wav`).
