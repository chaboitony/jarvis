# jarvis-install: tools
"""
tools/system.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
How the workstation is doing: CPU, memory, GPU, disks, temperatures, uptime,
loaded AI models, and pending updates. Read-only.

Sources: /proc (CPU, memory, uptime), nvidia-smi (GPU), lm_sensors (CPU
temperature), Python's disk_usage (disks), Ollama's API (loaded models), and
checkupdates from pacman-contrib + yay (updates; slow, so only when asked).
Anything missing is reported as unavailable instead of failing the whole report.
"""
import json
import os
import shutil
import subprocess
import time

from tools import tool

PROMPT = """System: use system_status for questions about the computer's health: CPU, memory, GPU, VRAM, disk space, temperatures, uptime, loaded AI models or pending updates.
Pick the matching part; use overview for general "how's the system" questions. Checking updates is slow, so only do it when asked."""

EXAMPLES = [
    "how is the system doing", "how hot is the GPU", "how much VRAM is in use", "how much disk space is left",
    "are there any updates", "what's the CPU load", "how long has the computer been on", "which models are loaded",
]

PARTS = ["overview", "cpu", "memory", "gpu", "disk", "temperature", "uptime", "models", "updates"]
DISKS = [("/", "Root"), ("/home", "Home")]


def _disks():
    """(name, usage) for each disk in DISKS, skipping a folder on the same partition as one already listed."""
    seen, found = set(), []
    for path, name in DISKS:
        try:
            device = os.stat(path).st_dev
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        if device not in seen:
            seen.add(device)
            found.append((name, usage))
    return found
WARN_DISK, WARN_MEMORY, WARN_GPU_TEMP = 90, 90, 85  # percent, percent, °C


def _run(cmd, timeout=10):
    """Run a command; return its output, or None if it's missing or fails."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _gb(n_bytes):
    return f"{n_bytes / 1e9:.0f} GB" if n_bytes >= 10e9 else f"{n_bytes / 1e9:.1f} GB"


# --------------------------------------------------------------------------
# Readings (each returns numbers or None; the sentences are built below)
# --------------------------------------------------------------------------
def _cpu_busy(sample=0.3):
    """Percent of CPU time in use, measured over a short sample of /proc/stat."""
    def snapshot():
        with open("/proc/stat") as f:
            values = [int(v) for v in f.readline().split()[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
        return idle, sum(values)
    try:
        idle1, total1 = snapshot()
        time.sleep(sample)
        idle2, total2 = snapshot()
    except (OSError, ValueError, IndexError):
        return None
    total = total2 - total1
    return 100 * (1 - (idle2 - idle1) / total) if total else None


def _memory():
    """(used bytes, total bytes) from /proc/meminfo."""
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, value = line.split(":", 1)
                info[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        return None
    if "MemTotal" not in info or "MemAvailable" not in info:
        return None
    return info["MemTotal"] - info["MemAvailable"], info["MemTotal"]


def _gpu():
    """dict with name, load, vram_used, vram_total (MiB), temp (°C), power (W), or None."""
    out = _run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"])
    if not out:
        return None
    fields = [f.strip() for f in out.splitlines()[0].split(",")]
    if len(fields) < 6:
        return None

    def num(x):
        try:
            return float(x)
        except ValueError:
            return None
    return {"name": fields[0], "load": num(fields[1]), "vram_used": num(fields[2]),
            "vram_total": num(fields[3]), "temp": num(fields[4]), "power": num(fields[5])}


def _cpu_temp():
    """CPU package temperature in °C from lm_sensors, or None."""
    out = _run(["sensors", "-j"])
    if not out:
        return None
    try:
        data = json.loads(out)
    except ValueError:
        return None
    for chip, readings in data.items():
        if not chip.startswith(("coretemp", "k10temp", "zenpower")):
            continue
        for label, values in readings.items():
            if isinstance(values, dict) and label.startswith(("Package id", "Tctl", "Tdie")):
                for key, value in values.items():
                    if key.endswith("_input"):
                        return float(value)
    return None


def _uptime():
    try:
        with open("/proc/uptime") as f:
            seconds = float(f.read().split()[0])
    except (OSError, ValueError):
        return None
    days, rest = divmod(int(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    parts = ([f"{days} day{'s' * (days != 1)}"] if days else []) + \
            ([f"{hours} hour{'s' * (hours != 1)}"] if hours else []) + \
            ([f"{minutes} minute{'s' * (minutes != 1)}"] if not days else [])
    return " ".join(parts) or "less than a minute"


# --------------------------------------------------------------------------
# Sentences
# --------------------------------------------------------------------------
def _cpu_text():
    busy = _cpu_busy()
    load1, _, _ = os.getloadavg()
    text = f"CPU: {busy:.0f}% busy" if busy is not None else "CPU usage unavailable"
    text += f", load average {load1:.2f} on {os.cpu_count()} threads"
    temp = _cpu_temp()
    return text + (f", {temp:.0f}°C." if temp is not None else ".")


def _memory_text():
    mem = _memory()
    if not mem:
        return "Memory: unavailable."
    used, total = mem
    return f"Memory: {_gb(used)} of {_gb(total)} in use ({100 * used / total:.0f}%)."


def _gpu_text():
    g = _gpu()
    if not g:
        return "GPU: unavailable (nvidia-smi didn't answer)."
    text = f"GPU ({g['name']}): {g['load']:.0f}% busy" if g["load"] is not None else f"GPU ({g['name']})"
    if g["vram_used"] is not None and g["vram_total"]:
        text += f", VRAM {g['vram_used'] / 1024:.1f} of {g['vram_total'] / 1024:.0f} GB"
    if g["temp"] is not None:
        text += f", {g['temp']:.0f}°C"
    if g["power"] is not None:
        text += f", {g['power']:.0f} W"
    return text + "."


def _disk_text():
    parts = []
    for name, usage in _disks():
        parts.append(f"{name} {_gb(usage.used)} of {_gb(usage.total)} used ({100 * usage.used / usage.total:.0f}%), "
                     f"{_gb(usage.free)} free")
    return ("Disks: " + "; ".join(parts) + ".") if parts else "Disks: unavailable."


def _temperature_text():
    cpu, g = _cpu_temp(), _gpu()
    parts = []
    if cpu is not None:
        parts.append(f"CPU {cpu:.0f}°C")
    if g and g["temp"] is not None:
        parts.append(f"GPU {g['temp']:.0f}°C")
    return ("Temperatures: " + ", ".join(parts) + ".") if parts else "Temperatures: unavailable."


def _models_text():
    try:
        import ollama
        models = ollama.ps().models
    except Exception:
        return "AI models: Ollama didn't answer."
    if not models:
        return "AI models: none loaded."
    described = []
    for m in models:
        size, vram = m.size or 0, m.size_vram or 0
        where = "fully on the GPU" if size and vram >= size else (
            f"{100 * vram / size:.0f}% on the GPU" if size else "location unknown")
        described.append(f"{m.model} ({_gb(size)}, {where})")
    return "AI models loaded: " + "; ".join(described) + "."


def _updates_text():
    if shutil.which("checkupdates") is None:
        official = "Official package updates can't be checked: install pacman-contrib (sudo pacman -S pacman-contrib)."
    else:
        out = _run(["checkupdates"], timeout=90)
        count = len(out.splitlines()) if out else 0  # checkupdates exits 2 when there are none
        official = f"{count} official package update{'s' * (count != 1)} available."
    aur = ""
    if shutil.which("yay"):
        out = _run(["yay", "-Qua"], timeout=90)
        count = len(out.splitlines()) if out else 0
        aur = f" {count} AUR update{'s' * (count != 1)}."
    return "Updates: " + official + aur


# --------------------------------------------------------------------------
# Tool
# --------------------------------------------------------------------------
@tool
def system_status(part: str = "overview") -> str:
    """Check the computer's health: CPU, memory, GPU and VRAM, disks, temperatures, uptime, loaded AI models or updates.

    Args:
        part: overview, cpu, memory, gpu, disk, temperature, uptime, models or updates
    """
    part = str(part or "overview").strip().lower()
    part = {"ram": "memory", "vram": "gpu", "graphics": "gpu", "storage": "disk", "disks": "disk",
            "space": "disk", "temp": "temperature", "temperatures": "temperature", "heat": "temperature",
            "ollama": "models", "model": "models", "update": "updates", "all": "overview",
            "processor": "cpu", "load": "cpu"}.get(part, part)
    if part not in PARTS:
        raise ValueError(f"Unknown part '{part}'. Use one of: {', '.join(PARTS)}.")
    uptime = _uptime()
    readers = {
        "cpu": _cpu_text, "memory": _memory_text, "gpu": _gpu_text, "disk": _disk_text,
        "temperature": _temperature_text, "models": _models_text, "updates": _updates_text,
        "uptime": lambda: f"Uptime: {uptime or 'unavailable'}.",
    }
    if part != "overview":
        return readers[part]()
    return " ".join([_cpu_text(), _memory_text(), _gpu_text(), _disk_text(),
                     f"Uptime: {uptime or 'unavailable'}."])


def briefing():
    """Only problems: a nearly full disk, low memory or a hot GPU. Nothing when all is well."""
    warnings = []
    for name, usage in _disks():
        percent = 100 * usage.used / usage.total
        if percent >= WARN_DISK:
            warnings.append(f"{name} disk is {percent:.0f}% full ({_gb(usage.free)} free)")
    mem = _memory()
    if mem and 100 * mem[0] / mem[1] >= WARN_MEMORY:
        warnings.append(f"memory is {100 * mem[0] / mem[1]:.0f}% used")
    g = _gpu()
    if g and g["temp"] is not None and g["temp"] >= WARN_GPU_TEMP:
        warnings.append(f"GPU is at {g['temp']:.0f}°C")
    return ("System warning: " + "; ".join(warnings) + ".") if warnings else None
