# jarvis-install: tools
"""
tools/mcp.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
The MCP bridge: lets J.A.R.V.I.S. use tools from MCP servers.

MCP (Model Context Protocol) is a standard way for an AI to use outside tools.
An MCP server is a small program that offers a list of tools (for example
"git_status" or "fetch") and runs them when asked. Thousands exist. This plugin
starts the servers listed in config.toml, keeps them running, and turns each
allowed tool into an ordinary J.A.R.V.I.S. tool, so the usual safety rules apply.

Safety (on top of the Toolbox's own rules):
  * Only tools named in a server's `allow` list are offered to the model.
  * Tools the server marks destructive are never offered, even if allowed.
  * Tools the server doesn't mark read-only count as `writes`.
  * Every result counts as `untrusted` (outside text), so after an MCP tool runs,
    tools that change things are blocked for the rest of that request.
  * `allowed_paths`: any argument named *path* must be inside these folders
    (resolved fully, so "..", "~" and symlinks can't escape).
  * `public_urls_only`: any argument named *url* must be http(s) and not point
    at this computer or the home network.
  * Servers get only a minimal environment (HOME, PATH, USER...); pass more
    with `env_pass` (names only, so secrets never go in config.toml).

Each server is one routing group ("mcp:git"), described by its `prompt` and
`examples`. Settings: [tools.mcp] and [tools.mcp.servers.<name>] in config.toml.
"""
import asyncio
import atexit
import ipaddress
import os
import re
import threading
from pathlib import Path
from urllib.parse import urlparse

PROMPT = ""            # each server group has its own prompt instead
RUNTIME_TOOLS = []     # filled by setup(); read by the Toolbox
GROUPS = {}            # "mcp:<server>" -> {"prompt", "examples"}
STATUS = {}            # server -> "ready (n tools)" or why it isn't
_bridge = None

DESCRIPTION_CHARS = 300   # tool descriptions are trimmed to this; long ones crowd the small model
_SCHEMA_KEYS = ("type", "description", "enum", "items", "properties", "required", "default",
                "minimum", "maximum", "minItems", "maxItems")


# --------------------------------------------------------------------------
# Turning MCP tool descriptions into the shape Ollama wants
# --------------------------------------------------------------------------
def _clean_schema(schema):
    """Keep the parts of a JSON schema the model needs; flatten 'X or null' to X."""
    if not isinstance(schema, dict):
        return {"type": "string"}
    options = schema.get("anyOf") or schema.get("oneOf")
    if options:
        real = [o for o in options if isinstance(o, dict) and o.get("type") != "null"]
        if len(real) == 1:
            merged = {**real[0], **{k: v for k, v in schema.items() if k not in ("anyOf", "oneOf")}}
            return _clean_schema(merged)
        return {"type": "string", **({"description": schema["description"]} if "description" in schema else {})}
    out = {k: v for k, v in schema.items() if k in _SCHEMA_KEYS}
    if "properties" in out:
        out["properties"] = {name: _clean_schema(p) for name, p in out["properties"].items()}
    if "items" in out:
        out["items"] = _clean_schema(out["items"])
    if "type" not in out:
        out["type"] = "object" if "properties" in out else "string"
    if "title" in schema and "description" not in out and out["type"] != "object":
        out["description"] = str(schema["title"])
    return out


def _short(text):
    """First paragraph of a description, at most DESCRIPTION_CHARS long."""
    first = re.split(r"\n\s*\n", (text or "").strip())[0]
    first = " ".join(first.split())
    return first if len(first) <= DESCRIPTION_CHARS else first[:DESCRIPTION_CHARS - 1].rsplit(" ", 1)[0] + "…"


# --------------------------------------------------------------------------
# Checking arguments before they reach a server
# --------------------------------------------------------------------------
def _expand(path):
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def _inside(path, roots):
    """True if path (fully resolved) is one of roots or inside one of them."""
    real = os.path.realpath(_expand(path))
    for root in roots:
        root = os.path.realpath(_expand(root))
        if real == root or real.startswith(root.rstrip(os.sep) + os.sep):
            return True
    return False


def _public_url(url):
    """Refuse anything that isn't a plain http(s) address on the internet."""
    parts = urlparse(str(url))
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("only http:// and https:// web addresses are allowed")
    host = parts.hostname.lower()
    if host in ("localhost", "localhost.localdomain") or host.endswith((".local", ".lan", ".internal", ".home.arpa")):
        raise ValueError(f"{host} is on this computer or the home network, which isn't allowed")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return  # a name like example.com
    if not ip.is_global:
        raise ValueError(f"{host} is a private or local address, which isn't allowed")


def _check_args(server, args):
    """
    Fill defaults, expand ~ in paths, and enforce allowed_paths / public_urls_only.
    A short path like "jarvis" means a folder inside the first allowed folder
    (so "what changed in jarvis" works by voice).
    """
    given = {k: v for k, v in (args if isinstance(args, dict) else {}).items() if v not in (None, "")}
    args = {**server["defaults"], **given}
    for name, value in list(args.items()):
        lname = name.lower()
        if "path" in lname and isinstance(value, str) and value:
            if not value.startswith(("/", "~", "$")) and server["allowed_paths"]:
                value = str(_expand(server["allowed_paths"][0]) / value)
            if server["allowed_paths"] and not _inside(value, server["allowed_paths"]):
                folders = ", ".join(server["allowed_paths"])
                raise ValueError(f"{value} is outside the folders this tool may use ({folders})")
            args[name] = os.path.realpath(_expand(value))
        if lname in ("url", "uri") and server["public_urls_only"]:
            _public_url(value)
    return args


def _result_text(result):
    """An MCP tool result -> plain text for the model."""
    parts = []
    for item in result.content or []:
        if getattr(item, "text", None) is not None:
            parts.append(item.text)
        elif getattr(item, "resource", None) is not None:
            parts.append(getattr(item.resource, "text", None) or f"[file: {getattr(item.resource, 'uri', '')}]")
        else:
            parts.append(f"[{getattr(item, 'type', 'item')} not shown]")
    text = "\n".join(p for p in parts if p).strip()
    if result.is_error:
        raise ValueError(text or "the tool reported an error")
    return text or "Done."


# --------------------------------------------------------------------------
# The bridge: one background thread running every server connection
# --------------------------------------------------------------------------
class Bridge:
    """
    The MCP library is asynchronous (asyncio); the rest of J.A.R.V.I.S. isn't.
    So the connections live on their own thread with its own event loop, and
    tool calls are handed to that loop and waited for.
    """

    def __init__(self, timeout):
        self.timeout = timeout
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="mcp", daemon=True)
        self.thread.start()
        self.clients = {}      # server name -> connected Client
        self.stops = {}        # server name -> asyncio.Event that ends its connection
        self.tasks = {}

    def _run(self, coro, timeout):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def connect(self, name, params, mode):
        """Start one server and wait until it's ready. Returns its tool list."""
        from mcp import Client

        ready = None

        async def keep_open():
            nonlocal ready
            ready = asyncio.get_running_loop().create_future()
            stop = asyncio.Event()
            self.stops[name] = stop
            try:
                async with Client(params, mode=mode, read_timeout_seconds=self.timeout) as client:
                    tools = (await client.list_tools()).tools
                    self.clients[name] = client
                    if not ready.done():
                        ready.set_result(tools)
                    await stop.wait()
            except BaseException as e:  # noqa: BLE001 (report anything, including cancellation)
                if ready is not None and not ready.done():
                    ready.set_exception(e if isinstance(e, Exception) else RuntimeError(repr(e)))
            finally:
                self.clients.pop(name, None)

        async def start():
            self.tasks[name] = asyncio.ensure_future(keep_open())
            while ready is None:
                await asyncio.sleep(0)
            return await asyncio.wait_for(asyncio.shield(ready), self.timeout)

        return self._run(start(), self.timeout + 5)

    def call(self, name, tool, args):
        client = self.clients.get(name)
        if client is None:
            raise ValueError(f"the {name} tool server isn't running; try 'reload tools'")
        from mcp.shared.exceptions import MCPError

        try:
            return self._run(client.call_tool(tool, args), self.timeout + 2)
        except TimeoutError:
            raise ValueError(f"{tool} took longer than {self.timeout:g} seconds")
        except MCPError as e:  # the server refused or timed out: a plain sentence for the model
            raise ValueError(f"the {name} tool server said: {e}")

    def stop(self):
        """End every connection (which stops the server programs) and the thread."""
        async def stop_all():
            for event in self.stops.values():
                event.set()
            tasks = [t for t in self.tasks.values() if not t.done()]
            if tasks:
                await asyncio.wait(tasks, timeout=5)

        if self.loop.is_running():
            try:
                self._run(stop_all(), 8)
            except Exception:
                pass
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=3)


# --------------------------------------------------------------------------
# Plugin hooks
# --------------------------------------------------------------------------
def _server_settings(name, raw):
    if not isinstance(raw, dict) or not raw.get("command"):
        raise ValueError(f"[tools.mcp.servers.{name}] needs command = \"...\"")
    if raw.get("enabled", True) is False:
        return None
    allow = [str(a) for a in raw.get("allow", [])]
    if not allow:
        raise ValueError(f"[tools.mcp.servers.{name}] needs allow = [...] listing the tools to offer")
    return {
        "command": str(_expand(raw["command"])) if "/" in str(raw["command"]) else str(raw["command"]),
        "args": [str(_expand(a)) if str(a).startswith("~") else str(a) for a in raw.get("args", [])],
        "allow": allow,
        "allowed_paths": [str(p) for p in raw.get("allowed_paths", [])],
        "public_urls_only": bool(raw.get("public_urls_only", True)),
        "defaults": dict(raw.get("defaults", {})),
        "env_pass": [str(v) for v in raw.get("env_pass", [])],
        "prompt": str(raw.get("prompt", "")),
        "examples": [str(e) for e in raw.get("examples", [])],
        "mode": str(raw.get("protocol", "legacy")),
    }


def _make_tool(bridge, server_name, server, tool, name):
    """One MCP tool -> a function plus schema the Toolbox understands."""
    def run(**kwargs):
        args = _check_args(server, kwargs)
        return _result_text(bridge.call(server_name, tool.name, args))

    run.__name__ = name
    params = _clean_schema(tool.model_dump(by_alias=True).get("inputSchema") or {"type": "object", "properties": {}})
    params.setdefault("properties", {})
    # Arguments with a configured default don't have to be given by the model
    params["required"] = [r for r in params.get("required", []) if r not in server["defaults"]]
    schema = {"type": "function", "function": {
        "name": name, "description": _short(tool.description) or name, "parameters": params}}
    return run, schema


def setup(settings):
    """Start the servers in [tools.mcp.servers] and build their tools."""
    global _bridge
    from mcp import StdioServerParameters

    shutdown()
    RUNTIME_TOOLS.clear()
    GROUPS.clear()
    STATUS.clear()
    servers = {}
    for name, raw in (settings.get("servers") or {}).items():
        server = _server_settings(name, raw)
        if server is not None:
            servers[name] = server
    if not servers:
        raise ValueError("no servers configured under [tools.mcp.servers]")

    _bridge = Bridge(float(settings.get("timeout", 30)))
    used_names = set()
    for name, server in servers.items():
        env = {k: os.environ[k] for k in server["env_pass"] if k in os.environ}
        params = StdioServerParameters(command=server["command"], args=server["args"], env=env or None)
        try:
            tools = _bridge.connect(name, params, server["mode"])
        except Exception as e:
            STATUS[name] = f"failed to start ({type(e).__name__}: {e})"
            print(f"  [mcp] {name}: {STATUS[name]}")
            continue
        offered, skipped = 0, []
        for tool in tools:
            if tool.name not in server["allow"]:
                continue
            hints = tool.annotations
            if hints is not None and hints.destructive_hint:
                skipped.append(f"{tool.name} (destructive)")
                continue
            clean = re.sub(r"[^A-Za-z0-9_]", "_", tool.name)
            tool_name = clean if clean not in used_names else f"{name}_{clean}"
            used_names.add(tool_name)
            func, schema = _make_tool(_bridge, name, server, tool, tool_name)
            read_only = hints is not None and hints.read_only_hint is True
            RUNTIME_TOOLS.append({"func": func, "schema": schema, "group": f"mcp:{name}",
                                  "writes": not read_only, "untrusted": True})
            offered += 1
        missing = [a for a in server["allow"] if a not in {t.name for t in tools}]
        GROUPS[f"mcp:{name}"] = {"prompt": server["prompt"], "examples": server["examples"]}
        STATUS[name] = f"ready ({offered} tools)"
        if skipped:
            print(f"  [mcp] {name}: not offered: {', '.join(skipped)}")
        if missing:
            print(f"  [mcp] {name}: allow lists tools the server doesn't have: {', '.join(missing)}")
    if not RUNTIME_TOOLS:
        problems = "; ".join(f"{n}: {s}" for n, s in STATUS.items())
        shutdown()
        raise ValueError(f"no MCP tools available ({problems})")


def shutdown():
    """Stop every server this copy of the plugin started."""
    global _bridge
    if _bridge is not None:
        bridge, _bridge = _bridge, None
        bridge.stop()


def briefing():
    failed = [n for n, s in STATUS.items() if not s.startswith("ready")]
    return f"Tool servers not running: {', '.join(failed)}." if failed else None


atexit.register(shutdown)
