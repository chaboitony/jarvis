# jarvis-install: tools
"""
tools/  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
The toolbox: finds J.A.R.V.I.S.'s abilities and runs them safely.

Every file in this folder is one group of tools (a "plugin"). Only the files
listed under [tools] enabled in config.toml are loaded.

A plugin file can contain:

    from tools import tool

    PROMPT = "Extra instructions for the model about these tools."   # optional

    EXAMPLES = ["will it rain tomorrow", "do I need a jacket"]  # optional; sample
                                    # requests that help routing pick this plugin

    def setup(settings):            # optional; gets the [tools.<file name>] table
        ...                         # from config.toml. Raise ValueError to refuse to load.

    def briefing():                 # optional; a sentence or two for the welcome
        return "Tasks: 2 due today."  # briefing (tools/briefing.py collects these).
                                    # Not a tool itself, so no @tool.

    @tool                           # marks a function as something the AI may call
    def get_weather(day: str = "today") -> str:
        '''Get the forecast for home.          <- first line: what the model sees

        Args:
            day: today, tomorrow or a weekday name
        '''
        return "Sunny"

@tool options:
    @tool(writes=True)      the tool changes something (adds, completes, edits...)
    @tool(untrusted=True)   the tool returns text from outside (web pages, email...).
                            Such text could contain instructions aimed at the AI, so
                            after an untrusted tool runs, write tools are blocked
                            for the rest of that request.
    @tool(schema={...})     give the JSON schema by hand instead of building it
                            from the function's signature and docstring.

Tools made while running (the MCP bridge, tools/mcp.py): a plugin can also fill
two module-level variables in setup():
    RUNTIME_TOOLS = [{"name", "func", "schema", "writes", "untrusted", "group"}, ...]
    GROUPS = {"mcp:git": {"prompt": "...", "examples": [...]}, ...}
and may define shutdown(), which is called when the plugin is reloaded or
J.A.R.V.I.S. exits (to stop the programs it started).

Routing: with many tools loaded, only the few groups that match a request are
offered to the model (see Toolbox.select). A group is normally one plugin file;
the MCP bridge adds one group per server (e.g. "mcp:git"). Each group is
described for routing by its name, PROMPT, tool descriptions and EXAMPLES, so
good EXAMPLES matter.

Rules for tool functions:
  * Return plain text (or something that turns into text).
  * Raise ValueError with a plain sentence when something is wrong; the model
    reads that sentence and can explain it or try again.
  * Never pass text from the model to a shell. Use argument lists.
"""
import importlib
import inspect
import math
import pkgutil
import re
import subprocess
import sys
import traceback
import typing

MAX_RESULT_CHARS = 4000  # keeps one tool result from filling the model's context window

# The plugin modules currently loaded, in config.toml order: name -> module.
# Plugins that combine others (like briefing) read this.
LOADED = {}

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean",
               list: "array", dict: "object"}


# --------------------------------------------------------------------------
# The @tool marker
# --------------------------------------------------------------------------
def tool(func=None, *, schema=None, writes=False, untrusted=False):
    """Mark a function as a tool. Works as @tool or @tool(writes=True, ...)."""
    def mark(f):
        f._jarvis_tool = {"schema": schema, "writes": writes, "untrusted": untrusted}
        return f
    return mark(func) if func is not None else mark


# --------------------------------------------------------------------------
# Building a schema from a function
# --------------------------------------------------------------------------
def _json_type(annotation):
    """Python type hint -> JSON schema type name. Unknown types become 'string'."""
    args = typing.get_args(annotation)
    if args and type(None) in args:  # Optional[int] or int | None -> int
        rest = [a for a in args if a is not type(None)]
        annotation = rest[0] if len(rest) == 1 else str
    origin = typing.get_origin(annotation) or annotation
    return _JSON_TYPES.get(origin, "string")


def _parse_docstring(doc):
    """Return (summary, {argument: description}) from a Google-style docstring."""
    lines = inspect.cleandoc(doc or "").splitlines()
    summary, arg_docs = [], {}
    in_summary, in_args, arg_indent, current = True, False, None, None

    for line in lines:
        text = line.strip()
        indent = len(line) - len(line.lstrip())
        if text in ("Args:", "Arguments:", "Parameters:"):
            in_summary, in_args = False, True
            continue
        if in_args:
            if text and indent == 0:  # the next section, e.g. "Returns:"
                in_args = False
                continue
            if not text:
                continue
            match = re.match(r"(\w+)\s*(\([^)]*\))?\s*:\s*(.*)", text)
            if match and (arg_indent is None or indent <= arg_indent):
                arg_indent = indent
                current = match.group(1)
                arg_docs[current] = match.group(3)
            elif current:  # a wrapped line belongs to the previous argument
                arg_docs[current] += " " + text
        elif in_summary:
            if text:
                summary.append(text)
            elif summary:
                in_summary = False
    return " ".join(summary), arg_docs


def build_schema(func):
    """
    Make the description the model sees from the function itself.
    Arguments with a default value are optional; the rest are required.
    """
    summary, arg_docs = _parse_docstring(func.__doc__)
    if not summary:
        raise ValueError(f"{func.__name__} needs a docstring; its first line tells the model what it does.")
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = {}

    properties, required = {}, []
    for name, param in inspect.signature(func).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        prop = {"type": _json_type(hints.get(name, str))}
        if name in arg_docs:
            prop["description"] = arg_docs[name]
        properties[name] = prop
        if param.default is param.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": summary,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _shutdown(module):
    """Call a plugin's shutdown(), if it has one, without letting it crash J.A.R.V.I.S."""
    stop = getattr(module, "shutdown", None)
    if callable(stop):
        try:
            stop()
        except Exception:
            traceback.print_exc()


def _cosine(a, b):
    """How alike two embeddings are: 1 = same direction, 0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


# --------------------------------------------------------------------------
# The toolbox
# --------------------------------------------------------------------------
class Toolbox:
    def __init__(self):
        self.entries = {}     # tool name -> {"func", "schema", "writes", "untrusted", "plugin", "group"}
        self.prompts = {}     # group name -> its PROMPT text
        self.examples = {}    # group name -> its EXAMPLES
        self.groups = []      # routing groups, in config order
        self.modules = {}     # plugin name -> module
        self.route_error = None
        self._doc_vectors = None  # plugin name -> embedding, built on first routed request
        self.loaded = []      # plugin names that loaded
        self.available = []   # every plugin file in the folder
        self.problems = []    # plugins that failed, and why
        self._tainted = False

    # ---- loading ---------------------------------------------------------
    def load(self, config):
        """
        (Re)load the enabled plugins. Everything is built first and swapped in at
        the end, so a reload that goes wrong never leaves a half-empty toolbox.
        """
        tools_cfg = config.get("tools", {})
        enabled = tools_cfg.get("enabled", [])
        available = sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_"))
        entries, prompts, examples, groups, loaded, problems = {}, {}, {}, [], [], []
        modules = {}

        for plugin in enabled:
            if plugin not in available:
                problems.append(f"{plugin}: there is no file tools/{plugin}.py")
                continue
            module = None
            try:
                module = self._import(plugin)
                settings = tools_cfg.get(plugin, {})
                if hasattr(module, "setup"):
                    module.setup(settings if isinstance(settings, dict) else {})
                found = self._collect(module, plugin)
            except Exception as e:
                problems.append(f"{plugin}: {type(e).__name__}: {e}")
                _shutdown(module)  # it may have started something before failing
                continue
            clashes = sorted(set(found) & set(entries))
            if clashes:
                problems.append(f"{plugin}: tool name already used by another plugin: {', '.join(clashes)}")
                _shutdown(module)
                continue
            entries.update(found)
            loaded.append(plugin)
            modules[plugin] = module
            plugin_prompt = str(getattr(module, "PROMPT", "") or "").strip()
            plugin_examples = list(getattr(module, "EXAMPLES", None) or [])
            runtime_groups = getattr(module, "GROUPS", None) or {}
            for group in dict.fromkeys(e["group"] for e in found.values()):
                groups.append(group)
                info = runtime_groups.get(group, {})
                text = str(info.get("prompt", "") or "").strip() or (plugin_prompt if group == plugin else "")
                if text:
                    prompts[group] = text
                examples[group] = list(info.get("examples", [])) or (plugin_examples if group == plugin else [])

        old = self.modules
        self.entries, self.prompts, self.examples, self.groups = entries, prompts, examples, groups
        self.loaded, self.available, self.problems = loaded, available, problems
        self.modules = modules
        self._doc_vectors = None  # plugins changed, so routing descriptions must be rebuilt
        LOADED.clear()
        LOADED.update(modules)
        # Stop what the previous copies of reloaded plugins started (e.g. MCP servers)
        for name, module in old.items():
            if modules.get(name) is not module:
                _shutdown(module)

    def close(self):
        """Stop everything the plugins started. Called when J.A.R.V.I.S. exits."""
        for module in self.modules.values():
            _shutdown(module)

    @staticmethod
    def _import(plugin):
        """Import a plugin fresh from its file, even if it was loaded before."""
        full_name = f"{__name__}.{plugin}"
        # Forget the old copy first. importlib.reload() would keep leftovers,
        # such as a tool you deleted from the file.
        sys.modules.pop(full_name, None)
        importlib.invalidate_caches()
        return importlib.import_module(full_name)

    @staticmethod
    def _collect(module, plugin):
        found = {}
        for obj in list(vars(module).values()):
            info = getattr(obj, "_jarvis_tool", None)
            # skip non-tools, and tools imported from another plugin
            if info is None or not callable(obj) or getattr(obj, "__module__", None) != module.__name__:
                continue
            schema = info["schema"] or build_schema(obj)
            name = schema.get("function", {}).get("name")
            if name != obj.__name__:
                raise ValueError(f"schema name '{name}' doesn't match function '{obj.__name__}'")
            found[name] = {**info, "func": obj, "schema": schema, "plugin": plugin, "group": plugin}
        for item in getattr(module, "RUNTIME_TOOLS", None) or []:
            name = item["schema"]["function"]["name"]
            if name in found:
                raise ValueError(f"two tools are called '{name}'")
            found[name] = {"func": item["func"], "schema": item["schema"],
                           "writes": bool(item.get("writes")), "untrusted": bool(item.get("untrusted")),
                           "plugin": plugin, "group": item.get("group") or plugin}
        if not found:
            raise ValueError("no @tool functions found")
        return found

    # ---- what the model sees ---------------------------------------------
    @property
    def schemas(self):
        return [entry["schema"] for entry in self.entries.values()]

    @property
    def prompt(self):
        return "\n\n".join(self.prompts.values())

    def schemas_for(self, groups):
        """The tool descriptions for only these groups."""
        return [e["schema"] for e in self.entries.values() if e["group"] in groups]

    def prompt_for(self, groups):
        """The PROMPT instructions for only these groups."""
        return "\n\n".join(text for name, text in self.prompts.items() if name in groups)

    # ---- routing ---------------------------------------------------------
    def describe_group(self, group):
        """The text routing compares requests against: name, instructions, tools, examples."""
        parts = [f"{group.replace('mcp:', '')}."]
        if group in self.prompts:
            parts.append(self.prompts[group])
        for entry in self.entries.values():
            if entry["group"] != group:
                continue
            fn = entry["schema"]["function"]
            parts.append(fn.get("description", ""))
            for prop in fn.get("parameters", {}).get("properties", {}).values():
                parts.append(prop.get("description", ""))
        examples = self.examples.get(group) or []
        if examples:
            parts.append("Examples: " + "; ".join(str(e) for e in examples))
        return " ".join(" ".join(parts).split())

    def select(self, text, embed, sticky=(), top=3):
        """
        Choose which groups' tools to offer for one request, so the model picks
        from a short list instead of every tool.

        embed(list of texts) -> list of vectors (an embedding model turns text into
        numbers so that texts with similar meaning get similar numbers). Each
        plugin's description is embedded once; each request is embedded and
        compared to them by cosine similarity. The best `top` groups are chosen,
        plus `sticky` ones (used in the last request, so follow-ups like
        "and tomorrow?" still work).

        Returns (chosen groups in config order, {group: score}). If embedding
        fails, every group is returned, so routing can never make tools vanish.
        """
        plugins = list(self.groups)
        if len(plugins) <= top:
            return plugins, {}
        try:
            if self._doc_vectors is None:
                docs = ["search_document: " + self.describe_group(p) for p in plugins]
                self._doc_vectors = dict(zip(plugins, embed(docs)))
            [query] = embed(["search_query: " + text])
            self.route_error = None
        except Exception as e:
            self.route_error = f"{type(e).__name__}: {e}"
            return plugins, {}
        scores = {p: _cosine(query, vec) for p, vec in self._doc_vectors.items()}
        chosen = set(sorted(plugins, key=scores.get, reverse=True)[:top])
        chosen.update(p for p in sticky if p in scores)
        return [p for p in plugins if p in chosen], scores

    def summary(self):
        """A short report for the screen: what loaded, what's off, what failed."""
        lines = []
        for group in self.groups:
            names = [n for n, e in self.entries.items() if e["group"] == group]
            lines.append(f"  {group}: {', '.join(names)}")
        off = [p for p in self.available if p not in self.loaded and
               not any(prob.startswith(f"{p}:") for prob in self.problems)]
        if off:
            lines.append(f"  not enabled: {', '.join(off)}")
        for problem in self.problems:
            lines.append(f"  FAILED {problem}")
        return "\n".join(lines) or "  (no tools loaded)"

    # ---- running a tool --------------------------------------------------
    def start_turn(self):
        """Call at the start of each request. Clears the untrusted-text flag."""
        self._tainted = False

    def call(self, name, arguments):
        """Run one tool call from the model and always return text it can read."""
        entry = self.entries.get(name)
        if entry is None:
            return f"Error: there is no tool named '{name}'."
        if entry["writes"] and self._tainted:
            return ("Error: blocked. This request already used outside information, so tools that "
                    "change things are switched off until Tony makes a new request. Tell him what you "
                    "would change and let him confirm.")
        if entry["untrusted"]:
            self._tainted = True  # set before running, so even an error message counts

        try:
            result = entry["func"](**(arguments if isinstance(arguments, dict) else {}))
        except TypeError as e:
            return f"Error: wrong arguments for {name} ({e})."
        except ValueError as e:
            return f"Error: {e}"
        except subprocess.TimeoutExpired:
            return f"Error: {name} took too long to respond."
        except FileNotFoundError as e:
            return f"Error: {name} needs a program or file that isn't available ({e.filename or e})."
        except Exception as e:  # a bug in a plugin shouldn't crash J.A.R.V.I.S.
            traceback.print_exc()
            return f"Error: {name} failed unexpectedly ({type(e).__name__}: {e})."

        text = "" if result is None else str(result)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + " ...(cut short)"
        return text or "Done."
