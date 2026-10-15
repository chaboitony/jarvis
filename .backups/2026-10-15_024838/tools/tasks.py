# jarvis-install: tools
"""
tools/tasks.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
The Taskwarrior actions the AI is allowed to take. (Formerly task_tools.py.)

Safety rules:
  * Commands run as an argument list, never through a shell, so text from the
    AI can't sneak in extra shell commands.
  * The AI can list, add, complete and change tasks. It cannot delete them.
    Any change it makes can be reverted with `task undo`.
  * Projects, dates and priorities are checked before anything runs.
"""
import json
import re
import subprocess
from datetime import date, datetime, timezone

from tools import tool

# Obsidian branch names double as Taskwarrior projects
PROJECTS = ["RDC", "RAIL", "AOC", "KIA", "SOC", "POD"]

DATE_WORDS = {
    "today", "tomorrow", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "eow", "eom", "eoy",
}
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SUBPROJECT = re.compile(r"^[A-Za-z0-9_-]+$")
MAX_LIST = 15

# Added to the system prompt while this plugin is enabled
PROMPT = """You manage Tony's Taskwarrior tasks with the task tools.
Projects match his Obsidian branches:
- RDC: research and technical theory
- RAIL: prototypes and engineering builds
- AOC: coursework and school
- KIA: archives and reference
- SOC: planning and strategy
- POD: project management

Task rules:
- Use a tool whenever a request involves tasks. Never guess a task ID; call list_tasks first.
- Choose the project that fits best. If none clearly fits, leave it out.
- For dates use YYYY-MM-DD, or today, tomorrow, a weekday name, eow, eom or eoy.
- You cannot delete tasks. If asked, say so and offer to mark the task done instead."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _run(args, verbose="nothing"):
    """Run `task` with no confirmation prompts and minimal chatter."""
    cmd = ["task", "rc.confirmation=off", f"rc.verbose={verbose}", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "Taskwarrior reported an error.")
    return result.stdout.strip()


def _check_project(project):
    parts = project.strip().split(".")
    top = parts[0].upper()
    if top not in PROJECTS:
        raise ValueError(
            f"Unknown project '{project}'. Use one of {', '.join(PROJECTS)} "
            "(sub-projects like RAIL.jarvis are fine)."
        )
    for sub in parts[1:]:
        if not SUBPROJECT.match(sub):
            raise ValueError(f"Bad sub-project name '{sub}'.")
    return ".".join([top, *parts[1:]])


def _check_due(due):
    due = due.strip().lower()
    if due in DATE_WORDS or ISO_DATE.match(due):
        return due
    raise ValueError(
        f"Can't use due date '{due}'. Use YYYY-MM-DD, today, tomorrow, "
        "a weekday name, eow, eom or eoy."
    )


def _check_priority(priority):
    p = priority.strip().upper()[:1]
    if p not in {"H", "M", "L"}:
        raise ValueError("Priority must be H (high), M (medium) or L (low).")
    return p


def _check_id(task_id):
    try:
        task_id = int(task_id)
    except (TypeError, ValueError):
        raise ValueError(f"'{task_id}' is not a task ID number.")
    if task_id < 1:
        raise ValueError("Task IDs start at 1.")
    return task_id


def _get_task(task_id):
    """Return the pending task with this ID, or raise if there isn't one."""
    data = json.loads(_run([str(task_id), "export"]) or "[]")
    if not data or data[0].get("status") != "pending":
        raise ValueError(f"There is no pending task with ID {task_id}.")
    return data[0]


def _fmt_due(stamp):
    """Taskwarrior stores dates in UTC like 20261016T050000Z."""
    if not stamp:
        return None
    dt = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime("%a %Y-%m-%d")


# --------------------------------------------------------------------------
# What the model sees. These schemas are written by hand (instead of built
# from docstrings) so the wording stays exactly as it was tuned in stage 1.
# --------------------------------------------------------------------------
_PROJECT_HELP = "Project: RDC, RAIL, AOC, KIA, SOC or POD, optionally with a sub-project like RAIL.jarvis"
_DUE_HELP = "Due date: YYYY-MM-DD, today, tomorrow, a weekday name, eow, eom or eoy"
_PRIORITY_HELP = "Priority: H, M or L"


def _schema(name, description, properties, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(required)},
        },
    }


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
@tool(schema=_schema(
    "list_tasks", "List pending tasks with their ID numbers, most urgent first.",
    {"project": {"type": "string", "description": "Optional. Only show this project."}},
))
def list_tasks(project=None):
    args = ["status:pending"]
    if project:
        args.append(f"project:{_check_project(project)}")
    tasks = json.loads(_run([*args, "export"]) or "[]")
    if not tasks:
        return "No pending tasks."
    tasks.sort(key=lambda t: t.get("urgency", 0), reverse=True)
    rows = [
        {
            "id": t.get("id"),
            "description": t.get("description"),
            "project": t.get("project"),
            "due": _fmt_due(t.get("due")),
            "priority": t.get("priority"),
        }
        for t in tasks[:MAX_LIST]
    ]
    note = f" (showing {MAX_LIST} of {len(tasks)})" if len(tasks) > MAX_LIST else ""
    return f"Pending tasks, most urgent first{note}: " + json.dumps(rows)


@tool(writes=True, schema=_schema(
    "add_task", "Add a new task.",
    {
        "description": {"type": "string", "description": "What needs to be done, short and clear."},
        "project": {"type": "string", "description": f"Optional. {_PROJECT_HELP}"},
        "due": {"type": "string", "description": f"Optional. {_DUE_HELP}"},
        "priority": {"type": "string", "description": f"Optional. {_PRIORITY_HELP}"},
    },
    required=["description"],
))
def add_task(description, project=None, due=None, priority=None):
    description = (description or "").strip()
    if not description:
        raise ValueError("A task needs a description.")
    if len(description) > 200:
        raise ValueError("Description is too long; keep it under 200 characters.")
    args = ["add"]
    if project:
        args.append(f"project:{_check_project(project)}")
    if due:
        args.append(f"due:{_check_due(due)}")
    if priority:
        args.append(f"priority:{_check_priority(priority)}")
    # `--` tells Taskwarrior everything after it is description text, so words
    # like "project:" inside the description can't change the task's settings.
    args += ["--", description]
    return _run(args, verbose="new-id") or "Task added."


@tool(writes=True, schema=_schema(
    "complete_task", "Mark a pending task as done, by its ID number from list_tasks.",
    {"task_id": {"type": "integer", "description": "The task's ID number."}},
    required=["task_id"],
))
def complete_task(task_id):
    task_id = _check_id(task_id)
    task = _get_task(task_id)
    _run([str(task_id), "done"])
    return f"Completed task {task_id}: {task['description']}. (Remaining task IDs may renumber.)"


@tool(writes=True, schema=_schema(
    "modify_task", "Change a pending task's due date, priority or project, by its ID number.",
    {
        "task_id": {"type": "integer", "description": "The task's ID number."},
        "due": {"type": "string", "description": f"Optional. {_DUE_HELP}"},
        "priority": {"type": "string", "description": f"Optional. {_PRIORITY_HELP}"},
        "project": {"type": "string", "description": f"Optional. {_PROJECT_HELP}"},
    },
    required=["task_id"],
))
def modify_task(task_id, due=None, priority=None, project=None):
    task_id = _check_id(task_id)
    task = _get_task(task_id)
    changes = []
    if due:
        changes.append(f"due:{_check_due(due)}")
    if priority:
        changes.append(f"priority:{_check_priority(priority)}")
    if project:
        changes.append(f"project:{_check_project(project)}")
    if not changes:
        raise ValueError("Nothing to change. Give a new due date, priority or project.")
    _run([str(task_id), "modify", *changes])
    return f"Updated task {task_id} ({task['description']}): {' '.join(changes)}"


def _due_date(stamp):
    """Taskwarrior's UTC timestamp -> the local calendar date."""
    dt = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return dt.astimezone().date()


def briefing():
    """Overdue and due-today tasks in one sentence for the welcome briefing."""
    tasks = json.loads(_run(["status:pending", "export"]) or "[]")
    if not tasks:
        return "Tasks: nothing pending."
    today = date.today()
    overdue = [t for t in tasks if t.get("due") and _due_date(t["due"]) < today]
    due_today = [t for t in tasks if t.get("due") and _due_date(t["due"]) == today]

    def names(group):
        group = sorted(group, key=lambda t: t.get("urgency", 0), reverse=True)
        shown = "; ".join(str(t.get("description", ""))[:60] for t in group[:3])
        return shown + ("; and more" if len(group) > 3 else "")

    parts = [f"Tasks: {len(tasks)} pending"]
    if overdue:
        parts.append(f"{len(overdue)} overdue ({names(overdue)})")
    if due_today:
        parts.append(f"{len(due_today)} due today ({names(due_today)})")
    if not overdue and not due_today:
        top = max(tasks, key=lambda t: t.get("urgency", 0))
        parts.append(f"nothing due today; most urgent is {str(top.get('description', ''))[:60]}")
    return ", ".join(parts) + "."
