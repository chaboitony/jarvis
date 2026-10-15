# jarvis-install: tools
"""
tools/calc.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
Exact math, so the language model never has to do arithmetic in its head
(small models get it wrong surprisingly often).
  calculate       arithmetic and functions: 2.5*sqrt(16), sin(30 deg), 10^3, log(100)
  convert_units   unit conversion with pint: 25 mph -> m/s, 300 N*m -> lbf*ft, 72 degF -> degC

Safety: expressions are never run as Python. calculate() reads them with
Python's parser (ast), then walks the tree and allows only numbers, + - * / ^ %,
parentheses and the functions listed in FUNCTIONS. Anything else is refused.
pint has its own expression parser that doesn't run code either, and unit text
is checked against a short list of allowed characters first.
"""
import ast
import math
import operator
import re

from tools import tool

PROMPT = """Math: never do arithmetic yourself. Use calculate for any calculation, and convert_units for unit conversions.
Write expressions with * for multiplication and ^ or ** for powers. Trig functions use radians; write "30 deg" for degrees, e.g. sin(30 deg).
Say results with sensible rounding."""

EXAMPLES = [
    "what's 15 percent of 84", "square root of 2", "what is 2 to the power of 10",
    "convert 25 miles per hour to meters per second", "how many feet is 3 meters",
    "100 Fahrenheit in Celsius", "torque of 300 newton meters in foot pounds", "calculate the area of a circle with radius 4",
]

FUNCTIONS = {
    "sqrt": math.sqrt, "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "exp": math.exp, "ln": math.log, "log": math.log10, "log10": math.log10, "log2": math.log2,
    "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
    "factorial": math.factorial, "hypot": math.hypot, "degrees": math.degrees, "radians": math.radians,
    "min": min, "max": max,
}
CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "g": 9.80665, "deg": math.pi / 180}
OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
MAX_LENGTH = 200
MAX_EXPONENT = 1000      # keeps 9^9^9 from freezing the computer
MAX_FACTORIAL = 500
UNIT_TEXT = re.compile(r"^[A-Za-z0-9 .,*/^()+\-°µΩ_]+$")


# --------------------------------------------------------------------------
# calculate
# --------------------------------------------------------------------------
def _prepare(expression):
    """Accept everyday notation: ^ for powers, x or × for times, 'percent of', '30 deg'."""
    text = str(expression or "").strip()
    if not text:
        raise ValueError("Give me something to calculate.")
    if len(text) > MAX_LENGTH:
        raise ValueError("That expression is too long.")
    text = text.replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
    text = re.sub(r"(\d)\s*[xX]\s*(?=[\d(])", r"\1*", text)                  # 3 x 4 -> 3*4
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+", r"\1/100*", text)  # 15% of 84
    text = re.sub(r"\bmod(?:ulo)?\b", "%", text)                               # 7 mod 3 -> 7 % 3
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\d(])", r"(\1/100)", text)       # 15% -> 0.15 (7 % 3 stays modulo)
    text = re.sub(r"(\d+(?:\.\d+)?|\))\s*(?:deg|degrees|°)\b", r"\1*deg", text)  # 30 deg -> 30*deg
    text = text.replace("°", "*deg")
    return text


def _evaluate(node):
    """Walk the parsed expression and compute it, allowing only safe pieces."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise ValueError(f"I don't know what '{node.id}' means here.")
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and (
                abs(right) > MAX_EXPONENT
                or (isinstance(left, int) and left.bit_length() * abs(right) > 100_000)):
            raise ValueError("That power is too large to calculate.")
        return OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        func = FUNCTIONS.get(node.func.id)
        if func is None:
            raise ValueError(f"'{node.func.id}' isn't a function I can use. "
                             f"Available: {', '.join(sorted(FUNCTIONS))}.")
        args = [_evaluate(a) for a in node.args]
        if node.func.id == "factorial" and (not args or args[0] > MAX_FACTORIAL or args[0] != int(args[0])):
            raise ValueError(f"factorial needs a whole number up to {MAX_FACTORIAL}.")
        if node.func.id == "factorial":
            args = [int(args[0])]
        return func(*args)
    raise ValueError("That expression uses something I'm not allowed to calculate.")


def _format(value):
    if isinstance(value, int) and value.bit_length() > 1000:
        raise ValueError("The result is too large to say.")
    if isinstance(value, complex):
        raise ValueError("The result isn't a real number.")
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            raise ValueError("The result isn't a finite number.")
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


@tool
def calculate(expression: str) -> str:
    """Calculate a math expression exactly: + - * / ^, parentheses, sqrt, sin, cos, tan, log, ln, pi and more.

    Args:
        expression: the math to work out, e.g. "2.5 * sqrt(16)" or "sin(30 deg)" or "15% of 84"
    """
    prepared = _prepare(expression)
    try:
        tree = ast.parse(prepared, mode="eval")
    except SyntaxError:
        raise ValueError(f"I couldn't read '{expression}' as math.")
    try:
        result = _evaluate(tree)
    except ZeroDivisionError:
        raise ValueError("That divides by zero.")
    except OverflowError:
        raise ValueError("The result is too large.")
    except (TypeError, ValueError) as e:
        if isinstance(e, ValueError) and str(e) and not str(e).startswith("math domain"):
            raise
        raise ValueError("That's outside what the function accepts (for example, the square root of a negative number).")
    return f"{expression} = {_format(result)}"


# --------------------------------------------------------------------------
# convert_units
# --------------------------------------------------------------------------
_registry = None


def _units():
    """Load pint the first time it's needed (it takes a moment)."""
    global _registry
    if _registry is None:
        try:
            import pint
        except ImportError:
            raise ValueError("Unit conversion needs the pint library. In ~/rima/jarvis run: uv add pint")
        _registry = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
    return _registry


def _unit_text(text, what):
    text = " ".join(str(text or "").split())
    if not text or len(text) > 60 or not UNIT_TEXT.match(text):
        raise ValueError(f"'{text}' doesn't look like a {what}.")
    # everyday spellings pint doesn't know
    swaps = {r"\bfahrenheit\b": "degF", r"\bcelsius\b": "degC", r"\bcentigrade\b": "degC",
             r"°F": "degF", r"°C": "degC", r"\bmiles per hour\b": "mph", r"\bkilometers per hour\b": "km/h",
             r"\bfoot[ -]pounds?\b": "ft*lbf", r"\bpound[ -]feet\b": "lbf*ft", r"\bnewton[ -]meters?\b": "N*m"}
    for pattern, replacement in swaps.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a measurement between units, e.g. mph to m/s, N*m to ft*lbf, degF to degC, psi to kPa.

    Args:
        value: the number to convert
        from_unit: the unit it's in, e.g. "mph", "ft", "degF", "N*m", "psi"
        to_unit: the unit wanted, e.g. "m/s", "m", "degC", "ft*lbf", "kPa"
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{value}' isn't a number.")
    ureg = _units()
    source, target = _unit_text(from_unit, "unit"), _unit_text(to_unit, "unit")
    import pint
    try:
        result = ureg.Quantity(number, source).to(target)
    except pint.errors.DimensionalityError:
        raise ValueError(f"Can't convert {from_unit} to {to_unit}; they measure different things.")
    except (pint.errors.PintError, AttributeError, AssertionError, TypeError, ValueError):
        raise ValueError(f"I don't recognize the unit in '{from_unit}' or '{to_unit}'.")
    return f"{_format(number)} {from_unit} = {_format(float(result.magnitude))} {to_unit}"
