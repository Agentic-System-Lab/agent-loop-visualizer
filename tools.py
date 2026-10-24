"""Tools the agent can call.

A tool has two halves: the schema the model reads (name, description, JSON Schema
for the arguments; the chat template turns it into text in the system turn), and
the Python function the harness runs. The model only ever produces the call.
"""

import ast
import math
import operator
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

WIKI_HEADERS = {"User-Agent": f"agent-loop-visualizer/0.1 (https://github.com/cyyeh/agent-atlas-learning) httpx/{httpx.__version__}"}
TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


class ToolError(Exception):
    pass


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    environment: str
    fn: Callable[..., Any]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    def validate(self, args: dict) -> tuple[dict, list[str]]:
        """Check arguments against the schema; return them with defaults filled in, plus any errors."""
        props = self.parameters.get("properties", {})
        errors = [f"缺少必要參數 {k}" for k in self.parameters.get("required", []) if k not in args]
        for key, value in args.items():
            spec = props.get(key)
            if spec is None:
                errors.append(f"沒有 {key} 這個參數")
            elif not TYPE_CHECKS[spec["type"]](value):
                errors.append(f"{key} 應該是 {spec['type']}，收到 {type(value).__name__}")
            elif "enum" in spec and value not in spec["enum"]:
                errors.append(f"{key} 只能是 {spec['enum']}")
        filled = {k: spec["default"] for k, spec in props.items() if "default" in spec}
        filled.update(args)
        return filled, errors


# --- calculator: arithmetic over a whitelisted AST, never eval() ---

BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
FUNCS = {
    name: getattr(math, name)
    for name in ["sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "log", "log2", "log10", "exp", "floor", "ceil", "factorial"]
}
FUNCS.update(abs=abs, round=round)
CONSTS = {"pi": math.pi, "e": math.e}
# Bigger integers are useless to the model, and past 4300 digits json.dumps refuses to print them.
MAX_DIGITS = 2000


def _digits(x) -> float:
    return math.log10(abs(x)) if abs(x) > 1 else 0.0


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in CONSTS:
        return CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        # Check the size before computing: ((10**999)**999)**999 would otherwise hang.
        if isinstance(node.op, ast.Pow) and right > 0 and _digits(left) * right > MAX_DIGITS:
            raise ToolError(f"結果超過 {MAX_DIGITS} 位數")
        return BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
        return UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS and not node.keywords:
        args = [_eval(a) for a in node.args]
        if node.func.id == "factorial" and args and args[0] > 500:
            raise ToolError("factorial 的參數太大")
        return FUNCS[node.func.id](*args)
    raise ToolError(f"不支援的運算：{ast.unparse(node)}")


def calculator(expression: str) -> dict:
    expr = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"算式語法錯誤：{e.msg}") from e
    try:
        result = _eval(tree)
    except OverflowError as e:  # the message depends on libc: "Result too large" on macOS, "Numerical result out of range" on glibc
        raise ToolError("結果超出浮點數範圍") from e
    except (ZeroDivisionError, ValueError, TypeError) as e:  # TypeError: sqrt(), factorial(2.5)
        raise ToolError(str(e)) from e
    if isinstance(result, complex):  # (-8) ** (1/3)
        raise ToolError("結果是複數，只支援實數運算")
    if isinstance(result, float) and not math.isfinite(result):  # 1e308 * 10 overflows without raising; inf - inf is nan
        raise ToolError("結果超出浮點數範圍")
    if _digits(result) > MAX_DIGITS:  # products of large powers, e.g. 10**999 * 10**999 * 10**999
        raise ToolError(f"結果超過 {MAX_DIGITS} 位數")
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        result = int(result)
    return {"expression": expression, "result": result}


# --- clock ---

def get_current_time(timezone: str = "Asia/Taipei") -> dict:
    try:
        now = datetime.now(ZoneInfo(timezone))
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ToolError(f"不認得時區 {timezone}，請用 IANA 名稱，例如 Asia/Taipei、America/New_York") from e
    return {
        "timezone": timezone,
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": now.strftime("%A"),
        "utc_offset": now.strftime("%z"),
    }


# --- Wikipedia: after ReAct's Wikipedia actions on HotpotQA (search, lookup, finish), as search and read ---

def search_wikipedia(query: str, lang: str = "en") -> dict:
    resp = httpx.get(
        f"https://{lang}.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": query, "srlimit": 5, "format": "json", "utf8": 1},
        headers=WIKI_HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    hits = resp.json().get("query", {}).get("search", [])
    results = [{"title": h["title"], "snippet": re.sub(r"<[^>]+>", "", h["snippet"])} for h in hits]
    return {"query": query, "results": results}


def read_wikipedia(title: str, lang: str = "en") -> dict:
    resp = httpx.get(
        f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title.replace(' ', '_'), safe='')}",
        headers=WIKI_HEADERS,
        timeout=10,
        follow_redirects=True,
    )
    if resp.status_code == 404:
        raise ToolError(f"找不到「{title}」這個頁面，可以先用 search_wikipedia 找正確的標題")
    resp.raise_for_status()
    data = resp.json()
    return {
        "title": data.get("title", title),
        "summary": data.get("extract", ""),
        "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
    }


LANG = {"type": "string", "enum": ["en", "zh"], "default": "en", "description": "Wikipedia language edition"}

TOOLS = {
    t.name: t
    for t in [
        Tool(
            name="get_current_time",
            description="Get the current date and time in a timezone.",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "default": "Asia/Taipei",
                        "description": "IANA timezone name, e.g. Asia/Taipei or America/New_York",
                    },
                },
            },
            environment="系統時鐘",
            fn=get_current_time,
        ),
        Tool(
            name="calculator",
            description="Evaluate an arithmetic expression exactly. Supports + - * / // % ** and sqrt, log, sin, cos, factorial, pi, e.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The expression, e.g. (37.5 * 12) ** 0.5"},
                },
                "required": ["expression"],
            },
            environment="Python 沙箱（AST 白名單）",
            fn=calculator,
        ),
        Tool(
            name="search_wikipedia",
            description="Search Wikipedia and return up to 5 matching page titles with snippets.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search keywords"}, "lang": LANG},
                "required": ["query"],
            },
            environment="Wikipedia API（網路）",
            fn=search_wikipedia,
        ),
        Tool(
            name="read_wikipedia",
            description="Read the summary of a Wikipedia page by its exact title.",
            parameters={
                "type": "object",
                "properties": {"title": {"type": "string", "description": "Exact page title"}, "lang": LANG},
                "required": ["title"],
            },
            environment="Wikipedia API（網路）",
            fn=read_wikipedia,
        ),
    ]
}
