"""Tools the agent can call.

A tool has two halves: the schema the model reads (name, description, JSON Schema
for the arguments; the chat template turns it into text in the system turn), and
the Python function the harness runs. The model only ever produces the call.

The schemas are in English for every page language. What goes wrong is worded in both of the page's
languages (i18n.py), and the model reads it in the language its run started in.

run_python is the one tool whose environment is not this process: it runs the model's code in a Docker
container built from sandbox/Dockerfile, with the conversation's files read-only at /data. It is offered
only when the sidebar lets the model write code, and Docker is needed for it alone: the page then asks
sandbox_problem() what is missing, tells the user, and leaves the tool out until it is there. The charts
the code draws with matplotlib come back with what it printed (sandbox/agent_loop_figures.py); they are
saved in the conversation's folder, and run_python returns them as a ToolResult, whose images the page
shows the user while the model reads only where they are.
"""

import ast
import base64
import math
import operator
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from harness.i18n import pick

WIKI_HEADERS = {"User-Agent": f"agent-loop-visualizer/0.1 (https://github.com/cyyeh/agent-atlas-learning) httpx/{httpx.__version__}"}
SANDBOX_IMAGE = "agent-loop-sandbox"  # docker build -t agent-loop-sandbox harness/sandbox
SANDBOX_VERSION = "2"  # the agent-loop-sandbox.version label in sandbox/Dockerfile; an older image is rebuilt
DOCKER_INSTALL = "https://docs.docker.com/get-started/get-docker/"
SANDBOX_TIMEOUT = 30  # seconds a piece of code may run
MAX_OUTPUT_CHARS = 1500  # of what the code prints, and of a traceback; the harness cuts the whole result at 2000
FIGURES_MARK = "<<agent-loop-figures>>"  # the line after which the sandbox writes the charts, one base64 PNG per line
MAX_FIGURES = 4  # sandbox/agent_loop_figures.py stops at as many
PNG = b"\x89PNG\r\n\x1a\n"
TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


class ToolError(Exception):
    """An expected failure, such as a timezone it does not know or a page that does not exist, in both
    languages. Python's own messages (division by zero) are the same in both."""

    def __init__(self, zh: str, en: str | None = None):
        super().__init__(zh)
        self.text = {"zh": zh, "en": en or zh}


@dataclass
class ToolResult:
    """What a tool returns when it also has something for the user to see: `data` is what the model reads, and
    `images` are files on this machine the page shows (run_python's charts), whose paths the model never reads."""
    data: dict
    images: list[str]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    environment: dict[str, str]  # language -> where the tool runs, as the page names it
    fn: Callable[..., Any]
    takes_files: bool = False  # the harness also hands fn the conversation's folder, as `files`

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    def validate(self, args: dict, lang: str = "zh") -> tuple[dict, list[str]]:
        """Check arguments against the schema; return them with defaults filled in, plus any errors in `lang`."""
        props = self.parameters.get("properties", {})
        errors = [pick(lang, f"缺少必要參數 {k}", f"Missing required argument {k}") for k in self.parameters.get("required", []) if k not in args]
        for key, value in args.items():
            spec = props.get(key)
            if spec is None:
                errors.append(pick(lang, f"沒有 {key} 這個參數", f"There is no argument {key}"))
            elif not TYPE_CHECKS[spec["type"]](value):
                got = type(value).__name__
                errors.append(pick(lang, f"{key} 應該是 {spec['type']}，收到 {got}", f"{key} should be {spec['type']}, got {got}"))
            elif "enum" in spec and value not in spec["enum"]:
                errors.append(pick(lang, f"{key} 只能是 {spec['enum']}", f"{key} must be one of {spec['enum']}"))
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
            raise ToolError(f"結果超過 {MAX_DIGITS} 位數", f"The result has more than {MAX_DIGITS} digits")
        return BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
        return UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS and not node.keywords:
        args = [_eval(a) for a in node.args]
        if node.func.id == "factorial" and args and args[0] > 500:
            raise ToolError("factorial 的參數太大", "The argument of factorial is too large")
        # round(5, -10**9) computes 10 ** 10**9 inside int.__round__, where the check on powers above cannot see it.
        if node.func.id == "round" and len(args) > 1 and abs(args[1]) > MAX_DIGITS:
            raise ToolError("round 的位數參數太大", "The digits argument of round is too large")
        return FUNCS[node.func.id](*args)
    raise ToolError(f"不支援的運算：{ast.unparse(node)}", f"Unsupported operation: {ast.unparse(node)}")


def calculator(expression: str) -> dict:
    expr = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"算式語法錯誤：{e.msg}", f"Syntax error in the expression: {e.msg}") from e
    try:
        result = _eval(tree)
    except OverflowError as e:  # the message depends on libc: "Result too large" on macOS, "Numerical result out of range" on glibc
        raise ToolError("結果超出浮點數範圍", "The result is out of floating-point range") from e
    except (ZeroDivisionError, ValueError, TypeError) as e:  # TypeError: sqrt(), factorial(2.5)
        raise ToolError(str(e)) from e
    if isinstance(result, complex):  # (-8) ** (1/3)
        raise ToolError("結果是複數，只支援實數運算", "The result is a complex number; only real numbers are supported")
    if isinstance(result, float) and not math.isfinite(result):  # 1e308 * 10 overflows without raising; inf - inf is nan
        raise ToolError("結果超出浮點數範圍", "The result is out of floating-point range")
    if _digits(result) > MAX_DIGITS:  # products of large powers, e.g. 10**999 * 10**999 * 10**999
        raise ToolError(f"結果超過 {MAX_DIGITS} 位數", f"The result has more than {MAX_DIGITS} digits")
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        result = int(result)
    return {"expression": expression, "result": result}


# --- clock ---

def get_current_time(timezone: str = "Asia/Taipei") -> dict:
    try:
        now = datetime.now(ZoneInfo(timezone))
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ToolError(
            f"不認得時區 {timezone}，請用 IANA 名稱，例如 Asia/Taipei、America/New_York",
            f"Unknown timezone {timezone}; use an IANA name such as Asia/Taipei or America/New_York",
        ) from e
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
        raise ToolError(
            f"找不到「{title}」這個頁面，可以先用 search_wikipedia 找正確的標題",
            f"There is no page titled {title!r}; search_wikipedia finds the exact title",
        )
    resp.raise_for_status()
    data = resp.json()
    return {
        "title": data.get("title", title),
        "summary": data.get("extract", ""),
        "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
    }


# --- Python: the model's code in a Docker container that cannot reach the network or change the files ---

def sandbox_user() -> str:
    """The uid:gid the container runs as: the host's own user, because on Linux a mounted folder keeps its permissions
    and another user may not be let in (pytest's tmp_path is 0700); nobody when that user is root or, on Windows, has
    no uid, since Docker Desktop shares the folder with every user there."""
    if not hasattr(os, "getuid") or os.getuid() == 0:
        return "65534:65534"
    return f"{os.getuid()}:{os.getgid()}"


def sandbox_command(name: str, files: Path | None) -> list[str]:
    """docker run for one piece of code, read from stdin. The container has no network, a read-only file system
    except a small /tmp, a cap on memory, CPU and processes, no Linux capabilities, the host's own user (never root),
    and the conversation's files read-only at /data; --rm deletes it afterwards. Its charts come out on stdout,
    after FIGURES_MARK."""
    cmd = [
        "docker", "run", "--rm", "-i", "--name", name, "--network", "none",
        "--memory", "512m", "--memory-swap", "512m", "--cpus", "1", "--pids-limit", "64",
        "--read-only", "--tmpfs", "/tmp:rw,size=64m", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", sandbox_user(), "-e", "HOME=/tmp", "-e", f"AGENT_LOOP_FIGURES={FIGURES_MARK}", "-w", "/tmp",
    ]
    if files is not None:
        cmd += ["-v", f"{Path(files).resolve()}:/data:ro"]
    return cmd + [SANDBOX_IMAGE, "python", "-I", "-"]


def sandbox_problem() -> dict[str, str] | None:
    """Why run_python cannot run right now, in both languages (Markdown), or None when Docker and an up-to-date
    sandbox image are there."""
    build = f"`docker build -t {SANDBOX_IMAGE} harness/sandbox`"
    label = '{{ index .Config.Labels "agent-loop-sandbox.version" }}'
    try:
        found = subprocess.run(["docker", "image", "inspect", "--format", label, SANDBOX_IMAGE], capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return {
            "zh": f"寫程式要用 Docker 當沙箱，但這台電腦還沒有安裝。請先[安裝 Docker]({DOCKER_INSTALL})，再到專案資料夾執行 {build}。",
            "en": f"Writing code needs Docker as its sandbox, and it is not installed on this computer. [Install Docker]({DOCKER_INSTALL}), "
            f"then run {build} in the project folder.",
        }
    except subprocess.TimeoutExpired:
        return {"zh": "Docker 沒有回應，確認它已經啟動。", "en": "Docker does not respond; make sure it is running."}
    if found.returncode == 0 and found.stdout.strip() == SANDBOX_VERSION:
        return None
    if found.returncode == 0:  # built from an older sandbox/Dockerfile: no matplotlib, and a pip that trusts revoked certificates
        return {"zh": f"沙箱的 image 是舊版的：在專案資料夾再執行一次 {build}。", "en": f"The sandbox image is out of date: run {build} in the project folder again."}
    if "no such image" in found.stderr.lower() or "not found" in found.stderr.lower():
        return {"zh": f"還沒有建好沙箱的 image：在專案資料夾執行 {build}（只要一次）。", "en": f"The sandbox image is not built yet: run {build} in the project folder (once)."}
    return {"zh": "Docker 已經安裝，但沒有在執行。啟動 Docker 之後再試一次。", "en": "Docker is installed but not running. Start Docker and try again."}


def save_figures(charts: str, folder: Path | None) -> list[Path]:
    """The PNGs the sandbox wrote after FIGURES_MARK, saved in the conversation's folder as figure-1.png, figure-2.png…
    under names no file there has yet, so the model's next code finds them in /data too."""
    pngs = []
    for line in charts.split()[:MAX_FIGURES]:
        try:
            png = base64.b64decode(line, validate=True)
        except ValueError:
            continue
        if png.startswith(PNG):
            pngs.append(png)
    if folder is None or not pngs:
        return []
    paths, n = [], 0
    for png in pngs:
        n += 1
        while (path := Path(folder) / f"figure-{n}.png").exists():
            n += 1
        path.write_bytes(png)
        paths.append(path)
    return paths


def run_python(code: str, files: Path | None = None) -> dict | ToolResult:
    name = f"agent-loop-{uuid.uuid4().hex[:12]}"
    if files is not None:  # made by Docker otherwise, as root on Linux, where the charts could not be saved
        Path(files).mkdir(parents=True, exist_ok=True)
    try:
        done = subprocess.run(sandbox_command(name, files), input=code, capture_output=True, text=True, timeout=SANDBOX_TIMEOUT)
    except FileNotFoundError as e:
        raise ToolError("沒有安裝 Docker，沒辦法執行程式", "Docker is not installed, so the code cannot run") from e
    except subprocess.TimeoutExpired as e:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)  # the container outlives a killed client
        raise ToolError(f"程式執行超過 {SANDBOX_TIMEOUT} 秒，已經停止", f"The code ran for more than {SANDBOX_TIMEOUT} s and was stopped") from e
    stdout, _, charts = done.stdout.partition(f"\n{FIGURES_MARK}\n")
    out, err = stdout[:MAX_OUTPUT_CHARS], done.stderr[-MAX_OUTPUT_CHARS:]
    if done.returncode == 125:  # docker run itself failed: no image, Docker not running
        raise ToolError(f"Docker 無法啟動沙箱：{err.strip()}", f"Docker could not start the sandbox: {err.strip()}")
    if done.returncode == 137:  # killed by the kernel, nearly always for going over --memory
        raise ToolError("程式用掉超過 512 MB 記憶體，被停止了", "The code used more than 512 MB of memory and was stopped")
    if done.returncode != 0:
        printed = "\n".join(text.strip("\n") for text in (out, err) if text.strip())
        raise ToolError(f"程式以結束代碼 {done.returncode} 結束：\n{printed}", f"The code exited with code {done.returncode}:\n{printed}")
    figures = save_figures(charts, files)
    if not out.strip() and not figures:
        raise ToolError("程式沒有印出任何東西；用 print() 印出想看的結果", "The code printed nothing; print() what you want to see")
    result = {"stdout": out + ("…(truncated)" if len(stdout) > MAX_OUTPUT_CHARS else "")} if out.strip() else {}
    if not figures:
        return result
    return ToolResult({**result, "figures": [f"/data/{path.name}" for path in figures]}, [str(path) for path in figures])


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
            environment={"zh": "系統時鐘", "en": "system clock"},
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
            environment={"zh": "Python 沙箱（AST 白名單）", "en": "Python sandbox (AST allowlist)"},
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
            environment={"zh": "Wikipedia API（網路）", "en": "Wikipedia API (network)"},
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
            environment={"zh": "Wikipedia API（網路）", "en": "Wikipedia API (network)"},
            fn=read_wikipedia,
        ),
        Tool(
            name="run_python",
            description=(
                "Run Python 3.12 code in a sandbox and return what it prints. Use it for calculations, data analysis "
                "and the files the user attached, which are under /data (read-only). numpy, pandas and matplotlib are installed; "
                "plt.show() shows a chart to the user. "
                f"There is no network, and the code may run for {SANDBOX_TIMEOUT} seconds. print() the results you need."
            ),
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string", "description": "A complete Python program"}},
                "required": ["code"],
            },
            environment={"zh": "Docker 沙箱（無網路）", "en": "Docker sandbox (no network)"},
            fn=run_python,
            takes_files=True,
        ),
    ]
}
