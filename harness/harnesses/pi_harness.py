"""pi as the harness: the page hands the turn to pi (https://pi.dev), a coding agent that runs a loop of its own,
and draws what pi says it does.

pi runs as a child process in RPC mode (`pi --mode rpc`): commands go in on its stdin and records come out on its
stdout, one JSON object per line. For each question PiHarness starts pi in the conversation's folder, picks the model and
the thinking level, sends the question as a prompt and turns pi's events into the page's (harness.py): one of pi's
turns (turn_start to turn_end: one request to the model and the tools its reply calls) is a step, the text the model
streams is delta, the finished assistant message is model_end and parsed, and each tool execution is tool_call,
tool_start and tool_result. pi keeps the conversation in a session file in the conversation's folder, so the next
question starts pi on the same session and the model reads all of it.

pi asks a model over HTTP, in one of two ways. It can ask the page's own model (PageModel: Gemma or Qwen on
transformers or vLLM), which model_endpoint.py serves on 127.0.0.1 as the provider agent-loop: then the endpoint renders
each request with the model's chat template and generates, and its events (context, prefill, token, model_end) go
into pi's step, which shows every token as SimpleHarness's do. Or it can ask a model at a provider (Anthropic, OpenAI,
Google, ... or any endpoint in its models.json), with the credentials it keeps itself (`pi` then /login, or an API
key in the environment): an API sends back text, not tokens, so a step shows the text as it streams (delta) and the
token counts the provider reported, and no probabilities. Either way pi_extension.ts reports each request pi sends,
and the step shows it (request): pi's system prompt, its tools and the conversation, as JSON.

pi's tools (read, bash, edit, write, grep, find, ls) run on this machine, in the conversation's folder, without a
sandbox: bash reaches whatever the user can. Ask me before running a tool goes through the same extension, which then
asks before every call. In RPC mode its question is a dialog on stdout (extension_ui_request), which becomes the
page's approval_request; the user's answer goes back on stdin.

The page starts pi without the user's own extensions, skills, prompt templates and context files (AGENTS.md), so
that what it draws is pi's own loop, the same on every machine; the models and the credentials are still the user's.
"""

import base64
import collections
import json
import mimetypes
import os
import queue
import shlex
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from harness import media
from harness.backends.backend import Backend, Sampling
from harness.harnesses import model_endpoint
from harness.harnesses.simple_harness import APPROVAL_TIMEOUT, common_prefix
from harness.i18n import pick

# How to start pi: the `pi` on the PATH, unless PI_COMMAND says otherwise (a path, or "npx -y @earendil-works/pi-coding-agent").
PI_COMMAND = shlex.split(os.environ.get("PI_COMMAND", "pi"))
INSTALL = "npm install -g @earendil-works/pi-coding-agent"
FLAGS = ["--mode", "rpc", "--no-extensions", "--no-skills", "--no-prompt-templates", "--no-themes", "--no-context-files"]
FILES = {"zh": "這台電腦的檔案", "en": "files on this computer"}
# pi's built-in tools: what each does and where it runs, as the page says it.
TOOLS = {
    "read": ({"zh": "讀一個檔案（文字或圖片）", "en": "reads a file (text or an image)"}, FILES),
    "bash": ({"zh": "執行 shell 指令", "en": "runs a shell command"}, {"zh": "這台電腦的 shell", "en": "shell on this computer"}),
    "edit": ({"zh": "把檔案裡的一段文字換成另一段", "en": "replaces exact text in a file"}, FILES),
    "write": ({"zh": "新建或覆寫一個檔案", "en": "creates or overwrites a file"}, FILES),
    "grep": ({"zh": "搜尋檔案的內容", "en": "searches the contents of files"}, FILES),
    "find": ({"zh": "用 glob 找檔案的路徑", "en": "finds paths by a glob pattern"}, FILES),
    "ls": ({"zh": "列出一個資料夾", "en": "lists a folder"}, FILES),
}
OTHER_TOOL = {"zh": "pi 的工具", "en": "a pi tool"}  # where a tool runs that pi has and the page does not know
DEFAULT_TOOLS = ("read", "bash", "edit", "write")  # what pi offers the model unless told otherwise
THINKING = ("off", "minimal", "low", "medium", "high")  # pi's thinking levels for a model that thinks
EXTENSION = Path(__file__).with_name("pi_extension.ts")
APPROVE_TITLE = "agent-loop:approve"  # the title pi_extension.ts gives its confirm dialog
REQUEST_MARK = "agent-loop:request "  # how pi_extension.ts starts the notification of a request
SESSION_DIR = ".pi-session"  # in the conversation's folder
START_TIMEOUT = 30  # seconds pi may take to start and to answer a command
STOP_GRACE = 10  # seconds pi has to settle after an abort, or to exit after its stdin closes, before it is killed
DIALOGS = ("select", "confirm", "input", "editor")  # the extension_ui_request methods that wait for an answer


class PiError(Exception):
    """pi could not be started, refused a command or exited; `text` says so in both of the page's languages."""

    def __init__(self, zh: str, en: str):
        super().__init__(en)
        self.text = {"zh": zh, "en": en}


def environment_of(tool: str) -> dict[str, str]:
    return TOOLS[tool][1] if tool in TOOLS else OTHER_TOOL


def not_found() -> PiError:
    command = " ".join(PI_COMMAND)
    return PiError(
        f"找不到 pi（`{command}`）。用 `{INSTALL}` 安裝它（需要 Node.js 22.19 以上），或用環境變數 `PI_COMMAND` 指定怎麼啟動它。",
        f"pi (`{command}`) was not found. Install it with `{INSTALL}` (it needs Node.js 22.19 or later), or say how to start it with `PI_COMMAND`.",
    )


class Process:
    """One pi in RPC mode: commands are written to its stdin, and a thread reads its stdout into `inbox`."""

    def __init__(self, args: list[str], cwd: Path, inbox: queue.Queue, env: dict | None = None):
        try:
            self.proc = subprocess.Popen(
                [*PI_COMMAND, *FLAGS, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=cwd, env={**os.environ, "PI_SKIP_VERSION_CHECK": "1", **(env or {})},
            )
        except (FileNotFoundError, PermissionError):
            raise not_found() from None
        self.inbox = inbox
        self.backlog: collections.deque = collections.deque()  # what came while request() waited for its response
        self.stderr: collections.deque = collections.deque(maxlen=20)
        self._lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read(self):
        # A binary pipe splits lines on LF alone, as RPC's framing asks: a JSON string may hold U+2028.
        for line in self.proc.stdout:
            try:
                self.inbox.put(("record", json.loads(line)))
            except json.JSONDecodeError:
                continue
        self.inbox.put(("exit", None))

    def _read_stderr(self):
        for line in self.proc.stderr:
            self.stderr.append(line.decode(errors="replace").rstrip())

    def send(self, command: dict):
        with self._lock:
            try:
                self.proc.stdin.write(json.dumps(command, ensure_ascii=False).encode() + b"\n")
                self.proc.stdin.flush()
            except (OSError, ValueError):  # pi has exited, which its stdout says too
                pass

    def request(self, command: dict, timeout: float = START_TIMEOUT) -> dict:
        """Send a command and wait for its response; return the response's data. What comes meanwhile is kept, in
        order, in `backlog`."""
        rid = uuid.uuid4().hex[:8]
        self.send({**command, "id": rid})
        deadline = time.monotonic() + timeout
        while True:
            try:
                kind, value = self.inbox.get(timeout=max(deadline - time.monotonic(), 0.01))
            except queue.Empty:
                raise PiError(f"pi 在 {timeout} 秒內沒有回應 `{command['type']}`。", f"pi did not answer `{command['type']}` within {timeout} s.") from None
            if kind == "exit":
                raise self.exited()
            if kind == "record" and value.get("type") == "response" and value.get("id") == rid:
                if not value.get("success"):
                    error = value.get("error") or ""
                    raise PiError(f"pi 拒絕了 `{command['type']}`：{error}", f"pi refused `{command['type']}`: {error}")
                return value.get("data") or {}
            self.backlog.append((kind, value))

    def exited(self) -> PiError:
        said = "\n".join(self.stderr)
        return PiError("pi 意外結束了。" + (f"它最後說：\n{said}" if said else ""), "pi exited unexpectedly." + (f" Its last words:\n{said}" if said else ""))

    def close(self, grace: float = STOP_GRACE):
        """Close stdin, which asks pi to shut down, and kill it if it has not within `grace` seconds."""
        with self._lock:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        try:
            self.proc.wait(grace)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def probe() -> dict:
    """What pi can run: its version, the models it has credentials for (pi's own model objects: provider, id, name,
    input, reasoning, contextWindow, maxTokens, cost) and the one it starts on by itself ("provider/id", or None).
    Raises PiError when pi does not start or answer."""
    try:
        done = subprocess.run([*PI_COMMAND, "--version"], capture_output=True, text=True, timeout=START_TIMEOUT)
    except (FileNotFoundError, PermissionError):
        raise not_found() from None
    except subprocess.TimeoutExpired:
        raise PiError(f"`{' '.join(PI_COMMAND)} --version` 沒有結束。", f"`{' '.join(PI_COMMAND)} --version` did not finish.") from None
    if done.returncode:
        said = (done.stderr or done.stdout).strip()[-1000:]
        raise PiError(f"pi 沒有啟動：\n{said}", f"pi did not start:\n{said}")
    with tempfile.TemporaryDirectory() as folder:  # somewhere with no project settings of its own
        process = Process(["--no-session", "--no-tools"], Path(folder), queue.Queue())
        try:
            models = process.request({"type": "get_available_models"})["models"]
            current = process.request({"type": "get_state"}).get("model")
        finally:
            process.close()
    names = [f"{m['provider']}/{m['id']}" for m in models]
    start = f"{current['provider']}/{current['id']}" if current else None
    return {"version": done.stdout.strip(), "models": models, "default": start if start in names else None}


@dataclass
class PageModel:
    """The page's own model, for pi to ask through model_endpoint.py instead of a provider's, with the sidebar's decoding."""
    llm: Backend
    sampling: Sampling


@dataclass
class PiSettings:
    provider: str  # the model, as pi names it: its provider and its id there (model_endpoint.PROVIDER for the page's own)
    model: str
    thinking: str = "off"  # one of THINKING
    tools: tuple[str, ...] = DEFAULT_TOOLS
    approve_tools: bool = False
    max_steps: int = 6
    lang: str = "zh"  # what pi_extension.ts tells the model when the user refuses a call is worded in this language


class PiHarness:
    """One turn, run by pi (harness.Harness)."""

    def __init__(self, settings: PiSettings, history: list[dict], text: str, emit: Callable[[dict], None],
                 attachments: list[media.Attachment] = (), folder: Path | None = None, session: str = "chat", page: PageModel | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.settings = settings
        self.page = page  # the page's own model, when pi asks it rather than a provider's
        self.history = history  # only for run_end: pi keeps the conversation in its session
        self.text = text
        self.attachments = list(attachments)
        self.folder = Path(folder or tempfile.mkdtemp())  # the conversation's: pi works in it and keeps its session there
        self.session = session  # the session's id, the same for every question of the conversation
        self._emit = emit
        self._start = time.perf_counter()
        self._inbox: queue.Queue = queue.Queue()  # pi's records, and the user's clicks, for run() to handle in order
        self._pi: Process | None = None
        self._step, self._open = 0, False  # the page's step, and whether pi's current turn is one (not after an abort)
        self._turn_t = self._stream_t = 0.0
        self._wrote = False  # whether the model has streamed anything yet in this message
        self._calls: dict[str, dict] = {}  # call id -> name, arguments and step of each call drawn
        self._pending: dict[str, tuple[str, float]] = {}  # call id -> the dialog's id and when it times out
        self._seen: list[tuple[str, str]] = []
        self._last: dict | None = None  # the last assistant message of a step
        self._error: str | None = None
        self._aborting: str | None = None  # why the turn is being stopped: user_stopped, max_steps, approval_timeout
        self._kill_at: float | None = None  # after an abort: when to stop waiting for pi to settle
        self._hung = False  # pi did not settle in time, so it is killed rather than asked to exit
        self._emit_lock = threading.Lock()  # the endpoint's thread sends the model's events, beside run()'s
        self._unserved = threading.Condition()  # a step opened whose request the endpoint has not taken yet
        self._waiting: int | None = None
        self._request_text = ""  # the last request pi sent, as JSON, to mark what the next one adds

    # --- controls called from the web server ---

    def stop(self):
        self._inbox.put(("stop", None))

    def decide(self, call_id: str, approved: bool):
        self._inbox.put(("decide", (call_id, approved)))

    # --- the turn ---

    def emit(self, event_type: str, **data):
        with self._emit_lock:
            self._emit({"type": event_type, "t": round((time.perf_counter() - self._start) * 1000), **data})

    def run(self):
        s, page = self.settings, self.page
        settings = {"model": f"{s.provider}/{s.model}", "thinking": s.thinking, "use_tools": bool(s.tools), "tools": list(s.tools),
                    "approve_tools": s.approve_tools, "max_steps": s.max_steps, "lang": s.lang, "page_model": page is not None}
        if page:  # what the step cards and the Model tab say of the decoding
            sm = page.sampling
            settings |= {"temperature": sm.temperature, "top_p": sm.top_p, "top_k": sm.top_k, "max_new_tokens": sm.max_new_tokens, "mtp": sm.mtp}
        self.emit(
            "run_start", run_id=self.id, model=page.llm.model_id if page else f"{s.provider}/{s.model}",
            family=page.llm.family.key if page else None, harness="pi", settings=settings, tools=list(s.tools),
        )
        stop_reason = "error"
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            self._pi = Process(self._args(), self.folder, self._inbox, self._env())
            # Each waits for its answer: pi handles commands concurrently, and a prompt must not go to the model it starts on.
            self._pi.request({"type": "set_model", "provider": s.provider, "modelId": s.model})
            self._pi.request({"type": "set_thinking_level", "level": s.thinking})
            # A failed request is retried by pi as a new turn, which would be a step of its own; the page shows the error instead.
            self._pi.request({"type": "set_auto_retry", "enabled": False})
            message, images = self._message()
            if not self._stopped_meanwhile():
                self._pi.request({"type": "prompt", "message": message, **({"images": images} if images else {})})
                self._loop()
            stop_reason = self._stop_reason()
            if stop_reason == "error":
                self.emit("error", message=f"pi: {self._error}")
        except PiError as e:  # pi did not start, refused, or exited
            self.emit("error", message=f"PiError: {e}", text={lang: f"PiError: {t}" for lang, t in e.text.items()})
        except Exception as e:  # a crash here is shown, not a silent hang
            self.emit("error", message=f"{type(e).__name__}: {e}")
        finally:
            if page:
                model_endpoint.endpoint().stop_serving()
            if self._pi:
                self._pi.close(0 if self._hung else STOP_GRACE)
        final = self._text(self._last) if self._last and stop_reason != "error" else ""
        self.emit("run_end", stop_reason=stop_reason, final=final,
                  messages=[*self.history, {"role": "user", "content": self.text}, {"role": "assistant", "content": final}])

    def _args(self) -> list[str]:
        s = self.settings
        args = ["--session-dir", str(self.folder / SESSION_DIR), "--session-id", self.session, "--extension", str(EXTENSION)]
        return args + (["--tools", ",".join(s.tools)] if s.tools else ["--no-tools"])

    def _env(self) -> dict[str, str]:
        """What pi_extension.ts is told: the refusal's words, whether to ask, and the page's model if pi asks it."""
        s, env = self.settings, {"AGENT_LOOP_REFUSAL": pick(self.settings.lang, "使用者拒絕執行這個工具呼叫", "The user refused to run this tool call")}
        if s.approve_tools:
            env["AGENT_LOOP_APPROVE"] = "1"
        if page := self.page:
            ep = model_endpoint.endpoint()
            ep.serve(model_endpoint.Serving(page.llm, page.sampling, s.lang, self.folder, self._serving_step, self.emit,
                                            lambda: self._aborting is not None))
            env |= {"AGENT_LOOP_ENDPOINT": ep.url, "AGENT_LOOP_TOKEN": ep.token, "AGENT_LOOP_MODEL": page.llm.model_id,
                    "AGENT_LOOP_CONTEXT": str(page.llm.context_length), "AGENT_LOOP_MAX_TOKENS": str(page.sampling.max_new_tokens),
                    # pi sends images alone of what a model reads
                    "AGENT_LOOP_INPUT": "text,image" if "image" in page.llm.modalities else "text"}
        return env

    def _serving_step(self) -> int | None:
        """For the endpoint: the step a request pi sent belongs to. pi starts a turn and then sends its request, and the two
        reach the page by different ways, so a request waits until the page has opened that step; one that comes outside
        every step (pi compacting the conversation, say) is served without being drawn."""
        with self._unserved:
            self._unserved.wait_for(lambda: self._waiting is not None or self._aborting is not None, timeout=model_endpoint.WAIT_FOR_STEP)
            step, self._waiting = self._waiting, None
            return step

    def _message(self) -> tuple[str, list[dict]]:
        """The question, with a line for the model on the files attached (in English, as all the model reads from the
        page is), and the images it reads inline."""
        images = [{"type": "image", "data": base64.b64encode(Path(a.path).read_bytes()).decode(), "mimeType": mimetypes.guess_type(a.name)[0] or "image/png"}
                  for a in self.attachments if a.kind == "image"]
        files = ", ".join(f"{a.name} ({media.fmt_size(a.size)})" for a in self.attachments)
        note = f"Attached, in the current folder: {files}" if files else ""
        return "\n\n".join(t for t in (self.text, note) if t), images

    def _stopped_meanwhile(self) -> bool:
        """Whether Stop was pressed while pi started, before the question went to it."""
        if any(kind == "stop" for kind, _ in self._pi.backlog):
            self._aborting = "user_stopped"
        return self._aborting is not None

    def _loop(self):
        while True:
            if self._pi.backlog:
                kind, value = self._pi.backlog.popleft()
            else:
                try:
                    kind, value = self._inbox.get(timeout=0.5)
                except queue.Empty:
                    kind, value = "tick", None
            now = time.monotonic()
            if kind == "exit":
                if not self._error and not self._aborting:
                    raise self._pi.exited()
                return
            if kind == "stop":
                self._abort("user_stopped")
            elif kind == "decide":
                self._decided(*value)
            elif kind == "record" and self._on(value):
                return
            if any(deadline < now for _, deadline in self._pending.values()):
                self._abort("approval_timeout")
            if self._kill_at and now > self._kill_at:  # pi did not settle after the abort
                self._hung = True
                return

    def _abort(self, reason: str):
        if self._aborting:
            return
        self._aborting = reason
        for call_id, (dialog, _) in list(self._pending.items()):
            timed_out = reason == "approval_timeout"
            self.emit("approval", step=self._calls[call_id]["step"], call_id=call_id, approved=False, timed_out=timed_out, stopped=not timed_out)
            self._pi.send({"type": "extension_ui_response", "id": dialog, "cancelled": True})
        self._pending.clear()
        self._pi.send({"type": "abort"})
        self._kill_at = time.monotonic() + STOP_GRACE
        with self._unserved:  # a request waiting for its step learns now, not after WAIT_FOR_STEP, that none will open
            self._unserved.notify_all()

    def _stop_reason(self) -> str:
        if self._aborting:
            return self._aborting
        if self._error:
            return "error"
        if self._last and self._last.get("stopReason") == "length":
            return "max_new_tokens"
        return "no_tool_call"

    # --- pi's records ---

    def _on(self, rec: dict) -> bool:
        """Turn one of pi's records into the page's events; True once pi has settled."""
        kind = rec.get("type")
        if kind == "turn_start":
            self._turn_start()
        elif kind == "message_start" and self._open and rec["message"].get("role") == "assistant":
            self._stream_t, self._wrote = time.perf_counter(), False
        elif kind == "message_update" and self._open:
            self._update(rec["assistantMessageEvent"])
        elif kind == "message_end" and self._open and rec["message"].get("role") == "assistant":
            self._message_end(rec["message"])
        elif kind == "tool_execution_start" and self._open and not rec.get("parentToolCallId"):
            self._tool_start(rec["toolCallId"], rec["toolName"], rec.get("args") or {})
        elif kind == "tool_execution_end" and rec.get("toolCallId") in self._calls and not rec.get("parentToolCallId"):
            self._tool_end(rec["toolCallId"], rec.get("result") or {}, rec.get("isError", False), rec.get("durationMs"))
        elif kind == "extension_ui_request":
            self._dialog(rec)
        return kind == "agent_settled"

    def _turn_start(self):
        if self._aborting:  # pi goes on to the turn its tool results start before the abort reaches it
            self._open = False
            return
        if self._step >= self.settings.max_steps:  # pi has no step limit of its own; the page's is kept by stopping it
            self._abort("max_steps")
            self._open = False
            return
        self._step += 1
        self._open = True
        self._turn_t = self._stream_t = time.perf_counter()
        self.emit("step_start", step=self._step)
        with self._unserved:
            self._waiting = self._step
            self._unserved.notify_all()

    def _update(self, e: dict):
        kind = e.get("type")
        if kind == "text_delta":
            self._delta("content", e.get("delta", ""))
        elif kind == "thinking_delta":
            self._delta("thought", e.get("delta", ""))
        elif kind == "toolcall_start":
            self._delta("tool_call", ("\n" if self._wrote else "") + e.get("toolName", "") + " ")
        elif kind == "toolcall_delta":
            self._delta("tool_call", e.get("delta", ""))

    def _delta(self, channel: str, text: str):
        if text and not self.page:  # the page's model sends its tokens instead
            self._wrote = True
            self.emit("delta", step=self._step, channel=channel, text=text)

    def _message_end(self, message: dict):
        now = time.perf_counter()
        usage = message.get("usage") or {}
        reason = message.get("stopReason")
        if reason == "error" and not self._aborting:
            self._error = message.get("errorMessage") or reason
        finish = "user_stopped" if self._aborting else {"length": "max_new_tokens", "aborted": "user_stopped", "error": "error"}.get(reason, "end_turn")
        n, seconds = usage.get("output") or 0, now - self._stream_t
        if not self.page:  # the page's model sent its own, with what the backend knows
            self.emit(
                "model_end", step=self._step, finish=finish, stop_piece=None, n_tokens=n,
                first_ms=round((self._stream_t - self._turn_t) * 1000), decode_ms=round(seconds * 1000),
                # tokens over the time they took to arrive, network included; nothing when they came all at once
                tok_per_s=round(n / seconds, 1) if n and seconds > 0.05 else None, mtp=None,
                usage={"prompt": (usage.get("input") or 0) + (usage.get("cacheRead") or 0) + (usage.get("cacheWrite") or 0),
                       "cached": usage.get("cacheRead") or 0, "output": n, "cost": (usage.get("cost") or {}).get("total")},
            )
        blocks = message.get("content") or []
        calls = [{"name": b.get("name", ""), "arguments": b.get("arguments") or {}, "raw_args": json.dumps(b.get("arguments") or {}, ensure_ascii=False),
                  "error": None} for b in blocks if b.get("type") == "toolCall"]
        thinking = "\n\n".join(b["thinking"] for b in blocks if b.get("type") == "thinking" and b.get("thinking"))
        self.emit("parsed", step=self._step, thinking=thinking, content=self._text(message), tool_calls=calls)
        self._last = message

    @staticmethod
    def _text(message: dict) -> str:
        return "".join(b.get("text", "") for b in message.get("content") or [] if b.get("type") == "text")

    def _tool_start(self, call_id: str, name: str, args: dict):
        key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        repeated = key in self._seen
        self._seen.append(key)
        self._calls[call_id] = {"name": name, "arguments": args, "step": self._step}
        environment = environment_of(name)
        self.emit("tool_call", step=self._step, call_id=call_id, name=name, arguments=args, raw_args=json.dumps(args, ensure_ascii=False),
                  errors=[], repeated=repeated, environment=environment)
        if not self.settings.approve_tools:  # otherwise it starts once the user allows it (_decided)
            self.emit("tool_start", step=self._step, call_id=call_id, name=name, environment=environment)

    def _tool_end(self, call_id: str, result: dict, is_error: bool, ms: float | None):
        call = self._calls[call_id]
        parts = result.get("content") or []
        text = "\n".join(p.get("text", "") if p.get("type") == "text" else f"[{p.get('type')}]" for p in parts)
        truncated = bool((result.get("structuredContent") or {}).get("truncated") or (result.get("details") or {}).get("truncation"))
        self.emit("tool_result", step=call["step"], call_id=call_id, name=call["name"], ok=not is_error, result=text,
                  truncated=truncated, ms=round(ms or 0, 1), images=[])

    def _dialog(self, rec: dict):
        method = rec.get("method")
        if method == "notify" and (rec.get("message") or "").startswith(REQUEST_MARK):
            if self._open:  # not one past an abort
                self._request(json.loads(rec["message"][len(REQUEST_MARK):]))
        elif method == "confirm" and rec.get("title") == APPROVE_TITLE:
            call_id = json.loads(rec.get("message") or "{}").get("toolCallId")
            if self._aborting or call_id not in self._calls:
                self._pi.send({"type": "extension_ui_response", "id": rec["id"], "cancelled": True})
                return
            self._pending[call_id] = (rec["id"], time.monotonic() + APPROVAL_TIMEOUT)
            call = self._calls[call_id]
            self.emit("approval_request", step=call["step"], call_id=call_id, name=call["name"], arguments=call["arguments"])
        elif method in DIALOGS:  # no other extension is loaded, so nothing else should ask; an answer keeps pi from waiting
            self._pi.send({"type": "extension_ui_response", "id": rec["id"], "cancelled": True})

    def _request(self, payload):
        """The request pi is sending the model, as JSON, the part that is new since its last request marked: the lines
        between what the two begin and end with alike (the step's new messages; the tools after them do not change)."""
        text = json.dumps(without_media(payload), ensure_ascii=False, indent=2)
        last, self._request_text = self._request_text, text
        added_from = added_to = None
        if last:
            start = common_prefix(text, last)
            end = common_prefix(text[::-1], last[::-1])
            end = min(end, len(text) - start, len(last) - start)
            added_from, added_to = text.rfind("\n", 0, start) + 1, len(text) - end
            line_end = text.find("\n", added_to)
            added_to = len(text) if line_end < 0 else line_end
        messages = next((payload[k] for k in ("messages", "input", "contents") if isinstance(payload, dict) and isinstance(payload.get(k), list)), [])
        roles = [m.get("role") for m in messages if isinstance(m, dict) and m.get("role")]
        self.emit("request", step=self._step, text=text, added_from=added_from, added_to=added_to,
                  roles={role: roles.count(role) for role in dict.fromkeys(roles)})

    def _decided(self, call_id: str, approved: bool):
        if call_id not in self._pending:
            return
        dialog, _ = self._pending.pop(call_id)
        call = self._calls[call_id]
        self.emit("approval", step=call["step"], call_id=call_id, approved=approved, timed_out=False, stopped=False)
        if approved:
            self.emit("tool_start", step=call["step"], call_id=call_id, name=call["name"], environment=environment_of(call["name"]))
        self._pi.send({"type": "extension_ui_response", "id": dialog, "confirmed": approved})


def without_media(value):
    """A request with each image's bytes left out, as the page shows it: a data URL keeps its type, and a long base64
    string says how long it was."""
    if isinstance(value, dict):
        return {k: without_media(v) for k, v in value.items()}
    if isinstance(value, list):
        return [without_media(v) for v in value]
    if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
        kind, data = value.split(";base64,", 1)
        return f"{kind};base64,…({len(data):,} characters)"
    if isinstance(value, str) and len(value) > 1000 and not value.strip(BASE64):
        return f"…({len(value):,} base64 characters)"
    return value


BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
