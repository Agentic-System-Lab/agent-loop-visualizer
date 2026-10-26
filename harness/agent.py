"""The harness: the program around the model.

Each step renders the conversation into one prompt, lets the model generate, and
reads what came out. A tool call is parsed, validated, optionally approved by the
user, executed, and its result goes back into the conversation for the next step.
A reply without a tool call ends the run; so do the step limit, the per-step token
limit, repeated malformed calls and the stop button. Every stage is emitted as an
event so the browser can draw it. The model is whatever Backend (backend.py) the page
hands it, so the loop is the same whichever runs the model.

A run keeps the page's language from when it started: the errors and refusals the model reads come
back in it. The system prompt is the same in both languages.

Files attached to the question (media.py) are saved in the conversation's folder: the images, clips and
videos the model reads become parts of the user's message, and run_python's sandbox sees the folder as /data.
"""

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from harness import media
from harness.backends.backend import Backend, Sampling
from harness.i18n import pick
from harness.tools import TOOLS, ToolError, ToolResult

MAX_RESULT_CHARS = 2000
MAX_FORMAT_ERRORS = 3
APPROVAL_TIMEOUT = 300  # seconds; an unanswered request (say, from a closed tab) must not hold the model forever

SYSTEM_WITH_TOOLS = """You are a helpful assistant running inside an agent loop.
Call a tool whenever it gives you something you cannot know or compute reliably yourself: the current time, exact arithmetic, facts to look up on Wikipedia, or the result of running code.
Call one tool at a time and wait for its result. When you have enough information, answer the user directly without calling a tool.
Reply in the same language and script the user writes in; if the user writes Traditional Chinese, reply in Traditional Chinese.
使用者用繁體中文時，請用繁體中文回答。"""

SYSTEM_NO_TOOLS = """You are a helpful assistant.
Reply in the same language and script the user writes in; if the user writes Traditional Chinese, reply in Traditional Chinese.
使用者用繁體中文時，請用繁體中文回答。"""


@dataclass
class RunSettings:
    sampling: Sampling = field(default_factory=Sampling)
    max_steps: int = 6
    thinking: bool = False
    use_tools: bool = True
    tools: tuple[str, ...] = tuple(TOOLS)  # the tools the model gets when use_tools is on; run_python only once the page allows it
    approve_tools: bool = False
    lang: str = "zh"  # what the model reads from the harness and the tools is worded in this language


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class Run:
    def __init__(self, llm: Backend, history: list[dict], user_text: str, settings: RunSettings, emit: Callable[[dict], None],
                 attachments: list[media.Attachment] = (), files: Path | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.llm = llm
        self.settings = settings
        self.messages = [*history, {"role": "user", "content": media.user_content(user_text, list(attachments))}]
        self.files = files  # the conversation's folder, which run_python's sandbox sees as /data
        self.tools = {name: t for name, t in TOOLS.items() if name in settings.tools} if settings.use_tools else {}
        self._emit = emit
        self._start = time.perf_counter()
        self._stop = threading.Event()
        self._timed_out = False
        self._decisions: dict[str, bool] = {}
        self._decided = threading.Condition()

    # --- controls called from the web server ---

    def stop(self):
        self._stop.set()
        with self._decided:
            self._decided.notify_all()

    def decide(self, call_id: str, approved: bool):
        with self._decided:
            self._decisions[call_id] = approved
            self._decided.notify_all()

    # --- the loop ---

    def emit(self, event_type: str, **data):
        self._emit({"type": event_type, "t": round((time.perf_counter() - self._start) * 1000), **data})

    def run(self):
        s = self.settings
        family = self.llm.family  # how this model writes its turns, thinking and tool calls
        tools = [t.schema() for t in self.tools.values()] or None
        schemas = {t.name: t.parameters for t in TOOLS.values()}
        system = {"role": "system", "content": SYSTEM_WITH_TOOLS if tools else SYSTEM_NO_TOOLS}
        self.emit(
            "run_start", run_id=self.id, model=self.llm.model_id, family=family.key,
            settings=_settings_dict(s), tools=list(self.tools),
        )

        seen: list[tuple[str, str]] = []
        format_errors = 0
        prev_sequence = ""
        stop_reason, final = "max_steps", ""

        try:
            for step in range(1, s.max_steps + 1):
                self.emit("step_start", step=step)
                prompt = self.llm.render([system, *self.messages], tools, s.thinking)
                encoded = self.llm.encode(prompt, media.parts(self.messages))
                # Media sit in user messages, all before what is new since the last step; so the new part is the same
                # text in the prompt and in what the model reads, only further along by what the media expanded into.
                added_from = len(encoded.text) - len(prompt) + common_prefix(prompt, prev_sequence) if prev_sequence else 0
                self.emit(
                    "context",
                    step=step,
                    prompt=encoded.text,
                    n_tokens=encoded.n_tokens,
                    added_from=added_from,
                    media_tokens=encoded.media_tokens,
                    context_length=self.llm.context_length,
                    roles=_role_counts([system, *self.messages]),
                )

                done = None
                for ev in self.llm.generate(encoded, s.sampling, self._stop.is_set):
                    kind = ev.pop("type")
                    if kind == "done":
                        done = ev
                    else:
                        self.emit(kind, step=step, **ev)
                raw = done.pop("raw")
                self.emit("model_end", step=step, raw=raw, **done)

                # The chat template may open the thinking itself; that part is in the prompt, not in raw.
                prefix = family.thought_prefix if prompt.endswith(family.thought_prefix) else ""
                parsed = family.parse(prefix + raw, schemas, s.lang)
                calls = parsed["tool_calls"] if tools else []
                self.emit("parsed", step=step, thinking=parsed["thinking"], content=parsed["content"], tool_calls=calls)

                if done["finish"] == "user_stopped":
                    stop_reason, final = "user_stopped", parsed["content"]
                    self.messages.append({"role": "assistant", "content": final})
                    break
                if not calls:
                    stop_reason = "max_new_tokens" if done["finish"] == "max_new_tokens" else "no_tool_call"
                    final = parsed["content"]
                    self.messages.append({"role": "assistant", "content": final})
                    break

                ids = [f"call_{step}_{i}" for i in range(len(calls))]
                assistant = {
                    "role": "assistant",
                    # Some templates cannot keep text in front of a tool call (see Family.text_with_calls).
                    "content": parsed["content"] if family.text_with_calls else "",
                    "tool_calls": [
                        {"id": cid, "type": "function", "function": {"name": c["name"], "arguments": c["arguments"] or {}}}
                        for cid, c in zip(ids, calls)
                    ],
                }
                if parsed["thinking"]:
                    assistant[family.reasoning_key] = parsed["thinking"]
                self.messages.append(assistant)
                # The template rewrites the model's turn when it re-renders history (sorted arguments,
                # its own thought markup), so compare the next prompt with this rendering, not prompt + raw.
                prev_sequence = self.llm.render([system, *self.messages], tools, s.thinking, generation_prompt=False)

                for cid, call in zip(ids, calls):
                    result = self._handle_call(step, cid, call, seen)
                    if call["error"]:
                        format_errors += 1
                    else:
                        format_errors = 0
                    self.messages.append({"role": "tool", "tool_call_id": cid, "name": call["name"], "content": result})
                    if self._stop.is_set():
                        break

                if self._stop.is_set():
                    stop_reason = "approval_timeout" if self._timed_out else "user_stopped"
                    break
                if format_errors >= MAX_FORMAT_ERRORS:
                    stop_reason = "format_errors"
                    break
        except Exception as e:  # surface a crash of the backend or the harness in the UI instead of a silent hang; a tool's becomes an observation
            # A backend's error says what went wrong in both languages (vllm_backend.ServerError), and the page shows
            # the one it is in; other errors are Python's own, in English.
            both = getattr(e, "text", None)
            texts = {"text": {lang: f"{type(e).__name__}: {t}" for lang, t in both.items()}} if isinstance(both, dict) else {}
            self.emit("error", message=f"{type(e).__name__}: {e}", **texts)
            stop_reason = "error"

        self.emit("run_end", stop_reason=stop_reason, final=final, messages=self.messages)

    def _handle_call(self, step: int, cid: str, call: dict, seen: list) -> str:
        """Validate, approve and execute one tool call. Returns the observation text for the model."""
        lang = self.settings.lang
        tool = self.tools.get(call["name"])
        args, errors = call["arguments"] or {}, []
        if call["error"]:
            errors = [pick(lang, f"無法解析工具呼叫：{call['error']}", f"Could not parse the tool call: {call['error']}")]
        elif tool is None:
            known = ", ".join(self.tools)
            errors = [pick(lang, f"沒有 {call['name']} 這個工具，可用的工具是 {known}", f"There is no tool {call['name']}; the tools are {known}")]
        else:
            args, errors = tool.validate(args, lang)

        key = (call["name"], json.dumps(args, sort_keys=True, ensure_ascii=False))
        repeated = key in seen
        seen.append(key)
        self.emit(
            "tool_call",
            step=step,
            call_id=cid,
            name=call["name"],
            arguments=args,
            raw_args=call["raw_args"],
            errors=errors,
            repeated=repeated,
            environment=tool.environment if tool else None,
        )
        if errors:
            return self._observe(step, cid, call["name"], {"error": "; ".join(errors)}, ok=False, ms=0)

        if self.settings.approve_tools:
            self.emit("approval_request", step=step, call_id=cid, name=call["name"], arguments=args)
            approved = self._wait_for_decision(cid)
            # Nobody answered when the wait timed out or the run was stopped; only then is it not a rejection.
            stopped = cid not in self._decisions and not self._timed_out
            self.emit("approval", step=step, call_id=cid, approved=approved, timed_out=self._timed_out, stopped=stopped)
            if not approved:
                if self._timed_out:
                    error = pick(lang, "等待使用者確認逾時，這個工具呼叫沒有執行", "No one confirmed the tool call in time, so it did not run")
                elif stopped:
                    error = pick(lang, "使用者停止了這次執行，這個工具呼叫沒有執行", "The user stopped this run, so the tool call did not run")
                else:
                    error = pick(lang, "使用者拒絕執行這個工具呼叫", "The user refused to run this tool call")
                return self._observe(step, cid, call["name"], {"error": error}, ok=False, ms=0)

        self.emit("tool_start", step=step, call_id=cid, name=call["name"], environment=tool.environment)
        start = time.perf_counter()
        try:
            result, ok = tool.fn(**args, **({"files": self.files} if tool.takes_files else {})), True
        except ToolError as e:
            result, ok = {"error": e.text[lang]}, False
        except Exception as e:
            result, ok = {"error": f"{type(e).__name__}: {e}"}, False
        return self._observe(step, cid, call["name"], result, ok=ok, ms=(time.perf_counter() - start) * 1000)

    def _observe(self, step: int, cid: str, name: str, result: dict | ToolResult, ok: bool, ms: float) -> str:
        # The model reads a ToolResult's data; its images (run_python's charts) are only for the page to show.
        result, images = (result.data, result.images) if isinstance(result, ToolResult) else (result, [])
        text = json.dumps(result, ensure_ascii=False)
        truncated = len(text) > MAX_RESULT_CHARS
        if truncated:
            text = text[:MAX_RESULT_CHARS] + "…(truncated)"
        self.emit("tool_result", step=step, call_id=cid, name=name, ok=ok, result=text, truncated=truncated, ms=round(ms, 1), images=images)
        return text

    def _wait_for_decision(self, cid: str) -> bool:
        deadline = time.monotonic() + APPROVAL_TIMEOUT
        with self._decided:
            while cid not in self._decisions and not self._stop.is_set():
                left = deadline - time.monotonic()
                if left <= 0:  # treat as rejected and end the run
                    self._timed_out = True
                    self._stop.set()
                    break
                self._decided.wait(timeout=min(left, 0.5))
            return self._decisions.get(cid, False)


def _settings_dict(s: RunSettings) -> dict:
    return {
        "temperature": s.sampling.temperature,
        "top_p": s.sampling.top_p,
        "top_k": s.sampling.top_k,
        "max_new_tokens": s.sampling.max_new_tokens,
        "mtp": s.sampling.mtp,
        "max_steps": s.max_steps,
        "thinking": s.thinking,
        "use_tools": s.use_tools,
        "tools": list(s.tools),
        "approve_tools": s.approve_tools,
        "lang": s.lang,
    }


def _role_counts(messages: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for m in messages:
        counts[m["role"]] = counts.get(m["role"], 0) + 1
    return counts
