"""The page's own model as an OpenAI-compatible chat completions endpoint, for a harness that asks a model's API (pi).

pi asks models over HTTP, so to have pi ask the model in memory here (SimpleBackend on this machine, or VLLMBackend), this
module serves it on 127.0.0.1, at a port of its own and behind a token. pi_extension.ts registers it in pi as the
provider agent-loop. Each request pi sends is the conversation and the tools in the OpenAI format; the endpoint turns
it into the conversation as SimpleHarness keeps it, renders it with the model's chat template, encodes it and
generates, as SimpleHarness does in a step. Its events (context, prefill, token, model_end) go to the turn being drawn
(Serving), into the step pi is in, so pi's steps show every token and the string the model read. The reply streams
back to pi as server-sent events: the thinking as reasoning_content, the reply as content, and at the end the tool
calls the family parses out of the model's own format as tool_calls, which pi runs.

pi checks a tool call's arguments itself, so the endpoint sends only the calls it could parse; one it could not is
not sent, and pi ends its turn there, since nothing told the model what went wrong (SimpleHarness would). Sampling,
thinking and the token limit are the sidebar's (Serving), not what the request asks for, except that thinking follows
pi's thinking level (chat_template_kwargs.enable_thinking).
"""

import base64
import hashlib
import json
import mimetypes
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from harness import media
from harness.backends.backend import Backend, Sampling
from harness.harnesses.simple_harness import common_prefix

PROVIDER = "agent-loop"  # what pi calls the endpoint, as pi_extension.ts registers it
WAIT_FOR_STEP = 10  # seconds a request waits for the turn it belongs to before it is served undrawn


@dataclass
class Serving:
    """What a request is served with, for one turn: the model and the sidebar's settings, and where its events go."""
    llm: Backend
    sampling: Sampling
    lang: str  # the language the errors of a tool call the family cannot parse are worded in
    folder: Path  # where the images in a request are saved for the processor: the conversation's folder
    step: Callable[[], int | None]  # waits for the step the request belongs to; None for a request outside every step
    emit: Callable[..., None]  # emit(event_type, **data), into the turn's events
    stopped: Callable[[], bool]  # whether the turn is being stopped


class Endpoint:
    """The HTTP server, one per process; it serves the turn that called serve(), and refuses requests otherwise."""

    def __init__(self):
        self.token = secrets.token_urlsafe(24)  # pi sends it as its API key; anything else on this machine is refused
        self._serving: Serving | None = None
        self._lock = threading.Lock()  # one request at a time: one model
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self))
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def serve(self, serving: Serving):
        self._serving = serving

    def stop_serving(self):
        self._serving = None


_endpoint: Endpoint | None = None
_endpoint_lock = threading.Lock()


def endpoint() -> Endpoint:
    """The process's endpoint, started the first time it is asked for."""
    global _endpoint
    with _endpoint_lock:
        if _endpoint is None:
            _endpoint = Endpoint()
        return _endpoint


# ---------- the request, as the conversation SimpleHarness keeps ----------

def text_of(content) -> str:
    if isinstance(content, str) or content is None:
        return content or ""
    return "".join(p.get("text", "") for p in content if p.get("type") == "text")


def image_part(url: str, folder: Path) -> dict | None:
    """An image sent as a data URL, saved in the conversation's folder under its hash, so that each request names the
    same file; as the part of a user message the chat template writes a placeholder for (media.Attachment.part)."""
    if not url.startswith("data:") or ";base64," not in url:
        return None
    mime, data = url[5:].split(";base64,", 1)
    raw = base64.b64decode(data)
    folder = folder / "pi-images"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (hashlib.sha256(raw).hexdigest()[:16] + (mimetypes.guess_extension(mime) or ".png"))
    if not path.exists():
        path.write_bytes(raw)
    return media.Attachment(path.name, "image", str(path), len(raw)).part()


def user_content(content, folder: Path) -> str | list[dict]:
    if isinstance(content, str) or content is None:
        return content or ""
    parts = []
    for p in content:
        if p.get("type") == "text":
            parts.append({"type": "text", "text": p.get("text", "")})
        elif p.get("type") == "image_url" and (image := image_part((p.get("image_url") or {}).get("url", ""), folder)):
            parts.append(image)
    return parts if any(p["type"] != "text" for p in parts) else "".join(p["text"] for p in parts)


def conversation(body: dict, llm: Backend, folder: Path) -> list[dict]:
    """The request's messages as SimpleHarness keeps its own (Hugging Face's format, which the chat templates take):
    the arguments of a tool call as a dict, the thinking under the family's key, a tool result with its tool's name."""
    family, names, out = llm.family, {}, []
    for m in body.get("messages", []):
        role = m.get("role")
        if role in ("system", "developer"):
            if out and out[-1]["role"] == "system":  # a template takes one system message, first
                out[-1]["content"] += "\n\n" + text_of(m.get("content"))
            else:
                out.append({"role": "system", "content": text_of(m.get("content"))})
        elif role == "user":
            out.append({"role": "user", "content": user_content(m.get("content"), folder)})
        elif role == "assistant":
            calls = []
            for c in m.get("tool_calls") or []:
                fn = c.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                names[c.get("id")] = fn.get("name", "")
                calls.append({"id": c.get("id"), "type": "function", "function": {"name": fn.get("name", ""), "arguments": args}})
            # Some templates cannot keep text in front of a tool call (see Family.text_with_calls).
            message = {"role": "assistant", "content": text_of(m.get("content")) if family.text_with_calls or not calls else ""}
            if calls:
                message["tool_calls"] = calls
            if thinking := m.get("reasoning_content") or m.get("reasoning"):
                message[family.reasoning_key] = thinking
            out.append(message)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id"), "name": m.get("name") or names.get(m.get("tool_call_id"), ""),
                        "content": text_of(m.get("content"))})
    return out


def seen_before(messages: list[dict]) -> list[dict]:
    """The conversation as of the model's last message: what the step before this one ended with, before the tool
    results that are new now."""
    last = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=None)
    return messages[: last + 1] if last is not None else []


# ---------- one request ----------

def _handler(ep: Endpoint):
    # HTTP/1.0, so a stream ends where the connection closes, with no length or chunks to say where.
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, status: int, data: dict):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if self.headers.get("Authorization") == f"Bearer {ep.token}":
                return True
            self._json(401, {"error": {"message": "This endpoint serves the Agent Loop Visualizer's pi only."}})
            return False

        def do_GET(self):
            if not self._authorized():
                return
            serving = ep._serving
            self._json(200, {"object": "list", "data": [{"id": serving.llm.model_id, "object": "model"}] if serving else []})

        def do_POST(self):
            if not self._authorized():
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            serving = ep._serving
            if self.path.rstrip("/") != "/v1/chat/completions" or serving is None:
                self._json(404 if serving else 503, {"error": {"message": "No turn is being served."}})
                return
            with ep._lock:
                self._complete(body, serving)

        def _complete(self, body: dict, serving: Serving):
            llm, family = serving.llm, serving.llm.family
            step = serving.step()  # pi starts its turn before it sends the request; this waits until the page has drawn it
            emit = (lambda kind, **data: serving.emit(kind, step=step, **data)) if step else (lambda kind, **data: None)
            tools = body.get("tools") or None
            schemas = {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools or []}
            thinking = bool((body.get("chat_template_kwargs") or {}).get("enable_thinking"))
            try:
                messages = conversation(body, llm, serving.folder)
                prompt = llm.render(messages, tools, thinking)
                encoded = llm.encode(prompt, media.parts(messages))
            except Exception as e:  # pi shows it as the request's error
                self._json(400, {"error": {"message": f"{type(e).__name__}: {e}"}})
                return
            before = seen_before(messages)
            # Media sit in user messages, all before what is new; so the new part is the same text, further along (SimpleHarness).
            if before:
                previous = llm.render(before, tools, thinking, generation_prompt=False)
                added_from = len(encoded.text) - len(prompt) + common_prefix(prompt, previous)
            else:
                added_from = 0
            emit("context", prompt=encoded.text, n_tokens=encoded.n_tokens, added_from=added_from, media_tokens=encoded.media_tokens,
                 context_length=llm.context_length, roles={role: sum(m["role"] == role for m in messages) for role in dict.fromkeys(m["role"] for m in messages)})

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            rid, gone = f"chatcmpl-{uuid.uuid4().hex[:12]}", False

            def send(delta: dict | None = None, finish: str | None = None, usage: dict | None = None) -> bool:
                nonlocal gone
                chunk = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": llm.model_id,
                         "choices": [] if usage else [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
                if usage:
                    chunk["usage"] = usage
                try:
                    self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                except OSError:  # pi went away (it was aborted): the model stops before its next token
                    gone = True
                return not gone

            send({"role": "assistant"})
            done = None
            try:
                for ev in llm.generate(encoded, serving.sampling, lambda: gone or serving.stopped()):
                    kind = ev.pop("type")
                    if kind == "done":
                        done = ev
                        continue
                    emit(kind, **ev)
                    if kind == "token" and ev["channel"] in ("thought", "content"):
                        send({"reasoning_content" if ev["channel"] == "thought" else "content": ev["text"]})
            except Exception as e:  # what runs the model failed (a vLLM server, say): a stream cannot say so but by its finish reason, which pi shows
                send(finish=f"error: {type(e).__name__}: {e}")
                return
            raw = done.pop("raw")
            emit("model_end", raw=raw, **done)
            if gone:
                return
            # The chat template may open the thinking itself; that part is in the prompt, not in raw.
            prefix = family.thought_prefix if prompt.endswith(family.thought_prefix) else ""
            calls = [c for c in family.parse(prefix + raw, schemas, serving.lang)["tool_calls"] if not c["error"]] if tools else []
            if calls:
                send({"tool_calls": [{"index": i, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                                      "function": {"name": c["name"], "arguments": json.dumps(c["arguments"] or {}, ensure_ascii=False)}}
                                     for i, c in enumerate(calls)]})
            finish = "length" if done["finish"] == "max_new_tokens" else "tool_calls" if calls else "stop"
            send(finish=finish)
            send(usage={"prompt_tokens": encoded.n_tokens, "completion_tokens": done["n_tokens"], "total_tokens": encoded.n_tokens + done["n_tokens"]})
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except OSError:
                pass

    return Handler
