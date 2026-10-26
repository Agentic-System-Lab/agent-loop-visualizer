"""A scripted stand-in for pi in RPC mode (agent/harnesses/pi_harness.py), run as a child process the way pi is.

It speaks pi's protocol as pi 1.4 does (docs/rpc.md, docs/json.md): commands on stdin, responses and events on
stdout, one JSON object per line. What its model says comes from the JSON file in FAKE_PI_SCRIPT:

    {"models": [pi model objects], "turns": [turn, ...], "delay": seconds between streamed pieces}

and each turn is {"stop": pi's stopReason, "error": its errorMessage, "usage": {...}, "say": [item, ...]}, where an item
is {"thinking": text}, {"text": text} or {"tool": name, "args": {...}, "result": text, "fails": bool}. A turn with
"forever": true streams until it is aborted, and with "deaf": true even then.

Loaded with --extension (pi_extension.ts), it does what the extension does: before each request to its model it
reports the request as a notification, with AGENT_LOOP_APPROVE it asks before each tool, and with AGENT_LOOP_ENDPOINT
it has the provider agent-loop, whose model is the page's own (model_endpoint.py): then each turn really asks that
endpoint over HTTP, as pi's openai-completions provider does, streams back what the model says and runs the tool calls
it gets, each answering what "results" says for its tool ("done" otherwise), until a reply calls no tool.

What it was started with and every command it read are appended to FAKE_PI_LOG, one JSON per line. FakePi, at the
end, puts it in pi's place for a test.
"""

import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request

ARGS = sys.argv[1:]
LOG = os.environ.get("FAKE_PI_LOG")
SCRIPT = json.loads(open(os.environ["FAKE_PI_SCRIPT"]).read()) if os.environ.get("FAKE_PI_SCRIPT") else {"turns": []}
DELAY = SCRIPT.get("delay", 0.005)
MODELS = SCRIPT.get("models", [{"id": "fake-model", "name": "Fake Model", "provider": "fake", "api": "openai-completions",
                                "reasoning": False, "input": ["text", "image"], "contextWindow": 32000, "maxTokens": 2048,
                                "cost": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0}}])
ENDPOINT = os.environ.get("AGENT_LOOP_ENDPOINT")
if ENDPOINT:  # what pi_extension.ts registers
    MODELS = [*MODELS, {"id": os.environ["AGENT_LOOP_MODEL"], "name": os.environ["AGENT_LOOP_MODEL"], "provider": "agent-loop",
                        "api": "openai-completions", "reasoning": True, "input": os.environ.get("AGENT_LOOP_INPUT", "text").split(","),
                        "contextWindow": int(os.environ["AGENT_LOOP_CONTEXT"]), "maxTokens": int(os.environ["AGENT_LOOP_MAX_TOKENS"]),
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}]
SYSTEM = "You are an expert coding assistant operating inside pi, a coding agent harness."
inbox: queue.Queue = queue.Queue()
model: dict | None = MODELS[0] if MODELS else None
thinking = "off"


def log(entry: dict):
    if LOG:
        with open(LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")


def out(record: dict):
    sys.stdout.write(json.dumps(record) + "\n")
    sys.stdout.flush()


def read_stdin():
    for line in sys.stdin:
        command = json.loads(line)
        log({"command": command})
        inbox.put(command)
    inbox.put(None)


class Aborted(Exception):
    pass


def check(abortable=True):
    """Handle the commands that came meanwhile; raise Aborted on an abort."""
    while True:
        try:
            command = inbox.get_nowait()
        except queue.Empty:
            return
        if command is None:
            sys.exit(0)
        if command.get("type") == "abort":
            out({"id": command.get("id"), "type": "response", "command": "abort", "success": True})
            if abortable and not SCRIPT.get("deaf"):
                raise Aborted
        else:
            answer(command)


def answer(command: dict):
    global model, thinking
    kind, rid = command.get("type"), command.get("id")
    ok = {"id": rid, "type": "response", "command": kind, "success": True}
    if kind == "get_available_models":
        out({**ok, "data": {"models": MODELS}})
    elif kind == "get_state":
        out({**ok, "data": {"model": model, "thinkingLevel": "off", "isStreaming": False}})
    elif kind == "set_model":
        found = next((m for m in MODELS if m["provider"] == command["provider"] and m["id"] == command["modelId"]), None)
        if found:
            model = found
            out({**ok, "data": found})
        else:
            out({**ok, "success": False, "error": f"Model not found: {command['provider']}/{command['modelId']}"})
    elif kind == "prompt":
        if SCRIPT.get("refuse"):
            out({**ok, "success": False, "error": SCRIPT["refuse"]})
            return
        out({**ok, "data": {"disposition": "started"}})
        run(command)
    elif kind == "set_thinking_level":
        thinking = command["level"]
        out(ok)
    elif kind == "extension_ui_response":
        pass  # an answer to a dialog nobody is waiting for any more
    else:
        out(ok)


def wait_for_answer(dialog: str) -> bool:
    """Block until the client answers the dialog, as pi's ctx.ui.confirm() does; an abort cancels it."""
    while True:
        command = inbox.get()
        if command is None:
            sys.exit(0)
        if command.get("type") == "extension_ui_response" and command.get("id") == dialog:
            return bool(command.get("confirmed"))
        if command.get("type") == "abort":
            out({"id": command.get("id"), "type": "response", "command": "abort", "success": True})
            raise Aborted
        answer(command)


def assistant(content, stop="pending", usage=None, error=None):
    message = {"role": "assistant", "content": content, "api": "openai-completions", "provider": model["provider"], "model": model["id"],
               "usage": usage or {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0, "cost": {"total": 0}},
               "stopReason": stop}
    return {**message, "errorMessage": error} if error else message


def stream(kind: str, index: int, text: str):
    for i in range(0, len(text), 4):
        check()
        out({"type": "message_update", "assistantMessageEvent": {"type": kind, "contentIndex": index, "delta": text[i:i + 4]}})
        time.sleep(DELAY)


def tools() -> list[dict]:
    """The tools pi declares to its model, as its openai-completions provider sends them."""
    names = ARGS[ARGS.index("--tools") + 1].split(",") if "--tools" in ARGS else []
    params = {"bash": {"command": {"type": "string"}}, "read": {"path": {"type": "string"}}}
    return [{"type": "function", "function": {"name": n, "description": f"pi's {n} tool",
                                              "parameters": {"type": "object", "properties": params.get(n, {"path": {"type": "string"}}),
                                                             "required": list(params.get(n, {"path": 0}))}}} for n in names]


def report(messages: list[dict]):
    """What pi_extension.ts does before each request: the request, to the page."""
    assert model  # set_model came first
    if "--extension" in ARGS:
        payload = {"model": model["id"], "messages": messages, "tools": tools(), "stream": True}
        out({"type": "extension_ui_request", "id": f"notify-{time.monotonic_ns()}", "method": "notify",
             "message": "agent-loop:request " + json.dumps(payload), "notifyType": "info"})


def run_tool(call: dict, item: dict) -> str:
    """Run one tool call: ask first if told to, then answer from the script; return the result the model reads."""
    out({"type": "tool_execution_start", "toolCallId": call["id"], "toolName": call["name"], "args": call["arguments"]})
    allowed = True
    if os.environ.get("AGENT_LOOP_APPROVE"):
        dialog = f"dialog-{call['id']}"
        out({"type": "extension_ui_request", "id": dialog, "method": "confirm", "title": "agent-loop:approve",
             "message": json.dumps({"toolCallId": call["id"], "toolName": call["name"], "input": call["arguments"]})})
        allowed = wait_for_answer(dialog)
    if allowed:
        time.sleep(DELAY)
        text = item.get("result", "")
        out({"type": "tool_execution_end", "toolCallId": call["id"], "toolName": call["name"],
             "result": {"content": [{"type": "text", "text": text}], "details": {}}, "isError": bool(item.get("fails")), "durationMs": 3})
        return text
    text = os.environ.get("AGENT_LOOP_REFUSAL", "")
    out({"type": "tool_execution_end", "toolCallId": call["id"], "toolName": call["name"],
         "result": {"content": [{"type": "text", "text": text}], "details": {}}, "isError": True})
    return text


def ask_endpoint(messages: list[dict]) -> dict:
    """One turn's request to the page's model, as pi's openai-completions provider makes it; the assistant message the
    stream adds up to, streamed on as pi streams it."""
    assert model and ENDPOINT  # set_model came first, with the page's model
    body = {"model": model["id"], "messages": messages, "tools": tools() or None, "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": model["maxTokens"], "chat_template_kwargs": {"enable_thinking": thinking != "off"}}
    request = urllib.request.Request(ENDPOINT + "/chat/completions", json.dumps({k: v for k, v in body.items() if v is not None}).encode(),
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['AGENT_LOOP_TOKEN']}"})
    text, reasoning, finish = "", "", None
    calls: list[dict] = []
    usage: dict = {}
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as e:  # pi's provider ends the message in the error, unless pi was aborted meanwhile
        check()
        return assistant([], "error", error=f"{e.code} {e.read().decode()}")
    with response:
        for line in response:
            check()
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta, finish = choice.get("delta") or {}, choice.get("finish_reason") or finish
                if piece := delta.get("reasoning_content"):
                    reasoning += piece
                    out({"type": "message_update", "assistantMessageEvent": {"type": "thinking_delta", "contentIndex": 0, "delta": piece}})
                if piece := delta.get("content"):
                    text += piece
                    out({"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "contentIndex": 1, "delta": piece}})
                calls += delta.get("tool_calls") or []
    content = ([{"type": "thinking", "thinking": reasoning}] if reasoning else []) + ([{"type": "text", "text": text}] if text else [])
    content += [{"type": "toolCall", "id": c["id"], "name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])} for c in calls]
    stop = {"tool_calls": "toolUse", "length": "length", "stop": "stop"}.get(finish or "", "error")
    usage = {"input": usage.get("prompt_tokens", 0), "output": usage.get("completion_tokens", 0), "cacheRead": 0, "cacheWrite": 0, "cost": {"total": 0}}
    return assistant(content, stop, usage, None if stop != "error" else f"Provider finish_reason: {finish}")


def run_on_endpoint(prompt: dict):
    """A run on the page's model: turns until a reply calls no tool, each a real request to the endpoint."""
    user = [{"type": "text", "text": prompt["message"]}] + [{"type": "image_url", "image_url": {"url": f"data:{i['mimeType']};base64,{i['data']}"}}
                                                             for i in prompt.get("images", [])]
    messages: list[dict] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    results = SCRIPT.get("results", {})
    for n in range(20):
        if n:
            check()
            out({"type": "turn_start"})
        report(messages)
        out({"type": "message_start", "message": assistant([])})
        message = ask_endpoint(messages)
        out({"type": "message_end", "message": message})
        calls = [b for b in message["content"] if b["type"] == "toolCall"]
        thinking_text = next((b["thinking"] for b in message["content"] if b["type"] == "thinking"), "")
        text = next((b["text"] for b in message["content"] if b["type"] == "text"), "")
        messages.append({"role": "assistant", "content": text or None, **({"reasoning_content": thinking_text} if thinking_text else {}),
                         **({"tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                            for c in calls]} if calls else {})})
        for call in calls:
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": run_tool(call, {"result": results.get(call["name"], "done")})})
        out({"type": "turn_end", "message": message, "toolResults": [c["id"] for c in calls]})
        if not calls:
            return


def run(prompt: dict):
    out({"type": "agent_start"})
    out({"type": "turn_start"})
    out({"type": "message_start", "message": {"role": "user", "content": [{"type": "text", "text": prompt["message"]}]}})
    out({"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": prompt["message"]}]}})
    assert model  # set_model came first
    aborted, calls = False, 0
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt["message"]}]
    try:
        if model["provider"] == "agent-loop":
            run_on_endpoint(prompt)
        for n, turn in enumerate(SCRIPT["turns"] if model["provider"] != "agent-loop" else []):
            if n:
                check()
                out({"type": "turn_start"})
            report(messages)
            time.sleep(DELAY)
            check()
            content: list = []
            out({"type": "message_start", "message": assistant([])})
            if turn.get("forever"):
                while True:
                    stream("text_delta", 0, "more ")
            for i, item in enumerate(turn.get("say", [])):
                if "thinking" in item:
                    stream("thinking_delta", i, item["thinking"])
                    content.append({"type": "thinking", "thinking": item["thinking"]})
                elif "text" in item:
                    stream("text_delta", i, item["text"])
                    content.append({"type": "text", "text": item["text"]})
                else:
                    calls += 1
                    call = {"type": "toolCall", "id": f"call_{calls}", "name": item["tool"], "arguments": item.get("args", {})}
                    out({"type": "message_update", "assistantMessageEvent": {"type": "toolcall_start", "contentIndex": i, "id": call["id"], "toolName": item["tool"]}})
                    stream("toolcall_delta", i, json.dumps(call["arguments"]))
                    content.append(call)
            usage = {"input": 1000, "output": 20, "cacheRead": 200, "cacheWrite": 0, "totalTokens": 1220, "cost": {"total": 0.0021}, **turn.get("usage", {})}
            calls_here = [b for b in content if b["type"] == "toolCall"]
            message = assistant(content, turn.get("stop", "toolUse" if calls_here else "stop"), usage, turn.get("error"))
            out({"type": "message_end", "message": message})
            results = []
            text = "".join(b.get("text", "") for b in content if b["type"] == "text")
            messages.append({"role": "assistant", "content": text or None,
                             **({"tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                                for c in calls_here]} if calls_here else {})})
            for call in calls_here:
                item = next(i for i in turn["say"] if i.get("tool") == call["name"])
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": run_tool(call, item)})
                results.append(call["id"])
            out({"type": "turn_end", "message": message, "toolResults": results})
            if message["stopReason"] in ("stop", "length", "error"):
                break
    except Aborted:
        aborted = True
        message = assistant([], "aborted")
        out({"type": "message_end", "message": message})
        out({"type": "turn_end", "message": message, "toolResults": []})
    out({"type": "agent_end", "messages": [], "willRetry": False})
    out({"type": "agent_settled", "aborted": aborted})


def main():
    log({"args": ARGS, "cwd": os.getcwd(), "refusal": os.environ.get("AGENT_LOOP_REFUSAL"), "approve": os.environ.get("AGENT_LOOP_APPROVE"),
         "endpoint": ENDPOINT})
    if ARGS == ["--version"]:
        print(SCRIPT.get("version", "1.4.0"))
        return
    if SCRIPT.get("crash"):
        sys.stderr.write(SCRIPT["crash"] + "\n")
        sys.exit(1)
    threading.Thread(target=read_stdin, daemon=True).start()
    while True:
        command = inbox.get()
        if command is None:
            return
        try:
            answer(command)
        except Aborted:
            pass


class FakePi:
    """For a test: this file in pi's place (pi_harness.PI_COMMAND). say() writes what its model says next; log() reads back
    what each pi started was given and was sent."""

    def __init__(self, monkeypatch, folder):
        from agent.harnesses import pi_harness  # only in the test's process

        self.script, self.log_file = folder / "fake_pi_script.json", folder / "fake_pi.log"
        monkeypatch.setattr(pi_harness, "PI_COMMAND", [sys.executable, __file__])
        monkeypatch.setenv("FAKE_PI_SCRIPT", str(self.script))
        monkeypatch.setenv("FAKE_PI_LOG", str(self.log_file))
        self.say()

    def say(self, *turns, **more):
        self.script.write_text(json.dumps({"turns": list(turns), **more}))

    def log(self) -> list[dict]:
        return [json.loads(line) for line in self.log_file.read_text().splitlines()] if self.log_file.exists() else []

    def commands(self) -> list[dict]:
        return [e["command"] for e in self.log() if "command" in e]

    def started(self) -> dict:
        """What the last pi that ran a turn was started with (not a `pi --version`)."""
        return [e for e in self.log() if "args" in e and e["args"] != ["--version"]][-1]


if __name__ == "__main__":
    main()
