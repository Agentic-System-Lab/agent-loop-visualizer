"""A scripted stand-in for pi in RPC mode (harness/harnesses/pi_harness.py), run as a child process the way pi is.

It speaks pi's protocol as pi 1.4 does (docs/rpc.md, docs/json.md): commands on stdin, responses and events on
stdout, one JSON object per line. What its model says comes from the JSON file in FAKE_PI_SCRIPT:

    {"models": [pi model objects], "turns": [turn, ...], "delay": seconds between streamed pieces}

and each turn is {"stop": pi's stopReason, "error": its errorMessage, "usage": {...}, "say": [item, ...]}, where an item
is {"thinking": text}, {"text": text} or {"tool": name, "args": {...}, "result": text, "fails": bool}. A turn with
"forever": true streams until it is aborted, and with "deaf": true even then. Loaded with --extension (pi_approve.ts), it asks before each tool, as the
extension does. What it was started with and every command it read are appended to FAKE_PI_LOG, one JSON per line.
FakePi, at the end, puts it in pi's place for a test.
"""

import json
import os
import queue
import sys
import threading
import time

ARGS = sys.argv[1:]
LOG = os.environ.get("FAKE_PI_LOG")
SCRIPT = json.loads(open(os.environ["FAKE_PI_SCRIPT"]).read()) if os.environ.get("FAKE_PI_SCRIPT") else {"turns": []}
DELAY = SCRIPT.get("delay", 0.005)
MODELS = SCRIPT.get("models", [{"id": "fake-model", "name": "Fake Model", "provider": "fake", "api": "openai-completions",
                                "reasoning": False, "input": ["text", "image"], "contextWindow": 32000, "maxTokens": 2048,
                                "cost": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0}}])
inbox: queue.Queue = queue.Queue()
model = MODELS[0] if MODELS else None


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
    global model
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


def run(prompt: dict):
    approve = "--extension" in ARGS
    out({"type": "agent_start"})
    out({"type": "turn_start"})
    out({"type": "message_start", "message": {"role": "user", "content": [{"type": "text", "text": prompt["message"]}]}})
    out({"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": prompt["message"]}]}})
    aborted, calls = False, 0
    try:
        for n, turn in enumerate(SCRIPT["turns"]):
            if n:
                check()
                out({"type": "turn_start"})
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
            for call in calls_here:
                item = next(i for i in turn["say"] if i.get("tool") == call["name"])
                out({"type": "tool_execution_start", "toolCallId": call["id"], "toolName": call["name"], "args": call["arguments"]})
                allowed = True
                if approve:
                    dialog = f"dialog-{call['id']}"
                    out({"type": "extension_ui_request", "id": dialog, "method": "confirm", "title": "agent-loop:approve",
                         "message": json.dumps({"toolCallId": call["id"], "toolName": call["name"], "input": call["arguments"]})})
                    allowed = wait_for_answer(dialog)
                if allowed:
                    time.sleep(DELAY)
                    result = {"content": [{"type": "text", "text": item.get("result", "")}], "details": {}}
                    out({"type": "tool_execution_end", "toolCallId": call["id"], "toolName": call["name"], "result": result,
                         "isError": bool(item.get("fails")), "durationMs": 3})
                else:
                    refusal = os.environ.get("AGENT_LOOP_REFUSAL", "")
                    out({"type": "tool_execution_end", "toolCallId": call["id"], "toolName": call["name"],
                         "result": {"content": [{"type": "text", "text": refusal}], "details": {}}, "isError": True})
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
    log({"args": ARGS, "cwd": os.getcwd(), "refusal": os.environ.get("AGENT_LOOP_REFUSAL")})
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
        from harness.harnesses import pi_harness  # only in the test's process

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
