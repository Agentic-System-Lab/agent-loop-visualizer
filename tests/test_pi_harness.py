"""harnesses/pi_harness.py: the page's turn run by pi, through pi's RPC mode.

Most tests start tests/fake_pi.py in pi's place, which speaks pi 1.4's protocol from a script, so they need neither
Node nor a model. With RUN_PI_TESTS=1 and pi installed (or PI_COMMAND set), the last ones start the real pi, pointed at
a scripted OpenAI-compatible server on 127.0.0.1, so they touch no network either.
"""

import base64
import json
import os
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from fake_pi import FakePi
from harness.harnesses import pi_harness
from harness.harnesses.harness import ALWAYS, SOMETIMES, Harness
from harness.harnesses.simple_harness import RunSettings, SimpleHarness
from harness.media import Attachment

TIME = {"tool": "bash", "args": {"command": "date"}, "result": "Mon Oct 26 14:00:00 CST 2026\n"}
ANSWER = {"text": "It is 2 pm."}


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """pi is tests/fake_pi.py, saying what the test scripts; the conversation's folder is `folder`."""
    fake = FakePi(monkeypatch, tmp_path)
    fake.folder = tmp_path / "chat"
    return fake


def start(fake, settings=None, text="What time is it?", attachments=()):
    """Run a turn in a thread, as the page does; return the run, its events and the thread."""
    events = []
    run = pi_harness.PiHarness(settings or pi_harness.PiSettings("fake", "fake-model", lang="en"), [], text, events.append, attachments, fake.folder, "chat1")
    thread = threading.Thread(target=run.run, daemon=True)
    thread.start()
    return run, events, thread


def finish(run, events, thread):
    thread.join(20)
    assert not thread.is_alive()
    assert events[-1]["type"] == "run_end"
    return events


def run_pi(fake, *turns, settings=None, **kw):
    fake.say(*turns)
    return finish(*start(fake, settings, **kw))


def wait_for(events, kind, timeout=10):
    deadline = time.monotonic() + timeout
    while not any(e["type"] == kind for e in events):
        assert time.monotonic() < deadline, f"no {kind}"
        time.sleep(0.01)
    return next(e for e in events if e["type"] == kind)


def of(events, kind):
    return [e for e in events if e["type"] == kind]


def end(events):
    return events[-1]


# ---------- the interface ----------

def test_both_harnesses_are_a_harness(fake):
    assert isinstance(pi_harness.PiHarness(pi_harness.PiSettings("fake", "fake-model"), [], "hi", print), Harness)
    assert isinstance(SimpleHarness(object(), [], "hi", RunSettings(), print), Harness)


def test_every_event_pi_sends_is_one_the_interface_names(fake):
    events = run_pi(fake, {"say": [{"thinking": "Use date."}, TIME]}, {"say": [ANSWER]})
    assert {e["type"] for e in events} <= set(ALWAYS) | set(SOMETIMES)
    assert [e["type"] for e in events][:2] == ["run_start", "step_start"]


# ---------- one turn ----------

def test_a_turn_becomes_the_pages_events(fake):
    events = run_pi(fake, {"say": [{"thinking": "Use date."}, {"text": "Let me check."}, TIME]}, {"say": [ANSWER]})
    first = events[0]
    assert first["type"] == "run_start" and first["harness"] == "pi" and first["family"] is None
    assert first["model"] == "fake/fake-model" and first["tools"] == list(pi_harness.DEFAULT_TOOLS)
    assert [e["step"] for e in of(events, "step_start")] == [1, 2]
    streamed = {}
    for e in of(events, "delta"):
        if e["step"] == 1:
            streamed[e["channel"]] = streamed.get(e["channel"], "") + e["text"]
    assert streamed == {"thought": "Use date.", "content": "Let me check.", "tool_call": '\nbash {"command": "date"}'}
    done = of(events, "model_end")[0]
    assert done["finish"] == "end_turn" and done["n_tokens"] == 20 and done["stop_piece"] is None
    assert done["usage"] == {"prompt": 1200, "cached": 200, "output": 20, "cost": 0.0021}  # the prompt counts what came from the cache
    parsed = of(events, "parsed")[0]
    assert parsed["thinking"] == "Use date." and parsed["content"] == "Let me check."
    assert parsed["tool_calls"] == [{"name": "bash", "arguments": {"command": "date"}, "raw_args": '{"command": "date"}', "error": None}]
    call = of(events, "tool_call")[0]
    assert call["call_id"] == "call_1" and call["environment"] == pi_harness.TOOLS["bash"][1] and not call["errors"] and not call["repeated"]
    assert of(events, "tool_start")[0]["call_id"] == "call_1"
    result = of(events, "tool_result")[0]
    assert result["ok"] and result["result"] == TIME["result"] and result["ms"] == 3 and result["step"] == 1
    assert end(events)["stop_reason"] == "no_tool_call" and end(events)["final"] == "It is 2 pm."


def test_the_order_of_a_step_follows_the_loop(fake):
    kinds = [e["type"] for e in run_pi(fake, {"say": [TIME]}, {"say": [ANSWER]}) if e["type"] != "delta"]
    assert kinds == ["run_start", "step_start", "model_end", "parsed", "tool_call", "tool_start", "tool_result",
                     "step_start", "model_end", "parsed", "run_end"]


def test_pi_starts_in_the_conversations_folder_on_its_session_with_the_pages_tools_only(fake):
    run_pi(fake, {"say": [ANSWER]}, settings=pi_harness.PiSettings("fake", "fake-model", tools=("read", "ls"), lang="en"))
    started = fake.started()
    assert Path(started["cwd"]).resolve() == fake.folder.resolve()
    args = started["args"]
    assert args[:len(pi_harness.FLAGS)] == pi_harness.FLAGS and "--no-extensions" in args and "--no-context-files" in args
    assert args[args.index("--session-id") + 1] == "chat1"
    assert Path(args[args.index("--session-dir") + 1]) == fake.folder / pi_harness.SESSION_DIR
    assert args[args.index("--tools") + 1] == "read,ls" and "--extension" not in args


def test_the_model_is_set_and_retries_turned_off_before_the_question_goes(fake):
    run_pi(fake, {"say": [ANSWER]}, settings=pi_harness.PiSettings("fake", "fake-model", thinking="high"))
    kinds = [c["type"] for c in fake.commands()]
    assert kinds[:4] == ["set_model", "set_thinking_level", "set_auto_retry", "prompt"]
    commands = fake.commands()
    assert commands[0]["provider"] == "fake" and commands[0]["modelId"] == "fake-model"
    assert commands[1]["level"] == "high" and commands[2]["enabled"] is False


def test_no_tools_means_none_at_all(fake):
    events = run_pi(fake, {"say": [ANSWER]}, settings=pi_harness.PiSettings("fake", "fake-model", tools=()))
    assert "--no-tools" in fake.started()["args"] and events[0]["settings"]["use_tools"] is False


def test_attachments_are_named_for_the_model_and_images_sent_inline(fake, tmp_path):
    photo, data = tmp_path / "photo.png", tmp_path / "sales.csv"
    photo.write_bytes(b"\x89PNG fake")
    data.write_text("month,revenue\n")
    attachments = [Attachment("photo.png", "image", str(photo), 9), Attachment("sales.csv", "file", str(data), 14)]
    run_pi(fake, {"say": [ANSWER]}, attachments=attachments, text="What is this?")
    prompt = next(c for c in fake.commands() if c["type"] == "prompt")
    assert prompt["message"] == "What is this?\n\nAttached, in the current folder: photo.png (9 B), sales.csv (14 B)"
    assert prompt["images"] == [{"type": "image", "data": base64.b64encode(b"\x89PNG fake").decode(), "mimeType": "image/png"}]


def test_a_repeated_call_is_marked(fake):
    events = run_pi(fake, {"say": [TIME]}, {"say": [TIME]}, {"say": [ANSWER]})
    assert [c["repeated"] for c in of(events, "tool_call")] == [False, True]


def test_a_failed_tool_is_a_failed_result(fake):
    events = run_pi(fake, {"say": [{**TIME, "result": "command not found", "fails": True}]}, {"say": [ANSWER]})
    result = of(events, "tool_result")[0]
    assert not result["ok"] and result["result"] == "command not found"


def test_a_reply_cut_at_the_token_limit_says_so(fake):
    events = run_pi(fake, {"say": [{"text": "A long"}], "stop": "length"})
    assert of(events, "model_end")[0]["finish"] == "max_new_tokens" and end(events)["stop_reason"] == "max_new_tokens"


def test_the_next_question_continues_the_same_session(fake):
    run_pi(fake, {"say": [ANSWER]})
    run_pi(fake, {"say": [ANSWER]}, text="And now?")
    sessions = [e["args"][e["args"].index("--session-id") + 1] for e in fake.log() if "--session-id" in e.get("args", [])]
    assert sessions == ["chat1", "chat1"]


# ---------- asking first ----------

def asking(fake, lang="en"):
    fake.say({"say": [TIME]}, {"say": [ANSWER]})
    return start(fake, pi_harness.PiSettings("fake", "fake-model", approve_tools=True, lang=lang))


def test_ask_first_loads_the_extension_and_holds_the_call(fake):
    run, events, thread = asking(fake)
    request = wait_for(events, "approval_request")
    assert request["call_id"] == "call_1" and request["name"] == "bash" and request["arguments"] == {"command": "date"}
    assert str(pi_harness.APPROVE) in fake.started()["args"] and not of(events, "tool_start")
    run.decide("call_1", True)
    finish(run, events, thread)
    kinds = [e["type"] for e in events]
    assert kinds.index("approval") < kinds.index("tool_start") < kinds.index("tool_result")
    assert of(events, "approval")[0]["approved"] and of(events, "tool_result")[0]["ok"]


@pytest.mark.parametrize("lang, refusal", [("zh", "使用者拒絕執行這個工具呼叫"), ("en", "The user refused to run this tool call")])
def test_a_rejected_call_does_not_run_and_the_model_reads_why_in_the_turns_language(fake, lang, refusal):
    run, events, thread = asking(fake, lang)
    wait_for(events, "approval_request")
    run.decide("call_1", False)
    finish(run, events, thread)
    assert not of(events, "tool_start")
    result = of(events, "tool_result")[0]
    assert not result["ok"] and result["result"] == refusal
    assert fake.started()["refusal"] == refusal
    assert end(events)["stop_reason"] == "no_tool_call"


def test_stopping_while_a_call_waits_answers_it_as_stopped(fake):
    run, events, thread = asking(fake)
    wait_for(events, "approval_request")
    run.stop()
    finish(run, events, thread)
    approval = of(events, "approval")[0]
    assert not approval["approved"] and approval["stopped"] and not approval["timed_out"]
    assert end(events)["stop_reason"] == "user_stopped"


def test_an_unanswered_call_times_out_and_stops_pi(fake, monkeypatch):
    monkeypatch.setattr(pi_harness, "APPROVAL_TIMEOUT", 0.2)
    run, events, thread = asking(fake)
    finish(run, events, thread)
    approval = of(events, "approval")[0]
    assert approval["timed_out"] and not approval["approved"]
    assert end(events)["stop_reason"] == "approval_timeout"


# ---------- stopping ----------

def test_stop_aborts_pi_mid_reply(fake):
    fake.say({"forever": True})
    run, events, thread = start(fake)
    wait_for(events, "delta")
    run.stop()
    finish(run, events, thread)
    assert end(events)["stop_reason"] == "user_stopped"
    assert any(c["type"] == "abort" for c in fake.commands())
    assert of(events, "model_end")[-1]["finish"] == "user_stopped"


def test_pi_that_does_not_stop_when_asked_is_killed(fake, monkeypatch):
    monkeypatch.setattr(pi_harness, "STOP_GRACE", 0.3)
    fake.say({"forever": True}, deaf=True)
    run, events, thread = start(fake)
    wait_for(events, "delta")
    run.stop()
    finish(run, events, thread)
    assert end(events)["stop_reason"] == "user_stopped"


def test_stop_before_pi_has_started_sends_no_question(fake):
    fake.say({"say": [ANSWER]})
    events = []
    run = pi_harness.PiHarness(pi_harness.PiSettings("fake", "fake-model"), [], "hi", events.append, (), fake.folder, "chat1")
    run.stop()
    run.run()
    assert end(events)["stop_reason"] == "user_stopped" and not of(events, "step_start")
    assert "prompt" not in [c["type"] for c in fake.commands()]


def test_the_step_limit_stops_pi_and_no_step_past_it_is_drawn(fake):
    fake.say(*[{"say": [TIME]}] * 4, {"say": [ANSWER]})
    events = finish(*start(fake, pi_harness.PiSettings("fake", "fake-model", max_steps=2)))
    assert [e["step"] for e in of(events, "step_start")] == [1, 2]
    assert end(events)["stop_reason"] == "max_steps" and events[0]["settings"]["max_steps"] == 2


# ---------- what goes wrong ----------

def test_a_failed_request_is_an_error(fake):
    events = run_pi(fake, {"say": [], "stop": "error", "error": "Connection error."})
    assert of(events, "error")[0]["message"] == "pi: Connection error."
    assert end(events)["stop_reason"] == "error" and end(events)["final"] == ""


def test_a_refused_question_is_said_in_both_languages(fake):
    fake.say(refuse="No API key found for the selected model.")
    events = finish(*start(fake))
    error = of(events, "error")[0]
    assert "No API key found" in error["text"]["en"] and "pi 拒絕了 `prompt`" in error["text"]["zh"]
    assert end(events)["stop_reason"] == "error"


def test_a_model_pi_does_not_have_is_an_error(fake):
    fake.say()
    events = finish(*start(fake, pi_harness.PiSettings("fake", "gone")))
    assert "Model not found: fake/gone" in of(events, "error")[0]["text"]["en"]


def test_pi_that_dies_at_start_says_its_last_words(fake):
    fake.say(crash="SyntaxError: Unexpected token (Node.js 20 is too old)")
    events = finish(*start(fake))
    error = of(events, "error")[0]["text"]
    assert "Node.js 20 is too old" in error["en"] and "pi 意外結束了" in error["zh"]


def test_pi_that_is_not_installed_says_how_to_install_it(fake, monkeypatch):
    monkeypatch.setattr(pi_harness, "PI_COMMAND", ["no-such-pi-anywhere"])
    events = finish(*start(fake))
    error = of(events, "error")[0]["text"]
    assert pi_harness.INSTALL in error["en"] and pi_harness.INSTALL in error["zh"] and "PI_COMMAND" in error["en"]


# ---------- what pi can run ----------

def test_probe_lists_the_models_pi_has_and_the_one_it_starts_on(fake):
    fake.say(version="1.4.0")
    found = pi_harness.probe()
    assert found["version"] == "1.4.0" and found["default"] == "fake/fake-model"
    assert [f'{m["provider"]}/{m["id"]}' for m in found["models"]] == ["fake/fake-model"]


def test_probe_without_models_has_no_default(fake):
    fake.say(models=[])
    assert pi_harness.probe() == {"version": "1.4.0", "models": [], "default": None}


def test_probe_of_pi_that_is_not_installed(monkeypatch):
    monkeypatch.setattr(pi_harness, "PI_COMMAND", ["no-such-pi-anywhere"])
    with pytest.raises(pi_harness.PiError) as e:
        pi_harness.probe()
    assert pi_harness.INSTALL in e.value.text["zh"]


# ---------- the real pi ----------

REAL = os.environ.get("RUN_PI_TESTS") == "1"


class ScriptedModel(BaseHTTPRequestHandler):
    """An OpenAI-compatible chat completions endpoint: it calls bash once, then answers from the result."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        has_result = any(m.get("role") == "tool" for m in body["messages"])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def chunk(delta, finish=None):
            return {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "scripted",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        if has_result:
            chunks = [chunk({"role": "assistant", "content": "It says "}), chunk({"content": "hi."}), chunk({}, "stop")]
        else:
            chunks = [chunk({"role": "assistant", "content": "Let me run it."}),
                      chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "echo hi"}'}}]}),
                      chunk({}, "tool_calls")]
        chunks.append({"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "scripted", "choices": [],
                       "usage": {"prompt_tokens": 1500, "completion_tokens": 9, "total_tokens": 1509}})
        for c in chunks:
            self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.02)
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


@pytest.fixture
def real_pi(monkeypatch, tmp_path):
    if not REAL:
        pytest.skip("RUN_PI_TESTS=1 starts the real pi")
    if not (shutil.which(pi_harness.PI_COMMAND[0]) or Path(pi_harness.PI_COMMAND[0]).exists()):
        pytest.skip(f"{pi_harness.PI_COMMAND[0]} is not installed")
    server = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedModel)
    server.requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    agent = tmp_path / "agent"  # pi's own folder, with only the scripted model in it
    agent.mkdir()
    (agent / "models.json").write_text(json.dumps({"providers": {"scripted": {
        "baseUrl": f"http://127.0.0.1:{server.server_port}/v1", "api": "openai-completions", "apiKey": "none",
        "models": [{"id": "scripted-model", "name": "Scripted", "contextWindow": 32000, "maxTokens": 1024,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}]}}}))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent))
    monkeypatch.setenv("PI_OFFLINE", "1")
    yield server
    server.shutdown()


def test_the_real_pi_lists_the_model_it_has(real_pi):
    found = pi_harness.probe()
    assert [f'{m["provider"]}/{m["id"]}' for m in found["models"]] == ["scripted/scripted-model"]


@pytest.mark.parametrize("approve", [False, True])
def test_the_real_pi_runs_a_turn_the_page_can_draw(real_pi, tmp_path, approve):
    events = []
    run = pi_harness.PiHarness(pi_harness.PiSettings("scripted", "scripted-model", approve_tools=approve, lang="en"), [], "Run echo hi", events.append,
                   folder=tmp_path / "chat", session="chat1")
    thread = threading.Thread(target=run.run, daemon=True)
    thread.start()
    if approve:
        run.decide(wait_for(events, "approval_request", timeout=60)["call_id"], True)
    thread.join(60)
    assert end(events)["stop_reason"] == "no_tool_call", [e for e in events if e["type"] == "error"]
    assert end(events)["final"] == "It says hi."
    assert of(events, "tool_result")[0]["result"].strip() == "hi"
    assert of(events, "model_end")[0]["usage"]["prompt"] == 1500
    assert "".join(e["text"] for e in of(events, "delta") if e["step"] == 2) == "It says hi."
    system = real_pi.requests[0]["messages"][0]
    assert system["role"] in ("system", "developer") and "pi" in str(system["content"])
