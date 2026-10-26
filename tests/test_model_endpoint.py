"""harnesses/model_endpoint.py: the page's model served to pi as an OpenAI-compatible chat completions endpoint.

The requests here are what pi's openai-completions provider sends (pi 1.4, as pi_extension.ts registers the
endpoint), made over HTTP on 127.0.0.1 to the endpoint with a FakeLLM behind it.
"""

import base64
import json
import threading
import urllib.error
import urllib.request

import pytest

from fakes import FakeLLM, qwen_tool_call, tool_call
from harness.backends.backend import Sampling
from harness.families import GEMMA4, QWEN3_5
from harness.harnesses import model_endpoint
from harness.harnesses.model_endpoint import conversation, endpoint

SYSTEM = {"role": "system", "content": "You are an expert coding assistant operating inside pi."}
BASH = {"type": "function", "function": {"name": "bash", "description": "Run a command",
                                          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}


@pytest.fixture
def served(tmp_path):
    """The endpoint serving a FakeLLM for a turn whose events are `events`, every request drawn in step 1."""
    state = {"events": [], "stopped": False, "step": 1}

    def serve(outputs, family=None, sampling=None):
        llm = FakeLLM(list(outputs), family=family)
        endpoint().serve(model_endpoint.Serving(llm, sampling or Sampling(), "en", tmp_path, lambda: state["step"],
                                                lambda kind, **data: state["events"].append({"type": kind, **data}), lambda: state["stopped"]))
        return llm

    state["serve"] = serve
    yield state
    endpoint().stop_serving()


def post(body: dict, token: str | None = None) -> tuple[int, list[dict] | dict]:
    """POST a chat completion; the stream's chunks, or the error's JSON."""
    request = urllib.request.Request(endpoint().url + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {token or endpoint().token}"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            lines = [line.decode().strip() for line in response if line.strip()]  # each event ends with a blank line
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    return 200, [json.loads(line[6:]) for line in lines if line.startswith("data: {")] + ([{"done": True}] if lines[-1] == "data: [DONE]" else [])


def ask(messages, tools=(BASH,), thinking=False) -> dict:
    return {"model": "m", "messages": messages, "tools": list(tools) or None, "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": thinking}}


def deltas(chunks, key):
    return "".join(c["choices"][0]["delta"].get(key) or "" for c in chunks if c.get("choices"))


def finish_reason(chunks):
    return next(c["choices"][0]["finish_reason"] for c in chunks if c.get("choices") and c["choices"][0]["finish_reason"])


# ---------- who it answers ----------

def test_it_listens_on_this_machine_alone():
    assert endpoint().url.startswith("http://127.0.0.1:") and endpoint().url.endswith("/v1")


def test_it_answers_only_with_its_token(served):
    served["serve"](["Hi.<turn|>"])
    status, error = post(ask([SYSTEM, {"role": "user", "content": "hi"}]), token="wrong")
    assert status == 401 and "Agent Loop Visualizer" in error["error"]["message"]


def test_it_answers_no_request_while_no_turn_is_served():
    endpoint().stop_serving()
    status, error = post(ask([SYSTEM, {"role": "user", "content": "hi"}]))
    assert status == 503


# ---------- what it sends back ----------

def test_a_reply_streams_as_content_with_its_usage(served):
    served["serve"](["It is 2 pm.<turn|>"])
    status, chunks = post(ask([SYSTEM, {"role": "user", "content": "What time is it?"}]))
    assert status == 200 and chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert deltas(chunks, "content") == "It is 2 pm." and finish_reason(chunks) == "stop"
    assert chunks[-1] == {"done": True}  # [DONE]
    usage, context, done = chunks[-2]["usage"], served["events"][0], served["events"][-1]
    assert usage["completion_tokens"] == done["n_tokens"] and usage["prompt_tokens"] == context["n_tokens"]


def test_a_tool_call_in_the_models_format_goes_back_as_tool_calls(served):
    served["serve"]([tool_call("bash", command="date")])
    _, chunks = post(ask([SYSTEM, {"role": "user", "content": "Time?"}]))
    [call] = next(c["choices"][0]["delta"]["tool_calls"] for c in chunks if c.get("choices") and c["choices"][0]["delta"].get("tool_calls"))
    assert call["function"] == {"name": "bash", "arguments": '{"command": "date"}'} and call["id"].startswith("call_")
    assert finish_reason(chunks) == "tool_calls" and deltas(chunks, "content") == ""  # the call's markup is not the reply


def test_qwens_tool_call_too(served):
    served["serve"]([qwen_tool_call("bash", command="date")], family=QWEN3_5)
    _, chunks = post(ask([SYSTEM, {"role": "user", "content": "Time?"}]))
    assert finish_reason(chunks) == "tool_calls"


def test_thinking_streams_as_reasoning_content(served):
    llm = served["serve"](["Use the clock.<channel|>Two.<turn|>"])
    _, chunks = post(ask([SYSTEM, {"role": "user", "content": "Time?"}], thinking=True))
    assert llm.renders[0]["thinking"] is True
    assert deltas(chunks, "reasoning_content") == "Use the clock." and deltas(chunks, "content") == "Two."


def test_a_call_the_family_cannot_parse_is_not_sent(served):
    served["serve"](["<|tool_call>call:bash{command:<|tool_call|>"])
    _, chunks = post(ask([SYSTEM, {"role": "user", "content": "Time?"}]))
    assert not any(c.get("choices") and c["choices"][0]["delta"].get("tool_calls") for c in chunks)


def test_the_sidebars_token_limit_is_length(served):
    served["serve"](["A long reply that goes on."], sampling=Sampling(max_new_tokens=3))
    _, chunks = post(ask([SYSTEM, {"role": "user", "content": "Talk."}]))
    assert finish_reason(chunks) == "length"


def test_a_backend_that_fails_says_so_in_the_finish_reason(served):
    served["serve"]([RuntimeError("the vLLM server went away")])
    _, chunks = post(ask([SYSTEM, {"role": "user", "content": "hi"}]))
    assert finish_reason(chunks) == "error: RuntimeError: the vLLM server went away"


# ---------- what the page draws ----------

def test_the_request_is_drawn_as_the_steps_context_prefill_tokens_and_end(served):
    served["serve"](["It is 2 pm.<turn|>"])
    post(ask([SYSTEM, {"role": "user", "content": "What time is it?"}]))
    events = served["events"]
    kinds = [e["type"] for e in events]
    assert kinds[:2] == ["context", "prefill"] and kinds[-1] == "model_end" and set(kinds[2:-1]) == {"token"}
    assert all(e["step"] == 1 for e in events)
    context = events[0]
    assert "operating inside pi" in context["prompt"] and context["roles"] == {"system": 1, "user": 1} and context["added_from"] == 0
    assert events[-1]["raw"] == "It is 2 pm.<turn|>" and events[-1]["stop_piece"] == "<turn|>"


def test_what_is_new_in_a_step_is_the_tool_result(served):
    served["serve"](["It is 2 pm.<turn|>"])
    call = {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"date"}'}}
    post(ask([SYSTEM, {"role": "user", "content": "Time?"}, {"role": "assistant", "content": None, "tool_calls": [call]},
              {"role": "tool", "tool_call_id": "call_1", "content": "14:00"}]))
    context = served["events"][0]
    new = context["prompt"][context["added_from"]:]
    assert "14:00" in new and "Time?" not in new


def test_a_request_outside_every_step_is_answered_but_not_drawn(served):
    served["serve"](["A summary.<turn|>"])
    served["step"] = None
    status, chunks = post(ask([SYSTEM, {"role": "user", "content": "Summarize."}]))
    assert status == 200 and deltas(chunks, "content") == "A summary." and served["events"] == []


def test_a_stopped_turn_stops_the_model_before_its_next_token(served):
    served["serve"](["It is 2 pm.<turn|>"])
    served["stopped"] = True
    post(ask([SYSTEM, {"role": "user", "content": "Time?"}]))
    assert served["events"][-1]["finish"] == "user_stopped" and not [e for e in served["events"] if e["type"] == "token"]


# ---------- the conversation, as the chat templates take it ----------

def test_the_conversation_is_the_one_simple_harness_keeps(tmp_path):
    call = {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"date"}'}}
    body = ask([SYSTEM, {"role": "system", "content": "More rules."}, {"role": "user", "content": [{"type": "text", "text": "Time?"}]},
                {"role": "assistant", "content": "Let me see.", "tool_calls": [call], "reasoning_content": "Use date."},
                {"role": "tool", "tool_call_id": "call_1", "content": "14:00"}])
    gemma, qwen = (conversation(body, FakeLLM(family=f), tmp_path) for f in (GEMMA4, QWEN3_5))
    assert gemma[0] == {"role": "system", "content": SYSTEM["content"] + "\n\nMore rules."}
    assert gemma[1] == {"role": "user", "content": "Time?"}
    assert gemma[2]["tool_calls"] == [{"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": {"command": "date"}}}]
    assert gemma[2]["content"] == ("Let me see." if GEMMA4.text_with_calls else "")
    assert gemma[2][GEMMA4.reasoning_key] == "Use date." and qwen[2][QWEN3_5.reasoning_key] == "Use date."
    assert gemma[3] == {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "14:00"}


def test_an_image_is_saved_once_under_its_hash_and_becomes_a_part(tmp_path):
    url = "data:image/png;base64," + base64.b64encode(b"\x89PNG fake").decode()
    body = ask([{"role": "user", "content": [{"type": "text", "text": "What is this?"}, {"type": "image_url", "image_url": {"url": url}}]}])
    first = conversation(body, FakeLLM(), tmp_path)[0]["content"]
    second = conversation(body, FakeLLM(), tmp_path)[0]["content"]
    assert first == second and first[0] == {"type": "text", "text": "What is this?"}
    assert first[1]["type"] == "image" and first[1]["size"] == 9 and len(list((tmp_path / "pi-images").iterdir())) == 1


def test_one_request_at_a_time(served):
    served["serve"](["One.<turn|>", "Two.<turn|>"])
    results = []
    threads = [threading.Thread(target=lambda: results.append(post(ask([SYSTEM, {"role": "user", "content": "hi"}])))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert sorted(deltas(chunks, "content") for _, chunks in results) == ["One.", "Two."]
