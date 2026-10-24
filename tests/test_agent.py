"""The harness loop, driven by a fake model that replays scripted outputs."""

import json
import threading
import time

import pytest

import agent
from agent import MAX_FORMAT_ERRORS, MAX_RESULT_CHARS, Run, RunSettings, common_prefix
from fakes import FakeLLM, record_run, tool_call
from llm import Sampling
from tools import TOOLS, ToolError

ANSWER = "現在是下午三點。<turn|>"


def kinds(events):
    """Event types with each run of tokens collapsed to one 'token*'."""
    out = []
    for ev in events:
        kind = "token*" if ev["type"] == "token" else ev["type"]
        if not (kind == "token*" and out and out[-1] == kind):
            out.append(kind)
    return out


def of_type(events, kind):
    return [e for e in events if e["type"] == kind]


def end_of(events):
    [end] = of_type(events, "run_end")
    assert events[-1] is end
    return end


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.01)


# ---------- a run that ends with a reply ----------

def test_plain_reply_stops_without_a_tool_call():
    run, events, _ = record_run([ANSWER])
    assert kinds(events) == ["run_start", "step_start", "context", "prefill", "token*", "model_end", "parsed", "run_end"]
    end = end_of(events)
    assert end["stop_reason"] == "no_tool_call"
    assert end["final"] == "現在是下午三點。"
    assert end["messages"] == [
        {"role": "user", "content": "台北現在幾點？"},
        {"role": "assistant", "content": "現在是下午三點。"},
    ]


def test_run_start_reports_settings_and_tools():
    _, events, _ = record_run([ANSWER], sampling=Sampling(0.5, 0.9, 10, 256), max_steps=3, thinking=True, approve_tools=True)
    start = events[0]
    assert start["type"] == "run_start" and len(start["run_id"]) == 12
    assert start["settings"] == {
        "temperature": 0.5, "top_p": 0.9, "top_k": 10, "max_new_tokens": 256,
        "max_steps": 3, "thinking": True, "use_tools": True, "approve_tools": True,
    }
    assert start["tools"] == list(TOOLS)


def test_every_event_has_a_rising_timestamp():
    _, events, _ = record_run([tool_call("calculator", expression="1+1"), ANSWER])
    times = [e["t"] for e in events]
    assert all(isinstance(t, int) for t in times)
    assert times == sorted(times)


def test_settings_reach_the_model():
    sampling = Sampling(temperature=0.0, top_p=0.5, top_k=3, max_new_tokens=99)
    _, _, fake = record_run([ANSWER], sampling=sampling)
    assert fake.samplings == [sampling]


def test_context_event_describes_the_prompt():
    _, events, fake = record_run([ANSWER])
    [ctx] = of_type(events, "context")
    assert ctx["prompt"] == fake.prompts[0]
    assert ctx["n_tokens"] == len(fake.tokenizer(ctx["prompt"]).input_ids)
    assert ctx["context_length"] == fake.context_length
    assert ctx["added_from"] == 0
    assert ctx["roles"] == {"system": 1, "user": 1}


def test_history_comes_before_the_new_question():
    history = [{"role": "user", "content": "嗨"}, {"role": "assistant", "content": "你好！"}]
    _, events, fake = record_run([ANSWER], history=history)
    assert fake.renders[0]["messages"][1:] == [*history, {"role": "user", "content": "台北現在幾點？"}]
    assert end_of(events)["messages"][:2] == history
    assert of_type(events, "context")[0]["roles"] == {"system": 1, "user": 2, "assistant": 1}


# ---------- a tool call goes round the loop ----------

def test_tool_call_round_trip():
    run, events, fake = record_run([tool_call("calculator", expression="6*7"), "答案是 42。<turn|>"])
    assert kinds(events) == [
        "run_start",
        "step_start", "context", "prefill", "token*", "model_end", "parsed", "tool_call", "tool_start", "tool_result",
        "step_start", "context", "prefill", "token*", "model_end", "parsed",
        "run_end",
    ]
    [call] = of_type(events, "tool_call")
    assert call["step"] == 1 and call["call_id"] == "call_1_0" and call["name"] == "calculator"
    assert call["arguments"] == {"expression": "6*7"}
    assert call["errors"] == [] and call["repeated"] is False
    assert call["environment"] == TOOLS["calculator"].environment

    [result] = of_type(events, "tool_result")
    assert result["ok"] is True and result["truncated"] is False and result["ms"] >= 0
    assert json.loads(result["result"]) == {"expression": "6*7", "result": 42}

    end = end_of(events)
    assert end["stop_reason"] == "no_tool_call" and end["final"] == "答案是 42。"
    assert end["messages"] == [
        {"role": "user", "content": "台北現在幾點？"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1_0", "type": "function", "function": {"name": "calculator", "arguments": {"expression": "6*7"}}},
        ]},
        {"role": "tool", "tool_call_id": "call_1_0", "name": "calculator", "content": result["result"]},
        {"role": "assistant", "content": "答案是 42。"},
    ]


def test_observation_is_in_the_next_prompt():
    _, events, fake = record_run([tool_call("calculator", expression="6*7"), ANSWER])
    [result] = of_type(events, "tool_result")
    assert result["result"] in fake.prompts[1]
    assert result["result"] not in fake.prompts[0]


def test_added_from_marks_where_the_new_context_starts():
    # the real template's behavior (sorted arguments, thinking) is covered in test_chat_template.py
    _, events, fake = record_run([tool_call("calculator", expression="6*7"), ANSWER])
    ctx1, ctx2 = of_type(events, "context")
    assert ctx2["prompt"][: ctx2["added_from"]].startswith(ctx1["prompt"])
    assert ctx2["prompt"][ctx2["added_from"]:].startswith("response:calculator")


def test_text_before_a_tool_call_is_shown_but_kept_out_of_history():
    _, events, _ = record_run(["我來算一下。" + tool_call("calculator", expression="6*7"), ANSWER])
    assert of_type(events, "parsed")[0]["content"] == "我來算一下。"  # the step card shows it
    assert end_of(events)["messages"][1]["content"] == ""


def test_two_calls_in_one_step_both_run():
    two = (
        "<|tool_call>call:get_current_time{timezone:<|\"|>Asia/Taipei<|\"|>}<tool_call|>"
        "<|tool_call>call:get_current_time{timezone:<|\"|>America/New_York<|\"|>}<tool_call|><|tool_response>"
    )
    _, events, _ = record_run([two, ANSWER])
    assert [c["call_id"] for c in of_type(events, "tool_call")] == ["call_1_0", "call_1_1"]
    results = of_type(events, "tool_result")
    assert [json.loads(r["result"])["timezone"] for r in results] == ["Asia/Taipei", "America/New_York"]
    assistant = end_of(events)["messages"][1]
    assert [c["id"] for c in assistant["tool_calls"]] == ["call_1_0", "call_1_1"]


def test_thinking_is_parsed_and_kept_as_reasoning():
    thought_then_call = "先算一下<channel|>" + tool_call("calculator", expression="2**8")
    _, events, fake = record_run([thought_then_call, "想好了<channel|>256<turn|>"], thinking=True)
    assert all(r["thinking"] for r in fake.renders)
    parsed = of_type(events, "parsed")
    assert parsed[0]["thinking"] == "先算一下" and parsed[0]["tool_calls"][0]["name"] == "calculator"
    assert parsed[1] == {**parsed[1], "thinking": "想好了", "content": "256"}
    messages = end_of(events)["messages"]
    assert messages[1]["reasoning"] == "先算一下"
    assert messages[-1] == {"role": "assistant", "content": "256"}


def test_without_tools_the_model_sees_none_and_calls_are_ignored():
    _, events, fake = record_run([tool_call("calculator", expression="1+1")], use_tools=False)
    assert fake.renders[0]["tools"] is None
    assert fake.renders[0]["messages"][0]["content"] == agent.SYSTEM_NO_TOOLS
    assert events[0]["tools"] == []
    assert of_type(events, "parsed")[0]["tool_calls"] == []
    assert of_type(events, "tool_call") == []
    assert end_of(events)["stop_reason"] == "no_tool_call"


def test_with_tools_the_model_sees_every_schema():
    _, _, fake = record_run([ANSWER])
    assert fake.renders[0]["tools"] == [t.schema() for t in TOOLS.values()]
    assert fake.renders[0]["messages"][0]["content"] == agent.SYSTEM_WITH_TOOLS


# ---------- the harness checks the call before running it ----------

def test_invalid_arguments_become_an_error_observation():
    _, events, _ = record_run([tool_call("calculator"), ANSWER])
    [call] = of_type(events, "tool_call")
    assert call["errors"] == ["缺少必要參數 expression"]
    assert of_type(events, "tool_start") == []
    [result] = of_type(events, "tool_result")
    assert result["ok"] is False and result["ms"] == 0
    assert json.loads(result["result"]) == {"error": "缺少必要參數 expression"}
    assert end_of(events)["stop_reason"] == "no_tool_call"  # the loop went on to step 2


def test_unknown_tool_lists_the_real_ones():
    _, events, _ = record_run([tool_call("send_email", to="boss"), ANSWER])
    [call] = of_type(events, "tool_call")
    assert call["environment"] is None
    assert "沒有 send_email 這個工具" in call["errors"][0] and "calculator" in call["errors"][0]
    assert of_type(events, "tool_start") == []


def test_defaults_are_filled_before_running():
    _, events, _ = record_run([tool_call("get_current_time"), ANSWER])
    [call] = of_type(events, "tool_call")
    assert call["arguments"] == {"timezone": "Asia/Taipei"}
    assert json.loads(of_type(events, "tool_result")[0]["result"])["timezone"] == "Asia/Taipei"


def test_repeated_call_is_flagged_after_defaults_are_filled():
    outputs = [tool_call("get_current_time"), tool_call("get_current_time", timezone="Asia/Taipei"), tool_call("calculator", expression="1"), ANSWER]
    _, events, _ = record_run(outputs)
    assert [c["repeated"] for c in of_type(events, "tool_call")] == [False, True, False]


def test_tool_error_is_reported_to_the_model():
    _, events, _ = record_run([tool_call("calculator", expression="1/0"), ANSWER])
    [result] = of_type(events, "tool_result")
    assert result["ok"] is False
    assert "division by zero" in json.loads(result["result"])["error"]
    assert of_type(events, "tool_start")  # it did run


@pytest.mark.parametrize("expression", ["(-8) ** (1/3)", "10**999 * 10**999 * 10**999", "factorial()"])
def test_calculator_edge_cases_do_not_end_the_run(expression):
    _, events, _ = record_run([tool_call("calculator", expression=expression), ANSWER])
    assert of_type(events, "error") == []
    [result] = of_type(events, "tool_result")
    assert result["ok"] is False and "error" in json.loads(result["result"])
    assert end_of(events)["stop_reason"] == "no_tool_call"


def test_unexpected_exception_in_a_tool_is_caught(monkeypatch):
    def boom(expression):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(TOOLS["calculator"], "fn", boom)
    _, events, _ = record_run([tool_call("calculator", expression="1"), ANSWER])
    assert json.loads(of_type(events, "tool_result")[0]["result"]) == {"error": "RuntimeError: disk on fire"}
    assert end_of(events)["stop_reason"] == "no_tool_call"


def test_long_results_are_truncated(monkeypatch):
    monkeypatch.setattr(TOOLS["calculator"], "fn", lambda expression: {"result": "x" * 5000})
    _, events, _ = record_run([tool_call("calculator", expression="1"), ANSWER])
    [result] = of_type(events, "tool_result")
    assert result["truncated"] is True
    assert result["result"] == json.dumps({"result": "x" * 5000})[:MAX_RESULT_CHARS] + "…(truncated)"
    assert end_of(events)["messages"][2]["content"] == result["result"]


# ---------- the ways a run stops ----------

def test_step_limit():
    _, events, fake = record_run([tool_call("calculator", expression="1")] * 5, max_steps=2)
    end = end_of(events)
    assert end["stop_reason"] == "max_steps" and end["final"] == ""
    assert [e["step"] for e in of_type(events, "step_start")] == [1, 2]
    assert len(fake.prompts) == 2


def test_token_limit_without_a_tool_call():
    _, events, _ = record_run(["這是一段很長很長的回覆"], sampling=Sampling(max_new_tokens=4))
    end = end_of(events)
    assert end["stop_reason"] == "max_new_tokens"
    assert end["final"] == "這是一段"
    assert of_type(events, "model_end")[0]["finish"] == "max_new_tokens"


def test_truncated_tool_call_counts_as_a_format_error():
    cut = tool_call("calculator", expression="12345")[:-20]
    _, events, _ = record_run([cut, ANSWER], sampling=Sampling(max_new_tokens=len(cut)))
    [call] = of_type(events, "tool_call")
    assert call["errors"] and call["errors"][0].startswith("無法解析工具呼叫")


def test_repeated_format_errors_stop_the_loop():
    bad = "<|tool_call>call:calculator{expression}<tool_call|><|tool_response>"
    _, events, _ = record_run([bad] * 5)
    assert end_of(events)["stop_reason"] == "format_errors"
    assert len(of_type(events, "step_start")) == MAX_FORMAT_ERRORS


def test_a_good_call_resets_the_format_error_count():
    bad = "<|tool_call>call:calculator{expression}<tool_call|><|tool_response>"
    good = tool_call("calculator", expression="1")
    _, events, _ = record_run([bad, bad, good, bad, bad, ANSWER], max_steps=8)
    assert end_of(events)["stop_reason"] == "no_tool_call"


def test_model_crash_is_reported_and_the_run_still_ends():
    _, events, _ = record_run([RuntimeError("MPS out of memory")])
    [error] = of_type(events, "error")
    assert error["message"] == "RuntimeError: MPS out of memory"
    end = end_of(events)
    assert end["stop_reason"] == "error" and end["final"] == ""


def test_stop_before_generation_ends_the_run():
    fake = FakeLLM([ANSWER])
    events = []
    run = Run(fake, [], "hi", RunSettings(), events.append)
    run.stop()
    run.run()
    assert of_type(events, "model_end")[0]["finish"] == "user_stopped"
    assert end_of(events)["stop_reason"] == "user_stopped"


# ---------- the human in the loop ----------

def start_in_thread(outputs, **settings):
    fake = FakeLLM(list(outputs))
    events = []
    run = Run(fake, [], "幾點？", RunSettings(**settings), events.append)
    thread = threading.Thread(target=run.run, daemon=True)
    thread.start()
    return run, events, thread


def test_approved_call_runs():
    run, events, thread = start_in_thread([tool_call("get_current_time"), ANSWER], approve_tools=True)
    wait_for(lambda: of_type(events, "approval_request"))
    assert thread.is_alive() and of_type(events, "tool_start") == []
    [request] = of_type(events, "approval_request")
    assert request["call_id"] == "call_1_0" and request["arguments"] == {"timezone": "Asia/Taipei"}

    run.decide("call_1_0", True)
    thread.join(5)
    assert kinds(events)[kinds(events).index("tool_call"):][:5] == ["tool_call", "approval_request", "approval", "tool_start", "tool_result"]
    assert of_type(events, "approval")[0]["approved"] is True
    assert of_type(events, "tool_result")[0]["ok"] is True
    assert end_of(events)["stop_reason"] == "no_tool_call"


def test_rejected_call_is_not_run_and_the_model_is_told():
    run, events, thread = start_in_thread([tool_call("get_current_time"), ANSWER], approve_tools=True)
    wait_for(lambda: of_type(events, "approval_request"))
    run.decide("call_1_0", False)
    thread.join(5)
    assert of_type(events, "approval")[0]["approved"] is False
    assert of_type(events, "approval")[0]["stopped"] is False
    assert of_type(events, "tool_start") == []
    [result] = of_type(events, "tool_result")
    assert result["ok"] is False and "拒絕" in result["result"]
    assert end_of(events)["messages"][2]["content"] == result["result"]


def test_stop_while_waiting_for_approval():
    run, events, thread = start_in_thread([tool_call("get_current_time"), ANSWER], approve_tools=True)
    wait_for(lambda: of_type(events, "approval_request"))
    run.stop()
    thread.join(5)
    assert not thread.is_alive()
    [approval] = of_type(events, "approval")
    assert approval["approved"] is False and approval["stopped"] is True and approval["timed_out"] is False
    assert of_type(events, "tool_start") == []
    end = end_of(events)
    assert end["stop_reason"] == "user_stopped"
    # the next turn's history says the run was stopped, not that the user rejected the call
    assert json.loads(end["messages"][2]["content"]) == {"error": "使用者停止了這次執行，這個工具呼叫沒有執行"}


def test_unanswered_approval_times_out(monkeypatch):
    monkeypatch.setattr(agent, "APPROVAL_TIMEOUT", 0.1)
    run, events, thread = start_in_thread([tool_call("get_current_time"), ANSWER], approve_tools=True)
    thread.join(5)
    assert not thread.is_alive()
    [approval] = of_type(events, "approval")
    assert approval["approved"] is False and approval["timed_out"] is True and approval["stopped"] is False
    assert of_type(events, "tool_start") == []
    end = end_of(events)
    assert end["stop_reason"] == "approval_timeout"
    assert json.loads(end["messages"][2]["content"]) == {"error": "等待使用者確認逾時，這個工具呼叫沒有執行"}


def test_answered_approval_is_not_a_timeout():
    run, events, thread = start_in_thread([tool_call("get_current_time"), ANSWER], approve_tools=True)
    wait_for(lambda: of_type(events, "approval_request"))
    run.decide("call_1_0", False)
    thread.join(5)
    assert of_type(events, "approval")[0]["timed_out"] is False


def test_no_approval_needed_when_the_setting_is_off():
    _, events, _ = record_run([tool_call("get_current_time"), ANSWER])
    assert of_type(events, "approval_request") == []


def test_invalid_calls_are_rejected_before_asking():
    _, events, _ = record_run([tool_call("calculator"), ANSWER], approve_tools=True)
    assert of_type(events, "approval_request") == []


# ---------- helpers ----------

@pytest.mark.parametrize("a, b, n", [("", "", 0), ("abc", "abd", 2), ("abc", "abc", 3), ("ab", "abcd", 2), ("x", "y", 0)])
def test_common_prefix(a, b, n):
    assert common_prefix(a, b) == n


def test_tool_error_class_is_shared():
    assert issubclass(ToolError, Exception)
