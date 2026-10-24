"""The Qwen3.5 family: its parser, and the harness running a Qwen-format conversation."""

import pytest

from families import QWEN3_5
from families.qwen3_5 import convert, parse_output
from fakes import qwen_call, qwen_tool_call, record_run
from tools import TOOLS

SCHEMAS = {t.name: t.parameters for t in TOOLS.values()}
TYPED = {"f": {"properties": {
    "n": {"type": "integer"}, "x": {"type": "number"}, "ok": {"type": "boolean"},
    "tags": {"type": "array"}, "opts": {"type": "object"}, "s": {"type": "string"},
}}}


# ---------- parse_output: splitting one model turn ----------

def test_plain_reply_strips_stop_marker():
    assert parse_output("台北現在是下午三點。<|im_end|>") == {"thinking": None, "tool_calls": [], "content": "台北現在是下午三點。"}


@pytest.mark.parametrize("marker", ["<|im_end|>", "<|endoftext|>"])
def test_every_end_marker_is_removed(marker):
    assert parse_output(f"hi{marker}")["content"] == "hi"


def test_tool_call_is_parsed_and_removed_from_content():
    raw = "<tool_call>\n<function=get_current_time>\n<parameter=timezone>\nAsia/Taipei\n</parameter>\n</function>\n</tool_call><|im_end|>"
    out = parse_output(raw, SCHEMAS)
    assert out["content"] == ""
    [call] = out["tool_calls"]
    assert call["name"] == "get_current_time" and call["arguments"] == {"timezone": "Asia/Taipei"} and call["error"] is None
    assert call["raw_args"] == "\n<parameter=timezone>\nAsia/Taipei\n</parameter>\n"


def test_text_before_a_call_stays_as_content():
    out = parse_output("我來算一下。\n\n" + qwen_tool_call("calculator", expression="1+1"), SCHEMAS)
    assert out["content"] == "我來算一下。"
    assert out["tool_calls"][0]["arguments"] == {"expression": "1+1"}


def test_several_calls_keep_their_order():
    raw = qwen_call("calculator", {"expression": "1"}) + "\n" + qwen_call("get_current_time", {"timezone": "UTC"}) + "<|im_end|>"
    assert [c["name"] for c in parse_output(raw, SCHEMAS)["tool_calls"]] == ["calculator", "get_current_time"]


def test_values_keep_their_lines_and_symbols():
    """Values are plain text: quotes, braces and inner newlines need no escaping."""
    expr = 'say "hi" {x}\n<b>2</b>'
    [call] = parse_output(qwen_call("calculator", {"expression": expr}), SCHEMAS)["tool_calls"]
    assert call["arguments"] == {"expression": expr}


def test_values_take_their_type_from_the_schema():
    args = {"n": 3, "x": 2.5, "ok": True, "tags": ["a", 1], "opts": {"k": [1]}, "s": "42"}
    [call] = parse_output(qwen_call("f", args), TYPED)["tool_calls"]
    assert call["arguments"] == args  # "42" stays a string because the schema says string


@pytest.mark.parametrize("value, kind, expected", [
    ("3", "integer", 3), ("abc", "integer", "abc"), ("False", "boolean", False), ("yes", "boolean", "yes"),
    ("[1, 2]", "array", [1, 2]), ("3", None, "3"), ("3", "string", "3"),
])
def test_convert(value, kind, expected):
    assert convert(value, kind) == expected


def test_without_a_schema_values_stay_text():
    [call] = parse_output(qwen_call("unknown_tool", {"n": 3}))["tool_calls"]
    assert call["arguments"] == {"n": "3"}


def test_thinking_is_split_out():
    out = parse_output("<think>\n先查時間\n</think>\n\n" + qwen_tool_call("get_current_time"), SCHEMAS)
    assert out["thinking"] == "先查時間" and out["content"] == "" and out["tool_calls"][0]["name"] == "get_current_time"


def test_thinking_cut_off_by_the_token_limit():
    assert parse_output("<think>\n想到一半") == {"thinking": "想到一半", "tool_calls": [], "content": ""}


def test_unterminated_call_is_a_format_error():
    out = parse_output("好。<tool_call>\n<function=calculator>\n<parameter=expression>\n1+", SCHEMAS)
    [call] = out["tool_calls"]
    assert call["name"] == "calculator" and call["arguments"] is None and "</tool_call>" in call["error"]
    assert out["content"] == "好。"


def test_call_without_a_function_is_a_format_error():
    [call] = parse_output("<tool_call>\n{\"name\": \"calculator\"}\n</tool_call>")["tool_calls"]
    assert call["name"] == "?" and call["arguments"] is None and "<function=" in call["error"]


def test_a_missing_parameter_close_still_reads():
    raw = "<tool_call>\n<function=read_wikipedia>\n<parameter=title>\n台北\n<parameter=lang>\nzh\n</function>\n</tool_call>"
    [call] = parse_output(raw, SCHEMAS)["tool_calls"]
    assert call["arguments"] == {"title": "台北", "lang": "zh"}


# ---------- the harness speaking Qwen ----------

def test_a_qwen_tool_run_end_to_end():
    run, events, fake = record_run(
        ["我查一下。\n\n" + qwen_tool_call("calculator", expression="6*7"), "答案是 42。<|im_end|>"], family=QWEN3_5,
    )
    assert events[0]["family"] == "qwen3_5" and events[-1]["stop_reason"] == "no_tool_call"
    assistant = run.messages[1]
    assert assistant["content"] == "我查一下。"  # Qwen's template keeps text in front of a call
    assert assistant["tool_calls"][0]["function"] == {"name": "calculator", "arguments": {"expression": "6*7"}}
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["ok"] and '"result": 42' in result["result"]
    second = [e for e in events if e["type"] == "context"][1]
    assert second["prompt"][second["added_from"]:].startswith("<|im_start|>user\n<tool_response>\n")
    done = [e for e in events if e["type"] == "model_end"]
    assert [d["stop_piece"] for d in done] == ["<|im_end|>", "<|im_end|>"]  # both kinds of turn stop on the same token


def test_qwen_thinking_is_kept_as_reasoning_content():
    run, events, fake = record_run(
        ["先算\n</think>\n\n" + qwen_tool_call("calculator", expression="1"), "好\n</think>\n\n是 1。<|im_end|>"],
        family=QWEN3_5, thinking=True,
    )
    assert fake.prompts[0].endswith("<think>\n")
    assert run.messages[1]["reasoning_content"] == "先算" and "reasoning" not in run.messages[1]
    parsed = [e for e in events if e["type"] == "parsed"]
    assert [p["thinking"] for p in parsed] == ["先算", "好"] and parsed[1]["content"] == "是 1。"


def test_errors_are_worded_in_the_language_asked_for():
    [cut] = parse_output("<tool_call>\n<function=calculator>", SCHEMAS, "en")["tool_calls"]
    assert cut["error"] == "The tool call has no closing </tool_call>"
    [bad] = parse_output("<tool_call>\n{}\n</tool_call>", lang="en")["tool_calls"]
    assert bad["error"] == "The tool call has no <function=name>"
