"""The harness, the parser and the UI all assume things about Gemma 4's real tokenizer
and chat template. These tests check those assumptions against the real files.
Only the tokenizer is loaded (a few seconds); skipped if it is not in the local cache."""

import os
import re

import pytest

import render
from agent import SYSTEM_NO_TOOLS, SYSTEM_WITH_TOOLS, Run, RunSettings, common_prefix
from fakes import FakeLLM
from gemma_format import STR, parse_output
from llm import CHANNEL_SWITCH, THOUGHT_PREFIX, LocalLLM
from tools import TOOLS

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-E2B-it")
CALL_RE = re.compile(r"<\|tool_call>.*?<tool_call\|>", re.S)


@pytest.fixture(scope="module")
def gemma():
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except OSError:
        pytest.skip(f"{MODEL_ID} is not in the local Hugging Face cache")
    m = LocalLLM.__new__(LocalLLM)  # tokenizer only, no weights
    m.tokenizer = tokenizer
    return m


def system(tools=True):
    return {"role": "system", "content": SYSTEM_WITH_TOOLS if tools else SYSTEM_NO_TOOLS}


def schemas():
    return [t.schema() for t in TOOLS.values()]


def after_call(name, args, result='{"ok": true}'):
    """The messages the harness has after one tool call, built the way agent.Run builds them."""
    return [
        system(),
        {"role": "user", "content": "問題"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1_0", "type": "function", "function": {"name": name, "arguments": args}},
        ]},
        {"role": "tool", "tool_call_id": "call_1_0", "name": name, "content": result},
    ]


@pytest.mark.parametrize("piece", [*CHANNEL_SWITCH, *render.STOP_TOKENS, STR, "<|turn>", "<|think|>"])
def test_control_strings_are_single_tokens(gemma, piece):
    ids = gemma.tokenizer(piece, add_special_tokens=False).input_ids
    assert len(ids) == 1
    assert gemma.piece(ids[0]) == piece


def test_tool_schemas_are_written_into_the_system_turn(gemma):
    prompt = gemma.render([system(), {"role": "user", "content": "幾點？"}], schemas(), thinking=False)
    system_turn = prompt.split("<turn|>")[0]
    assert SYSTEM_WITH_TOOLS in system_turn
    for name in TOOLS:
        assert f"<|tool>declaration:{name}" in system_turn
    assert prompt.endswith("<|turn>model\n")


def test_no_tools_means_no_declarations(gemma):
    prompt = gemma.render([system(False), {"role": "user", "content": "幾點？"}], None, thinking=False)
    assert "<|tool>" not in prompt


def test_the_ui_finds_the_latest_user_message(gemma):
    history = [{"role": "user", "content": "嗨"}, {"role": "assistant", "content": "你好"}]
    prompt = gemma.render([system(), *history, {"role": "user", "content": "最新的問題"}], schemas(), thinking=False)
    start = prompt.rfind("<|turn>user")  # what render.step_html and context_html look for
    assert start > 0 and prompt[start:].startswith("<|turn>user\n最新的問題")


@pytest.mark.parametrize("name, args", [
    ("get_current_time", {"timezone": "America/New_York"}),
    ("calculator", {"expression": "(37.5 * 12) ** 0.5 + 2 ** 20"}),
    ("search_wikipedia", {"query": 'ReAct: "reason, act" {2022}', "lang": "en"}),
    ("read_wikipedia", {"title": "台北", "lang": "zh"}),
])
def test_tool_calls_round_trip_through_the_template(gemma, name, args):
    """The call format the template writes into history is exactly what the parser reads."""
    prompt = gemma.render(after_call(name, args), schemas(), thinking=False)
    [written] = CALL_RE.findall(prompt)
    [call] = parse_output(written)["tool_calls"]
    assert call["error"] is None
    assert (call["name"], call["arguments"]) == (name, args)


def test_tool_result_goes_back_as_a_tool_response(gemma):
    result = '{"timezone": "Asia/Taipei", "datetime": "2026-10-24 15:00:00"}'
    prompt = gemma.render(after_call("get_current_time", {"timezone": "Asia/Taipei"}, result), schemas(), thinking=False)
    assert f"<|tool_response>response:get_current_time{{value:{STR}{result}{STR}}}<tool_response|>" in prompt


def test_new_context_is_exactly_the_observation(gemma):
    """Re-rendering keeps everything the model already read, so the step card's
    'added' highlight (common_prefix with the last prompt + output) is just the result."""
    first = gemma.render(after_call("calculator", {"expression": "6*7"})[:2], schemas(), thinking=False)
    raw = f"<|tool_call>call:calculator{{expression:{STR}6*7{STR}}}<tool_call|><|tool_response>"
    second = gemma.render(after_call("calculator", {"expression": "6*7"}, '{"result": 42}'), schemas(), thinking=False)
    added_from = common_prefix(second, first + raw)
    assert added_from == len(first + raw)
    assert second[added_from:].startswith("response:calculator{value:")


def test_thinking_mode(gemma):
    plain = gemma.render([system(), {"role": "user", "content": "嗨"}], schemas(), thinking=False)
    thinking = gemma.render([system(), {"role": "user", "content": "嗨"}], schemas(), thinking=True)
    assert "<|think|>" not in plain
    assert thinking.startswith("<bos><|turn>system\n<|think|>")
    # after a tool result the template opens the thought channel itself; the harness relies on this
    after = gemma.render(after_call("calculator", {"expression": "1"}), schemas(), thinking=True)
    assert after.endswith(THOUGHT_PREFIX)


def test_every_special_token_in_a_prompt_gets_a_chip(gemma):
    prompt = gemma.render(after_call("calculator", {"expression": "1"}), schemas(), thinking=True)
    ids = gemma.tokenizer(prompt, add_special_tokens=False).input_ids
    special = set(gemma.tokenizer.all_special_ids)
    pieces = {gemma.piece(i) for i in ids if i in special}
    assert pieces  # sanity
    for piece in pieces:
        assert render.SPECIAL_RE.fullmatch(piece), piece


# ---------- the harness with the real template ----------

class TemplateLLM(FakeLLM):
    """The real chat template with scripted model output."""

    def __init__(self, gemma, outputs):
        super().__init__(outputs)
        self.gemma = gemma

    def render(self, messages, tools, thinking):
        return self.gemma.render(messages, tools, thinking)


SEARCH = f"<|tool_call>call:search_wikipedia{{query:{STR}ReAct{STR}}}<tool_call|><|tool_response>"
SEARCH_ZH = f"<|tool_call>call:search_wikipedia{{query:{STR}ReAct{STR},lang:{STR}zh{STR}}}<tool_call|><|tool_response>"


@pytest.mark.parametrize("output, thinking", [
    (SEARCH, False),
    (SEARCH_ZH, False),                                          # the template sorts the arguments
    ("<|channel>thought\n查一下<channel|>" + SEARCH, True),       # and rewrites the thought markup
    ("<|channel>thought\n查一下\n<channel|>" + SEARCH_ZH, True),
    ("我來查一下。" + SEARCH, False),                               # and moves text after the result
], ids=["one-arg", "two-args", "thinking", "thinking-two-args", "text-before-call"])
def test_step_card_marks_only_the_observation_as_new(gemma, monkeypatch, output, thinking):
    monkeypatch.setattr(TOOLS["search_wikipedia"], "fn", lambda query, lang="en": {"query": query, "results": []})
    events = []
    Run(TemplateLLM(gemma, [output, "好。<turn|>"]), [], "ReAct 是誰提出的？", RunSettings(thinking=thinking), events.append).run()
    first, second = (e for e in events if e["type"] == "context")
    assert second["prompt"][second["added_from"]:].startswith("response:search_wikipedia{value:")
    assert second["prompt"].startswith(first["prompt"])
    # the next step is still inside the model's turn, not after a closed <turn|>
    assert second["prompt"].endswith(THOUGHT_PREFIX if thinking else "<tool_response|>")
