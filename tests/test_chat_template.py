"""The harness, the parsers and the UI all assume things about each model's real tokenizer
and chat template. These tests check those assumptions against the real files, for every
model in the menu. Only the tokenizer is loaded (a few seconds), and the processor for the
tests with media; a model not yet downloaded into models/ (simple_backend.MODELS_DIR) is skipped."""

import os
import re
from pathlib import Path

import pytest
from huggingface_hub import try_to_load_from_cache

import families
import render
import media
import simple_backend
from agent import SYSTEM_NO_TOOLS, SYSTEM_WITH_TOOLS, Run, RunSettings
from fakes import FakeLLM, gemma_call, qwen_call
from html_check import assert_balanced
from simple_backend import MODELS_DIR, LocalLLM
from tools import TOOLS

SAMPLES = Path(__file__).resolve().parent.parent / "assets" / "examples"

# Per family: where a tool call sits in a rendered prompt, how the model ends a turn that calls
# a tool, and how the next prompt begins the tool's result.
CALL_RE = {"gemma4": re.compile(r"<\|tool_call>.*?<tool_call\|>", re.S), "qwen3_5": re.compile(r"<tool_call>.*?</tool_call>", re.S)}
CALL_END = {"gemma4": "<|tool_response>", "qwen3_5": "<|im_end|>"}
OBSERVATION = {"gemma4": "response:{name}{{value:", "qwen3_5": "<|im_start|>user\n<tool_response>\n"}


@pytest.fixture(scope="module", params=list(families.MODELS))
def model(request):
    """A LocalLLM with the real tokenizer and family but no weights."""
    from transformers import AutoConfig, AutoTokenizer

    # Look for the files themselves: with only config.json cached, local_files_only=True fails on Gemma's
    # tokenizer with a ValueError and loads Qwen's empty, rather than raising an OSError.
    revision = families.REVISIONS.get(request.param)
    files = [try_to_load_from_cache(request.param, f, cache_dir=MODELS_DIR, revision=revision) for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")]
    if not all(isinstance(f, str) for f in files):
        pytest.skip(f"{request.param} at {revision} is not downloaded into {MODELS_DIR}")
    folder = os.path.dirname(files[0])
    m = LocalLLM.__new__(LocalLLM)
    m.tokenizer = AutoTokenizer.from_pretrained(folder)
    m.config = AutoConfig.from_pretrained(folder)
    m.family = families.for_model_type(m.config.model_type)
    m.folder, m.model_id = folder, request.param
    return m


@pytest.fixture(scope="module")
def with_processor(model):
    """The same, with the real processor in front of the tokenizer, as LocalLLM loads it, for prompts with media."""
    import torch
    from transformers import AutoProcessor

    m = LocalLLM.__new__(LocalLLM)
    m.processor = AutoProcessor.from_pretrained(model.folder)
    m.tokenizer, m.family, m.device, m.dtype = m.processor.tokenizer, model.family, "cpu", torch.float32
    m.media_token_ids = {kind: getattr(model.config, f"{kind}_token_id", None) for kind in ("image", "video")}
    return m


def system(tools=True):
    return {"role": "system", "content": SYSTEM_WITH_TOOLS if tools else SYSTEM_NO_TOOLS}


def schemas():
    return [t.schema() for t in TOOLS.values()]


def parameters():
    return {t.name: t.parameters for t in TOOLS.values()}


def call_text(family, name, args):
    return gemma_call(name, args) if family.key == "gemma4" else qwen_call(name, args)


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


def test_every_file_in_the_model_folder_is_explained_and_has_the_fields_its_note_names(model):
    folders = [(model.folder, model.family.file_notes)]
    if model.model_id in families.DRAFTERS:
        drafter = families.DRAFTERS[model.model_id]
        config = try_to_load_from_cache(drafter, "config.json", cache_dir=MODELS_DIR, revision=families.REVISIONS.get(drafter))
        if isinstance(config, str):
            folders.append((os.path.dirname(config), model.family.drafter_file_notes))
    for folder, notes in folders:
        files = simple_backend.describe_folder(folder, notes)["files"]
        assert [f["name"] for f in files if render.file_kind(f["name"]) is None] == []
        for f in files:
            if f["name"] in notes:
                assert set(f["fields"]) == set(notes[f["name"]].fields), f["name"]
        assert_balanced(render.files_html({"files": files}, notes, model.family, "en"))


def test_control_tokens_are_single_tokens(model):
    """Every token the family has a note on, stops at or switches channels on is one token id."""
    for piece in {*model.family.special_tokens, *model.family.stop_tokens, *model.family.channel_switch}:
        ids = model.tokenizer(piece, add_special_tokens=False).input_ids
        assert len(ids) == 1, piece
        assert model.piece(ids[0]) == piece


def test_every_special_token_is_explained(model):
    """The model tab lists the tokenizer's control tokens; each one needs a note for it and its hover popover."""
    added = [t.content for t in model.tokenizer.added_tokens_decoder.values()]
    for piece in {*model.tokenizer.all_special_tokens, *added}:
        assert piece in model.family.special_tokens, piece
        assert model.family.special_re.fullmatch(piece), piece


def test_tool_schemas_are_written_into_the_system_turn(model):
    prompt = model.render([system(), {"role": "user", "content": "幾點？"}], schemas(), thinking=False)
    system_turn = prompt[: prompt.index(model.family.user_turn)]
    assert SYSTEM_WITH_TOOLS.strip() in system_turn
    for name in TOOLS:
        assert name in system_turn


def test_no_tools_means_no_declarations(model):
    prompt = model.render([system(False), {"role": "user", "content": "幾點？"}], None, thinking=False)
    assert not any(name in prompt for name in TOOLS)


def test_the_ui_finds_the_latest_user_message(model):
    history = [{"role": "user", "content": "嗨"}, {"role": "assistant", "content": "你好"}]
    prompt = model.render([system(), *history, {"role": "user", "content": "最新的問題"}], schemas(), thinking=False)
    start = prompt.rfind(model.family.user_turn)  # what render.step_html and context_html look for
    assert start > 0 and prompt[start:].startswith(model.family.user_turn + "\n最新的問題")


@pytest.mark.parametrize("name, args", [
    ("get_current_time", {"timezone": "America/New_York"}),
    ("calculator", {"expression": "(37.5 * 12) ** 0.5 + 2 ** 20"}),
    ("search_wikipedia", {"query": 'ReAct: "reason, act" {2022}', "lang": "en"}),
    ("read_wikipedia", {"title": "台北", "lang": "zh"}),
])
def test_tool_calls_round_trip_through_the_template(model, name, args):
    """The call format the template writes into history is exactly what the family's parser reads."""
    prompt = model.render(after_call(name, args), schemas(), thinking=False)
    written = CALL_RE[model.family.key].findall(prompt)[-1]  # Qwen's system turn shows an example call first
    [call] = model.family.parse(written, parameters())["tool_calls"]
    assert call["error"] is None
    assert (call["name"], call["arguments"]) == (name, args)


def test_the_model_ends_a_tool_call_on_a_stop_token(model):
    assert CALL_END[model.family.key] in model.family.stop_tokens


def test_new_context_is_exactly_the_observation(model):
    """The harness renders the history without a generation prompt right after the model's call;
    the next prompt must extend it, and what it adds must be the tool's result."""
    messages = after_call("calculator", {"expression": "6*7"}, '{"result": 42}')
    before = model.render(messages[:3], schemas(), thinking=False, generation_prompt=False)
    after = model.render(messages, schemas(), thinking=False)
    assert after.startswith(before)
    assert after[len(before):].startswith(OBSERVATION[model.family.key].format(name="calculator"))


def test_thinking_mode(model):
    plain = model.render([system(), {"role": "user", "content": "嗨"}], schemas(), thinking=False)
    assert not plain.endswith(model.family.thought_prefix)
    if model.family.key == "gemma4":
        assert "<|think|>" not in plain
        thinking = model.render([system(), {"role": "user", "content": "嗨"}], schemas(), thinking=True)
        assert thinking.startswith("<bos><|turn>system\n<|think|>")
    # after a tool result the template opens the thinking itself; the harness relies on this
    after = model.render(after_call("calculator", {"expression": "1"}), schemas(), thinking=True)
    assert after.endswith(model.family.thought_prefix)


def test_a_token_popover_shows_the_text_the_token_stands_for(model):
    """However the vocabulary spells a token (Qwen's 如果 is å¦Ĥæŀľ), the popover shows its text; a rare
    character split into bytes shows each byte in hex."""
    text = "例如，如果現在是台北市 9 點 52 分\n😀 é"
    ids = model.tokenizer(text, add_special_tokens=False).input_ids
    assert "".join(model.family.show_piece(model.piece(i)) for i in ids) == text.replace(" ", "␣").replace("\n", "↵")
    ids = model.tokenizer("𠀀", add_special_tokens=False).input_ids
    assert len(ids) > 1 and "".join(model.family.show_piece(model.piece(i)) for i in ids) == "<0xF0><0xA0><0x80><0x80>"


def test_every_special_token_in_a_prompt_gets_a_chip(model):
    prompt = model.render(after_call("calculator", {"expression": "1"}), schemas(), thinking=True)
    ids = model.tokenizer(prompt, add_special_tokens=False).input_ids
    special = set(model.tokenizer.all_special_ids) | model.token_ids(model.family.special_tokens)
    pieces = {model.piece(i) for i in ids if i in special}
    assert pieces  # sanity
    for piece in pieces:
        assert model.family.special_re.fullmatch(piece), piece
    assert render.special_html(prompt, model.family, "zh").count('<span class="sp">') == sum(1 for i in ids if i in special)


# ---------- the harness with the real template ----------

class TemplateLLM(FakeLLM):
    """The real chat template with scripted model output."""

    def __init__(self, model, outputs):
        super().__init__(outputs, family=model.family)
        self.real = model

    def render(self, messages, tools, thinking, generation_prompt=True):
        return self.real.render(messages, tools, thinking, generation_prompt)


def scripted(family, before, args, thinking):
    """A model turn: optional thinking, optional text, then one search_wikipedia call."""
    call = call_text(family, "search_wikipedia", args) + CALL_END[family.key]
    if not thinking:
        return before + call
    close = "<channel|>" if family.key == "gemma4" else "\n</think>\n\n"
    open_ = "<|channel>thought\n" if family.key == "gemma4" else ""  # Qwen's template opens <think> itself
    return f"{open_}查一下{close}{before}{call}"


@pytest.mark.parametrize("before, args, thinking", [
    ("", {"query": "ReAct"}, False),
    ("", {"query": "ReAct", "lang": "zh"}, False),  # the template may sort the arguments
    ("", {"query": "ReAct"}, True),  # and rewrite the thought markup
    ("", {"query": "ReAct", "lang": "zh"}, True),
    ("我來查一下。", {"query": "ReAct"}, False),  # and move or drop text before the call
], ids=["one-arg", "two-args", "thinking", "thinking-two-args", "text-before-call"])
def test_step_card_marks_only_the_observation_as_new(model, monkeypatch, before, args, thinking):
    monkeypatch.setattr(TOOLS["search_wikipedia"], "fn", lambda query, lang="en": {"query": query, "results": []})
    family = model.family
    end = "好。" + next(iter(family.stop_tokens))
    events = []
    llm = TemplateLLM(model, [scripted(family, before, args, thinking), end])
    Run(llm, [], "ReAct 是誰提出的？", RunSettings(thinking=thinking), events.append).run()
    [parsed, _] = [e for e in events if e["type"] == "parsed"]
    assert [c["arguments"] for c in parsed["tool_calls"]] == [args]
    first, second = (e for e in events if e["type"] == "context")
    assert second["prompt"][second["added_from"]:].startswith(OBSERVATION[family.key].format(name="search_wikipedia"))
    assert second["prompt"].startswith(first["prompt"][: second["added_from"]])
    # the model writes next: Gemma is still inside its turn, Qwen gets a new assistant turn
    next_turn = {"gemma4": "<tool_response|>", "qwen3_5": "<|im_start|>assistant\n"}[family.key]
    assert second["prompt"].endswith(family.thought_prefix if thinking else "") or not thinking
    assert next_turn in second["prompt"][second["added_from"]:]


class MediaTemplateLLM(TemplateLLM):
    """The real chat template and processor with scripted model output."""

    def encode(self, prompt, parts):
        self.encodes.append(parts)
        return self.real.encode(prompt, parts)


def test_with_media_the_step_card_still_marks_only_the_observation_as_new(with_processor, monkeypatch):
    monkeypatch.setattr(TOOLS["search_wikipedia"], "fn", lambda query, lang="en": {"query": query, "results": []})
    family = with_processor.family
    files = [media.Attachment(name, kind, str(SAMPLES / name), (SAMPLES / name).stat().st_size) for name, kind in
             (("receipt.png", "image"), ("ball.mp4", "video"))]
    events = []
    llm = MediaTemplateLLM(with_processor, [scripted(family, "", {"query": "ReAct"}, False), "好。" + next(iter(family.stop_tokens))])
    Run(llm, [], "這是什麼？", RunSettings(), events.append, files).run()
    first, second = (e for e in events if e["type"] == "context")
    assert first["media_tokens"]["image"] > 100 and first["media_tokens"]["video"] > 100
    assert first["n_tokens"] == len(with_processor.tokenizer(first["prompt"], add_special_tokens=False).input_ids)
    # The expanded media come before the tool result, which is still all that is new in step 2.
    assert second["prompt"][second["added_from"]:].startswith(OBSERVATION[family.key].format(name="search_wikipedia"))
    assert second["prompt"].startswith(first["prompt"][: second["added_from"]])
