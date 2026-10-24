"""End to end with the real weights of every model in the menu (Gemma 4 E2B ~10 GB in memory,
Qwen3.5 0.8B ~2 GB; tens of seconds each), and Gemma's MTP drafter.
Opt in with:  RUN_MODEL_TESTS=1 uv run pytest -m model   (add  -k Qwen  for one model)"""

import os

import pytest
import torch

import families
import render
from agent import SYSTEM_WITH_TOOLS, Run, RunSettings
from llm import LocalLLM, Sampling
from test_render import assert_balanced, render_everything
from tools import TOOLS

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to load the real model"),
]


@pytest.fixture(scope="module", params=list(families.MODELS))
def gemma(request):
    return LocalLLM(request.param)


def test_model_info_feeds_the_ui(gemma):
    info = gemma.info
    assert info["context_length"] > 0 and info["layers"] > 0
    assert set(gemma.family.stop_tokens) <= set(info["stop_tokens"])
    assert set(info["special_tokens"]) == set(gemma.family.special_tokens)
    # the reset button writes these into the sliders, so they must be numbers
    for key in ("temperature", "top_p", "top_k"):
        assert isinstance(info["defaults"][key], (int, float)), key
    assert_balanced(render.model_html(info))
    assert_balanced(render.special_tokens_html(info))


def test_greedy_decoding_matches_generate(gemma):
    prompt = gemma.render([{"role": "user", "content": "用一句話介紹台北。"}], None, False)
    events = list(gemma.generate(prompt, Sampling(temperature=0, max_new_tokens=24), lambda: False))
    ours = [e["id"] for e in events if e["type"] == "token"]
    assert all(e["rank"] == 1 for e in events if e["type"] == "token")

    ids = gemma.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(gemma.device)
    with torch.inference_mode():
        out = gemma.model.generate(ids, do_sample=False, max_new_tokens=24, top_p=None, top_k=None, temperature=None)
    theirs = out[0, ids.shape[-1]:].tolist()
    assert ours == theirs[: len(ours)]


def test_agent_uses_a_tool_and_answers(gemma):
    events = []
    settings = RunSettings(sampling=Sampling(temperature=0, max_new_tokens=256), max_steps=4)
    Run(gemma, [], "請用 calculator 工具計算 12345 * 6789。", settings, events.append).run()

    assert not [e for e in events if e["type"] == "error"]
    calls = [e for e in events if e["type"] == "tool_call"]
    assert calls and calls[0]["name"] == "calculator" and not calls[0]["errors"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert results[0]["ok"] and "83810205" in results[0]["result"]
    end = events[-1]
    assert end["type"] == "run_end" and end["stop_reason"] == "no_tool_call"
    assert "83810205" in end["final"].replace(",", "")

    for markup in render_everything(render.build(events), live=False):
        assert_balanced(markup)


def test_mtp_writes_what_plain_decoding_writes(gemma):
    if not gemma.drafter_id:
        pytest.skip(f"{gemma.model_id} has no MTP drafter")
    gemma.load_drafter()
    assert gemma.info["drafter"]["draft_tokens"] > 0
    messages = [{"role": "system", "content": SYSTEM_WITH_TOOLS}, {"role": "user", "content": "台北現在幾點？紐約呢？"}]
    prompt = gemma.render(messages, [t.schema() for t in TOOLS.values()], False)

    def decode(mtp):
        events = list(gemma.generate(prompt, Sampling(temperature=0, max_new_tokens=64, mtp=mtp), lambda: False))
        return [e for e in events if e["type"] == "token"], events[-1]

    plain, _ = decode(False)
    drafted, done = decode(True)
    assert [t["id"] for t in drafted] == [t["id"] for t in plain]
    assert done["mtp"]["accepted"] > done["mtp"]["drafted"] // 2  # a tool call is easy to guess
    assert done["mtp"]["forwards"] < len(drafted) - 1
    assert sum(1 for t in drafted if t.get("draft", {}).get("accepted")) == done["mtp"]["accepted"]
