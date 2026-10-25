"""End to end with the real weights of every model in the menu (Gemma 4 E2B ~10 GB in memory,
Qwen3.5 0.8B ~2 GB; tens of seconds each), and Gemma's MTP drafter; with the sample image, recording
and video the page's examples attach, and with run_python when Docker and its image are there.
Opt in with:  RUN_MODEL_TESTS=1 uv run pytest -m model   (add  -k Qwen  for one model)"""

import os
import re
from pathlib import Path

import pytest
import torch

import families
import media
import render
import samples
import tools
from agent import SYSTEM_WITH_TOOLS, Run, RunSettings
from backend import Backend
from html_check import assert_balanced
from simple_backend import LocalLLM, Sampling
from test_render import render_everything
from tools import TOOLS

SAMPLES = Path(__file__).resolve().parent.parent / "assets" / "examples"

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
    assert_balanced(render.model_html(info, lang="zh"))
    assert_balanced(render.special_tokens_html(info, lang="zh"))
    assert_balanced(render.folder_html(info, lang="zh"))
    assert {f["name"] for f in info["folder"]["files"]} >= {"config.json", "tokenizer.json", "chat_template.jinja"}


def test_the_loaded_model_is_a_backend(gemma):
    assert isinstance(gemma, Backend)


def test_greedy_decoding_matches_generate(gemma):
    prompt = gemma.render([{"role": "user", "content": "用一句話介紹台北。"}], None, False)
    events = list(gemma.generate(gemma.encode(prompt, []), Sampling(temperature=0, max_new_tokens=24), lambda: False))
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
    assert "num_assistant_tokens" in next(f for f in gemma.info["drafter"]["folder"]["files"] if f["name"] == "generation_config.json")["fields"]
    messages = [{"role": "system", "content": SYSTEM_WITH_TOOLS}, {"role": "user", "content": "台北現在幾點？紐約呢？"}]
    prompt = gemma.render(messages, [t.schema() for t in TOOLS.values()], False)

    def decode(mtp):
        events = list(gemma.generate(gemma.encode(prompt, []), Sampling(temperature=0, max_new_tokens=64, mtp=mtp), lambda: False))
        return [e for e in events if e["type"] == "token"], events[-1]

    plain, _ = decode(False)
    drafted, done = decode(True)
    assert [t["id"] for t in drafted] == [t["id"] for t in plain]
    assert done["mtp"]["accepted"] > done["mtp"]["drafted"] // 2  # a tool call is easy to guess
    assert done["mtp"]["forwards"] < len(drafted) - 1
    assert sum(1 for t in drafted if t.get("draft", {}).get("accepted")) == done["mtp"]["accepted"]


# ---------- images, sound and video ----------

def attach(name: str, model: LocalLLM) -> media.Attachment:
    path = SAMPLES / name
    return media.Attachment(name, media.kind_of(name, model.modalities), str(path), path.stat().st_size)


def media_prompt(model: LocalLLM, names: list[str], question: str):
    files = [attach(name, model) for name in names]
    messages = [{"role": "user", "content": media.user_content(question, files)}]
    return model.encode(model.render(messages, None, False), media.parts(messages))


def test_each_model_reads_what_its_checkpoint_says(gemma):
    expected = {"gemma4": ("image", "audio", "video"), "qwen3_5": ("image", "video")}[gemma.family.key]
    assert gemma.modalities == expected and gemma.info["modalities"] == list(expected)


def test_greedy_decoding_with_media_matches_generate(gemma):
    names = ["receipt.png", "ball.mp4"] + (["speech.flac"] if "audio" in gemma.modalities else [])
    prompt = media_prompt(gemma, names, "Describe each attachment in one short sentence.")
    assert set(prompt.media_tokens) == {gemma_kind for gemma_kind in gemma.modalities}
    events = list(gemma.generate(prompt, Sampling(temperature=0, max_new_tokens=24), lambda: False))
    ours = [e["id"] for e in events if e["type"] == "token"]
    assert events[0]["n_tokens"] == prompt.n_tokens > 500  # the media are in the prefill

    with torch.inference_mode():
        out = gemma.model.generate(input_ids=prompt.ids.to(gemma.device), **prompt.media, do_sample=False, max_new_tokens=24,
                                   top_p=None, top_k=None, temperature=None)
    assert ours == out[0, prompt.n_tokens:].tolist()[: len(ours)]

    # A text-only prompt right after: nothing about the media's positions carries over.
    text = gemma.encode(gemma.render([{"role": "user", "content": "用一句話介紹台北。"}], None, False), [])
    after = [e["id"] for e in gemma.generate(text, Sampling(temperature=0, max_new_tokens=16), lambda: False) if e["type"] == "token"]
    with torch.inference_mode():
        alone = gemma.model.generate(text.ids.to(gemma.device), do_sample=False, max_new_tokens=16, top_p=None, top_k=None, temperature=None)
    assert after == alone[0, text.n_tokens:].tolist()[: len(after)]


def test_mtp_with_an_image_writes_what_plain_decoding_writes(gemma):
    if not gemma.drafter_id:
        pytest.skip(f"{gemma.model_id} has no MTP drafter")
    gemma.load_drafter()
    prompt = media_prompt(gemma, ["receipt.png"], "List the items on this receipt.")

    def decode(mtp):
        return [e["id"] for e in gemma.generate(prompt, Sampling(temperature=0, max_new_tokens=48, mtp=mtp), lambda: False) if e["type"] == "token"]

    assert decode(True) == decode(False)


def test_the_model_reads_the_total_on_the_receipt(gemma):
    events = []
    settings = RunSettings(sampling=Sampling(temperature=0, max_new_tokens=64), use_tools=False)
    Run(gemma, [], "What is the TOTAL printed on this receipt? Answer with the number only.", settings, events.append, [attach("receipt.png", gemma)]).run()
    assert not [e for e in events if e["type"] == "error"]
    assert "370" in events[-1]["final"]
    [context] = [e for e in events if e["type"] == "context"]
    assert context["media_tokens"]["image"] > 100


@pytest.mark.skipif(tools.sandbox_problem() is not None, reason="needs Docker and the agent-loop-sandbox image")
def test_the_model_writes_code_that_runs_in_the_sandbox(gemma, tmp_path):
    events = []
    settings = RunSettings(sampling=Sampling(temperature=0, max_new_tokens=512), max_steps=4, lang="en")
    question = "Use run_python to compute the sum of the squares of the integers from 1 to 100."
    Run(gemma, [], question, settings, events.append, files=tmp_path).run()
    assert not [e for e in events if e["type"] == "error"]
    calls = [e for e in events if e["type"] == "tool_call" and e["name"] == "run_python"]
    results = [e for e in events if e["type"] == "tool_result" and e["name"] == "run_python"]
    assert calls and any(r["ok"] and "338350" in r["result"] for r in results)
    assert "338350" in events[-1]["final"].replace(",", "")


# ---------- every format the page accepts, with both models ----------

def rolls_left_to_right(final: str) -> bool:
    """Says the ball rolls left to right, in English or in Chinese (從左側滾動到右側), and not the other way."""
    text = final.lower().replace("*", "")
    return bool(re.search(r"left to right|左.{0,4}到.{0,2}右", text)) and not re.search(r"right to left|右.{0,4}到.{0,2}左", text)


MEDIA_QUESTIONS = {  # with tools off, greedy: what each model must get right from each kind of file
    "image": ("What is the TOTAL printed on this receipt? Answer with the number only.", lambda final: samples.TOTAL_PRINTED in final),
    "audio": ("Transcribe the speech in this recording into English text.",
              lambda final: "apostle of the middle classes" in final.lower()),
    # Gemma often answers this one in Chinese
    "video": ("Does the red ball in this video roll from left to right, or from right to left?", rolls_left_to_right),
}


@pytest.mark.parametrize("ext", [ext for ext in samples.FORMATS if ext in media.KINDS])
def test_the_model_reads_every_media_format_it_has_an_encoder_for(gemma, tmp_path, ext):
    path = samples.make(tmp_path, ext)
    kind = media.KINDS[ext]
    attachment = media.Attachment(path.name, media.kind_of(path.name, gemma.modalities), str(path), path.stat().st_size)
    if kind not in gemma.modalities:  # Qwen3.5 and sound: a file for run_python, never sent to the model
        assert attachment.kind == "file" and media.parts([{"role": "user", "content": media.user_content("", [attachment])}]) == []
        return
    question, right = MEDIA_QUESTIONS[kind]
    events = []
    settings = RunSettings(sampling=Sampling(temperature=0, max_new_tokens=128), use_tools=False, lang="en")
    Run(gemma, [], question, settings, events.append, [attachment], tmp_path).run()
    assert not [e for e in events if e["type"] == "error"]
    assert next(e for e in events if e["type"] == "context")["media_tokens"][kind] > 0
    assert right(events[-1]["final"]), events[-1]["final"]


@pytest.mark.skipif(tools.sandbox_problem() is not None, reason="needs Docker and the agent-loop-sandbox image")
@pytest.mark.parametrize("ext", samples.DATA)
def test_the_model_opens_every_data_format_with_run_python(gemma, tmp_path, ext):
    path = samples.make(tmp_path, ext)
    attachment = media.Attachment(path.name, "file", str(path), path.stat().st_size)
    events = []
    settings = RunSettings(sampling=Sampling(temperature=0, max_new_tokens=512), max_steps=3, lang="en")
    Run(gemma, [], f"Use run_python to open /data/{path.name} and print its first two rows.", settings, events.append, [attachment], tmp_path).run()
    results = [e for e in events if e["type"] == "tool_result" and e["name"] == "run_python"]
    assert any(r["ok"] and "2025" in r["result"] for r in results), results
