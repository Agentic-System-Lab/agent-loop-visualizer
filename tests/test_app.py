"""The Streamlit page, driven through streamlit.testing with the fake model in place of Gemma.

Each test clicks and types like a user would, then waits for the agent thread
before the next rerun, the way the page's half-second refresh would. The browser
is a Chinese one unless a test says otherwise, so the page is in Chinese.
"""

import base64
import io
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import agent
import render
import simple_backend
import tools
import vllm_backend
from backend import Sampling
from families import GEMMA4, QWEN3_5
from fakes import MEDIA_TOKENS, FakeLLM, tool_call

APP = Path(__file__).resolve().parent.parent / "streamlit_app.py"
ANSWER = "答案是 42。<turn|>"
CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")
LANG_COOKIE = "agent_loop_lang"  # streamlit_app.LANG_COOKIE; the app is a script, not a module to import


@pytest.fixture
def outputs():
    """What the fake model will say next, one entry per step. Tests fill it in."""
    return []


@pytest.fixture
def models(monkeypatch, outputs, tmp_path, sandbox):
    made = []

    class AppLLM(FakeLLM):
        def __init__(self, model_id):
            super().__init__(outputs, model_id)
            made.append(self)

    monkeypatch.setattr(simple_backend, "LocalLLM", AppLLM)
    monkeypatch.setenv("AGENT_LOOP_UPLOADS", str(tmp_path / "uploads"))
    st.cache_resource.clear()  # a fresh model and a fresh generation lock per test
    st.cache_data.clear()  # and a fresh look for Docker
    yield made
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture
def sandbox(monkeypatch):
    """What the page finds when it looks for Docker: a problem in both languages, or None when it is ready.
    `looked` counts the times it looked. Docker is ready unless a test says otherwise."""
    state = {"problem": None, "looked": 0}

    def sandbox_problem():
        state["looked"] += 1
        return state["problem"]

    monkeypatch.setattr(tools, "sandbox_problem", sandbox_problem)
    return state


@pytest.fixture
def browser(monkeypatch):
    """What the browser tells the page: its language and its cookies. A test may change them before it opens a tab."""
    seen = {"locale": "zh-TW", "cookies": {}}
    context = type(st.context)
    monkeypatch.setattr(context, "locale", property(lambda self: seen["locale"]))
    monkeypatch.setattr(context, "cookies", property(lambda self: seen["cookies"]))
    return seen


@pytest.fixture
def new_tab(models, browser):
    """Open the page in a new browser tab (a new session sharing the cached model)."""
    tabs = []

    def open_tab():
        at = AppTest.from_file(str(APP), default_timeout=10).run()
        assert not at.exception, at.exception
        tabs.append(at)
        return at

    yield open_tab
    for at in tabs:  # never leave an agent thread waiting
        for run in at.session_state["runs"] if "runs" in at.session_state else []:
            run.run.stop()
            run.thread.join(5)


@pytest.fixture
def at(new_tab):
    return new_tab()


# ---------- helpers ----------

def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def settle(at):
    """Let every agent thread finish, then redraw."""
    for run in at.session_state["runs"]:
        run.thread.join(5)
        assert not run.live
    at.run()
    assert not at.exception, at.exception
    return at


def submit(at, text):
    at.chat_input[0].set_value(text).run()
    assert not at.exception, at.exception
    return at


def ask(at, text="6 乘以 7 是多少？"):
    return settle(submit(at, text))


def wait_for_event(at, kind):
    run = at.session_state["runs"][-1]
    wait_for(lambda: any(e["type"] == kind for e in run.events))
    at.run()
    assert not at.exception, at.exception
    return at


def buttons(at, label):
    return [b for b in at.button if b.label == label]


def click(at, label):
    [button] = buttons(at, label)
    button.click().run()
    assert not at.exception, at.exception
    return at


def page_html(at):
    return "\n".join(h.value for h in at.html)


def captions(at):
    return [c.value for c in at.caption]


def tab(at):
    return next(b for b in at.button_group if b.key == "viz_tab")


def lang_switch(at):
    return next(b for b in at.button_group if b.key == "lang")


def drawn_text(at) -> str:
    """Everything the page writes, except the stylesheets it loads."""
    html = [h.value for h in at.html if "<style" not in h.value and not h.value.lstrip().startswith("/*")]
    return "\n".join([*html, *captions(at), *(m.value for m in at.markdown), *(i.value for i in at.info), *(w.value for w in at.warning)])


# ---------- first load ----------

def example_labels(at) -> list[str]:
    return [b.label for b in at.button if b.key and b.key.startswith("example-")]


def test_first_load(at, sandbox):
    assert example_labels(at) == [
        "台北現在幾點？紐約呢？",
        "(37.5 × 12) 的平方根，再加上 2 的 20 次方，是多少？",
        "ReAct 這篇論文是誰提出的？用 Wikipedia 查。",
        "Gemma 是哪家公司開發的？第一版什麼時候發布？",
        "這張收據上的品項加起來是多少？和收據上寫的總計一樣嗎？",
        "影片裡的紅球是從左邊滾到右邊，還是從右邊滾到左邊？",
        "把這段錄音的內容寫下來，再翻成繁體中文。",
    ]
    assert {t.label: t.value for t in at.toggle} == {
        "使用工具": True, "讓模型寫程式": False, "思考模式": False, "執行工具前先問我": False, "MTP 推測解碼": False,
    }
    assert sandbox["looked"] == 0  # writing code is off, so Docker is not needed and not looked for
    assert {s.label: s.value for s in at.slider} == {
        "temperature": 1.0, "top-p": 0.95, "top-k": 64, "步數上限": 6, "每步 token 上限": 1024,
    }
    assert tab(at).value == "loop" and tab(at).options == ["迴圈", "脈絡", "模型"]
    assert 'class="intro"' in page_html(at)
    assert not at.chat_input[0].disabled
    assert at.session_state["runs"] == [] and at.session_state["history"] == []


def test_model_loads_once_across_reruns(at, models):
    at.run()
    at.run()
    assert len(models) == 1


# ---------- one question, the whole loop ----------

def test_question_runs_the_whole_loop(at, outputs, models):
    outputs += [tool_call("calculator", expression="6*7"), ANSWER]
    ask(at)

    [user, assistant] = at.chat_message
    assert user.name == "user" and assistant.name == "assistant"
    assert [m.value for m in user.markdown] == ["6 乘以 7 是多少？"]
    assert "答案是 42。" in [m.value for m in assistant.markdown]

    html = page_html(at)
    assert "Step 1 · calculator" in html                           # trace chip in the chat
    assert "STEP 1" in html and "STEP 2" in html                   # step cards on the right
    assert "&quot;result&quot;: 42" in html                        # the tool really ran
    assert "停止：模型沒有再呼叫工具" in html                         # stop card
    assert any(c.startswith("2 步 · ") and c.endswith("停止：模型沒有再呼叫工具") for c in captions(at))
    assert any(c.startswith("Step 2 / 6 · 脈絡 ") for c in captions(at))
    assert len(buttons(at, "看這一步的完整脈絡 →")) == 2

    history = at.session_state["history"]
    assert [m["role"] for m in history] == ["user", "assistant", "tool", "assistant"]
    assert history[-1]["content"] == "答案是 42。"
    assert not at.chat_input[0].disabled  # ready for the next question
    assert len(models[0].prompts) == 2


def test_example_button_asks_it(at):
    at.button(key="example-1").click().run()
    settle(at)
    [run] = at.session_state["runs"]
    assert run.text == "(37.5 × 12) 的平方根，再加上 2 的 20 次方，是多少？"
    assert not [b for b in at.button if b.key and b.key.startswith("example-")]  # the welcome is gone


def test_sidebar_settings_reach_the_model(at, models):
    next(t for t in at.toggle if t.label == "使用工具").set_value(False)
    next(t for t in at.toggle if t.label == "思考模式").set_value(True)
    next(s for s in at.slider if s.label == "temperature").set_value(0.0)
    next(s for s in at.slider if s.label == "top-k").set_value(0)
    next(s for s in at.slider if s.label == "步數上限").set_value(3)
    next(s for s in at.slider if s.label == "每步 token 上限").set_value(256)
    at.run()
    ask(at)

    fake = models[0]
    assert fake.renders[0]["tools"] is None
    assert fake.renders[0]["thinking"] is True
    assert fake.samplings == [Sampling(temperature=0.0, top_p=0.95, top_k=0, max_new_tokens=256)]
    settings = at.session_state["runs"][0].data()["settings"]
    assert settings["max_steps"] == 3 and settings["use_tools"] is False


def test_follow_up_question_sees_the_conversation(at, outputs, models):
    outputs += ["你好！<turn|>", "你剛剛說你好。<turn|>"]
    ask(at, "嗨")
    ask(at, "我剛剛說了什麼？")

    second_prompt = models[0].prompts[1]
    assert "嗨" in second_prompt and "你好！" in second_prompt and "我剛剛說了什麼？" in second_prompt
    assert [m.name for m in at.chat_message] == ["user", "assistant", "user", "assistant"]
    assert len(at.session_state["history"]) == 4

    assert at.session_state["view"] == 1
    assert any(c.startswith("第 2 輪：") for c in captions(at))
    [view_first] = buttons(at, "看流程")  # only the turn not on screen gets the button
    view_first.click().run()
    assert at.session_state["view"] == 0
    assert any(c.startswith("第 1 輪：嗨") for c in captions(at))


def test_new_chat_clears_everything(at):
    ask(at)
    click(at, "新對話")
    assert at.session_state["runs"] == [] and at.session_state["history"] == []
    assert len(at.chat_message) == 0
    assert 'class="intro"' in page_html(at)


def test_reset_restores_model_defaults(at):
    next(s for s in at.slider if s.label == "temperature").set_value(0.3)
    next(s for s in at.slider if s.label == "top-p").set_value(0.5)
    next(s for s in at.slider if s.label == "步數上限").set_value(2)
    at.run()
    click(at, "還原成模型建議值")
    values = {s.label: s.value for s in at.slider}
    assert values["temperature"] == 1.0 and values["top-p"] == 0.95 and values["top-k"] == 64
    assert values["步數上限"] == 6


def test_model_error_is_shown(at, outputs):
    outputs.append(RuntimeError("MPS backend out of memory"))
    ask(at)
    assert [e.value for e in at.error] == ["RuntimeError: MPS backend out of memory"]
    assert any(c.endswith("停止：執行錯誤") for c in captions(at))
    assert "（這一輪沒有最終回覆，看右邊的停止原因。）" in captions(at)
    assert "停止：執行錯誤" in page_html(at)


# ---------- the other two views ----------

def test_context_view(at, outputs):
    outputs += [tool_call("calculator", expression="6*7"), ANSWER]
    tab(at).set_value("context").run()
    assert [i.value for i in at.info] == ["送出問題後，這裡會顯示每一步模型實際讀到的完整字串。"]
    tab(at).set_value("loop").run()

    ask(at)
    at.button(key="ctx-0-1").click().run()
    assert tab(at).value == "context"
    steps = next(b for b in at.button_group if b.options == ["Step 1", "Step 2"])
    assert steps.value == 1
    html = page_html(at)
    assert 'class="ctx-view"' in html and '<mark class="added">' + render.chip("<|turn>", GEMMA4, "zh") + "user" in html

    steps.set_value(2).run()
    assert '<mark class="added">response:calculator' in page_html(at)
    assert any("system 1 · user 1 · assistant 1 · tool 1" in c for c in captions(at))


def test_model_view(at, outputs):
    tab(at).set_value("model").run()
    html = page_html(at)
    assert "huggingface.co/google/gemma-4-E2B-it" in html
    assert "解碼設定" in html and "動作空間：模型看得到的工具" in html
    assert "特殊 token（24 個）" in html
    assert "模型資料夾" in html and "models/models--fake--gemma-test/snapshots/0123456789abcdef" in html
    assert len(at.dataframe) == 0  # no steps yet

    outputs += [tool_call("calculator", expression="6*7"), ANSWER]
    ask(at)
    [table] = at.dataframe
    assert list(table.value["step"]) == [1, 2]


def test_model_view_shows_tools_off(at):
    next(t for t in at.toggle if t.label == "使用工具").set_value(False)
    tab(at).set_value("model").run()
    assert "目前關閉了「使用工具」" in page_html(at)


# ---------- switching models ----------

def test_switching_models_starts_over_with_the_new_model(at, models, outputs):
    ask(at)
    menu = next(s for s in at.selectbox if s.label == "模型")
    assert menu.value == "google/gemma-4-E2B-it" and models[-1].family is GEMMA4
    menu.set_value("Qwen/Qwen3.5-0.8B").run()
    assert not at.exception, at.exception
    assert at.session_state["runs"] == [] and at.session_state["history"] == []  # a new conversation
    assert models[-1].model_id == "Qwen/Qwen3.5-0.8B" and models[-1].family is QWEN3_5
    assert (at.session_state["temperature"], at.session_state["top_p"], at.session_state["top_k"]) == (1.0, 1.0, 20)
    assert any("Qwen3.5" in c for c in captions(at))

    outputs.append("你好。<|im_end|>")
    ask(at, "嗨")
    assert models[-1].prompts[-1].startswith("<|im_start|>system\n")
    html = page_html(at)
    assert "停止：模型沒有再呼叫工具" in html and "&lt;|im_end|&gt;" in html


def test_a_changed_setting_survives_reruns_but_not_a_model_switch(at, models):
    next(s for s in at.slider if s.label == "top-k").set_value(5).run()
    at.run()
    assert at.session_state["top_k"] == 5
    next(s for s in at.selectbox if s.label == "模型").set_value("Qwen/Qwen3.5-0.8B").run()
    assert at.session_state["top_k"] == 20


# ---------- MTP ----------

def mtp_toggle(at):
    return next(t for t in at.toggle if t.label == "MTP 推測解碼")


def test_mtp_loads_the_drafter_and_reaches_the_model(at, models, outputs):
    assert not mtp_toggle(at).disabled and models[0].drafter is None  # off by default, nothing loaded
    mtp_toggle(at).set_value(True).run()
    assert models[0].drafter is not None

    outputs += [tool_call("calculator", expression="6*7"), ANSWER]
    ask(at)
    assert [s.mtp for s in models[0].samplings] == [True, True]
    assert at.session_state["runs"][0].data()["settings"]["mtp"] is True
    assert "MTP 草稿收下" in page_html(at)

    tab(at).set_value("model").run()
    html = page_html(at)
    assert "google/gemma-4-E2B-it-assistant" in html and "每次猜 6 個 token" in html
    assert "MTP 推測解碼 <b>開</b>" in html
    [table] = at.dataframe
    assert "MTP 收下" in table.value.columns


def test_mtp_is_off_for_a_model_without_a_drafter(at, models):
    mtp_toggle(at).set_value(True).run()
    next(s for s in at.selectbox if s.label == "模型").set_value("Qwen/Qwen3.5-0.8B").run()
    assert not at.exception, at.exception
    assert at.session_state["mtp"] is False and mtp_toggle(at).disabled
    ask(at, "嗨")
    assert models[-1].samplings[-1].mtp is False


def test_reset_turns_mtp_off(at):
    mtp_toggle(at).set_value(True).run()
    click(at, "還原成模型建議值")
    assert mtp_toggle(at).value is False


# ---------- a vLLM server runs the model ----------

@pytest.fixture
def vllm(monkeypatch, models, outputs):
    """The page with VLLM_URL set. The server serves Qwen3.5 under the name "qwen", and a fake model with what a
    VLLMBackend has (the server in its info, nothing about weights loaded here) stands in for the backend; the server
    takes images but refused video, and sends sampling masks. `made` holds each one the page makes; set `down` and the
    server cannot be reached, or `defaults` for the decoding the checkpoint recommends."""
    state = {"made": [], "down": False}

    class ServedLLM(FakeLLM):
        def __init__(self, model_id, url):
            super().__init__(outputs, model_id)
            self.modalities = ("image",)
            self.drafter_id = None
            unknown = ("device", "dtype", "parameters", "parameter_parts", "load_seconds")
            refused = {"video": {"zh": "伺服器不收：At most 0 video(s) may be provided in one prompt.",
                                 "en": "the server refused it: At most 0 video(s) may be provided in one prompt."}}
            self.info = {k: v for k, v in self.info.items() if k not in unknown} | {"modalities": ["image"], "drafter_id": None, "server": {
                "engine": "vLLM", "version": "0.32.0", "url": url, "served_as": "qwen", "logprobs_mode": "processed_logprobs", "generate_api": True,
                "sampling_mask": True, "refused": refused}}
            self.info["defaults"] = state.get("defaults", self.info["defaults"])
            state["made"].append(self)

    def served_models(url, transport=None):
        if state["down"]:
            raise vllm_backend.ServerError(f"連不上 {url} 的 vLLM 伺服器。", f"Could not reach the vLLM server at {url}.")
        return [{"id": "qwen", "root": "Qwen/Qwen3.5-0.8B", "max_model_len": 8192}]

    monkeypatch.setenv("VLLM_URL", "http://localhost:8000")
    monkeypatch.setattr(vllm_backend, "VLLMBackend", ServedLLM)
    monkeypatch.setattr(vllm_backend, "served_models", served_models)
    return state


def test_with_vllm_url_the_page_runs_what_the_server_serves(vllm, at, models, outputs):
    menu = next(s for s in at.selectbox if s.label == "模型")
    assert menu.options == ["Qwen3.5 0.8B (Alibaba)"] and menu.value == "Qwen/Qwen3.5-0.8B"
    assert models == [] and [m.model_id for m in vllm["made"]] == ["Qwen/Qwen3.5-0.8B"]  # nothing loaded on this machine
    assert "`Qwen/Qwen3.5-0.8B` · Qwen3.5 · vLLM 0.32.0 · http://localhost:8000" in captions(at)
    outputs.append("答案是 42。<|im_end|>")
    ask(at)
    assert "答案是 42。" in drawn_text(at) and vllm["made"][0].prompts


def test_with_vllm_the_sidebar_says_the_server_decides_speculative_decoding(vllm, at):
    assert mtp_toggle(at).disabled and mtp_toggle(at).value is False
    assert "--speculative-config" in mtp_toggle(at).help and "--per-request-spec-decode-metrics" in mtp_toggle(at).help


def test_with_vllm_the_model_reads_what_the_server_takes_and_the_page_says_what_it_refused(vllm, at):
    labels = example_labels(at)
    assert "這張收據上的品項加起來是多少？和收據上寫的總計一樣嗎？" in labels
    assert "影片裡的紅球是從左邊滾到右邊，還是從右邊滾到左邊？" not in labels  # the server refused video
    assert "台北現在幾點？紐約呢？" in labels
    assert "這個模型讀得懂圖片：用輸入框左邊的 ＋ 附上檔案。" in captions(at)
    assert "這台 vLLM 伺服器不收影片，所以不能附上；原因寫在模型分頁。" in captions(at)
    assert ".png" in at.chat_input[0].proto.file_type and ".mp4" not in at.chat_input[0].proto.file_type
    tab(at).set_value("model").run()
    html = page_html(at)
    assert "這個 checkpoint 讀得懂、但這台伺服器不收的媒體" in html and "<b>影片</b>：伺服器不收：At most 0 video(s)" in html
    assert "看得到剩下幾個候選；這時它不接受 temperature 0 和 top-k 0" in html


def test_a_vllm_server_that_sends_sampling_masks_gets_no_greedy_or_unlimited_top_k_from_the_sidebar(vllm, browser, new_tab, outputs):
    vllm["defaults"] = {"temperature": 0.0, "top_p": 1.0, "top_k": 0}  # a checkpoint that recommends greedy decoding
    browser["locale"] = "en-US"
    at = new_tab()
    temperature, top_k = (next(s for s in at.slider if s.label == label) for label in ("temperature", "top-k"))
    assert (temperature.min, top_k.min) == (0.05, 1)
    assert "--return-sampling-mask" in temperature.help and "greedy" in temperature.help and "0 (no limit) is not allowed" in top_k.help
    assert (at.session_state["temperature"], at.session_state["top_k"]) == (0.05, 128)  # the nearest settings the server takes
    outputs.append("Hello.<|im_end|>")
    ask(at, "Hi")
    sampling = vllm["made"][0].samplings[-1]
    assert (sampling.temperature, sampling.top_k) == (0.05, 128)


def test_with_vllm_the_model_tab_says_the_server_runs_it(vllm, at):
    tab(at).set_value("model").run()
    html = page_html(at)
    assert "Qwen3.5 家族，由 http://localhost:8000 的 vLLM 伺服器跑" in html and "<dd>vLLM 0.32.0</dd>" in html
    assert "transformers 跑" not in html and 'class="params-bar"' not in html and "None" not in drawn_text(at)
    assert "伺服器自己讀權重" in html


def test_a_vllm_server_that_goes_down_later_is_said_not_a_traceback(vllm, at):
    vllm["down"] = True
    st.cache_data.clear()  # as when the menu's minute is up
    at.run()
    assert not at.exception and [e.value for e in at.error] == ["連不上 http://localhost:8000 的 vLLM 伺服器。"]


@pytest.mark.parametrize(("locale", "message"), [("zh-TW", "連不上 http://localhost:8000 的 vLLM 伺服器。"),
                                                 ("en-US", "Could not reach the vLLM server at http://localhost:8000.")])
def test_a_vllm_server_that_cannot_be_reached_is_said_in_the_page_language(vllm, browser, new_tab, locale, message):
    vllm["down"] = True
    browser["locale"] = locale
    at = new_tab()
    assert [e.value for e in at.error] == [message] and not at.chat_input
    vllm["down"] = False
    st.cache_data.clear()  # the menu is not cached while the server is down; this only makes sure of it
    at.run()
    assert not at.exception and not at.error and at.chat_input


# ---------- the guide ----------

@pytest.mark.parametrize("lang, title", [("zh", "運作說明"), ("en", "How it works")])
def test_guide_is_a_link_to_github_pages_in_the_page_language(at, monkeypatch, lang, title):
    pages = []
    navigation = st.navigation
    monkeypatch.setattr(st, "navigation", lambda p, **kw: pages.extend(p) or navigation(p, **kw))
    lang_switch(at).set_value(lang).run()
    assert not at.exception, at.exception
    guide = next(p for p in pages[-2:] if p.title == title)
    assert guide.external_url == f"https://cyyeh.github.io/agent-loop-visualizer/{lang}/"


# ---------- the human in the loop ----------

def start_waiting_for_approval(at, outputs):
    outputs += [tool_call("get_current_time", timezone="Asia/Taipei"), "台北現在下午三點。<turn|>"]
    next(t for t in at.toggle if t.label == "執行工具前先問我").set_value(True)
    at.run()
    submit(at, "台北現在幾點？")
    return wait_for_event(at, "approval_request")


def test_approve_a_tool_call(at, outputs):
    start_waiting_for_approval(at, outputs)
    assert "**Agent 想執行這個工具，要允許嗎？**  \n`get_current_time({\"timezone\": \"Asia/Taipei\"})`" in [m.value for m in at.markdown]
    assert "Step 1 · 等你確認要不要執行工具" in captions(at)
    assert at.chat_input[0].disabled
    assert buttons(at, "新對話")[0].disabled
    assert buttons(at, "停止這次執行")
    assert "等你在左邊按「允許」或「拒絕」" in page_html(at)

    click(at, "允許")
    settle(at)
    assert "台北現在下午三點。" in [m.value for m in at.markdown]
    assert "你允許執行" in page_html(at)
    assert '"timezone": "Asia/Taipei"' in at.session_state["history"][2]["content"]
    assert not buttons(at, "允許") and not buttons(at, "停止這次執行")


def test_reject_a_tool_call(at, outputs):
    start_waiting_for_approval(at, outputs)
    click(at, "拒絕")
    settle(at)
    html = page_html(at)
    assert "你拒絕了" in html and "未執行" in html
    assert "使用者拒絕執行這個工具呼叫" in at.session_state["history"][2]["content"]


def test_stop_button(at, outputs):
    start_waiting_for_approval(at, outputs)
    click(at, "停止這次執行")
    settle(at)
    assert any(c.endswith("停止：使用者中斷") for c in captions(at))
    html = page_html(at)
    assert "停止：使用者中斷" in html
    assert "執行被停止，這個呼叫沒有執行" in html and "你拒絕了" not in html
    assert not at.chat_input[0].disabled


def test_one_run_at_a_time_across_tabs(at, new_tab, outputs, models):
    start_waiting_for_approval(at, outputs)

    other = new_tab()
    [warning] = other.warning
    assert "另一個分頁的 agent 還在執行（「台北現在幾點？」）" in warning.value
    assert other.chat_input[0].disabled
    assert all(b.disabled for b in other.button if b.key and b.key.startswith("example-"))
    assert len(models) == 1  # both tabs share the one model

    click(at, "允許")
    settle(at)
    other.run()
    assert len(other.warning) == 0 and not other.chat_input[0].disabled
    ask(other, "現在換我")
    assert len(other.session_state["runs"]) == 1


def test_another_tab_can_stop_a_leftover_run(at, new_tab, outputs):
    """Reloading the tab that is waiting for approval leaves its run behind; a new tab can stop it."""
    start_waiting_for_approval(at, outputs)
    leftover = at.session_state["runs"][0]

    reloaded = new_tab()
    click(reloaded, "停止那個 agent")
    leftover.thread.join(5)
    assert not leftover.live
    assert leftover.events[-1]["stop_reason"] == "user_stopped"
    assert "使用者停止了這次執行" in leftover.events[-1]["messages"][2]["content"]

    reloaded.run()
    assert len(reloaded.warning) == 0 and not reloaded.chat_input[0].disabled
    outputs[:] = [ANSWER]  # the stopped run never used its second reply
    ask(reloaded, "重新整理之後的問題")
    assert "答案是 42。" in [m.value for m in reloaded.markdown]


def test_unanswered_approval_frees_the_model(at, new_tab, outputs, monkeypatch):
    monkeypatch.setattr(agent, "APPROVAL_TIMEOUT", 0.2)
    start_waiting_for_approval(at, outputs)
    settle(at)
    assert any(c.endswith("停止：等待確認逾時") for c in captions(at))
    assert "等待確認逾時，harness 當作拒絕並停止迴圈" in page_html(at)

    other = new_tab()
    outputs.append(ANSWER)
    ask(other, "換我")
    assert len(other.session_state["runs"]) == 1


def test_a_tab_that_has_not_redrawn_cannot_start_a_second_run(at, new_tab, outputs):
    """A tab opened before another tab's run started still shows an enabled input. Streamlit drops
    what it submits, because the input is disabled on the redraw, and the redraw shows why."""
    stale = new_tab()
    start_waiting_for_approval(at, outputs)
    assert not stale.chat_input[0].disabled  # its last drawing is from before the run

    submit(stale, "我也想問")
    assert stale.session_state["runs"] == []
    assert len(stale.warning) == 1 and stale.chat_input[0].disabled


def test_toast_when_two_tabs_start_at_once(at, new_tab, outputs):
    """Two tabs can pass the busy check at the same moment; the lock lets only one of them start."""
    start_waiting_for_approval(at, outputs)
    leftover = at.session_state["runs"][0]
    slot = leftover._slot
    slot.run = None  # the moment between the other tab taking the lock and registering its run
    try:
        racing = new_tab()
        assert not racing.chat_input[0].disabled
        submit(racing, "我也想問")
        assert [t.value for t in racing.toast] == ["另一個 agent 正在執行（可能在別的分頁），請稍等。"]
        assert racing.session_state["runs"] == []
    finally:
        slot.run = leftover


# ---------- the page's language ----------

def test_an_english_browser_gets_the_page_in_english(browser, new_tab, outputs):
    browser["locale"] = "en-US"
    at = new_tab()
    assert at.session_state["lang"] == "en" and lang_switch(at).value == "en"
    assert {t.label for t in at.toggle} == {"Use tools", "Let the model write code", "Thinking mode", "Ask me before running a tool", "MTP speculative decoding"}
    assert at.button(key="example-0").label == "What time is it in Taipei right now?"
    assert tab(at).options == ["Loop", "Context", "Model"]

    outputs += [tool_call("calculator", expression="6*7"), "It is 42.<turn|>"]
    ask(at, "What is 6 times 7?")
    assert any(c.endswith("stop: no more tool calls") for c in captions(at))
    for view in ("loop", "context", "model"):
        tab(at).set_value(view).run()
        text = re.sub(r'<pre class="ctx-view">.*?</pre>', "", drawn_text(at), flags=re.S)  # the prompt itself has Chinese in it
        assert not CJK.search(text), text[CJK.search(text).start() - 80:][:160]


def test_the_language_picked_last_time_wins_over_the_browser(browser, new_tab):
    browser["cookies"] = {LANG_COOKIE: "en"}
    assert new_tab().session_state["lang"] == "en"
    browser["cookies"] = {LANG_COOKIE: "klingon"}  # not a language the page has: the browser's then
    assert new_tab().session_state["lang"] == "zh"


def test_picking_a_language_remembers_it_in_a_cookie(at):
    assert "document.cookie" not in "".join(h.value for h in at.html)  # nothing picked, nothing written
    lang_switch(at).set_value("en").run()
    [script] = [h.value for h in at.html if "document.cookie" in h.value]
    assert f"document.cookie = '{LANG_COOKIE}=en; path=/; max-age=31536000; SameSite=Lax';" in script
    lang_switch(at).set_value("zh").run()
    assert any(f"'{LANG_COOKIE}=zh;" in h.value for h in at.html)


def test_switching_language_keeps_the_conversation_and_the_settings(at, outputs):
    next(s for s in at.slider if s.label == "temperature").set_value(0.3)
    next(s for s in at.slider if s.label == "步數上限").set_value(4)
    at.run()
    outputs += [tool_call("calculator", expression="6*7"), ANSWER]
    ask(at)
    tab(at).set_value("model").run()

    lang_switch(at).set_value("en").run()
    assert not at.exception, at.exception
    assert len(at.session_state["runs"]) == 1 and len(at.session_state["history"]) == 4
    assert at.session_state["model_id"] == "google/gemma-4-E2B-it" and tab(at).value == "model"
    assert next(s for s in at.slider if s.label == "temperature").value == 0.3
    assert next(s for s in at.slider if s.label == "Max steps").value == 4
    assert any(c.endswith("stop: no more tool calls") for c in captions(at))  # the finished turn, drawn again in English


def test_the_model_reads_errors_in_the_language_the_turn_started_in(at, outputs):
    lang_switch(at).set_value("en").run()
    outputs += [tool_call("calculator"), "Done.<turn|>"]
    ask(at, "Add it up.")
    assert '"Missing required argument expression"' in at.session_state["history"][2]["content"]


# ---------- examples, attachments and writing code ----------

def toggle(at, label):
    return next(t for t in at.toggle if t.label == label)


@pytest.mark.parametrize("locale", ["zh-TW", "en-US"])
def test_each_page_shows_examples_in_its_own_language_only(browser, new_tab, sandbox, locale):
    browser["locale"] = locale
    at = new_tab()
    toggle(at, "讓模型寫程式" if locale == "zh-TW" else "Let the model write code").set_value(True).run()
    labels = example_labels(at)
    assert len(labels) >= 9
    if locale == "zh-TW":
        assert all(CJK.search(label) for label in labels), labels
    else:
        assert not any(CJK.search(label) for label in labels), labels


def test_an_example_with_a_file_attaches_it(at, models, outputs):
    outputs.append("加起來是 360，收據上寫 370，不一樣。<turn|>")
    at.button(key="example-4").click().run()
    settle(at)
    [run] = at.session_state["runs"]
    [receipt] = run.attachments
    assert (receipt.name, receipt.kind) == ("receipt.png", "image")
    folder = Path(receipt.path).parent
    assert folder.parent == Path(os.environ["AGENT_LOOP_UPLOADS"]) and (folder / "receipt.png").read_bytes() == (APP.parent / "assets/examples/receipt.png").read_bytes()

    fake = models[0]
    assert [p["name"] for p in fake.encodes[0]] == ["receipt.png"]  # the image goes to the processor
    user = at.session_state["history"][0]["content"]
    assert user[0]["type"] == "image" and user[1] == {"type": "text", "text": run.text}
    [user_msg, _] = at.chat_message
    assert any(c == "📎 receipt.png · 19.8 KB" for c in (c.value for c in user_msg.caption))
    html = page_html(at)
    assert f'&lt;|image|&gt;<span class="sp-n">×{MEDIA_TOKENS}</span>' in html  # the image's tokens, as one chip
    assert f"其中 圖片 {MEDIA_TOKENS} tokens" in html


def test_media_examples_follow_what_the_model_reads(at):
    assert "把這段錄音的內容寫下來，再翻成繁體中文。" in example_labels(at)  # Gemma 4 E2B hears
    next(s for s in at.selectbox if s.label == "模型").set_value("Qwen/Qwen3.5-0.8B").run()
    labels = example_labels(at)
    assert "影片裡的紅球是從左邊滾到右邊，還是從右邊滾到左邊？" in labels and "把這段錄音的內容寫下來，再翻成繁體中文。" not in labels  # Qwen3.5 does not


def sample_picker(at):
    return next(b for b in at.button_group if b.key == "samples")


def test_a_sample_file_goes_with_a_question_of_ones_own(at, models, outputs):
    outputs.append("收據上有三個品項。<turn|>")
    sample_picker(at).set_value(["receipt.png"]).run()
    assert at.chat_input[0].placeholder == "問一個關於 receipt.png 的問題…"
    ask(at, "收據上有幾個品項？")
    [run] = at.session_state["runs"]
    [receipt] = run.attachments
    assert (run.text, receipt.name, receipt.kind) == ("收據上有幾個品項？", "receipt.png", "image")
    assert Path(receipt.path).read_bytes() == (APP.parent / "assets/examples/receipt.png").read_bytes()
    assert [p["name"] for p in models[0].encodes[0]] == ["receipt.png"]
    assert sample_picker(at).value == [] and at.chat_input[0].placeholder != "問一個關於 receipt.png 的問題…"  # it went with that question only


def test_the_sample_files_follow_what_the_model_can_use(at, sandbox):
    assert sample_picker(at).options == ["ball.mp4", "receipt.png", "speech.flac"]
    toggle(at, "讓模型寫程式").set_value(True).run()
    assert sample_picker(at).options == ["ball.mp4", "receipt.png", "sales.csv", "speech.flac"]  # once run_python can read it
    next(s for s in at.selectbox if s.label == "模型").set_value("Qwen/Qwen3.5-0.8B").run()
    assert sample_picker(at).options == ["ball.mp4", "receipt.png", "sales.csv"]  # Qwen3.5 does not hear


def test_a_sample_data_file_is_for_run_python(at, outputs, models, sandbox):
    toggle(at, "讓模型寫程式").set_value(True).run()
    sample_picker(at).set_value(["sales.csv"]).run()
    ask(at, "顯示 sales.csv 的前十行。")
    [run] = at.session_state["runs"]
    assert [(a.name, a.kind) for a in run.attachments] == [("sales.csv", "file")]
    assert at.session_state["history"][0]["content"][-1]["text"] == (
        "顯示 sales.csv 的前十行。\n\nAttached for run_python, which reads them:\n"
        "/data/sales.csv (942 B), whose first line is: month,region,units,revenue")


def test_writing_code_looks_for_docker_only_once_switched_on(at, sandbox):
    assert sandbox["looked"] == 0
    assert not any("寫 Python" in label for label in example_labels(at))
    assert "打開側欄的「讓模型寫程式」，還有要寫 Python 的範例。" in captions(at)

    sandbox["problem"] = {"zh": "寫程式要用 Docker 當沙箱，但這台電腦還沒有安裝。", "en": "Writing code needs Docker as its sandbox, and it is not installed."}
    toggle(at, "讓模型寫程式").set_value(True).run()
    assert sandbox["looked"] >= 1
    assert [w.value for w in at.warning] == ["寫程式要用 Docker 當沙箱，但這台電腦還沒有安裝。"]
    python = [b for b in at.button if b.key and b.key.startswith("example-") and "Python" in b.label]
    assert len(python) == 2 and all(b.disabled for b in python)  # shown, but they cannot run yet

    sandbox["problem"] = None
    st.cache_data.clear()  # as if 10 s had passed since the last look
    at.run()
    assert not at.warning
    assert not any(b.disabled for b in at.button if b.key and b.key.startswith("example-"))


def test_the_input_takes_data_files_only_once_the_model_can_run_code(at, sandbox):
    def types():
        return list(at.chat_input[0].proto.file_type)

    assert ".png" in types() and ".csv" not in types()
    sandbox["problem"] = {"zh": "Docker 沒有在執行。", "en": "Docker is not running."}
    toggle(at, "讓模型寫程式").set_value(True).run()
    assert ".csv" not in types()  # switched on, but no Docker yet: no tool to read it
    sandbox["problem"] = None
    st.cache_data.clear()
    at.run()
    assert ".csv" in types() and ".xlsx" in types()


def test_without_docker_the_model_does_not_get_run_python(at, outputs, models, sandbox):
    sandbox["problem"] = {"zh": "Docker 沒有在執行。", "en": "Docker is not running."}
    toggle(at, "讓模型寫程式").set_value(True).run()
    ask(at)
    assert "run_python" not in [t["function"]["name"] for t in models[0].renders[0]["tools"]]
    tab(at).set_value("model").run()
    assert "這一輪沒有提供給模型" in page_html(at)


def test_the_csv_example_runs_python_on_the_attached_file(at, outputs, models, monkeypatch):
    commands = []

    def docker(cmd, **kwargs):
        commands.append((cmd, kwargs))
        return type("Done", (), {"returncode": 0, "stdout": "2025-12 244320\n2131200\n", "stderr": ""})()

    monkeypatch.setattr(tools.subprocess, "run", docker)
    code = 'import pandas as pd\ndf = pd.read_csv("/data/sales.csv")\nprint(df.groupby("month").revenue.sum().idxmax())'
    outputs += [tool_call("run_python", code=code), "12 月最高，全年 2,131,200。<turn|>"]
    toggle(at, "讓模型寫程式").set_value(True).run()
    next(b for b in at.button if b.key and b.key.startswith("example-") and "銷售資料" in b.label).click().run()
    settle(at)

    [(cmd, kwargs)] = commands
    [run] = at.session_state["runs"]
    folder = Path(run.attachments[0].path).parent
    assert f"{folder.resolve()}:/data:ro" in cmd and kwargs["input"] == code  # the conversation's files, read-only
    assert run.attachments[0].kind == "file" and models[0].encodes[0] == []  # a data file is not media for the model
    question = at.session_state["history"][0]["content"]
    assert question[-1]["text"].endswith("Attached for run_python, which reads them:\n"
                                         "/data/sales.csv (942 B), whose first line is: month,region,units,revenue")  # the columns, for the code
    assert "run_python" in [t["function"]["name"] for t in models[0].renders[0]["tools"]]
    html = page_html(at)
    assert '<span class="fn">run_python</span>({&quot;code&quot;: &quot;…&quot;})' in html and "df.groupby(&quot;month&quot;)" in html  # the code, written out
    assert "&quot;stdout&quot;: |\n    2025-12 244320\n    2131200" in html


def test_a_chart_is_shown_in_the_chat_and_in_the_step_that_drew_it(at, outputs, monkeypatch):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(buffer, format="PNG")
    stdout = f"\n{tools.FIGURES_MARK}\n{base64.b64encode(buffer.getvalue()).decode()}\n"
    monkeypatch.setattr(tools.subprocess, "run", lambda cmd, **kwargs: SimpleNamespace(returncode=0, stdout=stdout, stderr=""))
    outputs += [tool_call("run_python", code="import matplotlib.pyplot as plt\nplt.plot([1, 2])\nplt.show()"), "畫好了。<turn|>"]
    toggle(at, "讓模型寫程式").set_value(True).run()
    ask(at, "畫一條線。")
    [chat, step] = at.image  # the chat is drawn before the loop's tab
    assert chat.captions == step.captions == ["figure-1.png"]
    folder = Path(at.session_state["runs"][0].run.files)
    assert (folder / "figure-1.png").read_bytes() == buffer.getvalue()
    assert '&quot;figures&quot;: [\n    &quot;/data/figure-1.png&quot;' in page_html(at)  # what the model read, in the step card


def test_a_new_chat_removes_the_conversations_files(at, outputs):
    at.button(key="example-4").click().run()
    settle(at)
    folder = Path(at.session_state["runs"][0].attachments[0].path).parent
    assert folder.exists()
    click(at, "新對話")
    assert not folder.exists() and at.session_state["runs"] == []
