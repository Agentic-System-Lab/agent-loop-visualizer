"""The Streamlit page, driven through streamlit.testing with the fake model in place of Gemma.

Each test clicks and types like a user would, then waits for the agent thread
before the next rerun, the way the page's half-second refresh would.
"""

import time
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import agent
import llm
from fakes import FakeLLM, tool_call
from llm import Sampling

APP = Path(__file__).resolve().parent.parent / "streamlit_app.py"
ANSWER = "答案是 42。<turn|>"


@pytest.fixture
def outputs():
    """What the fake model will say next, one entry per step. Tests fill it in."""
    return []


@pytest.fixture
def models(monkeypatch, outputs):
    made = []

    class AppLLM(FakeLLM):
        def __init__(self, model_id):
            super().__init__(outputs, model_id)
            made.append(self)

    monkeypatch.setattr(llm, "LocalLLM", AppLLM)
    st.cache_resource.clear()  # a fresh model and a fresh generation lock per test
    yield made
    st.cache_resource.clear()


@pytest.fixture
def new_tab(models):
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
    return next(b for b in at.button_group if b.options == ["迴圈", "脈絡", "模型"])


# ---------- first load ----------

def test_first_load(at):
    assert [b.label for b in at.button if b.key and b.key.startswith("example-")] == [
        "台北現在幾點？紐約呢？",
        "(37.5 × 12) 的平方根，再加上 2 的 20 次方，是多少？",
        "ReAct 這篇論文是誰提出的？用 Wikipedia 查。",
        "Gemma 是哪家公司開發的？第一版什麼時候發布？",
    ]
    assert {t.label: t.value for t in at.toggle} == {"使用工具": True, "思考模式": False, "執行工具前先問我": False}
    assert {s.label: s.value for s in at.slider} == {
        "temperature": 1.0, "top-p": 0.95, "top-k": 64, "步數上限": 6, "每步 token 上限": 1024,
    }
    assert tab(at).value == "迴圈"
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
    tab(at).set_value("脈絡").run()
    assert [i.value for i in at.info] == ["送出問題後，這裡會顯示每一步模型實際讀到的完整字串。"]
    tab(at).set_value("迴圈").run()

    ask(at)
    at.button(key="ctx-0-1").click().run()
    assert tab(at).value == "脈絡"
    steps = next(b for b in at.button_group if b.options == ["Step 1", "Step 2"])
    assert steps.value == 1
    html = page_html(at)
    assert 'class="ctx-view"' in html and '<mark class="added"><span class="sp">&lt;|turn&gt;</span>user' in html

    steps.set_value(2).run()
    assert '<mark class="added">response:calculator' in page_html(at)
    assert any("system 1 · user 1 · assistant 1 · tool 1" in c for c in captions(at))


def test_model_view(at, outputs):
    tab(at).set_value("模型").run()
    html = page_html(at)
    assert "huggingface.co/google/gemma-4-E2B-it" in html
    assert "解碼設定" in html and "動作空間：模型看得到的工具" in html
    assert len(at.dataframe) == 0  # no steps yet

    outputs += [tool_call("calculator", expression="6*7"), ANSWER]
    ask(at)
    [table] = at.dataframe
    assert list(table.value["step"]) == [1, 2]


def test_model_view_shows_tools_off(at):
    next(t for t in at.toggle if t.label == "使用工具").set_value(False)
    tab(at).set_value("模型").run()
    assert "目前關閉了「使用工具」" in page_html(at)


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
