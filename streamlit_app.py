"""Agent Loop 可視化: chat with a local model (Gemma 4 E2B or Qwen3.5 0.8B, picked in the
sidebar) on the left and watch every step of its agent loop on the right.

Run with:  uv run streamlit run streamlit_app.py

The agent runs in a background thread (agent.Run) and appends one event per loop
stage to a list. While a run is live, a fragment redraws the page from that list
every half second. Clicking stop or approve reruns the script but not the thread,
so the loop keeps going.

A second page, 運作說明 (guide_page.py), walks through the loop with the code behind it.
"""

import json
import os
import threading
from pathlib import Path

import streamlit as st

import families
import render
from agent import Run, RunSettings
from llm import LocalLLM, Sampling
from tools import TOOLS

# The model menu; MODEL_ID picks the one selected at first, and may name a model the menu does not list.
DEFAULT_MODEL = os.environ.get("MODEL_ID", next(iter(families.MODELS)))
MODELS = {**families.MODELS, DEFAULT_MODEL: families.MODELS.get(DEFAULT_MODEL, DEFAULT_MODEL)}
ASSETS = Path(__file__).parent / "assets"
POLL_SECONDS = 0.5
BOX_HEIGHT = 620  # the chat and each tab scroll in a box this tall; side by side, style.css fits them to the window
DEFAULTS = {
    "use_tools": True,
    "thinking": False,
    "approve_tools": False,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "mtp": False,
    "max_steps": 6,
    "max_new_tokens": 1024,
}
EXAMPLES = [
    "台北現在幾點？紐約呢？",
    "(37.5 × 12) 的平方根，再加上 2 的 20 次方，是多少？",
    "ReAct 這篇論文是誰提出的？用 Wikipedia 查。",
    "Gemma 是哪家公司開發的？第一版什麼時候發布？",
    "What time is it in Paris right now?",
    "Who created the Transformer? Ask Wikipedia.",
    "Write a Python function to test for primes.",
]


@st.cache_resource(max_entries=1, show_spinner=False)  # one model in memory: switching models frees the last one
def load_llm(model_id: str) -> LocalLLM:
    llm = LocalLLM(model_id)
    llm.warmup()
    return llm


def current_llm() -> LocalLLM:
    model_id = st.session_state.model_id
    with st.spinner(f"載入 {model_id}（第一次要下載權重；之後約 10–20 秒）…"):
        llm = load_llm(model_id)
    if st.session_state.sampling_for != model_id:  # a model just chosen starts from its own recommended decoding, MTP off
        for key, value in {**llm.info["defaults"], "mtp": False}.items():
            st.session_state[key] = value
        st.session_state.sampling_for = model_id
    if st.session_state.mtp and llm.drafter_id and llm.drafter is None:
        with st.spinner(f"載入 MTP 草稿模型 {llm.drafter_id}（第一次要下載約 160 MB）…"):
            llm.load_drafter()
    return llm


class Slot:
    """One model, one run at a time, across every browser tab. The run is kept here as well as in
    session state so another tab can see it and stop it, e.g. after the tab that started it was reloaded."""

    def __init__(self):
        self.lock = threading.Lock()
        self.run: LiveRun | None = None


@st.cache_resource
def generation_slot() -> Slot:
    return Slot()


class LiveRun:
    """One user turn: the harness thread and the events it has emitted so far."""

    def __init__(self, llm: LocalLLM, history: list[dict], text: str, settings: RunSettings, slot: Slot):
        self.text = text
        self.events: list[dict] = []
        self.absorbed = False
        self.run = Run(llm, history, text, settings, self.events.append)
        self._slot = slot
        slot.run = self
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def _work(self):
        try:
            self.run.run()
        finally:
            if self._slot.run is self:
                self._slot.run = None  # do not keep the model alive after a switch to another one
            self._slot.lock.release()

    @property
    def live(self) -> bool:
        return self.thread.is_alive()

    def data(self) -> dict:
        return render.build(self.events[:])


# ---------- state ----------

def init_state():
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("runs", [])
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("view", None)
    st.session_state.setdefault("viz_tab", "迴圈")
    st.session_state.setdefault("model_id", DEFAULT_MODEL)
    st.session_state.setdefault("sampling_for", None)


def current_settings(llm: LocalLLM) -> RunSettings:
    s = st.session_state
    return RunSettings(
        sampling=Sampling(s.temperature, s.top_p, s.top_k, s.max_new_tokens, mtp=s.mtp and llm.drafter is not None),
        max_steps=s.max_steps,
        thinking=s.thinking,
        use_tools=s.use_tools,
        approve_tools=s.approve_tools,
    )


def start_run(llm: LocalLLM, text: str) -> bool:
    slot = generation_slot()
    if not slot.lock.acquire(blocking=False):
        st.toast("另一個 agent 正在執行（可能在別的分頁），請稍等。")
        return False
    try:
        run = LiveRun(llm, st.session_state.history, text, current_settings(llm), slot)
    except Exception:
        slot.lock.release()
        raise
    st.session_state.runs.append(run)
    st.session_state.view = len(st.session_state.runs) - 1
    return True


def other_tab_run() -> LiveRun | None:
    """A run that is still going but was started by another tab, or by this tab before a reload."""
    run = generation_slot().run
    return run if run and run.live and run not in st.session_state.runs else None


def absorb_finished():
    """When a run ends, its full message list becomes the history for the next turn."""
    for run in st.session_state.runs:
        if not run.live and not run.absorbed:
            end = next((e for e in reversed(run.events) if e["type"] == "run_end"), None)
            if end:
                st.session_state.history = end["messages"]
            run.absorbed = True


def reset_settings(defaults: dict):
    for key, value in {**DEFAULTS, **defaults}.items():
        st.session_state[key] = value


def new_chat():
    st.session_state.runs = []
    st.session_state.history = []
    st.session_state.view = None


def show_context(run_index: int, step: int):
    st.session_state.viz_tab = "脈絡"
    st.session_state[f"ctx_step_{run_index}"] = step


def show_run(run_index: int):
    st.session_state.view = run_index


# ---------- sidebar ----------

def sidebar(llm: LocalLLM, live: bool, busy: bool):
    info = llm.info
    with st.sidebar:
        st.markdown("### Agent Loop 可視化")
        st.selectbox(
            "模型", list(MODELS), format_func=MODELS.get, key="model_id", on_change=new_chat, disabled=busy,
            help="換模型會開始新的對話，解碼設定也換成新模型的建議值。harness、工具和畫面都不變。",
        )
        st.caption(f"`{llm.model_id}` · {llm.family.name} · {info['device'].upper()} · {info['dtype']} · 載入 {info['load_seconds']} s")

        st.markdown("##### 迴圈")
        st.toggle("使用工具", key="use_tools", help="關掉就只是聊天機器人：模型看不到任何工具。")
        st.toggle("思考模式", key="thinking", help=llm.family.thinking_help)
        st.toggle("執行工具前先問我", key="approve_tools", help="由 harness 擋下等你確認，而不是寫在 prompt 裡請模型遵守。")

        st.markdown("##### 解碼與上限")
        st.slider("temperature", 0.0, 2.0, step=0.05, key="temperature", help="0 = greedy，永遠選機率最高的 token。")
        st.slider("top-p", 0.05, 1.0, step=0.05, key="top_p")
        st.slider("top-k", 0, 128, key="top_k", help="0 = 不限。")
        st.toggle("MTP 推測解碼", key="mtp", disabled=busy or not llm.drafter_id, help=mtp_help(llm))
        st.slider("步數上限", 1, 100, key="max_steps", help="harness 的迴圈最多跑幾圈（step），跑滿就停。")
        st.slider("每步 token 上限", 64, 8192, step=64, key="max_new_tokens", help="一次推論最多生成幾個 token。")
        st.button("還原成模型建議值", on_click=reset_settings, args=(info["defaults"],), width="stretch")
        st.button("新對話", on_click=new_chat, disabled=live, width="stretch")
        st.caption("設定會套用到下一個問題。")


def mtp_help(llm: LocalLLM) -> str:
    if not llm.drafter_id:
        return "這個模型沒有搭配的 MTP 草稿模型，目前只有 Gemma 4 E2B 有。"
    return (
        f"用 {llm.drafter_id} 當草稿模型：每生成一個 token，它就從主模型的狀態一口氣猜接下來幾個 token，"
        "主模型再用一次 forward 全部驗證。輸出的分布和不開一樣，只是比較快。第一次打開會載入草稿模型（約 160 MB）。"
    )


# ---------- left: chat ----------

def chat_panel(llm: LocalLLM, live_run: LiveRun | None, other: LiveRun | None):
    runs = st.session_state.runs
    busy = live_run is not None or other is not None
    box = st.container(height=BOX_HEIGHT, border=False, key="chat_box")
    with box:
        if not runs:
            st.markdown(
                "送出一個問題，右邊會一步一步畫出 agent 迴圈：模型讀進脈絡、提出動作、"
                "harness 去執行、結果接回脈絡，直到停止。"
            )
            st.caption("試試看")
            for i, example in enumerate(EXAMPLES):
                if st.button(example, key=f"example-{i}", disabled=busy):
                    if start_run(llm, example):
                        st.rerun()
        for i, run in enumerate(runs):
            with st.chat_message("user"):
                st.markdown(run.text)
            with st.chat_message("assistant"):
                assistant_message(i, run)

    if live_run:
        st.button("停止這次執行", icon=":material/stop_circle:", on_click=live_run.run.stop, width="stretch")
    if other:
        st.warning(f"另一個分頁的 agent 還在執行（「{other.text[:30]}」），可能是重新整理前留下的。它結束之前不能送出新問題。")
        st.button("停止那個 agent", icon=":material/stop_circle:", on_click=other.run.stop, width="stretch")
    prompt = st.chat_input("問一個問題…", key="chat_input", disabled=busy)
    if prompt and start_run(llm, prompt):
        st.rerun()


def assistant_message(i: int, run: LiveRun):
    data = run.data()
    trace = render.trace_html(data)
    if trace:
        st.html(trace)

    for call in render.pending_approvals(data):
        with st.container(border=True):
            args = json.dumps(call["arguments"], ensure_ascii=False)
            st.markdown(f"**Agent 想執行這個工具，要允許嗎？**  \n`{call['name']}({args})`")
            allow, reject = st.columns(2)
            allow.button("允許", key=f"allow-{i}-{call['call_id']}", type="primary", width="stretch",
                         on_click=run.run.decide, args=(call["call_id"], True))
            reject.button("拒絕", key=f"reject-{i}-{call['call_id']}", width="stretch",
                          on_click=run.run.decide, args=(call["call_id"], False))

    if run.live:
        st.caption(render.status(data))
    text = data["end"]["final"] if data["end"] else render.draft(data)
    if text:
        st.markdown(text)
    if data["error"]:
        st.error(data["error"])
    if data["end"]:
        if not text:
            st.caption("（這一輪沒有最終回覆，看右邊的停止原因。）")
        foot, button = st.columns([3, 1], vertical_alignment="center")
        foot.caption(render.footer(data))
        if st.session_state.view != i:
            button.button("看流程", key=f"view-{i}", type="tertiary", on_click=show_run, args=(i,))


# ---------- right: the loop ----------

def viz_panel(llm: LocalLLM):
    runs = st.session_state.runs
    index = st.session_state.view
    run = runs[index] if runs and index is not None else None
    data = run.data() if run else None

    head, label = st.columns([2, 3], vertical_alignment="center")
    with head:
        tab = st.segmented_control("檢視", ["迴圈", "脈絡", "模型"], key="viz_tab", required=True, label_visibility="collapsed")
    if run and len(runs) > 1:
        label.caption(f"第 {index + 1} 輪：{run.text[:40]}")

    # Each tab scrolls in its own box, so a scroll position never carries over to another tab.
    view = {"脈絡": "context", "模型": "model"}.get(tab, "loop")
    with st.container(height=BOX_HEIGHT, border=False, key=f"{view}_view"):
        if view == "context":
            context_panel(index, data)
        elif view == "model":
            model_panel(llm, data)
        else:
            loop_panel(index, run, data, llm.model_id)


def loop_panel(index: int | None, run: LiveRun | None, data: dict | None, model_id: str):
    st.html(render.diagram_html(data, dark=st.context.theme.type == "dark", model=model_id))
    if data is None:
        st.html(render.INTRO)
        return
    st.caption(render.run_stats(data))
    # A fixed height makes it scroll; style.css shrinks it to the space left under the diagram.
    with st.container(height=BOX_HEIGHT, border=False, autoscroll=run.live, key="timeline"):
        steps = data["steps"]
        for step in steps:
            with st.container(border=True):
                live_step = run.live and step is steps[-1]
                st.html(render.step_html(step, data, live=live_step))
                if step["context"]:
                    st.button("看這一步的完整脈絡 →", key=f"ctx-{index}-{step['n']}", type="tertiary",
                              on_click=show_context, args=(index, step["n"]))
        if data["end"]:
            st.html(render.stop_html(data))


def context_panel(index: int | None, data: dict | None):
    steps = [s for s in data["steps"] if s["context"]] if data else []
    if not steps:
        st.info("送出問題後，這裡會顯示每一步模型實際讀到的完整字串。")
        return
    key = f"ctx_step_{index}"
    if st.session_state.get(key) is None:
        st.session_state[key] = steps[-1]["n"]
    choice = st.segmented_control(
        "步驟", [s["n"] for s in steps], key=key, required=True, format_func=lambda n: f"Step {n}"
    )
    step = next((s for s in steps if s["n"] == choice), steps[-1])
    st.caption(render.context_summary(step))
    st.html(render.CONTEXT_LEGEND + render.context_html(step, data["family"]))


def model_panel(llm: LocalLLM, data: dict | None):
    st.html(render.model_html(llm.info))
    settings = data["settings"] if data else {**DEFAULTS, **{k: st.session_state[k] for k in DEFAULTS}}
    st.html(render.knobs_html(settings, llm.info["defaults"]))
    if data and data["steps"]:
        st.markdown("##### 每一步的推論")
        st.dataframe(render.step_rows(data), hide_index=True, width="stretch")
    st.html(render.tools_html(TOOLS, settings["use_tools"], llm.family))
    st.html(render.special_tokens_html(llm.info))


# ---------- pages ----------

def keep_widget_values():
    """Streamlit drops a widget's value while its page is not shown. Writing the values back as plain
    session state keeps the sidebar settings and the selected view across a visit to the guide."""
    for key in (*DEFAULTS, "viz_tab", "model_id"):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def agent_page():
    init_state()
    llm = current_llm()
    own_live = any(run.live for run in st.session_state.runs)
    polling = own_live or other_tab_run() is not None
    sidebar(llm, own_live, polling)

    @st.fragment(run_every=POLL_SECONDS if polling else None)
    def workspace():
        absorb_finished()
        live_run = next((run for run in st.session_state.runs if run.live), None)
        other = other_tab_run()
        if polling and live_run is None and other is None:
            st.rerun()  # the run just ended: redraw everything once and stop polling
        with st.container(key="workspace"):
            chat, viz = st.columns([5, 7], gap="large")
            with chat:
                chat_panel(llm, live_run, other)
            with viz:
                viz_panel(llm)

    workspace()


def main():
    st.set_page_config(layout="wide")
    st.html(ASSETS / "style.css")
    if st.context.theme.type == "dark":
        st.html(ASSETS / "dark.css")
    keep_widget_values()
    page = st.navigation(
        [
            st.Page(agent_page, title="Agent Loop 可視化", icon=":material/autorenew:", default=True),
            st.Page("guide_page.py", title="運作說明", icon=":material/menu_book:", url_path="guide"),
        ],
        position="top",
    )
    page.run()


main()
