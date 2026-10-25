"""Agent Loop 可視化 (Agent Loop Visualizer): chat with a local model (Gemma 4 E2B or Qwen3.5 0.8B, picked in the
sidebar) on the left and watch every step of its agent loop on the right.

Run with:  uv run streamlit run streamlit_app.py

The agent runs in a background thread (agent.Run) and appends one event per loop
stage to a list. While a run is live, a fragment redraws the page from that list
every half second. Clicking stop or approve reruns the script but not the thread,
so the loop keeps going.

With VLLM_URL set, a vLLM server runs the model instead of this machine (vllm_backend.py), and the
model menu lists what that server serves.

The top bar also links to 運作說明 (How it works), a walk through the loop with the code behind it:
a static page that guide.py builds and GitHub Pages serves.

The page is in Traditional Chinese or English (i18n.py). It starts in the language last picked on the
switch at the top right, kept in a cookie, or else in the browser's language.

A question can come with files (media.py): images, sound and video the model reads, and, when the sidebar
lets the model write code, data files for run_python. Only then does the page look for Docker, which
run_python's sandbox needs, and say how to install it when it is missing. The charts run_python draws are
shown in the chat and in the step that drew them.
"""

import json
import os
import shutil
import threading
import uuid
from pathlib import Path

import streamlit as st

import families
import media
import render
import tools
import vllm_backend
from agent import Run, RunSettings
from backend import Backend, Sampling
from i18n import LANGS, from_locale, pick
from simple_backend import LocalLLM
from tools import TOOLS

# The model menu; MODEL_ID picks the one selected at first, and may name a model the menu does not list.
DEFAULT_MODEL = os.environ.get("MODEL_ID", next(iter(families.MODELS)))
MODELS = {**families.MODELS, DEFAULT_MODEL: families.MODELS.get(DEFAULT_MODEL, DEFAULT_MODEL)}
# Where a vLLM server listens (http://localhost:8000, say); with it set, the server runs the model, not this machine.
VLLM_URL = os.environ.get("VLLM_URL")
ASSETS = Path(__file__).parent / "assets"
SAMPLES = ASSETS / "examples"  # the files some examples attach, and the sample files offered above the input box
# Each conversation's files go in a folder of their own under here. It is under the home folder because that is
# the one every Docker setup (Docker Desktop, OrbStack, Colima) shares with its containers.
UPLOADS = Path(os.environ.get("AGENT_LOOP_UPLOADS", Path.home() / ".cache" / "agent-loop-visualizer" / "uploads"))
MAX_UPLOAD_MB = 50
GUIDE_URL = "https://cyyeh.github.io/agent-loop-visualizer/"  # built from guide.py; zh/ and en/ under it are the two languages
LANG_COOKIE = "agent_loop_lang"  # the language last picked on the switch
POLL_SECONDS = 0.5
BOX_HEIGHT = 620  # the chat and each tab scroll in a box this tall; side by side, style.css fits them to the window
DEFAULTS = {
    "use_tools": True,
    "python": False,  # let the model write code: offers run_python, which needs Docker
    "thinking": False,
    "approve_tools": False,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "mtp": False,
    "max_steps": 6,
    "max_new_tokens": 1024,
}
# The examples under the welcome, in the page's language only: (question, what it needs). "python" needs run_python;
# a file name attaches that file from assets/examples, and the example shows only when the model can use it.
EXAMPLES = {
    "zh": [
        ("台北現在幾點？紐約呢？", None),
        ("(37.5 × 12) 的平方根，再加上 2 的 20 次方，是多少？", None),
        ("ReAct 這篇論文是誰提出的？用 Wikipedia 查。", None),
        ("Gemma 是哪家公司開發的？第一版什麼時候發布？", None),
        ("這張收據上的品項加起來是多少？和收據上寫的總計一樣嗎？", "receipt.png"),
        ("影片裡的紅球是從左邊滾到右邊，還是從右邊滾到左邊？", "ball.mp4"),
        ("把這段錄音的內容寫下來，再翻成繁體中文。", "speech.flac"),
        ("寫一個 Python 函式判斷質數，用它算出 10000 以內有幾個質數。", "python"),
        ("用 Python 讀這份銷售資料：哪個月的總營收最高？全年總營收是多少？", "sales.csv"),
    ],
    "en": [
        # One city: asked about two at once in English, Gemma often replies in Chinese or not at all.
        ("What time is it in Taipei right now?", None),
        ("What is the square root of (37.5 × 12), plus 2 to the power of 20?", None),
        ("Who proposed the ReAct paper? Look it up on Wikipedia.", None),
        ("Which company developed Gemma, and when was the first version released?", None),
        ("Who created the Transformer? Ask Wikipedia.", None),
        ("What do the items on this receipt add up to? Does it match the total printed on it?", "receipt.png"),
        ("Does the red ball in this video roll from left to right, or from right to left?", "ball.mp4"),
        # Asked to transcribe in English with tools on, Gemma tends to say it cannot hear; a question on the content works more often.
        ('The speaker in this recording calls someone "the apostle of the middle classes". Who?', "speech.flac"),
        ("Write a Python function that tests for primes, and use it to count the primes below 10,000.", "python"),
        ("Use Python to read this sales data: which month had the highest total revenue, and what was the revenue for the whole year?", "sales.csv"),
    ],
}
EXAMPLE_ICONS = {"image": ":material/image:", "audio": ":material/graphic_eq:", "video": ":material/movie:", "file": ":material/table_chart:", "python": ":material/code:"}
SAMPLE_FILES = sorted(p.name for p in SAMPLES.iterdir() if p.is_file() and not p.name.startswith("."))


def tr(zh: str, en: str) -> str:
    """The text in the language the page is in."""
    return pick(st.session_state.lang, zh, en)


@st.cache_data(ttl=60, show_spinner=False)
def model_menu() -> dict[str, str]:
    """The model menu, id -> label: MODELS, or with VLLM_URL the models the vLLM server serves, each by what it was
    loaded from, which is the Hugging Face id its tokenizer is downloaded by."""
    if not VLLM_URL:
        return MODELS
    served = (card.get("root") or card["id"] for card in vllm_backend.served_models(VLLM_URL))
    return {model_id: families.MODELS.get(model_id, model_id) for model_id in served}


def default_model() -> str:
    menu = model_menu()
    return DEFAULT_MODEL if DEFAULT_MODEL in menu else next(iter(menu))


@st.cache_resource(max_entries=1, show_spinner=False)  # one model in memory: switching models frees the last one
def load_llm(model_id: str) -> Backend:
    llm = vllm_backend.VLLMBackend(model_id, VLLM_URL) if VLLM_URL else LocalLLM(model_id)
    llm.warmup()
    return llm


def current_llm() -> Backend:
    model_id = st.session_state.model_id
    if VLLM_URL:
        loading = tr(f"連線到 {VLLM_URL} 的 vLLM 伺服器，載入 {model_id} 的 tokenizer…", f"Connecting to the vLLM server at {VLLM_URL} and loading {model_id}'s tokenizer…")
    else:
        loading = tr(f"載入 {model_id}（第一次要下載權重；之後約 10–20 秒）…", f"Loading {model_id} (the first time downloads the weights; after that it takes about 10–20 s)…")
    with st.spinner(loading):
        llm = load_llm(model_id)
    if st.session_state.sampling_for != model_id:  # a model just chosen starts from its own recommended decoding, MTP off
        for key, value in {**llm.info["defaults"], "mtp": False}.items():
            st.session_state[key] = value
        st.session_state.sampling_for = model_id
    if st.session_state.mtp and llm.drafter_id and llm.drafter is None:
        with st.spinner(tr(f"載入 MTP 草稿模型 {llm.drafter_id}（第一次要下載約 190 MB）…", f"Loading the MTP drafter {llm.drafter_id} (the first time downloads about 190 MB)…")):
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

    def __init__(self, llm: Backend, history: list[dict], text: str, attachments: list[media.Attachment], files: Path,
                 settings: RunSettings, slot: Slot):
        self.text = text
        self.attachments = attachments
        self.events: list[dict] = []
        self.absorbed = False
        self.run = Run(llm, history, text, settings, self.events.append, attachments, files)
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
    st.session_state.setdefault("viz_tab", "loop")
    if "model_id" not in st.session_state:  # asking the vLLM server what it serves only when there is no choice yet
        st.session_state.model_id = default_model()
    st.session_state.setdefault("sampling_for", None)
    st.session_state.setdefault("chat_id", uuid.uuid4().hex[:12])


def chat_folder() -> Path:
    """This conversation's files: what the user attached, and what run_python sees as /data."""
    return UPLOADS / st.session_state.chat_id


def current_settings(llm: Backend) -> RunSettings:
    s = st.session_state
    return RunSettings(
        sampling=Sampling(s.temperature, s.top_p, s.top_k, s.max_new_tokens, mtp=s.mtp and llm.drafter is not None),
        max_steps=s.max_steps,
        thinking=s.thinking,
        use_tools=s.use_tools,
        tools=offered_tools(),
        approve_tools=s.approve_tools,
        lang=s.lang,
    )


@st.cache_data(ttl=10, show_spinner=False)
def sandbox_problem() -> dict | None:
    """What keeps run_python from running (tools.sandbox_problem), asked again at most every 10 s."""
    return tools.sandbox_problem()


def code_allowed() -> bool:
    """Whether the sidebar lets the model write code; Docker is looked for only then."""
    return st.session_state.use_tools and st.session_state.python


def code_ready() -> bool:
    return code_allowed() and sandbox_problem() is None


def offered_tools() -> tuple[str, ...]:
    """Every tool, but run_python only when the model may write code and Docker can run it."""
    return tuple(name for name in TOOLS if name != "run_python" or code_ready())


def start_run(llm: Backend, text: str, uploads: list[tuple[str, bytes]] = ()) -> bool:
    """Ask the question in a new run, with `uploads` (file name, contents) saved in the conversation's folder first."""
    slot = generation_slot()
    if not slot.lock.acquire(blocking=False):
        st.toast(tr("另一個 agent 正在執行（可能在別的分頁），請稍等。", "Another agent is running (maybe in another tab). Please wait."))
        return False
    try:
        attachments = [media.save(chat_folder(), name, data, llm.modalities) for name, data in uploads]
        run = LiveRun(llm, st.session_state.history, text, attachments, chat_folder(), current_settings(llm), slot)
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
    if "chat_id" in st.session_state:  # the old conversation's files go with it
        shutil.rmtree(chat_folder(), ignore_errors=True)
    st.session_state.chat_id = uuid.uuid4().hex[:12]


def show_context(run_index: int, step: int):
    st.session_state.viz_tab = "context"
    st.session_state[f"ctx_step_{run_index}"] = step


def show_run(run_index: int):
    st.session_state.view = run_index


# ---------- sidebar ----------

def sidebar(llm: Backend, menu: dict[str, str], live: bool, busy: bool):
    info, lang = llm.info, st.session_state.lang
    with st.sidebar:
        st.markdown(tr("### Agent Loop 可視化", "### Agent Loop Visualizer"))
        st.selectbox(
            tr("模型", "Model"), list(menu), format_func=menu.get, key="model_id", on_change=new_chat, disabled=busy,
            help=tr("換模型會開始新的對話，解碼設定也換成新模型的建議值。harness、工具和畫面都不變。",
                    "Switching models starts a new conversation and resets the decoding settings to what the new model recommends. "
                    "The harness, the tools and the page stay the same."),
        )
        if server := info.get("server"):
            st.caption(f"`{llm.model_id}` · {llm.family.name} · {server['engine']} {server.get('version') or ''} · {server['url']}")
        else:
            st.caption(f"`{llm.model_id}` · {llm.family.name} · {info['device'].upper()} · {info['dtype']} · "
                       + tr(f"載入 {info['load_seconds']} s", f"loaded in {info['load_seconds']} s"))

        st.markdown(tr("##### 迴圈", "##### Loop"))
        st.toggle(tr("使用工具", "Use tools"), key="use_tools", help=tr("關掉就只是聊天機器人：模型看不到任何工具。", "Off, it is just a chatbot: the model sees no tools."))
        st.toggle(tr("讓模型寫程式", "Let the model write code"), key="python", disabled=not st.session_state.use_tools,
                  help=tr("打開後模型多一個 run_python 工具：它寫的 Python 在 Docker 容器裡執行（沒有網路，檔案唯讀），"
                          "也讀得到你附上的資料檔（CSV、Excel…）。需要先安裝 Docker；沒打開就不會用到 Docker。",
                          "Gives the model one more tool, run_python: the Python it writes runs in a Docker container (no network, files "
                          "read-only), and it can read the data files you attach (CSV, Excel…). It needs Docker; with this off, Docker is never used."))
        if code_allowed() and (problem := sandbox_problem()):
            st.warning(problem[lang], icon=":material/warning:")
        st.toggle(tr("思考模式", "Thinking mode"), key="thinking", help=llm.family.thinking_help[lang])
        st.toggle(tr("執行工具前先問我", "Ask me before running a tool"), key="approve_tools",
                  help=tr("由 harness 擋下等你確認，而不是寫在 prompt 裡請模型遵守。",
                          "The harness holds the call until you confirm it, instead of a line in the prompt asking the model to check first."))

        st.markdown(tr("##### 解碼與上限", "##### Decoding and limits"))
        if (info.get("server") or {}).get("sampling_mask"):
            # vLLM refuses greedy decoding and an unlimited top-k once it sends sampling masks, so the sliders stop short
            # of them, and a setting from before moves to the nearest one it takes.
            st.session_state.temperature = max(st.session_state.temperature, 0.05)
            st.session_state.top_k = st.session_state.top_k or 128
            st.slider("temperature", 0.05, 2.0, step=0.05, key="temperature",
                      help=tr("這台 vLLM 伺服器會送回 sampling mask（`--return-sampling-mask`），這時它不接受 greedy（temperature 0），所以最低是 0.05。",
                              "This vLLM server sends sampling masks (`--return-sampling-mask`), and then it refuses greedy decoding (temperature 0), so the lowest is 0.05."))
            st.slider("top-p", 0.05, 1.0, step=0.05, key="top_p")
            st.slider("top-k", 1, 128, key="top_k",
                      help=tr("這台 vLLM 伺服器會送回 sampling mask，要靠 top-k 限制它的大小，所以不能設 0（不限）。",
                              "This vLLM server sends sampling masks, and needs top-k to bound them, so 0 (no limit) is not allowed."))
        else:
            st.slider("temperature", 0.0, 2.0, step=0.05, key="temperature", help=tr("0 = greedy，永遠選機率最高的 token。", "0 = greedy: always the most likely token."))
            st.slider("top-p", 0.05, 1.0, step=0.05, key="top_p")
            st.slider("top-k", 0, 128, key="top_k", help=tr("0 = 不限。", "0 = no limit."))
        st.toggle(tr("MTP 推測解碼", "MTP speculative decoding"), key="mtp", disabled=busy or not llm.drafter_id, help=mtp_help(llm))
        st.slider(tr("步數上限", "Max steps"), 1, 100, key="max_steps",
                  help=tr("harness 的迴圈最多跑幾圈（step），跑滿就停。", "How many times (steps) the harness may go around the loop before it stops."))
        st.slider(tr("每步 token 上限", "Max tokens per step"), 64, 8192, step=64, key="max_new_tokens",
                  help=tr("一次推論最多生成幾個 token。", "The most tokens one inference may generate."))
        st.button(tr("還原成模型建議值", "Reset to the model's defaults"), on_click=reset_settings, args=(info["defaults"],), width="stretch")
        st.button(tr("新對話", "New chat"), on_click=new_chat, disabled=live, width="stretch")
        st.caption(tr("設定會套用到下一個問題。", "Settings apply to the next question."))


def mtp_help(llm: Backend) -> str:
    if server := llm.info.get("server"):
        return tr(
            f"{server['engine']} 伺服器啟動時就決定要不要推測解碼（`--speculative-config`），這裡不能切換。伺服器加上 "
            "`--per-request-spec-decode-metrics summary` 的話，每一步會顯示它收下幾個猜測，但不會說是哪幾個 token。",
            f"Whether the {server['engine']} server decodes speculatively is set when it starts (`--speculative-config`), not here. "
            "Started with `--per-request-spec-decode-metrics summary`, it says how many guesses it kept in each step, though not which tokens they were.",
        )
    if not llm.drafter_id:
        return tr("這個模型沒有搭配的 MTP 草稿模型，目前只有 Gemma 4 E2B 有。", "This model has no MTP drafter to go with it; for now only Gemma 4 E2B has one.")
    return tr(
        f"用 {llm.drafter_id} 當草稿模型：每生成一個 token，它就從主模型的狀態一口氣猜接下來幾個 token，"
        "主模型再用一次 forward 全部驗證。輸出的分布和不開一樣，只是比較快。第一次打開會載入草稿模型（約 160 MB）。",
        f"Uses {llm.drafter_id} as the drafter: after every token it guesses the next few at once from the model's state, "
        "and the model checks them all in one forward pass. The output follows the same distribution as without it, only faster. "
        "Switching it on the first time loads the drafter (about 160 MB).",
    )


# ---------- left: chat ----------

def chat_panel(llm: Backend, live_run: LiveRun | None, other: LiveRun | None):
    runs = st.session_state.runs
    busy = live_run is not None or other is not None
    box = st.container(height=BOX_HEIGHT, border=False, key="chat_box")
    with box:
        if not runs:
            st.markdown(tr(
                "送出一個問題，右邊會一步一步畫出 agent 迴圈：模型讀進脈絡、提出動作、"
                "harness 去執行、結果接回脈絡，直到停止。",
                "Send a question, and the agent loop is drawn on the right step by step: the model reads the context and proposes "
                "an action, the harness carries it out, and the result goes back into the context, until it stops.",
            ))
            if llm.modalities:
                kinds = pick(st.session_state.lang, "、", ", ").join(render.MODALITIES[st.session_state.lang][k] for k in llm.modalities)
                st.caption(tr(f"這個模型讀得懂{kinds}：用輸入框左邊的 ＋ 附上檔案"
                              + ("，或按麥克風錄一段話。" if "audio" in llm.modalities else "。"),
                              f"This model reads {kinds}: attach a file with the + at the left of the input box"
                              + (", or record yourself with the microphone." if "audio" in llm.modalities else ".")))
            if refused := (llm.info.get("server") or {}).get("refused"):  # what the vLLM server did not take, and so cannot be attached
                kinds = pick(st.session_state.lang, "、", ", ").join(render.MODALITIES[st.session_state.lang][k] for k in refused)
                st.caption(tr(f"這台 vLLM 伺服器不收{kinds}，所以不能附上；原因寫在模型分頁。",
                              f"This vLLM server does not take {kinds}, so they cannot be attached; the Model tab says why."))
            st.caption(tr("試試看", "Try one"))
            examples(llm, busy)
        for i, run in enumerate(runs):
            with st.chat_message("user"):
                attachments_view(run.attachments)
                if run.text:
                    st.markdown(run.text)
            with st.chat_message("assistant"):
                assistant_message(i, run)

    if live_run:
        st.button(tr("停止這次執行", "Stop this run"), icon=":material/stop_circle:", on_click=live_run.run.stop, width="stretch")
    if other:
        st.warning(tr(f"另一個分頁的 agent 還在執行（「{other.text[:30]}」），可能是重新整理前留下的。它結束之前不能送出新問題。",
                      f"An agent from another tab is still running (“{other.text[:30]}”), perhaps left over from before a reload. "
                      "No new question can be sent until it ends."))
        st.button(tr("停止那個 agent", "Stop that agent"), icon=":material/stop_circle:", on_click=other.run.stop, width="stretch")
    types = media.accepted(llm.modalities, python=code_ready())  # a data file is no use to the model without run_python
    samples = [name for name in SAMPLE_FILES if Path(name).suffix.lstrip(".") in types]
    picked = [name for name in st.session_state.get("samples") or [] if name in samples]
    sample_row = st.container() if samples else None  # filled after the input box, so that sending a question can clear the samples picked for it
    if picked:
        names = ", ".join(picked)
        placeholder = tr(f"問一個關於 {names} 的問題…", f"Ask a question about {names}…")
    else:
        placeholder = tr("問一個問題，也可以附上檔案…", "Ask a question; you can attach files…") if types else tr("問一個問題…", "Ask a question…")
    sent = st.chat_input(
        placeholder, key="chat_input", disabled=busy, accept_file="multiple" if types else False, file_type=types or None,
        accept_audio="audio" in llm.modalities, max_upload_size=MAX_UPLOAD_MB,
    )
    text, uploads = (sent, []) if isinstance(sent, str) or sent is None else (sent.text, uploaded(sent))
    if (text or uploads) and start_run(llm, text or "", uploads + [(name, (SAMPLES / name).read_bytes()) for name in picked]):
        st.session_state.samples = []
        st.rerun()
    if sample_row:
        sample_picker(sample_row, samples, llm, busy)


def sample_picker(row, samples: list[str], llm: Backend, busy: bool):
    """The sample files from assets/examples that this model can use, to ask a question of one's own about: a file
    picked here goes with the next question typed in the input box, as one attached with the + would."""
    def label(name: str) -> str:
        return f"{EXAMPLE_ICONS[media.kind_of(name, llm.modalities)]} {name}"

    with row:
        st.pills(tr("範例檔案", "Sample files"), samples, selection_mode="multi", format_func=label, key="samples", disabled=busy,
                 help=tr("選了的檔案會附在你接下來輸入的問題，和用 ＋ 附上的一樣。", "A file picked here goes with the next question you type, as if attached with the +."))


def uploaded(sent) -> list[tuple[str, bytes]]:
    """The files sent with a message, and a recording made with the microphone, as (file name, contents)."""
    files = [(f.name, f.getvalue()) for f in sent.get("files") or []]  # a key is missing when the input does not take its kind
    return files + ([("recording.wav", audio.getvalue())] if (audio := sent.get("audio")) else [])


def examples(llm: Backend, busy: bool):
    """The page language's examples that this model can try: one that attaches a file shows only if the model reads
    that kind, and one that needs Python only once the model may write code, disabled until Docker is ready."""
    lang, hidden = st.session_state.lang, False
    for i, (question, needs) in enumerate(EXAMPLES[lang]):
        sample = needs if needs not in (None, "python") else None
        kind = media.KINDS.get(Path(sample).suffix.lstrip("."), "file") if sample else needs
        if kind in media.MEDIA and kind not in llm.modalities:
            continue
        if kind in ("python", "file") and not code_allowed():
            hidden = True
            continue
        problem = sandbox_problem() if kind in ("python", "file") else None
        note = problem[lang] if problem else tr(f"會附上範例檔案 {sample}", f"Attaches the sample file {sample}") if sample else None
        if st.button(question, key=f"example-{i}", icon=EXAMPLE_ICONS.get(kind), help=note, disabled=busy or problem is not None):
            if start_run(llm, question, [(sample, (SAMPLES / sample).read_bytes())] if sample else []):
                st.rerun()
    if hidden:
        st.caption(tr("打開側欄的「讓模型寫程式」，還有要寫 Python 的範例。", "Turn on Let the model write code in the sidebar for examples that need Python."))


def attachments_view(attachments: list[media.Attachment]):
    for a in attachments:
        if a.kind == "image":
            st.image(a.path, width=240)
        elif a.kind == "audio":
            st.audio(a.path, format=media.mime(a.name) or "audio/wav")
        elif a.kind == "video" and media.mime(a.name):
            st.video(a.path, format=media.mime(a.name))
        elif a.kind == "video":  # a video some browsers cannot play: the frames the model reads instead
            st.image(frame_grid(a.path), width=320, caption=tr("瀏覽器不一定播得了這種影片；這是模型讀到的其中幾張畫格",
                                                     "Browsers may not play this kind of video; here are some of the frames the model reads"))
        st.caption(f"📎 {a.name} · {media.fmt_size(a.size)}")


@st.cache_data(show_spinner=False, max_entries=8)
def frame_grid(path: str):
    """Made once per video, not on every redraw of a live run."""
    return media.frame_grid(path)


def assistant_message(i: int, run: LiveRun):
    lang = st.session_state.lang
    data = run.data()
    trace = render.trace_html(data, lang)
    if trace:
        st.html(trace)
    for path in render.images(data["steps"]):  # the charts run_python drew, as soon as they are back
        st.image(path, caption=Path(path).name)

    for call in render.pending_approvals(data):
        with st.container(border=True):
            # An argument over several lines (code) is shown below the call rather than squeezed into its JSON.
            blocks = {k: v for k, v in call["arguments"].items() if isinstance(v, str) and "\n" in v.strip()}
            args = json.dumps({k: "…" if k in blocks else v for k, v in call["arguments"].items()}, ensure_ascii=False)
            st.markdown(tr("**Agent 想執行這個工具，要允許嗎？**", "**The agent wants to run this tool. Allow it?**") + f"  \n`{call['name']}({args})`")
            for code in blocks.values():
                st.code(code)
            allow, reject = st.columns(2)
            allow.button(tr("允許", "Allow"), key=f"allow-{i}-{call['call_id']}", type="primary", width="stretch",
                         on_click=run.run.decide, args=(call["call_id"], True))
            reject.button(tr("拒絕", "Reject"), key=f"reject-{i}-{call['call_id']}", width="stretch",
                          on_click=run.run.decide, args=(call["call_id"], False))

    if run.live:
        st.caption(render.status(data, lang))
    text = data["end"]["final"] if data["end"] else render.draft(data)
    if text:
        st.markdown(text)
    if data["error"]:
        st.error(data["error"])
    if data["end"]:
        if not text:
            st.caption(tr("（這一輪沒有最終回覆，看右邊的停止原因。）", "(No final reply in this turn; the right side says why it stopped.)"))
        foot, button = st.columns([3, 1], vertical_alignment="center")
        foot.caption(render.footer(data, lang))
        if st.session_state.view != i:
            button.button(tr("看流程", "Show the loop"), key=f"view-{i}", type="tertiary", on_click=show_run, args=(i,))


# ---------- right: the loop ----------

def viz_panel(llm: Backend):
    runs = st.session_state.runs
    index = st.session_state.view
    run = runs[index] if runs and index is not None else None
    data = run.data() if run else None

    # The switcher keeps its own width; in a narrow column the caption wraps under it.
    with st.container(horizontal=True, vertical_alignment="center"):
        names = {"loop": tr("迴圈", "Loop"), "context": tr("脈絡", "Context"), "model": tr("模型", "Model")}
        view = st.segmented_control(tr("檢視", "View"), list(names), format_func=names.get, key="viz_tab", required=True, label_visibility="collapsed")
        if run and len(runs) > 1:
            st.caption(tr(f"第 {index + 1} 輪：{run.text[:40]}", f"Turn {index + 1}: {run.text[:40]}"))

    # Each tab scrolls in its own box, so a scroll position never carries over to another tab.
    with st.container(height=BOX_HEIGHT, border=False, key=f"{view}_view"):
        if view == "context":
            context_panel(index, data)
        elif view == "model":
            model_panel(llm, data)
        else:
            loop_panel(index, run, data, llm.model_id)


def loop_panel(index: int | None, run: LiveRun | None, data: dict | None, model_id: str):
    lang = st.session_state.lang
    st.html(render.diagram_html(data, st.context.theme.type == "dark", lang, model=model_id))
    if data is None:
        st.html(render.INTRO[lang])
        return
    st.caption(render.run_stats(data, lang))
    # A fixed height makes it scroll; style.css shrinks it to the space left under the diagram.
    with st.container(height=BOX_HEIGHT, border=False, autoscroll=run.live, key="timeline"):
        steps = data["steps"]
        for step in steps:
            with st.container(border=True):
                live_step = run.live and step is steps[-1]
                st.html(render.step_html(step, data, lang, live=live_step))
                for path in render.images([step]):
                    st.image(path, width=320, caption=Path(path).name)
                if step["context"]:
                    st.button(tr("看這一步的完整脈絡 →", "See this step's full context →"), key=f"ctx-{index}-{step['n']}", type="tertiary",
                              on_click=show_context, args=(index, step["n"]))
        if data["end"]:
            st.html(render.stop_html(data, lang))


def context_panel(index: int | None, data: dict | None):
    lang = st.session_state.lang
    steps = [s for s in data["steps"] if s["context"]] if data else []
    if not steps:
        st.info(tr("送出問題後，這裡會顯示每一步模型實際讀到的完整字串。", "Once you send a question, this shows the whole string the model read in each step."))
        return
    key = f"ctx_step_{index}"
    if st.session_state.get(key) is None:
        st.session_state[key] = steps[-1]["n"]
    choice = st.segmented_control(
        tr("步驟", "Step"), [s["n"] for s in steps], key=key, required=True, format_func=lambda n: f"Step {n}"
    )
    step = next((s for s in steps if s["n"] == choice), steps[-1])
    st.caption(render.context_summary(step, lang))
    st.html(render.CONTEXT_LEGEND[lang] + render.context_html(step, data["family"], lang))


def model_panel(llm: Backend, data: dict | None):
    lang = st.session_state.lang
    st.html(render.model_html(llm.info, lang))
    settings = data["settings"] if data else {**DEFAULTS, **{k: st.session_state[k] for k in DEFAULTS}}
    st.html(render.knobs_html(settings, llm.info["defaults"], lang))
    if data and data["steps"]:
        st.markdown(tr("##### 每一步的推論", "##### Inference in each step"))
        st.dataframe(render.step_rows(data, lang), hide_index=True, width="stretch")
    offered = data["settings"].get("tools") if data else list(offered_tools())
    st.html(render.tools_html(TOOLS, settings["use_tools"], llm.family, lang, offered))
    st.html(render.special_tokens_html(llm.info, lang))
    st.html(render.folder_html(llm.info, lang))


# ---------- pages ----------

def agent_page():
    try:
        init_state()
        menu = model_menu()
        llm = current_llm()
    except vllm_backend.ServerError as e:  # the vLLM server is down or refused: say so, rather than show a traceback
        st.error(e.text[st.session_state.lang], icon=":material/cloud_off:")
        st.stop()
    own_live = any(run.live for run in st.session_state.runs)
    polling = own_live or other_tab_run() is not None
    sidebar(llm, menu, own_live, polling)

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


def pick_lang():
    st.session_state.lang_picked = True


def language_switch():
    """中文 | English, pinned to the top right over Streamlit's header by style.css. A page starts in the
    language last picked here, else in the browser's. Streamlit can read cookies but only the browser can
    write them, so a pick is remembered by a script that sets the cookie."""
    if "lang" not in st.session_state:
        picked = st.context.cookies.get(LANG_COOKIE)
        st.session_state.lang = picked if picked in LANGS else from_locale(st.context.locale)
    with st.container(key="lang_switch"):
        st.segmented_control("Language / 語言", list(LANGS), format_func=LANGS.get, key="lang", required=True,
                             label_visibility="collapsed", on_change=pick_lang)
        if st.session_state.get("lang_picked"):
            cookie = f"{LANG_COOKIE}={st.session_state.lang}; path=/; max-age=31536000; SameSite=Lax"  # a year
            st.html(f"<script>document.cookie = {cookie!r};</script>", unsafe_allow_javascript=True)


def main():
    st.set_page_config(layout="wide", page_icon=ASSETS / "logo.svg")
    st.logo(ASSETS / "logo.svg", size="large")
    st.html(ASSETS / "style.css")
    if st.context.theme.type == "dark":
        st.html(ASSETS / "dark.css")
    language_switch()
    lang = st.session_state.lang
    page = st.navigation(
        [
            st.Page(agent_page, title=pick(lang, "Agent Loop 可視化", "Agent Loop Visualizer"), icon=":material/autorenew:", default=True),
            # opens in a new tab, in the page's language
            st.Page(f"{GUIDE_URL}{lang}/", title=pick(lang, "運作說明", "How it works"), icon=":material/menu_book:", url_path="guide"),
        ],
        position="top",
    )
    page.run()


main()
