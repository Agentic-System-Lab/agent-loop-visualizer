"""Agent Loop 可視化 (Agent Loop Visualizer): chat with a local model (Gemma 4 E2B or Qwen3.5 0.8B, picked in the
sidebar) on the left and watch every step of its agent loop on the right.

Run with:  uv run streamlit run streamlit_app.py  (which calls main() here on every rerun)

This is the app's side of the project: the agent (agent/) runs the turn, and this page only starts it, draws
its events (render.py) and passes on the user's clicks. The settings read from the environment below are read once,
when the module is first imported.

The agent runs in a background thread (agent.harnesses.simple_harness.SimpleHarness) and appends one event per loop
stage to a list. While a run is live, a fragment redraws the page from that list
every half second. Clicking stop or approve reruns the script but not the thread,
so the loop keeps going.

The sidebar can also hand the loop to pi (agent/harnesses/pi_harness.py), a coding agent that runs its own, and the
page draws what pi says it does, with the request pi sends the model. pi asks either the page's own model, which
agent/harnesses/model_endpoint.py serves it on 127.0.0.1, so every token still shows, or a model it has credentials
for (Claude, GPT…), of which only the text shows, and then no model loads here.

With VLLM_URL set, a vLLM server runs the model instead of this machine (agent/model_backends/vllm_backend.py), and the
model menu lists what that server serves, marking those the page does not support; picking one of those says why
and offers a GitHub issue asking for it, already filled in. Without it, the sidebar can still pick vLLM: the page then starts a
vLLM server in Docker from docker-compose/compose.yaml (agent/model_backends/vllm_docker.py). Either way, the model in
memory is let go before the next one loads, so switching models or what runs them never holds two.

The top bar also links to 運作說明 (How it works), a walk through the loop with the code behind it:
a static page that app/guide.py builds and GitHub Pages serves.

The page is in Traditional Chinese or English (agent/i18n.py). It starts in the language last picked on the
switch at the top right, kept in a cookie, or else in the browser's language.

A question can come with files (agent/media.py): images, sound and video the model reads, and, when the sidebar
lets the model write code, data files for run_python. Only then does the page look for Docker, which
run_python's sandbox needs, and say how to install it when it is missing. The charts run_python draws are
shown in the chat and in the step that drew them.
"""

import gc
import json
import os
import shutil
import threading
import uuid
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlencode

import streamlit as st

from agent import media, model_families, tools
from agent.harnesses import model_endpoint, pi_harness
from agent.harnesses.harness import Harness
from agent.harnesses.simple_harness import RunSettings, SimpleHarness
from agent.i18n import LANGS, from_locale, pick
from agent.model_backends import vllm_backend, vllm_docker
from agent.model_backends.backend import Backend, Sampling
from agent.model_backends.simple_backend import SimpleBackend, empty_cache
from agent.tools import TOOLS
from app import render

# The model menu; MODEL_ID picks the one selected at first, and may name a model the menu does not list.
DEFAULT_MODEL = os.environ.get("MODEL_ID", next(iter(model_families.MODELS)))
MODELS = {**model_families.MODELS, DEFAULT_MODEL: model_families.MODELS.get(DEFAULT_MODEL, DEFAULT_MODEL)}
# Where a vLLM server listens (http://localhost:8000, say); with it set, the server runs the model, not this machine.
VLLM_URL = os.environ.get("VLLM_URL")
ASSETS = Path(__file__).parent / "assets"
SAMPLES = ASSETS / "examples"  # the files some examples attach, and the sample files offered above the input box
# Each conversation's files go in a folder of their own under here. It is under the home folder because that is
# the one every Docker setup (Docker Desktop, OrbStack, Colima) shares with its containers.
UPLOADS = Path(os.environ.get("AGENT_LOOP_UPLOADS", Path.home() / ".cache" / "agent-loop-visualizer" / "uploads"))
MAX_UPLOAD_MB = 50
GUIDE_URL = "https://cyyeh.github.io/agent-loop-visualizer/"  # built from app/guide.py; zh/ and en/ under it are the two languages
ISSUES_URL = "https://github.com/cyyeh/agent-loop-visualizer/issues"  # where a model the page does not support yet can be asked for
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
EXAMPLE_ICONS: dict[str | None, str] = {"image": ":material/image:", "audio": ":material/graphic_eq:", "video": ":material/movie:", "file": ":material/table_chart:", "python": ":material/code:"}
SAMPLE_FILES = sorted(p.name for p in SAMPLES.iterdir() if p.is_file() and not p.name.startswith("."))


def tr(zh: str, en: str) -> str:
    """The text in the language the page is in."""
    return pick(st.session_state.lang, zh, en)


@st.cache_data(ttl=60, show_spinner=False)
def model_menu() -> dict[str, str]:
    """The model menu, id -> label: MODELS, or with VLLM_URL the models the vLLM server serves, each by what it was
    loaded from, which is the Hugging Face id its tokenizer is downloaded by, and marked ⚠ when the page does not
    support it. The mark is the same in both languages, as every label in a selectbox must be."""
    if not VLLM_URL:
        return MODELS
    # A LoRA adapter (a card with a parent) is left out: its root is the adapter's folder on the server.
    served = (card.get("root") or card["id"] for card in vllm_backend.served_models(VLLM_URL) if not card.get("parent"))
    return {model_id: ("⚠ " if supported(model_id) is False else "") + model_families.MODELS.get(model_id, model_id) for model_id in served}


@st.cache_data(show_spinner=False)
def served_family(model_id: str) -> str | None:
    """The family of a model the vLLM server serves, from its config.json alone (vllm_backend.family_of), or None for
    one the page does not support. It is kept while the app runs, so the menu does not ask the Hub every minute; what
    could not be told raises, which is not kept, and is asked again the next time the menu is made."""
    family = vllm_backend.family_of(model_id)
    return family.key if family else None


def supported(model_id: str) -> bool | None:
    """Whether the page runs a model the menu lists; None when that cannot be told (no way to the Hub, or a folder
    only on the server, which then says why it does not load)."""
    if model_id in model_families.MODELS:
        return True
    try:
        return served_family(model_id) is not None
    except (OSError, ValueError):
        return None


def default_model() -> str:
    """The model a new tab starts on: MODEL_ID's when the menu has it, else the first the page supports, else the first."""
    menu = model_menu()
    if DEFAULT_MODEL in menu:
        return DEFAULT_MODEL
    return next((model_id for model_id in menu if supported(model_id)), next(iter(menu)))


@st.cache_resource(max_entries=1, show_spinner=False)  # one model in memory: switching models frees the last one
def load_llm(model_id: str, url: str | None) -> Backend:
    llm = vllm_backend.VLLMBackend(model_id, url) if url else SimpleBackend(model_id)
    llm.warmup()
    return llm


def release_model():
    """Let the model in memory go now, rather than once the next one has loaded beside it."""
    load_llm.clear()
    gc.collect()
    empty_cache()


def docker_vllm() -> bool:
    """Whether the model runs on the vLLM server the page starts in Docker: vLLM picked in the sidebar, no VLLM_URL."""
    return not VLLM_URL and st.session_state.engine == "vllm"


def vllm_server(model_id: str) -> dict:
    """The vLLM server in Docker for the model (vllm_docker.py), started first if it is not up, with what it is doing
    shown meanwhile, decoding speculatively if MTP is on and its GPU can. A start that failed is not tried again on every
    rerun, only when the user asks to (Try again)."""
    if failed := st.session_state.get("vllm_failed"):
        raise failed
    if server := vllm_docker.serving(model_id, st.session_state.mtp):
        st.session_state.vllm_device, st.session_state.mtp = server["device"], server["mtp"]
        return server
    stages = {
        "download": tr(f"下載 {model_id} 的權重到 models/（只有第一次）…", f"Downloading {model_id}'s weights into models/ (the first time only)…"),
        "up": tr("用 Docker Compose 啟動 vLLM 的容器（第一次要下載或建好 image，要幾分鐘）…",
                 "Starting vLLM's container with Docker Compose (the first time pulls or builds its image, which takes minutes)…"),
        "load": tr(f"vLLM 正在載入 {model_id}（第一次要編譯，在 CPU 上要幾分鐘）…", f"vLLM is loading {model_id} (the first time it compiles, which takes minutes on a CPU)…"),
    }
    with st.status(tr("啟動 vLLM 伺服器", "Starting the vLLM server"), expanded=True) as status:
        stage, line = st.empty(), st.empty()

        def say(what: str, last: str):
            stage.markdown(stages[what])
            if last:
                line.code(last, language=None, wrap_lines=True)
            else:
                line.empty()

        try:
            server = vllm_docker.start(model_id, say, st.session_state.mtp)
        except vllm_docker.DockerError as e:
            st.session_state.vllm_failed = e
            status.update(label=tr("vLLM 伺服器沒有啟動", "The vLLM server did not start"), state="error")
            raise
        status.update(label=tr("vLLM 伺服器啟動了", "The vLLM server is up"), state="complete", expanded=False)
    st.session_state.vllm_device, st.session_state.mtp = server["device"], server["mtp"]  # off on a CPU, which cannot
    return server


def current_llm() -> Backend:
    model_id = st.session_state.model_id
    if st.session_state.pop("stop_vllm", False):  # transformers was just picked again: vLLM's memory goes back first
        with st.spinner(tr("停止 vLLM 的容器…", "Stopping vLLM's container…")):
            vllm_docker.stop()
    st.session_state.pop("engine_picked", None)  # vLLM is stopped or about to start: from now on this tab follows it too
    url = vllm_server(model_id)["url"] if docker_vllm() else VLLM_URL
    if url:
        loading = tr(f"連線到 {url} 的 vLLM 伺服器，載入 {model_id} 的 tokenizer…", f"Connecting to the vLLM server at {url} and loading {model_id}'s tokenizer…")
    else:
        loading = tr(f"載入 {model_id}（第一次要下載權重；之後約 10–20 秒）…這時候切換的設定，載入完才會生效。",
                     f"Loading {model_id} (the first time downloads the weights; after that it takes about 10–20 s)… What you switch meanwhile takes effect once it has loaded.")
    with st.spinner(loading):
        llm = load_llm(model_id, url)
    if st.session_state.sampling_for != model_id:  # a model just chosen starts from its own recommended decoding, MTP off
        # but on vLLM in Docker, where MTP is how the server was just started
        for key, value in {**llm.info["defaults"], **({} if docker_vllm() else {"mtp": False})}.items():
            st.session_state[key] = value
        st.session_state.sampling_for = model_id
    if st.session_state.mtp and llm.drafter_id and llm.drafter is None:
        with st.spinner(tr(f"載入 MTP 草稿模型 {llm.drafter_id}（第一次要下載約 190 MB）…", f"Loading the MTP drafter {llm.drafter_id} (the first time downloads about 190 MB)…")):
            llm.load_drafter()
    return llm


# ---------- pi ----------

@dataclass
class PiAgent:
    """pi and the model it asks, where the page would otherwise have a Backend: what the chat needs to know of it. The
    model is the page's own (`llm`, served to pi by model_endpoint.py) or one of pi's own (`model`, at a provider)."""
    model_id: str  # the page's model id, or "provider/id", as pi's own model menu names it
    version: str
    llm: Backend | None = None
    model: dict = field(default_factory=dict)  # pi's model object for one of its own (pi_harness.probe)

    @property
    def modalities(self) -> tuple[str, ...]:
        """What pi sends the model besides text: images alone, and those only to a model that reads them."""
        reads = self.llm.modalities if self.llm else self.model.get("input", ())
        return ("image",) if "image" in reads else ()

    @property
    def info(self) -> dict:
        return self.llm.info if self.llm else {}  # a provider's model has no server or folder here


def pi_picked() -> bool:
    return st.session_state.get("harness") == "pi"


@st.cache_data(ttl=60, show_spinner=False)
def pi_found() -> dict:
    """What pi can run (pi_harness.probe), asked again at most every minute; a PiError is not kept, and is asked again."""
    return pi_harness.probe()


def retry_pi():
    pi_found.clear()


def pi_settings(who: PiAgent) -> pi_harness.PiSettings:
    s = st.session_state
    if who.llm:  # Thinking mode, which pi sends as its thinking level and the endpoint turns into the template's switch
        provider, model, thinking = model_endpoint.PROVIDER, who.llm.model_id, "medium" if s.thinking else "off"
    else:
        provider, model, thinking = who.model["provider"], who.model["id"], s.pi_thinking
    return pi_harness.PiSettings(provider, model, thinking=thinking, tools=tuple(s.pi_tools), approve_tools=s.approve_tools,
                                 max_steps=s.max_steps, lang=s.lang)


def pi_picker(busy: bool) -> PiAgent:
    """Which model pi asks: the page's own, with the model menu and what runs it as the built-in harness has them, or one
    of the models pi has credentials for. pi is started once to ask which (and again after a minute)."""
    with st.spinner(tr("問 pi 它能用哪些模型…", "Asking pi which models it can use…")):
        found = pi_found()
    with st.sidebar:
        sources = {"page": tr("這個網頁的模型", "This page's model"), "pi": tr("pi 自己的模型", "pi's own models")}
        st.segmented_control(
            tr("pi 問的模型", "The model pi asks"), list(sources), format_func=sources.__getitem__, key="pi_source", required=True,
            on_change=new_chat, disabled=busy,
            help=tr("這個網頁的模型：Gemma 或 Qwen，跑在 transformers 或 vLLM 上，由網頁在 127.0.0.1 上當成一個相容 OpenAI 的端點提供給 pi；"
                    "每個 token 和模型讀到的字串都看得到。pi 自己的模型：pi 登入過的 provider 的模型（Claude、GPT…），只看得到文字。",
                    "This page's model: Gemma or Qwen, on transformers or vLLM, which the page serves to pi as an OpenAI-compatible endpoint on 127.0.0.1; "
                    "every token and the string the model reads show. pi's own models: those of the providers pi is logged in to (Claude, GPT…); only their text shows."),
        )
    if st.session_state.pi_source == "page":
        model_picker(model_menu(), busy)
        return PiAgent(st.session_state.model_id, found["version"], llm=current_llm())
    models = {f'{m["provider"]}/{m["id"]}': m for m in found["models"]}
    if not models:
        raise pi_harness.PiError(
            "pi 還沒有可以用的模型。在終端機執行 `pi`，用 `/login` 登入一個 provider；或在啟動這個 app 之前設好 API key"
            "（例如 `ANTHROPIC_API_KEY`、`OPENAI_API_KEY`），或在 pi 的 `models.json` 加上一個相容的端點。然後按「再試一次」。",
            "pi has no model it can use yet. Run `pi` in a terminal and log in to a provider with `/login`, or set an API key "
            "(such as `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) before starting this app, or add a compatible endpoint to pi's `models.json`. Then press Try again.",
        )
    if st.session_state.get("pi_model") not in models:
        st.session_state.pi_model = found["default"] if found["default"] in models else next(iter(models))
    with st.sidebar:
        st.selectbox(tr("模型", "Model"), list(models), key="pi_model", on_change=new_chat, disabled=busy,
                     help=tr("pi 有登入資訊的模型（provider/模型 id）。換模型會開始新的對話。",
                             "The models pi has credentials for (provider/model id). Switching models starts a new conversation."))
    return PiAgent(st.session_state.pi_model, found["version"], model=models[st.session_state.pi_model])


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
    """One user turn: the harness thread and the events it has emitted so far. `make` makes the harness (simple_harness.SimpleHarness
    or pi_harness.PiHarness) given the function it sends its events to."""

    def __init__(self, make: Callable[[Callable[[dict], None]], Harness], text: str, attachments: list[media.Attachment], slot: Slot):
        self.text = text
        self.attachments = attachments
        self.events: list[dict] = []
        self.absorbed = False
        self.run = make(self.events.append)
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
    st.session_state.setdefault("harness", "simple")  # what runs the loop: this project's harness, or pi
    st.session_state.setdefault("pi_tools", list(pi_harness.DEFAULT_TOOLS))
    st.session_state.setdefault("pi_thinking", "off")
    st.session_state.setdefault("pi_source", "page")  # the model pi asks: the page's own, or one of pi's
    # The sidebar draws only one harness's settings; Streamlit would forget the other's, whose widgets are not drawn,
    # unless they are set again here.
    for key in [*DEFAULTS, "model_id", "pi_model", "pi_tools", "pi_thinking", "pi_source"]:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    st.session_state.setdefault("runs", [])
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("view", None)
    st.session_state.setdefault("viz_tab", "loop")
    on_vllm = None if VLLM_URL else vllm_docker.chosen()  # what vLLM's container runs for this app, and how, if it does
    if "model_id" not in st.session_state:  # asking the vLLM server what it serves only when there is no choice yet
        # A new tab or a reload joins the container rather than replacing it.
        st.session_state.model_id = on_vllm["model"] if on_vllm else default_model()
    st.session_state.setdefault("sampling_for", None)
    st.session_state.setdefault("chat_id", uuid.uuid4().hex[:12])
    # What runs the model without VLLM_URL: transformers here, or vLLM in Docker. It is one choice for every tab, as the
    # model in memory is one for every tab: a tab follows a container another started or stopped, rather than loading
    # the model here beside it, and decodes the way the server does, rather than starting it again the other way. Only
    # once this tab picked one or the other, until it has started or stopped vLLM, does it lead.
    if not st.session_state.get("engine_picked"):
        engine = "vllm" if on_vllm else "local"
        if engine != st.session_state.get("engine", "local"):
            st.session_state.mtp = False  # the drafter here is not loaded
            st.session_state.pop("vllm_failed", None)
        if on_vllm:
            st.session_state.mtp = on_vllm["mtp"]
        st.session_state.engine = engine


def chat_folder() -> Path:
    """This conversation's files: what the user attached, and what run_python sees as /data."""
    return UPLOADS / st.session_state.chat_id


def current_settings(llm: Backend) -> RunSettings:
    s = st.session_state
    return RunSettings(
        sampling=Sampling(s.temperature, s.top_p, s.top_k, s.max_new_tokens, mtp=s.mtp and (llm.drafter is not None or docker_vllm())),
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
    """Whether the sidebar lets the model write code: run_python, for which Docker is looked for only then, or with
    pi its bash tool."""
    if pi_picked():
        return "bash" in st.session_state.pi_tools
    return st.session_state.use_tools and st.session_state.python


def code_ready() -> bool:
    return code_allowed() and (pi_picked() or sandbox_problem() is None)


def offered_tools() -> tuple[str, ...]:
    """Every tool, but run_python only when the model may write code and Docker can run it."""
    return tuple(name for name in TOOLS if name != "run_python" or code_ready())


def start_run(who: "Backend | PiAgent", text: str, uploads: Sequence[tuple[str, bytes]] = ()) -> bool:
    """Ask the question in a new run, with `uploads` (file name, contents) saved in the conversation's folder first."""
    slot = generation_slot()
    if not slot.lock.acquire(blocking=False):
        st.toast(tr("另一個 agent 正在執行（可能在別的分頁），請稍等。", "Another agent is running (maybe in another tab). Please wait."))
        return False
    try:
        attachments = [media.save(chat_folder(), name, data, who.modalities) for name, data in uploads]
        run = LiveRun(harness_for(who, text, attachments), text, attachments, slot)
    except Exception:
        slot.lock.release()
        raise
    st.session_state.runs.append(run)
    st.session_state.view = len(st.session_state.runs) - 1
    return True


def harness_for(who: "Backend | PiAgent", text: str, attachments: list[media.Attachment]) -> Callable[[Callable[[dict], None]], Harness]:
    """What runs the turn: pi, or this project's harness on the model in memory, with the sidebar's settings now."""
    history, folder = st.session_state.history, chat_folder()
    if isinstance(who, PiAgent):
        settings, session = pi_settings(who), st.session_state.chat_id
        page = pi_harness.PageModel(who.llm, current_settings(who.llm).sampling) if who.llm else None
        return lambda emit: pi_harness.PiHarness(settings, history, text, emit, attachments, folder, session, page)
    run_settings = current_settings(who)
    return lambda emit: SimpleHarness(who, history, text, run_settings, emit, attachments, folder)


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
    if docker_vllm():  # where MTP is on unless switched off, and switching starts the server again
        st.session_state.mtp = True
        switch_mtp()


def new_chat():
    st.session_state.runs = []
    st.session_state.history = []
    st.session_state.view = None
    if "chat_id" in st.session_state:  # the old conversation's files go with it
        shutil.rmtree(chat_folder(), ignore_errors=True)
    st.session_state.chat_id = uuid.uuid4().hex[:12]


def switch_model():
    """Another model picked in the menu: a new conversation, and the model in memory let go before the next loads."""
    new_chat()
    release_model()
    st.session_state.pop("vllm_failed", None)


def switch_engine():
    """transformers or vLLM picked: the one that held the model lets it go before the other loads it, the weights
    here now and vLLM's container at the next run, under a spinner. MTP goes off with transformers, whose drafter
    loads only when it is switched on, and on with vLLM, whose server decodes speculatively where its GPU can."""
    switch_model()
    st.session_state.mtp = st.session_state.engine == "vllm"
    st.session_state.stop_vllm = st.session_state.engine == "local"
    st.session_state.engine_picked = True  # this run starts or stops vLLM, which every other tab then follows


def switch_mtp():
    """MTP switched: on vLLM in Docker that starts the server again, decoding the other way, which every other tab then
    follows."""
    if docker_vllm():
        st.session_state.engine_picked = True


def retry_vllm():
    st.session_state.pop("vllm_failed", None)


def show_context(run_index: int, step: int):
    st.session_state.viz_tab = "context"
    st.session_state[f"ctx_step_{run_index}"] = step


def show_run(run_index: int):
    st.session_state.view = run_index


# ---------- sidebar ----------

def harness_picker(busy: bool):
    """The very top of the sidebar: what runs the loop. It is drawn before anything can fail, so that the other can
    always be picked."""
    with st.sidebar:
        st.markdown(tr("### Agent Loop 可視化", "### Agent Loop Visualizer"))
        names = {"simple": tr("內建 harness", "Built-in harness"), "pi": "pi"}
        st.segmented_control(
            tr("用什麼跑迴圈", "What runs the loop"), list(names), format_func=names.__getitem__, key="harness", required=True,
            on_change=new_chat, disabled=busy,
            help=tr("內建 harness：這個專案自己寫的迴圈（`agent/harnesses/simple_harness.py`），模型在這台電腦或 vLLM 上跑，每個 token 都看得到。"
                    "pi：把迴圈交給 pi 這個 coding agent，網頁畫出它每一步做了什麼，和它送給模型的請求。pi 可以問這個網頁的模型，每個 token 都看得到；"
                    "或問你在 pi 裡設定好的模型（Claude、GPT…），只看得到文字。pi 的工具在這台電腦上執行，沒有沙箱。換過去會開始新的對話。",
                    "Built-in harness: this project's own loop (`agent/harnesses/simple_harness.py`), with the model on this computer or on vLLM, and every token in view. "
                    "pi: hands the loop to pi, a coding agent, and the page draws what it does at each step, with the request it sends the model. "
                    "pi can ask this page's model, with every token in view, or a model you set up in pi (Claude, GPT…), of which only the text shows. "
                    "pi's tools run on this computer, without a sandbox. Switching starts a new conversation."),
        )


def model_picker(menu: dict[str, str], busy: bool):
    """The top of the sidebar: the model, and, without VLLM_URL, what runs it. It is drawn before the model loads, so
    that while vLLM starts, or after it failed to, another model or transformers can still be picked."""
    with st.sidebar:
        about = tr("換模型會開始新的對話，解碼設定也換成新模型的建議值。harness、工具和畫面都不變。",
                   "Switching models starts a new conversation and resets the decoding settings to what the new model recommends. "
                   "The harness, the tools and the page stay the same.")
        if VLLM_URL:
            names = [f.name for f in model_families.FAMILIES.values()]
            about += tr(f"\n\n標 ⚠ 的模型，這個網頁還不支援它的家族（目前支援 {'、'.join(names)}）；選了會說明原因，也可以直接到 GitHub 提 issue。",
                        f"\n\nA model marked ⚠ is of a family this page does not support yet (it supports {', '.join(names)}); "
                        "picking it says why, and offers a GitHub issue asking for it.")
        st.selectbox(tr("模型", "Model"), list(menu), format_func=menu.__getitem__, key="model_id", on_change=switch_model, disabled=busy, help=about)
        if not VLLM_URL:
            # The same in both languages: with full-width parentheses the two options no longer fit side by side.
            engines = {"local": "transformers", "vllm": "vLLM (Docker)"}
            st.segmented_control(
                tr("用什麼跑模型", "What runs the model"), list(engines), format_func=engines.__getitem__, key="engine", required=True,
                on_change=switch_engine, disabled=busy,
                help=tr("transformers：模型載入這個程式，在這台電腦上跑，每個 token 的細節都看得到。"
                        "vLLM：用專案的 docker-compose/compose.yaml 在 Docker 裡啟動 vLLM 伺服器，有 NVIDIA GPU 就用 GPU，否則用 CPU；"
                        "第一次要建好 image、載入並編譯模型，要幾分鐘。在 GPU 上預設開 MTP 推測解碼。換過去時，原本那一邊的模型會先釋放，也會開始新的對話。",
                        "transformers: the model loads into this app and runs on this computer, and every detail of each token shows. "
                        "vLLM: a vLLM server in Docker, started from the project's docker-compose/compose.yaml, on an NVIDIA GPU if there is one, "
                        "else on the CPU; the first start builds its image and loads and compiles the model, which takes minutes. On a GPU, MTP speculative decoding is on by default. "
                        "Switching lets the other one's model go first, and starts a new conversation."),
            )


def model_caption(llm: Backend) -> str:
    """The model in memory and what runs it, under the model menu."""
    info = llm.info
    if server := info.get("server"):
        where = tr(f" · Docker（{st.session_state.vllm_device.upper()}）", f" · Docker ({st.session_state.vllm_device.upper()})") if docker_vllm() else ""
        where += " · MTP" if docker_vllm() and st.session_state.mtp else ""
        return f"`{llm.model_id}` · {llm.family.name} · {server['engine']} {server.get('version') or ''} · {server['url']}{where}"
    return (f"`{llm.model_id}` · {llm.family.name} · {info['device'].upper()} · {info['dtype']} · "
            + tr(f"載入 {info['load_seconds']} s", f"loaded in {info['load_seconds']} s"))


def sidebar(llm: Backend, live: bool, busy: bool):
    lang = st.session_state.lang
    with st.sidebar:
        st.caption(model_caption(llm))

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
    decoding_controls(llm, live, busy, tr("harness 的迴圈最多跑幾圈（step），跑滿就停。", "How many times (steps) the harness may go around the loop before it stops."))


def decoding_controls(llm: Backend, live: bool, busy: bool, steps_help: str):
    """The decoding settings of the model in memory and the loop's limits, for either harness that runs on it."""
    info = llm.info
    with st.sidebar:
        st.markdown(tr("##### 解碼與上限", "##### Decoding and limits"))
        if (info.get("server") or {}).get("refuses_greedy"):
            # vLLM refuses greedy decoding and an unlimited top-k once it was started with --return-sampling-mask, so the
            # sliders stop short of them, and a setting from before moves to the nearest one it takes.
            st.session_state.temperature = max(st.session_state.temperature, 0.05)
            st.session_state.top_k = st.session_state.top_k or 128
            st.slider("temperature", 0.05, 2.0, step=0.05, key="temperature",
                      help=tr("這台 vLLM 伺服器啟動時加了 `--return-sampling-mask`，所以不接受 greedy（temperature 0），最低是 0.05。",
                              "This vLLM server was started with `--return-sampling-mask`, so it refuses greedy decoding (temperature 0), and the lowest is 0.05."))
            st.slider("top-p", 0.05, 1.0, step=0.05, key="top_p")
            st.slider("top-k", 1, 128, key="top_k",
                      help=tr("這台 vLLM 伺服器啟動時加了 `--return-sampling-mask`，要靠 top-k 限制 sampling mask 的大小，所以不能設 0（不限）。",
                              "This vLLM server was started with `--return-sampling-mask` and needs top-k to bound its sampling masks, so 0 (no limit) is not allowed."))
        else:
            st.slider("temperature", 0.0, 2.0, step=0.05, key="temperature", help=tr("0 = greedy，永遠選機率最高的 token。", "0 = greedy: always the most likely token."))
            st.slider("top-p", 0.05, 1.0, step=0.05, key="top_p")
            st.slider("top-k", 0, 128, key="top_k", help=tr("0 = 不限。", "0 = no limit."))
        st.toggle(tr("MTP 推測解碼", "MTP speculative decoding"), key="mtp", on_change=switch_mtp, disabled=busy or not can_mtp(llm), help=mtp_help(llm))
        st.slider(tr("步數上限", "Max steps"), 1, 100, key="max_steps", help=steps_help)
        st.slider(tr("每步 token 上限", "Max tokens per step"), 64, 8192, step=64, key="max_new_tokens",
                  help=tr("一次推論最多生成幾個 token。", "The most tokens one inference may generate."))
        st.button(tr("還原成模型建議值", "Reset to the model's defaults"), on_click=reset_settings, args=(info["defaults"],), width="stretch")
        st.button(tr("新對話", "New chat"), on_click=new_chat, disabled=live, width="stretch")
        st.caption(tr("設定會套用到下一個問題。", "Settings apply to the next question."))


PI_STEPS_HELP = {"zh": "pi 自己沒有步數上限；迴圈跑滿這麼多步，網頁就請 pi 中止。",
                 "en": "pi has no step limit of its own; once the loop has gone around this many times, the page asks pi to abort."}


def pi_sidebar(who: PiAgent, live: bool, busy: bool):
    """pi's settings: its tools, how much its model thinks, asking first, and the step limit the page keeps for it; on
    the page's own model, its decoding settings too."""
    lang = st.session_state.lang
    with st.sidebar:
        if who.llm:
            st.caption(f"pi {who.version} · " + model_caption(who.llm))
        else:
            st.caption(f"pi {who.version} · `{who.model_id}` · " + tr("在 provider 那邊跑", "runs at the provider"))
        st.markdown(tr("##### 迴圈", "##### Loop"))
        st.pills(tr("pi 的工具", "pi's tools"), list(pi_harness.TOOLS), selection_mode="multi", key="pi_tools",
                 help=tr("模型看得到哪些 pi 的工具。全部取消就只是聊天機器人。", "Which of pi's tools the model sees. With none, it is just a chatbot."))
        st.caption(tr("pi 的工具在這台電腦上、這段對話的資料夾裡執行，沒有沙箱：bash 碰得到你碰得到的一切。",
                      "pi's tools run on this computer, in this conversation's folder, without a sandbox: bash reaches whatever you can."))
        if who.llm:
            st.toggle(tr("思考模式", "Thinking mode"), key="thinking",
                      help=who.llm.family.thinking_help[lang] + tr("（pi 把它當成思考等級送出，模型端點再換成 chat template 的思考開關。）",
                                                                    " (pi sends it as its thinking level, which the model endpoint turns into the chat template's thinking switch.)"))
        else:
            levels = pi_harness.THINKING if who.model.get("reasoning") else ("off",)
            if st.session_state.pi_thinking not in levels:
                st.session_state.pi_thinking = "off"
            st.segmented_control(tr("思考", "Thinking"), list(levels), key="pi_thinking", required=True,
                                 help=tr("pi 的思考等級：越高，模型回覆前想得越多。模型不會思考時只有 off。",
                                         "pi's thinking level: the higher, the more the model thinks before it replies. A model that does not think has only off."))
        st.toggle(tr("執行工具前先問我", "Ask me before running a tool"), key="approve_tools",
                  help=tr("pi 自己不問；網頁讓 pi 載入的 extension（`pi_extension.ts`）在每次執行工具前都先等你確認。",
                          "pi does not ask by itself; the extension the page loads into pi (`pi_extension.ts`) waits for you before every tool call."))
    if who.llm:
        decoding_controls(who.llm, live, busy, PI_STEPS_HELP[lang])
        return
    with st.sidebar:
        st.markdown(tr("##### 上限", "##### Limits"))
        st.slider(tr("步數上限", "Max steps"), 1, 100, key="max_steps", help=PI_STEPS_HELP[lang])
        st.button(tr("新對話", "New chat"), on_click=new_chat, disabled=live, width="stretch")
        st.caption(tr("設定會套用到下一個問題。", "Settings apply to the next question."))


def can_mtp(llm: Backend) -> bool:
    """Whether the MTP switch does anything: with transformers, for a model with a drafter; on vLLM in Docker, on a GPU
    and for a model vLLM drafts for. A server at VLLM_URL decides for itself."""
    if docker_vllm():
        return vllm_docker.speculative(llm.model_id, st.session_state.vllm_device) is not None
    return bool(llm.drafter_id)


def mtp_help(llm: Backend) -> str:
    if docker_vllm():
        if st.session_state.vllm_device != "gpu":
            return tr("vLLM 0.32 的 CPU 版 image 兩個模型都不能推測解碼：Qwen3.5 要用的 Triton kernel 沒有 CPU 版，Gemma 4 的草稿模型要建 CUDA graph。"
                      "所以在 CPU 上的 vLLM (Docker) 不開 MTP；在 NVIDIA GPU 上可以。",
                      "vLLM 0.32's CPU image cannot decode either model speculatively: Qwen3.5 needs a Triton kernel without a CPU version, and Gemma 4's "
                      "drafter sets up CUDA graphs. So vLLM (Docker) runs without MTP on a CPU; on an NVIDIA GPU it can.")
        if not (config := vllm_docker.speculative(llm.model_id, "gpu")):
            return tr("vLLM 推測解碼這個模型的方式還沒有設定（`model_families.VLLM_SPECULATIVE`）。",
                      "How vLLM decodes this model speculatively is not set up (`model_families.VLLM_SPECULATIVE`).")
        n, drafter = config["num_speculative_tokens"], config.get("model")
        how = (tr(f"用草稿模型 {drafter}", f"with the drafter {drafter}") if drafter
               else tr("用它自己 checkpoint 裡的 MTP 層（transformers 用不了）", "with the MTP layer in its own checkpoint, which transformers cannot use"))
        return tr(f"Docker 裡的 vLLM 伺服器在 GPU 上推測解碼，{how}：每次猜 {n} 個 token，主模型一次 forward 全部驗證。"
                  "step 結束時，收下的猜測會加上底線。切換會用另一種方式重新啟動伺服器，要幾分鐘。",
                  f"The vLLM server in Docker decodes speculatively on the GPU, {how}: it guesses {n} tokens at a time, and the model checks "
                  "them all in one forward pass. Once a step ends, the guesses it kept are underlined. Switching starts the server again the other way, which takes a few minutes.")
    if server := llm.info.get("server"):
        return tr(
            f"{server['engine']} 伺服器啟動時就決定要不要推測解碼（`--speculative-config`），這裡不能切換。伺服器加上 "
            "`--per-request-spec-decode-metrics summary` 的話，每一步會顯示它收下幾個猜測；設成 `detailed`，step 結束時猜中的 token 還會加上底線。",
            f"Whether the {server['engine']} server decodes speculatively is set when it starts (`--speculative-config`), not here. "
            "Started with `--per-request-spec-decode-metrics summary`, it says how many guesses it kept in each step; with `detailed`, "
            "the guesses are also underlined once the step ends.",
        )
    if not llm.drafter_id:
        on_vllm = (tr("在有 NVIDIA GPU 的 vLLM (Docker) 上，它可以推測解碼。", " On vLLM (Docker) with an NVIDIA GPU it can decode speculatively.")
                   if llm.model_id in model_families.VLLM_SPECULATIVE and not VLLM_URL else "")
        return tr("transformers 沒有這個模型搭配的 MTP 草稿模型，目前只有 Gemma 4 E2B 有。" + on_vllm,
                  "transformers has no MTP drafter for this model; for now only Gemma 4 E2B has one." + on_vllm)
    return tr(
        f"用 {llm.drafter_id} 當草稿模型：每生成一個 token，它就從主模型的狀態一口氣猜接下來幾個 token，"
        "主模型再用一次 forward 全部驗證。輸出的分布和不開一樣，只是比較快。第一次打開會載入草稿模型（約 160 MB）。",
        f"Uses {llm.drafter_id} as the drafter: after every token it guesses the next few at once from the model's state, "
        "and the model checks them all in one forward pass. The output follows the same distribution as without it, only faster. "
        "Switching it on the first time loads the drafter (about 160 MB).",
    )


# ---------- left: chat ----------

def chat_panel(llm: "Backend | PiAgent", live_run: LiveRun | None, other: LiveRun | None):
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


def sample_picker(row, samples: list[str], llm: "Backend | PiAgent", busy: bool):
    """The sample files from assets/examples that this model can use, to ask a question of one's own about: a file
    picked here goes with the next question typed in the input box, as one attached with the + would."""
    readable = llm.modalities  # not the model: Streamlit keeps format_func after the run, and with it the model

    def label(name: str) -> str:
        return f"{EXAMPLE_ICONS[media.kind_of(name, readable)]} {name}"

    with row:
        st.pills(tr("範例檔案", "Sample files"), samples, selection_mode="multi", format_func=label, key="samples", disabled=busy,
                 help=tr("選了的檔案會附在你接下來輸入的問題，和用 ＋ 附上的一樣。", "A file picked here goes with the next question you type, as if attached with the +."))


def uploaded(sent) -> list[tuple[str, bytes]]:
    """The files sent with a message, and a recording made with the microphone, as (file name, contents)."""
    files = [(f.name, f.getvalue()) for f in sent.get("files") or []]  # a key is missing when the input does not take its kind
    return files + ([("recording.wav", audio.getvalue())] if (audio := sent.get("audio")) else [])


def examples(llm: "Backend | PiAgent", busy: bool):
    """The page language's examples that this model can try: one that attaches a file shows only if the model reads
    that kind, and one that needs Python only once the model may write code, disabled until Docker is ready (with pi,
    once it has bash)."""
    lang, hidden = st.session_state.lang, False
    for i, (question, needs) in enumerate(EXAMPLES[lang]):
        sample = needs if needs not in (None, "python") else None
        kind = media.KINDS.get(Path(sample).suffix.lstrip("."), "file") if sample else needs
        if kind in media.MEDIA and kind not in llm.modalities:
            continue
        if kind in ("python", "file") and not code_allowed():
            hidden = True
            continue
        problem = sandbox_problem() if kind in ("python", "file") and not pi_picked() else None
        note = problem[lang] if problem else tr(f"會附上範例檔案 {sample}", f"Attaches the sample file {sample}") if sample else None
        if st.button(question, key=f"example-{i}", icon=EXAMPLE_ICONS.get(kind), help=note, disabled=busy or problem is not None):
            if start_run(llm, question, [(sample, (SAMPLES / sample).read_bytes())] if sample else []):
                st.rerun()
    if hidden and pi_picked():
        st.caption(tr("在側欄給 pi 它的 bash 工具，還有要寫 Python 的範例。", "Give pi its bash tool in the sidebar for examples that need Python."))
    elif hidden:
        st.caption(tr("打開側欄的「讓模型寫程式」，還有要寫 Python 的範例。", "Turn on Let the model write code in the sidebar for examples that need Python."))


def attachments_view(attachments: list[media.Attachment]):
    for a in attachments:
        if a.kind == "image":
            st.image(a.path, width=240)
        elif a.kind == "audio":
            st.audio(a.path, format=media.mime(a.name) or "audio/wav")
        elif a.kind == "video" and (mime := media.mime(a.name)):
            st.video(a.path, format=mime)
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
    if error := render.error_text(data, lang):
        st.error(error)
    if data["end"]:
        if not text:
            st.caption(tr("（這一輪沒有最終回覆，看右邊的停止原因。）", "(No final reply in this turn; the right side says why it stopped.)"))
        foot, button = st.columns([3, 1], vertical_alignment="center")
        foot.caption(render.footer(data, lang))
        if st.session_state.view != i:
            button.button(tr("看流程", "Show the loop"), key=f"view-{i}", type="tertiary", on_click=show_run, args=(i,))


# ---------- right: the loop ----------

def viz_panel(llm: "Backend | PiAgent"):
    runs = st.session_state.runs
    index = st.session_state.view
    run = runs[index] if runs and index is not None else None
    data = run.data() if run else None

    # The switcher keeps its own width; in a narrow column the caption wraps under it.
    with st.container(horizontal=True, vertical_alignment="center"):
        names = {"loop": tr("迴圈", "Loop"), "context": tr("脈絡", "Context"), "model": tr("模型", "Model")}
        view = st.segmented_control(tr("檢視", "View"), list(names), format_func=names.__getitem__, key="viz_tab", required=True, label_visibility="collapsed")
        if run and len(runs) > 1:
            st.caption(tr(f"第 {index + 1} 輪：{run.text[:40]}", f"Turn {index + 1}: {run.text[:40]}"))

    # Each tab scrolls in its own box, so a scroll position never carries over to another tab.
    with st.container(height=BOX_HEIGHT, border=False, key=f"{view}_view"):
        if view == "context":
            context_panel(index, data)
        elif view == "model":
            if isinstance(llm, PiAgent):
                pi_model_panel(llm, data)
            else:
                model_panel(llm, data)
        else:
            loop_panel(index, run, data, llm.model_id)


def loop_panel(index: int | None, run: LiveRun | None, data: dict | None, model_id: str):
    lang, harness = st.session_state.lang, st.session_state.harness
    st.html(render.diagram_html(data, st.context.theme.type == "dark", lang, model=model_id, harness=harness))
    if data is None or run is None:
        st.html(render.PI_INTRO[st.session_state.pi_source][lang] if harness == "pi" else render.INTRO[lang])
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
                if step["context"] or step["request"]:
                    st.button(tr("看這一步的完整脈絡 →", "See this step's full context →"), key=f"ctx-{index}-{step['n']}", type="tertiary",
                              on_click=show_context, args=(index, step["n"]))
        if data["end"]:
            st.html(render.stop_html(data, lang))


def context_panel(index: int | None, data: dict | None):
    lang = st.session_state.lang
    steps = [s for s in data["steps"] if s["context"] or s["request"]] if data else []
    if not steps and pi_picked():
        st.info(tr("送出問題後，這裡會顯示 pi 每一步送給模型的請求；pi 問的是這個網頁的模型時，還有模型實際讀到的完整字串。",
                   "Once you send a question, this shows the request pi sends the model in each step, and, when pi asks this page's model, "
                   "the whole string the model read."))
        return
    if not steps or data is None:
        st.info(tr("送出問題後，這裡會顯示每一步模型實際讀到的完整字串。", "Once you send a question, this shows the whole string the model read in each step."))
        return
    key = f"ctx_step_{index}"
    if st.session_state.get(key) is None:
        st.session_state[key] = steps[-1]["n"]
    choice = st.segmented_control(
        tr("步驟", "Step"), [s["n"] for s in steps], key=key, required=True, format_func=lambda n: f"Step {n}"
    )
    step = next((s for s in steps if s["n"] == choice), steps[-1])
    if step["context"]:
        st.caption(render.context_summary(step, lang))
        st.html(render.CONTEXT_LEGEND[lang] + render.context_html(step, data["family"], lang))
    if step["request"] and step["context"]:  # pi asking the page's model: what pi sent, before the template wrote it out
        with st.expander(tr("pi 送出的請求（JSON），chat template 把它寫成上面那個字串", "The request pi sent (JSON), which the chat template wrote out as the string above")):
            st.caption(render.request_summary(step, lang))
            st.html(render.REQUEST_LEGEND[lang] + render.request_html(step))
    elif step["request"]:
        st.caption(render.request_summary(step, lang))
        st.html(render.REQUEST_LEGEND[lang] + render.request_html(step))
        st.caption(tr("API 那邊怎麼把這個請求寫成模型讀的字串，pi 和 provider 都不說。",
                      "How the API writes this request out as the string the model reads, neither pi nor the provider says."))


def model_panel(llm: Backend, data: dict | None):
    lang = st.session_state.lang
    st.html(render.model_html(llm.info, lang))
    settings = data["settings"] if data else {**DEFAULTS, **{k: st.session_state[k] for k in DEFAULTS}}
    if VLLM_URL:  # whether it drafts is the server's, which says so only in a step's counts
        settings = {**settings, "mtp": None}
    st.html(render.knobs_html(settings, llm.info["defaults"], lang))
    if data and data["steps"]:
        st.markdown(tr("##### 每一步的推論", "##### Inference in each step"))
        st.dataframe(render.step_rows(data, lang), hide_index=True, width="stretch")
    offered = data["settings"].get("tools") if data else list(offered_tools())
    st.html(render.tools_html(TOOLS, settings["use_tools"], llm.family, lang, offered))
    st.html(render.special_tokens_html(llm.info, lang))
    st.html(render.folder_html(llm.info, lang))


def pi_model_panel(who: PiAgent, data: dict | None):
    lang, s = st.session_state.lang, st.session_state
    if who.llm:  # the page's own model, as the built-in harness shows it, with what pi does with it
        llm = who.llm
        st.html(render.model_html(llm.info, lang))
        settings = data["settings"] if data else {**DEFAULTS, **{k: s[k] for k in DEFAULTS}, "tools": s.pi_tools, "page_model": True,
                                                  "thinking": "medium" if s.thinking else "off"}
        st.html(render.knobs_html({**settings, "mtp": None} if VLLM_URL else settings, llm.info["defaults"], lang))
    else:
        st.html(render.pi_model_html(who.model, who.version, lang))
        settings = data["settings"] if data else {"thinking": s.pi_thinking, "approve_tools": s.approve_tools, "max_steps": s.max_steps, "tools": s.pi_tools}
    st.html(render.pi_settings_html(settings, lang))
    if data and data["steps"]:
        st.markdown(tr("##### 每一步的推論", "##### Inference in each step") if who.llm else tr("##### 每一步的請求", "##### The request in each step"))
        st.dataframe(render.step_rows(data, lang), hide_index=True, width="stretch")
    st.html(render.pi_tools_html(pi_harness.TOOLS, settings["tools"], lang))
    if who.llm:
        st.html(render.special_tokens_html(who.llm.info, lang))
        st.html(render.folder_html(who.llm.info, lang))


def issue_url(error: vllm_backend.UnsupportedModel) -> str:
    """A new GitHub issue asking the page to support a model, its title and description filled in in the page's
    language. It names the model and its model_type, never VLLM_URL, which may be a private address, and a checkpoint
    in a folder by the folder's name alone."""
    on_hub = not os.path.isdir(error.model_id)
    name = error.model_id if on_hub else Path(error.model_id).name
    kind = error.model_type
    title = tr(f"支援 {name}" + (f"（{kind}）" if kind else ""), f"Support {name}" + (f" ({kind})" if kind else ""))
    where = tr(f"（https://huggingface.co/{name}）" if on_hub else "（這台電腦上的資料夾）", f" (https://huggingface.co/{name})" if on_hub else " (a folder on my machine)")
    body = tr(
        "想在 Agent Loop 可視化裡跑這個模型，網頁說它還不支援。\n\n"
        f"- 模型：`{name}`{where}\n"
        f"- config.json 的 model_type：" + (f"`{kind}`" if kind else "（沒有 config.json，或讀不出來）") + "\n"
        "- 跑在：vLLM 伺服器（`VLLM_URL`）\n\n"
        "## 想用它做什麼\n\n"
        "（例如：想拿它的 agent loop 和網頁已經支援的模型比較。）\n",
        "I would like to run this model in the Agent Loop Visualizer, and the page says it does not support it yet.\n\n"
        f"- Model: `{name}`{where}\n"
        f"- model_type in its config.json: " + (f"`{kind}`" if kind else "(no config.json, or it could not be read)") + "\n"
        "- Runs on: a vLLM server (`VLLM_URL`)\n\n"
        "## What I would like to use it for\n\n"
        "(Say, to compare its agent loop with those of the models the page already supports.)\n",
    )
    return f"{ISSUES_URL}/new?{urlencode({'title': title, 'body': body}, quote_via=quote)}"


def ask_for_model(error: vllm_backend.UnsupportedModel):
    """Under the error for a model the page does not support: an invitation to ask for it on GitHub, with the issue
    filled in for the person to read and change before sending it, and a search for one someone has opened already."""
    search = f"{ISSUES_URL}?{urlencode({'q': f'is:issue {error.model_type or Path(error.model_id).name}'}, quote_via=quote)}"
    st.markdown(tr(f"想讓網頁支援這個模型嗎？歡迎到 GitHub 提 issue：按下面的按鈕，標題和內容都已經填好，送出前可以再改。"
                   f"也可以先[看看有沒有人提過]({search})。",
                   f"Would you like the page to support this model? You are welcome to open an issue on GitHub: the button below fills in "
                   f"its title and description, which you can change before you submit it. You can also [see whether someone has asked already]({search})."))
    st.link_button(tr("到 GitHub 提 issue", "Open an issue on GitHub"), issue_url(error), icon=":material/add_comment:")


# ---------- pages ----------

def agent_page():
    try:
        init_state()
        own_live = any(run.live for run in st.session_state.runs)
        polling = own_live or other_tab_run() is not None
        harness_picker(polling)
        if pi_picked():
            llm: Backend | PiAgent = pi_picker(polling)
        else:
            model_picker(model_menu(), polling)
            llm = current_llm()
    except (vllm_backend.ServerError, vllm_docker.DockerError, pi_harness.PiError) as e:  # vLLM is down, refused or did not start, or pi
        unsupported = isinstance(e, vllm_backend.UnsupportedModel)
        st.error(e.text[st.session_state.lang], icon=":material/extension_off:" if unsupported else ":material/cloud_off:")
        if isinstance(e, vllm_docker.DockerError):  # the sidebar still has the menu and the switch back to transformers
            st.button(tr("再試一次", "Try again"), icon=":material/refresh:", on_click=retry_vllm)
        if isinstance(e, pi_harness.PiError):  # and the switch back to the built-in harness
            st.button(tr("再試一次", "Try again"), icon=":material/refresh:", on_click=retry_pi)
        if isinstance(e, vllm_backend.UnsupportedModel):  # and the menu has the models the page does support
            ask_for_model(e)
        st.stop()
    model: Callable[[], Backend | PiAgent | None]  # what the workspace draws, None once it has been let go
    if isinstance(llm, PiAgent):
        pi_sidebar(llm, own_live, polling)
        if llm.llm:  # the page's own model, held weakly as below
            backend, model_id, version = weakref.ref(llm.llm), llm.model_id, llm.version
            model = lambda: PiAgent(model_id, version, llm=b) if (b := backend()) else None
        else:  # pi's own model: nothing in memory here to let go
            model = lambda: llm
    else:
        sidebar(llm, own_live, polling)
        # Streamlit keeps the fragment after the run, to run it again on its own; held weakly, the model can still be let go
        # (release_model()) the moment another is picked, rather than once the next run has replaced the fragment.
        model = weakref.ref(llm)

    @st.fragment(run_every=POLL_SECONDS if polling else None)
    def workspace():
        if (llm := model()) is None:  # let go meanwhile: the full rerun that follows draws the next one
            st.rerun()
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
        st.segmented_control("Language / 語言", list(LANGS), format_func=LANGS.__getitem__, key="lang", required=True,
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
