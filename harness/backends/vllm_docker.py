"""Starting a vLLM server in Docker, from docker-compose/compose.yaml, for the sidebar's vLLM (Docker).

Without VLLM_URL, the sidebar's "What runs the model" can still pick vLLM: the page then starts the server that
docker-compose/compose.yaml describes, with the model picked in the sidebar, waits until it answers, and runs the model through
VLLMBackend (vllm_backend.py) as on any other vLLM server. Picking transformers again stops it, and so does quitting
the app, so that only one of the two holds the model at a time; every tab follows what the app asked for (chosen()).

Docker Compose keeps track of the container: `docker compose up -d` with the model already running leaves it be,
with another model replaces it, and the first time pulls or builds the image. Where Docker has the nvidia runtime
(Linux with the NVIDIA Container Toolkit), compose.gpu.yaml goes on top and runs vLLM's CUDA image on the GPU;
elsewhere, a Mac say, compose.yaml runs vLLM's CPU image (vllm-cpu/). AGENT_LOOP_VLLM_DEVICE=cpu or gpu decides instead.

The server never downloads anything. The weights come into models/ here first, at the commit families.REVISIONS pins,
and the container reads that folder read-only, so it loads the very files the page renders the prompt with. As with
run_python's sandbox, the page runs `docker` only once vLLM is picked.
"""

import atexit
import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import httpx
from huggingface_hub import snapshot_download

from harness import families
from harness.backends.simple_backend import MODELS_DIR
from harness.tools import DOCKER_INSTALL

COMPOSE_DIR = Path(__file__).resolve().parents[2] / "docker-compose"  # compose.yaml, compose.gpu.yaml and vllm-cpu/
PROJECT = "agent-loop-vllm"  # the name in compose.yaml
SERVICE = "vllm"
COMPOSE = {"cpu": ["compose.yaml"], "gpu": ["compose.yaml", "compose.gpu.yaml"]}  # where the server runs -> its compose files
START_TIMEOUT = 20 * 60  # seconds for vLLM to load the model once its container runs; the first time it also compiles
POLL = 1.0  # seconds between two looks at a server that is loading
LOG_LINES = 15  # lines of output shown when something fails
# The message of an exception in vLLM's log, after the prefix of the process and the logger that wrote it.
EXCEPTION = re.compile(r"\b((?:\w+\.)*\w*(?:Error|Exception): .+)")

_lock = threading.Lock()  # one start or stop at a time, from every tab
_server: dict = {}  # the server this process started and saw answer: model, url, device
# The model this process last asked the server to run, from start() until stop(), whether it answered yet or not: what
# every tab follows, since they share the one model in memory.
_chosen: dict = {}


class DockerError(Exception):
    """Why the vLLM server could not be started, in both of the page's languages (Markdown)."""

    def __init__(self, zh: str, en: str):
        super().__init__(en)
        self.text = {"zh": zh, "en": en}


def _not_installed() -> DockerError:
    return DockerError(
        f"vLLM 要在 Docker 裡跑，但這台電腦還沒有安裝 Docker。請先[安裝 Docker]({DOCKER_INSTALL})，或改回 transformers。",
        f"vLLM runs in Docker, which is not installed on this computer. [Install Docker]({DOCKER_INSTALL}), or go back to transformers.",
    )


def _run(args: list[str], env: dict | None = None, timeout: float = 60, merge: bool = False) -> subprocess.CompletedProcess:
    """docker <args>, with what it prints captured; with `merge`, what it prints on stderr goes in stdout, in order."""
    try:
        return subprocess.run(["docker", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT if merge else subprocess.PIPE,
                              text=True, timeout=timeout, env=env)
    except FileNotFoundError as e:
        raise _not_installed() from e
    except subprocess.TimeoutExpired as e:
        raise DockerError("Docker 沒有回應，確認它已經啟動。", "Docker does not respond; make sure it is running.") from e


def _stream(args: list[str], env: dict, on_line: Callable[[str], None]) -> tuple[int, list[str]]:
    """docker <args>, handing on_line each line it prints as it prints it; then its exit code and its last lines.
    If on_line raises (the page reran while Docker was busy), Docker is stopped too, so that the next start does not
    run alongside it."""
    try:
        process = subprocess.Popen(["docker", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    except FileNotFoundError as e:
        raise _not_installed() from e
    tail: deque[str] = deque(maxlen=LOG_LINES)
    try:
        for line in process.stdout:
            if line := line.strip():
                tail.append(line)
                on_line(line)
        return process.wait(), list(tail)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(30)


def _compose(device: str, *args: str) -> list[str]:
    """The arguments of a `docker compose` command on the server, with the compose files for `device`."""
    files = [arg for name in COMPOSE[device] for arg in ("-f", str(COMPOSE_DIR / name))]
    return ["compose", "--progress", "plain", "-p", PROJECT, *files, *args]


def _env(model_id: str) -> dict:
    """What compose.yaml reads: the model, the commit to load it at, and the folder it was downloaded into."""
    return {**os.environ, "VLLM_MODEL": model_id, "VLLM_REVISION": families.REVISIONS.get(model_id, "main"),
            "AGENT_LOOP_MODELS": str(MODELS_DIR)}


def url() -> str:
    """Where the server listens: compose.yaml publishes it on this machine only, at VLLM_PORT."""
    return f"http://127.0.0.1:{os.environ.get('VLLM_PORT') or 8000}"


def device() -> str:
    """Where the server runs: "gpu" where Docker can hand a container NVIDIA GPUs (it has the nvidia runtime), else
    "cpu"; AGENT_LOOP_VLLM_DEVICE=cpu or gpu decides instead. Raises DockerError when Docker is missing or not running."""
    info = _run(["info", "--format", "{{json .Runtimes}}"])
    if info.returncode != 0:
        raise DockerError("Docker 已經安裝，但沒有在執行。啟動 Docker 之後再試一次。", "Docker is installed but not running. Start Docker and try again.")
    if (chosen := os.environ.get("AGENT_LOOP_VLLM_DEVICE")) in COMPOSE:
        return chosen
    return "gpu" if "nvidia" in json.loads(info.stdout or "{}") else "cpu"


def _serves(address: str, model_id: str) -> bool:
    """Whether the server at `address` answers and serves model_id."""
    try:
        response = httpx.get(f"{address}/v1/models", timeout=2)
        cards = response.json().get("data") or [] if response.status_code == 200 else []
    except (httpx.HTTPError, ValueError):
        return False
    return any(card.get("id") == model_id for card in cards)


def chosen() -> str | None:
    """The model this process asked the server to run and has not stopped, even while it starts or after it failed to:
    a tab that finds one runs the model there rather than loading it beside it. Runs no `docker`."""
    return _chosen.get("model")


def serving(model_id: str) -> dict | None:
    """The server this process started, {"model", "url", "device"}, if it still answers and serves model_id."""
    if _server.get("model") == model_id and _serves(_server["url"], model_id):
        return dict(_server)
    return None


def start(model_id: str, say: Callable[[str, str], None]) -> dict:
    """Start the server for model_id, or replace the one that serves another model, and wait until it answers; then
    {"model", "url", "device"}. say(stage, line) tells what is going on, with the last line of output: "download"
    (the weights into models/), "up" (Docker Compose pulling or building the image and starting the container) and
    "load" (vLLM loading the model)."""
    _chosen["model"] = model_id
    if os.path.isdir(model_id):
        raise DockerError(f"vLLM 的容器只讀得到 models/ 裡的模型，讀不到 {model_id} 這個資料夾；請用 Hugging Face 的 id。",
                          f"vLLM's container reads models from models/ only, not the folder {model_id}; use a Hugging Face id.")
    with _lock:
        if server := serving(model_id):  # started from another tab meanwhile
            return server
        _server.clear()
        where = device()
        if _run(["compose", "version"]).returncode != 0:
            raise DockerError("這台電腦的 Docker 沒有 Compose（`docker compose`）。請更新 Docker，或另外安裝 Docker Compose。",
                              "This computer's Docker has no Compose (`docker compose`). Update Docker, or install Docker Compose.")
        say("download", "")
        snapshot_download(model_id, revision=families.REVISIONS.get(model_id), cache_dir=MODELS_DIR)
        env = _env(model_id)
        code, tail = _stream(_compose(where, "up", "--detach"), env, lambda line: say("up", line))
        if code != 0:
            printed = "\n".join(tail)
            raise DockerError(f"`docker compose up` 失敗了（結束代碼 {code}）：\n\n```\n{printed}\n```",
                              f"`docker compose up` failed (exit code {code}):\n\n```\n{printed}\n```")
        container = _run(_compose(where, "ps", "--all", "--quiet", SERVICE), env).stdout.strip()
        address, deadline = url(), time.monotonic() + START_TIMEOUT
        while not _serves(address, model_id):
            state = json.loads(_run(["inspect", "--format", "{{json .State}}", container]).stdout or "{}")
            if not state.get("Running"):
                raise _failed(container, state, model_id)
            if time.monotonic() > deadline:
                raise DockerError(
                    f"vLLM 過了 {START_TIMEOUT // 60} 分鐘還沒載入好 {model_id}。用 `docker compose -p {PROJECT} logs` 看它在做什麼。",
                    f"vLLM has not loaded {model_id} after {START_TIMEOUT // 60} minutes. `docker compose -p {PROJECT} logs` shows what it is doing.")
            log = _run(["logs", "--tail", "1", container], merge=True).stdout.strip()
            say("load", log.splitlines()[-1] if log else "")
            time.sleep(POLL)
        _server.update(model=model_id, url=address, device=where)
        return dict(_server)


def _failed(container: str, state: dict, model_id: str) -> DockerError:
    """Why the server's container stopped while it loaded the model: out of memory, or the exceptions in its log."""
    if state.get("OOMKilled"):
        total = _run(["info", "--format", "{{.MemTotal}}"]).stdout.strip()
        gib = f"{int(total) / 2**30:.1f} GiB" if total.isdigit() else "?"
        return DockerError(
            f"vLLM 載入 {model_id} 時用完了 Docker 能用的記憶體（{gib}），容器被停掉了。給 Docker 更多記憶體"
            "（Docker Desktop：Settings → Resources），或換一個小一點的模型。",
            f"vLLM ran out of the memory Docker has ({gib}) while loading {model_id}, and its container was stopped. Give Docker more "
            "memory (Docker Desktop: Settings → Resources), or pick a smaller model.")
    log = _run(["logs", "--tail", "300", container], merge=True).stdout.splitlines()
    # The first exception is the cause; vLLM's processes then each fail in turn, and the last lines are only that.
    found = list(dict.fromkeys(m.group(1) for line in log if (m := EXCEPTION.search(line))))
    printed = "\n".join(found[:5] or log[-LOG_LINES:])
    code = state.get("ExitCode")
    return DockerError(
        f"vLLM 的伺服器沒有啟動成功（結束代碼 {code}）：\n\n```\n{printed}\n```\n\n完整的紀錄：`docker compose -p {PROJECT} logs`",
        f"vLLM's server did not start (exit code {code}):\n\n```\n{printed}\n```\n\nThe whole log: `docker compose -p {PROJECT} logs`")


def stop():
    """Stop and remove the server, if one runs, so that the memory it holds goes back to the machine. Without Docker
    there is nothing to stop."""
    with _lock:
        _server.clear()
        _chosen.clear()
        try:
            _run(_compose("cpu", "down"), timeout=120)
        except DockerError:
            pass


def _stop_at_exit():
    """When the app quits (Ctrl+C, or a SIGTERM, both of which Streamlit turns into a normal exit), the server it
    started goes too, rather than holding the model while the next start loads it with transformers. Without one,
    Docker is left alone."""
    if _chosen:
        stop()


atexit.register(_stop_at_exit)
