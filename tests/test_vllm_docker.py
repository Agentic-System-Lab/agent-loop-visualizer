"""Starting vLLM in Docker for the page (harness/backends/vllm_docker.py), and the compose files it starts it from.

Docker is faked at the level of subprocess: each `docker` command the module runs is recorded and answered as Docker
would, so every way a start can go (no Docker, Docker not running, no Compose, `docker compose up` failing, a container
killed for memory or exiting with an exception, a server that never answers) is checked without Docker. The compose
files are read as YAML, and, where Docker Compose is installed, by Compose itself.

With RUN_VLLM_DOCKER_TESTS=1 the last test starts the real server with Qwen3.5 0.8B (on the CPU, or the GPU where
Docker has one), asks it a question through VLLMBackend and stops it:
    RUN_VLLM_DOCKER_TESTS=1 uv run pytest tests/test_vllm_docker.py -k real
"""

import atexit
import json
import os
import re
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from harness import families
from harness.backends import simple_backend, vllm_docker
from harness.backends.backend import Sampling
from harness.backends.simple_backend import MODELS_DIR
from harness.tools import DOCKER_INSTALL

COMPOSE_DIR = vllm_docker.COMPOSE_DIR  # docker-compose/
QWEN = "Qwen/Qwen3.5-0.8B"
GEMMA = "google/gemma-4-E2B-it"
CONTAINER = "0123456789ab"


class FakeDocker:
    """What `docker` answers. Tests change its fields before starting the server: `installed`, `daemon` (running),
    `compose` (the Compose plugin), `runtimes`, `up` (exit code and lines of `docker compose up`), `state` (the
    container's, once up), `log`, and `answers_after` (how many looks before the server answers)."""

    def __init__(self):
        self.installed, self.daemon, self.compose = True, True, True
        self.runtimes = {"io.containerd.runc.v2": {}, "runc": {}}
        self.up = (0, [" Container agent-loop-vllm-vllm-1  Started"])
        self.state = {"Running": True, "OOMKilled": False, "ExitCode": 0}
        self.log = "(APIServer pid=1) INFO 10-25 14:28:31 [model.py:2152] Using max model len 131072\n"
        self.answers_after = 2
        self.commands: list[list[str]] = []
        self.envs: list[dict | None] = []
        self.served: str | None = None  # the model the last `up` started
        self.looks = 0
        self.terminated = False

    def run(self, cmd, stdout=None, stderr=None, text=None, timeout=None, env=None, capture_output=None):
        return subprocess.CompletedProcess(cmd, *self._answer(cmd, env))

    def _answer(self, cmd, env) -> tuple[int, str, str]:
        if not self.installed:
            raise FileNotFoundError(2, "No such file or directory", "docker")
        self.commands.append(cmd)
        self.envs.append(env)
        args = cmd[1:]
        if args[0] == "info":
            if not self.daemon:
                return 1, "", "Cannot connect to the Docker daemon at unix:///var/run/docker.sock."
            return 0, (json.dumps(self.runtimes) if "{{json .Runtimes}}" in args else "25232723968") + "\n", ""
        if args[:2] == ["compose", "version"]:
            return (0, "Docker Compose version v5.5.0\n", "") if self.compose else (1, "", "docker: 'compose' is not a docker command.")
        if args[0] == "compose" and "ps" in args:
            return 0, CONTAINER + "\n", ""
        if args[0] == "compose" and "down" in args:
            self.served = None
            return 0, "", ""
        if args[0] == "inspect":
            return 0, json.dumps(self.state), ""
        if args[0] == "logs":
            lines = self.log.splitlines()
            return 0, "\n".join(lines[-int(args[2]):]) + "\n", ""
        raise AssertionError(f"unexpected docker command {cmd}")

    def popen(self, cmd, stdout=None, stderr=None, text=None, env=None):
        if not self.installed:
            raise FileNotFoundError(2, "No such file or directory", "docker")
        self.commands.append(cmd)
        self.envs.append(env)
        assert cmd[1] == "compose" and "up" in cmd, cmd
        code, lines = self.up
        if code == 0:
            self.served = env["VLLM_MODEL"]
        fake = self

        class Process:
            stdout = iter(line + "\n" for line in lines)

            def wait(self, timeout=None):
                return code

            def poll(self):
                return None if not fake.terminated and lines and code is None else code

            def terminate(self):
                fake.terminated = True

        return Process()

    def serves(self, address: str, model_id: str) -> bool:
        self.looks += 1
        return self.state.get("Running") and self.served == model_id and self.looks > self.answers_after

    def compose_calls(self, what: str) -> list[list[str]]:
        return [c for c in self.commands if c[1] == "compose" and what in c]

    def models_up(self) -> list[str]:
        """The model of each `docker compose up`, in order."""
        return [env["VLLM_MODEL"] for c, env in zip(self.commands, self.envs) if c[1] == "compose" and "up" in c]


@pytest.fixture
def docker(monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr(vllm_docker.subprocess, "run", fake.run)
    monkeypatch.setattr(vllm_docker.subprocess, "Popen", fake.popen)
    monkeypatch.setattr(vllm_docker, "_serves", fake.serves)
    monkeypatch.setattr(vllm_docker, "POLL", 0)
    monkeypatch.delenv("AGENT_LOOP_VLLM_DEVICE", raising=False)
    monkeypatch.delenv("VLLM_PORT", raising=False)
    downloads = []
    monkeypatch.setattr(vllm_docker, "snapshot_download", lambda repo, **kw: downloads.append((repo, kw)))
    fake.downloads = downloads
    # The fake starts nothing, so the app's exit must not stop a real server: nothing is left chosen after a test.
    vllm_docker._server.clear()
    vllm_docker._chosen.clear()
    yield fake
    vllm_docker._server.clear()
    vllm_docker._chosen.clear()


def start(model_id=QWEN):
    said = []
    server = vllm_docker.start(model_id, lambda stage, line: said.append((stage, line)))
    return server, said


def failure(model_id=QWEN) -> vllm_docker.DockerError:
    with pytest.raises(vllm_docker.DockerError) as e:
        start(model_id)
    return e.value


# ---------- starting the server ----------

def test_the_server_starts_from_compose_yaml_with_the_model_at_its_pinned_revision(docker):
    server, said = start()
    assert server == {"model": QWEN, "url": "http://127.0.0.1:8000", "device": "cpu"}
    assert docker.downloads == [(QWEN, {"revision": families.REVISIONS[QWEN], "cache_dir": MODELS_DIR})]  # the weights, before the server
    (up,) = docker.compose_calls("up")
    assert up == ["docker", "compose", "--progress", "plain", "-p", "agent-loop-vllm", "-f", str(COMPOSE_DIR / "compose.yaml"), "up", "--detach"]
    env = docker.envs[docker.commands.index(up)]
    assert docker.models_up() == [QWEN]
    assert (env["VLLM_MODEL"], env["VLLM_REVISION"], env["AGENT_LOOP_MODELS"]) == (QWEN, families.REVISIONS[QWEN], str(MODELS_DIR))
    assert [stage for stage, _ in said] == ["download", "up", "load", "load"]
    assert said[1] == ("up", "Container agent-loop-vllm-vllm-1  Started") and "Using max model len" in said[2][1]


def test_a_machine_whose_docker_has_the_nvidia_runtime_runs_the_server_on_the_gpu(docker):
    docker.runtimes["nvidia"] = {"path": "nvidia-container-runtime"}
    server, _ = start(GEMMA)
    assert server["device"] == "gpu"
    (up,) = docker.compose_calls("up")
    assert up[up.index("-f") + 1:up.index("up")] == [str(COMPOSE_DIR / "compose.yaml"), "-f", str(COMPOSE_DIR / "compose.gpu.yaml")]


@pytest.mark.parametrize(("runtimes", "chosen"), [({}, "gpu"), ({"nvidia": {}}, "cpu")])
def test_agent_loop_vllm_device_decides_where_the_server_runs(docker, monkeypatch, runtimes, chosen):
    docker.runtimes.update(runtimes)
    monkeypatch.setenv("AGENT_LOOP_VLLM_DEVICE", chosen)
    assert start()[0]["device"] == chosen


def test_vllm_port_moves_where_the_page_looks_for_the_server(docker, monkeypatch):
    monkeypatch.setenv("VLLM_PORT", "8123")
    assert start()[0]["url"] == "http://127.0.0.1:8123"


def test_a_server_that_already_serves_the_model_is_not_started_again(docker):
    first, _ = start()
    assert vllm_docker.serving(QWEN) == first and vllm_docker.serving(GEMMA) is None
    again, said = start()
    assert again == first and said == [] and len(docker.compose_calls("up")) == 1


def test_another_model_goes_through_compose_up_again_which_replaces_the_container(docker):
    start()
    docker.looks = 0
    server, _ = start(GEMMA)
    assert server["model"] == GEMMA and docker.served == GEMMA
    assert docker.models_up() == [QWEN, GEMMA]


def test_a_checkpoint_folder_cannot_be_served_from_docker(docker, tmp_path):
    error = failure(str(tmp_path))
    assert "models/" in error.text["en"] and "models/" in error.text["zh"] and docker.commands == []


# ---------- what keeps it from starting, in both languages ----------

def test_without_docker_the_page_says_how_to_install_it(docker):
    docker.installed = False
    error = failure()
    assert DOCKER_INSTALL in error.text["zh"] and DOCKER_INSTALL in error.text["en"] and "transformers" in error.text["en"]


def test_docker_that_is_not_running_is_said(docker):
    docker.daemon = False
    error = failure()
    assert error.text == {"zh": "Docker 已經安裝，但沒有在執行。啟動 Docker 之後再試一次。", "en": "Docker is installed but not running. Start Docker and try again."}


def test_docker_without_compose_is_said(docker):
    docker.compose = False
    error = failure()
    assert "docker compose" in error.text["en"] and "Compose" in error.text["zh"] and docker.downloads == []


def test_compose_up_failing_shows_what_it_printed(docker):
    docker.up = (1, ["Error response from daemon: Bind for 127.0.0.1:8000 failed: port is already allocated"])
    error = failure()
    assert "exit code 1" in error.text["en"] and "結束代碼 1" in error.text["zh"]
    assert all("port is already allocated" in text for text in error.text.values())


def test_a_container_killed_for_memory_says_to_give_docker_more_memory(docker):
    docker.state = {"Running": False, "OOMKilled": True, "ExitCode": 137}
    error = failure(GEMMA)
    assert "23.5 GiB" in error.text["en"] and "Settings → Resources" in error.text["en"] and GEMMA in error.text["en"]
    assert "23.5 GiB" in error.text["zh"] and "記憶體" in error.text["zh"]


def test_a_server_that_exits_shows_the_exceptions_in_its_log_cause_first(docker):
    docker.state = {"Running": False, "OOMKilled": False, "ExitCode": 1}
    docker.log = "\n".join([
        "(APIServer pid=1) INFO 10-25 13:49:09 [core.py:133] Initializing a V1 LLM engine (v0.32.0)",
        "(VllmWorker-0 pid=112) ERROR 10-25 13:49:15 [multiproc_executor.py:953] ValueError: Available memory on node 0 (10.38/15.66 GiB) on startup is less than desired",
        "(EngineCore pid=74) ERROR 10-25 13:49:16 [core.py:1510] Exception: WorkerProc initialization failed due to an exception in a background process.",
        "Traceback (most recent call last):",
        "RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}",
        "RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}",
    ])
    shown = re.search(r"```\n(.*)\n```", failure().text["en"], re.S).group(1).splitlines()
    assert shown == ["ValueError: Available memory on node 0 (10.38/15.66 GiB) on startup is less than desired",
                     "Exception: WorkerProc initialization failed due to an exception in a background process.",
                     "RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}"]


def test_a_server_that_never_answers_gives_up_after_the_time_allowed(docker, monkeypatch):
    monkeypatch.setattr(vllm_docker, "START_TIMEOUT", 0)
    docker.answers_after = 10**9
    error = failure()
    assert "docker compose -p agent-loop-vllm logs" in error.text["en"] and "docker compose -p agent-loop-vllm logs" in error.text["zh"]


def test_a_rerun_while_compose_is_busy_stops_compose_too(docker):
    docker.up = (None, ["vllm Pulling", "vllm Downloading"])  # still running when the page reran

    def rerun(stage, line):
        if stage == "up":
            raise KeyboardInterrupt  # what Streamlit's rerun raises in the script thread is no Exception either

    with pytest.raises(KeyboardInterrupt):
        vllm_docker.start(QWEN, rerun)
    assert docker.terminated


# ---------- stopping it ----------

def test_stopping_runs_compose_down_and_forgets_the_server(docker):
    start()
    vllm_docker.stop()
    assert docker.compose_calls("down") and vllm_docker.serving(QWEN) is None and docker.served is None


def test_the_model_the_app_asked_vllm_for_is_kept_through_a_failed_start_until_it_is_stopped(docker):
    assert vllm_docker.chosen() is None
    docker.up = (1, ["Error response from daemon: pull access denied"])
    failure()
    assert vllm_docker.chosen() == QWEN  # every tab still says vLLM, and why it did not start
    docker.up = (0, [])
    start(GEMMA)
    assert vllm_docker.chosen() == GEMMA
    vllm_docker.stop()
    assert vllm_docker.chosen() is None


def test_quitting_the_app_stops_the_server_it_started_and_without_one_leaves_docker_alone(docker, monkeypatch):
    registered = []
    monkeypatch.setattr(atexit, "register", registered.append)
    module = runpy.run_path(vllm_docker.__file__)  # the hook is registered at import, apart from the imported module
    assert registered == [module["_stop_at_exit"]]
    vllm_docker._stop_at_exit()
    assert docker.commands == []
    start()
    vllm_docker._stop_at_exit()
    assert len(docker.compose_calls("down")) == 1 and vllm_docker.chosen() is None


def test_stopping_without_docker_is_quiet(docker):
    docker.installed = False
    vllm_docker.stop()


# ---------- the compose files ----------

class ComposeLoader(yaml.SafeLoader):
    """YAML with Compose's merge tags (`!reset`), which the plain loader refuses."""


ComposeLoader.add_constructor("!reset", lambda loader, node: None)
ComposeLoader.add_constructor("!override", lambda loader, node: loader.construct_sequence(node) if isinstance(node, yaml.SequenceNode) else loader.construct_mapping(node))


def service(name: str) -> dict:
    return yaml.load((COMPOSE_DIR / name).read_text(), Loader=ComposeLoader)["services"]["vllm"]


def default(value: str) -> str:
    """What ${VAR:-default} stands for when VAR is unset."""
    return re.sub(r"\$\{\w+:-([^}]*)\}", r"\1", value)


def test_the_server_listens_on_this_machine_only_like_the_page():
    (port,) = service("compose.yaml")["ports"]
    assert port.startswith("127.0.0.1:") and default(port) == "127.0.0.1:8000:8000"


def test_the_server_reads_the_projects_models_read_only_and_never_downloads():
    cpu = service("compose.yaml")
    models = next(v for v in cpu["volumes"] if v.split(":")[-2] == "/models" or ":/models:" in v)
    assert models.endswith(":/models:ro") and (COMPOSE_DIR / default(models).split(":")[0]).resolve() == MODELS_DIR
    assert cpu["environment"]["HF_HUB_OFFLINE"] == "1" and cpu["environment"]["HF_HUB_CACHE"] == "/models"


def test_by_default_the_server_serves_qwen_at_its_pinned_revision_under_its_own_id():
    model, revision, served, *_ = (default(arg) for arg in service("compose.yaml")["command"])
    assert model == QWEN and revision == f"--revision={families.REVISIONS[QWEN]}" and served == f"--served-model-name={QWEN}"


GPU_MEMORY = ("--kv-cache-memory-bytes=", "--gpu-memory-utilization=")  # the flags only the GPU file has


def test_the_gpu_file_runs_the_same_vllm_with_the_same_flags_but_the_cpu_only_one_and_its_memory():
    cpu, gpu = service("compose.yaml"), service("compose.gpu.yaml")
    assert [arg for arg in cpu["command"] if arg != "--skip-mm-profiling"] == [arg for arg in gpu["command"] if not arg.startswith(GPU_MEMORY)]
    assert "--enable-scale-out" in gpu["command"] and gpu["build"] is None
    assert gpu["deploy"]["resources"]["reservations"]["devices"][0]["driver"] == "nvidia"
    version = re.search(r"FROM vllm/vllm-openai-cpu:v([\d.]+)", (COMPOSE_DIR / "vllm-cpu" / "Dockerfile").read_text()).group(1)
    assert gpu["image"] == f"vllm/vllm-openai:v{version}" and cpu["image"].startswith(f"agent-loop-vllm-cpu:{version}-")


def test_on_the_gpu_the_kv_cache_is_the_cpus_size_and_vllm_starts_without_most_of_the_memory_free():
    # vLLM 0.32 refuses to start unless gpu_memory_utilization (0.92 by default) of the GPU's memory is free, which a
    # small card beside the page's own CUDA context, or a GPU that shares the machine's memory, does not have.
    flags = dict(default(arg).split("=", 1) for arg in service("compose.gpu.yaml")["command"] if arg.startswith(GPU_MEMORY))
    gib = int(default(service("compose.yaml")["environment"]["VLLM_CPU_KVCACHE_SPACE"]))
    assert flags["--kv-cache-memory-bytes"] == f"{gib}G"  # vLLM reads an upper-case G as GiB
    assert 0 < float(flags["--gpu-memory-utilization"]) <= 0.5


def test_the_cpu_image_keeps_a_step_that_takes_minutes_from_killing_the_engine():
    env = service("compose.yaml")["environment"]
    assert int(default(env["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"])) > 300 and int(default(env["VLLM_CPU_KVCACHE_SPACE"])) > 0


compose = pytest.mark.skipif(not shutil.which("docker") or subprocess.run(["docker", "compose", "version"], capture_output=True).returncode != 0,
                             reason="Docker Compose is not installed")


@compose
@pytest.mark.parametrize("files", [["compose.yaml"], ["compose.yaml", "compose.gpu.yaml"]])
def test_docker_compose_reads_the_files_as_the_page_runs_them(files):
    args = [arg for name in files for arg in ("-f", str(COMPOSE_DIR / name))]
    found = subprocess.run(["docker", "compose", "-p", "agent-loop-vllm", *args, "config", "--format", "json"], capture_output=True, text=True)
    assert found.returncode == 0, found.stderr
    vllm = json.loads(found.stdout)["services"]["vllm"]
    assert vllm["ports"][0]["host_ip"] == "127.0.0.1" and ("build" in vllm) == (files == ["compose.yaml"])


@compose
def test_a_relative_agent_loop_models_mounts_the_folder_the_weights_went_to(monkeypatch, tmp_path):
    # Compose takes a relative path in a volume from the compose file's folder, docker-compose/, not from where the
    # page started, and a name without a slash as a Docker volume; the server has to read the folder the page
    # downloaded the weights into.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_LOOP_MODELS", "elsewhere")
    monkeypatch.setattr(vllm_docker, "MODELS_DIR", runpy.run_path(simple_backend.__file__)["MODELS_DIR"])
    found = subprocess.run(["docker", *vllm_docker._compose("cpu", "config", "--format", "json")], capture_output=True, text=True, env=vllm_docker._env(QWEN))
    assert found.returncode == 0, found.stderr
    (models,) = [v for v in json.loads(found.stdout)["services"]["vllm"]["volumes"] if v["target"] == "/models"]
    assert models["source"] == str(tmp_path / "elsewhere") and models["read_only"]


# ---------- the real server ----------

@pytest.mark.skipif(os.environ.get("RUN_VLLM_DOCKER_TESTS") != "1", reason="set RUN_VLLM_DOCKER_TESTS=1 to start vLLM in Docker for real")
def test_real_server_in_docker_answers_through_vllm_backend():
    from harness.backends.vllm_backend import VLLMBackend

    vllm_docker._server.clear()
    server = vllm_docker.start(QWEN, lambda stage, line: None)
    try:
        llm = VLLMBackend(server["model"], server["url"])
        assert llm.info["server"]["generate_api"] and llm.info["server"]["logprobs_mode"] == "raw_logprobs"
        prompt = llm.encode(llm.render([{"role": "user", "content": "What is 2 + 2? Answer with the number only."}], None, False), [])
        events = list(llm.generate(prompt, Sampling(temperature=0.0, max_new_tokens=20), lambda: False))
        tokens = [e for e in events if e["type"] == "token"]
        assert "4" in events[-1]["raw"] and all(t["rank"] for t in tokens) and events[-1]["finish"] == "stop_token"
    finally:
        vllm_docker.stop()
