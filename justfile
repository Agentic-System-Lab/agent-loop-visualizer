# The commands in AGENTS.md and the READMEs, by name: `just` lists them, `just <name>` runs one. Each is the command
# the READMEs give, so every one of them also works without just. Recipes run in the project's folder and read the
# environment as those commands do: `MODEL_ID=Qwen/Qwen3.5-0.8B just run` starts on Qwen, `VLLM_URL=... just run` on
# a vLLM server. Arguments after a name go to the command, quoted as given (`just test -k render`, `just lint --fix`).

set positional-arguments

# List the recipes
default:
    @just --list --unsorted

# Install Python 3.12 and the packages uv.lock pins
sync:
    uv sync

# Download models into models/ ($AGENT_LOOP_MODELS if set) at their commits in REVISIONS: all of them (about 12 GB), or the ids given
download *ids:
    #!/usr/bin/env bash
    set -euo pipefail
    # The commits come from agent/model_families/__init__.py, so they stay the ones the page downloads at.
    pins=$(uv run python -c 'import sys; from agent.model_families import REVISIONS; print(*(i + "@" + REVISIONS.get(i, "main") for i in sys.argv[1:] or REVISIONS))' "$@")
    for pin in $pins; do
        uv run hf download "${pin%@*}" --revision "${pin#*@}" --cache-dir "${AGENT_LOOP_MODELS:-models}"
    done

# Build run_python's sandbox image, agent-loop-sandbox (again whenever agent/sandbox/Dockerfile changes)
sandbox:
    docker build -t agent-loop-sandbox agent/sandbox

# Start the app on http://localhost:8501
run *args:
    uv run streamlit run streamlit_app.py "$@"

# What checks.yml runs on every pull request: lint, types, the tests, and the guide with pygments alone
check: lint types test guide-check

# Lint, the order of imports included (--fix sorts them)
lint *args:
    uv run ruff check "$@"

# Check types: agent/ and app/ inside every function, tests/ where annotated
types *args:
    uv run mypy "$@"

# Run the tests that load no weights and touch no network, about 40 s
test *args:
    uv run pytest "$@"

# Load the real Gemma (with its MTP drafter) and Qwen, about 4 minutes (-k Qwen for one)
test-models *args:
    RUN_MODEL_TESTS=1 uv run pytest tests/test_real_model.py "$@"

# Ask a running vLLM server through VLLMBackend
test-vllm url="http://localhost:8000":
    VLLM_TEST_URL="$1" uv run pytest tests/test_vllm_backend.py -k real_server

# Start vLLM in Docker with Qwen3.5 and ask it through VLLMBackend, about 2 minutes
test-vllm-docker:
    RUN_VLLM_DOCKER_TESTS=1 uv run pytest tests/test_vllm_docker.py -k real

# Run the real pi (on the PATH, or PI_COMMAND) against a scripted model on 127.0.0.1
test-pi:
    RUN_PI_TESTS=1 uv run pytest tests/test_pi_harness.py -k real

# Build How it works: <out>/zh/, <out>/en/ and a redirecting <out>/index.html
guide out="_site":
    uv run python -m app.guide "$1"

# Test and build the guide as the Pages workflow does: pygments and pytest alone, not .venv
guide-check:
    uv run --isolated --no-project --with pygments --with pytest pytest tests/test_guide.py
    uv run --isolated --no-project --with pygments python -m app.guide _site

# Rewrite tests/guide_excerpts.txt, only after rereading in both languages the sections test_guide.py named
guide-snapshot:
    UPDATE_GUIDE_SNAPSHOT=1 uv run pytest tests/test_guide.py

# Start by hand the vLLM server the sidebar's vLLM (Docker) starts, on the cpu or on NVIDIA GPUs (gpu)
vllm-up device="cpu":
    docker compose -f docker-compose/compose.yaml {{ if device == "gpu" { "-f docker-compose/compose.gpu.yaml" } else if device == "cpu" { "" } else { error("the device is cpu or gpu") } }} up -d

# Stop that server, or one the app left running when it was stopped other than with Ctrl+C
vllm-down:
    docker compose -p agent-loop-vllm down
