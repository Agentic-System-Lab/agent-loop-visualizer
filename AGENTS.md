# AGENTS.md

Agent Loop 可視化 is a Streamlit app that runs a small model on the user's machine (Gemma 4 E2B or
Qwen3.5 0.8B, through Hugging Face `transformers`, or on vLLM: in Docker, started from the sidebar, or a
server at `VLLM_URL`) and draws every step of its agent loop: context,
model, harness, environment, stop, down to each token's probability. A question can carry images, sound
and video for the model, and, when the sidebar lets the model write code, data files for a `run_python`
tool that runs in a Docker sandbox. It is a teaching project for the
Agent Atlas course. The page and its 運作說明 (How it works) guide are in Traditional Chinese and
English. README.md (English) and README.zh-TW.md (Traditional Chinese) are the full account of what the
page shows, the architecture, the tests and the known limits; read the parts you are about to change.

## Commands

```bash
uv sync                                                    # Python 3.12, pinned by uv.lock; uv >= 0.13.4 (required-version in pyproject.toml)
uv run pytest                                              # ~40 s; loads no weights, touches no network
RUN_MODEL_TESTS=1 uv run pytest tests/test_real_model.py   # real Gemma (with its MTP drafter) and Qwen, ~4 min; -k Qwen for one
docker build -t agent-loop-sandbox harness/sandbox         # optional: run_python's sandbox; its tests skip without it; bump SANDBOX_VERSION with the Dockerfile
uv run streamlit run streamlit_app.py                      # http://localhost:8501; MODEL_ID=Qwen/Qwen3.5-0.8B starts on Qwen
VLLM_URL=http://localhost:8000 uv run streamlit run streamlit_app.py   # the model on a vLLM server: vllm serve <model> --revision <its REVISIONS commit> [--enable-scale-out [--return-sampling-mask --logprobs-mode processed_logprobs]]
VLLM_TEST_URL=http://localhost:8000 uv run pytest tests/test_vllm_backend.py -k real_server   # vllm_backend.py against that real server
docker compose -f docker-compose/compose.yaml up -d        # the vLLM server the sidebar's vLLM (Docker) starts, by hand (+ -f docker-compose/compose.gpu.yaml on NVIDIA GPUs)
RUN_VLLM_DOCKER_TESTS=1 uv run pytest tests/test_vllm_docker.py -k real   # starts that server with Qwen3.5 and asks it through VLLMBackend, ~2 min
uv run python -m app.guide _site                           # the guide: _site/zh/, _site/en/ and a redirecting _site/index.html
UPDATE_GUIDE_SNAPSHOT=1 uv run pytest tests/test_guide.py  # after rereading the guide sections a code change touched
```

No linter or formatter is configured. Match the surrounding code by hand and don't reformat lines you
are not changing.

## Layout

Two packages: `harness/` runs a turn, `app/` is what a person sees. `app/` imports `harness/`, never the
other way round.

```
streamlit_app.py         where Streamlit starts: imports app.page and calls main() on every rerun
app/                     the Streamlit app
  page.py                layout, session state, model menu, background thread; redraws every 0.5 s while a run is live
  render.py              pure functions from the event list to HTML: loop diagram, step cards, token heat and popovers, context
  guide.py               the 運作說明 page: content in both languages, code references by text, static HTML for GitHub Pages
  assets/                style.css (palette and app), dark.css, guide.css, guide.js (the guide's one inlined script), logo.svg, examples/ (files the examples attach, also offered above the input box)
harness/                 the agent, the model and what runs it; no Streamlit
  agent.py               the harness (Run): prompt → generate → parse → validate → (ask) → run tool → back into context → stop?
  backends/
    backend.py           Backend: what agent.py and the page ask of whatever runs the model (SimpleBackend, VLLMBackend, tests' FakeLLM); Sampling, Encoded, the token event's fields
    simple_backend.py    SimpleBackend: the hand-written decoding loop (KV cache, temperature / top-k / top-p, MTP draft and verify)
    vllm_backend.py      VLLMBackend (VLLM_URL): the tokenizer and chat template here, the token ids to a vLLM server (/v1/completions, or /inference/v1/generate with --enable-scale-out: media, sampling masks), what it says about each token back
    vllm_docker.py       the sidebar's vLLM (Docker): downloads the model, `docker compose up -d` with it, waits until it answers, `docker compose down`; DockerError in both languages
  families/              everything model-specific: one Family per model family; MODELS and DRAFTERS in __init__.py
  tools.py               tools: the schema the model reads plus the Python function the harness runs; run_python's docker run
  media.py               attachments: saved per conversation, written into the user message, decoded for the processor (PyAV)
  i18n.py                the two languages: LANGS, pick(lang, zh, en), from_locale(); no dependencies
  sandbox/               run_python's sandbox image, agent-loop-sandbox: the Dockerfile, and the matplotlib backend (agent_loop_figures.py) and matplotlibrc that send charts back on stdout
docker-compose/          the vLLM server vllm_docker.py starts: compose.yaml (CPU, any machine), compose.gpu.yaml (on top: NVIDIA GPUs), vllm-cpu/ (vLLM 0.32.0's CPU image + patch_vllm.py's two fixes)
tests/                   pytest; fakes.py is a scripted stand-in for SimpleBackend; samples.py makes a file of every accepted format; guide_excerpts.txt is the code the guide quotes
models/                  downloaded models (simple_backend.MODELS_DIR), which git ignores; AGENT_LOOP_MODELS moves them, HF_HUB_CACHE does not
```

Imports are absolute from the project's folder (`from harness import families`, `from harness.backends.backend
import Backend`, `from app import render`); pytest puts that folder on the path (`pythonpath` in pyproject.toml).

## Rules that keep it working

- **The harness never knows the app.** Nothing in `harness/` imports `app/`, Streamlit or anything that draws; it
  reports what happens only as events. `harness/__init__.py`, `harness/families/`, `harness/i18n.py` and
  `app/__init__.py` import nothing heavy, because the guide's build loads them with pygments alone.
- **app/page.py is imported once, not run as a script.** streamlit_app.py imports it and calls `main()` on every
  rerun, so everything a rerun does is inside functions, and what the module reads from the environment (`VLLM_URL`,
  `MODEL_ID`, `AGENT_LOOP_UPLOADS`) is read once per process. Tests change those on the module (`page.VLLM_URL`,
  `page.UPLOADS`, `page.SimpleBackend`), not in the environment.
- **Events are the interface.** `agent.Run` runs in a background thread and appends one event per
  stage; the order is in the READMEs' Architecture (架構) section. `app/render.py` is a pure function of that list and the
  page only draws it, which is why stop and approve can rerun the script without interrupting the loop.
  A new or changed event needs render.py, the `EVENTS` table in app/guide.py (tests check that it lists
  every event the harness sends and points at the line that sends it) and the event list in both READMEs.
  The harness and the page reach the model only through `backend.Backend`; a token event's probabilities,
  rank, candidates and MTP draft (`backend.TOKEN_DETAILS`) are optional, and render.py leaves out what is missing.
- **Nothing model-specific outside `harness/families/`.** agent.py, simple_backend.py and render.py have no "if Gemma".
  Parsing output, the tokens that switch to thinking / tool call / reply, stop tokens, special-token
  explanations, notes on the files of its checkpoints and recommended sampling all live on the
  `Family`. Another checkpoint of a supported family is one line in `MODELS` and one in `REVISIONS`,
  the commit it downloads at (and `DRAFTERS` and its drafter's commit if it has an MTP drafter, and `VLLM_SPECULATIVE` for how
  vLLM drafts for it on a GPU); a new family is a module like
  `harness/families/qwen3_5.py` added to `FAMILIES`. Moving a model to a newer commit means rerunning
  tests/test_chat_template.py and tests/test_real_model.py and rereading the guide's model folder
  section, whose numbers come from those files.
- **The decoding loop stays exact.** Greedy output equals `model.generate()`. MTP only changes how many
  forward passes run: greedy with MTP equals greedy without, and sampling with MTP keeps each token's
  distribution. `tests/test_simple_backend.py` and `tests/test_real_model.py` check this; don't loosen them.
  This is about simple_backend.py: vLLM runs its own kernels, so VLLMBackend promises only that the server reads
  exactly the prompt's token ids and that nothing it shows was made up where the server sent nothing, nor shown as
  something else: a processed --logprobs-mode is the sampling distribution, never the model's p.
- **Escape everything the model writes** before it reaches `st.html`. tests/test_render.py checks it,
  and also that every prefix of a run's events renders, because the page redraws mid-run.
- **app/guide.py finds code by text, not line number.** `Code("harness/agent.py", "def run(self):")` and inline
  `[[harness/agent.py|def run(self):]]` are looked up when the page is built, by their path from the project's folder,
  and each must match exactly one line. Before renaming, moving or rewriting a line, grep app/guide.py for it; if a reference breaks,
  tests/test_guide.py fails. Fix the reference, and when the behavior changed, the prose around it too.
- **The guide's text is reread whenever the code it quotes changes.** tests/guide_excerpts.txt holds
  every excerpt and the line behind every chip, without line numbers. When an edit changes any of
  them, tests/test_guide.py fails and names the sections. Reread each one in both languages, fix
  what is no longer true, then rewrite the file with `UPDATE_GUIDE_SNAPSHOT=1 uv run pytest
  tests/test_guide.py` and commit it with the change. Never regenerate it without rereading.
- **app/guide.py builds with pygments alone.** The Pages workflow runs
  `uv run --no-project --with pygments python -m app.guide _site`, and guide.py imports app.render and
  harness.i18n, and render imports harness.families. Keep those (and the packages' `__init__.py`) free of torch,
  transformers, streamlit and httpx. Before building, it runs tests/test_guide.py with pygments and pytest alone,
  so that file imports only app.guide, harness.i18n and tests/html_check.py (not test_render.py, which imports torch).
- **Every word a person sees exists in both languages, side by side.** In code it is
  `pick(lang, "中文", "English")`; in data (special-token notes, stop reasons, tool environments,
  guide paragraphs, the guide's event table) it is a `{"zh": ..., "en": ...}` dict. Never add a
  string in one language only. render.py functions take `lang` explicitly and stay pure: the events
  carry no language, so a switch redraws finished turns too. tests/test_render.py, test_guide.py and
  test_app.py fail when Chinese shows up on an English page, and test_guide.py and test_families.py
  when the two languages drift apart (different refs, links or tokens).
- **The model reads harness and tool errors in the run's language.** `RunSettings.lang` is the page's
  language when the turn started; `Tool.validate`, `family.parse` and the harness's own messages word
  errors in it, and `ToolError` / `ParseError` carry both texts. The system prompts in agent.py are the
  same for both languages, so the model's behavior does not depend on the page's language.
- **The language switch.** The app starts in the language in the `agent_loop_lang` cookie, else the
  browser's (`st.context.locale` through `i18n.from_locale`); picking one writes the cookie with an
  `st.html(..., unsafe_allow_javascript=True)` script, because Streamlit only reads cookies. The
  switch is pinned to the top right by style.css (`.st-key-lang_switch`). The guide's pages remember a
  pick in localStorage for the root index.html. A widget whose options are formatted per language must
  be a segmented control or similar, never a selectbox: a selectbox sends its label back, and a label
  that changes with the language becomes its value (this is why `MODELS` labels are the same in both).
- **One model in memory, one run at a time.** `load_llm()` is `st.cache_resource(max_entries=1)`, and switching models
  or what runs them calls `release_model()` first, so the next model never loads beside the last; picking transformers
  again stops vLLM's container. `generation_slot()` is a lock shared by every tab.
- The app listens on localhost only (`.streamlit/config.toml`) because it runs a model on the user's
  machine. Keep it that way. With `VLLM_URL` the prompts go to that server; the page itself stays local. The vLLM
  server in Docker publishes its port on 127.0.0.1 only (docker-compose/compose.yaml) and reads models/ read-only and
  offline, so it loads the commit in `REVISIONS`. The same file keeps `runner.postScriptGC` off: a full garbage
  collection after every script run held the GIL long enough for the agent's thread to read a vLLM step's tokens all
  at once, at thousands of tok/s; and `runner.fastReruns` off: a rerun in a new thread beside the one loading a model let
  that load finish after the model was let go, beside vLLM (tests/test_app.py checks both). `simple_backend.empty_cache()`
  frees cuBLAS's workspaces before emptying CUDA's cache, or the segment holding the weights stays reserved.
- **What a model reads comes from the checkpoint and the backend.** `SimpleBackend.modalities` is worked out from the config
  and the processor, not the family; `VLLMBackend.modalities` is that minus what the server refused when it was probed at
  load (all of it without `--enable-scale-out`), and the page says which it refused and why. A file of a kind the model does not read is a plain file for
  `run_python`. `encode()` expands the media into their tokens, and the `context` event carries that
  expanded text, because it is what the model reads.
- **Docker only when asked for.** `run_python` is offered only when Let the model write code is on and
  `tools.sandbox_problem()` finds Docker and the image, and vLLM in Docker starts only once What runs the model is
  vLLM. Until one of those is on, the page never runs `docker`, so the app works without it. Keep the sandbox's flags
  (no network, read-only, limits, unprivileged) in `tools.sandbox_command()`. docker-compose/compose.yaml tags the
  CPU image `agent-loop-vllm-cpu:<vLLM version>-<n>`: bump `n` when vllm-cpu/ changes, so machines with the old
  image build the new one, and keep compose.gpu.yaml's command the same as compose.yaml's but `--skip-mm-profiling`, plus its two flags on GPU
  memory and two on speculative decoding (tests/test_vllm_docker.py checks it). The app stops the container it started when it quits (`_stop_at_exit`), so a test
  that fakes a start leaves `vllm_docker._chosen` empty.

## Tests

- Most tests use `FakeLLM` from tests/fakes.py: same interface as `SimpleBackend`, prompts in a simplified
  form of each family's chat format, scripted replies streamed token by token. Cover new harness or UI
  behavior with it; tests/test_app.py drives the whole page through Streamlit's `AppTest`.
- tests/test_chat_template.py checks each family against the model's real tokenizer and chat template
  and skips models not yet downloaded into models/. tests/test_real_model.py loads the real
  weights and runs only with `RUN_MODEL_TESTS=1`. After changing parsing, prompts, stop tokens or the
  decoding loop, run them if the models are available, and say whether they ran or were skipped.
- tests/test_vllm_backend.py fakes the vLLM server with `httpx.MockTransport`, in the stream formats of vLLM
  0.32's /v1/completions and /inference/v1/generate (with and without --enable-scale-out, sampling masks, media, in each
  --logprobs-mode, refusing what vLLM refuses);
  when the models' tokenizers are in models/ it also runs them through the real chat templates, and with
  `VLLM_TEST_URL` it asks a real server. Read vLLM's source before changing what is sent or read.
- Wikipedia is faked over HTTP; the default suite never touches the network. Tests that need Docker
  run only when `tools.sandbox_problem()` finds it, and skip otherwise. tests/test_vllm_docker.py fakes Docker at the
  level of `subprocess`; only `RUN_VLLM_DOCKER_TESTS=1` starts a real vLLM container.
- Every format `harness/media.py` accepts has a sample in tests/samples.py (test_media.py fails otherwise), a
  decoding test, and a real-model test that asks both models about it. A new format needs all three.

## Language and style

- Code, comments, docstrings, test names and commit messages are in English. So is the text the model
  reads (system prompts, tool schemas). What a person sees is in both Traditional Chinese with Taiwan
  usage and English: UI strings, stop reasons, tool errors, the READMEs, the 運作說明 page. In Chinese,
  use the terms the page already uses: 脈絡 for context, 模型, harness, 環境 for environment, 停止,
  工具, 特殊 token, 推測解碼, 滑動視窗 for sliding window, 隱藏層維度 for hidden size, 注意力頭 for
  attention head; keep token, prefill, KV cache, top-k and the like in English. In English,
  use the page's terms too: context, model, harness, environment, stop, tool, special token, drafter,
  speculative decoding, observation, turn (one question) and step (one time around the loop), and the
  UI's own labels (Use tools, Thinking mode, Ask me before running a tool, Max steps, How it works).
- Every module opens with a docstring on what it is for and how it fits with the rest. Comments say
  why, in full sentences. Dataclasses, type hints, CAPS module constants, f-strings; long lines are
  fine where the neighbors have them.
- Test names are sentences about behavior, e.g. `test_observation_is_in_the_next_prompt`.
- Commit messages: an imperative subject that says what changed for the user, then a body on what and
  why; group bigger changes under short headings or `-` bullets.

## Keep the README in step

README.md (English) and README.zh-TW.md (Traditional Chinese) say the same things: the screens, the
experiments, the architecture, the two languages, the decoding loop and MTP, a table of what each test
file covers, measured speeds and known limitations. A change that makes any of it untrue updates both
in the same commit, each in its own language and in the README's voice; each links to the other at the
top. `docs/screenshot-en.png` and `docs/screenshot-zh.png` are their images, each showing the page in
its README's language, taken from a real Gemma run: if a UI change makes them stale, say so rather than
leaving them silently outdated.
