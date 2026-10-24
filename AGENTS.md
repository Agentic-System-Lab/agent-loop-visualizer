# AGENTS.md

Agent Loop 可視化 is a Streamlit app that runs a small model on the user's machine (Gemma 4 E2B or
Qwen3.5 0.8B, through Hugging Face `transformers`) and draws every step of its agent loop: context,
model, harness, environment, stop, down to each token's probability. It is a teaching project for the
Agent Atlas course. The page and its 運作說明 (How it works) guide are in Traditional Chinese and
English. README.md (English) and README.zh-TW.md (Traditional Chinese) are the full account of what the
page shows, the architecture, the tests and the known limits; read the parts you are about to change.

## Commands

```bash
uv sync                                                    # Python 3.12, pinned by uv.lock
uv run pytest                                              # ~25 s; loads no weights, touches no network
RUN_MODEL_TESTS=1 uv run pytest tests/test_real_model.py   # real Gemma (with its MTP drafter) and Qwen; -k Qwen for one
uv run streamlit run streamlit_app.py                      # http://localhost:8501; MODEL_ID=Qwen/Qwen3.5-0.8B starts on Qwen
uv run python guide.py _site                               # the guide: _site/zh/, _site/en/ and a redirecting _site/index.html
```

No linter or formatter is configured. Match the surrounding code by hand and don't reformat lines you
are not changing.

## Layout

```
streamlit_app.py   layout, session state, model menu, background thread; redraws every 0.5 s while a run is live
agent.py           the harness (Run): prompt → generate → parse → validate → (ask) → run tool → back into context → stop?
llm.py             LocalLLM: the hand-written decoding loop (KV cache, temperature / top-k / top-p, MTP draft and verify)
families/          everything model-specific: one Family per model family; MODELS and DRAFTERS in __init__.py
tools.py           tools: the schema the model reads plus the Python function the harness runs
render.py          pure functions from the event list to HTML: loop diagram, step cards, token heat and popovers, context
guide.py           the 運作說明 page: content in both languages, code references by text, static HTML for GitHub Pages
i18n.py            the two languages: LANGS, pick(lang, zh, en), from_locale(); no dependencies
assets/            style.css (palette and app), dark.css, guide.css, guide.js (the guide's one inlined script) and guide.js (the 運作說明 page)
tests/             pytest; fakes.py is a scripted stand-in for LocalLLM
```

## Rules that keep it working

- **Events are the interface.** `agent.Run` runs in a background thread and appends one event per
  stage; the order is in the READMEs' Architecture (架構) section. `render.py` is a pure function of that list and the
  page only draws it, which is why stop and approve can rerun the script without interrupting the loop.
  A new or changed event needs render.py, the `EVENTS` table in guide.py (tests check that it lists
  every event the harness sends and points at the line that sends it) and the event list in both READMEs.
- **Nothing model-specific outside `families/`.** agent.py, llm.py and render.py have no "if Gemma".
  Parsing output, the tokens that switch to thinking / tool call / reply, stop tokens, special-token
  explanations and recommended sampling all live on the `Family`. Another checkpoint of a supported
  family is one line in `MODELS` (and `DRAFTERS` if it has an MTP drafter); a new family is a module
  like `families/qwen3_5.py` added to `FAMILIES`.
- **The decoding loop stays exact.** Greedy output equals `model.generate()`. MTP only changes how many
  forward passes run: greedy with MTP equals greedy without, and sampling with MTP keeps each token's
  distribution. `tests/test_llm.py` and `tests/test_real_model.py` check this; don't loosen them.
- **Escape everything the model writes** before it reaches `st.html`. tests/test_render.py checks it,
  and also that every prefix of a run's events renders, because the page redraws mid-run.
- **guide.py finds code by text, not line number.** `Code("agent.py", "def run(self):")` and inline
  `[[agent.py|def run(self):]]` are looked up when the page is built, and each must match exactly one
  line. Before renaming or rewriting a line, grep guide.py for it; if a reference breaks,
  tests/test_guide.py fails. Fix the reference, and when the behavior changed, the prose around it too.
- **guide.py builds with pygments alone.** The Pages workflow runs
  `uv run --no-project --with pygments python guide.py _site`, and guide.py imports render and i18n,
  and render imports families. Keep those free of torch, transformers, streamlit and httpx.
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
- **One model in memory, one run at a time.** `current_llm()` is `st.cache_resource(max_entries=1)` and
  `generation_slot()` is a lock shared by every tab.
- The app listens on localhost only (`.streamlit/config.toml`) because it runs a model on the user's
  machine. Keep it that way.

## Tests

- Most tests use `FakeLLM` from tests/fakes.py: same interface as `LocalLLM`, prompts in a simplified
  form of each family's chat format, scripted replies streamed token by token. Cover new harness or UI
  behavior with it; tests/test_app.py drives the whole page through Streamlit's `AppTest`.
- tests/test_chat_template.py checks each family against the model's real tokenizer and chat template
  and skips models missing from the local Hugging Face cache. tests/test_real_model.py loads the real
  weights and runs only with `RUN_MODEL_TESTS=1`. After changing parsing, prompts, stop tokens or the
  decoding loop, run them if the models are available, and say whether they ran or were skipped.
- Wikipedia is faked over HTTP; the default suite never touches the network.

## Language and style

- Code, comments, docstrings, test names and commit messages are in English. So is the text the model
  reads (system prompts, tool schemas). What a person sees is in both Traditional Chinese with Taiwan
  usage and English: UI strings, stop reasons, tool errors, the READMEs, the 運作說明 page. In Chinese,
  use the terms the page already uses: 脈絡 for context, 模型, harness, 環境 for environment, 停止,
  工具, 特殊 token, 推測解碼; keep token, prefill, KV cache, top-k and the like in English. In English,
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
