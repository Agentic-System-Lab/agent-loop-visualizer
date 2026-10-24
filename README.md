# Agent Loop Visualizer

English | [繁體中文](README.zh-TW.md)

One of the projects of [Agent Atlas](https://cyyeh.github.io/agent-atlas-learning/) (in Traditional Chinese).

- Models: pick [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) or [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) in the sidebar; they run on your machine with Hugging Face `transformers`
- MTP speculative decoding: for Gemma 4 E2B you can switch it on in the sidebar (it is off by default); it uses [google/gemma-4-E2B-it-assistant](https://huggingface.co/google/gemma-4-E2B-it-assistant) as the drafter, so one forward pass produces several tokens
- Languages: the page and the How it works guide are in English and Traditional Chinese. They start in your browser's language, the switch at the top right changes it, and the language you pick is remembered for your next visit
- License: [Apache License 2.0](LICENSE)

On the left is a chat like ChatGPT's; on the right the agent loop is drawn step by step: the model reads the context and proposes an action, the harness carries it out, and the result goes back into the context, until it stops. The model runs on your own computer, so every token's probability, the candidate tokens, the special tokens, the prefill and the generation speed are all in view. The models run the same harness, so switching models changes only their text formats, which makes them easy to compare.

[How it works](https://cyyeh.github.io/agent-loop-visualizer/en/) maps each step of one turn, from sending a question to the reply, to the code that does it, with file names, line numbers and excerpts. It lives on GitHub Pages; How it works at the top of the app opens it in a new tab, in the language the app is in.

![The top bar links to the How it works page, the switch at the top right changes the language, and the sidebar picks the model; on the left is the chat, on the right the loop diagram and the tokens of step 1. The pointer rests on the special token <|tool_call>, whose popover says what it is for and shows its probability, its rank and the top 5 candidates.](docs/screenshot-en.png)

## What the page shows

| On the page | What you see | Unit 01 reading (in Chinese) |
|---|---|---|
| Loop diagram | Which box you are in (context, model, harness, environment, stop) and the arrow that is flowing | §4 An agent is a model in a loop |
| Context | The whole string the model reads in each step: the system prompt, the tool schemas, the conversation and the tool result added after the last step. Hovering over a dark special token says what it is for | §1 The context has a limit, §3 Tool definitions are tokens too |
| Model | Every generated token, darker the less likely the chosen token was. Hovering over a token shows its probability, its rank, the top 5 candidates and which ones top-k / top-p cut; a special token also says what it is for. With MTP on, tokens the drafter guessed right are underlined, the popover says what the drafter guessed and whether it was kept, and the step card shows how many were kept and how many forward passes the model saved | §1 An LLM samples from a conditional probability distribution; decoding settings |
| Harness | The tool calls parsed from the model's own format (Gemma 4 or Qwen3.5), the schema check, warnings about repeated calls, and your Allow or Reject | §3 The model never touches the environment |
| Environment | What the tool really returned and how long it took | §2 The environment and observable state |
| Stop | Why it stopped and at which token. Gemma stops at `<turn\|>` after an answer (like end_turn) and at `<\|tool_response>` when it wants a tool (like tool_use); Qwen stops at `<\|im_end\|>` either way. Then there are the step limit, the token limit and the user pressing stop | §5 When to stop |
| Model tab | The parameter breakdown, the number and kinds of layers, attention heads, the context limit, the MTP drafter, what each decoding setting does, each step's prefill time and tok/s (with MTP, also how many guesses were kept and how many decoding forward passes ran), and the id and purpose of every special token of the tokenizer (24 for Gemma, 33 for Qwen) | §1, §7 An agent is a system |
| How it works | The 15 stages of one turn (including how MTP guesses and checks), each with the code that does it and its line numbers, in English and in Chinese. It is on GitHub Pages; the line numbers are looked up in the source when the page is built, and it is rebuilt on every push to main, so it follows the code | All of it |

## Quick start

You need:

- macOS (Apple Silicon, using MPS), Linux with CUDA, or a CPU alone (very slow)
- Disk space for the weights: about 10 GB for Gemma 4 E2B, 1.8 GB for Qwen3.5 0.8B, and about 160 MB more for Gemma's drafter if you want MTP
- Memory: with MPS or CUDA, 16 GB or more for Gemma (the bfloat16 weights take about 10 GB); on a CPU it loads in float32, about 20.5 GB, so 24 GB or more. Qwen3.5 0.8B is much smaller and runs on an 8 GB machine. Only one model is loaded at a time; switching models lets the old one go
- Linux + CUDA: the pinned torch is built for CUDA 13, so the CUDA Version at the top right of `nvidia-smi` must be 13.0 or newer. With an older driver torch does not see the GPU and runs on the CPU; the top of the sidebar shows the device in use
- [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you)
- A network connection: to download the models the first time, and for the Wikipedia tools

```bash
uv sync
uv run hf download google/gemma-4-E2B-it   # download the weights first (about 10 GB); otherwise they download the first time the model is picked
uv run hf download Qwen/Qwen3.5-0.8B       # about 1.8 GB
uv run hf download google/gemma-4-E2B-it-assistant   # the MTP drafter, about 160 MB; otherwise it downloads the first time MTP is switched on
uv run streamlit run streamlit_app.py      # loads Gemma first; MODEL_ID=Qwen/Qwen3.5-0.8B loads Qwen first
```

Your browser opens http://localhost:8501, in its own language (English unless the browser is set to Chinese); the 中文 | English switch at the top right changes it, and the next visit starts in the one you picked. Loading Gemma and warming it up takes about 10–20 seconds the first time, Qwen a few seconds; reloading the page does not load it again. The Model menu at the top of the sidebar switches models; switching starts a new conversation, resets the decoding settings to the new model's recommendations and turns MTP off. MTP speculative decoding in the sidebar loads the drafter the first time it is switched on (in under a second), and it stays in memory after that.

> A 401 while downloading, though the model is public? Most likely the Hugging Face token saved on your machine has expired. Log in again (`uv run hf auth login`), or add `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` to send no token.

## Experiments to try

1. **Just a chatbot**: switch off Use tools in the sidebar and ask "What time is it in Taipei right now?". The model sees no tools, so it can only guess or admit it does not know. Switch tools back on, ask again, and compare the two in the Context tab.
2. **Sampling**: ask the same question with temperature 0 and with 1.5, and watch how the token colors and the candidates left in the popover change.
3. **Stop conditions**: set Max steps to 1 and ask something that needs a tool; the stop reason becomes "step limit reached".
4. **Where the human sits**: switch on Ask me before running a tool, then press Reject. It is the harness that blocks the call; the model only sees an error observation in the next step's context. Compare the mail deletion incident in §5 of the reading: a "confirm first" written in the prompt can get compressed away, one written into the harness cannot.
5. **Reasoning is tokens too**: switch on Thinking mode, and italic thinking tokens appear before the tool call; in the Context tab, the system turn starts with an extra `<|think|>`.
6. **The loop goes in circles**: ask "Who proposed the ReAct paper? Look it up on Wikipedia." E2B often repeats the same search, and the harness flags "exactly the same call as before"; compare the error analysis of ReAct in §4 of the reading.
7. **Switch models, keep the harness**: ask Gemma and Qwen the same question and compare them in the Context tab: the turn markers (`<|turn>` against `<|im_start|>`), where the tool schemas go, how a tool call is written (`call:name{…}` against XML) and where the tool result goes back all differ, yet the harness, environment and stop sections of the step cards are exactly the same.
8. **MTP is fast only when it guesses right**: with Gemma, switch on MTP speculative decoding, press "What time is it in Taipei right now?" and "Write a Python function to test for primes.", then ask "用三段話介紹台北。" (introduce Taipei in three paragraphs, in Chinese). Compare each step's tok/s and underlines: the drafter guesses almost every token of a tool call, about half of English code and only about a tenth of long Chinese text. Set temperature to 0 and run with MTP on and off: the output is exactly the same, because the drafter only decides how many forward passes run, not what gets written.

## Architecture

```
streamlit_app.py   layout, session state, model menu, language switch, background thread; redraws every 0.5 s while a run is live
guide.py           the How it works page: its content in both languages, line lookups, and the static HTML for GitHub Pages
render.py          turns the event list into HTML: the loop diagram, step cards, token heat and popovers, the context view
agent.py           the harness: build the prompt → generate → parse → validate → (ask you) → run the tool → back into the context → stop?
llm.py             local inference with a hand-written decoding loop; MTP's drafting and checking are here too
i18n.py            the two languages: pick() chooses the text for one, from_locale() picks the default from the browser's language
families/          everything that goes with the model: how each model family writes turns, thinking and tool calls, which tokens stop it, what each means
  __init__.py      the Family definition, the model menu MODELS, the MTP drafters DRAFTERS, finding a family by config.json's model_type
  gemma4.py        Gemma 4: the output parser and notes on its 24 special tokens
  qwen3_5.py       Qwen3.5: the output parser and notes on its 33 special tokens
tools.py           tools: the schema (for the model) plus the Python function (for the harness to run)
tests/             pytest tests; see Tests below
```

The agent runs in a background thread and appends an event at every stage; the UI only draws the event list. So when you press Stop or Allow and Streamlit reruns the script, the loop in the background is not interrupted.

How it works is a static page: `uv run python guide.py _site` writes `_site/en/index.html` and `_site/zh/index.html` (styles and a small script inlined, so they open straight in a browser), and `_site/index.html`, which sends a reader to the language they last picked on the page's switch, or else their browser's. On a push to main, `.github/workflows/pages.yml` does the same and publishes it to GitHub Pages.

The events of one run, in order: `run_start`, then for each step `step_start` → `context` → `prefill` → `token` ×N → `model_end` → `parsed`, with a tool call also `tool_call` → (`approval_request` → `approval`) → `tool_start` → `tool_result`, and finally `run_end`.

### Two languages

Every piece of text on the page is written in both languages side by side in the code: `pick(lang, "中文", "English")` in code, and `{"zh": …, "en": …}` in data (the special token notes, the stop reasons, the paragraphs of How it works). Every function in `render.py` takes a `lang`, and the events themselves have no language, so switching the language redraws the same events in the other one, finished turns included. The page starts in the language last picked, kept in a cookie, or else in the browser's `navigator.language` (anything starting with `zh` is Chinese, the rest English); picking one writes the cookie with a small script, because Streamlit can read cookies but not write them.

What the model reads from the harness (wrong arguments, a failed tool, your refusal) is worded in the language the page was in when the turn started, because it becomes the observation word for word; the system prompt is the same in both, so the model's behavior does not depend on the page's language.

### Why a hand-written decoding loop

`model.generate()` does not hand out the probability distribution at each step, so `llm.py` runs the loop itself: one forward pass per token (with a KV cache), then temperature, top-k and top-p in turn, then a sample. So every token can report its probability and rank in the model's own distribution, its probability when sampled, how many candidates were left after top-k / top-p, and the top 5 candidates. In greedy mode the output matches `generate()`, as measured.

### MTP speculative decoding

Decoding runs a full forward pass for every token, and the time goes mostly into reading the weights from memory; computing a few more positions in the same pass is hardly slower (Gemma 4 E2B on an Apple M5: about 56 ms for 1 position, about 70 ms for 7). So if the next few tokens are guessed cheaply first, the model can check them all in one forward pass. Google trained such a drafter for Gemma 4 (MTP, multi-token prediction), and `transformers`' `generate(assistant_model=…)` uses it, but that again hides the distribution at each step, so `llm.py` wires it into its own decoding loop:

1. **Guess** (`_draft`): the drafter `gemma-4-E2B-it-assistant` has about 78M parameters and 4 layers. It has no KV cache of its own and reads the model's keys and values directly (the last 20 layers of Gemma 4 E2B already share those two earlier layers computed); its input is the last token's embedding next to the model's last-layer hidden state. It makes 6 greedy guesses at a time (the checkpoint's `num_assistant_tokens`).
2. **Check** (`_drafted`): the model runs one forward pass over the token just settled plus the 6 guesses, gets back the distributions at 7 positions and hands them to `_sample()` in order. A kept guess needs no forward pass of its own; the first guess turned down is replaced by the model's own pick at that position, everything after it is dropped, and the KV cache is cut back with `crop()`; when all 6 hold, the last position gives one more token.
3. **Same output**: in greedy decoding a guess is kept only when it is the top token, so the output is the same as without MTP (`tests/test_real_model.py` compares them with the real model). When sampling, a guess is kept with its probability after temperature, top-k and top-p, and otherwise the token is drawn from the other candidates, so every token comes out exactly as often as without MTP (speculative sampling; `tests/test_llm.py` checks it over twenty thousand samples).

How many guesses are kept depends a lot on the content. Measured in the page with Gemma's default decoding settings (Apple M5):

| What the step writes | Without MTP | With MTP | Guesses kept |
|---|---|---|---|
| A tool call (step 1 of "台北現在幾點？紐約呢？", "What time is it in Taipei now? And New York?") | 12.6 tok/s | 40.9 tok/s | 34 / 35 |
| The answer from the tool result (step 2 of the same question) | 13.0 tok/s | 22.6 tok/s | 42 / 106 |
| English code, about 750 tokens | 8.8 tok/s | 11.6 tok/s (`llm.py` alone: 8.1 → 29.0) | 51% |
| Long Chinese text ("用三段話介紹台北。") | 10.3 tok/s | 10.3 tok/s | 7% |

### Supporting more models

The harness (`agent.py`), the decoding loop (`llm.py`) and the page (`render.py`) are written once, without a single "if Gemma" branch. Whatever differs between models is gathered in one `Family` object in `families/`:

- Reading the output: `parse` (splits the output into thinking, tool calls and the reply), `thought_prefix`, `channel_switch` (which tokens switch to thinking, a tool call or the reply), `stop_tokens`
- Writing the conversation back: `reasoning_key` (the field the thinking is kept in, so the chat template can read it), `text_with_calls` (whether text in front of a tool call is kept)
- The page: `user_turn`, a note on each special token, the help text for the sidebar and the Model tab, and the recommended decoding settings when there is no generation_config.json; each in English and in Chinese

To add a model:

1. Another checkpoint of a family that is already here (a bigger Gemma 4 or Qwen3.5, say): add a line to `MODELS` in `families/__init__.py`. Other Gemma 4 checkpoints have MTP drafters too (`google/gemma-4-*-it-assistant`); add a line to `DRAFTERS` and the sidebar's MTP switch works for them.
2. A new family: write a module like `families/qwen3_5.py` (a parser for the output plus notes on the special tokens, in both languages) and add its `Family` to `FAMILIES`. Loading a model finds it automatically by the `model_type` in the checkpoint's `config.json`. `tests/test_chat_template.py` checks a new family against the real tokenizer and chat template: every special token has a note, tool calls the template writes into the history parse back, and what the step card marks as new is exactly the tool result.

### The two models' formats

Gemma 4 wraps strings in the special token `<|"|>` and leaves keys unquoted, so it is not JSON, and `families/gemma4.py` has its own small parser:

```
<|tool_call>call:get_current_time{timezone:<|"|>Asia/Taipei<|"|>}<tool_call|><|tool_response>
```

After writing a tool call, Gemma stops at `<|tool_response>` to wait for the harness to add the result; after an answer it stops at `<turn|>`. These two stop tokens line up with `stop_reason: tool_use` and `end_turn` in the Anthropic Messages API.

Qwen3.5 uses XML, and every value is plain text, typed by the tool's schema (as vLLM's `qwen3_coder` parser does):

```
<tool_call>
<function=get_current_time>
<parameter=timezone>
Asia/Taipei
</parameter>
</function>
</tool_call><|im_end|>
```

Qwen stops at `<|im_end|>` both after an answer and for a tool, so the harness decides whether to end by whether parsing found a tool call, not by the stop token. The tool result goes into a new `user` message (`<tool_response>…</tool_response>`) instead of the same model turn as with Gemma.

## Tests

```bash
uv run pytest                                          # about 25 s; loads no weights and touches no network
RUN_MODEL_TESTS=1 uv run pytest tests/test_real_model.py   # loads the real Gemma (with its MTP drafter) and Qwen, about 40 s; add -k Qwen for one
```

Most tests use the fake model in `tests/fakes.py`: it replays scripted output one token at a time, with the same interface as `LocalLLM`, so the harness and the page run from start to finish.

| File | What it tests |
|---|---|
| `test_gemma4.py` | Gemma 4's parser: splitting thinking, tool calls and the reply; `<\|"\|>` strings, nested arguments, all kinds of format errors, in both languages |
| `test_qwen3_5.py` | Qwen3.5's parser: XML tool calls, types from the schema, thinking, all kinds of format errors; a whole harness run in Qwen's format |
| `test_families.py` | Every family explains the tokens it uses, in both languages and for the same tokens; `model_type` finds the family |
| `test_tools.py` | Schema validation and defaults, the calculator's AST allowlist (rejecting `__import__` and the like), timezones, Wikipedia (with fake HTTP); errors in both languages |
| `test_llm.py` | The hand-written decoding loop: KV cache, stop tokens, the token limit, interruption, channel switches, Chinese characters split across tokens; the candidates and probabilities of temperature / top-k / top-p; MTP with a scripted fake drafter: kept and rejected guesses, the extra token when all hold, how far the KV cache is cut back, the drafter never seeing dropped guesses, and a check that guesses do not change the distribution when sampling |
| `test_agent.py` | The harness: event order, tool results back in the context, validation errors, repeated calls, truncated results, every stop reason, Allow / Reject / Stop while waiting; the model reads errors in the turn's language, and the system prompt is the same in both |
| `test_render.py` | Every prefix of a run's events renders in both languages (the page redraws every 0.5 s mid-run), HTML tags pair up, the loop diagram is valid SVG, model output is always escaped, special token popovers and lists; MTP's underlines, popovers, counts and drafter card; no Chinese left on an English page |
| `test_app.py` | Drives the whole page with Streamlit's `AppTest`: sending a question, the example buttons, sidebar settings reaching the model, multi-turn chat, the Context and Model tabs, Allow / Reject / Stop, a new chat, resetting the settings, two tabs never running at once, switching models, How it works linking to GitHub Pages in the page's language; MTP off by default, the drafter loaded only when it is switched on, and switched off and disabled for Qwen, which has none; an English browser gets English, the last pick beats the browser's language, switching languages keeps the conversation and the settings, and the model's errors follow the language |
| `test_chat_template.py` | With each model's real tokenizer and chat template (only when cached locally): every special token is a single token and has a note, tool calls the template writes into the history parse back; a harness run with the real template checks that what the step card marks as new is exactly the tool result (two arguments, thinking mode, text before a tool call) |
| `test_guide.py` | Every excerpt and line number on How it works resolves, the event table points at the line that really sends each event, HTML tags pair up; each paragraph points at the same code and sections in both languages, with no Chinese left in the English one; the built pages have light and dark palettes and loop diagrams and name their commit, and the root page sends readers to the right language |
| `test_real_model.py` | Skipped by default. For each model: greedy decoding matches `model.generate()`, and the real model calls the calculator and answers; for Gemma with MTP, greedy output is exactly the same as without it, and the drafter's guesses really are kept |

## Known limitations

- Speed: on an Apple M5 (32 GB), Gemma runs at about 7–13 tok/s and Qwen3.5 0.8B at about 20–25 tok/s. In thinking mode a step often has two or three hundred thinking tokens, about 30 seconds of waiting. With MTP on, easily guessed steps such as tool calls are about 3× faster, and long Chinese text barely changes (see the table under MTP speculative decoding).
- With long replies in the page, MTP speeds things up much less than in `llm.py` alone (English code: 8 → 29 tok/s alone, 9 → 12 tok/s in the page), for reasons not yet pinned down. Perhaps the page redrawing every 0.5 s and the browser competing with the model for the same machine slow MTP down more than plain decoding, since each MTP round has several small drafter forward passes.
- Only Gemma 4 E2B can use MTP: `transformers` supports a drafter reading the model's KV cache this way only for Gemma 4, and Qwen3.5 has no matching drafter.
- With MTP on, the model computes 7 positions at once, which gives very slightly different numbers from computing 1; when two tokens are almost equally likely, greedy decoding may now and then pick the other one.
- 18 of Qwen3.5's 24 layers are Gated DeltaNet. transformers has two kernels to speed them up, `flash-linear-attention` (Triton, Linux wheels only) and `causal-conv1d` (CUDA), and neither installs on a Mac, so the PyTorch reference implementation runs instead, with the same results, only slower; on MPS and CPU, `llm.py` silences transformers' notice about that fallback. On Linux with CUDA the notice shows as usual, and installing those two packages speeds things up.
- On MPS, transformers hangs loading Qwen3.5's weights with several threads, so on MPS `llm.py` loads with one thread (`HF_DEACTIVATE_ASYNC_LOAD=1`), which does not make either model load slower.
- Qwen3.5 0.8B has thinking off by default; on, it easily keeps thinking without stopping, and lowering Max tokens per step helps. It ships no generation_config.json, so its recommended decoding settings are in `families/qwen3_5.py` (the presence penalty it recommends is not implemented here).
- Only one model is kept in memory. If two tabs pick different models, using them in turn keeps reloading; it is best for every tab to use the same model.
- There is one copy of the model, so only one agent runs at a time and other tabs wait. Reloading or closing the tab during a run does not stop the agent; other tabs (including the reloaded one) show it is still running and offer a Stop that agent button.
- With Ask me before running a tool on, if no one answers within 5 minutes the harness takes it as a rejection and stops the loop, with "confirmation timed out" as the stop reason.
- Token popovers appear only once the step finishes generating; while it generates there are only the heat colors (special tokens still explain themselves), so each redraw does not send a lot of HTML. The popovers are pure CSS, and one for a token too near the bottom of the timeline may be cut off.
- Special token popovers are placed next to the token with CSS anchor positioning, so a scrolling box does not cut them off; browsers without it open them right under the token, where they may be cut off near the bottom of the box.
- Both models are small: they may answer in Simplified Chinese, repeat the same search, answer half the question or get their sums wrong while thinking. Asked in English about two cities' times at once, Gemma calls the tool twice and then usually answers in Chinese, or stops right after the tool results without a reply; asked about one city, it answers in English. All of that is material to look at.
- The Wikipedia tools need the network.
- The `MODEL_ID` environment variable picks the model selected at first; it may also name a checkpoint the menu does not list, as long as it belongs to a supported family (Gemma 4 or Qwen3.5); another family needs one added first, as in Supporting more models.

## License

This project's code is licensed under the [Apache License 2.0](LICENSE). The Gemma 4 weights and its MTP drafter are released by Google under Apache 2.0 (see the [model card](https://huggingface.co/google/gemma-4-E2B-it) and the [drafter's card](https://huggingface.co/google/gemma-4-E2B-it-assistant)), and the Qwen3.5 weights by Alibaba's Qwen team under Apache 2.0 (see the [model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)); none of them are included in this project.

About the use of AI: the code was written with the help of Claude Code, run on this machine with gemma-4-E2B-it (and its MTP drafter gemma-4-E2B-it-assistant) and Qwen3.5-0.8B, and checked in a browser.
