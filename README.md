<img src="assets/logo.svg" alt="" width="72">

# Agent Loop Visualizer

English | [繁體中文](README.zh-TW.md)

![The top bar links to the How it works page, the switch at the top right changes the language, and the sidebar picks the model; on the left is the chat, with the sample files to ask about above the input box, on the right the loop diagram and the tokens of step 1. The pointer rests on the special token <|tool_call>, whose popover says what it is for and shows its probability, its rank and the top 5 candidates.](docs/screenshot-en.png)

This project is for learning: it shows how an agent loop works underneath, in a way more intuitive and visual than reading about it. Behind an agent's chat box runs a loop: step after step, the model proposes an action and the harness carries it out, until it stops. A chat app shows only the question and the answer; this page draws every step as it happens, down to each token the model writes, so the whole mechanism is laid out in front of you. Each part of the page matches a section of the Unit 01 reading of [Agent Atlas](https://cyyeh.github.io/agent-atlas-learning/) (in Traditional Chinese), the course this project belongs to, and [Experiments to try](#experiments-to-try) turns those ideas into things you can try yourself.

- Models: pick [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) or [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) in the sidebar; they run on your machine with Hugging Face `transformers`
- MTP speculative decoding: for Gemma 4 E2B you can switch it on in the sidebar (it is off by default); it uses [google/gemma-4-E2B-it-assistant](https://huggingface.co/google/gemma-4-E2B-it-assistant) as the drafter, so one forward pass produces several tokens
- Files: attach images and videos (both models) and sound (Gemma 4 E2B, which also takes a recording from the microphone) with the + in the input box, or pick one of the sample files above it (a receipt, a recording, a video, sales data) and ask your own question about it; each becomes a run of tokens in the context, and the page shows them
- Writing code (optional): Let the model write code in the sidebar gives the model a `run_python` tool, which runs its Python in a Docker sandbox, reads the data files you attach (CSV, TSV, JSON, text, Markdown, Excel) and draws charts with matplotlib, which appear in the chat. Docker is needed for this alone
- Languages: the page and the How it works guide are in English and Traditional Chinese. They start in your browser's language, the switch at the top right changes it, and the language you pick is remembered for your next visit
- License: [Apache License 2.0](LICENSE)

On the left is a chat like ChatGPT's; on the right the agent loop is drawn step by step: the model reads the context and proposes an action, the harness carries it out, and the result goes back into the context, until it stops. The model runs on your own computer, so every token's probability, the candidate tokens, the special tokens, the prefill and the generation speed are all in view. The models run the same harness, so switching models changes only their text formats, which makes them easy to compare.

[How it works](https://cyyeh.github.io/agent-loop-visualizer/en/) maps each step of one turn, from sending a question to the reply, to the code that does it, with file names, line numbers and excerpts. It lives on GitHub Pages; How it works at the top of the app opens it in a new tab, in the language the app is in.

## What the page shows

| On the page | What you see | Unit 01 reading (in Chinese) |
|---|---|---|
| Loop diagram | Which box you are in (context, model, harness, environment, stop) and the arrow that is flowing | §4 An agent is a model in a loop |
| Context | The whole string the model reads in each step: the system prompt, the tool schemas, the conversation and the tool result added after the last step. Hovering over a dark special token says what it is for. An attached image, clip or video is the run of tokens it expands into, drawn as one chip with its count (`<\|image\|>` ×266), and the step card says how many tokens each kind took | §1 The context has a limit, §3 Tool definitions are tokens too |
| Model | Every generated token, darker the less likely the chosen token was. Hovering over a token shows its probability, its rank, the top 5 candidates and which ones top-k / top-p cut; a special token also says what it is for. With MTP on, tokens the drafter guessed right are underlined, the popover says what the drafter guessed and whether it was kept, and the step card shows how many were kept and how many forward passes the model saved | §1 An LLM samples from a conditional probability distribution; decoding settings |
| Harness | The tool calls parsed from the model's own format (Gemma 4 or Qwen3.5), the schema check, warnings about repeated calls, and your Allow or Reject | §3 The model never touches the environment |
| Environment | What the tool really returned and how long it took; for `run_python`, the code is written out above and what it printed, or its traceback, below | §2 The environment and observable state |
| Stop | Why it stopped and at which token. Gemma stops at `<turn\|>` after an answer (like end_turn) and at `<\|tool_response>` when it wants a tool (like tool_use); Qwen stops at `<\|im_end\|>` either way. Then there are the step limit, the token limit and the user pressing stop | §5 When to stop |
| Model tab | The parameter breakdown, the number and kinds of layers, attention heads, the context limit, what the model reads (text, images, sound, video), the MTP drafter, what each decoding setting does, each step's prefill time and tok/s (with MTP, also how many guesses were kept and how many decoding forward passes ran), the id and purpose of every special token of the tokenizer (24 for Gemma, 33 for Qwen), and every file in the model's folder in `models/`: its size, what it is for, what it holds for that model and the values of its key fields | §1, §7 An agent is a system |
| How it works | The 17 stages of one turn (including how an image becomes tokens and how MTP guesses and checks), each with the code that does it and its line numbers, in English and in Chinese. It is on GitHub Pages; the line numbers are looked up in the source when the page is built, and it is rebuilt on every push to main, so it follows the code | All of it |

## Quick start

You need:

- macOS (Apple Silicon, using MPS), Linux with CUDA, or a CPU alone (very slow)
- Disk space for the weights: about 10 GB for Gemma 4 E2B, 1.8 GB for Qwen3.5 0.8B, and about 190 MB more for Gemma's drafter if you want MTP
- Memory: with MPS or CUDA, 16 GB or more for Gemma (the bfloat16 weights take about 10 GB); on a CPU it loads in float32, about 20.5 GB, so 24 GB or more. Qwen3.5 0.8B is much smaller and runs on an 8 GB machine. Only one model is loaded at a time; switching models lets the old one go
- Linux + CUDA: the pinned torch is built for CUDA 13, so the CUDA Version at the top right of `nvidia-smi` must be 13.0 or newer. With an older driver torch does not see the GPU and runs on the CPU; the top of the sidebar shows the device in use
- [uv](https://docs.astral.sh/uv/) 0.13.4 or newer (it installs Python 3.12 for you). Older versions still trust eMudhra's compromised emSign certificates ([the announcement](https://discuss.python.org/t/109576)), so `uv sync` refuses to run with them; `uv self update` (or `brew upgrade uv`) updates it
- A network connection: to download the models the first time, and for the Wikipedia tools
- Optional: [Docker](https://docs.docker.com/get-started/get-docker/) (Docker Desktop, OrbStack or Colima), only to let the model write code

```bash
uv sync
uv run hf download google/gemma-4-E2B-it --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 --cache-dir models   # download the weights first (about 10 GB); otherwise they download the first time the model is picked
uv run hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --cache-dir models       # about 1.8 GB
uv run hf download google/gemma-4-E2B-it-assistant --revision 2d874ef7d29f9a30599a1e4b3c1cbc9595f005df --cache-dir models   # the MTP drafter, about 190 MB; otherwise it downloads the first time MTP is switched on
uv run streamlit run streamlit_app.py      # loads Gemma first; MODEL_ID=Qwen/Qwen3.5-0.8B loads Qwen first
docker build -t agent-loop-sandbox sandbox # optional: the sandbox for Let the model write code (about 400 MB); again whenever sandbox/Dockerfile changes
```

Your browser opens http://localhost:8501, in its own language (English unless the browser is set to Chinese); the 中文 | English switch at the top right changes it, and the next visit starts in the one you picked. Loading Gemma and warming it up takes about 10–20 seconds the first time, Qwen a few seconds; reloading the page does not load it again. The Model menu at the top of the sidebar switches models; switching starts a new conversation, resets the decoding settings to the new model's recommendations and turns MTP off. MTP speculative decoding in the sidebar loads the drafter the first time it is switched on (in under a second), and it stays in memory after that.

Let the model write code in the sidebar is off by default, and while it is off the page never looks for Docker. Switched on, it checks for Docker and the `agent-loop-sandbox` image, and if either is missing, or the image was built from an older `sandbox/Dockerfile`, it says in the sidebar how to install Docker or build the image; the model gets `run_python` only once both are there, and the examples that need Python appear. Files you attach are saved in a folder per conversation under `~/.cache/agent-loop-visualizer/uploads/` (New chat removes it), which the sandbox sees, read-only, as `/data`.

The models download into `models/` in the project folder, which git ignores, not into the `~/.cache/huggingface/hub` other projects share; deleting the folder frees the space. Each downloads at a fixed revision, the commit in `REVISIONS` in `families/__init__.py` (the `--revision` above), so whenever you download it you get the files this page was written and tested against, the same on every operating system. To keep them somewhere else, set `AGENT_LOOP_MODELS` to that folder (and give the downloads above the same `--cache-dir`); `AGENT_LOOP_MODELS=$HOME/.cache/huggingface/hub` uses the models already downloaded there. `HF_HUB_CACHE` does not move them, since it is often set once for every project on a machine.

To run the model on a vLLM server instead, see [Running the model on a vLLM server](#running-the-model-on-a-vllm-server).

> A 401 while downloading, though the model is public? Most likely the Hugging Face token saved on your machine has expired. Log in again (`uv run hf auth login`), or add `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` to send no token.

## Experiments to try

1. **Just a chatbot**: switch off Use tools in the sidebar and ask "What time is it in Taipei right now?". The model sees no tools, so it can only guess or admit it does not know. Switch tools back on, ask again, and compare the two in the Context tab.
2. **Sampling**: ask the same question with temperature 0 and with 1.5, and watch how the token colors and the candidates left in the popover change.
3. **Stop conditions**: set Max steps to 1 and ask something that needs a tool; the stop reason becomes "step limit reached".
4. **Where the human sits**: switch on Ask me before running a tool, then press Reject. It is the harness that blocks the call; the model only sees an error observation in the next step's context. Compare the mail deletion incident in §5 of the reading: a "confirm first" written in the prompt can get compressed away, one written into the harness cannot.
5. **Reasoning is tokens too**: switch on Thinking mode, and italic thinking tokens appear before the tool call; in the Context tab, the system turn starts with an extra `<|think|>`.
6. **The loop goes in circles**: ask "Who proposed the ReAct paper? Look it up on Wikipedia." E2B often repeats the same search, and the harness flags "exactly the same call as before"; compare the error analysis of ReAct in §4 of the reading.
7. **Switch models, keep the harness**: ask Gemma and Qwen the same question and compare them in the Context tab: the turn markers (`<|turn>` against `<|im_start|>`), where the tool schemas go, how a tool call is written (`call:name{…}` against XML) and where the tool result goes back all differ, yet the harness, environment and stop sections of the step cards are exactly the same.
8. **MTP is fast only when it guesses right**: with Gemma, switch on MTP speculative decoding, press "What time is it in Taipei right now?", type "Write a Python function to test for primes." (with Let the model write code off), then ask "用三段話介紹台北。" (introduce Taipei in three paragraphs, in Chinese). Compare each step's tok/s and underlines: the drafter guesses almost every token of a tool call, about half of English code and only about a tenth of long Chinese text. Set temperature to 0 and run with MTP on and off: the output is exactly the same, because the drafter only decides how many forward passes run, not what gets written.

9. **An image is tokens too**: press the receipt example. In the step card's context section and in the Context tab, the image is a run of a few hundred `<|image|>` (Gemma) or `<|image_pad|>` (Qwen): what it costs in the context. The items add up to 360, but the receipt says 370; see whether the model adds them up itself or calls a tool.
10. **The context decides whether the model listens**: with Gemma, press the recording example on the Chinese page, and it writes the English sentence down and translates it. Ask "Transcribe this recording." on the English page with Use tools on, and it often says it cannot hear audio, sometimes after writing the very words into a tool call; switch Use tools off and ask again, and it transcribes it. The model and the audio tokens are the same; only the context differs.
11. **Code in a sandbox**: switch on Let the model write code (Docker needed) and press the sales data example. The note at the end of your message gives the file's path and first line; the model's code is written out in the step card, and what the sandbox printed, or its traceback, comes back as the observation, so a failing program can be fixed in the next step. Try Qwen too: it usually gets as far as the totals per month, then breaks its own code.
12. **A chart is for you, not the model**: with Let the model write code on, pick sales.csv among the sample files above the input box and ask for a chart of the revenue per month. The chart appears in the chat and under the step that drew it, but the observation the model reads is only `{"figures": ["/data/figure-1.png"]}`: it describes a chart it has never seen.

## Architecture

```
streamlit_app.py   layout, session state, model menu, language switch, background thread; redraws every 0.5 s while a run is live
guide.py           the How it works page: its content in both languages, line lookups, and the static HTML for GitHub Pages
render.py          turns the event list into HTML: the loop diagram, step cards, token heat and popovers, the context view
agent.py           the harness: build the prompt → generate → parse → validate → (ask you) → run the tool → back into the context → stop?
simple_backend.py  local inference with a hand-written decoding loop; MTP's drafting and checking are here too
vllm_backend.py    the model on a vLLM server (VLLM_URL): the prompt rendered and tokenized here, the token ids to the server, what it can tell about each token back; media, sampling masks and MTP's guesses on a server started with the flags for them
backend.py         the Backend that the harness and the page run the model through: render, encode, generate, and the fields of a token event; simple_backend.py is the one the app runs
i18n.py            the two languages: pick() chooses the text for one, from_locale() picks the default from the browser's language
families/          everything that goes with the model: how each model family writes turns, thinking and tool calls, which tokens stop it, what each means
  __init__.py      the Family definition, the model menu MODELS, the MTP drafters DRAFTERS, finding a family by config.json's model_type
  gemma4.py        Gemma 4: the output parser and notes on its 24 special tokens
  qwen3_5.py       Qwen3.5: the output parser and notes on its 33 special tokens
tools.py           tools: the schema (for the model) plus the Python function (for the harness to run); run_python's Docker sandbox
media.py           attachments: saving them, writing them into the user's message, decoding images, sound and video for the processor
sandbox/           run_python's sandbox: the Dockerfile (Python 3.12 with numpy, pandas, openpyxl, tabulate and matplotlib), and the matplotlib backend and settings that send the charts back
assets/            the page's styles and the guide's script; logo.svg; examples/, the files the examples attach and the sample files above the input box
tests/             pytest tests; see Tests below
models/            the downloaded models (git ignores it; AGENT_LOOP_MODELS moves it)
```

The agent runs in a background thread and appends an event at every stage; the UI only draws the event list. So when you press Stop or Allow and Streamlit reruns the script, the loop in the background is not interrupted.

How it works is a static page: `uv run python guide.py _site` writes `_site/en/index.html` and `_site/zh/index.html` (styles and a small script inlined, so they open straight in a browser), and `_site/index.html`, which sends a reader to the language they last picked on the page's switch, or else their browser's. On a push to main, `.github/workflows/pages.yml` runs `tests/test_guide.py`, then does the same and publishes it to GitHub Pages; if the tests fail, the page already up stays as it was.

The events of one run, in order: `run_start`, then for each step `step_start` → `context` → `prefill` → `token` ×N → `model_end` → `parsed`, with a tool call also `tool_call` → (`approval_request` → `approval`) → `tool_start` → `tool_result`, and finally `run_end`, after an `error` if something failed: its `message` is the exception's, and a backend's error (a vLLM server's, say) also comes in both languages (`text`), so the page shows the one it is in. `context` carries the prompt as the model reads it, with each attachment expanded into its tokens, and how many tokens each kind of media took (`media_tokens`). `token` carries the token and, when the backend knows them, its probability, its rank, the top 5 candidates and what MTP's drafter guessed (`backend.TOKEN_DETAILS`): `simple_backend.py` knows them all, and the page leaves out whatever a backend does not report rather than showing a number nobody measured. A backend that learns which tokens were MTP's guesses only when the step ends (a vLLM server) says so in `model_end`'s MTP counts (`kept_at`, `turned_down_at`), and the page marks those tokens then. `tool_result` carries the observation the model reads and, apart from it, the images the tool made for you (`images`: the paths of `run_python`'s charts), which the model never reads.

### Two languages

Every piece of text on the page is written in both languages side by side in the code: `pick(lang, "中文", "English")` in code, and `{"zh": …, "en": …}` in data (the special token notes, the stop reasons, the paragraphs of How it works). Every function in `render.py` takes a `lang`, and the events themselves have no language, so switching the language redraws the same events in the other one, finished turns included. The page starts in the language last picked, kept in a cookie, or else in the browser's `navigator.language` (anything starting with `zh` is Chinese, the rest English); picking one writes the cookie with a small script, because Streamlit can read cookies but not write them.

What the model reads from the harness (wrong arguments, a failed tool, your refusal) is worded in the language the page was in when the turn started, because it becomes the observation word for word; the system prompt is the same in both, so the model's behavior does not depend on the page's language.

### Why a hand-written decoding loop

`model.generate()` does not hand out the probability distribution at each step, so `simple_backend.py` runs the loop itself: one forward pass per token (with a KV cache), then temperature, top-k and top-p in turn, then a sample. So every token can report its probability and rank in the model's own distribution, its probability when sampled, how many candidates were left after top-k / top-p, and the top 5 candidates. In greedy mode the output matches `generate()`, as measured; of two equally likely tokens (in bfloat16, logits tie more often than one would think) it takes the lower id, as `generate()`'s argmax does.

### MTP speculative decoding

Decoding runs a full forward pass for every token, and the time goes mostly into reading the weights from memory; computing a few more positions in the same pass is hardly slower (Gemma 4 E2B on an Apple M5: about 56 ms for 1 position, about 70 ms for 7). So if the next few tokens are guessed cheaply first, the model can check them all in one forward pass. Google trained such a drafter for Gemma 4 (MTP, multi-token prediction), and `transformers`' `generate(assistant_model=…)` uses it, but that again hides the distribution at each step, so `simple_backend.py` wires it into its own decoding loop:

1. **Guess** (`_draft`): the drafter `gemma-4-E2B-it-assistant` has about 78M parameters and 4 layers. It has no KV cache of its own and reads the model's keys and values directly (the last 20 layers of Gemma 4 E2B already share those two earlier layers computed); its input is the last token's embedding next to the model's last-layer hidden state. It makes 6 greedy guesses at a time (the checkpoint's `num_assistant_tokens`).
2. **Check** (`_drafted`): the model runs one forward pass over the token just settled plus the 6 guesses, gets back the distributions at 7 positions and hands them to `_sample()` in order. A kept guess needs no forward pass of its own; the first guess turned down is replaced by the model's own pick at that position, everything after it is dropped, and the KV cache is cut back with `crop()`; when all 6 hold, the last position gives one more token.
3. **Same output**: in greedy decoding a guess is kept only when it is the top token, so the output is the same as without MTP (`tests/test_real_model.py` compares them with the real model). When sampling, a guess is kept with its probability after temperature, top-k and top-p, and otherwise the token is drawn from the other candidates, so every token comes out exactly as often as without MTP (speculative sampling; `tests/test_simple_backend.py` checks it over twenty thousand samples).

How many guesses are kept depends a lot on the content. Measured in the page with Gemma's default decoding settings (Apple M5):

| What the step writes | Without MTP | With MTP | Guesses kept |
|---|---|---|---|
| A tool call (step 1 of "台北現在幾點？紐約呢？", "What time is it in Taipei now? And New York?") | 12.6 tok/s | 40.9 tok/s | 34 / 35 |
| The answer from the tool result (step 2 of the same question) | 13.0 tok/s | 22.6 tok/s | 42 / 106 |
| English code, about 750 tokens | 8.8 tok/s | 11.6 tok/s (`simple_backend.py` alone: 8.1 → 29.0) | 51% |
| Long Chinese text ("用三段話介紹台北。") | 10.3 tok/s | 10.3 tok/s | 7% |

### Images, sound and video

Which of them a model reads is found out from the checkpoint when it loads: an encoder in its config and a part of its processor to feed it. Gemma 4 E2B reads images, sound and video; Qwen3.5 0.8B reads images and video. The input box takes only what the model reads:

| Kind | Formats | Gemma 4 E2B | Qwen3.5 0.8B |
|---|---|---|---|
| Image | png, jpg, jpeg, webp, gif, bmp | up to 280 tokens | depends on the size (about 300 for 640×480) |
| Sound | wav, mp3, m4a, flac, ogg | one token per 40 ms, the first 30 seconds | — (with Let the model write code, a file for `run_python`) |
| Video | mp4, mov, webm, mkv | 32 frames (all of a shorter video), each after its time as `mm:ss`, up to 70 tokens each | 2 frames a second, in pairs, each pair after its time |
| Data | csv, tsv, json, txt, md, xlsx | with Let the model write code, for `run_python` | the same |

The chat template writes one placeholder where each file goes (`<|image|>` for Gemma, `<|vision_start|><|image_pad|><|vision_end|>` for Qwen), and `LocalLLM.encode()` hands the processor every file in the conversation, which expands each placeholder into the file's tokens; the Context tab shows that expanded string, since it is what the model reads. `media.py` decodes the files first: images to RGB, turned upright by their EXIF; sound with PyAV to mono at the feature extractor's 16 kHz; video to at most 64 evenly spaced frames, 768 pixels on the longer side (transformers' own video loading needs torchcodec). Each file is decoded once. In the prefill, images and videos go through the vision encoder first, as `generate()` does it (handed to the forward pass as they are, videos lose their frame counts in transformers 5.20's Gemma 4), and Qwen3.5's position offset for images is cleared before every prefill. After the prefill decoding is the same as for text, and `tests/test_real_model.py` checks that greedy output with an image, a video and a recording matches `generate()`.

### Writing code in a Docker sandbox

With Let the model write code on and Docker ready, the model gets one more tool, `run_python`. Each call starts a new container from the image `sandbox/Dockerfile` builds: no network, a read-only file system except a small `/tmp`, 512 MB of memory, one CPU, at most 64 processes, no Linux capabilities and your own user, never root (on Linux the mounted folder keeps its permissions, which might not let another user in), with the conversation's folder read-only at `/data`; the code goes in on stdin, and the container is deleted afterwards. Code that runs past 30 seconds is stopped; a nonzero exit, using too much memory or printing nothing (and drawing nothing) comes back to the model as an error, in the turn's language, with the traceback, so it can fix the code in the next step.

matplotlib's charts come back too, although the container cannot write anywhere the page can read: the image sets matplotlib's backend to `sandbox/agent_loop_figures.py`, which turns each figure into a PNG at `plt.show()` or when the program ends and, as it exits, writes them after what the code printed, behind a marker line that `run_python` passes in. `run_python` cuts them off what the model reads (at most four), saves them in the conversation's folder as `figure-1.png`, `figure-2.png`… and returns them apart from the observation: the model reads only `{"stdout": …, "figures": ["/data/figure-1.png"]}`, and the page shows the images in the chat and under the step. A label in Chinese is drawn with WenQuanYi Micro Hei, the font the image installs for it. Importing matplotlib makes a call about half a second slower (0.8 s against 0.2 s for plain Python).

The image installs pip 26.2.2 first, from its wheel by hash, and everything else with it: an older pip also trusts its own copy of certifi, which still has the compromised emSign certificates. The page reads the `agent-loop-sandbox.version` label of the image, and when it is older than `sandbox/Dockerfile` the sidebar asks for a rebuild instead of offering `run_python`.

The model cannot see into a data file, so the harness adds a note to the user's message: where `run_python` finds each file, and the first line of a text file (for a CSV, its column names). Without that line a small model guesses columns in its code, a `Date` that is not there, and fails; with a few rows of data as well, Gemma copied the rows into its code and answered from them alone. `tabulate` is in the image because `DataFrame.to_markdown()`, which small models like to print tables with, needs it.

### Running the model on a vLLM server

The model can also run on a [vLLM](https://docs.vllm.ai/) server, on this machine or on a Linux machine with a GPU, while the page stays on yours. Set `VLLM_URL` to where the server listens:

```bash
# Where vLLM runs (vLLM 0.32 or newer, on a Linux machine with a GPU). --revision is the commit in REVISIONS, so the
# server and the page read the same tokenizer and chat template. The other flags let the page see more (below).
vllm serve Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --enable-scale-out
# Or Gemma 4 E2B with its MTP drafter, saying which tokens were guesses:
vllm serve google/gemma-4-E2B-it --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 --enable-scale-out \
  --speculative-config '{"method": "mtp", "model": "google/gemma-4-E2B-it-assistant", "revision": "2d874ef7d29f9a30599a1e4b3c1cbc9595f005df", "num_speculative_tokens": 6}' \
  --per-request-spec-decode-metrics detailed
# Or with sampling masks, which show how many candidates were left in place of the model's own probabilities (below):
vllm serve Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --enable-scale-out \
  --return-sampling-mask --logprobs-mode processed_logprobs

# Where the page runs:
VLLM_URL=http://localhost:8000 uv run streamlit run streamlit_app.py
```

The Model menu then lists what the server serves. If the server was started with `--api-key`, give the page the same key in `VLLM_API_KEY` (the variable `vllm serve` reads it from). If the server cannot hold the model's whole context in its KV cache, lower it with `--max-model-len`; the page's context limit follows the server's.

`vllm_backend.py` is a second `Backend` (`backend.py`) next to `simple_backend.py`; the harness, the tools and the page do not change. The prompt stays exact: the page downloads the model's config, tokenizer and chat template into `models/` (not the weights), renders and tokenizes the prompt itself, and sends the server the token ids, so the server reads exactly what the Context tab shows. Each token comes back as its id with its log probability and those of the most likely tokens at its position. With vLLM's default `--logprobs-mode raw_logprobs` those are the model's own distribution, as in `simple_backend.py`, and give the popover's model probability and top 5 candidates. How much more the page sees depends on the server's flags; at load, the page asks the server, with one-token requests, which of them it was started with:

| Server flags | Where the steps go | What the page shows |
|---|---|---|
| none | `/v1/completions` | the probability, the top 5, the rank only when the token is one of them; text only |
| `--enable-scale-out` | `/inference/v1/generate` | also every token's rank, and the images, sound and video the server takes |
| `--return-sampling-mask --logprobs-mode processed_logprobs` too | the same | the candidates left and the sampling probability, in place of the model probability, the heat and the top 5: vLLM sends sampling masks only with processed log probabilities, and then refuses temperature 0 and top-k 0, so the sidebar starts at 0.05 and 1 |
| `--per-request-spec-decode-metrics summary` / `detailed` | either | MTP's counts in each step; with `detailed`, the underlines on the guesses too, once the step is done |

What the page leaves out it does not make up (`backend.TOKEN_DETAILS`). In detail:

- **`--logprobs-mode`** decides what the numbers are, and no route says which one the server runs, so at load the page asks for one token at temperature 1 with top-k 2 and the 5 most likely tokens: in a processed mode the 3 that top-k cut come back as -9999, and log probabilities, unlike logits, add up to at most 1. With `processed_logprobs` (which `--return-sampling-mask` needs; vLLM refuses to start without it, or without its Model Runner V2, which it picks by default where it can), a token's log probability is its probability in the distribution it was drawn from, after temperature, top-k and top-p, so the popover shows it as the sampling probability, with the rank, and shows no model probability, heat or top 5. How many candidates were left comes from the sampling mask, or, without one, shows when top-k / top-p cut some of the most likely tokens listed. With `raw_logits` or `processed_logits`, the numbers are not probabilities, and the page shows none of them as one. Which of the top 5 top-k / top-p cut (gray in `simple_backend.py`'s popover) shows in no mode: the model's own probabilities and what the sampler did never come back together. The Model tab says which mode the server runs.
- **Greedy decoding** (temperature 0) needs no server to say that one candidate was left and drawn with probability 1, so the popover shows that on any server that takes it. A server started with `--return-sampling-mask` does not: vLLM refuses temperature 0 and top-k 0 there, on every route, so the sidebar's sliders stop at 0.05 and 1, and a recommended setting below them moves up to them. Without `--enable-scale-out` no response shows the masks, so at load the page also asks such a server (one with processed log probabilities) for a greedy token, and a refusal says the same.
- **Images, sound and video** go next to the placeholders the chat template wrote, as the server's `content_parts`: an image as PNG, turned upright by its EXIF; sound as WAV, decoded here; a video as the file itself, whose frames the server picks. The server expands each placeholder into the file's tokens. So that the Context tab shows the prompt as the server reads it, the page asks it for that expansion first, with a request that generates one token; if the server expands differently when generating, the step is an error. At load the page sends the server a tiny image, clip and video to see which kinds it takes; a kind it refuses (or every kind, without `--enable-scale-out`) cannot be attached, the line above the input box says so, and the Model tab lists why.
- **MTP** is the server's (`--speculative-config`), so the sidebar's switch is off and disabled. With `--per-request-spec-decode-metrics summary` (which vLLM takes only with `--speculative-config`), each step says how many guesses were kept and how many decoding forward passes ran. vLLM counts a verify step's kept guesses before it cuts the step at a stop token, so when the stop token is one of them, the guesses after it count too. With `detailed` it also says how many guesses each verify step kept, from which the page marks the guesses: the first token comes from the prefill, then each verify step gives the guesses it kept and one token of the model's own. vLLM does not count a step without guesses, so the page marks the tokens only when the counts account for every one of them, and only once the step is done, counting as kept only the guesses up to the stop token, as `simple_backend.py` does; what a turned-down guess was, the server does not say.
- **Prefill** is the time to the first token, the network included. vLLM keeps the KV cache of what a prompt shares with the last one (prefix caching), so from step 2 on it reads only the new part, where `simple_backend.py` reads the whole prompt every step. With media, the expansion request before it has already read most of the prompt.
- **The Model tab** names the server and its version; the specs come from `config.json` and the context limit from the server. The device, the precision and the parameter count are the server's and are not shown, and the folder card lists the weights as on the server only.
- **Not identical to `generate()`**: vLLM batches requests and has kernels of its own, so greedy output need not match `transformers`' token for token, as `simple_backend.py`'s does.

[How it works](https://cyyeh.github.io/agent-loop-visualizer/en/) has a section on the `Backend` interface that sets the two backends side by side, with their code.

### Supporting more models

The harness (`agent.py`), the decoding loop (`simple_backend.py`) and the page (`render.py`) are written once, without a single "if Gemma" branch. Whatever differs between models is gathered in one `Family` object in `families/`:

- Reading the output: `parse` (splits the output into thinking, tool calls and the reply), `thought_prefix`, `channel_switch` (which tokens switch to thinking, a tool call or the reply), `stop_tokens`
- Writing the conversation back: `reasoning_key` (the field the thinking is kept in, so the chat template can read it), `text_with_calls` (whether text in front of a tool call is kept)
- The page: `user_turn`, `byte_level` (whether the vocabulary spells bytes as characters, as Qwen's does, so a token popover shows 如果 instead of `å¦Ĥæŀľ`), a note on each special token, `file_notes` and `drafter_file_notes` (what each file of a checkpoint's folder holds, and the fields whose values the Model tab shows), the help text for the sidebar and the Model tab, and the recommended decoding settings when there is no generation_config.json; each in English and in Chinese
- Media: `processor_kwargs`, settings for the processor (Qwen caps the pixels of each video frame); which kinds a model reads comes from the checkpoint, not the family

To add a model:

1. Another checkpoint of a family that is already here (a bigger Gemma 4 or Qwen3.5, say): add a line to `MODELS` in `families/__init__.py`, and one to `REVISIONS` with the commit to download it at. Other Gemma 4 checkpoints have MTP drafters too (`google/gemma-4-*-it-assistant`); add a line to `DRAFTERS` (and the drafter's commit to `REVISIONS`) and the sidebar's MTP switch works for them.
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
uv run pytest                                          # about 40 s; loads no weights and touches no network
RUN_MODEL_TESTS=1 uv run pytest tests/test_real_model.py   # loads the real Gemma (with its MTP drafter) and Qwen, about 4 minutes; add -k Qwen for one
VLLM_TEST_URL=http://localhost:8000 uv run pytest tests/test_vllm_backend.py -k real_server   # asks a running vLLM server
```

Most tests use the fake model in `tests/fakes.py`: it replays scripted output one token at a time and is a `Backend` (`backend.py`) like `LocalLLM`, so the harness and the page run from start to finish.

Every file format the page accepts has a test: `tests/samples.py` makes a small file of each one from the examples' files (the receipt in every image format, the recording in every audio format, the rolling ball in every video container, the sales data in every data format). `test_media.py` fails if `media.py` accepts a format without a sample, and `test_real_model.py` asks both models about each one. The tests that need Docker run only when Docker and the sandbox image are there.

How it works quotes code, and `tests/guide_excerpts.txt` keeps everything it quotes, without line numbers. When a change touches quoted code, `test_guide.py` fails and names the sections that quote it: reread them, fix what they say if it is no longer true, then rewrite the file with `UPDATE_GUIDE_SNAPSHOT=1 uv run pytest tests/test_guide.py`.

| File | What it tests |
|---|---|
| `test_gemma4.py` | Gemma 4's parser: splitting thinking, tool calls and the reply; `<\|"\|>` strings, nested arguments, all kinds of format errors, in both languages |
| `test_qwen3_5.py` | Qwen3.5's parser: XML tool calls, types from the schema, thinking, all kinds of format errors; a whole harness run in Qwen's format |
| `test_families.py` | Every family explains the tokens it uses, in both languages and for the same tokens; `model_type` finds the family; a token shows as the text it stands for, and a byte that is only part of a character as `<0xF0>`; every family explains the files of its checkpoints in both languages, naming the same fields and tokens; every model and drafter downloads at a pinned commit |
| `test_tools.py` | Schema validation and defaults, the calculator's AST allowlist (rejecting `__import__` and the like), timezones, Wikipedia (with fake HTTP); `run_python`'s locked-down `docker run`, its errors (a traceback, no output, too much memory, too long, no image) and what is missing when Docker is not ready or the image is out of date, in both languages; the charts after the marker saved without overwriting a file, only PNGs, at most four, none from code that failed; the image's version label and its pip installed by hash; with Docker, the real sandbox reads `/data` but cannot change it or reach the network, reads every data format, sends back its charts with Chinese labels drawn, and has a pip (with its certifi) past the emSign fix |
| `test_simple_backend.py` | `LocalLLM` has every method and attribute of a `Backend`, and its tokens carry every field and detail; the hand-written decoding loop: KV cache, stop tokens, the token limit, interruption, channel switches, Chinese characters split across tokens; `encode()` handing the processor each kind of media in order with the family's settings, and a short video all its frames; images and videos through the vision encoder before the prefill, and no position offset left from the last prompt; greedy taking the lower id of two tied tokens, as `generate()` does; the candidates and probabilities of temperature / top-k / top-p; MTP with a scripted fake drafter: kept and rejected guesses, the extra token when all hold, how far the KV cache is cut back, the drafter never seeing dropped guesses, and a check that guesses do not change the distribution when sampling; models downloading at their pinned revision into `models/`, which `AGENT_LOOP_MODELS` moves and `HF_HUB_CACHE` does not, and a model folder's files, their sizes (through the cache's links) and the fields its notes name |
| `test_agent.py` | The fake model is a `Backend` whose tokens carry what `LocalLLM`'s do; the harness: event order, tool results back in the context, validation errors, repeated calls, truncated results, every stop reason, Allow / Reject / Stop while waiting; the model reads errors in the turn's language, and the system prompt is the same in both; attachments as parts of the question and a note for `run_python`, the media of every turn going to the processor, the expanded context still marking only the tool result as new, `run_python` getting the conversation's folder, and a turn without `run_python`; the model reading where the charts are while the page gets the images |
| `test_render.py` | Every prefix of a run's events renders in both languages (the page redraws every 0.5 s mid-run), HTML tags pair up, the loop diagram is valid SVG, model output is always escaped, special token popovers and lists; MTP's underlines, popovers, counts and drafter card; a run of one special token as one chip with its count, a cut snippet never splitting a token, how many tokens the media took, code and output over several lines written out, tools left out of a turn, the charts of a run and of each step; the model folder card: the files in order with their sizes and notes, a file the family notes but the checkpoint lacks, the drafter's folder, escaping; tokens from a backend that reports no probabilities, or only log probabilities, drawn with what they carry and no more; no Chinese left on an English page |
| `test_app.py` | Drives the whole page with Streamlit's `AppTest`: sending a question, the example buttons, sidebar settings reaching the model, multi-turn chat, the Context and Model tabs, Allow / Reject / Stop, a new chat, resetting the settings, two tabs never running at once, switching models, How it works linking to GitHub Pages in the page's language; MTP off by default, the drafter loaded only when it is switched on, and switched off and disabled for Qwen, which has none; an English browser gets English, the last pick beats the browser's language, switching languages keeps the conversation and the settings, and the model's errors follow the language; each page shows only its own language's examples, an example attaches its file and shows only when the model reads that kind, a sample file picked above the input box goes with the next question typed and is offered only when the model can use it, Let the model write code looks for Docker only once it is on and says what is missing, the sales example runs Python on the attached file, a chart shows in the chat and in the step that drew it, and New chat removes the conversation's files; with `VLLM_URL`, the menu lists what the server serves, the sidebar and the Model tab name the server, MTP is the server's, a server with sampling masks gets no greedy decoding or unlimited top-k from the sidebar, the model reads the media the server takes and the page says which it refused and why, the menu leaves out the LoRA adapters the server serves, an error the server sends mid-run is shown in the page's language, and an unreachable server, at load or later, is said in the page's language |
| `test_chat_template.py` | With each model's real tokenizer and chat template (only when downloaded into `models/`): every special token is a single token and has a note, every token's popover shows the text it stands for, tool calls the template writes into the history parse back; a harness run with the real template checks that what the step card marks as new is exactly the tool result (two arguments, thinking mode, text before a tool call), and with the real processor, with an image and a video in the question too; every file in each model's folder (and its drafter's) has a note, and every field a note names is in the file |
| `test_vllm_backend.py` | The vLLM backend against a fake server that speaks the streams of vLLM 0.32: `/v1/completions` gets the rendered prompt's token ids, every sidebar setting with every other sampling setting neutral, the stop tokens and a `max_tokens` that fits the server's context; each token's probability and the top 5, the rank only within them; with `--enable-scale-out`, `/inference/v1/generate` and every token's rank, the top 5 by rank though the chosen token comes first; with sampling masks, which vLLM sends only with processed log probabilities, the candidates left and the sampling probability but not the model's own probabilities, and temperature 0 and top-k 0 refused on every route, as vLLM refuses them, which the page finds out at load on a server with masks but without `--enable-scale-out` too; the server's `--logprobs-mode` worked out at load with one token, processed log probabilities without masks showing how many were left when some of the top were cut, logits showing only the rank; greedy as one candidate drawn for sure; which media the server takes, each sent as a file with the placeholders and the Context tab showing the server's expansion, a different expansion at generation time an error, a refused kind a plain file; MTP's counts on either route, the per-step ones marking the guesses only when they add up and counting as kept only those up to the stop token; the stop token, the token limit, Stop closing the stream; an unreachable server, a model it does not serve, one started from a folder of its own or with a model of a family the page (or its transformers) does not know, a missing or failing `/inference/v1/generate`, its refusals and errors, in both languages, and an error mid-run shown in the page's language; warming up, and a step that ends before its first token; the model card's specs, the server, what its numbers are in each logprobs mode, its sampling masks and the media it refused; a whole turn with a tool call and one with an image, every moment of each rendered; with the real tokenizer and chat template (when in `models/`), the server gets exactly the prompt the Context tab shows; with `VLLM_TEST_URL`, a real server, each feature it was started with |
| `test_media.py` | Every format the page accepts has a sample and is read as what it is (sound is a file for Qwen) and decodes; names saved distinct and plain; the question's parts and the note naming a text file's columns; photos turned upright, sound at the model's rate, a video's frames evenly spaced at their own rate; the type each clip plays back as, and the frames shown for an MKV video |
| `test_guide.py` | Every excerpt and line number on How it works resolves, the event table points at the line that really sends each event, HTML tags pair up; the code each section quotes matches `tests/guide_excerpts.txt`, which an edit to that code changes and a line merely moving does not; each paragraph points at the same code and sections in both languages, with no Chinese left in the English one; the built pages have light and dark palettes and loop diagrams and name their commit, and the root page sends readers to the right language |
| `test_real_model.py` | Skipped by default. For each model: the loaded model is a `Backend`; greedy decoding matches `model.generate()`, with text and with an image, a video and (Gemma) a recording, and a text prompt after them too; the real model calls the calculator and answers; it reads the receipt's total in every image format, transcribes the recording in every audio format (Gemma; Qwen gets it as a file), tells which way the ball rolls in every video container, opens every data format with `run_python`, and writes code that runs in the sandbox (with Docker); for Gemma with MTP, greedy output is exactly the same as without it, with an image too, and the drafter's guesses really are kept |

## Known limitations

- Speed: on an Apple M5 (32 GB), Gemma runs at about 7–13 tok/s and Qwen3.5 0.8B at about 20–25 tok/s. In thinking mode a step often has two or three hundred thinking tokens, about 30 seconds of waiting. With MTP on, easily guessed steps such as tool calls are about 3× faster, and long Chinese text barely changes (see the table under MTP speculative decoding).
- With long replies in the page, MTP speeds things up much less than in `simple_backend.py` alone (English code: 8 → 29 tok/s alone, 9 → 12 tok/s in the page), for reasons not yet pinned down. Perhaps the page redrawing every 0.5 s and the browser competing with the model for the same machine slow MTP down more than plain decoding, since each MTP round has several small drafter forward passes.
- Only Gemma 4 E2B can use MTP: `transformers` supports a drafter reading the model's KV cache this way only for Gemma 4, and Qwen3.5 has no matching drafter.
- With MTP on, the model computes 7 positions at once, which gives very slightly different numbers from computing 1; when two tokens are almost equally likely, greedy decoding may now and then pick the other one.
- 18 of Qwen3.5's 24 layers are Gated DeltaNet. transformers has two kernels to speed them up, `flash-linear-attention` (Triton) and `causal-conv1d` (CUDA). Both need a CUDA GPU, so neither is a dependency: the PyTorch reference implementation runs instead, with the same results, only slower, and `simple_backend.py` silences transformers' notice about that fallback on every device. With a GPU the difference is small for a 0.8B model. On an NVIDIA GB10 (DGX Spark), flash-linear-attention made prefill about 4× faster (3,571 tokens in 93 ms instead of 368 ms) but decoding barely changed (56 → 60 tok/s), and both paths stayed equally close to float32. causal-conv1d has no wheel for that machine and would have to be compiled. Don't install flash-linear-attention on a Linux machine without a GPU: transformers still picks it, and the first forward pass fails.
- On MPS, transformers hangs loading Qwen3.5's weights with several threads, so on MPS `simple_backend.py` loads with one thread (`HF_DEACTIVATE_ASYNC_LOAD=1`), which does not make either model load slower.
- Qwen3.5 0.8B has thinking off by default; on, it easily keeps thinking without stopping, and lowering Max tokens per step helps. It ships no generation_config.json, so its recommended decoding settings are in `families/qwen3_5.py` (the presence penalty it recommends is not implemented here).
- Only one model is kept in memory. If two tabs pick different models, using them in turn keeps reloading; it is best for every tab to use the same model.
- There is one copy of the model, so only one agent runs at a time and other tabs wait. Reloading or closing the tab during a run does not stop the agent; other tabs (including the reloaded one) show it is still running and offer a Stop that agent button.
- With Ask me before running a tool on, if no one answers within 5 minutes the harness takes it as a rejection and stops the loop, with "confirmation timed out" as the stop reason.
- Token popovers appear only once the step finishes generating; while it generates there are only the heat colors (special tokens still explain themselves), so each redraw does not send a lot of HTML. The popovers are pure CSS, and one for a token too near the bottom of the timeline may be cut off.
- Special token popovers are placed next to the token with CSS anchor positioning, so a scrolling box does not cut them off; browsers without it open them right under the token, where they may be cut off near the bottom of the box.
- Both models are small: they may answer in Simplified Chinese, repeat the same search, answer half the question or get their sums wrong while thinking. Asked in English about two cities' times at once, Gemma calls the tool twice and then usually answers in Chinese, or stops right after the tool results without a reply; asked about one city, it answers in English. All of that is material to look at.
- Only Gemma 4 E2B hears, and only the first 30 seconds of a clip. Neither model hears a video's sound: a video is frames. Each frame costs tokens: Gemma takes 32 frames of any video, about 2,000 tokens of prefill; Qwen two a second, so its count grows with the video's length (640 tokens for the 8-second example).
- In English with Use tools on, Gemma 4 E2B often says it cannot hear an attached recording, sometimes after writing its words into a tool call; asked in Chinese, or with Use tools off, it transcribes it. So the English recording example asks about what is said rather than for a transcription, which works about two times in three.
- Qwen3.5 0.8B opens every data format with `run_python` but rarely gets multi-step pandas code right; Gemma 4 E2B answers the sales example.
- The sandbox has no network, so the model cannot `pip install`; it has numpy, pandas, openpyxl, tabulate and matplotlib. Each call is a new container, so nothing (variables, files in `/tmp`) carries over to the next call; only its charts, saved in the conversation's folder.
- The model never sees the charts it draws: they reach the page, not the context, so what it says about one comes from its own code and numbers. A chart saved with `plt.savefig()` and then closed is not sent; one shown or left open is.
- In the chat, a clip plays back in the browser's own player; Safari cannot play an MKV video, so for one the chat shows four of the frames the model reads instead.
- Attachments are up to 50 MB each. They stay under `~/.cache/agent-loop-visualizer/uploads/` until New chat or a model switch; a closed tab leaves its folder behind.
- The Wikipedia tools need the network.
- On a vLLM server (`VLLM_URL`) greedy output may differ from `generate()`'s, and what the page shows depends on the server's flags (see Running the model on a vLLM server): without `--enable-scale-out` the model reads text only, and a token's popover has either the model's own probabilities and the top 5 (the default) or, with `--logprobs-mode processed_logprobs` (which `--return-sampling-mask` needs), the sampling probability and how many candidates were left, never both; which of the top 5 were cut never shows, and a server with sampling masks takes no greedy decoding or unlimited top-k. MTP's guesses are underlined only once the step is done, only when the server's counts account for every token, and without what a turned-down guess was. With media, each step asks the server for the expansion first, a second request that reads the prompt. The page cannot see which revision the server loaded, so start it with the `--revision` there.
- `vllm_backend.py` follows the protocols of vLLM 0.32, read from its source, and is tested against a fake server that speaks them, with the real tokenizers and in the real page; `/inference/v1/generate` and its sampling masks and per-step counts are marked experimental in vLLM and may change. Its tests against a real server (`VLLM_TEST_URL`) have not yet been run on a GPU.
- The `MODEL_ID` environment variable picks the model selected at first; it may also name a checkpoint the menu does not list, as long as it belongs to a supported family (Gemma 4 or Qwen3.5); another family needs one added first, as in Supporting more models.

## License

This project's code is licensed under the [Apache License 2.0](LICENSE). The Gemma 4 weights and its MTP drafter are released by Google under Apache 2.0 (see the [model card](https://huggingface.co/google/gemma-4-E2B-it) and the [drafter's card](https://huggingface.co/google/gemma-4-E2B-it-assistant)), and the Qwen3.5 weights by Alibaba's Qwen team under Apache 2.0 (see the [model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)); none of them are included in this project. `assets/examples/speech.flac` is utterance 1272-128104-0000 of the [LibriSpeech ASR corpus](https://www.openslr.org/12) (Panayotov et al., 2015), read from LibriVox, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the other example files (`receipt.png`, `ball.mp4`, `sales.csv`) and the logo were made for this project.

About the use of AI: the code was written with the help of Claude Code, run on this machine with gemma-4-E2B-it (and its MTP drafter gemma-4-E2B-it-assistant) and Qwen3.5-0.8B, and checked in a browser.
