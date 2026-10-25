"""HTML for the visualization, built from the events of one agent run.

Streamlit redraws the page on every refresh, so everything here is a pure function
of the event list: build() folds the events into steps, and the other functions
turn steps into HTML strings for st.html. Every function that writes words takes the
page's language, "zh" or "en" (i18n.py), so switching it redraws a finished run too.

A token shows as much of the model's distribution as the backend reported (backend.TOKEN_DETAILS):
simple_backend.LocalLLM reports all of it, and what a backend leaves out is left out here too, never filled in.
"""

import base64
import html
import json
import re

from harness import families
from harness.i18n import pick

STOP_REASONS = {
    "zh": {
        "no_tool_call": ("模型沒有再呼叫工具", "模型這一步只回覆文字，harness 把它交給使用者，迴圈結束。"),
        "max_steps": ("達到步數上限", "harness 的防呆：迴圈跑滿設定的步數就停，不管模型想不想繼續。"),
        "max_new_tokens": ("達到單步 token 上限", "這一步生成的 token 數到了上限，回覆可能被截斷。"),
        "user_stopped": ("使用者中斷", "你按了停止，harness 在下一個 token 前停下。"),
        "format_errors": ("連續格式錯誤", "模型連續 3 次寫出無法解析的工具呼叫，harness 停止迴圈。"),
        "approval_timeout": ("等待確認逾時", "一直沒有人按「允許」或「拒絕」（例如分頁已經關掉或重新整理），harness 當作拒絕並停止迴圈，讓模型可以給其他分頁使用。"),
        "error": ("執行錯誤", "程式發生錯誤，訊息在下面。"),
    },
    "en": {
        "no_tool_call": ("no more tool calls", "This step the model only replied in text; the harness hands the reply to the user and the loop ends."),
        "max_steps": ("step limit reached", "A harness safeguard: the loop stops after the set number of steps, whether or not the model wants to go on."),
        "max_new_tokens": ("token limit in a step", "This step generated as many tokens as it may; the reply may be cut off."),
        "user_stopped": ("stopped by the user", "You pressed stop, and the harness stopped before the next token."),
        "format_errors": ("repeated format errors", "The model wrote a tool call that could not be parsed 3 times in a row, so the harness stopped the loop."),
        "approval_timeout": ("confirmation timed out", "No one pressed Allow or Reject (the tab was closed or reloaded, say), so the harness took it as a refusal and stopped the loop, freeing the model for other tabs."),
        "error": ("error", "Something went wrong in the code; the message is below."),
    },
}
LAYER_TYPES = {
    "zh": {"sliding_attention": "滑動視窗", "full_attention": "full attention", "linear_attention": "線性 attention"},
    "en": {"sliding_attention": "sliding window", "full_attention": "full attention", "linear_attention": "linear attention"},
}
MODALITIES = {
    "zh": {"text": "文字", "image": "圖片", "audio": "聲音", "video": "影片"},
    "en": {"text": "text", "image": "image", "audio": "audio", "video": "video"},
}
RUN_MIN = 4  # this many of one special token in a row are drawn as one chip with a count, like the hundreds an image becomes
PARAM_PARTS = {
    "zh": {"language_model": "語言模型", "audio_tower": "音訊編碼器", "vision_tower": "視覺編碼器", "visual": "視覺編碼器"},
    "en": {"language_model": "language model", "audio_tower": "audio encoder", "vision_tower": "vision encoder", "visual": "vision encoder"},
}
# Each kind of file a checkpoint's folder may hold, in the order the Model tab lists them: language -> (what it is,
# what it is for and what reads it). What one holds in a family's checkpoints is the family's (families.FileNote).
MODEL_FILES = {
    "config.json": {
        "zh": ("架構", "模型由什麼組成：類別、層數、寬度、head 數、詞彙量和脈絡上限。`AutoConfig` 最先讀它：`model_type` 決定 family，這個分頁上方的規格也來自它。"),
        "en": ("architecture", "What the model is made of: its class, layers, widths, heads, vocabulary and context limit. `AutoConfig` reads it first: "
                               "its `model_type` picks the family, and the specs at the top of this tab come from it."),
    },
    "generation_config.json": {
        "zh": ("生成預設值", "生成的預設值：停止 token（`eos_token_id`）和建議的 temperature / top-p / top-k。停止 token 和 family 的合在一起；建議值就是「解碼設定」裡寫的那些。"),
        "en": ("generation defaults", "Defaults for generating: the stop tokens (`eos_token_id`) and the recommended temperature / top-p / top-k. "
                                      "The stop tokens join the family's, and the recommendations are those under Decoding settings."),
    },
    "model.safetensors": {
        "zh": ("權重", "權重：開頭是每個 tensor 的名稱、形狀和型別，後面是原始數字。safetensors 載入時不執行任何程式（pickle 存的 checkpoint 會），"
                       "也能直接映射進記憶體。`from_pretrained` 用上面寫的精度把它放到裝置上；在 CPU 上是 float32，記憶體是 bfloat16 權重的兩倍。"),
        "en": ("weights", "The weights: a header with every tensor's name, shape and type, then the raw numbers. Loading safetensors runs no code, "
                          "unlike a pickled checkpoint, and the file can be mapped straight into memory. `from_pretrained` puts them on the device in the precision above, "
                          "which on a CPU is float32: twice the memory of bfloat16 weights."),
    },
    "model.safetensors.index.json": {
        "zh": ("權重索引", "權重拆成好幾個檔案時，記下每個 tensor 在哪個檔案（`weight_map`）。"),
        "en": ("weight index", "For weights split into several files: which file holds each tensor (`weight_map`)."),
    },
    "tokenizer.json": {
        "zh": ("tokenizer", "tokenizers 函式庫的完整 tokenizer：文字怎麼正規化、怎麼切，詞彙表、BPE merges 和 added tokens。processor 載入它；頁面上每個 token 都是它的一個條目。"),
        "en": ("tokenizer", "The whole tokenizer for the tokenizers library: how text is normalized and split, the vocabulary, the BPE merges and the added tokens. "
                            "The processor loads it, and every token on the page is one of its entries."),
    },
    "tokenizer_config.json": {
        "zh": ("tokenizer 設定", "transformers 怎麼包裝這個 tokenizer：類別、哪些 token 扮演什麼角色（開頭、結尾、padding），以及輸入最長多長。"),
        "en": ("tokenizer settings", "How transformers wraps that tokenizer: its class, which tokens play which part (beginning, end, padding) and the longest input it takes."),
    },
    "vocab.json": {
        "zh": ("詞彙表", "只有詞彙表（token → id），舊的格式；`tokenizer.json` 裡也有。"),
        "en": ("vocabulary", "The vocabulary alone, token → id, in the older format; `tokenizer.json` holds it too."),
    },
    "merges.txt": {
        "zh": ("merges", "只有 BPE merges，一行一組，舊的格式；`tokenizer.json` 裡也有。"),
        "en": ("merges", "The BPE merges alone, one pair per line, in the older format; `tokenizer.json` holds them too."),
    },
    "chat_template.jinja": {
        "zh": ("chat template", "chat template：一段 Jinja 程式，把訊息、工具 schema 和思考開關寫成模型讀的那一個字串。`render()` 執行它，「脈絡」分頁顯示它寫出來的東西。"),
        "en": ("chat template", "The chat template: a Jinja program that turns the messages, the tool schemas and the thinking switch into the one string the model reads. "
                                "`render()` runs it, and the Context tab shows what it wrote."),
    },
    "processor_config.json": {
        "zh": ("processor", "processor 怎麼把附件變成編碼器的輸入：圖片的大小和 patch、聲音的特徵、影片的畫面。`AutoProcessor` 讀它。"),
        "en": ("processor", "How the processor turns attachments into the encoders' inputs: image sizes and patches, sound features, video frames. `AutoProcessor` reads it."),
    },
    "preprocessor_config.json": {
        "zh": ("圖片 processor", "processor 怎麼把圖片變成視覺編碼器的輸入。`AutoProcessor` 讀它。"),
        "en": ("image processor", "How the processor turns images into the vision encoder's input. `AutoProcessor` reads it."),
    },
    "video_preprocessor_config.json": {
        "zh": ("影片 processor", "processor 怎麼把影片變成給視覺編碼器的畫面。`AutoProcessor` 讀它。"),
        "en": ("video processor", "How the processor turns videos into frames for the vision encoder. `AutoProcessor` reads it."),
    },
    "README.md": {
        "zh": ("模型卡", "Hub 上的模型卡：模型是什麼、授權、benchmark 和建議用法。app 不讀它。"),
        "en": ("model card", "The model card on the Hub: what the model is, its license, benchmarks and recommended use. The app does not read it."),
    },
    "LICENSE": {
        "zh": ("授權", "權重的授權條款全文。app 不讀它。"),
        "en": ("license", "The license the weights are released under. The app does not read it."),
    },
    ".gitattributes": {
        "zh": ("git 設定", "Hub 上 git repository 的設定（哪些檔案當大型檔案存）。app 不讀它。"),
        "en": ("git settings", "Settings for the model's repository on the Hub (which files are stored as large files). The app does not read it."),
    },
}

INTRO = {
    "zh": """
<div class="intro">
  <p class="intro-lede">送出問題後，這裡會一步一步畫出 agent 迴圈。每一步都是同一個循環：</p>
  <dl class="intro-list">
    <div style="--stage: var(--c-context)"><dt>脈絡</dt><dd>chat template 把 system prompt、工具 schema 和對話接成一長串 token，這就是模型這一步能看到的全部。</dd></div>
    <div style="--stage: var(--c-model)"><dt>模型</dt><dd>一次推論。每個 token 都是從機率分布抽出來的：顏色越深，代表被選中的 token 機率越低。滑過 token 可以看候選。</dd></div>
    <div style="--stage: var(--c-harness)"><dt>Harness</dt><dd>解析模型寫出的工具呼叫、用 schema 驗證、需要時先問你。模型從不直接碰環境。</dd></div>
    <div style="--stage: var(--c-env)"><dt>環境</dt><dd>真正執行工具的地方。結果包成觀察接回脈絡，開始下一步。</dd></div>
    <div style="--stage: var(--c-stop)"><dt>停止</dt><dd>模型不再呼叫工具、達到步數或 token 上限，或是你按下停止。</dd></div>
  </dl>
</div>
""",
    "en": """
<div class="intro">
  <p class="intro-lede">Send a question, and the agent loop is drawn here step by step. Every step goes around the same cycle:</p>
  <dl class="intro-list">
    <div style="--stage: var(--c-context)"><dt>Context</dt><dd>The chat template joins the system prompt, the tool schemas and the conversation into one long run of tokens. That is all the model can see in a step.</dd></div>
    <div style="--stage: var(--c-model)"><dt>Model</dt><dd>One inference. Every token is drawn from a probability distribution: the darker a token, the less likely it was. Hover over a token to see the candidates.</dd></div>
    <div style="--stage: var(--c-harness)"><dt>Harness</dt><dd>Parses the tool calls the model wrote, validates them against the schema and asks you first when it should. The model never touches the environment itself.</dd></div>
    <div style="--stage: var(--c-env)"><dt>Environment</dt><dd>Where the tools really run. The result goes back into the context as an observation, and the next step begins.</dd></div>
    <div style="--stage: var(--c-stop)"><dt>Stop</dt><dd>The model calls no more tools, a step or token limit is reached, or you press stop.</dd></div>
  </dl>
</div>
""",
}

CONTEXT_LEGEND = {
    "zh": """
<p class="legend">
  <span><span class="sp">特殊 token</span> 滑過看用途，全部列在「模型」分頁</span>
  <mark class="added">這一步新加入</mark>
  <span class="gen">模型這一步的輸出</span>
</p>
""",
    "en": """
<p class="legend">
  <span><span class="sp">special token</span> hover to see what it is for; all of them are in the Model tab</span>
  <mark class="added">new in this step</mark>
  <span class="gen">the model's output in this step</span>
</p>
""",
}


# ---------- formatting ----------

def esc(value) -> str:
    return html.escape(str(value), quote=True)


def fmt_ms(ms: float) -> str:
    if ms < 1:
        return "<1 ms"
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def pct(p: float) -> str:
    if p >= 0.9995:
        return "100%"
    if p >= 0.1:
        return f"{p * 100:.1f}%"
    if p >= 0.001:
        return f"{p * 100:.2f}%"
    return f"{p * 100:.1e}%"


def fmt_params(n: int) -> str:
    return f"{n / 1e9:.2f}B" if n >= 1e8 else f"{n / 1e6:.1f}M"


def fmt_bytes(n: int) -> str:
    for unit, size in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if round(n / size, 1) >= 1:
            return f"{n / size:.1f} {unit}"
    return f"{n} B"


def n_steps(n: int, lang: str) -> str:
    return pick(lang, f"{n} 步", f"{n} step" + ("" if n == 1 else "s"))


def prose_html(text: str) -> str:
    """Escape a description and set its `backticked` parts in monospace."""
    parts = esc(text).split("`")
    return "".join(f'<span class="tk">{p}</span>' if i % 2 else p for i, p in enumerate(parts))


def meaning_html(piece: str, family: families.Family, lang: str) -> str:
    """A special token's name and what it is for; empty for tokens we have no note on."""
    if piece not in family.special_tokens:
        return ""
    name, desc = family.special_tokens[piece][lang]
    return f"<b>{esc(name)}</b><span>{prose_html(desc)}</span>"


def stop_meaning(piece: str, family: families.Family, lang: str) -> str:
    """What stopping at this token means, e.g. that the model waits for a tool result."""
    return family.stop_tokens.get(piece, {}).get(lang, "")


def chip(piece: str, family: families.Family, lang: str, count: int = 1) -> str:
    """A special token as a dark chip, with ×count for a run of it; hovering it pops up what the token is for."""
    meaning = meaning_html(piece, family, lang)
    tip = f'<span class="sp-tip">{meaning}</span>' if meaning else ""
    times = f'<span class="sp-n">×{count:,}</span>' if count > 1 else ""
    return f'<span class="sp">{esc(piece)}{times}{tip}</span>'


def segments(text: str, family: families.Family) -> list[tuple[str, int]]:
    """Text split into plain text, (text, 0), and runs of one control token, (token, how many in a row)."""
    out, last = [], 0
    for m in family.special_re.finditer(text):
        if m.start() > last:
            out.append((text[last:m.start()], 0))
        if out and out[-1][1] and out[-1][0] == m.group(0):
            out[-1] = (m.group(0), out[-1][1] + 1)
        else:
            out.append((m.group(0), 1))
        last = m.end()
    if last < len(text):
        out.append((text[last:], 0))
    return out


def special_html(text: str, family: families.Family, lang: str, last: int | None = None) -> str:
    """Escape text and turn the family's control tokens, such as <|turn>, into chips; a run of RUN_MIN or more of one
    token is one chip with its count. With `last`, only about the last that many characters are drawn, after an
    ellipsis, a run counting as one token, so a cut never lands inside one."""
    segs = segments(text, family)
    cut = False
    if last is not None:
        kept, left = [], last
        for piece, count in reversed(segs):
            size = len(piece) * (1 if count >= RUN_MIN else max(count, 1))
            if size > left:
                if not count and left > 0:
                    kept.append((piece[-left:], 0))
                cut = True
                break
            kept.append((piece, count))
            left -= size
        segs = kept[::-1]
    out = ["…"] if cut else []
    for piece, count in segs:
        if not count:
            out.append(esc(piece))
        elif count >= RUN_MIN:
            out.append(chip(piece, family, lang, count))
        else:
            out.append(chip(piece, family, lang) * count)
    return "".join(out)


def pretty(text: str) -> str:
    """A tool result for reading: its JSON indented, and a value that runs over several lines (what code printed,
    a traceback) written out line by line after a |, instead of as one string full of \\n."""
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    if not isinstance(value, dict) or not any(isinstance(v, str) and "\n" in v.strip() for v in value.values()):
        return json.dumps(value, ensure_ascii=False, indent=2)
    lines = []
    for n, (key, v) in enumerate(value.items(), 1):
        name = json.dumps(key, ensure_ascii=False)
        if isinstance(v, str) and "\n" in v.strip():
            lines += [f"  {name}: |", *("    " + line for line in v.rstrip("\n").split("\n"))]
        else:
            body = json.dumps(v, ensure_ascii=False, indent=2).replace("\n", "\n  ")
            lines.append(f"  {name}: {body}" + ("," if n < len(value) else ""))
    return "\n".join(["{", *lines, "}"])


def media_line(counts: dict, lang: str) -> str:
    """How many of the context's tokens each kind of media took, e.g. "圖片 266 · 影片 2,016"."""
    return " · ".join(f"{MODALITIES[lang].get(kind, kind)} {n:,}" for kind, n in counts.items())


# ---------- events -> steps ----------

def build(events: list[dict]) -> dict:
    data = {"model": None, "family": None, "settings": {}, "steps": [], "end": None, "error": None, "last": None, "t": 0}
    for ev in events:
        kind = ev["type"]
        data["last"], data["t"] = kind, ev["t"]
        if kind == "run_start":
            data["model"], data["family"] = ev["model"], families.get(ev["family"])
            data["settings"] = ev["settings"]
        elif kind == "step_start":
            data["steps"].append({
                "n": ev["step"], "t0": ev["t"], "t1": ev["t"], "context": None, "prefill": None,
                "tokens": [], "done": None, "parsed": None, "calls": {},
            })
        elif kind == "run_end":
            data["end"] = ev
        elif kind == "error":
            data["error"] = ev.get("text") or ev["message"]
        elif data["steps"]:
            step = data["steps"][-1]
            step["t1"] = ev["t"]
            if kind == "token":
                step["tokens"].append(ev)
            elif kind in ("context", "prefill", "parsed"):
                step[kind] = ev
            elif kind == "model_end":
                step["done"] = ev
                step["tokens"] = with_drafts(step["tokens"], ev.get("mtp") or {})
            elif kind == "tool_call":
                step["calls"][ev["call_id"]] = {**ev, "approval": None, "started": False, "result": None}
            elif kind == "approval_request":
                step["calls"][ev["call_id"]]["approval"] = "pending"
            elif kind == "approval":
                if ev["approved"]:
                    approval = "approved"
                elif ev.get("timed_out"):
                    approval = "timeout"
                elif ev.get("stopped"):
                    approval = "stopped"
                else:
                    approval = "rejected"
                step["calls"][ev["call_id"]]["approval"] = approval
            elif kind == "tool_start":
                step["calls"][ev["call_id"]]["started"] = True
            elif kind == "tool_result":
                step["calls"][ev["call_id"]]["result"] = ev
    return data


def with_drafts(tokens: list[dict], mtp: dict) -> list[dict]:
    """Mark the guesses a server reported only at the end of a step (vllm_backend.drafted_tokens: the places in kept_at
    and turned_down_at), as simple_backend.py marks them while generating. The server does not say what a guess turned down was."""
    kept, turned_down = set(mtp.get("kept_at", ())), set(mtp.get("turned_down_at", ()))
    if not kept and not turned_down:
        return tokens
    marks = {**{i: {"piece": None, "accepted": False} for i in turned_down}, **{i: {"accepted": True} for i in kept}}
    return [{**t, "draft": {"piece": t["piece"], **marks[t["index"]]}} if t["index"] in marks and "draft" not in t else t for t in tokens]


def images(steps: list[dict]) -> list[str]:
    """The files the tools in these steps made for the user to see, such as run_python's charts, in order."""
    return [path for s in steps for c in s["calls"].values() if c["result"] for path in c["result"].get("images", [])]


def all_calls(data: dict) -> list[dict]:
    return [{**c, "step": s["n"]} for s in data["steps"] for c in s["calls"].values()]


def pending_approvals(data: dict) -> list[dict]:
    return [c for c in all_calls(data) if c["approval"] == "pending"]


def draft(data: dict) -> str:
    """The reply as it streams: content tokens of the step in progress."""
    if not data["steps"]:
        return ""
    return "".join(t["text"] for t in data["steps"][-1]["tokens"] if t["channel"] == "content")


def stop_title(stop_reason: str, lang: str) -> str:
    return STOP_REASONS[lang].get(stop_reason, (stop_reason,))[0]


def status(data: dict, lang: str) -> str:
    if not data["steps"]:
        return pick(lang, "啟動中…", "Starting…")
    step, last = data["steps"][-1], data["last"]
    n = step["n"]
    if last in ("step_start", "context"):
        return f"Step {n} · " + pick(lang, "把脈絡送進模型", "sending the context to the model")
    if last == "prefill":
        read = step["prefill"]["n_tokens"]
        return f"Step {n} · " + pick(lang, f"模型讀完 {read:,} 個 token，開始生成", f"the model has read {read:,} tokens and starts generating")
    if last == "token":
        channel = step["tokens"][-1]["channel"]
        if channel == "thought":
            return f"Step {n} · " + pick(lang, "模型思考中…", "the model is thinking…")
        if channel == "tool_call":
            return f"Step {n} · " + pick(lang, "模型正在寫工具呼叫…", "the model is writing a tool call…")
        return f"Step {n} · " + pick(lang, "模型生成中…", "the model is generating…")
    if last == "approval_request":
        return f"Step {n} · " + pick(lang, "等你確認要不要執行工具", "waiting for you to allow the tool call")
    if last == "tool_start":
        name = list(step["calls"].values())[-1]["name"]
        return f"Step {n} · " + pick(lang, f"在環境執行 {name}", f"running {name} in the environment")
    if last == "tool_result":
        return f"Step {n} · " + pick(lang, "觀察接回脈絡", "the observation goes back into the context")
    return f"Step {n} · " + pick(lang, "harness 解析與驗證", "the harness parses and validates")


def footer(data: dict, lang: str) -> str:
    end = data["end"]
    title = stop_title(end["stop_reason"], lang)
    return f"{n_steps(len(data['steps']), lang)} · {fmt_ms(end['t'])} · " + pick(lang, f"停止：{title}", f"stop: {title}")


def run_stats(data: dict, lang: str) -> str:
    s = data["settings"]
    steps = data["steps"]
    ctx = next((st["context"]["n_tokens"] for st in reversed(steps) if st["context"]), 0)
    gen = sum(len(st["tokens"]) for st in steps)
    return f"Step {len(steps)} / {s.get('max_steps', '?')} · " + pick(
        lang, f"脈絡 {ctx:,} tokens · 共生成 {gen:,} tokens", f"context {ctx:,} tokens · {gen:,} tokens generated"
    ) + f" · {fmt_ms(data['t'])}"


# ---------- chat side ----------

def trace_html(data: dict, lang: str) -> str:
    chips = []
    for c in all_calls(data):
        r = c["result"]
        if r is None:
            mark, ok = "…", ""
        else:
            mark = ("✓ " if r["ok"] else "✗ ") + (fmt_ms(r["ms"]) if c["started"] else pick(lang, "未執行", "not run"))
            ok = "true" if r["ok"] else "false"
        chips.append(f'<span class="trace-chip" data-ok="{ok}">Step {c["step"]} · {esc(c["name"])} <span class="dot">{mark}</span></span>')
    return f'<div class="trace">{"".join(chips)}</div>' if chips else ""


# ---------- the loop diagram ----------

NODES = [  # key, x, y, title and caption in each language
    ("context", 22, 16, {"zh": ("脈絡 Context", "system · 工具 · 對話"), "en": ("Context", "system · tools · chat")}),
    ("model", 300, 16, {"zh": ("模型 LLM", "LLM"), "en": ("Model", "LLM")}),
    ("harness", 300, 152, {"zh": ("Harness", "解析 · 驗證 · 權限"), "en": ("Harness", "parse · check · ask")}),
    ("env", 22, 152, {"zh": ("環境 Environment", "時鐘 · Wiki · Python"), "en": ("Environment", "clock · wiki · python")}),
    ("stop", 560, 152, {"zh": ("停止", "回覆使用者"), "en": ("Stop", "reply to the user")}),
]
EDGES = [  # key, path, label position and anchor, label in each language
    ("context-model", "M182 48 H294", 239, 38, "middle", {"zh": "送入模型", "en": "into the model"}),
    ("model-harness", "M380 80 V148", 390, 120, "start", {"zh": "輸出 token", "en": "tokens out"}),
    ("harness-env", "M298 184 H188", 242, 174, "middle", {"zh": "執行工具", "en": "run a tool"}),
    ("env-context", "M102 150 V84", 92, 120, "end", {"zh": "觀察", "en": "observation"}),
    ("harness-stop", "M462 184 H554", 509, 174, "middle", {"zh": "沒有工具呼叫", "en": "no tool call"}),
]


def loop_state(data: dict | None, lang: str) -> dict:
    """Which node and edge are active, what each node's caption says, and the step counter."""
    state = {"active": None, "edge": None, "visited": set(), "waiting": False, "captions": {},
             "counter": pick(lang, "等待問題", "waiting for a question"), "ended": False}
    if not data or not data["last"]:
        return state
    caps, steps = state["captions"], data["steps"]
    for s in steps:
        state["visited"].add("context")
        if s["prefill"]:
            state["visited"].add("model")
        if s["done"]:
            state["visited"].add("harness")
        if any(c["started"] for c in s["calls"].values()):
            state["visited"].add("env")

    step = steps[-1] if steps else None
    if step:
        if step["context"]:
            caps["context"] = f"{step['context']['n_tokens']:,} tokens"
        if step["prefill"]:
            caps["model"] = f"prefill {fmt_ms(step['prefill']['ms'])}"
        if step["tokens"]:
            caps["model"] = pick(lang, f"已生成 {len(step['tokens'])} tokens", f"{len(step['tokens'])} tokens so far")
        if step["done"]:
            caps["model"] = f"{step['done']['n_tokens']} tok · {step['done']['tok_per_s']} tok/s"
        calls = list(step["calls"].values())
        if step["parsed"]:
            caps["harness"] = pick(lang, "、", ", ").join(c["name"] for c in calls) or pick(lang, "沒有工具呼叫", "no tool call")
        if calls:
            c = calls[-1]
            if c["approval"] == "pending":
                caps["harness"] = pick(lang, "等你確認", "waiting for you")
            if c["started"]:
                caps["env"] = c["environment"][lang]
            if c["result"] and c["started"]:
                done = pick(lang, "完成", "done") if c["result"]["ok"] else pick(lang, "錯誤", "error")
                caps["env"] = f"{done} · {fmt_ms(c['result']['ms'])}"
        state["counter"] = f"Step {step['n']} / {data['settings'].get('max_steps', '?')}"

    last = data["last"]
    started = bool(step and step["calls"] and list(step["calls"].values())[-1]["started"])
    state["active"], state["edge"] = {
        "step_start": ("context", "env-context" if step and step["n"] > 1 else None),
        "context": ("context", "context-model"),
        "prefill": ("model", "context-model"),
        "token": ("model", None),
        "model_end": ("harness", "model-harness"),
        "parsed": ("harness", "model-harness"),
        "tool_call": ("harness", None),
        "approval_request": ("harness", None),
        "approval": ("harness", None),
        "tool_start": ("env", "harness-env"),
        "tool_result": ("env", "env-context") if started else ("harness", None),
        "run_end": ("stop", "harness-stop"),
        "error": ("stop", None),
    }.get(last, (None, None))
    state["waiting"] = last == "approval_request"
    if data["end"]:
        state["ended"] = True
        state["visited"].add("stop")
        caps["stop"] = STOP_REASONS[lang].get(data["end"]["stop_reason"], ("",))[0]
        state["counter"] = pick(lang, "完成 · ", "done · ") + n_steps(len(steps), lang)
    return state


# An <img> SVG cannot see the page's CSS, so the diagram carries its own colors per theme.
PALETTES = {
    "light": {"surface": "#fbf9f3", "ink": "#18212b", "ink2": "#414c58", "muted": "#5f6772", "line": "#d8d0bf", "accent": "#b73d23",
              "context": "#3d5c9a", "model": "#b73d23", "harness": "#6b4c8a", "env": "#286f60", "stop": "#414c58"},
    "dark": {"surface": "#1a1f26", "ink": "#e8e3d8", "ink2": "#c4beb2", "muted": "#9aa1aa", "line": "#333a44", "accent": "#e3795f",
             "context": "#93b3f0", "model": "#e3795f", "harness": "#b9a0e0", "env": "#6fbfa8", "stop": "#c4beb2"},
}

DIAGRAM_CSS = """
text { font-family: -apple-system, "PingFang TC", "Noto Sans TC", "Microsoft JhengHei", sans-serif; fill: {muted}; text-anchor: middle; }
.edge path { fill: none; stroke: {line}; stroke-width: 2; marker-end: url(#a); }
.edge text { font-size: 12px; }
.edge.on path { stroke: {accent}; stroke-dasharray: 6 5; animation: flow .7s linear infinite; marker-end: url(#a-on); }
.edge.on text { fill: {ink}; font-weight: 600; }
.ended .edge.on path { animation: none; stroke-dasharray: none; }
.node rect { fill: {surface}; stroke: {line}; stroke-width: 1.5; }
.node .title { font-size: 14px; font-weight: 700; fill: {ink2}; }
.node .caption { font-size: 11.5px; font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.node.visited rect { stroke: color-mix(in srgb, var(--stage) 55%, {line}); }
.node.on rect { stroke: var(--stage); stroke-width: 2.5; fill: color-mix(in srgb, var(--stage) 14%, {surface}); }
.node.on .title { fill: var(--stage); }
.node.on .caption { fill: {ink}; }
.node.waiting rect { animation: pulse 1.2s ease-in-out infinite; }
.counter { font: 700 13px ui-monospace, "SF Mono", Menlo, monospace; fill: {ink2}; letter-spacing: .06em; }
@keyframes flow { to { stroke-dashoffset: -11; } }
@keyframes pulse { 50% { stroke-width: 4.5; } }
@media (prefers-reduced-motion: reduce) { .edge.on path, .node.waiting rect { animation: none; } }
"""


def diagram_html(data: dict | None, dark: bool, lang: str, counter: str | None = None, model: str | None = None) -> str:
    """The loop diagram as an <img>: Streamlit's HTML sanitizer removes inline <svg>.
    `model` names the model when no run says which one it is."""
    st = loop_state(data, lang)
    if counter:
        st["counter"] = counter
    model = (data or {}).get("model") or model
    if model:
        st["captions"].setdefault("model", model.split("/")[-1])
    pal = PALETTES["dark" if dark else "light"]
    css = DIAGRAM_CSS
    for key, value in pal.items():
        css = css.replace("{" + key + "}", value)
    edges = []
    for key, path, x, y, anchor, label in EDGES:
        cls = "edge on" if key == st["edge"] else "edge"
        edges.append(f'<g class="{cls}"><path d="{path}"/><text x="{x}" y="{y}" text-anchor="{anchor}">{label[lang]}</text></g>')
    nodes = []
    for key, x, y, words in NODES:
        title, caption = words[lang]
        cls = ["node"]
        if key in st["visited"]:
            cls.append("visited")
        if key == st["active"]:
            cls.append("on")
            if st["waiting"]:
                cls.append("waiting")
        cx = x + 80
        nodes.append(
            f'<g class="{" ".join(cls)}" style="--stage: {pal[key]}">'
            f'<rect x="{x}" y="{y}" width="160" height="64" rx="10"/>'
            f'<text class="title" x="{cx}" y="{y + 26}">{title}</text>'
            f'<text class="caption" x="{cx}" y="{y + 47}">{esc(st["captions"].get(key, caption))}</text></g>'
        )
    arrow = '<marker id="{id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="{fill}"/></marker>'
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 232" class="{"ended" if st["ended"] else ""}">'
        f'<style>{css}</style><defs>{arrow.format(id="a", fill=pal["line"])}{arrow.format(id="a-on", fill=pal["accent"])}</defs>'
        f'{"".join(edges)}{"".join(nodes)}<text class="counter" x="242" y="122">{esc(st["counter"])}</text></svg>'
    )
    src = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    alt = pick(
        lang,
        "Agent 迴圈：脈絡送入模型，模型輸出交給 harness；有工具呼叫就在環境執行，觀察接回脈絡；沒有就停止。目前：",
        "The agent loop: the context goes into the model and its output to the harness; a tool call runs in the environment "
        "and its observation goes back into the context; without one, the loop stops. Now: ",
    ) + esc(st["counter"])
    return f'<div class="diagram-wrap"><img class="diagram" src="{src}" alt="{alt}"></div>'


# ---------- one step of the loop ----------

def heat(p: float) -> str:
    return f"{min(1.0, max(0.0, 1 - p)) * 0.6:.3f}"


def special_note(tok: dict, family: families.Family, lang: str) -> str:
    meaning = meaning_html(tok["piece"], family, lang) if tok.get("special") else ""
    return f'<span class="tip-what">{meaning}</span>' if meaning else ""


def draft_note(tok: dict, family: families.Family, lang: str) -> str:
    """What MTP did at this position: the drafter's guess was kept, or turned down and replaced."""
    draft = tok.get("draft")
    if not draft:
        return ""
    if draft["accepted"]:
        return '<span class="tip-draft">' + pick(
            lang,
            "<b>MTP：草稿猜中</b>草稿模型先猜了這個 token，主模型在同一次 forward 裡驗證後收下"
            "（greedy 時它得是第 1 名；抽樣時收下的機率就是取樣時的機率）。",
            "<b>MTP: the drafter guessed it</b>The drafter guessed this token first, and the model kept it after checking it "
            "in the same forward pass (in greedy decoding it had to be the top token; when sampling, it is kept with its sampling probability).",
        ) + "</span>"
    if draft["piece"] is None:  # a server that counts the guesses it turned down but does not say what they were
        return '<span class="tip-draft">' + pick(
            lang,
            "<b>MTP：草稿被退回</b>草稿模型在這裡猜了另一個 token（伺服器沒說是哪一個），主模型沒有收下，"
            "這個 token 由主模型自己選，後面的猜測一起作廢。",
            "<b>MTP: the guess was turned down</b>The drafter guessed another token here (the server does not say which), and the "
            "model did not keep it. The model picked this token itself, and the guesses after it were dropped.",
        ) + "</span>"
    guess = f'<span class="tk">{esc(family.show_piece(draft["piece"]))}</span>'
    return '<span class="tip-draft">' + pick(
        lang,
        f"<b>MTP：草稿被退回</b>草稿模型猜的是 {guess}，"
        "主模型沒有收下，這個 token 由主模型自己選（greedy 取第 1 名，抽樣時從其他候選裡抽），後面的猜測一起作廢。",
        f"<b>MTP: the guess was turned down</b>The drafter guessed {guess}, and the model did not keep it. "
        "The model picked this token itself (the top one in greedy decoding, otherwise drawn from the other candidates), and the guesses after it were dropped.",
    ) + "</span>"


def tip_html(tok: dict, family: families.Family, lang: str) -> str:
    """A finished token's popover: the token, what it means if it is special, what MTP did there, and the numbers
    and candidates the backend reported for it (backend.TOKEN_DETAILS); a number it did not report is left out."""
    numbers = [
        (pick(lang, "模型給的機率", "model probability"), pct(tok["p"]) if known(tok, "p") else None),
        (pick(lang, "排名", "rank"), f'#{tok["rank"]}' if known(tok, "rank") else None),
        (pick(lang, "取樣時的機率", "sampling probability"), pct(tok["p_sampled"]) if known(tok, "p_sampled") else None),
        (pick(lang, "剩下的候選", "candidates left"), f'{tok["kept"]:,}' if known(tok, "kept") else None),
    ]
    grid = "".join(f"<span>{label}</span><b>{value}</b>" for label, value in numbers if value is not None)
    n = tok["index"] + 1
    return (
        f'<span class="tip"><span class="tip-piece">{esc(family.show_piece(tok["piece"]))}</span>'
        f'<span class="tip-sub">token id {tok["id"]} · {pick(lang, f"第 {n} 個生成的 token", f"generated token #{n}")}</span>'
        f'{special_note(tok, family, lang)}{draft_note(tok, family, lang)}'
        + (f'<span class="tip-grid">{grid}</span>' if grid else "")
        + f'{candidates_html(tok, family, lang)}</span>'
    )


def candidates_html(tok: dict, family: families.Family, lang: str) -> str:
    """The most likely tokens at this position, the chosen one highlighted and those top-k / top-p cut grayed out,
    when the backend reported them; a candidate it did not say was cut or kept is drawn as it is."""
    top = tok.get("top")
    if not top:
        return ""
    rows = []
    for c in top:
        cls = "c" + (" on" if c["id"] == tok["id"] else "") + (" cut" if c.get("kept") is False else "")
        rows.append(
            f'<span class="{cls}"><span class="cp">{esc(family.show_piece(c["piece"]))}</span>'
            f'<span class="b"><span style="width:{max(c["p"] * 100, 1):.1f}%"></span></span><span>{pct(c["p"])}</span></span>'
        )
    if any(c["id"] == tok["id"] for c in top):
        outside = ""
    elif known(tok, "rank"):
        outside = pick(lang, f'抽到排名 #{tok["rank"]} 的 token，不在前 5 名。', f'Drew the token ranked #{tok["rank"]}, outside the top 5.')
    else:
        outside = pick(lang, "抽到的 token 不在前 5 名。", "Drew a token outside the top 5.")
    if any(known(c, "kept") for c in top):
        label = pick(lang, "前 5 名候選（模型原始分布，灰色 = 被剃掉）", "top 5 by model probability; gray = cut")
    else:
        label = pick(lang, "前 5 名候選（模型原始分布）", "top 5 by model probability")
    return f'<span class="tip-label">{label}</span>{"".join(rows)}' + (f'<span class="tip-foot">{outside}</span>' if outside else "")


def known(item: dict, key: str) -> bool:
    """Whether the backend reported `key` (one of backend.TOKEN_DETAILS, or a candidate's kept)."""
    return item.get(key) is not None


def reported(tokens: list[dict], key: str) -> bool:
    """Whether the backend reports `key` for its tokens: some token has it, or there are no tokens yet to tell."""
    return not tokens or any(known(t, key) for t in tokens)


def tokens_html(tokens: list[dict], with_tips: bool, family: families.Family, lang: str) -> str:
    """Generated tokens with heat. Finished steps get a popover per token; while a step is still
    generating, only special tokens get one, and it just says what the token is for."""
    out = []
    for tok in tokens:
        cls = f"tok ch-{tok['channel']}" + (" special" if tok["special"] else "")
        if (tok.get("draft") or {}).get("accepted"):
            cls += " drafted"
        if tok["special"]:
            body, style = esc(tok["piece"]), ""
        else:
            body = esc(tok["text"] or family.show_piece(tok["piece"])).replace("\n", '<span class="nl">↵</span>\n')
            style = f' style="--h:{heat(tok["p"])}"' if known(tok, "p") else ""
        if with_tips:
            tip = tip_html(tok, family, lang)
        elif note := special_note(tok, family, lang):
            tip = f'<span class="tip"><span class="tip-piece">{esc(tok["piece"])}</span>{note}</span>'
        else:
            tip = ""
        out.append(f'<span class="{cls}"{style}>{body}{tip}</span>')
    return f'<div class="tokens">{"".join(out)}</div>'


def token_legend(s: dict, live: bool, lang: str, tokens: list[dict] = ()) -> str:
    """How to read the tokens. The heat scale and the candidates are mentioned only if the backend reports them
    for `tokens` (the run's tokens so far)."""
    if s.get("temperature", 1) <= 0:
        how = pick(lang, "temperature 0 = greedy：每一步都直接選機率最高的 token。", "temperature 0 = greedy: every token is simply the most likely one.")
    else:
        t, k, p = s.get("temperature"), s.get("top_k"), s.get("top_p")
        how = pick(
            lang,
            f"每一步：logits ÷ T {t} → 留前 {k or '全部'} 個 → 累積機率到 {p} 為止 → 抽樣。",
            f"Each token: logits ÷ T {t} → keep {f'the top {k}' if k else 'them all'} → up to a cumulative probability of {p} → sample.",
        )
    if reported(tokens, "top"):
        hint = pick(lang, "生成中，完成後可以滑過 token 看候選", "generating; hover over the tokens once it is done") if live else pick(
            lang, "滑過 token 看候選", "hover over a token to see the candidates"
        )
    else:
        hint = pick(lang, "生成中，完成後可以滑過 token 看它的 id", "generating; hover over the tokens once it is done") if live else pick(
            lang, "滑過 token 看它的 id", "hover over a token to see its id"
        )
    scale = f'<span class="heat-scale">{pick(lang, "機率高 <i></i> 低", "likely <i></i> unlikely")}</span>' if reported(tokens, "p") else ""
    drafts = s.get("mtp") or any(t.get("draft") for t in tokens)  # asked for in the sidebar, or reported by a server
    drafted = pick(
        lang, '<span><span class="tok drafted">底線</span> = MTP 草稿猜中</span>', '<span><span class="tok drafted">underline</span> = guessed by the MTP drafter</span>'
    ) if drafts else ""
    mtp = '<p class="tok-legend">' + pick(
        lang,
        "MTP：每個沒有底線的 token 之後，草稿模型先猜接下來幾個，主模型一次 forward 全部驗證；"
        "一串底線加上後面那個沒有底線的 token，是主模型同一次 forward 的產出。",
        "MTP: after each token without an underline, the drafter guesses the next few and the model checks them all in one forward pass; "
        "a run of underlined tokens and the plain token after it come from the same forward pass of the model.",
    ) + "</p>" if drafts else ""
    return (
        f'<p class="tok-legend">{scale}'
        + pick(
            lang,
            '<span><em>斜體</em> = 思考</span><span><span class="mono">等寬</span> = 工具呼叫</span>'
            '<span><span class="sp">深色</span> = 特殊 token，滑過看用途</span>',
            '<span><em>italic</em> = thinking</span><span><span class="mono">monospace</span> = tool call</span>'
            '<span><span class="sp">dark</span> = special token, hover to see what it is for</span>',
        )
        + f'{drafted}<span>{hint}</span></p>'
        f'<p class="tok-legend">{how}</p>{mtp}'
    )


def step_html(step: dict, data: dict, lang: str, live: bool) -> str:
    s, family = data["settings"], data["family"]
    calls = list(step["calls"].values())
    if calls:
        badge = pick(lang, f"呼叫工具 ×{len(calls)}", f"tool call ×{len(calls)}")
    elif step["done"]:
        badge = pick(lang, "回覆", "reply")
    else:
        badge = pick(lang, "進行中", "in progress")
    parts = [
        '<div class="step">'
        f'<div class="step-head"><span class="step-no">STEP {step["n"]}</span><span class="step-badge">{badge}</span>'
        f'<span class="step-time">{fmt_ms(step["t1"] - step["t0"])}</span></div>'
    ]

    ctx = step["context"]
    if ctx:
        prompt = ctx["prompt"]
        if step["n"] == 1:
            start = max(prompt.rfind(family.user_turn), 0)
            label = pick(lang, "最新的使用者訊息（前面還有 system、工具 schema 與對話歷史）",
                         "The newest user message (the system prompt, the tool schemas and the conversation so far come before it)")
        else:
            start = ctx["added_from"]
            label = pick(lang, "上一步的觀察接回脈絡", "The last step's observation, back in the context")
        roles = " · ".join(f"{r} {n}" for r, n in ctx["roles"].items())
        share = pct(ctx["n_tokens"] / ctx["context_length"])
        media = ctx.get("media_tokens")
        media = pick(lang, f'<span class="sep">|</span>其中 {media_line(media, lang)} tokens', f'<span class="sep">|</span>of which {media_line(media, lang)} tokens') if media else ""
        parts.append(
            '<section class="stage stage-context"><h4 class="stage-title">'
            + pick(lang, "脈絡 <small>Context · 這一步模型看到的全部</small>", "Context <small>everything the model sees in this step</small>")
            + f'</h4><p class="meta-line"><b>{ctx["n_tokens"]:,}</b> tokens · {pick(lang, f"上限的 {share}", f"{share} of the limit")} · {esc(roles)}{media}</p>'
            f'<p class="snippet-label">{label}</p><pre class="snippet">{special_html(prompt[start:], family, lang, last=1500)}</pre></section>'
        )

    if step["prefill"]:
        pf = step["prefill"]
        meta = (
            f'prefill <b>{pf["n_tokens"]:,}</b> tokens · {fmt_ms(pf["ms"])}<span class="sep">|</span>'
            + pick(lang, f'生成 <b>{len(step["tokens"])}</b> tokens', f'generated <b>{len(step["tokens"])}</b> tokens')
        )
        done = step["done"]
        if done:
            meta += f' · {done["tok_per_s"]} tok/s'
            if piece := done["stop_piece"]:
                stop_chip, meaning = chip(piece, family, lang), stop_meaning(piece, family, lang)
                meta += '<span class="sep">|</span>' + pick(lang, f"停在 {stop_chip} {meaning}", f"stopped at {stop_chip} ({meaning})")
            elif done["finish"] == "max_new_tokens":
                meta += '<span class="sep">|</span>' + pick(lang, "到達 token 上限", "hit the token limit")
            elif done["finish"] == "user_stopped":
                meta += '<span class="sep">|</span>' + pick(lang, "使用者中斷", "stopped by the user")
            if mtp := done.get("mtp"):
                rate = pct(mtp["accepted"] / mtp["drafted"]) if mtp["drafted"] else ""
                plain = max(done["n_tokens"] - 1, 0)
                meta += '<span class="sep">|</span>' + pick(
                    lang,
                    f'MTP 草稿收下 <b>{mtp["accepted"]}</b> / {mtp["drafted"]}{f"（{rate}）" if rate else ""} · '
                    f'主模型解碼 forward {mtp["forwards"]} 次（不開 MTP 要 {plain} 次）',
                    f'MTP kept <b>{mtp["accepted"]}</b> / {mtp["drafted"]} guesses{f" ({rate})" if rate else ""} · '
                    f'{mtp["forwards"]} decoding forward passes ({plain} without MTP)',
                )
        if s.get("mtp") or (step["done"] or {}).get("mtp"):  # asked for in the sidebar, or done by a server started with it
            how = pick(lang, "一次推論；草稿模型先猜，主模型一次驗證好幾個 token", "one inference; the drafter guesses, the model checks several tokens at once")
        else:
            how = pick(lang, "一次推論，一次一個 token", "one inference, one token at a time")
        parts.append(
            f'<section class="stage stage-model"><h4 class="stage-title">{pick(lang, "模型", "Model")} <small>LLM · {how}</small></h4>'
            f'<p class="meta-line">{meta}</p>{token_legend(s, live, lang, [t for st in data["steps"] for t in st["tokens"]])}'
            f'{tokens_html(step["tokens"], not live, family, lang)}</section>'
        )

    if step["parsed"]:
        body = []
        if not calls:
            if step["done"] and step["done"]["finish"] == "max_new_tokens":
                body.append('<p class="check bad">' + pick(lang, "沒有完整的工具呼叫，而且生成被 token 上限截斷",
                                                           "No complete tool call, and the generation was cut off at the token limit") + "</p>")
            else:
                body.append('<p class="check ok">' + pick(lang, "沒有工具呼叫 → 把回覆交給使用者，迴圈結束",
                                                          "No tool call → the reply goes to the user, and the loop ends") + "</p>")
        for c in calls:
            body.append(call_html(c["name"], c["arguments"]))
            if c["errors"]:
                errors = esc(pick(lang, "；", "; ").join(c["errors"]))
                body.append(f'<p class="check bad">✗ {errors} → ' + pick(lang, "把錯誤當作觀察送回模型", "the error goes back to the model as an observation") + "</p>")
            else:
                name = esc(family.name)
                body.append('<p class="check ok">✓ ' + pick(lang, f"從 {name} 格式解析成 JSON，符合工具 schema",
                                                            f"parsed from the {name} format into JSON; it matches the tool schema") + "</p>")
            if c["repeated"]:
                body.append('<p class="check warn">⚠ ' + pick(lang, "跟之前一模一樣的呼叫：迴圈可能在打轉",
                                                              "exactly the same call as before: the loop may be going around in circles") + "</p>")
            if c["approval"] == "pending":
                body.append('<p class="check warn">⏸ ' + pick(lang, "等你在左邊按「允許」或「拒絕」", "waiting for you to press Allow or Reject on the left") + "</p>")
            elif c["approval"] == "approved":
                body.append('<p class="check ok">✓ ' + pick(lang, "你允許執行", "you allowed it") + "</p>")
            elif c["approval"] == "rejected":
                body.append('<p class="check bad">✗ ' + pick(lang, "你拒絕了，harness 把拒絕當作觀察送回模型",
                                                             "you rejected it; the harness sends the refusal back to the model as an observation") + "</p>")
            elif c["approval"] == "timeout":
                body.append('<p class="check bad">✗ ' + pick(lang, "等待確認逾時，harness 當作拒絕並停止迴圈",
                                                             "the confirmation timed out; the harness took it as a refusal and stopped the loop") + "</p>")
            elif c["approval"] == "stopped":
                body.append('<p class="check warn">⏹ ' + pick(lang, "執行被停止，這個呼叫沒有執行", "the run was stopped, so this call did not run") + "</p>")
        parts.append(
            f'<section class="stage stage-harness"><h4 class="stage-title">Harness <small>{pick(lang, "解析 · 驗證 · 權限", "parse · check · ask")}</small></h4>'
            f'{"".join(body)}</section>'
        )

    env = []
    for c in calls:
        r = c["result"]
        if c["started"] and r is None:
            env.append(f'<p class="result-line">{esc(c["environment"][lang])} · {pick(lang, "執行中…", "running…")}</p>')
        elif r:
            if c["started"]:
                outcome = pick(lang, "回傳", "returned") if r["ok"] else pick(lang, "錯誤", "error")
                head = f'{esc(c["environment"][lang])} · {outcome} · {fmt_ms(r["ms"])}' + (pick(lang, " · 已截斷", " · truncated") if r["truncated"] else "")
            else:
                head = pick(lang, "沒有執行：harness 直接回報錯誤", "not run: the harness reports the error itself")
            env.append(f'<p class="result-line{"" if r["ok"] else " bad"}">{head}</p><pre class="snippet">{esc(pretty(r["result"]))}</pre>')
    if env:
        parts.append(
            '<section class="stage stage-env"><h4 class="stage-title">'
            + pick(lang, "環境 <small>Environment · 真正執行工具的地方</small>", "Environment <small>where the tools really run</small>")
            + f'</h4>{"".join(env)}</section>'
        )
    parts.append("</div>")
    return "".join(parts)


def call_html(name: str, arguments: dict | None) -> str:
    """A tool call as name(arguments JSON). An argument that runs over several lines, such as code, is shown as …
    in the JSON and written out below it."""
    args = arguments or {}
    blocks = {k: v for k, v in args.items() if isinstance(v, str) and "\n" in v.strip()}
    shown = json.dumps({k: "…" if k in blocks else v for k, v in args.items()} if arguments is not None else None, ensure_ascii=False)
    out = f'<p class="call-line"><span class="fn">{esc(name)}</span>({esc(shown)})</p>'
    return out + "".join(f'<p class="snippet-label">{esc(k)}</p><pre class="snippet arg">{esc(v)}</pre>' for k, v in blocks.items())


def error_text(data: dict, lang: str) -> str | None:
    """The run's error, in the page's language when the error said it in both (a backend's)."""
    error = data["error"]
    return error[lang] if isinstance(error, dict) else error


def stop_html(data: dict, lang: str) -> str:
    end = data["end"]
    title, desc = STOP_REASONS[lang].get(end["stop_reason"], (end["stop_reason"], ""))
    last = data["steps"][-1] if data["steps"] else None
    piece = last["done"]["stop_piece"] if last and last["done"] else None
    if end["stop_reason"] == "no_tool_call" and piece:
        stop_chip, meaning = chip(piece, data["family"], lang), stop_meaning(piece, data["family"], lang)
        desc += pick(lang, f" 最後停在 {stop_chip}（{meaning}）。", f" It stopped at {stop_chip} ({meaning}).")
    gen = sum(len(s["tokens"]) for s in data["steps"])
    error = f'<p class="check bad">{esc(error_text(data, lang))}</p>' if data["error"] else ""
    cls = "stop-card is-error" if end["stop_reason"] == "error" else "stop-card"
    totals = pick(lang, f'共 {len(data["steps"])} 步 · 生成 {gen:,} tokens', f'{n_steps(len(data["steps"]), lang)} · {gen:,} tokens generated')
    return (
        f'<div class="{cls}"><h4>{pick(lang, "停止：", "Stop: ")}{title}</h4><p>{desc}</p>{error}'
        f'<p class="meta-line">{totals} · {fmt_ms(end["t"])}</p></div>'
    )


# ---------- context tab ----------

def context_summary(step: dict, lang: str) -> str:
    ctx = step["context"]
    roles = " · ".join(f"{r} {n}" for r, n in ctx["roles"].items())
    share, chars = pct(ctx["n_tokens"] / ctx["context_length"]), len(ctx["prompt"])
    media = ctx.get("media_tokens")
    return f"{ctx['n_tokens']:,} / {ctx['context_length']:,} tokens" + pick(
        lang, f"（{share}）· {chars:,} 字元", f" ({share}) · {chars:,} characters"
    ) + f" · {roles}" + (pick(lang, f" · 其中 {media_line(media, lang)} tokens", f" · of which {media_line(media, lang)} tokens") if media else "")


def context_html(step: dict, family: families.Family, lang: str) -> str:
    prompt = step["context"]["prompt"]
    start = max(prompt.rfind(family.user_turn), 0) if step["n"] == 1 else step["context"]["added_from"]
    gen = step["done"]["raw"] if step["done"] else "".join(t["text"] for t in step["tokens"])
    return (
        f'<pre class="ctx-view">{special_html(prompt[:start], family, lang)}<mark class="added">{special_html(prompt[start:], family, lang)}</mark>'
        f'<span class="gen">{special_html(gen, family, lang)}</span></pre>'
    )


# ---------- model tab ----------

def model_html(info: dict, lang: str) -> str:
    """The model's card. A model a server runs (info["server"], vllm_backend.py) has no device, precision, parameter
    count or load time here, since its weights are not loaded on this machine; the card leaves those out."""
    family = families.get(info["family"])
    server = info.get("server")
    colors = {"language_model": "var(--c-model)", "audio_tower": "var(--c-env)", "vision_tower": "var(--c-context)", "visual": "var(--c-context)"}
    other = pick(lang, "其他", "other")
    parts = {PARAM_PARTS[lang].get(k, other): 0 for k in info.get("parameter_parts", {})}
    color_of = {}
    for key, n in info.get("parameter_parts", {}).items():
        label = PARAM_PARTS[lang].get(key, other)
        parts[label] += n
        color_of[label] = colors.get(key, "var(--muted)")
    bar = "".join(f'<span style="flex:{n};background:{color_of[label]}" title="{label}"></span>' for label, n in parts.items())
    legend = "".join(f'<li><i style="background:{color_of[label]}"></i>{label} {fmt_params(n)}</li>' for label, n in parts.items())
    specs = [
        (pick(lang, "伺服器", "server"), f'{server["engine"]} {server.get("version") or ""}'.strip()) if server else
        (pick(lang, "裝置", "device"), info["device"].upper()),
        (pick(lang, "精度", "precision"), info.get("dtype")),
        (pick(lang, "總參數", "parameters"), fmt_params(info["parameters"]) if info.get("parameters") else None),
        (pick(lang, "層數", "layers"), info["layers"]),
        (pick(lang, "隱藏層維度", "hidden size"), f"{info['hidden_size']:,}"),
        (pick(lang, "注意力頭數", "attention heads"), f"{info['attention_heads']} Q / {info['kv_heads']} KV"),
        (pick(lang, "層的種類", "layer types"), " · ".join(f"{LAYER_TYPES[lang].get(k, k)} {n}" for k, n in info["layer_types"].items())),
        (pick(lang, "滑動視窗", "sliding window"), f"{info['sliding_window']:,} tokens" if info["sliding_window"] else ""),
        (pick(lang, "脈絡上限", "context limit"), f"{info['context_length']:,} tokens"),
        (pick(lang, "詞彙量", "vocabulary"), f"{info['vocab_size']:,}"),
        (pick(lang, "讀得懂", "reads"), " · ".join(MODALITIES[lang][k] for k in ("text", *info.get("modalities", ())))),
        (pick(lang, "載入時間", "load time"), f"{info['load_seconds']} s" if info.get("load_seconds") is not None else None),
    ]
    spec_html = "".join(f"<div><dt>{k}</dt><dd>{esc(v)}</dd></div>" for k, v in specs if v)
    stops = "".join(f'<li><span class="sp">{esc(p)}</span> {stop_meaning(p, family, lang)}</li>' for p in info["stop_tokens"])
    name, arch = esc(family.name), esc(info["architecture"])
    if server:
        engine, url = esc(server["engine"]), esc(server["url"])
        mode = server.get("logprobs_mode", "raw_logprobs")
        if mode == "raw_logprobs":
            back = pick(lang, "每個 token 回來時帶著它的機率和前 5 名候選；取樣時剃掉了哪些候選，留在伺服器上。",
                        "Each token comes back with its probability and the top 5 candidates; which candidates the sampling cut stays on the server. ")
        elif mode == "processed_logprobs":
            back = pick(lang, "伺服器用 `--logprobs-mode processed_logprobs` 啟動：每個 token 帶回來的是它被抽出來時的機率"
                              "（套過 temperature、top-k、top-p 之後），不是模型原本的機率，所以看不到模型給的機率和前 5 名。",
                        "The server was started with `--logprobs-mode processed_logprobs`: each token comes back with its probability in the distribution "
                        "it was drawn from (after temperature, top-k and top-p), not the model's own, so the model probability and the top 5 do not show. ")
            if server.get("sampling_mask"):
                back += pick(lang, "它也送回 sampling mask，所以看得到剩下幾個候選。", "It also sends sampling masks, so how many candidates were left shows too. ")
            if server.get("refuses_greedy"):
                back += pick(lang, "它啟動時加了 `--return-sampling-mask`，所以不接受 temperature 0 和 top-k 0，側欄就從 0.05 和 1 開始。",
                             "It was started with `--return-sampling-mask`, so it refuses temperature 0 and top-k 0, and the sidebar starts at 0.05 and 1. ")
        else:
            back = pick(lang, f"伺服器用 `--logprobs-mode {mode}` 啟動，回來的是 logits 而不是機率，所以都不會當成機率顯示。",
                        f"The server was started with `--logprobs-mode {mode}`, so what comes back is logits, not probabilities, and none of them shows as one. ")
        where = pick(lang, f"{name} 家族，由 {url} 的 {engine} 伺服器跑（{arch}）：prompt 在這裡 render 成 token id 送過去。",
                     f"The {name} family, run by the {engine} server at {url} ({arch}): the prompt is rendered into token ids here and sent there. ") + prose_html(back)
    else:
        where = pick(lang, f"{name} 家族，在你的電腦上用 transformers 跑（{arch}）。", f"The {name} family, running on your computer with transformers ({arch}). ")
    return (
        '<div class="card">'
        f'<h4><a href="https://huggingface.co/{esc(info["model_id"])}" target="_blank" rel="noopener">{esc(info["model_id"])}</a></h4>'
        '<p class="card-sub">'
        + where
        + f'{prose_html(family.model_note.get(lang, ""))}</p>'
        f'<dl class="specs">{spec_html}</dl>'
        + refused_html(server, lang)
        + (f'<div class="params-bar">{bar}</div><ul class="params-legend">{legend}</ul>' if parts else "")
        + f'<p class="card-sub stops-label">{pick(lang, "停止 token：生成到這些 token 就交回 harness", "Stop tokens: generating one of these hands control back to the harness")}</p>'
        f'<ul class="stop-list">{stops}</ul>'
        f'{drafter_html(info, lang)}</div>'
    )


def refused_html(server: dict | None, lang: str) -> str:
    """The media the checkpoint reads but the server did not take (vllm_backend.VLLMBackend._probe), and why."""
    refused = (server or {}).get("refused")
    if not refused:
        return ""
    reasons: dict[str, list[str]] = {}  # one reason, such as no --enable-scale-out, often covers every kind
    for kind, why in refused.items():
        reasons.setdefault(why[lang], []).append(MODALITIES[lang][kind])
    items = "".join(f'<li><b>{pick(lang, "、", ", ").join(kinds)}</b>{pick(lang, "：", ": ")}{prose_html(why)}</li>' for why, kinds in reasons.items())
    return (
        f'<p class="card-sub stops-label">{pick(lang, "這個 checkpoint 讀得懂、但這台伺服器不收的媒體：", "Media the checkpoint reads but this server does not take:")}</p>'
        f'<ul class="refused-list">{items}</ul>'
    )


def drafter_html(info: dict, lang: str) -> str:
    """The MTP drafter that goes with the model, if it has one."""
    drafter_id = info.get("drafter_id")
    if not drafter_id:
        return ""
    link = f'<a href="https://huggingface.co/{esc(drafter_id)}" target="_blank" rel="noopener">{esc(drafter_id)}</a>'
    d = info.get("drafter")
    if not d:
        return '<p class="card-sub stops-label">' + pick(
            lang, f"MTP 草稿模型：{link}。打開側欄的「MTP 推測解碼」就會載入。", f"MTP drafter: {link}. Switching on MTP speculative decoding in the sidebar loads it."
        ) + "</p>"
    params = fmt_params(d["parameters"])
    return '<p class="card-sub stops-label">' + pick(
        lang,
        f'MTP 草稿模型：{link} · {params} 參數 · {d["layers"]} 層 · '
        f'隱藏層維度 {d["hidden_size"]} · 每次猜 {d["draft_tokens"]} 個 token · 載入 {d["load_seconds"]} s。'
        "它沒有自己的 KV cache：Gemma 4 後段的層本來就共用前面兩層算好的 key / value，草稿模型也讀同一份，"
        "再加上主模型最後一層的 hidden state，所以猜一個 token 只花主模型一次 forward 的一小部分時間。",
        f'MTP drafter: {link} · {params} parameters · {d["layers"]} layers · '
        f'hidden size {d["hidden_size"]} · guesses {d["draft_tokens"]} tokens at a time · loaded in {d["load_seconds"]} s. '
        "It has no KV cache of its own: Gemma 4's later layers already share the keys and values that two earlier layers computed, "
        "and the drafter reads those too, along with the model's last hidden state, so guessing a token takes a small fraction of one forward pass of the model.",
    ) + "</p>"


def knobs_html(s: dict, defaults: dict, lang: str) -> str:
    rows = [
        ("temperature", s["temperature"], pick(lang, "logits 除以 T 再 softmax。小於 1 讓分布更尖，大於 1 更平；0 等於 greedy。",
                                               "The logits are divided by T before the softmax. Below 1 sharpens the distribution, above 1 flattens it; 0 is greedy.")),
        ("top-k", s["top_k"] or pick(lang, "不限", "no limit"), pick(lang, "只保留機率最高的 k 個候選。", "Keeps only the k most likely candidates.")),
        ("top-p", s["top_p"], pick(lang, "從機率最高的開始累加，總和達到 p 就停（nucleus sampling）。",
                                   "Adds up probabilities from the most likely down and stops once the sum reaches p (nucleus sampling).")),
        (pick(lang, "MTP 推測解碼", "MTP speculative decoding"), pick(lang, "開", "on") if s.get("mtp") else pick(lang, "關", "off"),
         pick(lang, "草稿模型先猜接下來幾個 token，主模型一次 forward 全部驗證。收下的 token 跟不開時的分布一樣，只是少跑幾次 forward。",
              "The drafter guesses the next few tokens, and the model checks them all in one forward pass. "
              "The tokens kept follow the same distribution as without it; it just takes fewer forward passes.")),
        (pick(lang, "每步 token 上限", "Max tokens per step"), s["max_new_tokens"],
         pick(lang, "一次推論最多生成幾個 token，到了就截斷（相當於 max_tokens）。", "The most tokens one inference may generate before it is cut off (like max_tokens).")),
        (pick(lang, "步數上限", "Max steps"), s["max_steps"],
         pick(lang, "harness 的迴圈上限，跑滿就停（相當於最大迭代次數）。", "How many times the harness goes around the loop at most (like a maximum number of iterations).")),
    ]
    body = "".join(f"<div><dt>{k} <b>{esc(v)}</b></dt><dd>{d}</dd></div>" for k, v, d in rows)
    t, p, k = defaults["temperature"], defaults["top_p"], defaults["top_k"]
    return (
        f'<div class="card"><h4>{pick(lang, "解碼設定", "Decoding settings")}</h4><p class="card-sub">'
        + pick(lang, f"我們看到的「模型行為」其實是模型 × prompt × 解碼設定。模型建議值：T {t}、top-p {p}、top-k {k}。",
               f"What we see as the model's behavior is really model × prompt × decoding settings. The model recommends T {t}, top-p {p}, top-k {k}.")
        + f'</p><dl class="knobs">{body}</dl></div>'
    )


def step_rows(data: dict, lang: str) -> list[dict]:
    rows = []
    for s in data["steps"]:
        if not s["context"]:
            continue
        probs = [t["p"] for t in s["tokens"] if not t["special"] and known(t, "p")]
        done = s["done"] or {}
        rows.append({
            "step": s["n"],
            pick(lang, "脈絡 tokens", "context tokens"): s["context"]["n_tokens"],
            "prefill ms": s["prefill"]["ms"] if s["prefill"] else None,
            pick(lang, "生成 tokens", "generated tokens"): len(s["tokens"]),
            "tok/s": done.get("tok_per_s"),
            pick(lang, "平均機率", "mean probability"): round(sum(probs) / len(probs), 3) if probs else None,
            pick(lang, "停止", "stop"): done.get("stop_piece") or done.get("finish"),
        })
        if data["settings"].get("mtp") or any((st["done"] or {}).get("mtp") for st in data["steps"]):
            mtp = done.get("mtp") or {}
            rows[-1][pick(lang, "MTP 收下", "MTP kept")] = f'{mtp["accepted"]}/{mtp["drafted"]}' if mtp else None
            rows[-1][pick(lang, "解碼 forward", "decoding forwards")] = mtp.get("forwards")
    return rows


def tools_html(tools: dict, enabled: bool, family: families.Family, lang: str, offered: list[str] | None = None) -> str:
    """Every tool with its schema and environment; one not in `offered` (when given) is marked as kept from the model."""
    items = []
    for t in tools.values():
        params = ", ".join(
            f'{k}{"" if k in t.parameters.get("required", []) else "?"}: {v["type"]}' for k, v in t.parameters["properties"].items()
        )
        off = "" if offered is None or t.name in offered else '<p class="check warn">' + pick(
            lang, "這一輪沒有提供給模型，見側欄的設定", "not offered to the model in this turn; see the sidebar"
        ) + "</p>"
        items.append(
            f'<li><p class="call-line"><span class="fn">{esc(t.name)}</span>({esc(params)})</p>'
            f'<p class="tool-desc">{esc(t.description)}</p><p class="result-line">{pick(lang, "環境：", "environment: ")}{esc(t.environment[lang])}</p>{off}</li>'
        )
    note = "" if enabled else '<p class="check warn">' + pick(
        lang, "目前關閉了「使用工具」，模型這一輪看不到這些工具。", "Use tools is off, so the model cannot see these tools in this turn."
    ) + "</p>"
    schemas = prose_html(family.tools_note[lang])
    return (
        f'<div class="card"><h4>{pick(lang, "動作空間：模型看得到的工具", "Action space: the tools the model can see")}</h4>'
        '<p class="card-sub">'
        + pick(lang, f"每個工具是一份 JSON Schema。{schemas}可以在「脈絡」分頁看到。", f"Each tool is a JSON Schema. {schemas} You can see it in the Context tab.")
        + f'</p>{note}<ul class="tool-list">{"".join(items)}</ul></div>'
    )


def special_tokens_html(info: dict, lang: str) -> str:
    """Every special token of the model's tokenizer, grouped, with its id and what it is for."""
    family = families.get(info["family"])
    ids = info["special_tokens"]
    groups = [(title, sub, [p for p, _, _ in rows if p in ids]) for title, sub, rows in family.special_groups[lang]]
    extra = [p for p in ids if p not in family.special_tokens]
    if extra:
        groups.append((pick(lang, "其他", "Other"), pick(lang, "這個模型另外定義的", "defined by this model besides those"), extra))
    sections = []
    for title, sub, pieces in groups:
        if not pieces:
            continue
        rows = []
        for p in pieces:
            name, desc = family.special_tokens.get(p, {}).get(lang, (pick(lang, "沒有說明", "No note"), ""))
            stop = f'<i class="stop-tag">{pick(lang, "停止 token", "stop token")}</i>' if p in info["stop_tokens"] else ""
            rows.append(
                f'<div><dt><span class="sp">{esc(p)}</span><small>id {ids[p]}</small></dt>'
                f'<dd><b>{esc(name)}</b>{stop}<span>{prose_html(desc)}</span></dd></div>'
            )
        sections.append(f'<h5 class="gloss-head">{esc(title)} <small>{esc(sub)}</small></h5><dl class="glossary">{"".join(rows)}</dl>')
    return (
        f'<div class="card"><h4>{pick(lang, f"特殊 token（{len(ids)} 個）", f"Special tokens ({len(ids)})")}</h4><p class="card-sub">'
        + pick(
            lang,
            "tokenizer 保留給格式用的 token，每個都是詞彙表裡的一個 id，不是一串字元。"
            "chat template 用它們標出訊息、思考和工具；模型也靠寫出它們切換正在寫的東西、決定什麼時候停。"
            "在「迴圈」和「脈絡」分頁，滑過深色的 token 也會看到這裡的說明。",
            "Tokens the tokenizer keeps for the format; each one is a single id in the vocabulary, not a string of characters. "
            "The chat template marks messages, thinking and tools with them, and the model writes them to switch what it is writing and to decide when to stop. "
            "In the Loop and Context tabs, hovering over a dark token shows its note from here.",
        )
        + f'</p>{"".join(sections)}</div>'
    )


def file_kind(name: str) -> str | None:
    """Which entry of MODEL_FILES a file is; the shards of a model's weights all count as model.safetensors."""
    if name in MODEL_FILES:
        return name
    return "model.safetensors" if name.endswith(".safetensors") else None


def field_value(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 80 else text[:79] + "…"


def files_html(folder: dict, notes: dict, family: families.Family, lang: str, served: bool = False) -> str:
    """One folder's files, in MODEL_FILES order: size, what the kind of file is for, what the family's note says it
    holds and the values of the fields the note names. A file the family has a note on but the folder lacks is listed too;
    when a server runs the model (`served`), the weights it lacks are on the server."""
    present = {file_kind(f["name"]) or f["name"] for f in folder["files"]}
    rows = [(f["name"], f) for f in folder["files"]] + [(name, None) for name in notes if name not in present]
    order = list(MODEL_FILES)
    rows.sort(key=lambda row: (order.index(file_kind(row[0])) if file_kind(row[0]) else len(order), row[0]))
    out, said = [], set()
    for name, file in rows:
        kind = file_kind(name)
        label, what = MODEL_FILES[kind][lang] if kind else (pick(lang, "其他", "other"), pick(lang, "這一頁沒有它的說明。", "This page has no note on it."))
        note = notes.get(kind or name)
        held = f"<span>{esc(family.name)}{pick(lang, '：', ': ')}{prose_html(note.text[lang])}</span>" if note else ""
        if kind and kind in said:  # another shard of the weights: the notes are on the first
            what, held = pick(lang, "權重的另一個分片。", "Another shard of the weights."), ""
        said.add(kind)
        if file:
            size = fmt_bytes(file["bytes"])
        elif served and kind == "model.safetensors":
            size = pick(lang, "只在伺服器上", "on the server only")
        else:
            size = pick(lang, "這個 checkpoint 沒有", "not in this checkpoint")
        fields = "".join(f'<li><span class="tk">{esc(k)}</span> {esc(field_value(v))}</li>' for k, v in (file or {}).get("fields", {}).items())
        out.append(
            f'<div><dt><span class="tk">{esc(name)}</span><small>{size}</small></dt>'
            f'<dd><b>{label}</b><span>{prose_html(what)}</span>{held}'
            + (f'<ul class="file-fields">{fields}</ul>' if fields else "") + "</dd></div>"
        )
    return f'<dl class="glossary files">{"".join(out)}</dl>'


def folder_html(info: dict, lang: str) -> str:
    """The files in the model's folder, and in its MTP drafter's once that is loaded."""
    family = families.get(info["family"])
    folder = info["folder"]
    where = f'<span class="tk">{esc(folder["path"])}</span>'
    if folder["revision"]:
        rev = folder["revision"][:7]
        if families.REVISIONS.get(info["model_id"]) == folder["revision"]:
            which = pick(lang, f"從 Hub 下載的 checkpoint，revision {rev}：app 固定下載這個 commit（`REVISIONS`），所以每次下載都是同樣的檔案。",
                         f"A checkpoint downloaded from the Hub at revision {rev}, the commit the app pins it to (`REVISIONS`), so every download gets these same files. ")
        else:
            which = pick(lang, f"從 Hub 下載的 checkpoint，revision {rev}，是下載當時 `main` 指向的 commit；之後再下載可能拿到更新的。",
                         f"A checkpoint downloaded from the Hub at revision {rev}, the commit `main` pointed to then; a later download may get a newer one. ")
        how = which + pick(
            lang,
            "檔案放在 `snapshots/<revision>/`，每個都是連結，指向 `blobs/` 裡用 hash 命名的檔案（系統不能建連結時，例如沒開開發者模式的 Windows，就直接複製），"
            "所以兩個 revision 共用的檔案只存一份。同一個 revision 在每台電腦、每種作業系統上都是同樣的檔案，不同的只有載入的方式"
            + ("。" if "server" in info else "（上面的裝置和精度）。"),
            "The files sit in `snapshots/<revision>/`, each a link to a file in `blobs/` named by its hash (a copy where the system cannot make links, "
            "as on Windows without developer mode), so a file two revisions share is stored once. A revision is the same files on every computer and "
            "every operating system; only how they are loaded differs" + ("." if "server" in info else " (the device and precision above)."),
        )
    else:
        how = pick(lang, "一個磁碟上的 checkpoint，直接從這裡載入。", "A checkpoint on disk, loaded in place.")
    if server := info.get("server"):  # how goes through prose_html, which escapes it
        how += pick(
            lang,
            f"{server['engine']} 伺服器自己讀權重，所以這裡只下載了 tokenizer、chat template 和設定檔，用來把 prompt render 成 token id；"
            "伺服器啟動時用 `--revision` 指定同一個 commit，兩邊讀的就是同樣的檔案。",
            f" The {server['engine']} server reads the weights itself, so only the tokenizer, the chat template and the settings are downloaded "
            "here, to render the prompts into token ids; a server started with `--revision` at the same commit reads the same files.",
        )
    drafter = info.get("drafter") or {}
    extra = ""
    if drafter.get("folder"):
        extra = (
            f'<h5 class="gloss-head">{pick(lang, "MTP 草稿模型", "The MTP drafter")} <small>{esc(drafter["folder"]["path"])}</small></h5>'
            + files_html(drafter["folder"], family.drafter_file_notes, family, lang)
        )
    return (
        f'<div class="card"><h4>{pick(lang, "模型資料夾", "The model folder")}</h4><p class="card-sub">'
        + pick(lang, f"模型的每個檔案都在 {where}。", f"Every file of the model is in {where}. ")
        + prose_html(how)
        + pick(lang, "下面是每個檔案的大小和用途，以及它在這個家族的 checkpoint 裡裝了什麼；欄位的值是從檔案讀出來的。",
               " Below is each file's size and purpose, and what it holds in this family's checkpoints; the values of the fields are read from the files.")
        + f'</p>{files_html(folder, family.file_notes, family, lang, served="server" in info)}{extra}</div>'
    )
