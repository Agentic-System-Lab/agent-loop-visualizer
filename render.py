"""HTML for the visualization, built from the events of one agent run.

Streamlit redraws the page on every refresh, so everything here is a pure function
of the event list: build() folds the events into steps, and the other functions
turn steps into HTML strings for st.html. Every function that writes words takes the
page's language, "zh" or "en" (i18n.py), so switching it redraws a finished run too.
"""

import base64
import html
import json
import re

import families
from i18n import pick

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
    "zh": {"sliding_attention": "sliding window", "full_attention": "full attention", "linear_attention": "線性 attention"},
    "en": {"sliding_attention": "sliding window", "full_attention": "full attention", "linear_attention": "linear attention"},
}
PARAM_PARTS = {
    "zh": {"language_model": "語言模型", "audio_tower": "音訊編碼器", "vision_tower": "視覺編碼器", "visual": "視覺編碼器"},
    "en": {"language_model": "language model", "audio_tower": "audio encoder", "vision_tower": "vision encoder", "visual": "vision encoder"},
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


def n_steps(n: int, lang: str) -> str:
    return pick(lang, f"{n} 步", f"{n} step" + ("" if n == 1 else "s"))


def show_piece(piece: str) -> str:
    """SentencePiece marks a leading space with ▁; show it as ␣ so it reads as a space."""
    return piece.replace("▁", "␣").replace("\n", "↵")


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


def chip(piece: str, family: families.Family, lang: str) -> str:
    """A special token as a dark chip; hovering it pops up what the token is for."""
    meaning = meaning_html(piece, family, lang)
    tip = f'<span class="sp-tip">{meaning}</span>' if meaning else ""
    return f'<span class="sp">{esc(piece)}{tip}</span>'


def special_html(text: str, family: families.Family, lang: str) -> str:
    """Escape text and turn the family's control tokens, such as <|turn>, into chips."""
    out, last = [], 0
    for m in family.special_re.finditer(text):
        out.append(esc(text[last:m.start()]))
        out.append(chip(m.group(0), family, lang))
        last = m.end()
    out.append(esc(text[last:]))
    return "".join(out)


def pretty(text: str) -> str:
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, TypeError):
        return text


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
            data["error"] = ev["message"]
        elif data["steps"]:
            step = data["steps"][-1]
            step["t1"] = ev["t"]
            if kind == "token":
                step["tokens"].append(ev)
            elif kind in ("context", "prefill", "parsed"):
                step[kind] = ev
            elif kind == "model_end":
                step["done"] = ev
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
    ("env", 22, 152, {"zh": ("環境 Environment", "時鐘 · 計算 · Wikipedia"), "en": ("Environment", "clock · math · wiki")}),
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


def draft_note(tok: dict, lang: str) -> str:
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
    guess = f'<span class="tk">{esc(show_piece(draft["piece"]))}</span>'
    return '<span class="tip-draft">' + pick(
        lang,
        f"<b>MTP：草稿被退回</b>草稿模型猜的是 {guess}，"
        "主模型沒有收下，這個 token 由主模型自己選（greedy 取第 1 名，抽樣時從其他候選裡抽），後面的猜測一起作廢。",
        f"<b>MTP: the guess was turned down</b>The drafter guessed {guess}, and the model did not keep it. "
        "The model picked this token itself (the top one in greedy decoding, otherwise drawn from the other candidates), and the guesses after it were dropped.",
    ) + "</span>"


def tip_html(tok: dict, family: families.Family, lang: str) -> str:
    rows = []
    for c in tok["top"]:
        cls = "c" + (" on" if c["id"] == tok["id"] else "") + ("" if c["kept"] else " cut")
        rows.append(
            f'<span class="{cls}"><span class="cp">{esc(show_piece(c["piece"]))}</span>'
            f'<span class="b"><span style="width:{max(c["p"] * 100, 1):.1f}%"></span></span><span>{pct(c["p"])}</span></span>'
        )
    outside = "" if any(c["id"] == tok["id"] for c in tok["top"]) else (
        '<span class="tip-foot">' + pick(lang, f'抽到排名 #{tok["rank"]} 的 token，不在前 5 名。', f'Drew the token ranked #{tok["rank"]}, outside the top 5.') + "</span>"
    )
    n = tok["index"] + 1
    top = pick(lang, "前 5 名候選（模型原始分布，灰色 = 被剃掉）", "top 5 candidates (the model's own distribution; gray = cut)")
    return (
        f'<span class="tip"><span class="tip-piece">{esc(show_piece(tok["piece"]))}</span>'
        f'<span class="tip-sub">token id {tok["id"]} · {pick(lang, f"第 {n} 個生成的 token", f"generated token #{n}")}</span>'
        f'{special_note(tok, family, lang)}{draft_note(tok, lang)}'
        f'<span class="tip-grid"><span>{pick(lang, "模型給的機率", "model probability")}</span><b>{pct(tok["p"])}</b>'
        f'<span>{pick(lang, "排名", "rank")}</span><b>#{tok["rank"]}</b>'
        f'<span>{pick(lang, "取樣時的機率", "sampling probability")}</span><b>{pct(tok["p_sampled"])}</b>'
        f'<span>{pick(lang, "剩下的候選", "candidates left")}</span><b>{tok["kept"]:,}</b></span>'
        f'<span class="tip-label">{top}</span>{"".join(rows)}{outside}</span>'
    )


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
            body = esc(tok["text"] or tok["piece"]).replace("\n", '<span class="nl">↵</span>\n')
            style = f' style="--h:{heat(tok["p"])}"'
        if with_tips:
            tip = tip_html(tok, family, lang)
        elif note := special_note(tok, family, lang):
            tip = f'<span class="tip"><span class="tip-piece">{esc(tok["piece"])}</span>{note}</span>'
        else:
            tip = ""
        out.append(f'<span class="{cls}"{style}>{body}{tip}</span>')
    return f'<div class="tokens">{"".join(out)}</div>'


def token_legend(s: dict, live: bool, lang: str) -> str:
    if s.get("temperature", 1) <= 0:
        how = pick(lang, "temperature 0 = greedy：每一步都直接選機率最高的 token。", "temperature 0 = greedy: every token is simply the most likely one.")
    else:
        t, k, p = s.get("temperature"), s.get("top_k"), s.get("top_p")
        how = pick(
            lang,
            f"每一步：logits ÷ T {t} → 留前 {k or '全部'} 個 → 累積機率到 {p} 為止 → 抽樣。",
            f"Each token: logits ÷ T {t} → keep {f'the top {k}' if k else 'them all'} → up to a cumulative probability of {p} → sample.",
        )
    hint = pick(lang, "生成中，完成後可以滑過 token 看候選", "generating; hover over the tokens once it is done") if live else pick(
        lang, "滑過 token 看候選", "hover over a token to see the candidates"
    )
    drafted = pick(
        lang, '<span><span class="tok drafted">底線</span> = MTP 草稿猜中</span>', '<span><span class="tok drafted">underline</span> = guessed by the MTP drafter</span>'
    ) if s.get("mtp") else ""
    mtp = '<p class="tok-legend">' + pick(
        lang,
        "MTP：每個沒有底線的 token 之後，草稿模型先猜接下來幾個，主模型一次 forward 全部驗證；"
        "一串底線加上後面那個沒有底線的 token，是主模型同一次 forward 的產出。",
        "MTP: after each token without an underline, the drafter guesses the next few and the model checks them all in one forward pass; "
        "a run of underlined tokens and the plain token after it come from the same forward pass of the model.",
    ) + "</p>" if s.get("mtp") else ""
    return (
        f'<p class="tok-legend"><span class="heat-scale">{pick(lang, "機率高 <i></i> 低", "likely <i></i> unlikely")}</span>'
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
        added = prompt[start:]
        if len(added) > 1500:
            added = "…" + added[-1500:]
        roles = " · ".join(f"{r} {n}" for r, n in ctx["roles"].items())
        share = pct(ctx["n_tokens"] / ctx["context_length"])
        parts.append(
            '<section class="stage stage-context"><h4 class="stage-title">'
            + pick(lang, "脈絡 <small>Context · 這一步模型看到的全部</small>", "Context <small>everything the model sees in this step</small>")
            + f'</h4><p class="meta-line"><b>{ctx["n_tokens"]:,}</b> tokens · {pick(lang, f"上限的 {share}", f"{share} of the limit")} · {esc(roles)}</p>'
            f'<p class="snippet-label">{label}</p><pre class="snippet">{special_html(added, family, lang)}</pre></section>'
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
        if s.get("mtp"):
            how = pick(lang, "一次推論；草稿模型先猜，主模型一次驗證好幾個 token", "one inference; the drafter guesses, the model checks several tokens at once")
        else:
            how = pick(lang, "一次推論，一次一個 token", "one inference, one token at a time")
        parts.append(
            f'<section class="stage stage-model"><h4 class="stage-title">{pick(lang, "模型", "Model")} <small>LLM · {how}</small></h4>'
            f'<p class="meta-line">{meta}</p>{token_legend(s, live, lang)}{tokens_html(step["tokens"], not live, family, lang)}</section>'
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
            body.append(f'<p class="call-line"><span class="fn">{esc(c["name"])}</span>({esc(json.dumps(c["arguments"], ensure_ascii=False))})</p>')
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


def stop_html(data: dict, lang: str) -> str:
    end = data["end"]
    title, desc = STOP_REASONS[lang].get(end["stop_reason"], (end["stop_reason"], ""))
    last = data["steps"][-1] if data["steps"] else None
    piece = last["done"]["stop_piece"] if last and last["done"] else None
    if end["stop_reason"] == "no_tool_call" and piece:
        stop_chip, meaning = chip(piece, data["family"], lang), stop_meaning(piece, data["family"], lang)
        desc += pick(lang, f" 最後停在 {stop_chip}（{meaning}）。", f" It stopped at {stop_chip} ({meaning}).")
    gen = sum(len(s["tokens"]) for s in data["steps"])
    error = f'<p class="check bad">{esc(data["error"])}</p>' if data["error"] else ""
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
    return f"{ctx['n_tokens']:,} / {ctx['context_length']:,} tokens" + pick(
        lang, f"（{share}）· {chars:,} 字元", f" ({share}) · {chars:,} characters"
    ) + f" · {roles}"


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
    family = families.get(info["family"])
    total = info["parameters"]
    colors = {"language_model": "var(--c-model)", "audio_tower": "var(--c-env)", "vision_tower": "var(--c-context)", "visual": "var(--c-context)"}
    other = pick(lang, "其他", "other")
    parts = {PARAM_PARTS[lang].get(k, other): 0 for k in info["parameter_parts"]}
    color_of = {}
    for key, n in info["parameter_parts"].items():
        label = PARAM_PARTS[lang].get(key, other)
        parts[label] += n
        color_of[label] = colors.get(key, "var(--muted)")
    bar = "".join(f'<span style="flex:{n};background:{color_of[label]}" title="{label}"></span>' for label, n in parts.items())
    legend = "".join(f'<li><i style="background:{color_of[label]}"></i>{label} {fmt_params(n)}</li>' for label, n in parts.items())
    specs = [
        (pick(lang, "裝置", "device"), info["device"].upper()),
        (pick(lang, "精度", "precision"), info["dtype"]),
        (pick(lang, "總參數", "parameters"), fmt_params(total)),
        (pick(lang, "層數", "layers"), info["layers"]),
        ("hidden size", f"{info['hidden_size']:,}"),
        ("attention heads", f"{info['attention_heads']} Q / {info['kv_heads']} KV"),
        (pick(lang, "層的種類", "layer types"), " · ".join(f"{LAYER_TYPES[lang].get(k, k)} {n}" for k, n in info["layer_types"].items())),
        ("sliding window", f"{info['sliding_window']:,} tokens" if info["sliding_window"] else ""),
        (pick(lang, "脈絡上限", "context limit"), f"{info['context_length']:,} tokens"),
        (pick(lang, "詞彙量", "vocabulary"), f"{info['vocab_size']:,}"),
        (pick(lang, "載入時間", "load time"), f"{info['load_seconds']} s"),
    ]
    spec_html = "".join(f"<div><dt>{k}</dt><dd>{esc(v)}</dd></div>" for k, v in specs if v)
    stops = "".join(f'<li><span class="sp">{esc(p)}</span> {stop_meaning(p, family, lang)}</li>' for p in info["stop_tokens"])
    name, arch = esc(family.name), esc(info["architecture"])
    return (
        '<div class="card">'
        f'<h4><a href="https://huggingface.co/{esc(info["model_id"])}" target="_blank" rel="noopener">{esc(info["model_id"])}</a></h4>'
        '<p class="card-sub">'
        + pick(lang, f"{name} 家族，在你的電腦上用 transformers 跑（{arch}）。", f"The {name} family, running on your computer with transformers ({arch}). ")
        + f'{prose_html(family.model_note.get(lang, ""))}</p>'
        f'<dl class="specs">{spec_html}</dl>'
        f'<div class="params-bar">{bar}</div><ul class="params-legend">{legend}</ul>'
        f'<p class="card-sub stops-label">{pick(lang, "停止 token：生成到這些 token 就交回 harness", "Stop tokens: generating one of these hands control back to the harness")}</p>'
        f'<ul class="stop-list">{stops}</ul>'
        f'{drafter_html(info, lang)}</div>'
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
        f'hidden size {d["hidden_size"]} · 每次猜 {d["draft_tokens"]} 個 token · 載入 {d["load_seconds"]} s。'
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
        probs = [t["p"] for t in s["tokens"] if not t["special"]]
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
        if data["settings"].get("mtp"):
            mtp = done.get("mtp") or {}
            rows[-1][pick(lang, "MTP 收下", "MTP kept")] = f'{mtp["accepted"]}/{mtp["drafted"]}' if mtp else None
            rows[-1][pick(lang, "解碼 forward", "decoding forwards")] = mtp.get("forwards")
    return rows


def tools_html(tools: dict, enabled: bool, family: families.Family, lang: str) -> str:
    items = []
    for t in tools.values():
        params = ", ".join(
            f'{k}{"" if k in t.parameters.get("required", []) else "?"}: {v["type"]}' for k, v in t.parameters["properties"].items()
        )
        items.append(
            f'<li><p class="call-line"><span class="fn">{esc(t.name)}</span>({esc(params)})</p>'
            f'<p class="tool-desc">{esc(t.description)}</p><p class="result-line">{pick(lang, "環境：", "environment: ")}{esc(t.environment[lang])}</p></li>'
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
