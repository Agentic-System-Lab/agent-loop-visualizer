"""HTML for the visualization, built from the events of one agent run.

Streamlit redraws the page on every refresh, so everything here is a pure function
of the event list: build() folds the events into steps, and the other functions
turn steps into HTML strings for st.html.
"""

import base64
import html
import json
import re

SPECIAL_RE = re.compile(r"<\|[^<>|\s]*\|?>|<[a-z_]+\|>|<bos>|<eos>|<pad>")

STOP_REASONS = {
    "no_tool_call": ("模型沒有再呼叫工具", "模型這一步只回覆文字，harness 把它交給使用者，迴圈結束。"),
    "max_steps": ("達到步數上限", "harness 的防呆：迴圈跑滿設定的步數就停，不管模型想不想繼續。"),
    "max_new_tokens": ("達到單步 token 上限", "這一步生成的 token 數到了上限，回覆可能被截斷。"),
    "user_stopped": ("使用者中斷", "你按了停止，harness 在下一個 token 前停下。"),
    "format_errors": ("連續格式錯誤", "模型連續 3 次寫出無法解析的工具呼叫，harness 停止迴圈。"),
    "approval_timeout": ("等待確認逾時", "一直沒有人按「允許」或「拒絕」（例如分頁已經關掉或重新整理），harness 當作拒絕並停止迴圈，讓模型可以給其他分頁使用。"),
    "error": ("執行錯誤", "程式發生錯誤，訊息在下面。"),
}
STOP_TOKENS = {
    "<turn|>": "回合結束，相當於 end_turn",
    "<|tool_response>": "等工具結果，相當於 tool_use",
    "<eos>": "序列結束",
}
PARAM_PARTS = {"language_model": "語言模型", "audio_tower": "音訊編碼器", "vision_tower": "視覺編碼器"}

INTRO = """
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
"""

CONTEXT_LEGEND = """
<p class="legend">
  <span class="sp">特殊 token</span>
  <mark class="added">這一步新加入</mark>
  <span class="gen">模型這一步的輸出</span>
</p>
"""


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


def show_piece(piece: str) -> str:
    """SentencePiece marks a leading space with ▁; show it as ␣ so it reads as a space."""
    return piece.replace("▁", "␣").replace("\n", "↵")


def special_html(text: str) -> str:
    """Escape text and turn special tokens such as <|turn> into chips."""
    out, last = [], 0
    for m in SPECIAL_RE.finditer(text):
        out.append(esc(text[last:m.start()]))
        out.append(f'<span class="sp">{esc(m.group(0))}</span>')
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
    data = {"settings": {}, "steps": [], "end": None, "error": None, "last": None, "t": 0}
    for ev in events:
        kind = ev["type"]
        data["last"], data["t"] = kind, ev["t"]
        if kind == "run_start":
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


def status(data: dict) -> str:
    if not data["steps"]:
        return "啟動中…"
    step, last = data["steps"][-1], data["last"]
    n = step["n"]
    if last in ("step_start", "context"):
        return f"Step {n} · 把脈絡送進模型"
    if last == "prefill":
        return f"Step {n} · 模型讀完 {step['prefill']['n_tokens']:,} 個 token，開始生成"
    if last == "token":
        channel = step["tokens"][-1]["channel"]
        return f"Step {n} · " + {"thought": "模型思考中…", "tool_call": "模型正在寫工具呼叫…"}.get(channel, "模型生成中…")
    if last == "approval_request":
        return f"Step {n} · 等你確認要不要執行工具"
    if last == "tool_start":
        return f"Step {n} · 在環境執行 {list(step['calls'].values())[-1]['name']}"
    if last == "tool_result":
        return f"Step {n} · 觀察接回脈絡"
    return f"Step {n} · harness 解析與驗證"


def footer(data: dict) -> str:
    end = data["end"]
    title = STOP_REASONS.get(end["stop_reason"], (end["stop_reason"],))[0]
    return f"{len(data['steps'])} 步 · {fmt_ms(end['t'])} · 停止：{title}"


def run_stats(data: dict) -> str:
    s = data["settings"]
    steps = data["steps"]
    ctx = next((st["context"]["n_tokens"] for st in reversed(steps) if st["context"]), 0)
    gen = sum(len(st["tokens"]) for st in steps)
    return f"Step {len(steps)} / {s.get('max_steps', '?')} · 脈絡 {ctx:,} tokens · 共生成 {gen:,} tokens · {fmt_ms(data['t'])}"


# ---------- chat side ----------

def trace_html(data: dict) -> str:
    chips = []
    for c in all_calls(data):
        r = c["result"]
        if r is None:
            mark, ok = "…", ""
        else:
            mark = ("✓ " if r["ok"] else "✗ ") + (fmt_ms(r["ms"]) if c["started"] else "未執行")
            ok = "true" if r["ok"] else "false"
        chips.append(f'<span class="trace-chip" data-ok="{ok}">Step {c["step"]} · {esc(c["name"])} <span class="dot">{mark}</span></span>')
    return f'<div class="trace">{"".join(chips)}</div>' if chips else ""


# ---------- the loop diagram ----------

NODES = [
    ("context", 22, 16, "脈絡 Context", "system · 工具 · 對話"),
    ("model", 300, 16, "模型 LLM", "gemma-4-E2B-it"),
    ("harness", 300, 152, "Harness", "解析 · 驗證 · 權限"),
    ("env", 22, 152, "環境 Environment", "時鐘 · 計算 · Wikipedia"),
    ("stop", 560, 152, "停止", "回覆使用者"),
]
EDGES = [
    ("context-model", "M182 48 H294", 239, 38, "middle", "送入模型"),
    ("model-harness", "M380 80 V148", 390, 120, "start", "輸出 token"),
    ("harness-env", "M298 184 H188", 242, 174, "middle", "執行工具"),
    ("env-context", "M102 150 V84", 92, 120, "end", "觀察"),
    ("harness-stop", "M462 184 H554", 509, 174, "middle", "沒有工具呼叫"),
]


def loop_state(data: dict | None) -> dict:
    """Which node and edge are active, what each node's caption says, and the step counter."""
    state = {"active": None, "edge": None, "visited": set(), "waiting": False, "captions": {}, "counter": "等待問題", "ended": False}
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
            caps["model"] = f"已生成 {len(step['tokens'])} tokens"
        if step["done"]:
            caps["model"] = f"{step['done']['n_tokens']} tok · {step['done']['tok_per_s']} tok/s"
        calls = list(step["calls"].values())
        if step["parsed"]:
            caps["harness"] = "、".join(c["name"] for c in calls) or "沒有工具呼叫"
        if calls:
            c = calls[-1]
            if c["approval"] == "pending":
                caps["harness"] = "等你確認"
            if c["started"]:
                caps["env"] = c["environment"]
            if c["result"] and c["started"]:
                caps["env"] = f"{'完成' if c['result']['ok'] else '錯誤'} · {fmt_ms(c['result']['ms'])}"
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
        caps["stop"] = STOP_REASONS.get(data["end"]["stop_reason"], ("",))[0]
        state["counter"] = f"完成 · {len(steps)} 步"
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


def diagram_html(data: dict | None, dark: bool) -> str:
    """The loop diagram as an <img>: Streamlit's HTML sanitizer removes inline <svg>."""
    st = loop_state(data)
    pal = PALETTES["dark" if dark else "light"]
    css = DIAGRAM_CSS
    for key, value in pal.items():
        css = css.replace("{" + key + "}", value)
    edges = []
    for key, path, x, y, anchor, label in EDGES:
        cls = "edge on" if key == st["edge"] else "edge"
        edges.append(f'<g class="{cls}"><path d="{path}"/><text x="{x}" y="{y}" text-anchor="{anchor}">{label}</text></g>')
    nodes = []
    for key, x, y, title, caption in NODES:
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
    alt = "Agent 迴圈：脈絡送入模型，模型輸出交給 harness；有工具呼叫就在環境執行，觀察接回脈絡；沒有就停止。目前：" + esc(st["counter"])
    return f'<div class="diagram-wrap"><img class="diagram" src="{src}" alt="{alt}"></div>'


# ---------- one step of the loop ----------

def heat(p: float) -> str:
    return f"{min(1.0, max(0.0, 1 - p)) * 0.6:.3f}"


def tip_html(tok: dict) -> str:
    rows = []
    for c in tok["top"]:
        cls = "c" + (" on" if c["id"] == tok["id"] else "") + ("" if c["kept"] else " cut")
        rows.append(
            f'<span class="{cls}"><span class="cp">{esc(show_piece(c["piece"]))}</span>'
            f'<span class="b"><span style="width:{max(c["p"] * 100, 1):.1f}%"></span></span><span>{pct(c["p"])}</span></span>'
        )
    outside = "" if any(c["id"] == tok["id"] for c in tok["top"]) else f'<span class="tip-foot">抽到排名 #{tok["rank"]} 的 token，不在前 5 名。</span>'
    return (
        f'<span class="tip"><span class="tip-piece">{esc(show_piece(tok["piece"]))}</span>'
        f'<span class="tip-sub">token id {tok["id"]} · 第 {tok["index"] + 1} 個生成的 token</span>'
        f'<span class="tip-grid"><span>模型給的機率</span><b>{pct(tok["p"])}</b><span>排名</span><b>#{tok["rank"]}</b>'
        f'<span>取樣時的機率</span><b>{pct(tok["p_sampled"])}</b><span>剩下的候選</span><b>{tok["kept"]:,}</b></span>'
        f'<span class="tip-label">前 5 名候選（模型原始分布，灰色 = 被剃掉）</span>{"".join(rows)}{outside}</span>'
    )


def tokens_html(tokens: list[dict], with_tips: bool) -> str:
    out = []
    for tok in tokens:
        cls = f"tok ch-{tok['channel']}" + (" special" if tok["special"] else "")
        if tok["special"]:
            body, style = esc(tok["piece"]), ""
        else:
            body = esc(tok["text"] or tok["piece"]).replace("\n", '<span class="nl">↵</span>\n')
            style = f' style="--h:{heat(tok["p"])}"'
        tip = tip_html(tok) if with_tips else ""
        out.append(f'<span class="{cls}"{style}>{body}{tip}</span>')
    return f'<div class="tokens">{"".join(out)}</div>'


def token_legend(s: dict, live: bool) -> str:
    if s.get("temperature", 1) <= 0:
        how = "temperature 0 = greedy：每一步都直接選機率最高的 token。"
    else:
        how = f"每一步：logits ÷ T {s.get('temperature')} → 留前 {s.get('top_k') or '全部'} 個 → 累積機率到 {s.get('top_p')} 為止 → 抽樣。"
    hint = "生成中，完成後可以滑過 token 看候選" if live else "滑過 token 看候選"
    return (
        '<p class="tok-legend"><span class="heat-scale">機率高 <i></i> 低</span>'
        f'<span><em>斜體</em> = 思考</span><span><span class="mono">等寬</span> = 工具呼叫</span><span>{hint}</span></p>'
        f'<p class="tok-legend">{how}</p>'
    )


def step_html(step: dict, data: dict, live: bool) -> str:
    s = data["settings"]
    calls = list(step["calls"].values())
    if calls:
        badge = f"呼叫工具 ×{len(calls)}"
    elif step["done"]:
        badge = "回覆"
    else:
        badge = "進行中"
    parts = [
        '<div class="step">'
        f'<div class="step-head"><span class="step-no">STEP {step["n"]}</span><span class="step-badge">{badge}</span>'
        f'<span class="step-time">{fmt_ms(step["t1"] - step["t0"])}</span></div>'
    ]

    ctx = step["context"]
    if ctx:
        prompt = ctx["prompt"]
        if step["n"] == 1:
            start = max(prompt.rfind("<|turn>user"), 0)
            label = "最新的使用者訊息（前面還有 system、工具 schema 與對話歷史）"
        else:
            start = ctx["added_from"]
            label = "上一步的觀察接回脈絡"
        added = prompt[start:]
        if len(added) > 1500:
            added = "…" + added[-1500:]
        roles = " · ".join(f"{r} {n}" for r, n in ctx["roles"].items())
        share = ctx["n_tokens"] / ctx["context_length"]
        parts.append(
            '<section class="stage stage-context"><h4 class="stage-title">脈絡 <small>Context · 這一步模型看到的全部</small></h4>'
            f'<p class="meta-line"><b>{ctx["n_tokens"]:,}</b> tokens · 上限的 {pct(share)} · {esc(roles)}</p>'
            f'<p class="snippet-label">{label}</p><pre class="snippet">{special_html(added)}</pre></section>'
        )

    if step["prefill"]:
        pf = step["prefill"]
        meta = f'prefill <b>{pf["n_tokens"]:,}</b> tokens · {fmt_ms(pf["ms"])}<span class="sep">|</span>生成 <b>{len(step["tokens"])}</b> tokens'
        done = step["done"]
        if done:
            meta += f' · {done["tok_per_s"]} tok/s'
            if done["stop_piece"]:
                meta += f'<span class="sep">|</span>停在 <span class="sp">{esc(done["stop_piece"])}</span> {STOP_TOKENS.get(done["stop_piece"], "")}'
            elif done["finish"] == "max_new_tokens":
                meta += '<span class="sep">|</span>到達 token 上限'
            elif done["finish"] == "user_stopped":
                meta += '<span class="sep">|</span>使用者中斷'
        parts.append(
            '<section class="stage stage-model"><h4 class="stage-title">模型 <small>LLM · 一次推論，一次一個 token</small></h4>'
            f'<p class="meta-line">{meta}</p>{token_legend(s, live)}{tokens_html(step["tokens"], with_tips=not live)}</section>'
        )

    if step["parsed"]:
        body = []
        if not calls:
            if step["done"] and step["done"]["finish"] == "max_new_tokens":
                body.append('<p class="check bad">沒有完整的工具呼叫，而且生成被 token 上限截斷</p>')
            else:
                body.append('<p class="check ok">沒有工具呼叫 → 把回覆交給使用者，迴圈結束</p>')
        for c in calls:
            body.append(f'<p class="call-line"><span class="fn">{esc(c["name"])}</span>({esc(json.dumps(c["arguments"], ensure_ascii=False))})</p>')
            if c["errors"]:
                body.append(f'<p class="check bad">✗ {esc("；".join(c["errors"]))} → 把錯誤當作觀察送回模型</p>')
            else:
                body.append('<p class="check ok">✓ 從 Gemma 格式解析成 JSON，符合工具 schema</p>')
            if c["repeated"]:
                body.append('<p class="check warn">⚠ 跟之前一模一樣的呼叫：迴圈可能在打轉</p>')
            if c["approval"] == "pending":
                body.append('<p class="check warn">⏸ 等你在左邊按「允許」或「拒絕」</p>')
            elif c["approval"] == "approved":
                body.append('<p class="check ok">✓ 你允許執行</p>')
            elif c["approval"] == "rejected":
                body.append('<p class="check bad">✗ 你拒絕了，harness 把拒絕當作觀察送回模型</p>')
            elif c["approval"] == "timeout":
                body.append('<p class="check bad">✗ 等待確認逾時，harness 當作拒絕並停止迴圈</p>')
            elif c["approval"] == "stopped":
                body.append('<p class="check warn">⏹ 執行被停止，這個呼叫沒有執行</p>')
        parts.append(
            '<section class="stage stage-harness"><h4 class="stage-title">Harness <small>解析 · 驗證 · 權限</small></h4>'
            f'{"".join(body)}</section>'
        )

    env = []
    for c in calls:
        r = c["result"]
        if c["started"] and r is None:
            env.append(f'<p class="result-line">{esc(c["environment"])} · 執行中…</p>')
        elif r:
            if c["started"]:
                head = f'{esc(c["environment"])} · {"回傳" if r["ok"] else "錯誤"} · {fmt_ms(r["ms"])}' + (" · 已截斷" if r["truncated"] else "")
            else:
                head = "沒有執行：harness 直接回報錯誤"
            env.append(f'<p class="result-line{"" if r["ok"] else " bad"}">{head}</p><pre class="snippet">{esc(pretty(r["result"]))}</pre>')
    if env:
        parts.append(
            '<section class="stage stage-env"><h4 class="stage-title">環境 <small>Environment · 真正執行工具的地方</small></h4>'
            f'{"".join(env)}</section>'
        )
    parts.append("</div>")
    return "".join(parts)


def stop_html(data: dict) -> str:
    end = data["end"]
    title, desc = STOP_REASONS.get(end["stop_reason"], (end["stop_reason"], ""))
    last = data["steps"][-1] if data["steps"] else None
    piece = last["done"]["stop_piece"] if last and last["done"] else None
    if end["stop_reason"] == "no_tool_call" and piece:
        desc += f' 最後停在 <span class="sp">{esc(piece)}</span>（{STOP_TOKENS.get(piece, "")}）。'
    gen = sum(len(s["tokens"]) for s in data["steps"])
    error = f'<p class="check bad">{esc(data["error"])}</p>' if data["error"] else ""
    cls = "stop-card is-error" if end["stop_reason"] == "error" else "stop-card"
    return (
        f'<div class="{cls}"><h4>停止：{title}</h4><p>{desc}</p>{error}'
        f'<p class="meta-line">共 {len(data["steps"])} 步 · 生成 {gen:,} tokens · {fmt_ms(end["t"])}</p></div>'
    )


# ---------- context tab ----------

def context_summary(step: dict) -> str:
    ctx = step["context"]
    roles = " · ".join(f"{r} {n}" for r, n in ctx["roles"].items())
    return f"{ctx['n_tokens']:,} / {ctx['context_length']:,} tokens（{pct(ctx['n_tokens'] / ctx['context_length'])}）· {len(ctx['prompt']):,} 字元 · {roles}"


def context_html(step: dict) -> str:
    prompt = step["context"]["prompt"]
    start = max(prompt.rfind("<|turn>user"), 0) if step["n"] == 1 else step["context"]["added_from"]
    gen = step["done"]["raw"] if step["done"] else "".join(t["text"] for t in step["tokens"])
    return (
        f'<pre class="ctx-view">{special_html(prompt[:start])}<mark class="added">{special_html(prompt[start:])}</mark>'
        f'<span class="gen">{special_html(gen)}</span></pre>'
    )


# ---------- model tab ----------

def model_html(info: dict) -> str:
    total = info["parameters"]
    colors = {"language_model": "var(--c-model)", "audio_tower": "var(--c-env)", "vision_tower": "var(--c-context)"}
    parts = {PARAM_PARTS.get(k, "其他"): 0 for k in info["parameter_parts"]}
    color_of = {}
    for key, n in info["parameter_parts"].items():
        label = PARAM_PARTS.get(key, "其他")
        parts[label] += n
        color_of[label] = colors.get(key, "var(--muted)")
    bar = "".join(f'<span style="flex:{n};background:{color_of[label]}" title="{label}"></span>' for label, n in parts.items())
    legend = "".join(f'<li><i style="background:{color_of[label]}"></i>{label} {fmt_params(n)}</li>' for label, n in parts.items())
    specs = [
        ("裝置", info["device"].upper()),
        ("精度", info["dtype"]),
        ("總參數", fmt_params(total)),
        ("層數", info["layers"]),
        ("hidden size", f"{info['hidden_size']:,}"),
        ("attention heads", f"{info['attention_heads']} Q / {info['kv_heads']} KV"),
        ("sliding window", f"{info['sliding_window']:,} tokens"),
        ("脈絡上限", f"{info['context_length']:,} tokens"),
        ("詞彙量", f"{info['vocab_size']:,}"),
        ("載入時間", f"{info['load_seconds']} s"),
    ]
    spec_html = "".join(f"<div><dt>{k}</dt><dd>{esc(v)}</dd></div>" for k, v in specs)
    stops = "".join(f'<li><span class="sp">{esc(p)}</span> {STOP_TOKENS.get(p, "")}</li>' for p in info["stop_tokens"])
    return (
        '<div class="card">'
        f'<h4><a href="https://huggingface.co/{esc(info["model_id"])}" target="_blank" rel="noopener">{esc(info["model_id"])}</a></h4>'
        f'<p class="card-sub">在你的電腦上用 transformers 跑（{esc(info["architecture"])}）。官方標示 2.3B effective：'
        '每層各自的 embedding 表（Per-Layer Embeddings）只做查表，不算進有效參數。</p>'
        f'<dl class="specs">{spec_html}</dl>'
        f'<div class="params-bar">{bar}</div><ul class="params-legend">{legend}</ul>'
        f'<p class="card-sub stops-label">停止 token：生成到這些 token 就交回 harness</p><ul class="stop-list">{stops}</ul>'
        '</div>'
    )


def knobs_html(s: dict, defaults: dict) -> str:
    rows = [
        ("temperature", s["temperature"], "logits 除以 T 再 softmax。小於 1 讓分布更尖，大於 1 更平；0 等於 greedy。"),
        ("top-k", s["top_k"] or "不限", "只保留機率最高的 k 個候選。"),
        ("top-p", s["top_p"], "從機率最高的開始累加，總和達到 p 就停（nucleus sampling）。"),
        ("每步 token 上限", s["max_new_tokens"], "一次推論最多生成幾個 token，到了就截斷（相當於 max_tokens）。"),
        ("步數上限", s["max_steps"], "harness 的迴圈上限，跑滿就停（相當於最大迭代次數）。"),
    ]
    body = "".join(f"<div><dt>{k} <b>{esc(v)}</b></dt><dd>{d}</dd></div>" for k, v, d in rows)
    return (
        '<div class="card"><h4>解碼設定</h4>'
        '<p class="card-sub">我們看到的「模型行為」其實是模型 × prompt × 解碼設定（unit 01 第 1 節）。'
        f'模型建議值：T {defaults["temperature"]}、top-p {defaults["top_p"]}、top-k {defaults["top_k"]}。</p>'
        f'<dl class="knobs">{body}</dl></div>'
    )


def step_rows(data: dict) -> list[dict]:
    rows = []
    for s in data["steps"]:
        if not s["context"]:
            continue
        probs = [t["p"] for t in s["tokens"] if not t["special"]]
        done = s["done"] or {}
        rows.append({
            "step": s["n"],
            "脈絡 tokens": s["context"]["n_tokens"],
            "prefill ms": s["prefill"]["ms"] if s["prefill"] else None,
            "生成 tokens": len(s["tokens"]),
            "tok/s": done.get("tok_per_s"),
            "平均機率": round(sum(probs) / len(probs), 3) if probs else None,
            "停止": done.get("stop_piece") or done.get("finish"),
        })
    return rows


def tools_html(tools: dict, enabled: bool) -> str:
    items = []
    for t in tools.values():
        params = ", ".join(
            f'{k}{"" if k in t.parameters.get("required", []) else "?"}: {v["type"]}' for k, v in t.parameters["properties"].items()
        )
        items.append(
            f'<li><p class="call-line"><span class="fn">{esc(t.name)}</span>({esc(params)})</p>'
            f'<p class="tool-desc">{esc(t.description)}</p><p class="result-line">環境：{esc(t.environment)}</p></li>'
        )
    note = "" if enabled else '<p class="check warn">目前關閉了「使用工具」，模型這一輪看不到這些工具。</p>'
    return (
        '<div class="card"><h4>動作空間：模型看得到的工具</h4>'
        '<p class="card-sub">每個工具是一份 JSON Schema，chat template 會把它寫成 <span class="sp">&lt;|tool&gt;</span>declaration:…'
        '<span class="sp">&lt;tool|&gt;</span> 放進 system turn（可以在「脈絡」分頁看到）。</p>'
        f'{note}<ul class="tool-list">{"".join(items)}</ul></div>'
    )
