"""The 運作說明 page: one user turn through the agent loop, with the code behind every part.

The content points at code by text, not by line number: Code("agent.py", "def run(self):") is
looked up in the file every time the page is drawn, so the line numbers on the page follow the
code when it changes. A start text must match exactly one line of its file; tests/test_guide.py
checks every reference, so an edit that breaks one fails the tests instead of the page.
"""

import ast
import html
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).parent
STAGES = {
    "ui": ("畫面", "var(--ink-2)"),
    "harness": ("Harness", "var(--c-harness)"),
    "context": ("脈絡", "var(--c-context)"),
    "model": ("模型", "var(--c-model)"),
    "env": ("環境", "var(--c-env)"),
    "stop": ("停止", "var(--c-stop)"),
}
REF_RE = re.compile(r"\[\[([\w./]+)\|(.+?)\]\]")


@dataclass(frozen=True)
class Code:
    """An excerpt: from the line containing `start` through the statement that begins there, or
    through the statement on the first line at or after it that contains `end`."""

    file: str
    start: str
    end: str | None = None
    caption: str = ""


@dataclass(frozen=True)
class Section:
    key: str
    stage: str
    title: str
    lead: str
    body: tuple  # paragraphs (str), bullet lists (tuple of str) and Code excerpts, in reading order


# ---------- finding code ----------

def source(file: str) -> tuple[list[str], ast.Module]:
    text = (ROOT / file).read_text(encoding="utf-8")
    return text.splitlines(), ast.parse(text)


def find_line(file: str, text: str) -> int:
    """The 1-based number of the only line in `file` that contains `text`."""
    lines, _ = source(file)
    hits = [n for n, line in enumerate(lines, 1) if text in line]
    if len(hits) != 1:
        raise LookupError(f"{file}: {text!r} is on {len(hits)} lines, expected exactly 1")
    return hits[0]


def statement_end(tree: ast.Module, line: int) -> int:
    """Last line of the outermost statement that starts on `line`; failing that, of the innermost
    statement around it (the line is the middle of a call or a dict); failing that, the line."""
    statements = [n for n in ast.walk(tree) if isinstance(n, ast.stmt)]
    starting = [n for n in statements if n.lineno == line]
    if starting:
        return max(n.end_lineno for n in starting)
    around = [n for n in statements if n.lineno <= line <= n.end_lineno and not hasattr(n, "body")]
    return min(around, key=lambda n: n.end_lineno - n.lineno).end_lineno if around else line


def span(code: Code) -> tuple[int, int]:
    lines, tree = source(code.file)
    lo = find_line(code.file, code.start)
    if code.end is None:
        return lo, statement_end(tree, lo)
    last = next((n for n in range(lo, len(lines) + 1) if code.end in lines[n - 1]), None)
    if last is None:
        raise LookupError(f"{code.file}: {code.end!r} is not on or after line {lo}")
    return lo, statement_end(tree, last)


# ---------- HTML ----------

def esc(value) -> str:
    return html.escape(str(value), quote=True)


def ref_html(file: str, text: str) -> str:
    """An inline file:line chip; hovering it shows the line itself."""
    line = find_line(file, text)
    src = source(file)[0][line - 1].strip()
    return f'<code class="ref" title="{esc(src)}">{esc(file)}:{line}</code>'


def xref_html(key: str) -> str:
    """A link to another section by its key, with the section's current number and title."""
    n, sec = next((n, sec) for n, sec in enumerate(SECTIONS, 1) if sec.key == key)
    return f'<a class="g-xref" href="#g-{key}">第 {n} 段「{esc(sec.title)}」</a>'


def _inline(text: str) -> str:
    out = html.escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    return re.sub(r"\{\{(\w+)\}\}", lambda m: xref_html(m.group(1)), out)


def prose_html(text: str) -> str:
    """Escape a paragraph and mark it up: `code`, **bold**, [[file|text]] becomes a file:line chip,
    {{key}} a link to that section."""
    out, last = [], 0
    for m in REF_RE.finditer(text):
        out.append(_inline(text[last:m.start()]))
        out.append(ref_html(m.group(1), m.group(2)))
        last = m.end()
    out.append(_inline(text[last:]))
    return "".join(out)


def code_html(code: Code) -> str:
    lo, hi = span(code)
    lines = source(code.file)[0][lo - 1:hi]
    body = highlight(textwrap.dedent("\n".join(lines)), PythonLexer(), HtmlFormatter(nowrap=True))
    rows = body.split("\n")[: hi - lo + 1]
    gutter = "".join(f'<span class="ln">{n}</span>{row}\n' for n, row in zip(range(lo, hi + 1), rows))
    where = f"第 {lo} 行" if lo == hi else f"第 {lo}–{hi} 行"
    caption = f'<span class="src-cap">{prose_html(code.caption)}</span>' if code.caption else ""
    return (
        f'<figure class="src"><figcaption><b>{esc(code.file)}</b><span class="src-lines">{where}</span>{caption}</figcaption>'
        f"<pre>{gutter}</pre></figure>"
    )


def body_html(item) -> str:
    if isinstance(item, Code):
        return code_html(item)
    if isinstance(item, tuple):
        return "<ul>" + "".join(f"<li>{prose_html(i)}</li>" for i in item) + "</ul>"
    return f"<p>{prose_html(item)}</p>"


def section_html(n: int, sec: Section) -> str:
    label, color = STAGES[sec.stage]
    return (
        f'<section class="g-sec" id="g-{sec.key}" style="--stage: {color}">'
        f'<header class="g-head"><span class="g-num">{n}</span><div><p class="g-stage">{label}</p>'
        f'<h2>{esc(sec.title)}</h2><p class="g-lead">{prose_html(sec.lead)}</p></div></header>'
        f'{"".join(body_html(item) for item in sec.body)}</section>'
    )


def toc_html() -> str:
    items = []
    for n, sec in enumerate(SECTIONS, 1):
        label, color = STAGES[sec.stage]
        items.append(
            f'<li style="--stage: {color}"><a href="#g-{sec.key}"><span class="g-num">{n}</span>'
            f'<span>{esc(sec.title)}</span><small>{label}</small></a></li>'
        )
    items.append('<li style="--stage: var(--muted)"><a href="#g-events"><span class="g-num">＃</span><span>事件一覽</span><small>參考</small></a></li>')
    return f'<nav class="g-toc"><ol>{"".join(items)}</ol></nav>'


def events_html() -> str:
    rows = "".join(
        f"<tr><td><code>{esc(kind)}</code></td><td>{ref_html(file, text)}</td><td>{prose_html(carries)}</td><td>{prose_html(shows)}</td></tr>"
        for kind, file, text, carries, shows in EVENTS
    )
    return (
        '<section class="g-sec" id="g-events" style="--stage: var(--muted)">'
        '<header class="g-head"><span class="g-num">＃</span><div><p class="g-stage">參考</p><h2>事件一覽</h2>'
        '<p class="g-lead">harness 每走到一個階段就 append 一個事件；畫面上的一切都是從這張列表畫出來的。'
        "依一輪裡出現的順序排列。</p></div></header>"
        '<div class="g-table"><table><thead><tr><th>事件</th><th>在哪裡送出</th><th>帶了什麼</th><th>畫面上的用途</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div></section>"
    )


def page_html(diagram: str) -> str:
    sections = "".join(section_html(n, sec) for n, sec in enumerate(SECTIONS, 1))
    return (
        '<div class="guide"><header class="g-hero"><p class="g-kicker">運作說明</p>'
        "<h1>一個問題在 agent 迴圈裡怎麼走</h1>"
        f'<p class="g-intro">{prose_html(INTRO)}</p></header>'
        f'<div class="g-diagram">{diagram}</div>{toc_html()}{sections}{events_html()}</div>'
    )


# ---------- content ----------

INTRO = (
    "從你按下送出到看到回覆，中間發生的每一件事，照實際執行的順序走一遍。側欄可以選 Gemma 4 E2B 或 Qwen3.5 0.8B，"
    "兩個模型跑的是同一個迴圈，不一樣的地方會並排說明。每段都附上做這件事的程式碼；`agent.py:98` 這類標記是檔名和行號，"
    "滑過去會顯示那一行。行號是打開這一頁時從原始碼查出來的，程式改了也會跟著更新。"
)

SECTIONS = [
    Section("overview", "ui", "總覽：兩條 thread、一張事件列表", "你送出一個問題就是一輪（一個 `Run`）；一輪裡是一個迴圈，每一圈叫一個 step。", (
        "每個 step 做同一件事：harness 把整段對話 render 成一個字串（脈絡）→ 模型一次推論、一個 token 一個 token 生成 → "
        "harness 解析輸出。有工具呼叫，就驗證、（需要時問你）、在環境執行，把結果接回對話，進下一圈；"
        "沒有工具呼叫，就把回覆交給你，這一輪結束。",
        "整個 app 有兩條 thread。Streamlit 的 script thread 負責畫面；agent 跑在另一條背景 thread"
        "（[[streamlit_app.py|self.thread = threading.Thread]]）。兩邊只透過三樣東西溝通：",
        (
            "agent 每走到一個階段，就把一個事件 append 到 `events` 列表（[[agent.py|def emit(self, event_type]]）；畫面只負責把這張列表畫出來。",
            "按「停止」時，畫面呼叫 `Run.stop()`（[[agent.py|def stop(self):]]）。",
            "按「允許」或「拒絕」時，畫面呼叫 `Run.decide()`（[[agent.py|def decide(self, call_id]]）。",
        ),
        "每個檔案的分工：",
        (
            "`streamlit_app.py`：版面、側欄設定、模型選單、背景 thread、執行中每 0.5 秒重畫一次。",
            "`agent.py`：harness，也就是迴圈本體 `Run.run()`（[[agent.py|def run(self):]]）。",
            "`llm.py`：本機推論，自己寫的解碼迴圈 `generate()`（[[llm.py|def generate(self, prompt]]）和抽樣 `_sample()`。",
            "`families/`：跟著模型走的東西。`gemma4.py`、`qwen3_5.py` 各自負責讀懂自己模型的輸出，並說明它的特殊 token（{{families}}）。",
            "`tools.py`：每個工具的 schema（給模型看）和 Python 函式（給 harness 執行），全部列在 `TOOLS`（[[tools.py|TOOLS = {]]）。",
            "`render.py`：把事件列表折成 step（[[render.py|def build(events: list[dict])]]），再畫成 HTML。",
        ),
    )),
    Section("families", "model", "模型家族：換模型只換這一層", "harness、解碼迴圈和畫面都只寫一次；每個模型不一樣的地方，集中在一個 `Family` 物件裡。", (
        "兩個模型的迴圈完全一樣，不一樣的是它們寫字的格式：",
        (
            "回合：Gemma 用 `<|turn>user` … `<turn|>`；Qwen 用 ChatML 的 `<|im_start|>user` … `<|im_end|>`。",
            "工具呼叫：Gemma 寫 `<|tool_call>call:名稱{key:<|\"|>值<|\"|>}<tool_call|>`；Qwen 寫 XML："
            "`<tool_call>` 裡一個 `<function=名稱>`，每個參數一段 `<parameter=名稱>`，值一律是純文字。",
            "工具結果：Gemma 的結果接在同一個 model turn 裡（`<|tool_response>` … `<tool_response|>`）；"
            "Qwen 的結果放進一則新的 `user` 訊息（`<tool_response>` … `</tool_response>`），再開一個新的 assistant turn。",
            "停止 token：Gemma 回答完停在 `<turn|>`、要用工具時停在 `<|tool_response>`，看停止 token 就知道是哪一種；"
            "Qwen 兩種都停在 `<|im_end|>`，要看輸出裡有沒有工具呼叫才知道。",
            "思考：Gemma 在 system turn 開頭放 `<|think|>`，推理寫在 `<|channel>thought` 到 `<channel|>` 之間；"
            "Qwen 由 chat template 在 assistant 訊息開頭先寫好 `<think>`，推理寫到 `</think>`。",
        ),
        "這些差異全部寫進 `Family`，其他程式只問 family，沒有任何 `if 是 Gemma` 的分支：",
        Code("families/__init__.py", "class Family:", end="def special_re(self)",
             caption="欄位分三組：讀模型輸出用的、把對話寫回 chat template 用的、畫面用的。"),
        "載入模型時，用 checkpoint 的 `config.json` 裡的 `model_type` 找出 family。停止 token 是 generation config 列的，再加上 family 宣告的："
        "Qwen3.5 的 checkpoint 只列了 `<|endoftext|>`，沒有列每個回合結尾的 `<|im_end|>`，少了 family 這一層，模型回答完會繼續寫下去。"
        "Qwen 的 `<think>`、`<tool_call>` 在 tokenizer 裡也沒有標成 special，同樣由 family 補上。",
        Code("llm.py", "self.family = families.for_model_type(", end="self.special_ids = set("),
        "要加模型時：同一個家族的另一個 checkpoint（例如更大的 Gemma 4），在 `MODELS` 加一行就好（[[families/__init__.py|MODELS = {]]）。"
        "新的家族，照 `families/qwen3_5.py` 寫一個模組，內容是一個輸出的 parser 和特殊 token 的說明，加進 `FAMILIES`（[[families/__init__.py|FAMILIES = {]]），"
        "`model_type` 對得上就會自動選到。",
        Code("families/qwen3_5.py", "QWEN3_5 = Family(", caption="Qwen3.5 的 family 幾乎都是資料，只有 `parse` 是程式。"),
    )),
    Section("submit", "ui", "送出問題：載入模型、開一條 thread 跑 agent", "輸入框和範例按鈕都走 `start_run()`：拿鎖、凍結設定、開 thread。", (
        "畫面每次重跑，先由 `current_llm()` 照側欄選的模型拿到 `LocalLLM`。`load_llm()` 用 `st.cache_resource(max_entries=1)` 快取，"
        "記憶體裡同時只留一個模型，換模型時舊的就會被放掉。第一次用到某個模型時，把它建議的 temperature、top-p、top-k 寫進側欄；"
        "Qwen3.5 的 checkpoint 沒附 generation_config.json，建議值來自 family（[[llm.py|def _sampling_defaults(self)]]）。"
        "側欄的模型選單換了之後會開新對話（[[streamlit_app.py|\"模型\", list(MODELS)]]）。",
        Code("streamlit_app.py", "@st.cache_resource(max_entries=1", end="def current_llm() -> LocalLLM:"),
        "輸入框（[[streamlit_app.py|prompt = st.chat_input(]]）和範例按鈕都會呼叫 `start_run()`。"
        "它先用不等待的方式拿一把全域的鎖（[[streamlit_app.py|if not slot.lock.acquire(blocking=False):]]）："
        "模型只有一份，所以不管開幾個分頁，同一時間只能有一個 agent 在跑，拿不到鎖就跳出提示。",
        Code("streamlit_app.py", "def start_run(", caption="鎖是 `generation_slot()` 裡的 `threading.Lock`，用 `st.cache_resource` 讓所有分頁共用。"),
        "`current_settings()` 把側欄的值打包成 `RunSettings`，解碼參數包在裡面的 `Sampling`。這份設定這一輪從頭用到尾，"
        "執行中改側欄只會影響下一輪。",
        Code("streamlit_app.py", "def current_settings("),
        "`LiveRun` 建立 `Run`，把 `self.events.append` 當成 `emit` 傳進去，然後開背景 thread。"
        "thread 跑完 `run.run()` 之後才在 `finally` 裡放開鎖，所以就算 agent 出錯，鎖也一定會還回去。",
        Code("streamlit_app.py", "class LiveRun:", caption="`daemon=True`：關掉 server 時不用等 agent 跑完。"),
    )),
    Section("prepare", "harness", "Run 準備：訊息、工具、system prompt", "`self.messages` 是 agent 的全部記憶，之後每一步都把它整份重新 render。", (
        "`Run.__init__` 把之前的對話歷史接上這次的問題，存成 `self.messages`（[[agent.py|self.messages = [*history,]]）。"
        "之後的工具呼叫和工具結果都 append 在後面；模型本身不記得任何事，每一步看到的都是這份 list 重新 render 出來的字串。"
        "訊息用的是 Hugging Face 的通用格式（`role`、`content`、`tool_calls`），兩個家族都吃得下，格式差異交給各自的 chat template。"
        "`_stop` 和 `_decided` 是 UI thread 用來通知 agent 的兩個同步工具。",
        Code("agent.py", "def __init__(self, llm: LocalLLM, history"),
        "`run()` 一開始拿出這個模型的 family，決定這一輪模型看得到什麼：開著「使用工具」時，收集每個工具的 `schema()` 給 chat template，"
        "另外留一份「工具名稱 → 參數 schema」給 parser（Qwen 的參數值是純文字，要靠 schema 才知道型別）；"
        "system prompt 也分有工具（[[agent.py|SYSTEM_WITH_TOOLS = ]]）和沒工具兩個版本。"
        "第一個事件 `run_start` 帶著模型和 family，畫面就知道這一輪要用哪個模型的格式來畫。",
        Code("agent.py", "def run(self):", end='stop_reason, final = "max_steps", ""',
             caption="`stop_reason` 的預設值是 `max_steps`：迴圈跑完所有步數都沒有 break，停止原因就是步數上限。"),
        "工具的 schema 就是一份 JSON Schema，包成 Hugging Face chat template 認得的 function 格式：",
        Code("tools.py", "def schema(self) -> dict:"),
    )),
    Section("context", "context", "脈絡：把整段對話 render 成一個字串", "chat template 把 system、工具、對話接成一長串；模型這一步能看到的就只有它。", (
        "每一圈從 `for step in range(1, s.max_steps + 1)` 開始，側欄的「步數上限」就是這個 range 的上界。第一件事是 render：",
        Code("agent.py", "for step in range(1, s.max_steps + 1):", end="roles=_role_counts([system, *self.messages])",
             caption="`step_start` 讓畫面開一張新的 step 卡片；`context` 帶著完整字串送出。"),
        "`render()` 呼叫 tokenizer 自帶的 chat template。格式是模型自己的，harness 不用知道：",
        (
            "Gemma 4：`<bos><|turn>system` 開頭；開了思考模式，system turn 最前面多一個 `<|think|>`；"
            "每個工具寫成 `<|tool>declaration:名稱{…}<tool|>`；每則訊息包在 `<|turn>角色` … `<turn|>` 裡。",
            "Qwen3.5：沒有 `<bos>`，`<|im_start|>system` 開頭；工具 schema 原樣以 JSON 放進 `<tools>` … `</tools>`，"
            "後面附上 XML 呼叫格式的說明；每則訊息包在 `<|im_start|>角色` … `<|im_end|>` 裡。",
        ),
        "`generation_prompt=True`（`add_generation_prompt`）在最後打開模型的回合，請模型接著寫：Gemma 是 `<|turn>model`；"
        "Qwen 是 `<|im_start|>assistant` 加上 `<think>`（開思考模式）或一段空的 `<think></think>`（不開）。",
        Code("llm.py", "def render(self, messages"),
        "接著數 token（[[agent.py|n_tokens=len(self.llm.tokenizer(prompt]]），畫面拿它和脈絡上限比，顯示用掉幾 %。"
        "`added_from` 是這一步的字串和 `prev_sequence` 共同開頭的長度（{{record}}會說 `prev_sequence` 是什麼）："
        "從這個位置之後就是這一步新加入的內容，第 2 步以後正好是工具結果。",
        Code("agent.py", "def common_prefix(", caption="逐字元比較，回傳兩個字串共同開頭的長度。"),
        "畫面上的「脈絡」分頁就是把 `context` 事件裡的字串原樣印出來（[[render.py|def context_html(]]），"
        "`added_from` 之後的部分標成「這一步新加入」，family 列出的特殊 token 換成可以滑過看說明的深色標籤。",
    )),
    Section("model", "model", "模型：prefill，然後一次一個 token", "一次 forward 讀完整段 prompt，之後每生成一個 token 再 forward 一次。", (
        "harness 用一個 for 迴圈消費 `generate()` 產生的事件：`prefill` 和每個 `token` 都加上 step 編號轉送出去，"
        "最後的 `done` 留下來，改名成 `model_end` 送出。",
        Code("agent.py", "done = None", end='self.emit("model_end"',
             caption="傳進去的 `self._stop.is_set` 是一個函式：模型每生成一個 token 前都會呼叫它，問「要停了嗎？」。"),
        "**prefill**：把整個 prompt 轉成 token id，一次 forward 算完。`use_cache=True` 留下每一層的狀態（一般 attention 層是 key / value，"
        "Qwen3.5 的 Gated DeltaNet 層是一個固定大小的狀態），之後每個新 token 只需要算自己那一個位置；"
        "`logits_to_keep=1` 只取最後一個位置的 logits，因為只需要「下一個 token」的分布。"
        "prefill 的時間主要看 prompt 有多長，模型分頁的表格裡有每一步的 prefill ms。",
        Code("llm.py", "def generate(self, prompt", end='yield {"type": "prefill"',
             caption="prompt 如果以 family 的 `thought_prefix` 結尾（Gemma 的 `<|channel>thought`、Qwen 的 `<think>`），模型一開始就在思考裡。"),
        "**解碼迴圈**最多跑 `max_new_tokens` 次（側欄的「每步 token 上限」）。每一圈：檢查停止 → 抽樣出一個 token → "
        "判斷它是不是特殊 token、有沒有切換 channel：哪些 token 切到思考、工具呼叫或回覆，由 family 的 `channel_switch` 決定"
        "（Gemma：[[families/gemma4.py|channel_switch={]]；Qwen：[[families/qwen3_5.py|channel_switch={]]）→ "
        "把目前為止的所有 token 整段 decode 再取差值，這樣被拆成好幾個 byte token 的中文字會等到湊齊才輸出 → 送出 `token` 事件。",
        Code("llm.py", "decode_start = time.perf_counter()", end="**details,"),
        "生成到停止 token 就跳出迴圈。還沒停，就把剛抽到的這一個 token 餵回模型，帶著快取算出下一個位置的 logits。",
        Code("llm.py", "if token_id in self.eos_ids:", end="logits = out.logits[0, -1].float().cpu()"),
        "最後的 `done` 告訴 harness 為什麼停（`finish`：`stop_token`、`max_new_tokens` 或 `user_stopped`）、停在哪個 token、"
        "生成速度，以及整段原始輸出 `raw`（保留特殊 token，因為 harness 要靠它們解析）。",
        Code("llm.py", "decode_s = time.perf_counter() - decode_start", end='"raw": self.tokenizer.decode(generated'),
    )),
    Section("sampling", "model", "抽樣：每個 token 是怎麼選出來的", "temperature → top-k → top-p → 抽一個；同時記下模型原本的分布，給浮窗用。", (
        "`_sample()` 每個 token 跑一次。先對原始 logits 做 softmax，得到模型自己的分布（T = 1、什麼都沒剃掉）："
        "token 浮窗裡的「模型給的機率」、排名和前 5 名候選都來自這個分布。",
        "temperature ≤ 0 是 greedy，直接拿機率最高的。否則依序：logits ÷ T → 只留前 k 個（top-k，0 = 不限）→ "
        "重新 softmax → 從機率最高的往下累加，累積機率達到 top-p 為止（nucleus，至少留一個）→ 重新正規化 → "
        "`torch.multinomial` 抽一個。這段跟模型無關，兩個家族共用。",
        Code("llm.py", "def _sample(self, logits"),
        "回傳的資料就是浮窗裡的數字：`p` 是被選中的 token 在原始分布的機率；`rank` 是它的名次；"
        "`kept` 是剃完還剩幾個候選；`p_sampled` 是它在剃完、重新正規化後的分布裡的機率。"
        "step 卡片上 token 的底色深淺就是用 `p` 算的（[[render.py|def heat(p: float)]]）：越深，代表模型原本越不看好它，卻被抽中了。",
    )),
    Section("parse", "harness", "Harness：把輸出拆成思考、工具呼叫、回覆", "模型只會吐文字；哪段是工具呼叫，要 harness 用 family 的 parser 讀出來。", (
        "chat template 有時會在 prompt 結尾先打開思考（family 的 `thought_prefix`）；這段在 prompt 裡、不在 `raw` 裡，"
        "所以先補回來，再交給 family 的 `parse`。關掉「使用工具」時，就算模型寫了工具呼叫也當作沒有。",
        Code("agent.py", "prefix = family.thought_prefix if", end='self.emit("parsed"'),
        "兩個 parser 回傳一樣的結構：`thinking`、`tool_calls`（每個有 `name`、`arguments`、`error`）、`content`，所以後面的 harness 不用管是哪個模型。",
        "**Gemma 4**：(1) 切出 `<|channel>thought` 到 `<channel|>` 之間的思考；(2) 用正規表達式找出每一個 `<|tool_call>call:名稱{…}<tool_call|>`"
        "（[[families/gemma4.py|CALL_RE = re.compile]]），參數交給小 parser；(3) 有開頭、沒有結尾的工具呼叫（例如被 token 上限截斷）記成格式錯誤。"
        "剩下的文字去掉停止 token，就是給使用者的回覆。",
        Code("families/gemma4.py", "def parse_output("),
        "Gemma 的參數不是 JSON：key 不加引號，字串用 `<|\"|>` 這個特殊 token 包起來，例如 `{timezone:<|\"|>Asia/Taipei<|\"|>}`。"
        "`_Parser` 是遞迴下降 parser：看到 `<|\"|>` 就一路讀到下一個 `<|\"|>`，`{` 讀物件、`[` 讀陣列，"
        "其他的當作數字、`true` / `false` 或 `null`。",
        Code("families/gemma4.py", "def value(self):"),
        "**Qwen3.5**：思考在 `</think>` 之前；每個 `<tool_call>` 區塊裡找 `<function=名稱>` 和一段段 `<parameter=名稱>`。"
        "值都是純文字，所以要照工具的 schema 轉型別：schema 說是字串就保持原樣，說是數字、陣列或物件就用 JSON 讀，"
        "說是布林就看是不是 `true` / `false`；轉不過去就留著文字，讓下一關的 schema 驗證把型別錯誤回報給模型。",
        Code("families/qwen3_5.py", "def parse_output("),
        Code("families/qwen3_5.py", "def convert(value: str, kind"),
    )),
    Section("stop", "stop", "要不要停：由 harness 決定", "模型只是停止生成；這一輪要不要結束，是 harness 看完輸出才決定的。", (
        "解析完先檢查兩種馬上結束的情況：你按了停止；或這一步沒有任何工具呼叫。沒有工具呼叫，回覆就是最終答案，停止原因是 `no_tool_call`；"
        "如果是被 token 上限截斷而沒有工具呼叫，停止原因記成 `max_new_tokens`。兩種情況都把回覆存成 assistant 訊息，下一輪會當成歷史。",
        "這裡看的是「解析出來有沒有工具呼叫」，不是「停在哪個 token」。Gemma 停在 `<|tool_response>` 就代表要用工具，"
        "但 Qwen 不管回答完或要用工具都停在 `<|im_end|>`；用解析結果判斷，兩個家族才能共用同一段程式。",
        Code("agent.py", 'if done["finish"] == "user_stopped":', end="if not calls:"),
        "其他停止條件分散在迴圈各處：",
        (
            "for 迴圈跑完所有步數都沒 break → `max_steps`（{{prepare}}設定的預設值）。",
            "等待確認時被停止或逾時 → `user_stopped` 或 `approval_timeout`（[[agent.py|stop_reason = \"approval_timeout\" if self._timed_out]]）。",
            "連續 `MAX_FORMAT_ERRORS`（3）次寫出無法解析的工具呼叫 → `format_errors`。",
            "模型或工具丟出任何例外 → `error`，錯誤訊息會顯示在畫面上，而不是讓 thread 默默死掉。",
        ),
        Code("agent.py", "if format_errors >= MAX_FORMAT_ERRORS:", end='self.emit("run_end"',
             caption="不管怎麼停，最後一定送出 `run_end`，帶著停止原因、最終回覆和整份訊息列表。"),
    )),
    Section("record", "harness", "把工具呼叫寫回對話", "工具呼叫先變成一則 assistant 訊息，下一步由 chat template 寫回模型自己的格式。", (
        "有工具呼叫時，harness 給每個呼叫一個 id（`call_{step}_{i}`），寫成一則 assistant 訊息，"
        "格式和 OpenAI / Hugging Face 的 `tool_calls` 一樣；下一步 chat template 會把它寫回 Gemma 的 `<|tool_call>…<tool_call|>` 或 Qwen 的 `<tool_call>…</tool_call>`。",
        "兩個家族在這裡有兩點不同，都由 family 決定：模型在工具呼叫前寫的文字，Qwen 的模板會保留，Gemma 的模板會把它挪到工具結果後面並關掉回合，"
        "所以 Gemma 丟掉這段文字（`text_with_calls`）；思考內容要存在哪個欄位，模板才讀得到，Gemma 是 `reasoning`、Qwen 是 `reasoning_content`（`reasoning_key`）。",
        "接著先 render 一次「加上這則 assistant 訊息、但還沒有工具結果」的字串，存成 `prev_sequence`；`generation_prompt=False`，"
        "因為這時還輪不到模型寫。下一步的 prompt 和它比，多出來的就是工具結果。直接拿「上一步 prompt + 模型輸出」來比會不準，"
        "因為模板重新 render 時會改寫模型那一段（排序參數、換掉思考的標記）。",
        Code("agent.py", 'ids = [f"call_{step}_{i}"', end="prev_sequence = self.llm.render("),
    )),
    Section("check", "harness", "驗證與權限：模型從不直接碰環境", "每個呼叫要過三關：格式和 schema、重複檢查、（開著時）你的確認。", (
        "每個呼叫交給 `_handle_call()`。第一關是驗證：解析失敗、沒有這個工具、參數不符合 schema，都不會執行。",
        Code("agent.py", "def _handle_call(", end="args, errors = tool.validate(args)"),
        "`Tool.validate()` 檢查必要參數有沒有給、有沒有多出不認得的參數、型別和 enum 對不對，然後補上預設值。"
        "錯誤訊息是中文句子，因為它會原封不動變成模型下一步讀到的觀察。",
        Code("tools.py", "def validate(self, args: dict)"),
        "第二關記錄重複：把「工具名稱 + 排序後的參數 JSON」當成 key，跟這一輪之前的呼叫比；一模一樣就在 step 卡片標出"
        "「迴圈可能在打轉」。這只是提醒，不會擋。然後送出 `tool_call` 事件；有錯誤就直接把錯誤包成觀察回報，不進環境。",
        Code("agent.py", 'key = (call["name"], json.dumps', end='{"error": "; ".join(errors)}'),
        "第三關是你。開著「執行工具前先問我」時，harness 送出 `approval_request`，然後停在 `_wait_for_decision()` 等。"
        "拒絕、停止、逾時都會變成一則錯誤觀察，模型下一步只看得到這段文字。",
        Code("agent.py", "if self.settings.approve_tools:", end='return self._observe(step, cid, call["name"], {"error": error}'),
        "等待用的是 `threading.Condition`：每 0.5 秒醒來一次，直到你按下按鈕（`decide()` 會 `notify_all`）、按了停止，"
        "或超過 `APPROVAL_TIMEOUT`（[[agent.py|APPROVAL_TIMEOUT = ]]，5 分鐘）。逾時就當作拒絕並停止整輪，免得一個關掉的分頁把模型佔住。",
        Code("agent.py", "def _wait_for_decision("),
        "畫面這邊，等待中的呼叫會在聊天區顯示「允許／拒絕」按鈕，按下去就呼叫 `run.run.decide()`。"
        "「先問我」寫在 harness 的程式裡，而不是寫在 prompt 裡請模型遵守，所以模型沒有辦法繞過。",
        Code("streamlit_app.py", "for call in render.pending_approvals(data):"),
    )),
    Section("execute", "env", "環境：真的去執行工具", "通過三關才會呼叫 Python 函式；錯誤也會變成觀察，不會讓迴圈當掉。", (
        "送出 `tool_start` 後呼叫工具的 Python 函式。工具自己丟的 `ToolError` 是可以預期的錯誤（時區打錯、找不到頁面），"
        "訊息直接給模型看；其他例外也會被接住，變成 `{\"error\": …}`。",
        Code("agent.py", 'self.emit("tool_start"', end="return self._observe(step, cid, call[\"name\"], result, ok=ok"),
        "四個工具各自連到不同的環境：",
        (
            "`get_current_time`：讀系統時鐘，用 IANA 時區名稱換算（[[tools.py|def get_current_time(]]）。",
            "`calculator`：在 AST 白名單上計算，從不呼叫 `eval()`。只允許數字、四則運算、次方和少數數學函式；"
            "算次方前先估計位數，避免 `10**10**10` 這種算式把 thread 卡死。",
            "`search_wikipedia` / `read_wikipedia`：用 httpx 呼叫 Wikipedia API（[[tools.py|def search_wikipedia(]]），"
            "和 ReAct 論文在 HotpotQA 上用的 search / read 是同一組動作。",
        ),
        Code("tools.py", "def _eval(node):", caption="遞迴走過語法樹，遇到白名單以外的節點就丟 `ToolError`。"),
    )),
    Section("observe", "context", "觀察接回脈絡，進下一圈", "工具結果變成一則 `tool` 訊息；下一圈 render 時，它就出現在模型的工具呼叫後面。", (
        "工具結果（或錯誤）在 `_observe()` 裡轉成 JSON 字串，超過 `MAX_RESULT_CHARS`（2000 字元）就截斷並加上 `…(truncated)`，"
        "免得一個長網頁塞爆脈絡；送出 `tool_result` 事件，再把字串回傳。",
        Code("agent.py", "def _observe("),
        "回傳的字串變成一則 `role: tool` 的訊息 append 到 `self.messages`。同時更新連續格式錯誤的次數：這次是解析錯誤就加一，正常就歸零。",
        Code("agent.py", "for cid, call in zip(ids, calls):", end="if format_errors >= MAX_FORMAT_ERRORS:"),
        "然後回到 for 迴圈開頭，下一圈 render 時 chat template 把工具結果寫進脈絡，這就是 step 卡片上「上一步的觀察接回脈絡」那一段：",
        (
            "Gemma 4：`<|tool_response>response:名稱{value:<|\"|>…<|\"|>}<tool_response|>`，緊接在模型的工具呼叫後面，"
            "還在同一個 model turn 裡，模型讀到結果後接著寫。",
            "Qwen3.5：`<|im_start|>user` 加上 `<tool_response>` … `</tool_response>`，結果算是一則 user 訊息；"
            "後面再開一個新的 `<|im_start|>assistant`，模型從新的回合開始寫。",
        ),
        "不管哪一種，模型下一步可能再呼叫一次工具，也可能直接回答。",
    )),
    Section("ui", "ui", "畫面：事件怎麼變成你看到的東西", "背景 thread 只管 append 事件；畫面每 0.5 秒把整張列表重畫一次。", (
        "`workspace` 是一個 fragment：有 agent 在跑時，它每 0.5 秒自己重跑一次，只重畫聊天和右邊的檢視，不用整頁重跑。"
        "跑完的那一刻再整頁重跑一次，然後停止輪詢。",
        Code("streamlit_app.py", "@st.fragment(run_every=", end="viz_panel(llm)"),
        "每次重畫，`LiveRun.data()` 複製一份目前的事件列表交給 `render.build()`，按順序折成一個個 step。"
        "`run_start` 告訴它這一輪是哪個模型、哪個 family，之後畫特殊 token、停止 token 和使用者訊息的位置都問這個 family；"
        "`step_start` 開一個新的 step，`context`、`prefill`、`token`、`parsed` 填進去，工具相關的事件依 `call_id` 歸到各自的呼叫。"
        "之後所有 HTML 都是這個結構的純函式，所以執行到一半的任何時刻都畫得出來。",
        Code("render.py", "def build(events: list[dict])"),
        "迴圈圖哪一格亮、哪條箭頭在流動，由最後一個事件決定：",
        Code("render.py", 'state["active"], state["edge"] = {'),
        "一輪結束後，`absorb_finished()` 把 `run_end` 帶回來的完整訊息列表存成 `history`。下一個問題就接在它後面，"
        "所以 agent 記得前面聊過什麼，包括之前的工具結果。",
        Code("streamlit_app.py", "def absorb_finished("),
        "停止按鈕呼叫 `Run.stop()`：設定 `_stop` 並叫醒正在等確認的 thread。解碼迴圈每個 token 前都會檢查 `_stop`，"
        "所以按下去之後最多再生成一個 token 就停。",
        Code("agent.py", "def stop(self):", end="def decide(self, call_id"),
    )),
]

# (event, file, text on the line that sends it, what it carries, what the page does with it)
EVENTS = [
    ("run_start", "agent.py", '"run_start", run_id=', "模型和 family、這一輪的設定、可用的工具", "用哪個模型的格式畫這一輪；步數上限、解碼說明"),
    ("step_start", "agent.py", 'self.emit("step_start"', "step 編號", "開一張新的 step 卡片；迴圈圖亮起「脈絡」"),
    ("context", "agent.py", '"context",', "完整 prompt、token 數、`added_from`、脈絡上限、各角色訊息數", "step 卡片的脈絡段、「脈絡」分頁"),
    ("prefill", "llm.py", 'yield {"type": "prefill"', "prompt 的 token 數、prefill 耗時", "迴圈圖「模型」格、模型分頁的 prefill ms"),
    ("token", "llm.py", '"type": "token",', "id、piece、文字、channel、是否為特殊 token、機率、排名、前 5 名候選", "token 的熱度顏色與浮窗、左邊逐字出現的回覆"),
    ("model_end", "agent.py", 'self.emit("model_end"', "停止原因、停止 token、token 數、tok/s、原始輸出", "「停在 …」、tok/s、模型分頁的表格"),
    ("parsed", "agent.py", 'self.emit("parsed"', "思考、回覆、工具呼叫", "Harness 段開始出現"),
    ("tool_call", "agent.py", '"tool_call",', "名稱、驗證後的參數、錯誤、是否重複、環境", "Harness 段的呼叫、驗證結果、重複警告"),
    ("approval_request", "agent.py", 'self.emit("approval_request"', "等待確認的呼叫", "聊天區的「允許／拒絕」按鈕；迴圈圖閃爍"),
    ("approval", "agent.py", 'self.emit("approval",', "允許、拒絕、逾時或被停止", "Harness 段顯示你的決定"),
    ("tool_start", "agent.py", 'self.emit("tool_start"', "工具名稱、環境", "迴圈圖亮起「環境」"),
    ("tool_result", "agent.py", 'self.emit("tool_result"', "結果字串、成功與否、是否截斷、耗時", "環境段的結果、聊天區的工具標籤"),
    ("error", "agent.py", 'self.emit("error"', "例外的類型和訊息", "聊天區和停止卡片的錯誤訊息"),
    ("run_end", "agent.py", 'self.emit("run_end"', "停止原因、最終回覆、整份訊息列表", "停止卡片；訊息列表成為下一輪的 `history`"),
]
