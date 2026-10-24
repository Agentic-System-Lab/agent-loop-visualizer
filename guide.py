"""The 運作說明 (How it works) page: one user turn through the agent loop, with the code behind every part.

It is a static page on GitHub Pages, in Traditional Chinese and in English; the app's top bar links to
the one in the app's language. Build it with

    uv run python guide.py _site

which writes _site/zh/index.html and _site/en/index.html, each one file with its styles and script
inlined, and _site/index.html, which sends a reader on to the language they picked on the page's
switch before, or else their browser's. .github/workflows/pages.yml builds and publishes them on
every push to main.

The content points at code by text, not by line number: Code("agent.py", "def run(self):") is
looked up in the file every time the page is built, so the line numbers on the page follow the
code when it changes. A start text must match exactly one line of its file; tests/test_guide.py
checks every reference, so an edit that breaks one fails the tests instead of the page.

Every paragraph is written in both languages next to each other ({"zh": ..., "en": ...}), and the
code excerpts between them are shared, so the two pages always show the same code.
"""

import ast
import html
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

import render
from i18n import LANGS, pick

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
REPO = "https://github.com/cyyeh/agent-loop-visualizer"
FONTS = "https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap"
HTML_LANG = {"zh": "zh-Hant", "en": "en"}
STAGES = {
    "ui": ({"zh": "畫面", "en": "Page"}, "var(--ink-2)"),
    "harness": ({"zh": "Harness", "en": "Harness"}, "var(--c-harness)"),
    "context": ({"zh": "脈絡", "en": "Context"}, "var(--c-context)"),
    "model": ({"zh": "模型", "en": "Model"}, "var(--c-model)"),
    "env": ({"zh": "環境", "en": "Environment"}, "var(--c-env)"),
    "stop": ({"zh": "停止", "en": "Stop"}, "var(--c-stop)"),
}
REF_RE = re.compile(r"\[\[([\w./]+)\|(.+?)\]\]")


@dataclass(frozen=True)
class Code:
    """An excerpt: from the line containing `start` through the statement that begins there, or
    through the statement on the first line at or after it that contains `end`."""

    file: str
    start: str
    end: str | None = None
    caption: dict | None = None  # {"zh": ..., "en": ...}


@dataclass(frozen=True)
class Section:
    key: str
    stage: str
    title: dict
    lead: dict
    body: tuple  # paragraphs ({"zh", "en"} dicts), bullet lists (tuples of them) and Code excerpts, in reading order


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


def xref_html(key: str, lang: str) -> str:
    """A link to another section by its key, with the section's current number and title."""
    n, sec = next((n, sec) for n, sec in enumerate(SECTIONS, 1) if sec.key == key)
    title = esc(sec.title[lang])
    return f'<a class="g-xref" href="#g-{key}">{pick(lang, f"第 {n} 段「{title}」", f"§{n} “{title}”")}</a>'


def _inline(text: str, lang: str) -> str:
    out = html.escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    return re.sub(r"\{\{(\w+)\}\}", lambda m: xref_html(m.group(1), lang), out)


def prose_html(text: str, lang: str) -> str:
    """Escape a paragraph and mark it up: `code`, **bold**, [[file|text]] becomes a file:line chip,
    {{key}} a link to that section."""
    out, last = [], 0
    for m in REF_RE.finditer(text):
        out.append(_inline(text[last:m.start()], lang))
        out.append(ref_html(m.group(1), m.group(2)))
        last = m.end()
    out.append(_inline(text[last:], lang))
    return "".join(out)


def code_html(code: Code, lang: str) -> str:
    lo, hi = span(code)
    lines = source(code.file)[0][lo - 1:hi]
    body = highlight(textwrap.dedent("\n".join(lines)), PythonLexer(), HtmlFormatter(nowrap=True))
    rows = body.split("\n")[: hi - lo + 1]
    gutter = "".join(f'<span class="ln">{n}</span>{row}\n' for n, row in zip(range(lo, hi + 1), rows))
    if lo == hi:
        where = pick(lang, f"第 {lo} 行", f"line {lo}")
    else:
        where = pick(lang, f"第 {lo}–{hi} 行", f"lines {lo}–{hi}")
    caption = f'<span class="src-cap">{prose_html(code.caption[lang], lang)}</span>' if code.caption else ""
    return (
        f'<figure class="src"><figcaption><b>{esc(code.file)}</b><span class="src-lines">{where}</span>{caption}</figcaption>'
        f"<pre>{gutter}</pre></figure>"
    )


def body_html(item, lang: str) -> str:
    if isinstance(item, Code):
        return code_html(item, lang)
    if isinstance(item, tuple):
        return "<ul>" + "".join(f"<li>{prose_html(i[lang], lang)}</li>" for i in item) + "</ul>"
    return f"<p>{prose_html(item[lang], lang)}</p>"


def section_html(n: int, sec: Section, lang: str) -> str:
    label, color = STAGES[sec.stage]
    return (
        f'<section class="g-sec" id="g-{sec.key}" style="--stage: {color}">'
        f'<header class="g-head"><span class="g-num">{n}</span><div><p class="g-stage">{label[lang]}</p>'
        f'<h2>{esc(sec.title[lang])}</h2><p class="g-lead">{prose_html(sec.lead[lang], lang)}</p></div></header>'
        f'{"".join(body_html(item, lang) for item in sec.body)}</section>'
    )


def toc_html(lang: str) -> str:
    """A bar that stays at the top of the window and opens into the list of sections. Closed it
    shows the section you are reading; guide.js keeps that current and closes the list after a jump."""
    items = []
    for n, sec in enumerate(SECTIONS, 1):
        label, color = STAGES[sec.stage]
        items.append(
            f'<li style="--stage: {color}"><a href="#g-{sec.key}"><span class="g-num">{n}</span>'
            f'<span>{esc(sec.title[lang])}</span><small>{label[lang]}</small></a></li>'
        )
    items.append(
        '<li style="--stage: var(--muted)"><a href="#g-events"><span class="g-num">＃</span>'
        f'<span>{pick(lang, "事件一覽", "Events")}</span><small>{pick(lang, "參考", "Reference")}</small></a></li>'
    )
    contents = pick(lang, "目錄", "Contents")
    idle = pick(lang, f"{len(SECTIONS)} 節，加上事件一覽", f"{len(SECTIONS)} sections, plus the events")
    return (
        f'<nav class="g-toc" aria-label="{contents}"><details><summary><span class="g-toc-label">{contents}</span>'
        f'<span class="g-toc-now"><span>{idle}</span></span></summary>'
        f'<ol>{"".join(items)}</ol></details></nav>'
    )


def events_html(lang: str) -> str:
    rows = "".join(
        f"<tr><td><code>{esc(kind)}</code></td><td>{ref_html(file, text)}</td>"
        f"<td>{prose_html(carries[lang], lang)}</td><td>{prose_html(shows[lang], lang)}</td></tr>"
        for kind, file, text, carries, shows in EVENTS
    )
    heads = pick(lang, ("事件", "在哪裡送出", "帶了什麼", "畫面上的用途"), ("Event", "Sent from", "Carries", "What the page does with it"))
    return (
        '<section class="g-sec" id="g-events" style="--stage: var(--muted)">'
        f'<header class="g-head"><span class="g-num">＃</span><div><p class="g-stage">{pick(lang, "參考", "Reference")}</p>'
        f'<h2>{pick(lang, "事件一覽", "Events")}</h2><p class="g-lead">'
        + pick(
            lang,
            "harness 每走到一個階段就 append 一個事件；畫面上的一切都是從這張列表畫出來的。依一輪裡出現的順序排列。",
            "The harness appends an event at every stage; everything on the page is drawn from this list. They are in the order they come in a turn.",
        )
        + '</p></div></header><div class="g-table"><table><thead><tr>'
        + "".join(f"<th>{h}</th>" for h in heads)
        + f"</tr></thead><tbody>{rows}</tbody></table></div></section>"
    )


def diagram_html(lang: str) -> str:
    """The loop diagram once per theme; guide.css shows the one that matches the reader's."""
    counter = pick(lang, "每一圈 = 一個 step", "one lap = one step")
    return "".join(
        f'<div class="g-{theme}">{render.diagram_html(None, theme == "dark", lang, counter=counter, model="Gemma 4 · Qwen3.5")}</div>'
        for theme in ("light", "dark")
    )


def lang_switch_html(lang: str) -> str:
    """中文 | English at the top right. Both pages have the same sections, so guide.js keeps the reader's place."""
    links = "".join(
        f'<a href="../{code}/" hreflang="{HTML_LANG[code]}" lang="{HTML_LANG[code]}" data-lang="{code}"'
        + (' aria-current="page"' if code == lang else "")
        + f">{name}</a>"
        for code, name in LANGS.items()
    )
    return f'<nav class="g-lang" aria-label="{pick(lang, "語言", "Language")}">{links}</nav>'


# _site/index.html: the language picked on a page's switch (guide.js remembers it), else the browser's (i18n.from_locale).
REDIRECT = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>運作說明 · How it works · Agent Loop 可視化</title>
<script>
var lang;
try { lang = localStorage.getItem("lang"); } catch (err) {}
if (lang !== "zh" && lang !== "en") lang = /^zh/i.test(navigator.language || "") ? "zh" : "en";
location.replace(lang + "/" + location.hash);
</script>
<style>body { margin: 48px 24px; font: 16px/1.6 -apple-system, "PingFang TC", "Noto Sans TC", sans-serif; background: #f4f0e6; color: #18212b; }
a { color: #2f5191; } @media (prefers-color-scheme: dark) { body { background: #14181d; color: #e8e3d8; } a { color: #93b3f0; } }</style>
</head><body><p><a href="zh/">運作說明（中文）</a> · <a href="en/">How it works (English)</a></p></body></html>
"""


def page_html(commit: str, lang: str) -> str:
    """The page body; `commit` is the commit its line numbers were looked up in."""
    sections = "".join(section_html(n, sec, lang) for n, sec in enumerate(SECTIONS, 1))
    link = f'<a href="{REPO}/tree/{esc(commit)}"><code>{esc(commit[:7])}</code></a>'
    return (
        f'{lang_switch_html(lang)}<main class="guide"><header class="g-hero"><p class="g-kicker">{pick(lang, "運作說明", "HOW IT WORKS")}</p>'
        f'<h1>{pick(lang, "一個問題在 agent 迴圈裡怎麼走", "How one question travels through the agent loop")}</h1>'
        f'<p class="g-intro">{prose_html(INTRO[lang], lang)}</p>'
        f'<p class="g-meta"><a href="{REPO}">{pick(lang, "原始碼和執行方式在 GitHub", "The source and how to run it are on GitHub")}</a> · '
        f'{pick(lang, "行號對應 commit", "line numbers from commit")} {link}</p></header>'
        f'<div class="g-diagram">{diagram_html(lang)}</div>{toc_html(lang)}{sections}{events_html(lang)}</main>'
    )


def stylesheet() -> str:
    """The palette from style.css, dark.css's for a reader in dark mode, then the page's own rules."""
    palette = re.search(r"^:root \{.*?^\}", (ASSETS / "style.css").read_text(encoding="utf-8"), re.S | re.M).group()
    dark = (ASSETS / "dark.css").read_text(encoding="utf-8")
    return f"{palette}\n@media (prefers-color-scheme: dark) {{\n{dark}}}\n{(ASSETS / 'guide.css').read_text(encoding='utf-8')}"


def document_html(commit: str, lang: str) -> str:
    title = pick(lang, "運作說明 · Agent Loop 可視化", "How it works · Agent Loop Visualizer")
    description = pick(
        lang,
        "一輪對話在本機 agent 迴圈裡怎麼走：每個階段都附上程式碼片段和行號。",
        "How one turn of a conversation goes through a local agent loop, with the code and its line numbers at every stage.",
    )
    return (
        f'<!doctype html><html lang="{HTML_LANG[lang]}"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title>"
        f'<meta name="description" content="{description}">'
        f'<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="{FONTS}">'
        f"<style>\n{stylesheet()}</style></head><body>{page_html(commit, lang)}"
        f"<script>\n{(ASSETS / 'guide.js').read_text(encoding='utf-8')}</script></body></html>\n"
    )


def build(out: Path) -> list[Path]:
    """Write out/index.html and the page in each language, with line numbers from the checked-out code."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    out.mkdir(parents=True, exist_ok=True)
    pages = [out / "index.html"]
    pages[0].write_text(REDIRECT, encoding="utf-8")
    for lang in LANGS:
        page = out / lang / "index.html"
        page.parent.mkdir(exist_ok=True)
        page.write_text(document_html(commit, lang), encoding="utf-8")
        pages.append(page)
    return pages


# ---------- content ----------

INTRO = {
    "zh": "從你按下送出到看到回覆，中間發生的每一件事，照實際執行的順序走一遍。app 的側欄可以選 Gemma 4 E2B 或 Qwen3.5 0.8B，"
    "兩個模型跑的是同一個迴圈，不一樣的地方會並排說明。每段都附上做這件事的程式碼；`agent.py:98` 這類標記是檔名和行號，"
    "滑過去會顯示那一行。行號是產生這一頁時從原始碼查出來的；每次推上 main，GitHub Actions 都會重新產生，程式改了也會跟著更新。",
    "en": "Everything that happens between pressing send and seeing the reply, in the order it really runs. The app's sidebar picks "
    "Gemma 4 E2B or Qwen3.5 0.8B; both models run the same loop, and where they differ, the two are described side by side. "
    "Every part comes with the code that does it; a mark like `agent.py:98` is a file and a line number, and hovering over it shows that line. "
    "The line numbers are looked up in the source when this page is built; GitHub Actions rebuilds it on every push to main, so they follow the code as it changes.",
}

SECTIONS = [
    Section("overview", "ui",
            {"zh": "總覽：兩條 thread、一張事件列表", "en": "Overview: two threads, one event list"},
            {"zh": "你送出一個問題就是一輪（一個 `Run`）；一輪裡是一個迴圈，每一圈叫一個 step。",
             "en": "Each question you send is one turn (one `Run`); inside a turn is a loop, and each time around it is a step."}, (
        {"zh": "每個 step 做同一件事：harness 把整段對話 render 成一個字串（脈絡）→ 模型一次推論、一個 token 一個 token 生成 → "
               "harness 解析輸出。有工具呼叫，就驗證、（需要時問你）、在環境執行，把結果接回對話，進下一圈；"
               "沒有工具呼叫，就把回覆交給你，這一輪結束。",
         "en": "Every step does the same thing: the harness renders the whole conversation into one string (the context) → the model runs "
               "one inference, generating one token at a time → the harness parses the output. If there is a tool call, it validates it, "
               "asks you if it should, runs it in the environment and adds the result to the conversation for the next step; "
               "if there is none, it hands you the reply, and the turn ends."},
        {"zh": "整個 app 有兩條 thread。Streamlit 的 script thread 負責畫面；agent 跑在另一條背景 thread"
               "（[[streamlit_app.py|self.thread = threading.Thread]]）。兩邊只透過三樣東西溝通：",
         "en": "The app runs on two threads. Streamlit's script thread draws the page; the agent runs on a background thread "
               "([[streamlit_app.py|self.thread = threading.Thread]]). The two talk through just three things:"},
        (
            {"zh": "agent 每走到一個階段，就把一個事件 append 到 `events` 列表（[[agent.py|def emit(self, event_type]]）；畫面只負責把這張列表畫出來。",
             "en": "At every stage the agent appends an event to the `events` list ([[agent.py|def emit(self, event_type]]); the page only draws that list."},
            {"zh": "按「停止」時，畫面呼叫 `Run.stop()`（[[agent.py|def stop(self):]]）。",
             "en": "Pressing Stop calls `Run.stop()` ([[agent.py|def stop(self):]])."},
            {"zh": "按「允許」或「拒絕」時，畫面呼叫 `Run.decide()`（[[agent.py|def decide(self, call_id]]）。",
             "en": "Pressing Allow or Reject calls `Run.decide()` ([[agent.py|def decide(self, call_id]])."},
        ),
        {"zh": "每個檔案的分工：", "en": "What each file does:"},
        (
            {"zh": "`streamlit_app.py`：版面、側欄設定、模型選單、背景 thread、執行中每 0.5 秒重畫一次。",
             "en": "`streamlit_app.py`: the layout, the sidebar settings, the model menu and the background thread; it redraws every 0.5 s while a run is live."},
            {"zh": "`agent.py`：harness，也就是迴圈本體 `Run.run()`（[[agent.py|def run(self):]]）。",
             "en": "`agent.py`: the harness, that is, the loop itself, `Run.run()` ([[agent.py|def run(self):]])."},
            {"zh": "`llm.py`：本機推論，自己寫的解碼迴圈 `generate()`（[[llm.py|def generate(self, prompt]]）和抽樣 `_sample()`；"
                   "打開 MTP 時再加上一個草稿模型（{{mtp}}）。",
             "en": "`llm.py`: local inference, with a hand-written decoding loop, `generate()` ([[llm.py|def generate(self, prompt]]), "
                   "and sampling, `_sample()`; with MTP on, a drafter joins in ({{mtp}})."},
            {"zh": "`families/`：跟著模型走的東西。`gemma4.py`、`qwen3_5.py` 各自負責讀懂自己模型的輸出，並說明它的特殊 token（{{families}}）。",
             "en": "`families/`: everything that goes with the model. `gemma4.py` and `qwen3_5.py` each read their own model's output "
                   "and explain its special tokens ({{families}})."},
            {"zh": "`tools.py`：每個工具的 schema（給模型看）和 Python 函式（給 harness 執行），全部列在 `TOOLS`（[[tools.py|TOOLS = {]]）。",
             "en": "`tools.py`: each tool's schema (for the model to read) and Python function (for the harness to run), "
                   "all listed in `TOOLS` ([[tools.py|TOOLS = {]])."},
            {"zh": "`render.py`：把事件列表折成 step（[[render.py|def build(events: list[dict])]]），再畫成 HTML。",
             "en": "`render.py`: folds the event list into steps ([[render.py|def build(events: list[dict])]]) and draws them as HTML."},
            {"zh": "`i18n.py`：頁面的兩種語言，繁體中文和英文；`pick()` 照語言挑出其中一段文字（[[i18n.py|def pick(]]）。",
             "en": "`i18n.py`: the page's two languages, Traditional Chinese and English; `pick()` chooses the text for one of them ([[i18n.py|def pick(]])."},
        ),
    )),
    Section("families", "model",
            {"zh": "模型家族：換模型只換這一層", "en": "Model families: switching models changes only this layer"},
            {"zh": "harness、解碼迴圈和畫面都只寫一次；每個模型不一樣的地方，集中在一個 `Family` 物件裡。",
             "en": "The harness, the decoding loop and the page are written once; what differs between models is gathered in one `Family` object."}, (
        {"zh": "兩個模型的迴圈完全一樣，不一樣的是它們寫字的格式：",
         "en": "The two models run exactly the same loop; what differs is the format they write in:"},
        (
            {"zh": "回合：Gemma 用 `<|turn>user` … `<turn|>`；Qwen 用 ChatML 的 `<|im_start|>user` … `<|im_end|>`。",
             "en": "Turns: Gemma uses `<|turn>user` … `<turn|>`; Qwen uses ChatML's `<|im_start|>user` … `<|im_end|>`."},
            {"zh": "工具呼叫：Gemma 寫 `<|tool_call>call:名稱{key:<|\"|>值<|\"|>}<tool_call|>`；Qwen 寫 XML："
                   "`<tool_call>` 裡一個 `<function=名稱>`，每個參數一段 `<parameter=名稱>`，值一律是純文字。",
             "en": "Tool calls: Gemma writes `<|tool_call>call:name{key:<|\"|>value<|\"|>}<tool_call|>`; Qwen writes XML: "
                   "one `<function=name>` inside `<tool_call>`, then a `<parameter=name>` for each argument, every value plain text."},
            {"zh": "工具結果：Gemma 的結果接在同一個 model turn 裡（`<|tool_response>` … `<tool_response|>`）；"
                   "Qwen 的結果放進一則新的 `user` 訊息（`<tool_response>` … `</tool_response>`），再開一個新的 assistant turn。",
             "en": "Tool results: Gemma's result goes into the same model turn (`<|tool_response>` … `<tool_response|>`); "
                   "Qwen's goes into a new `user` message (`<tool_response>` … `</tool_response>`), followed by a new assistant turn."},
            {"zh": "停止 token：Gemma 回答完停在 `<turn|>`、要用工具時停在 `<|tool_response>`，看停止 token 就知道是哪一種；"
                   "Qwen 兩種都停在 `<|im_end|>`，要看輸出裡有沒有工具呼叫才知道。",
             "en": "Stop tokens: Gemma stops at `<turn|>` after an answer and at `<|tool_response>` when it wants a tool, so the stop token "
                   "tells which; Qwen stops at `<|im_end|>` either way, and only a tool call in the output tells them apart."},
            {"zh": "思考：Gemma 在 system turn 開頭放 `<|think|>`，推理寫在 `<|channel>thought` 到 `<channel|>` 之間；"
                   "Qwen 由 chat template 在 assistant 訊息開頭先寫好 `<think>`，推理寫到 `</think>`。",
             "en": "Thinking: Gemma puts `<|think|>` at the start of the system turn and writes its reasoning between `<|channel>thought` "
                   "and `<channel|>`; for Qwen, the chat template opens the assistant message with `<think>`, and the reasoning runs until `</think>`."},
        ),
        {"zh": "這些差異全部寫進 `Family`，其他程式只問 family，沒有任何 `if 是 Gemma` 的分支：",
         "en": "All of these differences go into a `Family`, and the rest of the code just asks the family; there is no `if Gemma` branch anywhere:"},
        Code("families/__init__.py", "class Family:", end="def special_re(self)",
             caption={"zh": "欄位分三組：讀模型輸出用的、把對話寫回 chat template 用的、畫面用的。",
                      "en": "The fields come in three groups: for reading the model's output, for writing the conversation back for the chat template, and for the page."}),
        {"zh": "載入模型時，用 checkpoint 的 `config.json` 裡的 `model_type` 找出 family。停止 token 是 generation config 列的，再加上 family 宣告的："
               "Qwen3.5 的 checkpoint 只列了 `<|endoftext|>`，沒有列每個回合結尾的 `<|im_end|>`，少了 family 這一層，模型回答完會繼續寫下去。"
               "Qwen 的 `<think>`、`<tool_call>` 在 tokenizer 裡也沒有標成 special，同樣由 family 補上。",
         "en": "When a model loads, the `model_type` in the checkpoint's `config.json` finds its family. The stop tokens are those the generation "
               "config lists plus those the family declares: the Qwen3.5 checkpoint lists only `<|endoftext|>`, not the `<|im_end|>` that ends "
               "every turn, so without the family the model would keep writing after its answer. Qwen's tokenizer does not mark `<think>` "
               "and `<tool_call>` as special either, and the family adds those too."},
        Code("llm.py", "self.family = families.for_model_type(", end="self.special_ids = set("),
        {"zh": "要加模型時：同一個家族的另一個 checkpoint（例如更大的 Gemma 4），在 `MODELS` 加一行就好（[[families/__init__.py|MODELS = {]]）。"
               "新的家族，照 `families/qwen3_5.py` 寫一個模組，內容是一個輸出的 parser 和特殊 token 的說明（中英文各一份），"
               "加進 `FAMILIES`（[[families/__init__.py|FAMILIES = {]]），`model_type` 對得上就會自動選到。",
         "en": "To add a model: another checkpoint of a family that is already here (a bigger Gemma 4, say) is one line in `MODELS` "
               "([[families/__init__.py|MODELS = {]]). A new family is a module like `families/qwen3_5.py`, with a parser for the output and "
               "notes on the special tokens (in Chinese and in English), added to `FAMILIES` ([[families/__init__.py|FAMILIES = {]]); "
               "a matching `model_type` picks it automatically."},
        Code("families/qwen3_5.py", "QWEN3_5 = Family(",
             caption={"zh": "Qwen3.5 的 family 幾乎都是資料，只有 `parse` 是程式。", "en": "Qwen3.5's family is almost all data; only `parse` is code."}),
    )),
    Section("submit", "ui",
            {"zh": "送出問題：載入模型、開一條 thread 跑 agent", "en": "Sending a question: load the model, start a thread for the agent"},
            {"zh": "輸入框和範例按鈕都走 `start_run()`：拿鎖、凍結設定、開 thread。",
             "en": "The input box and the example buttons both go through `start_run()`: take the lock, freeze the settings, start a thread."}, (
        {"zh": "畫面每次重跑，先由 `current_llm()` 照側欄選的模型拿到 `LocalLLM`。`load_llm()` 用 `st.cache_resource(max_entries=1)` 快取，"
               "記憶體裡同時只留一個模型，換模型時舊的就會被放掉。第一次用到某個模型時，把它建議的 temperature、top-p、top-k 寫進側欄；"
               "Qwen3.5 的 checkpoint 沒附 generation_config.json，建議值來自 family（[[llm.py|def _sampling_defaults(self)]]）。"
               "側欄的模型選單換了之後會開新對話（[[streamlit_app.py|key=\"model_id\", on_change=new_chat]]），MTP 也回到關閉；"
               "打開「MTP 推測解碼」時，`current_llm()` 第一次會順便載入草稿模型（[[llm.py|def load_drafter(self):]]）。",
         "en": "On every rerun, `current_llm()` first gets the `LocalLLM` for the model picked in the sidebar. `load_llm()` is cached with "
               "`st.cache_resource(max_entries=1)`, so only one model stays in memory, and switching models lets the old one go. The first time "
               "a model is used, its recommended temperature, top-p and top-k go into the sidebar; the Qwen3.5 checkpoint ships no "
               "generation_config.json, so its recommendations come from the family ([[llm.py|def _sampling_defaults(self)]]). "
               "Switching models in the sidebar's menu starts a new conversation ([[streamlit_app.py|key=\"model_id\", on_change=new_chat]]) "
               "and turns MTP back off; the first time MTP speculative decoding is switched on, `current_llm()` also loads the drafter "
               "([[llm.py|def load_drafter(self):]])."},
        Code("streamlit_app.py", "@st.cache_resource(max_entries=1", end="def current_llm() -> LocalLLM:"),
        {"zh": "輸入框（[[streamlit_app.py|prompt = st.chat_input(]]）和範例按鈕都會呼叫 `start_run()`。"
               "它先用不等待的方式拿一把全域的鎖（[[streamlit_app.py|if not slot.lock.acquire(blocking=False):]]）："
               "模型只有一份，所以不管開幾個分頁，同一時間只能有一個 agent 在跑，拿不到鎖就跳出提示。",
         "en": "The input box ([[streamlit_app.py|prompt = st.chat_input(]]) and the example buttons both call `start_run()`. "
               "It first tries to take a global lock without waiting ([[streamlit_app.py|if not slot.lock.acquire(blocking=False):]]): "
               "there is one copy of the model, so however many tabs are open, only one agent can run at a time, and without the lock the tab shows a notice."},
        Code("streamlit_app.py", "def start_run(",
             caption={"zh": "鎖是 `generation_slot()` 裡的 `threading.Lock`，用 `st.cache_resource` 讓所有分頁共用。",
                      "en": "The lock is the `threading.Lock` in `generation_slot()`, shared by every tab through `st.cache_resource`."}),
        {"zh": "`current_settings()` 把側欄的值打包成 `RunSettings`，解碼參數（包括要不要用 MTP）包在裡面的 `Sampling`。這份設定這一輪從頭用到尾，"
               "執行中改側欄只會影響下一輪。它也記下頁面現在的語言：harness 和工具回給模型的錯誤，這一輪都用這個語言寫。",
         "en": "`current_settings()` packs the sidebar's values into `RunSettings`, with the decoding parameters (MTP included) in its `Sampling`. "
               "These settings hold for the whole turn; changing the sidebar during a run only affects the next one. It also records the page's "
               "language: the errors that the harness and the tools send the model are worded in it for the whole turn."},
        Code("streamlit_app.py", "def current_settings("),
        {"zh": "`LiveRun` 建立 `Run`，把 `self.events.append` 當成 `emit` 傳進去，然後開背景 thread。"
               "thread 跑完 `run.run()` 之後才在 `finally` 裡放開鎖，所以就算 agent 出錯，鎖也一定會還回去。",
         "en": "`LiveRun` creates the `Run`, passing `self.events.append` as its `emit`, and starts the background thread. "
               "The thread releases the lock in a `finally` only after `run.run()` returns, so the lock always comes back, even when the agent fails."},
        Code("streamlit_app.py", "class LiveRun:",
             caption={"zh": "`daemon=True`：關掉 server 時不用等 agent 跑完。", "en": "`daemon=True`: stopping the server does not wait for the agent to finish."}),
    )),
    Section("prepare", "harness",
            {"zh": "Run 準備：訊息、工具、system prompt", "en": "Preparing a run: messages, tools, system prompt"},
            {"zh": "`self.messages` 是 agent 的全部記憶，之後每一步都把它整份重新 render。",
             "en": "`self.messages` is all the agent remembers; every step renders the whole of it again."}, (
        {"zh": "`Run.__init__` 把之前的對話歷史接上這次的問題，存成 `self.messages`（[[agent.py|self.messages = [*history,]]）。"
               "之後的工具呼叫和工具結果都 append 在後面；模型本身不記得任何事，每一步看到的都是這份 list 重新 render 出來的字串。"
               "訊息用的是 Hugging Face 的通用格式（`role`、`content`、`tool_calls`），兩個家族都吃得下，格式差異交給各自的 chat template。"
               "`_stop` 和 `_decided` 是 UI thread 用來通知 agent 的兩個同步工具。",
         "en": "`Run.__init__` appends this question to the conversation so far and keeps it as `self.messages` ([[agent.py|self.messages = [*history,]]). "
               "Tool calls and tool results are appended after it; the model itself remembers nothing, and what it sees in each step is this "
               "list rendered into a string again. The messages use Hugging Face's common format (`role`, `content`, `tool_calls`), which "
               "both families accept; each family's chat template takes care of its own format. `_stop` and `_decided` are the two "
               "synchronization tools the UI thread signals the agent with."},
        Code("agent.py", "def __init__(self, llm: LocalLLM, history"),
        {"zh": "`run()` 一開始拿出這個模型的 family，決定這一輪模型看得到什麼：開著「使用工具」時，收集每個工具的 `schema()` 給 chat template，"
               "另外留一份「工具名稱 → 參數 schema」給 parser（Qwen 的參數值是純文字，要靠 schema 才知道型別）；"
               "system prompt 也分有工具（[[agent.py|SYSTEM_WITH_TOOLS = ]]）和沒工具兩個版本。"
               "第一個事件 `run_start` 帶著模型和 family，畫面就知道這一輪要用哪個模型的格式來畫。",
         "en": "`run()` starts by taking the model's family and deciding what the model gets to see this turn: with Use tools on, it collects "
               "every tool's `schema()` for the chat template, and keeps a map from tool name to parameters schema for the parser (Qwen's "
               "argument values are plain text, and only the schema says their types); the system prompt comes in two versions, with tools "
               "([[agent.py|SYSTEM_WITH_TOOLS = ]]) and without. The first event, `run_start`, carries the model and the family, so the page "
               "knows whose format to draw the turn in."},
        Code("agent.py", "def run(self):", end='stop_reason, final = "max_steps", ""',
             caption={"zh": "`stop_reason` 的預設值是 `max_steps`：迴圈跑完所有步數都沒有 break，停止原因就是步數上限。",
                      "en": "`stop_reason` starts out as `max_steps`: when the loop uses up its steps without a break, the step limit is why it stopped."}),
        {"zh": "工具的 schema 就是一份 JSON Schema，包成 Hugging Face chat template 認得的 function 格式：",
         "en": "A tool's schema is a JSON Schema, wrapped in the function format that Hugging Face chat templates know:"},
        Code("tools.py", "def schema(self) -> dict:"),
    )),
    Section("context", "context",
            {"zh": "脈絡：把整段對話 render 成一個字串", "en": "Context: rendering the whole conversation into one string"},
            {"zh": "chat template 把 system、工具、對話接成一長串；模型這一步能看到的就只有它。",
             "en": "The chat template joins the system prompt, the tools and the conversation into one long string; it is all the model can see in this step."}, (
        {"zh": "每一圈從 `for step in range(1, s.max_steps + 1)` 開始，側欄的「步數上限」就是這個 range 的上界。第一件事是 render：",
         "en": "Each step starts at `for step in range(1, s.max_steps + 1)`; the sidebar's Max steps is the top of that range. The first thing is to render:"},
        Code("agent.py", "for step in range(1, s.max_steps + 1):", end="roles=_role_counts([system, *self.messages])",
             caption={"zh": "`step_start` 讓畫面開一張新的 step 卡片；`context` 帶著完整字串送出。",
                      "en": "`step_start` has the page open a new step card; `context` goes out with the whole string."}),
        {"zh": "`render()` 呼叫 tokenizer 自帶的 chat template。格式是模型自己的，harness 不用知道：",
         "en": "`render()` calls the chat template that comes with the tokenizer. The format is the model's own, and the harness does not need to know it:"},
        (
            {"zh": "Gemma 4：`<bos><|turn>system` 開頭；開了思考模式，system turn 最前面多一個 `<|think|>`；"
                   "每個工具寫成 `<|tool>declaration:名稱{…}<tool|>`；每則訊息包在 `<|turn>角色` … `<turn|>` 裡。",
             "en": "Gemma 4: it starts with `<bos><|turn>system`; with thinking mode on, a `<|think|>` comes first in the system turn; "
                   "each tool is written as `<|tool>declaration:name{…}<tool|>`; each message is wrapped in `<|turn>role` … `<turn|>`."},
            {"zh": "Qwen3.5：沒有 `<bos>`，`<|im_start|>system` 開頭；工具 schema 原樣以 JSON 放進 `<tools>` … `</tools>`，"
                   "後面附上 XML 呼叫格式的說明；每則訊息包在 `<|im_start|>角色` … `<|im_end|>` 裡。",
             "en": "Qwen3.5: no `<bos>`; it starts with `<|im_start|>system`; the tool schemas go in as JSON, as they are, between `<tools>` and "
                   "`</tools>`, followed by how to write a call in XML; each message is wrapped in `<|im_start|>role` … `<|im_end|>`."},
        ),
        {"zh": "`generation_prompt=True`（`add_generation_prompt`）在最後打開模型的回合，請模型接著寫：Gemma 是 `<|turn>model`；"
               "Qwen 是 `<|im_start|>assistant` 加上 `<think>`（開思考模式）或一段空的 `<think></think>`（不開）。",
         "en": "`generation_prompt=True` (`add_generation_prompt`) opens the model's turn at the end, asking the model to write next: `<|turn>model` "
               "for Gemma; for Qwen, `<|im_start|>assistant` followed by `<think>` (thinking mode on) or an empty `<think></think>` (off)."},
        Code("llm.py", "def render(self, messages"),
        {"zh": "接著數 token（[[agent.py|n_tokens=len(self.llm.tokenizer(prompt]]），畫面拿它和脈絡上限比，顯示用掉幾 %。"
               "`added_from` 是這一步的字串和 `prev_sequence` 共同開頭的長度（{{record}}會說 `prev_sequence` 是什麼）："
               "從這個位置之後就是這一步新加入的內容，第 2 步以後正好是工具結果。",
         "en": "Next it counts the tokens ([[agent.py|n_tokens=len(self.llm.tokenizer(prompt]]), which the page compares with the context "
               "limit to show how much of it is used. `added_from` is the length of the start that this step's string shares with "
               "`prev_sequence` ({{record}} says what `prev_sequence` is): everything after it is new in this step, which from step 2 on "
               "is exactly the tool result."},
        Code("agent.py", "def common_prefix(",
             caption={"zh": "逐字元比較，回傳兩個字串共同開頭的長度。", "en": "Compares character by character and returns the length of the start the two strings share."}),
        {"zh": "畫面上的「脈絡」分頁就是把 `context` 事件裡的字串原樣印出來（[[render.py|def context_html(]]），"
               "`added_from` 之後的部分標成「這一步新加入」，family 列出的特殊 token 換成可以滑過看說明的深色標籤。",
         "en": "The page's Context tab prints the string in the `context` event as it is ([[render.py|def context_html(]]), marks everything "
               "after `added_from` as new in this step, and turns the special tokens the family lists into dark chips you can hover over for a note."},
    )),
    Section("model", "model",
            {"zh": "模型：prefill，然後一次一個 token", "en": "Model: prefill, then one token at a time"},
            {"zh": "一次 forward 讀完整段 prompt，之後每生成一個 token 再 forward 一次。",
             "en": "One forward pass reads the whole prompt; after that, each new token takes another forward pass."}, (
        {"zh": "harness 用一個 for 迴圈消費 `generate()` 產生的事件：`prefill` 和每個 `token` 都加上 step 編號轉送出去，"
               "最後的 `done` 留下來，改名成 `model_end` 送出。",
         "en": "The harness consumes the events that `generate()` yields in a for loop: `prefill` and each `token` are passed on with the step "
               "number, and the final `done` is kept and sent on as `model_end`."},
        Code("agent.py", "done = None", end='self.emit("model_end"',
             caption={"zh": "傳進去的 `self._stop.is_set` 是一個函式：模型每生成一個 token 前都會呼叫它，問「要停了嗎？」。",
                      "en": "The `self._stop.is_set` passed in is a function: the model calls it before every token to ask whether to stop."}),
        {"zh": "**prefill**：把整個 prompt 轉成 token id，一次 forward 算完。`use_cache=True` 留下每一層的狀態（一般 attention 層是 key / value，"
               "Qwen3.5 的 Gated DeltaNet 層是一個固定大小的狀態），之後每個新 token 只需要算自己那一個位置；"
               "`logits_to_keep=1` 只取最後一個位置的 logits，因為只需要「下一個 token」的分布（[[llm.py|def _forward(]]）。"
               "prefill 的時間主要看 prompt 有多長，模型分頁的表格裡有每一步的 prefill ms。",
         "en": "**prefill**: the whole prompt becomes token ids and goes through one forward pass. `use_cache=True` keeps each layer's state "
               "(the keys and values in ordinary attention layers, a fixed-size state in Qwen3.5's Gated DeltaNet layers), so each new token "
               "needs only its own position computed; `logits_to_keep=1` returns the logits for the last position only, since only the "
               "distribution of the next token is needed ([[llm.py|def _forward(]]). Prefill time depends mostly on the prompt's length; "
               "the table in the Model tab has each step's prefill ms."},
        Code("llm.py", "def generate(self, prompt", end='yield {"type": "prefill"',
             caption={"zh": "prompt 如果以 family 的 `thought_prefix` 結尾（Gemma 的 `<|channel>thought`、Qwen 的 `<think>`），模型一開始就在思考裡。",
                      "en": "If the prompt ends with the family's `thought_prefix` (`<|channel>thought` for Gemma, `<think>` for Qwen), the model starts out inside its thinking."}),
        {"zh": "**解碼迴圈**從 `tokens` 一個一個拿抽好的 token：平常由 `_one_at_a_time()` 產生，打開 MTP 時改由 `_drafted()`（{{mtp}}），"
               "迴圈本身不用知道是哪一種。每一圈：檢查停止 → 拿到一個 token → "
               "判斷它是不是特殊 token、有沒有切換 channel：哪些 token 切到思考、工具呼叫或回覆，由 family 的 `channel_switch` 決定"
               "（Gemma：[[families/gemma4.py|channel_switch={]]；Qwen：[[families/qwen3_5.py|channel_switch={]]）→ "
               "把目前為止的所有 token 整段 decode 再取差值，這樣被拆成好幾個 byte token 的中文字會等到湊齊才輸出 → 送出 `token` 事件。",
         "en": "**The decoding loop** takes sampled tokens from `tokens` one at a time: they normally come from `_one_at_a_time()`, and from "
               "`_drafted()` with MTP on ({{mtp}}); the loop itself does not need to know which. Each time around: check for stop → get a token → "
               "work out whether it is a special token and whether it switches channel, where the family's `channel_switch` says which tokens "
               "switch to thinking, a tool call or the reply (Gemma: [[families/gemma4.py|channel_switch={]]; Qwen: "
               "[[families/qwen3_5.py|channel_switch={]]) → decode all the tokens so far and take the difference, so that a Chinese character "
               "split across several byte tokens comes out only once it is whole → send a `token` event."},
        Code("llm.py", "decode_start = time.perf_counter()", end="**details,"),
        {"zh": "生成到停止 token，或到了 `max_new_tokens`（側欄的「每步 token 上限」），就跳出迴圈。",
         "en": "A stop token, or reaching `max_new_tokens` (the sidebar's Max tokens per step), ends the loop."},
        Code("llm.py", "if token_id in self.eos_ids:", end="if len(generated) == sampling.max_new_tokens:"),
        {"zh": "還沒停，迴圈就再跟 `tokens` 要下一個。`_one_at_a_time()` 這時才把剛抽到的 token 餵回模型，帶著快取算出下一個位置的 logits；"
               "它是一個 generator，迴圈跳出之後就不會再多跑一次 forward。",
         "en": "If it has not stopped, the loop asks `tokens` for the next one. Only then does `_one_at_a_time()` feed the token it just sampled "
               "back into the model, with the cache, for the next position's logits; it is a generator, so once the loop ends, no extra forward pass runs."},
        Code("llm.py", "def _one_at_a_time("),
        {"zh": "最後的 `done` 告訴 harness 為什麼停（`finish`：`stop_token`、`max_new_tokens` 或 `user_stopped`）、停在哪個 token、"
               "生成速度，以及整段原始輸出 `raw`（保留特殊 token，因為 harness 要靠它們解析）。",
         "en": "The final `done` tells the harness why it stopped (`finish`: `stop_token`, `max_new_tokens` or `user_stopped`), at which token, "
               "how fast it generated, and the whole raw output, `raw` (special tokens included, because the harness parses by them)."},
        Code("llm.py", "decode_s = time.perf_counter() - decode_start", end='"raw": self.tokenizer.decode(generated'),
    )),
    Section("sampling", "model",
            {"zh": "抽樣：每個 token 是怎麼選出來的", "en": "Sampling: how each token is picked"},
            {"zh": "temperature → top-k → top-p → 抽一個；同時記下模型原本的分布，給浮窗用。",
             "en": "temperature → top-k → top-p → draw one; and keep the model's own distribution for the popover."}, (
        {"zh": "`_sample()` 每個 token 跑一次。先對原始 logits 做 softmax，得到模型自己的分布（T = 1、什麼都沒剃掉）："
               "token 浮窗裡的「模型給的機率」、排名和前 5 名候選都來自這個分布。",
         "en": "`_sample()` runs once per token. It first takes the softmax of the raw logits, the model's own distribution (T = 1, nothing cut): "
               "the model probability, the rank and the top 5 candidates in a token's popover all come from it."},
        {"zh": "temperature ≤ 0 是 greedy，直接拿機率最高的。否則依序：logits ÷ T → 只留前 k 個（top-k，0 = 不限）→ "
               "重新 softmax → 從機率最高的往下累加，累積機率達到 top-p 為止（nucleus，至少留一個）→ 重新正規化 → "
               "`torch.multinomial` 抽一個。這段跟模型無關，兩個家族共用。",
         "en": "A temperature of 0 or less is greedy: it takes the most likely token. Otherwise, in order: logits ÷ T → keep only the top k "
               "(top-k; 0 = no limit) → softmax again → add up from the most likely down until the cumulative probability reaches top-p "
               "(nucleus, keeping at least one) → renormalize → draw one with `torch.multinomial`. None of this depends on the model; both families share it."},
        Code("llm.py", "def _sample(self, logits"),
        {"zh": "回傳的資料就是浮窗裡的數字：`p` 是被選中的 token 在原始分布的機率；`rank` 是它的名次；"
               "`kept` 是剃完還剩幾個候選；`p_sampled` 是它在剃完、重新正規化後的分布裡的機率。"
               "step 卡片上 token 的底色深淺就是用 `p` 算的（[[render.py|def heat(p: float)]]）：越深，代表模型原本越不看好它，卻被抽中了。",
         "en": "What it returns are the numbers in the popover: `p` is the chosen token's probability in the model's own distribution; `rank` "
               "is its place; `kept` is how many candidates were left after the cuts; `p_sampled` is its probability in the cut and "
               "renormalized distribution. The shade behind a token in a step card comes from `p` ([[render.py|def heat(p: float)]]): "
               "the darker it is, the less the model expected that token, and yet it was drawn."},
    )),
    Section("mtp", "model",
            {"zh": "MTP：草稿模型先猜，主模型一次驗證", "en": "MTP: the drafter guesses, the model checks them all at once"},
            {"zh": "側欄打開「MTP 推測解碼」後，主模型的一次 forward 可以產生好幾個 token，而每個 token 的機率都不變。",
             "en": "With MTP speculative decoding on in the sidebar, one forward pass of the model can produce several tokens, and every token's probability stays the same."}, (
        {"zh": "一般的解碼每個 token 都要跑一次完整的 forward，而這段時間主要花在把幾 GB 的權重從記憶體讀出來；"
               "同一次 forward 多算幾個位置，幾乎不會更慢（實測 Gemma 4 E2B 在 Apple M5 上，1 個位置約 56 ms，7 個位置約 70 ms）。"
               "所以只要有人便宜地先猜好接下來幾個 token，主模型就能一次 forward 把它們全部驗證。這就是推測解碼（speculative decoding）；"
               "MTP（multi-token prediction）是 Google 為 Gemma 4 訓練的猜法。",
         "en": "Ordinary decoding runs a full forward pass for every token, and most of that time goes into reading gigabytes of weights from "
               "memory; computing a few more positions in the same pass is hardly any slower (measured with Gemma 4 E2B on an Apple M5: about "
               "56 ms for 1 position, about 70 ms for 7). So if something cheap guesses the next few tokens first, the model can check them "
               "all in one forward pass. That is speculative decoding; MTP (multi-token prediction) is the guesser Google trained for Gemma 4."},
        {"zh": "猜的是草稿模型 `google/gemma-4-E2B-it-assistant`（[[families/__init__.py|DRAFTERS = {]]）：約 78M 參數、4 層、hidden size 256，"
               "主模型是 5.1B。它沒有自己的 KV cache：Gemma 4 E2B 的 35 層裡，後 20 層本來就不自己算 key / value，而是共用前面兩層"
               "（最後一個 sliding window 層和最後一個 full attention 層）算好的，草稿模型也讀這一份。第一次打開 MTP 時才載入，"
               "每次猜幾個 token 寫在它的 generation config（`num_assistant_tokens` = 6）。",
         "en": "The guesses come from the drafter `google/gemma-4-E2B-it-assistant` ([[families/__init__.py|DRAFTERS = {]]): about 78M "
               "parameters, 4 layers, hidden size 256, next to the model's 5.1B. It has no KV cache of its own: of Gemma 4 E2B's 35 layers, "
               "the last 20 do not compute their own keys and values but share those of two earlier layers (the last sliding window layer "
               "and the last full attention layer), and the drafter reads the same ones. It loads the first time MTP is switched on, and how "
               "many tokens it guesses at a time is in its generation config (`num_assistant_tokens` = 6)."},
        Code("llm.py", "def load_drafter(self):"),
        {"zh": "**猜**：輸入是「上一個 token 的 embedding」接上「主模型在上一個位置最後一層的 hidden state」，輸出下一個 token 的 logits，"
               "取機率最高的當作猜測。接著把猜到的 token 和草稿模型自己的 hidden state 當作下一次的輸入，再猜一個，最多 6 個。"
               "6 次都站在同一個位置，因為它看到的 key / value 只到主模型真正讀過的地方。",
         "en": "**Guess**: the input is the last token's embedding next to the model's last-layer hidden state at the last position; the output "
               "is the logits for the next token, and the most likely one is the guess. The guessed token and the drafter's own hidden state "
               "are then the input for another guess, up to 6. All 6 sit at the same position, because the keys and values it sees go only "
               "as far as the model has really read."},
        Code("llm.py", "def _draft("),
        {"zh": "**驗證**：主模型對「剛確定的 token + 6 個猜測」做一次 forward，`logits_to_keep=7` 拿回 7 個位置的分布："
               "第 i 個位置的分布，就是讀到第 i 個猜測之前時，下一個 token 的分布。從第一個猜測開始依序交給 `_sample()` 檢查："
               "收下就是一個不用另外 forward 的 token；第一個被退回的猜測，由主模型自己在那個位置選的 token 取代，後面的猜測全部作廢；"
               "6 個都收下時，最後一個位置的分布還能多給一個 token。所以一次 forward 產生 1 到 7 個 token。",
         "en": "**Check**: the model runs one forward pass over the token just settled plus the 6 guesses, and `logits_to_keep=7` returns the "
               "distributions at 7 positions: the one at position i is the distribution of the next token before reading guess i. Starting "
               "from the first guess, `_sample()` checks them in order: a guess it keeps is a token without a forward pass of its own; the "
               "first guess it turns down is replaced by the model's own pick at that position, and the guesses after it are dropped; when "
               "all 6 hold, the last position's distribution gives one more token. So one forward pass produces between 1 and 7 tokens."},
        Code("llm.py", "def _drafted("),
        {"zh": "作廢的猜測已經寫進了 KV cache，要用 `past.crop()` 拿掉；sliding window 層平常會馬上丟掉太舊的 key / value，"
               "`activate_past_recording()` 讓它先留著，等 `crop()` 之後再縮回去。給草稿模型的那份 key / value 也同樣切掉作廢的部分。",
         "en": "The dropped guesses are already in the KV cache, and `past.crop()` takes them out; sliding window layers normally drop keys and "
               "values that are too old right away, and `activate_past_recording()` keeps them until after the `crop()`. The keys and values "
               "for the drafter are cut back the same way."},
        {"zh": "**為什麼輸出不變**：`_sample()` 收到猜測時，greedy（T = 0）只在猜測剛好是第 1 名時收下，所以輸出和不開 MTP 一樣"
               "（一次算 7 個位置的數值和一次算 1 個會有極小的差異，兩個 token 機率幾乎相同時偶爾會選到另一個）。"
               "抽樣時，猜測被收下的機率就是 temperature、top-k、top-p 處理完之後它的機率 p；沒收下，就把它從候選裡拿掉，從其他候選抽。"
               "任何一個 token 最後出現的機率還是 p：草稿模型只決定要花幾次 forward，不決定寫出什麼。",
         "en": "**Why the output does not change**: given a guess, `_sample()` in greedy decoding (T = 0) keeps it only when it is the top token, "
               "so the output is the same as without MTP (computing 7 positions at once gives very slightly different numbers from computing 1, "
               "so when two tokens are almost equally likely it may now and then pick the other). When sampling, a guess is kept with its "
               "probability p after temperature, top-k and top-p; if it is not kept, it is taken out of the candidates and the token is drawn "
               "from the rest. Every token still comes out with probability p: the drafter decides how many forward passes it takes, not what gets written."},
        Code("llm.py", "at = (cand_ids == draft)", end="pick = int(torch.multinomial(weights, 1))"),
        {"zh": "畫面上：", "en": "On the page:"},
        (
            {"zh": "step 卡片裡被收下的猜測有底線；一串底線加上後面那個沒有底線的 token，就是主模型同一次 forward 的產出。",
             "en": "Kept guesses are underlined in the step card; a run of underlined tokens and the plain token after it come from one forward pass of the model."},
            {"zh": "token 浮窗會寫「草稿猜中」，或「草稿被退回」和草稿原本猜的是什麼。",
             "en": "A token's popover says the drafter guessed it, or that the guess was turned down and what the drafter had guessed."},
            {"zh": "step 卡片的模型段多一行：草稿收下幾個、解碼 forward 跑了幾次、不開 MTP 要跑幾次；模型分頁的表格也有這兩欄。",
             "en": "The step card's model section gets a line on how many guesses were kept, how many decoding forward passes ran and how many "
                   "it would take without MTP; the table in the Model tab has these two columns too."},
        ),
        {"zh": "收下的比例很看內容。在 Apple M5 上用 greedy 實測：工具呼叫幾乎都猜得中（35 個收下 34 個，約 15 → 43 tok/s），"
               "英文程式碼約一半，繁體中文的長回答只有一成左右，快不了多少。格式固定、可以預測的文字，草稿模型才猜得準。",
         "en": "How many are kept depends a lot on the content. Measured with greedy decoding on an Apple M5: tool calls are almost all guessed "
               "right (34 of 35 kept, about 15 → 43 tok/s), English code about half, and long answers in Traditional Chinese only about a "
               "tenth, hardly any faster. The drafter guesses well only where the text is fixed in form and predictable."},
    )),
    Section("parse", "harness",
            {"zh": "Harness：把輸出拆成思考、工具呼叫、回覆", "en": "Harness: splitting the output into thinking, tool calls and the reply"},
            {"zh": "模型只會吐文字；哪段是工具呼叫，要 harness 用 family 的 parser 讀出來。",
             "en": "The model only writes text; which part is a tool call, the harness reads with the family's parser."}, (
        {"zh": "chat template 有時會在 prompt 結尾先打開思考（family 的 `thought_prefix`）；這段在 prompt 裡、不在 `raw` 裡，"
               "所以先補回來，再交給 family 的 `parse`，`s.lang` 是它寫錯誤訊息用的語言。關掉「使用工具」時，就算模型寫了工具呼叫也當作沒有。",
         "en": "The chat template sometimes opens the thinking at the end of the prompt (the family's `thought_prefix`); that part is in the "
               "prompt, not in `raw`, so it is put back before the output goes to the family's `parse`, with `s.lang` as the language for its "
               "error messages. With Use tools off, a tool call the model writes anyway counts as none."},
        Code("agent.py", "prefix = family.thought_prefix if", end='self.emit("parsed"'),
        {"zh": "兩個 parser 回傳一樣的結構：`thinking`、`tool_calls`（每個有 `name`、`arguments`、`error`）、`content`，所以後面的 harness 不用管是哪個模型。",
         "en": "Both parsers return the same structure, `thinking`, `tool_calls` (each with `name`, `arguments` and `error`) and `content`, "
               "so the rest of the harness does not care which model it was."},
        {"zh": "**Gemma 4**：(1) 切出 `<|channel>thought` 到 `<channel|>` 之間的思考；(2) 用正規表達式找出每一個 `<|tool_call>call:名稱{…}<tool_call|>`"
               "（[[families/gemma4.py|CALL_RE = re.compile]]），參數交給小 parser；(3) 有開頭、沒有結尾的工具呼叫（例如被 token 上限截斷）記成格式錯誤。"
               "剩下的文字去掉停止 token，就是給使用者的回覆。",
         "en": "**Gemma 4**: (1) cut out the thinking between `<|channel>thought` and `<channel|>`; (2) find every "
               "`<|tool_call>call:name{…}<tool_call|>` with a regular expression ([[families/gemma4.py|CALL_RE = re.compile]]) and hand the "
               "arguments to a small parser; (3) record a tool call that opens but never closes (cut off at the token limit, say) as a format "
               "error. The text left over, without stop tokens, is the reply for the user."},
        Code("families/gemma4.py", "def parse_output("),
        {"zh": "Gemma 的參數不是 JSON：key 不加引號，字串用 `<|\"|>` 這個特殊 token 包起來，例如 `{timezone:<|\"|>Asia/Taipei<|\"|>}`。"
               "`_Parser` 是遞迴下降 parser：看到 `<|\"|>` 就一路讀到下一個 `<|\"|>`，`{` 讀物件、`[` 讀陣列，"
               "其他的當作數字、`true` / `false` 或 `null`。",
         "en": "Gemma's arguments are not JSON: keys have no quotes, and strings are wrapped in the special token `<|\"|>`, as in "
               "`{timezone:<|\"|>Asia/Taipei<|\"|>}`. `_Parser` is a recursive descent parser: at `<|\"|>` it reads up to the next `<|\"|>`, "
               "at `{` an object, at `[` an array, and anything else is a number, `true` / `false` or `null`."},
        Code("families/gemma4.py", "def value(self):"),
        {"zh": "**Qwen3.5**：思考在 `</think>` 之前；每個 `<tool_call>` 區塊裡找 `<function=名稱>` 和一段段 `<parameter=名稱>`。"
               "值都是純文字，所以要照工具的 schema 轉型別：schema 說是字串就保持原樣，說是數字、陣列或物件就用 JSON 讀，"
               "說是布林就看是不是 `true` / `false`；轉不過去就留著文字，讓下一關的 schema 驗證把型別錯誤回報給模型。",
         "en": "**Qwen3.5**: the thinking is before `</think>`; in each `<tool_call>` block it finds `<function=name>` and the "
               "`<parameter=name>` sections. Every value is plain text, so its type comes from the tool's schema: a string stays as it is, "
               "a number, array or object is read as JSON, and a boolean is whether it says `true` or `false`; text that does not convert "
               "stays text, and the schema check that follows reports the wrong type to the model."},
        Code("families/qwen3_5.py", "def parse_output("),
        Code("families/qwen3_5.py", "def convert(value: str, kind"),
    )),
    Section("stop", "stop",
            {"zh": "要不要停：由 harness 決定", "en": "Whether to stop: the harness decides"},
            {"zh": "模型只是停止生成；這一輪要不要結束，是 harness 看完輸出才決定的。",
             "en": "The model only stops generating; whether the turn ends is up to the harness, once it has read the output."}, (
        {"zh": "解析完先檢查兩種馬上結束的情況：你按了停止；或這一步沒有任何工具呼叫。沒有工具呼叫，回覆就是最終答案，停止原因是 `no_tool_call`；"
               "如果是被 token 上限截斷而沒有工具呼叫，停止原因記成 `max_new_tokens`。兩種情況都把回覆存成 assistant 訊息，下一輪會當成歷史。",
         "en": "After parsing, it first checks the two cases that end the turn at once: you pressed stop, or this step has no tool call. Without "
               "a tool call, the reply is the final answer and the stop reason is `no_tool_call`; if the generation was cut off at the token "
               "limit without a tool call, the stop reason is `max_new_tokens`. Either way, the reply is kept as an assistant message, part of "
               "the history for the next turn."},
        {"zh": "這裡看的是「解析出來有沒有工具呼叫」，不是「停在哪個 token」。Gemma 停在 `<|tool_response>` 就代表要用工具，"
               "但 Qwen 不管回答完或要用工具都停在 `<|im_end|>`；用解析結果判斷，兩個家族才能共用同一段程式。",
         "en": "What counts here is whether parsing found a tool call, not which token it stopped at. Gemma stopping at `<|tool_response>` does "
               "mean it wants a tool, but Qwen stops at `<|im_end|>` both after an answer and for a tool; deciding by the parse lets both "
               "families share this code."},
        Code("agent.py", 'if done["finish"] == "user_stopped":', end="if not calls:"),
        {"zh": "其他停止條件分散在迴圈各處：", "en": "The other ways to stop are spread around the loop:"},
        (
            {"zh": "for 迴圈跑完所有步數都沒 break → `max_steps`（{{prepare}}設定的預設值）。",
             "en": "The for loop uses up its steps without a break → `max_steps` (the default set in {{prepare}})."},
            {"zh": "等待確認時被停止或逾時 → `user_stopped` 或 `approval_timeout`（[[agent.py|stop_reason = \"approval_timeout\" if self._timed_out]]）。",
             "en": "Stopped or timed out while waiting for a confirmation → `user_stopped` or `approval_timeout` "
                   "([[agent.py|stop_reason = \"approval_timeout\" if self._timed_out]])."},
            {"zh": "連續 `MAX_FORMAT_ERRORS`（3）次寫出無法解析的工具呼叫 → `format_errors`。",
             "en": "`MAX_FORMAT_ERRORS` (3) tool calls in a row that cannot be parsed → `format_errors`."},
            {"zh": "模型或工具丟出任何例外 → `error`，錯誤訊息會顯示在畫面上，而不是讓 thread 默默死掉。",
             "en": "Any exception from the model or a tool → `error`; the message shows on the page instead of the thread dying silently."},
        ),
        Code("agent.py", "if format_errors >= MAX_FORMAT_ERRORS:", end='self.emit("run_end"',
             caption={"zh": "不管怎麼停，最後一定送出 `run_end`，帶著停止原因、最終回覆和整份訊息列表。",
                      "en": "However it stops, `run_end` always goes out last, with the stop reason, the final reply and the whole message list."}),
    )),
    Section("record", "harness",
            {"zh": "把工具呼叫寫回對話", "en": "Writing the tool call back into the conversation"},
            {"zh": "工具呼叫先變成一則 assistant 訊息，下一步由 chat template 寫回模型自己的格式。",
             "en": "A tool call first becomes an assistant message, which the chat template writes back in the model's own format in the next step."}, (
        {"zh": "有工具呼叫時，harness 給每個呼叫一個 id（`call_{step}_{i}`），寫成一則 assistant 訊息，"
               "格式和 OpenAI / Hugging Face 的 `tool_calls` 一樣；下一步 chat template 會把它寫回 Gemma 的 `<|tool_call>…<tool_call|>` 或 Qwen 的 `<tool_call>…</tool_call>`。",
         "en": "With tool calls, the harness gives each one an id (`call_{step}_{i}`) and writes them into an assistant message in the format "
               "of OpenAI's and Hugging Face's `tool_calls`; in the next step the chat template writes it back as Gemma's "
               "`<|tool_call>…<tool_call|>` or Qwen's `<tool_call>…</tool_call>`."},
        {"zh": "兩個家族在這裡有兩點不同，都由 family 決定：模型在工具呼叫前寫的文字，Qwen 的模板會保留，Gemma 的模板會把它挪到工具結果後面並關掉回合，"
               "所以 Gemma 丟掉這段文字（`text_with_calls`）；思考內容要存在哪個欄位，模板才讀得到，Gemma 是 `reasoning`、Qwen 是 `reasoning_content`（`reasoning_key`）。",
         "en": "The two families differ in two ways here, both decided by the family: text the model writes before a tool call is kept by "
               "Qwen's template, while Gemma's moves it after the tool results and closes the turn there, so Gemma's is dropped "
               "(`text_with_calls`); and the thinking has to go in the field the template reads it from, `reasoning` for Gemma and "
               "`reasoning_content` for Qwen (`reasoning_key`)."},
        {"zh": "接著先 render 一次「加上這則 assistant 訊息、但還沒有工具結果」的字串，存成 `prev_sequence`；`generation_prompt=False`，"
               "因為這時還輪不到模型寫。下一步的 prompt 和它比，多出來的就是工具結果。直接拿「上一步 prompt + 模型輸出」來比會不準，"
               "因為模板重新 render 時會改寫模型那一段（排序參數、換掉思考的標記）。",
         "en": "Then it renders the conversation with this assistant message but without the tool results yet, and keeps that as "
               "`prev_sequence`; `generation_prompt=False`, since it is not the model's turn to write yet. Compared with it, what the next "
               "step's prompt has in addition is the tool result. Comparing with the last prompt plus the model's output would be off, "
               "because the template rewrites the model's part when it renders it again (it sorts the arguments and replaces the thinking markup)."},
        Code("agent.py", 'ids = [f"call_{step}_{i}"', end="prev_sequence = self.llm.render("),
    )),
    Section("check", "harness",
            {"zh": "驗證與權限：模型從不直接碰環境", "en": "Validation and permission: the model never touches the environment"},
            {"zh": "每個呼叫要過三關：格式和 schema、重複檢查、（開著時）你的確認。",
             "en": "Each call has three checks to pass: format and schema, repeats, and (when it is on) your confirmation."}, (
        {"zh": "每個呼叫交給 `_handle_call()`。第一關是驗證：解析失敗、沒有這個工具、參數不符合 schema，都不會執行。",
         "en": "Each call goes to `_handle_call()`. The first check is validation: a call that failed to parse, names no known tool or has "
               "arguments that do not fit the schema never runs."},
        Code("agent.py", "def _handle_call(", end="args, errors = tool.validate(args, lang)"),
        {"zh": "`Tool.validate()` 檢查必要參數有沒有給、有沒有多出不認得的參數、型別和 enum 對不對，然後補上預設值。"
               "錯誤訊息是完整的句子，用這一輪開始時頁面的語言寫，因為它會原封不動變成模型下一步讀到的觀察。",
         "en": "`Tool.validate()` checks that the required arguments are there, that there are no unknown ones, and that the types and enums "
               "are right, then fills in the defaults. Each error is a full sentence in the language the page was in when the turn started, "
               "because it becomes, word for word, the observation the model reads in the next step."},
        Code("tools.py", "def validate(self, args: dict"),
        {"zh": "第二關記錄重複：把「工具名稱 + 排序後的參數 JSON」當成 key，跟這一輪之前的呼叫比；一模一樣就在 step 卡片標出"
               "「迴圈可能在打轉」。這只是提醒，不會擋。然後送出 `tool_call` 事件；有錯誤就直接把錯誤包成觀察回報，不進環境。",
         "en": "The second check records repeats: the tool name plus the arguments as sorted JSON is the key, compared with the earlier calls "
               "in this turn; an exact repeat is flagged in the step card as the loop maybe going in circles. It is only a warning and blocks "
               "nothing. Then the `tool_call` event goes out; with errors, the errors go straight back as the observation, without reaching the environment."},
        Code("agent.py", 'key = (call["name"], json.dumps', end='{"error": "; ".join(errors)}'),
        {"zh": "第三關是你。開著「執行工具前先問我」時，harness 送出 `approval_request`，然後停在 `_wait_for_decision()` 等。"
               "拒絕、停止、逾時都會變成一則錯誤觀察，模型下一步只看得到這段文字。",
         "en": "The third check is you. With Ask me before running a tool on, the harness sends `approval_request` and waits in "
               "`_wait_for_decision()`. A rejection, a stop or a timeout each becomes an error observation, and that text is all the model sees in the next step."},
        Code("agent.py", "if self.settings.approve_tools:", end='return self._observe(step, cid, call["name"], {"error": error}'),
        {"zh": "等待用的是 `threading.Condition`：每 0.5 秒醒來一次，直到你按下按鈕（`decide()` 會 `notify_all`）、按了停止，"
               "或超過 `APPROVAL_TIMEOUT`（[[agent.py|APPROVAL_TIMEOUT = ]]，5 分鐘）。逾時就當作拒絕並停止整輪，免得一個關掉的分頁把模型佔住。",
         "en": "The wait uses a `threading.Condition`: it wakes every 0.5 s until you press a button (`decide()` calls `notify_all`), press stop, "
               "or `APPROVAL_TIMEOUT` passes ([[agent.py|APPROVAL_TIMEOUT = ]], 5 minutes). A timeout counts as a rejection and stops the "
               "whole turn, so a closed tab cannot hold on to the model."},
        Code("agent.py", "def _wait_for_decision("),
        {"zh": "畫面這邊，等待中的呼叫會在聊天區顯示「允許／拒絕」按鈕，按下去就呼叫 `run.run.decide()`。"
               "「先問我」寫在 harness 的程式裡，而不是寫在 prompt 裡請模型遵守，所以模型沒有辦法繞過。",
         "en": "On the page, a waiting call shows Allow and Reject buttons in the chat, and pressing one calls `run.run.decide()`. Asking first "
               "is written into the harness's code, not into the prompt as a request to the model, so the model has no way around it."},
        Code("streamlit_app.py", "for call in render.pending_approvals(data):"),
    )),
    Section("execute", "env",
            {"zh": "環境：真的去執行工具", "en": "Environment: really running the tool"},
            {"zh": "通過三關才會呼叫 Python 函式；錯誤也會變成觀察，不會讓迴圈當掉。",
             "en": "Only after the three checks is the Python function called; errors become observations too, and never crash the loop."}, (
        {"zh": "送出 `tool_start` 後呼叫工具的 Python 函式。工具自己丟的 `ToolError` 是可以預期的錯誤（時區打錯、找不到頁面），"
               "訊息用這一輪的語言直接給模型看；其他例外也會被接住，變成 `{\"error\": …}`。",
         "en": "After `tool_start` goes out, the tool's Python function is called. A `ToolError` from the tool is an expected failure (a mistyped "
               "timezone, a page that does not exist), and its message goes to the model as it is, in the turn's language; any other exception "
               "is caught too and becomes `{\"error\": …}`."},
        Code("agent.py", 'self.emit("tool_start"', end="return self._observe(step, cid, call[\"name\"], result, ok=ok"),
        {"zh": "四個工具各自連到不同的環境：", "en": "The four tools reach different environments:"},
        (
            {"zh": "`get_current_time`：讀系統時鐘，用 IANA 時區名稱換算（[[tools.py|def get_current_time(]]）。",
             "en": "`get_current_time`: reads the system clock and converts with an IANA timezone name ([[tools.py|def get_current_time(]])."},
            {"zh": "`calculator`：在 AST 白名單上計算，從不呼叫 `eval()`。只允許數字、四則運算、次方和少數數學函式；"
                   "算次方前先估計位數，避免 `10**10**10` 這種算式把 thread 卡死。",
             "en": "`calculator`: computes over an AST allowlist and never calls `eval()`. Only numbers, arithmetic, powers and a few math functions "
                   "are allowed; before a power it estimates the number of digits, so an expression like `10**10**10` cannot hang the thread."},
            {"zh": "`search_wikipedia` / `read_wikipedia`：用 httpx 呼叫 Wikipedia API（[[tools.py|def search_wikipedia(]]），"
                   "和 ReAct 論文在 HotpotQA 上用的 search / read 是同一組動作。",
             "en": "`search_wikipedia` / `read_wikipedia`: call the Wikipedia API with httpx ([[tools.py|def search_wikipedia(]]), the same pair "
                   "of actions, search and read, that the ReAct paper used on HotpotQA."},
        ),
        Code("tools.py", "def _eval(node):",
             caption={"zh": "遞迴走過語法樹，遇到白名單以外的節點就丟 `ToolError`。",
                      "en": "Walks the syntax tree recursively and raises `ToolError` at any node outside the allowlist."}),
    )),
    Section("observe", "context",
            {"zh": "觀察接回脈絡，進下一圈", "en": "The observation goes back into the context for the next step"},
            {"zh": "工具結果變成一則 `tool` 訊息；下一圈 render 時，它就出現在模型的工具呼叫後面。",
             "en": "The tool result becomes a `tool` message; when the next step renders, it appears after the model's tool call."}, (
        {"zh": "工具結果（或錯誤）在 `_observe()` 裡轉成 JSON 字串，超過 `MAX_RESULT_CHARS`（2000 字元）就截斷並加上 `…(truncated)`，"
               "免得一個長網頁塞爆脈絡；送出 `tool_result` 事件，再把字串回傳。",
         "en": "In `_observe()` the tool result (or error) becomes a JSON string, cut at `MAX_RESULT_CHARS` (2000 characters) with "
               "`…(truncated)` added, so one long web page cannot flood the context; the `tool_result` event goes out, and the string is returned."},
        Code("agent.py", "def _observe("),
        {"zh": "回傳的字串變成一則 `role: tool` 的訊息 append 到 `self.messages`。同時更新連續格式錯誤的次數：這次是解析錯誤就加一，正常就歸零。",
         "en": "The returned string is appended to `self.messages` as a `role: tool` message. The count of format errors in a row is updated at "
               "the same time: one more for a parse error, back to zero otherwise."},
        Code("agent.py", "for cid, call in zip(ids, calls):", end="if format_errors >= MAX_FORMAT_ERRORS:"),
        {"zh": "然後回到 for 迴圈開頭，下一圈 render 時 chat template 把工具結果寫進脈絡，這就是 step 卡片上「上一步的觀察接回脈絡」那一段：",
         "en": "Then it is back to the top of the for loop, and when the next step renders, the chat template writes the tool result into the "
               "context; this is what the step card shows as the last step's observation, back in the context:"},
        (
            {"zh": "Gemma 4：`<|tool_response>response:名稱{value:<|\"|>…<|\"|>}<tool_response|>`，緊接在模型的工具呼叫後面，"
                   "還在同一個 model turn 裡，模型讀到結果後接著寫。",
             "en": "Gemma 4: `<|tool_response>response:name{value:<|\"|>…<|\"|>}<tool_response|>`, right after the model's tool call and "
                   "still in the same model turn; the model reads the result and carries on writing."},
            {"zh": "Qwen3.5：`<|im_start|>user` 加上 `<tool_response>` … `</tool_response>`，結果算是一則 user 訊息；"
                   "後面再開一個新的 `<|im_start|>assistant`，模型從新的回合開始寫。",
             "en": "Qwen3.5: `<|im_start|>user` followed by `<tool_response>` … `</tool_response>`, so the result is a user message; a new "
                   "`<|im_start|>assistant` follows, and the model writes in a new turn."},
        ),
        {"zh": "不管哪一種，模型下一步可能再呼叫一次工具，也可能直接回答。",
         "en": "Either way, in the next step the model may call a tool again or answer right away."},
    )),
    Section("ui", "ui",
            {"zh": "畫面：事件怎麼變成你看到的東西", "en": "The page: how events become what you see"},
            {"zh": "背景 thread 只管 append 事件；畫面每 0.5 秒把整張列表重畫一次。",
             "en": "The background thread only appends events; the page redraws the whole list every 0.5 s."}, (
        {"zh": "`workspace` 是一個 fragment：有 agent 在跑時，它每 0.5 秒自己重跑一次，只重畫聊天和右邊的檢視，不用整頁重跑。"
               "跑完的那一刻再整頁重跑一次，然後停止輪詢。",
         "en": "`workspace` is a fragment: while an agent runs, it reruns itself every 0.5 s and redraws only the chat and the views on the "
               "right, not the whole page. The moment the run ends, the whole page reruns once, and the polling stops."},
        Code("streamlit_app.py", "@st.fragment(run_every=", end="viz_panel(llm)"),
        {"zh": "每次重畫，`LiveRun.data()` 複製一份目前的事件列表交給 `render.build()`，按順序折成一個個 step。"
               "`run_start` 告訴它這一輪是哪個模型、哪個 family，之後畫特殊 token、停止 token 和使用者訊息的位置都問這個 family；"
               "`step_start` 開一個新的 step，`context`、`prefill`、`token`、`parsed` 填進去，工具相關的事件依 `call_id` 歸到各自的呼叫。"
               "之後所有 HTML 都是這個結構（和頁面語言）的純函式，所以執行到一半的任何時刻都畫得出來；"
               "切換語言也只是用另一種語言重畫同一份事件，跑完的對話也會跟著換。",
         "en": "On every redraw, `LiveRun.data()` copies the event list so far and hands it to `render.build()`, which folds it into steps in "
               "order. `run_start` says which model and family the turn uses, and from then on the special tokens, the stop tokens and where "
               "the user's message starts are all asked of that family; `step_start` opens a new step, `context`, `prefill`, `token` and "
               "`parsed` fill it in, and the tool events go to their calls by `call_id`. All the HTML after that is a pure function of this "
               "structure (and of the page's language), so it can be drawn at any moment of a run; switching the language just redraws the "
               "same events in the other one, finished turns included."},
        Code("render.py", "def build(events: list[dict])"),
        {"zh": "迴圈圖哪一格亮、哪條箭頭在流動，由最後一個事件決定：",
         "en": "Which box of the loop diagram lights up and which arrow flows is decided by the last event:"},
        Code("render.py", 'state["active"], state["edge"] = {'),
        {"zh": "一輪結束後，`absorb_finished()` 把 `run_end` 帶回來的完整訊息列表存成 `history`。下一個問題就接在它後面，"
               "所以 agent 記得前面聊過什麼，包括之前的工具結果。",
         "en": "When a turn ends, `absorb_finished()` keeps the full message list from `run_end` as `history`. The next question is appended "
               "to it, so the agent remembers what was said before, earlier tool results included."},
        Code("streamlit_app.py", "def absorb_finished("),
        {"zh": "停止按鈕呼叫 `Run.stop()`：設定 `_stop` 並叫醒正在等確認的 thread。解碼迴圈每個 token 前都會檢查 `_stop`，"
               "所以按下去之後最多再生成一個 token 就停。",
         "en": "The stop button calls `Run.stop()`: it sets `_stop` and wakes a thread that is waiting for a confirmation. The decoding loop "
               "checks `_stop` before every token, so after the press at most one more token is generated."},
        Code("agent.py", "def stop(self):", end="def decide(self, call_id"),
    )),
]

# (event, file, text on the line that sends it, what it carries, what the page does with it)
EVENTS = [
    ("run_start", "agent.py", '"run_start", run_id=',
     {"zh": "模型和 family、這一輪的設定、可用的工具", "en": "the model and family, the turn's settings, the available tools"},
     {"zh": "用哪個模型的格式畫這一輪；步數上限、解碼說明", "en": "whose format to draw the turn in; the step limit, the decoding notes"}),
    ("step_start", "agent.py", 'self.emit("step_start"',
     {"zh": "step 編號", "en": "the step number"},
     {"zh": "開一張新的 step 卡片；迴圈圖亮起「脈絡」", "en": "opens a new step card; Context lights up in the loop diagram"}),
    ("context", "agent.py", '"context",',
     {"zh": "完整 prompt、token 數、`added_from`、脈絡上限、各角色訊息數",
      "en": "the whole prompt, its token count, `added_from`, the context limit, the number of messages per role"},
     {"zh": "step 卡片的脈絡段、「脈絡」分頁", "en": "the step card's context section, the Context tab"}),
    ("prefill", "llm.py", 'yield {"type": "prefill"',
     {"zh": "prompt 的 token 數、prefill 耗時", "en": "the prompt's token count, the prefill time"},
     {"zh": "迴圈圖「模型」格、模型分頁的 prefill ms", "en": "the Model box in the loop diagram, prefill ms in the Model tab"}),
    ("token", "llm.py", '"type": "token",',
     {"zh": "id、piece、文字、channel、是否為特殊 token、機率、排名、前 5 名候選；開 MTP 時還有 `draft`（草稿猜了什麼、有沒有收下）",
      "en": "id, piece, text, channel, whether it is special, probability, rank, the top 5 candidates; with MTP, also `draft` (what the drafter guessed and whether it was kept)"},
     {"zh": "token 的熱度顏色、底線與浮窗、左邊逐字出現的回覆", "en": "the token's heat color, underline and popover; the reply appearing bit by bit on the left"}),
    ("model_end", "agent.py", 'self.emit("model_end"',
     {"zh": "停止原因、停止 token、token 數、tok/s、MTP 統計、原始輸出", "en": "why it stopped, the stop token, the token count, tok/s, the MTP counts, the raw output"},
     {"zh": "「停在 …」、tok/s、MTP 收下幾個、模型分頁的表格", "en": "“stopped at …”, tok/s, how many MTP guesses were kept, the table in the Model tab"}),
    ("parsed", "agent.py", 'self.emit("parsed"',
     {"zh": "思考、回覆、工具呼叫", "en": "the thinking, the reply, the tool calls"},
     {"zh": "Harness 段開始出現", "en": "the Harness section appears"}),
    ("tool_call", "agent.py", '"tool_call",',
     {"zh": "名稱、驗證後的參數、錯誤、是否重複、環境", "en": "the name, the validated arguments, errors, whether it repeats, the environment"},
     {"zh": "Harness 段的呼叫、驗證結果、重複警告", "en": "the call, the validation result and the repeat warning in the Harness section"}),
    ("approval_request", "agent.py", 'self.emit("approval_request"',
     {"zh": "等待確認的呼叫", "en": "the call waiting for a confirmation"},
     {"zh": "聊天區的「允許／拒絕」按鈕；迴圈圖閃爍", "en": "the Allow and Reject buttons in the chat; the loop diagram pulses"}),
    ("approval", "agent.py", 'self.emit("approval",',
     {"zh": "允許、拒絕、逾時或被停止", "en": "allowed, rejected, timed out or stopped"},
     {"zh": "Harness 段顯示你的決定", "en": "the Harness section shows your decision"}),
    ("tool_start", "agent.py", 'self.emit("tool_start"',
     {"zh": "工具名稱、環境", "en": "the tool name, the environment"},
     {"zh": "迴圈圖亮起「環境」", "en": "Environment lights up in the loop diagram"}),
    ("tool_result", "agent.py", 'self.emit("tool_result"',
     {"zh": "結果字串、成功與否、是否截斷、耗時", "en": "the result string, whether it succeeded, whether it was cut, the time it took"},
     {"zh": "環境段的結果、聊天區的工具標籤", "en": "the result in the Environment section, the tool chip in the chat"}),
    ("error", "agent.py", 'self.emit("error"',
     {"zh": "例外的類型和訊息", "en": "the exception's type and message"},
     {"zh": "聊天區和停止卡片的錯誤訊息", "en": "the error message in the chat and on the stop card"}),
    ("run_end", "agent.py", 'self.emit("run_end"',
     {"zh": "停止原因、最終回覆、整份訊息列表", "en": "the stop reason, the final reply, the whole message list"},
     {"zh": "停止卡片；訊息列表成為下一輪的 `history`", "en": "the stop card; the message list becomes the next turn's `history`"}),
]


if __name__ == "__main__":
    print(*build(Path(sys.argv[1] if len(sys.argv) > 1 else "_site")), sep="\n")
