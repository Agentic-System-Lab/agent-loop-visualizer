"""The 運作說明 (How it works) page: one user turn through the agent loop, with the code behind every part.

It is a static page on GitHub Pages, in Traditional Chinese and in English; the app's top bar links to
the one in the app's language. Build it from the project's folder with

    uv run python -m app.guide _site

which writes _site/zh/index.html and _site/en/index.html, each one file with its styles and script
inlined, and _site/index.html, which sends a reader on to the language they picked on the page's
switch before, or else their browser's. On every push to main, .github/workflows/pages.yml runs
tests/test_guide.py, then builds and publishes them.

The content points at code by text, not by line number: Code("agent/harnesses/simple_harness.py", "def run(self):") is
looked up in the file every time the page is built, so the line numbers on the page follow the
code when it changes, and each file:line chip and excerpt links to its lines on GitHub at the commit
the page was built from. A start text must match exactly one line of its file; tests/test_guide.py
checks every reference, so an edit that breaks one fails the tests instead of the page. It also
compares the code the page quotes with tests/guide_excerpts.txt, so an edit to quoted code fails
too, naming the sections whose text to reread before writing that file again.

Every paragraph is written in both languages next to each other ({"zh": ..., "en": ...}), and the
code excerpts between them are shared, so the two pages always show the same code.

Where a part of a section differs by what runs the model, or by the model, it is told once per
backend or once per model, in Tabs (TAB_KINDS): radio buttons that guide.css switches without any
script, and that guide.js keeps on the backend, or the model, the reader last picked, in every set of
tabs of that kind on the page.
"""

import ast
import base64
import html
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_for_filename

from app import render
from agent.i18n import LANGS, pick

ROOT = Path(__file__).resolve().parent.parent  # the project's folder, which the files the guide quotes are named from
ASSETS = Path(__file__).parent / "assets"
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
REF_RE = re.compile(r"\[\[([\w./-]+)\|(.+?)\]\]")
# The tabs a part of a section can come in: one per backend where what runs the model changes it, one per model where
# the model does, and one per harness where what runs the loop does. Each kind has the name of its row of tabs, and each tab its label and the small print beside it (the
# class that runs the model, or the family's module). A key with a dash is a narrower one of the key before the dash:
# "vllm-docker" and "vllm-url" are kinds of "vllm" (any vLLM server, where the two do not differ), "pi-page" and
# "pi-provider" of "pi" (pi asking this page's model or one of its own), and guide.js takes either for the other.
TAB_KINDS = {
    "backend": ({"zh": "用什麼跑模型", "en": "What runs the model"}, {
        "transformers": ({"zh": "transformers", "en": "transformers"}, "SimpleBackend"),
        "vllm": ({"zh": "vLLM", "en": "vLLM"}, "VLLMBackend"),
        "vllm-docker": ({"zh": "vLLM (Docker)", "en": "vLLM (Docker)"}, "VLLMBackend"),
        "vllm-url": ({"zh": "vLLM 伺服器（VLLM_URL）", "en": "vLLM server (VLLM_URL)"}, "VLLMBackend"),
        "fake": ({"zh": "測試", "en": "Tests"}, "FakeLLM"),
    }),
    "model": ({"zh": "模型", "en": "Model"}, {
        "gemma4": ({"zh": "Gemma 4", "en": "Gemma 4"}, "gemma4.py"),
        "qwen3_5": ({"zh": "Qwen3.5", "en": "Qwen3.5"}, "qwen3_5.py"),
        "gemma4-drafter": ({"zh": "Gemma 4 的草稿模型", "en": "Gemma 4's drafter"}, "MTP"),
    }),
    "harness": ({"zh": "用什麼跑迴圈", "en": "What runs the loop"}, {
        "simple": ({"zh": "內建 harness", "en": "Built-in harness"}, "SimpleHarness"),
        "pi": ({"zh": "pi", "en": "pi"}, "PiHarness"),
        "pi-page": ({"zh": "pi · 這個網頁的模型", "en": "pi · this page's model"}, "model_endpoint.py"),
        "pi-provider": ({"zh": "pi · pi 自己的模型", "en": "pi · pi's own models"}, "models.json · /login"),
    }),
}
MAX_TABS = 4  # guide.css hides the panels of the tabs not picked, by place, up to this many


@dataclass(frozen=True)
class Code:
    """An excerpt: from the line containing `start` through the statement that begins there, or
    through the statement on the first line at or after it that contains `end`."""

    file: str
    start: str
    end: str | None = None
    caption: dict | None = None  # {"zh": ..., "en": ...}


@dataclass(frozen=True)
class Tabs:
    """One part of a section told once for each backend or each model, {key: body}, in tabs in that order; the keys are
    all of one kind in TAB_KINDS."""

    panels: dict

    @property
    def kind(self) -> str:
        return next(kind for kind, (_, tabs) in TAB_KINDS.items() if set(self.panels) <= set(tabs))


@dataclass(frozen=True)
class Section:
    key: str
    stage: str
    title: dict
    lead: dict
    body: tuple  # paragraphs ({"zh", "en"} dicts), bullet lists (tuples of them), Code excerpts and Tabs, in reading order


def parts(body: tuple):
    """The paragraphs, lists and excerpts of a body in reading order, those in each tab included."""
    for item in body:
        if isinstance(item, Tabs):
            for panel in item.panels.values():
                yield from parts(panel)
        else:
            yield item


# ---------- finding code ----------

def source(file: str) -> tuple[list[str], ast.Module | None]:
    """A file's lines, and for Python its syntax tree, which says where a statement ends; any other file (compose.yaml,
    say) is quoted line by line."""
    text = (ROOT / file).read_text(encoding="utf-8")
    return text.splitlines(), ast.parse(text) if file.endswith(".py") else None


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
    """The excerpt's first and last lines; outside Python it ends on the line with `end`, or is the one line."""
    lines, tree = source(code.file)
    lo = find_line(code.file, code.start)
    if code.end is None:
        return lo, statement_end(tree, lo) if tree else lo
    last = next((n for n in range(lo, len(lines) + 1) if code.end in lines[n - 1]), None)
    if last is None:
        raise LookupError(f"{code.file}: {code.end!r} is not on or after line {lo}")
    return lo, statement_end(tree, last) if tree else last


# ---------- HTML ----------

def esc(value) -> str:
    return html.escape(str(value), quote=True)


def lines_text(lo: int, hi: int, lang: str) -> str:
    return pick(lang, f"第 {lo} 行", f"line {lo}") if lo == hi else pick(lang, f"第 {lo}–{hi} 行", f"lines {lo}–{hi}")


def github_html(file: str, lo: int, hi: int, lang: str, commit: str) -> str:
    """An icon that opens the lines on GitHub, at the commit their numbers were looked up in, in a new tab, so the
    reader keeps their place on the page. guide.css draws it."""
    anchor = f"L{lo}" if lo == hi else f"L{lo}-L{hi}"
    title = pick(lang, "在 GitHub 上看這一行", "See this line on GitHub") if lo == hi else pick(lang, "在 GitHub 上看這幾行", "See these lines on GitHub")
    label = pick(lang, f"在 GitHub 上看 {file} {lines_text(lo, hi, lang)}", f"{file}, {lines_text(lo, hi, lang)}, on GitHub")
    return (
        f'<a class="g-gh" href="{REPO}/blob/{esc(commit)}/{esc(file)}#{anchor}" target="_blank" rel="noopener" '
        f'title="{title}" aria-label="{esc(label)}"></a>'
    )


def ref_html(file: str, text: str, lang: str, commit: str) -> str:
    """An inline file:line chip, whose hover shows the line itself, and the icon that opens it on GitHub."""
    line = find_line(file, text)
    src = source(file)[0][line - 1].strip()
    return f'<code class="ref" title="{esc(src)}">{esc(file)}:{line}</code>{github_html(file, line, line, lang, commit)}'


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


def prose_html(text: str, lang: str, commit: str) -> str:
    """Escape a paragraph and mark it up: `code`, **bold**, [[file|text]] becomes a file:line chip
    linked to that line at `commit` on GitHub, {{key}} a link to that section."""
    out, last = [], 0
    for m in REF_RE.finditer(text):
        out.append(_inline(text[last:m.start()], lang))
        out.append(ref_html(m.group(1), m.group(2), lang, commit))
        last = m.end()
    out.append(_inline(text[last:], lang))
    return "".join(out)


def code_html(code: Code, lang: str, commit: str) -> str:
    lo, hi = span(code)
    lines = source(code.file)[0][lo - 1:hi]
    body = highlight(textwrap.dedent("\n".join(lines)), get_lexer_for_filename(code.file), HtmlFormatter(nowrap=True))
    rows = body.split("\n")[: hi - lo + 1]
    gutter = "".join(f'<span class="ln">{n}</span>{row}\n' for n, row in zip(range(lo, hi + 1), rows))
    caption = f'<span class="src-cap">{prose_html(code.caption[lang], lang, commit)}</span>' if code.caption else ""
    return (
        f'<figure class="src"><figcaption><b>{esc(code.file)}</b><span class="src-lines">{lines_text(lo, hi, lang)}</span>'
        f"{github_html(code.file, lo, hi, lang, commit)}{caption}</figcaption><pre>{gutter}</pre></figure>"
    )


def tabs_html(tabs: Tabs, name: str, lang: str, commit: str) -> str:
    """A tab per backend or per model: radio buttons named `name`, the first one checked, then a panel for each.
    guide.css shows the panel of the one checked, so they switch without guide.js; with it, every set of tabs of the
    same kind follows the reader's pick."""
    title, labels = TAB_KINDS[tabs.kind]
    bar = "".join(
        f'<label><input type="radio" name="{name}" value="{key}"{" checked" if i == 0 else ""}>'
        f"<span>{esc(labels[key][0][lang])}</span><small>{esc(labels[key][1])}</small></label>"
        for i, key in enumerate(tabs.panels)
    )
    panels = "".join(
        f'<div class="g-panel">{"".join(body_html(item, lang, commit) for item in body)}</div>' for body in tabs.panels.values()
    )
    return (
        f'<div class="g-tabs" data-kind="{tabs.kind}"><div class="g-tab-bar" role="radiogroup" aria-label="{esc(title[lang])}">'
        f"{bar}</div>{panels}</div>"
    )


def body_html(item, lang: str, commit: str, name: str = "") -> str:
    if isinstance(item, Code):
        return code_html(item, lang, commit)
    if isinstance(item, Tabs):
        return tabs_html(item, name, lang, commit)
    if isinstance(item, tuple):
        return "<ul>" + "".join(f"<li>{prose_html(i[lang], lang, commit)}</li>" for i in item) + "</ul>"
    return f"<p>{prose_html(item[lang], lang, commit)}</p>"


def section_html(n: int, sec: Section, lang: str, commit: str) -> str:
    label, color = STAGES[sec.stage]
    return (
        f'<section class="g-sec" id="g-{sec.key}" style="--stage: {color}">'
        f'<header class="g-head"><span class="g-num">{n}</span><div><p class="g-stage">{label[lang]}</p>'
        f'<h2>{esc(sec.title[lang])}</h2><p class="g-lead">{prose_html(sec.lead[lang], lang, commit)}</p></div></header>'
        f'{"".join(body_html(item, lang, commit, f"g-{sec.key}-{i}") for i, item in enumerate(sec.body))}</section>'
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


def events_html(lang: str, commit: str) -> str:
    rows = "".join(
        f"<tr><td><code>{esc(kind)}</code></td><td>{ref_html(file, text, lang, commit)}</td>"
        f"<td>{prose_html(carries[lang], lang, commit)}</td><td>{prose_html(shows[lang], lang, commit)}</td></tr>"
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


def logo_url() -> str:
    """The site's logo (assets/logo.svg, the app's too) as a data URL, so the page stays one file."""
    return "data:image/svg+xml;base64," + base64.b64encode((ASSETS / "logo.svg").read_bytes()).decode()


def page_html(commit: str, lang: str) -> str:
    """The page body; `commit` is the commit its line numbers were looked up in, and which its links to GitHub open."""
    sections = "".join(section_html(n, sec, lang, commit) for n, sec in enumerate(SECTIONS, 1))
    link = f'<a href="{REPO}/tree/{esc(commit)}"><code>{esc(commit[:7])}</code></a>'
    return (
        f'{lang_switch_html(lang)}<main class="guide"><header class="g-hero"><img class="g-logo" src="{logo_url()}" alt="">'
        f'<p class="g-kicker">{pick(lang, "運作說明", "HOW IT WORKS")}</p>'
        f'<h1>{pick(lang, "一個問題在 agent 迴圈裡怎麼走", "How one question travels through the agent loop")}</h1>'
        f'<p class="g-intro">{prose_html(INTRO[lang], lang, commit)}</p>'
        f'<p class="g-meta"><a href="{REPO}">{pick(lang, "原始碼和執行方式在 GitHub", "The source and how to run it are on GitHub")}</a> · '
        f'{pick(lang, "行號對應 commit", "line numbers from commit")} {link}</p></header>'
        f'<div class="g-diagram">{diagram_html(lang)}</div>{toc_html(lang)}{sections}{events_html(lang, commit)}</main>'
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
        f'<link rel="icon" href="{logo_url()}">'
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
    "兩個模型跑的是同一個迴圈；迴圈由這個專案自己的 harness 跑，也可以交給 pi 這個 coding agent（{{overview}}）。問題可以附上圖片、聲音或影片（{{media}}），打開「讓模型寫程式」時，"
    "模型還能在 Docker 沙箱裡執行 Python（{{execute}}）。模型可以在這台電腦上用 transformers 跑，也可以交給一台 vLLM 伺服器（{{backends}}）。"
    "兩個模型、兩種跑法或兩個 harness 不一樣的地方，分成頁籤（tab）各自說明，一個模型、一種 backend 或一個 harness 一個；在任何一段選了一個，整頁同一類的頁籤都跟著換。每段都附上做這件事的程式碼；`agent/harnesses/simple_harness.py:99` 這類標記是檔案的路徑和行號，"
    "滑過去會顯示那一行，點旁邊的 GitHub 圖示會在新分頁打開那個檔案、跳到那一行，程式片段標題上的圖示也一樣。"
    "行號是產生這一頁時從原始碼查出來的，連結也指向那個 commit；每次推上 main，GitHub Actions 都會重新產生，程式改了也會跟著更新。",
    "en": "Everything that happens between pressing send and seeing the reply, in the order it really runs. The app's sidebar picks "
    "Gemma 4 E2B or Qwen3.5 0.8B; both models run the same loop, which this project's own harness runs, or pi, a coding agent ({{overview}}). "
    "A question can carry images, sound or video ({{media}}), and with Let the model write code on, the model can also run Python in a Docker sandbox ({{execute}}). "
    "The model runs on this machine with transformers, or on a vLLM server ({{backends}}). Where the two models, the two ways of running "
    "them or the two harnesses differ, a section tells each in a tab of its own, one per model, per backend or per harness, and picking one in any section switches every tab "
    "of that kind on the page. "
    "Every part comes with the code that does it; a mark like `agent/harnesses/simple_harness.py:99` is a file's path and a line number, hovering over it shows that line, "
    "and the GitHub icon beside it opens the file at that line in a new tab, as the icon on an excerpt's title does for its lines. "
    "The line numbers are looked up in the source when this page is built, and the links go to that commit; GitHub Actions rebuilds it on every push to main, so they follow the code as it changes.",
}

SECTIONS = [
    Section("overview", "ui",
            {"zh": "總覽：兩條 thread、一張事件列表", "en": "Overview: two threads, one event list"},
            {"zh": "你送出一個問題就是一輪（一個 `SimpleHarness`）；一輪裡是一個迴圈，每一圈叫一個 step。",
             "en": "Each question you send is one turn (one `SimpleHarness`); inside a turn is a loop, and each time around it is a step."}, (
        {"zh": "每個 step 做同一件事：harness 把整段對話 render 成一個字串（脈絡）→ 模型一次推論、一個 token 一個 token 生成 → "
               "harness 解析輸出。有工具呼叫，就驗證、（需要時問你）、在環境執行，把結果接回對話，進下一圈；"
               "沒有工具呼叫，就把回覆交給你，這一輪結束。",
         "en": "Every step does the same thing: the harness renders the whole conversation into one string (the context) → the model runs "
               "one inference, generating one token at a time → the harness parses the output. If there is a tool call, it validates it, "
               "asks you if it should, runs it in the environment and adds the result to the conversation for the next step; "
               "if there is none, it hands you the reply, and the turn ends."},
        {"zh": "整個 app 有兩條 thread。Streamlit 的 script thread 負責畫面；agent 跑在另一條背景 thread"
               "（[[app/page.py|self.thread = threading.Thread]]）。兩邊只透過三樣東西溝通：",
         "en": "The app runs on two threads. Streamlit's script thread draws the page; the agent runs on a background thread "
               "([[app/page.py|self.thread = threading.Thread]]). The two talk through just three things:"},
        (
            {"zh": "agent 每走到一個階段，就把一個事件 append 到 `events` 列表（[[agent/harnesses/simple_harness.py|def emit(self, event_type]]）；畫面只負責把這張列表畫出來。",
             "en": "At every stage the agent appends an event to the `events` list ([[agent/harnesses/simple_harness.py|def emit(self, event_type]]); the page only draws that list."},
            {"zh": "按「停止」時，畫面呼叫 `SimpleHarness.stop()`（[[agent/harnesses/simple_harness.py|def stop(self):]]）。",
             "en": "Pressing Stop calls `SimpleHarness.stop()` ([[agent/harnesses/simple_harness.py|def stop(self):]])."},
            {"zh": "按「允許」或「拒絕」時，畫面呼叫 `SimpleHarness.decide()`（[[agent/harnesses/simple_harness.py|def decide(self, call_id]]）。",
             "en": "Pressing Allow or Reject calls `SimpleHarness.decide()` ([[agent/harnesses/simple_harness.py|def decide(self, call_id]])."},
        ),
        {"zh": "這三樣就是畫面對「跑迴圈的那一層」的全部要求，`Harness`（[[agent/harnesses/harness.py|class Harness(Protocol):]]）。"
               "有兩個 harness 跑得了它：這個專案自己寫的 `SimpleHarness`，以及側欄的「用什麼跑迴圈」選 pi 時的 `PiHarness`，它把迴圈交給 pi 這個 coding agent。"
               "兩者做法不同的地方，下面每段都分成頁籤；pi 問這個網頁的模型和問它自己的模型也不一樣時，再各分一個頁籤。"
               "另一個會把自己的迴圈報成事件的 agent，例如 Claude Agent SDK 或 Codex，也可以照這樣再寫一個 harness。",
         "en": "These three are all the page asks of whatever runs the loop, the `Harness` ([[agent/harnesses/harness.py|class Harness(Protocol):]]). "
               "Two harnesses run it: this project's own, `SimpleHarness`, and, with pi picked under What runs the loop in the sidebar, `PiHarness`, which hands "
               "the loop to pi, a coding agent. Where the two do a part differently, the sections below tell it in a tab for each; where pi asking this page's "
               "model and pi asking one of its own differ too, in a tab for each of those. Another agent that reports its loop as events, such as the "
               "Claude Agent SDK or Codex, could be one more harness written the same way."},
        Tabs({
            "simple": (
                {"zh": "`SimpleHarness.run()` 就是迴圈本身（[[agent/harnesses/simple_harness.py|def run(self):]]）：它 render 脈絡、讓模型生成、解析輸出、執行工具，"
                       "每走到一個階段就送出一個事件。模型在這台電腦上用 transformers 跑，或在 vLLM 伺服器上（{{backends}}），每個 token 都看得到。",
                 "en": "`SimpleHarness.run()` is the loop itself ([[agent/harnesses/simple_harness.py|def run(self):]]): it renders the context, has the model "
                       "generate, parses the output and runs the tools, sending an event at every stage. The model runs on this machine with transformers, or on a "
                       "vLLM server ({{backends}}), and every token shows."},
            ),
            "pi": (
                {"zh": "`PiHarness` 每個問題都用 RPC 模式啟動一個 pi 子行程（[[agent/harnesses/pi_harness.py|class Process:]]）：指令從 stdin 送進去，紀錄從 stdout 讀出來，"
                       "一行一個 JSON。迴圈是 pi 在跑，`PiHarness` 只把 pi 的紀錄翻成頁面的事件；pi 的一個 turn（向模型送一次請求，加上回覆裡要執行的工具）就是一個 step。",
                 "en": "`PiHarness` starts pi as a child process in RPC mode for each question ([[agent/harnesses/pi_harness.py|class Process:]]): commands go in on stdin, "
                       "and records come out on stdout, one JSON object per line. pi runs the loop, and `PiHarness` only turns pi's records into the page's events; one "
                       "of pi's turns (one request to the model, and the tools its reply calls) is a step."},
                Code("agent/harnesses/pi_harness.py", "def _on(self, rec: dict) -> bool:"),
                {"zh": "pi 可以問這個網頁的模型，也可以問它自己登入過的模型（Claude、GPT…），由側欄的「pi 問的模型」決定。問這個網頁的模型時，"
                       "`model_endpoint.py` 把記憶體裡的模型當成一個相容 OpenAI 的端點提供給 pi，每個 token 照樣看得到；問 pi 自己的模型時，provider 的 API 只傳回文字。",
                 "en": "pi asks either this page's model or one it is logged in to (Claude, GPT…), as the sidebar's The model pi asks says. On this page's model, "
                       "`model_endpoint.py` serves the model in memory to pi as an OpenAI-compatible endpoint, and every token still shows; on one of pi's own, the "
                       "provider's API sends back text alone."},
            ),
        }),
        {"zh": "專案分成兩塊。`agent/` 跑一輪：跑迴圈的 harness、工具、模型，以及跑模型的那一層，它從不 import Streamlit。"
               "`app/` 是你看到的東西，站在 agent 上面：開始一輪、把 harness 送出的事件畫出來、把你的點擊傳回去。"
               "根目錄的 `streamlit_app.py` 只負責啟動 `app/page.py`。`agent/` 裡每個檔案的分工：",
         "en": "The project has two parts. `agent/` runs a turn: the harness that runs the loop, the tools, the model and what runs it, and it never imports Streamlit. "
               "`app/` is what you see, on top of the agent: it starts a turn, draws the events the harness sends and passes your clicks back. "
               "`streamlit_app.py` at the top only starts `app/page.py`. What each file in `agent/` does:"},
        (
            {"zh": "`agent/harnesses/`：跑迴圈的那一層。`harness.py` 是畫面對它的要求 `Harness`，以及每個 harness 送出的事件；"
                   "`simple_harness.py` 是這個專案的 harness，迴圈本體 `SimpleHarness.run()`（[[agent/harnesses/simple_harness.py|def run(self):]]）；"
                   "`pi_harness.py` 把迴圈交給 pi；`pi_extension.ts` 是網頁讓 pi 載入的 extension，回報 pi 送出的每個請求、在 pi 執行工具前先問你；"
                   "`model_endpoint.py` 把這個網頁的模型當成一個相容 OpenAI 的端點提供給 pi。",
             "en": "`agent/harnesses/`: what runs the loop. `harness.py` is what the page asks of it, `Harness`, and the events every harness sends; "
                   "`simple_harness.py` is this project's harness, with the loop itself, `SimpleHarness.run()` ([[agent/harnesses/simple_harness.py|def run(self):]]); "
                   "`pi_harness.py` hands the loop to pi; `pi_extension.ts` is the extension the page has pi load, which reports every request pi sends and asks you "
                   "before pi runs a tool; and `model_endpoint.py` serves this page's model to pi as an OpenAI-compatible endpoint."},
            {"zh": "`agent/model_backends/backend.py`：harness 和畫面對「跑模型的那一層」的全部要求，`Backend`（[[agent/model_backends/backend.py|class Backend(Protocol):]]）："
                   "`render()`、`encode()`、`generate()`，以及 token 事件裡哪些欄位一定有、哪些要 backend 知道才有。"
                   "`simple_backend.py` 的 `SimpleBackend` 是其中一種，`vllm_backend.py` 是另一種（{{backends}}）。",
             "en": "`agent/model_backends/backend.py`: everything the harness and the page ask of whatever runs the model, `Backend` ([[agent/model_backends/backend.py|class Backend(Protocol):]]): "
                   "`render()`, `encode()` and `generate()`, and which fields of a token event are always there and which only when the backend knows them. "
                   "`SimpleBackend` in `simple_backend.py` is one backend, and `vllm_backend.py` another ({{backends}})."},
            {"zh": "`agent/model_backends/simple_backend.py`：本機推論，自己寫的解碼迴圈 `generate()`（[[agent/model_backends/simple_backend.py|def generate(self, prompt]]）和抽樣 `_sample()`；"
                   "打開 MTP 時再加上一個草稿模型（{{mtp}}）。",
             "en": "`agent/model_backends/simple_backend.py`: local inference, with a hand-written decoding loop, `generate()` ([[agent/model_backends/simple_backend.py|def generate(self, prompt]]), "
                   "and sampling, `_sample()`; with MTP on, a drafter joins in ({{mtp}})."},
            {"zh": "`agent/model_backends/vllm_backend.py`：模型在 vLLM 伺服器上跑時用的 `Backend`（[[agent/model_backends/vllm_backend.py|class VLLMBackend:]]），"
                   "伺服器是 `VLLM_URL` 指的那台，或是畫面在 Docker 裡啟動的那台。prompt 照樣在這裡 render、切成 token，"
                   "token id 送到 vLLM 伺服器，每個 token 帶著它的 log 機率和前 5 名候選回來；取樣時剃掉了哪些候選，留在伺服器上。",
             "en": "`agent/model_backends/vllm_backend.py`: the `Backend` for a model on a vLLM server ([[agent/model_backends/vllm_backend.py|class VLLMBackend:]]), "
                   "the one at `VLLM_URL` or the one the page starts in Docker. The prompt is still rendered "
                   "and tokenized here, the token ids go to a vLLM server, and each token comes back with its log probability and the top 5 candidates; "
                   "which candidates the sampling cut stays on the server."},
            {"zh": "`agent/model_backends/vllm_docker.py`：沒設 `VLLM_URL`、側欄又選了 vLLM 時，照 `docker-compose/compose.yaml` 用 Docker Compose 啟動 vLLM 伺服器，"
                   "等到它能回應（[[agent/model_backends/vllm_docker.py|def start(model_id]]），在 GPU 上預設開 MTP 推測解碼；換回 transformers 時停掉它（{{backends}}）。",
             "en": "`agent/model_backends/vllm_docker.py`: without `VLLM_URL`, when the sidebar picks vLLM, it starts a vLLM server with Docker Compose from "
                   "`docker-compose/compose.yaml` and waits until it answers ([[agent/model_backends/vllm_docker.py|def start(model_id]]), on a GPU with MTP "
                   "speculative decoding on by default; picking transformers again stops it ({{backends}})."},
            {"zh": "`agent/model_families/`：跟著模型走的東西。`gemma4.py`、`qwen3_5.py` 各自負責讀懂自己模型的輸出，並說明它的特殊 token（{{families}}）。",
             "en": "`agent/model_families/`: everything that goes with the model. `gemma4.py` and `qwen3_5.py` each read their own model's output "
                   "and explain its special tokens ({{families}})."},
            {"zh": "`agent/tools.py`：每個工具的 schema（給模型看）和 Python 函式（給 harness 執行），全部列在 `TOOLS`（[[agent/tools.py|TOOLS = {]]）。"
                   "其中 `run_python` 在 `agent/sandbox/Dockerfile` 建的 Docker 容器裡跑模型寫的程式（{{execute}}）。",
             "en": "`agent/tools.py`: each tool's schema (for the model to read) and Python function (for the harness to run), "
                   "all listed in `TOOLS` ([[agent/tools.py|TOOLS = {]]). One of them, `run_python`, runs the model's code in a Docker container "
                   "built from `agent/sandbox/Dockerfile` ({{execute}})."},
            {"zh": "`agent/media.py`：問題附上的檔案：存進這段對話的資料夾、寫進使用者訊息，再解碼成模型的 processor 要的格式（{{media}}）。",
             "en": "`agent/media.py`: the files attached to a question: it saves them in the conversation's folder, writes them into the user's "
                   "message and decodes them for the model's processor ({{media}})."},
            {"zh": "`agent/i18n.py`：頁面的兩種語言，繁體中文和英文；`pick()` 照語言挑出其中一段文字（[[agent/i18n.py|def pick(]]）。"
                   "harness 也用它：模型讀到的錯誤訊息，用這一輪開始時頁面的語言寫。",
             "en": "`agent/i18n.py`: the page's two languages, Traditional Chinese and English; `pick()` chooses the text for one of them ([[agent/i18n.py|def pick(]]). "
                   "The harness uses it too: the errors the model reads are in the language the page was in when the turn started."},
        ),
        {"zh": "`app/` 裡：", "en": "In `app/`:"},
        (
            {"zh": "`app/page.py`：版面、側欄設定、模型選單、背景 thread、執行中每 0.5 秒重畫一次。",
             "en": "`app/page.py`: the layout, the sidebar settings, the model menu and the background thread; it redraws every 0.5 s while a run is live."},
            {"zh": "`app/render.py`：把事件列表折成 step（[[app/render.py|def build(events: list[dict])]]），再畫成 HTML。",
             "en": "`app/render.py`: folds the event list into steps ([[app/render.py|def build(events: list[dict])]]) and draws them as HTML."},
            {"zh": "`app/guide.py`：這一頁，從它引用的程式產生出來。",
             "en": "`app/guide.py`: this page, built from the code it quotes."},
        ),
        {"zh": "兩塊之外，`models/` 放下載的模型，每個模型一個資料夾，git 不追蹤（{{files}}）；`docker-compose/` 放畫面在 Docker 裡啟動 vLLM 用的設定（{{backends}}）。",
         "en": "Beside the two, `models/` holds the downloaded models, one folder each, which git ignores ({{files}}), and `docker-compose/` what the page "
               "starts vLLM in Docker from ({{backends}})."},
    )),
    Section("families", "model",
            {"zh": "模型家族：換模型只換這一層", "en": "Model families: switching models changes only this layer"},
            {"zh": "harness、解碼迴圈和畫面都只寫一次；每個模型不一樣的地方，集中在一個 `Family` 物件裡。",
             "en": "The harness, the decoding loop and the page are written once; what differs between models is gathered in one `Family` object."}, (
        {"zh": "兩個模型的迴圈完全一樣，不一樣的是它們寫字的格式：",
         "en": "The two models run exactly the same loop; what differs is the format they write in:"},
        Tabs({
            "gemma4": ((
                {"zh": "回合：`<|turn>user` … `<turn|>`。", "en": "Turns: `<|turn>user` … `<turn|>`."},
                {"zh": "工具呼叫：`<|tool_call>call:名稱{key:<|\"|>值<|\"|>}<tool_call|>`。",
                 "en": "Tool calls: `<|tool_call>call:name{key:<|\"|>value<|\"|>}<tool_call|>`."},
                {"zh": "工具結果：接在同一個 model turn 裡（`<|tool_response>` … `<tool_response|>`）。",
                 "en": "Tool results: in the same model turn (`<|tool_response>` … `<tool_response|>`)."},
                {"zh": "停止 token：回答完停在 `<turn|>`、要用工具時停在 `<|tool_response>`，看停止 token 就知道是哪一種。",
                 "en": "Stop tokens: `<turn|>` after an answer and `<|tool_response>` when it wants a tool, so the stop token tells which."},
                {"zh": "思考：system turn 開頭放 `<|think|>`，推理寫在 `<|channel>thought` 到 `<channel|>` 之間。",
                 "en": "Thinking: `<|think|>` at the start of the system turn, and the reasoning between `<|channel>thought` and `<channel|>`."},
            ),),
            "qwen3_5": ((
                {"zh": "回合：ChatML 的 `<|im_start|>user` … `<|im_end|>`。", "en": "Turns: ChatML's `<|im_start|>user` … `<|im_end|>`."},
                {"zh": "工具呼叫：XML：`<tool_call>` 裡一個 `<function=名稱>`，每個參數一段 `<parameter=名稱>`，值一律是純文字。",
                 "en": "Tool calls: XML, one `<function=name>` inside `<tool_call>`, then a `<parameter=name>` for each argument, every value plain text."},
                {"zh": "工具結果：放進一則新的 `user` 訊息（`<tool_response>` … `</tool_response>`），再開一個新的 assistant turn。",
                 "en": "Tool results: in a new `user` message (`<tool_response>` … `</tool_response>`), followed by a new assistant turn."},
                {"zh": "停止 token：回答完和要用工具都停在 `<|im_end|>`，要看輸出裡有沒有工具呼叫才知道是哪一種。",
                 "en": "Stop tokens: `<|im_end|>` after an answer and for a tool alike, and only a tool call in the output tells them apart."},
                {"zh": "思考：chat template 在 assistant 訊息開頭先寫好 `<think>`，推理寫到 `</think>`。",
                 "en": "Thinking: the chat template opens the assistant message with `<think>`, and the reasoning runs until `</think>`."},
            ),),
        }),
        {"zh": "這些差異全部寫進 `Family`，其他程式只問 family，沒有任何 `if 是 Gemma` 的分支：",
         "en": "All of these differences go into a `Family`, and the rest of the code just asks the family; there is no `if Gemma` branch anywhere:"},
        Code("agent/model_families/__init__.py", "class Family:", end="def special_re(self)",
             caption={"zh": "欄位分三組：讀模型輸出用的、把對話寫回 chat template 用的、畫面用的。最後幾個有預設值，只有某些家族需要，"
                            "例如 Qwen 在 `processor_kwargs` 裡給 processor 的影片設定；`file_notes` 和 `drafter_file_notes` 說明模型資料夾裡的每個檔案（{{files}}）。",
                      "en": "The fields come in three groups: for reading the model's output, for writing the conversation back for the chat template, and for the page. "
                            "The last few have defaults and only some families need them, such as the video setting Qwen gives its processor in `processor_kwargs`; "
                            "`file_notes` and `drafter_file_notes` explain each file in the model folder ({{files}})."}),
        {"zh": "載入模型時，用 checkpoint 的 `config.json` 裡的 `model_type` 找出 family。停止 token 是 generation config 列的，再加上 family 宣告的："
               "Qwen3.5 的 checkpoint 只列了 `<|endoftext|>`，沒有列每個回合結尾的 `<|im_end|>`，少了 family 這一層，模型回答完會繼續寫下去。"
               "Qwen 的 `<think>`、`<tool_call>` 在 tokenizer 裡也沒有標成 special，同樣由 family 補上。"
               "tokenizer 放在模型的 processor 裡；processor 也負責把圖片、聲音和影片變成模型的輸入，一個模型讀得懂哪幾種由 checkpoint 決定（{{media}}）。",
         "en": "When a model loads, the `model_type` in the checkpoint's `config.json` finds its family. The stop tokens are those the generation "
               "config lists plus those the family declares: the Qwen3.5 checkpoint lists only `<|endoftext|>`, not the `<|im_end|>` that ends "
               "every turn, so without the family the model would keep writing after its answer. Qwen's tokenizer does not mark `<think>` "
               "and `<tool_call>` as special either, and the family adds those too. The tokenizer comes inside the model's processor, which "
               "also turns images, sound and video into the model's inputs; which of those a model reads, the checkpoint says ({{media}})."},
        Code("agent/model_backends/simple_backend.py", "self.family = model_families.for_model_type(", end="self.special_ids = set("),
        {"zh": "要加模型時：同一個家族的另一個 checkpoint（例如更大的 Gemma 4），在 `MODELS` 加一行（[[agent/model_families/__init__.py|MODELS = {]]），"
               "`REVISIONS` 再加一行它要下載的 commit；有 MTP 草稿模型的話，`DRAFTERS` 和草稿模型的 commit 也要加，`VLLM_SPECULATIVE` 寫 vLLM 在 GPU 上怎麼替它推測解碼。"
               "新的家族，照 `agent/model_families/qwen3_5.py` 寫一個模組，內容是一個輸出的 parser 和特殊 token 的說明（中英文各一份），"
               "加進 `FAMILIES`（[[agent/model_families/__init__.py|FAMILIES = {]]），`model_type` 對得上就會自動選到。",
         "en": "To add a model: another checkpoint of a family that is already here (a bigger Gemma 4, say) is one line in `MODELS` "
               "([[agent/model_families/__init__.py|MODELS = {]]) and one in `REVISIONS`, the commit it downloads at; with an MTP drafter, also `DRAFTERS` "
               "and the drafter's commit, and `VLLM_SPECULATIVE` for how vLLM drafts for it on a GPU. A new family is a module like `agent/model_families/qwen3_5.py`, with a parser for the output and "
               "notes on the special tokens (in Chinese and in English), added to `FAMILIES` ([[agent/model_families/__init__.py|FAMILIES = {]]); "
               "a matching `model_type` picks it automatically."},
        Code("agent/model_families/qwen3_5.py", "QWEN3_5 = Family(",
             caption={"zh": "Qwen3.5 的 family 幾乎都是資料，只有 `parse` 是程式。", "en": "Qwen3.5's family is almost all data; only `parse` is code."}),
    )),
    Section("files", "model",
            {"zh": "模型資料夾：models/ 裡的每個檔案", "en": "The model folder: every file in models/"},
            {"zh": "模型的檔案放在 `models/` 裡，每個模型一個資料夾：transformers 從這裡載入整個模型，vLLM 也從這裡讀 tokenizer 和 chat template，在 Docker 裡連權重都從這裡讀。"
                   "兩個模型和 Gemma 的草稿模型分別說明。",
             "en": "A model's files are in a folder of its own in `models/`: transformers loads the whole model from it, vLLM reads the tokenizer and the chat template there too, "
                   "and in Docker the weights as well. Here is each file, for both models and Gemma's drafter."}, (
        {"zh": "不管用什麼跑模型，下載都是 `snapshot_download()` 下載到 `MODELS_DIR`（[[agent/model_backends/simple_backend.py|MODELS_DIR = Path(]]）：預設是專案裡的 `models/`，git 不會追蹤它；"
               "設定 `AGENT_LOOP_MODELS` 就換到別的地方（`HF_HUB_CACHE` 不行，因為它常是整台電腦所有專案共用的設定）。下載的是 `REVISIONS` 固定的 commit（[[agent/model_families/__init__.py|REVISIONS = {]]），不是當下的 `main`，"
               "所以不管什麼時候下載，拿到的都是這一頁寫作和測試時用的檔案。已經下載過的檔案不會再抓，連不上 Hub 時直接用資料夾裡現有的。",
         "en": "Whatever runs the model, a download is `snapshot_download()` into `MODELS_DIR` ([[agent/model_backends/simple_backend.py|MODELS_DIR = Path(]]): by default the project's `models/`, "
               "which git ignores; `AGENT_LOOP_MODELS` moves it elsewhere (`HF_HUB_CACHE` does not, since it is often set once for every project on a machine). It downloads the commit pinned in `REVISIONS` "
               "([[agent/model_families/__init__.py|REVISIONS = {]]), not whatever `main` is by then, so whenever it runs, it gets the files this page was "
               "written and tested against. Files already there are not fetched again, and when the Hub cannot be reached, what the folder holds is used."},
        {"zh": "`models/` 裡是 Hugging Face 的快取格式：", "en": "Inside, `models/` is laid out as Hugging Face's cache:"},
        (
            {"zh": "`models--google--gemma-4-E2B-it/`：每個 repository 一個資料夾，id 裡的 `/` 換成 `--`。",
             "en": "`models--google--gemma-4-E2B-it/`: one folder per repository, with the `/` in its id turned into `--`."},
            {"zh": "`refs/main`：下載時 `main` 指向的 revision（一個 commit hash），只有照 `main` 下載時才會寫，例如沒加 `--revision` 的 `hf download`；"
                   "app 照 `REVISIONS` 的 commit 下載，用不到它。",
             "en": "`refs/main`: the revision (a commit hash) that `main` pointed to, written only by a download that asks for `main`, such as "
                   "`hf download` without `--revision`; the app downloads the commit in `REVISIONS` and does not need it."},
            {"zh": "`snapshots/<revision>/`：照 repository 原樣排好的檔案，`from_pretrained` 拿到的就是這個資料夾。裡面每個檔案都是連結（Windows 不能建連結時是複製）。",
             "en": "`snapshots/<revision>/`: the files as the repository has them; this is the folder `from_pretrained` is given. "
                   "Each file in it is a link (a copy on Windows when links cannot be made)."},
            {"zh": "`blobs/`：檔案本身，用 hash 命名，兩個 revision 共用的檔案只存一份。Hub 用 Xet 存的檔案（權重和 `tokenizer.json`）放在 `models/` 最上層的 `blobs/`，"
                   "不同的 repository 也共用，模型自己的 `blobs/` 再連過去。",
             "en": "`blobs/`: the files themselves, named by their hash, so a file two revisions share is stored once. Files the Hub stores with Xet "
                   "(the weights and `tokenizer.json`) sit in the `blobs/` at the top of `models/`, shared across repositories too, and the model's "
                   "own `blobs/` links there."},
            {"zh": "`trees/<revision>.json`：那個 revision 有哪些檔案的清單，存下來備用。",
             "en": "`trees/<revision>.json`: a cached list of that revision's files."},
            {"zh": "`CACHEDIR.TAG`：標明這是快取資料夾，備份工具可以跳過。",
             "en": "`CACHEDIR.TAG`: marks the folder as a cache, so backup tools can skip it."},
        ),
        {"zh": "下載哪些檔案、權重由誰讀，看是用什麼跑模型：", "en": "Which files are downloaded, and what reads the weights, depends on what runs the model:"},
        Tabs({
            "transformers": (
                {"zh": "`SimpleBackend` 用 `_download()` 下載整個 checkpoint，權重也在內，全部在這台電腦上載入。它先只下載 `config.json`，看出模型屬於哪個 family，"
                       "不支援的模型就不會白白下載好幾 GB 的權重（{{families}}）。MTP 的草稿模型也是這樣下載，第一次打開 MTP 時才抓（{{mtp}}）。",
                 "en": "`SimpleBackend` has `_download()` fetch the whole checkpoint, the weights included, and loads all of it on this machine. It fetches `config.json` "
                       "alone first, to tell the model's family from it, so a model the page does not support fails before gigabytes of weights come down ({{families}}). "
                       "The MTP drafter downloads the same way, the first time MTP is switched on ({{mtp}})."},
                Code("agent/model_backends/simple_backend.py", "def _download("),
            ),
            "vllm-docker": (
                {"zh": "啟動伺服器之前，`vllm_docker.start()` 先把整個 checkpoint 下載到 `models/`，權重也在內；GPU 上開 MTP 時，連 Gemma 4 的草稿模型也一起下載"
                       "（Qwen3.5 的 MTP 層就在它自己的權重裡）。伺服器自己什麼都不下載：",
                 "en": "Before it starts the server, `vllm_docker.start()` downloads the whole checkpoint into `models/`, the weights included, and with MTP on a GPU "
                       "Gemma 4's drafter too (Qwen3.5's MTP layer is in its own weights). The server itself downloads nothing:"},
                Code("agent/model_backends/vllm_docker.py", 'say("download", "")', end="snapshot_download(drafter"),
                {"zh": "`docker-compose/compose.yaml` 把這個資料夾唯讀掛進容器的 `/models`，Hugging Face 的快取指到那裡，而且不連網路。"
                       "`_env()` 把完整路徑的 `MODELS_DIR` 當成 `AGENT_LOOP_MODELS` 交給 Docker Compose（[[agent/model_backends/vllm_docker.py|\"AGENT_LOOP_MODELS\": str(MODELS_DIR)]]），"
                       "所以 `models/` 換了地方，掛上的也是那裡。vLLM 編譯模型的結果（第一次啟動最久的那一段）存在另一個 Docker volume，不放在 `models/`：",
                 "en": "`docker-compose/compose.yaml` mounts that folder in the container at `/models`, read-only, and points Hugging Face's cache there, with no network. "
                       "`_env()` hands Docker Compose the full path of `MODELS_DIR` as `AGENT_LOOP_MODELS` ([[agent/model_backends/vllm_docker.py|\"AGENT_LOOP_MODELS\": str(MODELS_DIR)]]), "
                       "so a `models/` moved elsewhere is the one mounted. What vLLM compiles for the model, the first start's longest wait, goes in a Docker volume of its own, "
                       "not in `models/`:"},
                Code("docker-compose/compose.yaml", "- ${AGENT_LOOP_MODELS:-../models}:/models:ro", end='HF_HUB_OFFLINE: "1"'),
                {"zh": "vLLM 拿到的是模型的 id 和 `REVISIONS` 裡的 commit（`--revision`），在快取裡就是 `models--<org>--<name>/snapshots/<revision>/` 這個資料夾，"
                       "和 transformers 載入的是同一份檔案。快取裡的連結都是相對路徑，掛到 `/models` 底下一樣指得到 `blobs/`。不連網路時，vLLM 會用資料夾的路徑稱呼這個模型，"
                       "所以再用 `--served-model-name` 把名字改回網頁用的 id：",
                 "en": "vLLM is given the model's id and the commit in `REVISIONS` (`--revision`), which in the cache is the folder `models--<org>--<name>/snapshots/<revision>/`, "
                       "the very files transformers would load. The cache's links are relative, so under `/models` they still lead to `blobs/`. Offline, vLLM would call the "
                       "model by that folder's path, so `--served-model-name` gives it back the id the page asks for:"},
                Code("docker-compose/compose.yaml", "- ${VLLM_MODEL:-Qwen/Qwen3.5-0.8B}", end="--served-model-name="),
                {"zh": "網頁這邊讀的也是同一個資料夾：`VLLMBackend` 的 `_checkpoint_files()` 只要權重以外的檔案，而它們早就在裡面了，所以什麼都不必再下載。"
                       "磁碟上只有一份：伺服器從它載入權重，網頁從它讀 tokenizer 和 chat template，把 prompt 變成 token id（{{backends}}）。",
                 "en": "The page reads that same folder: `VLLMBackend`'s `_checkpoint_files()` wants every file but the weights, which are all there already, so nothing more "
                       "downloads. There is one copy on disk: the server loads the weights from it, and the page reads the tokenizer and the chat template there to turn "
                       "the prompt into token ids ({{backends}})."},
            ),
            "vllm-url": (
                {"zh": "`VLLM_URL` 的伺服器自己有一份模型，在它跑的那台電腦上，所以網頁只把權重以外的檔案下載到 `models/`"
                       "（`WEIGHTS` 列的那幾種不抓，[[agent/model_backends/vllm_backend.py|WEIGHTS = []]），用的是同一個 commit：",
                 "en": "A server at `VLLM_URL` has a copy of the model of its own, on the machine it runs on, so the page downloads every file but the weights into `models/` "
                       "(none of the kinds `WEIGHTS` lists, [[agent/model_backends/vllm_backend.py|WEIGHTS = []]), at the same commit:"},
                Code("agent/model_backends/vllm_backend.py", "def _checkpoint_files("),
                {"zh": "兩邊要讀到同樣的 tokenizer 和 chat template，伺服器就要用 `REVISIONS` 的 commit 啟動，例如 `vllm serve Qwen/Qwen3.5-0.8B --revision 2fc0636…`。"
                       "網頁照伺服器啟動時的名字（`/v1/models` 的 `root`）去下載，所以從伺服器自己的資料夾啟動的（`vllm serve /data/gemma`），"
                       "要這台電腦的同一個路徑也有同一個資料夾，網頁才讀得到，而且就地讀、不下載；否則網頁會說它為什麼載入不了。",
                 "en": "For the two to read the same tokenizer and chat template, start the server at the commit in `REVISIONS`, as in "
                       "`vllm serve Qwen/Qwen3.5-0.8B --revision 2fc0636…`. The page downloads by the name the server was started with (the `root` in its `/v1/models`), "
                       "so a server started from a folder of its own (`vllm serve /data/gemma`) works only when the same folder is at the same path on this machine, "
                       "which the page then reads where it is, downloading nothing; otherwise the page says why it cannot load."},
                {"zh": "在那之前，模型選單只下載伺服器上每個模型的 `config.json`，用它的 `model_type` 看出是哪個 family、網頁支不支援"
                       "（[[agent/model_backends/vllm_backend.py|def family_of(]]）；其他檔案等選了那個模型才下載。",
                 "en": "Before that, the model menu downloads only the `config.json` of each model the server serves, whose `model_type` tells its family, and whether the "
                       "page supports it ([[agent/model_backends/vllm_backend.py|def family_of(]]); the other files come once that model is picked."},
            ),
        }),
        {"zh": "載入時，每個檔案由誰讀：", "en": "When the model loads, which code reads which file:"},
        Tabs({
            "transformers": (
                {"zh": "同一個 revision 在 macOS、Linux 加 CUDA 或只有 CPU 的電腦上，都是一模一樣的檔案。不同的是載入的方式：`pick_device()` 選裝置（[[agent/model_backends/simple_backend.py|def pick_device()]]），"
                       "權重在 GPU 上照檔案的 bfloat16 載入，在 CPU 上轉成 float32，記憶體是兩倍。",
                 "en": "A revision is exactly the same files on macOS, on Linux with CUDA or on a computer with only a CPU. What differs is how they load: "
                       "`pick_device()` picks the device ([[agent/model_backends/simple_backend.py|def pick_device()]]), and the weights load in the files' bfloat16 on a GPU but become "
                       "float32 on a CPU, twice the memory."},
                (
                    {"zh": "`config.json`：`AutoConfig` 先讀它，`model_type` 決定 family（[[agent/model_backends/simple_backend.py|self.family = model_families.for_model_type(]]），模型分頁上方的規格也來自它。",
                     "en": "`config.json`: `AutoConfig` reads it first, and its `model_type` picks the family ([[agent/model_backends/simple_backend.py|self.family = model_families.for_model_type(]]); "
                           "the specs at the top of the Model tab come from it too."},
                    {"zh": "`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja` 和 processor 的設定檔：`AutoProcessor` 把它們讀成一個 processor"
                           "（[[agent/model_backends/simple_backend.py|self.processor = AutoProcessor.from_pretrained(files)]]）；chat template 由 `render()` 執行（[[agent/model_backends/simple_backend.py|def render(self, messages]]）。",
                     "en": "`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and the processor's settings: `AutoProcessor` reads them into one "
                           "processor ([[agent/model_backends/simple_backend.py|self.processor = AutoProcessor.from_pretrained(files)]]); `render()` runs the chat template ([[agent/model_backends/simple_backend.py|def render(self, messages]])."},
                    {"zh": "權重：`AutoModelForMultimodalLM` 照 `config.json` 建好模型，再把 `.safetensors` 裡的 tensor 依名稱填進去"
                           "（[[agent/model_backends/simple_backend.py|self.model = AutoModelForMultimodalLM.from_pretrained(]]）。",
                     "en": "The weights: `AutoModelForMultimodalLM` builds the model from `config.json` and fills it with the tensors of the `.safetensors` file "
                           "by name ([[agent/model_backends/simple_backend.py|self.model = AutoModelForMultimodalLM.from_pretrained(]])."},
                    {"zh": "`generation_config.json`：停止 token（[[agent/model_backends/simple_backend.py|eos = self.model.generation_config.eos_token_id]]）和建議的抽樣設定（[[agent/model_backends/simple_backend.py|def _sampling_defaults(self)]]）。",
                     "en": "`generation_config.json`: the stop tokens ([[agent/model_backends/simple_backend.py|eos = self.model.generation_config.eos_token_id]]) and the recommended sampling "
                           "settings ([[agent/model_backends/simple_backend.py|def _sampling_defaults(self)]])."},
                    {"zh": "`README.md`、`LICENSE`、`.gitattributes`：給人看的模型卡、授權和 Hub 上 git 的設定，app 不讀。",
                     "en": "`README.md`, `LICENSE` and `.gitattributes`: the model card, the license and the settings of the Hub's git, for people; the app does not read them."},
                ),
            ),
            "vllm": (
                {"zh": "`VLLMBackend` 讀的是同一份檔案、用同樣的方式讀，只有權重不在這裡讀：",
                 "en": "`VLLMBackend` reads the same files the same way, all but the weights:"},
                (
                    {"zh": "`config.json`：`AutoConfig` 讀它，`model_type` 決定 family（[[agent/model_backends/vllm_backend.py|self.family = model_families.for_model_type(config.model_type)]]），"
                           "模型分頁上方的規格也來自它；脈絡上限則是伺服器說的（`--max-model-len`，[[agent/model_backends/vllm_backend.py|self.context_length = card.get(]]）。",
                     "en": "`config.json`: `AutoConfig` reads it, and its `model_type` picks the family ([[agent/model_backends/vllm_backend.py|self.family = model_families.for_model_type(config.model_type)]]); "
                           "the specs at the top of the Model tab come from it too, but the context limit is the server's (`--max-model-len`, "
                           "[[agent/model_backends/vllm_backend.py|self.context_length = card.get(]])."},
                    {"zh": "`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja` 和 processor 的設定檔：`AutoProcessor` 讀成一個 processor，和 `SimpleBackend` 一樣"
                           "（[[agent/model_backends/vllm_backend.py|self.processor = AutoProcessor.from_pretrained(files)]]），所以 chat template 在這裡執行，不在伺服器上（{{context}}）。"
                           "伺服器也從它那一份（在 Docker 裡就是這個資料夾）載入 tokenizer 和 processor，用來展開媒體（{{media}}）。",
                     "en": "`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and the processor's settings: `AutoProcessor` reads them into one processor, as "
                           "`SimpleBackend` does ([[agent/model_backends/vllm_backend.py|self.processor = AutoProcessor.from_pretrained(files)]]), so the chat template runs here, "
                           "not on the server ({{context}}). The server loads a tokenizer and a processor from its copy too (in Docker, this very folder), to expand the media ({{media}})."},
                    {"zh": "權重：伺服器在它的裝置上、用它的精度載入，網頁從來不載入，所以模型分頁沒有裝置、精度和參數量（{{backends}}）。",
                     "en": "The weights: the server loads them, on its device and in its precision, and the page never does, so the Model tab has no device, precision or "
                           "parameter count ({{backends}})."},
                    {"zh": "`generation_config.json`：網頁用它的停止 token 和建議的抽樣設定（[[agent/model_backends/vllm_backend.py|def _generation_config(]]）。"
                           "伺服器也讀它，當成自己的抽樣預設值，所以每個請求都把它們設成不起作用（{{sampling}}）。",
                     "en": "`generation_config.json`: the page takes the stop tokens and the recommended sampling settings from it ([[agent/model_backends/vllm_backend.py|def _generation_config(]]). "
                           "The server reads it too, for sampling defaults of its own, which every request sets to do nothing ({{sampling}})."},
                    {"zh": "`README.md`、`LICENSE`、`.gitattributes`：網頁和伺服器都不讀。",
                     "en": "`README.md`, `LICENSE` and `.gitattributes`: neither the page nor the server reads them."},
                ),
            ),
        }),
        {"zh": "每個模型資料夾裡的檔案：", "en": "The files in each model's folder:"},
        Tabs({
            "gemma4": (
                {"zh": "**Gemma 4 E2B**（`google/gemma-4-E2B-it`，9 個檔案，約 10.3 GB）：",
                 "en": "**Gemma 4 E2B** (`google/gemma-4-E2B-it`, 9 files, about 10.3 GB):"},
                (
                    {"zh": "`config.json`（5 kB）：`Gemma4ForConditionalGeneration`，分三部分。`text_config` 是 35 層的語言模型：28 層滑動視窗（只看最後 512 個 token）、"
                           "7 層 full attention，隱藏層維度 1536，8 個注意力頭共用 1 組 key / value；最後 20 層（`num_kv_shared_layers`）不自己算 key / value，"
                           "用前面的層算好的。`vision_config` 是 16 層的視覺編碼器，一張圖最多變成 280 個 token；`audio_config` 是 12 層的音訊編碼器。脈絡上限 131,072 個 token。",
                     "en": "`config.json` (5 kB): `Gemma4ForConditionalGeneration`, in three parts. `text_config` is the 35-layer language model: 28 sliding window "
                           "layers, which see only the last 512 tokens, and 7 full attention layers, hidden size 1536, 8 attention heads sharing 1 set of keys and "
                           "values; the last 20 layers (`num_kv_shared_layers`) compute no keys and values of their own but use those of earlier layers. "
                           "`vision_config` is a 16-layer vision encoder that makes an image at most 280 tokens, and `audio_config` a 12-layer audio encoder. "
                           "The context limit is 131,072 tokens."},
                    {"zh": "`generation_config.json`（208 B）：停止 token `[1, 106, 50]`，也就是 `<eos>`、`<turn|>`、`<|tool_response>`；建議 T 1.0、top-p 0.95、top-k 64。",
                     "en": "`generation_config.json` (208 B): the stop tokens `[1, 106, 50]`, that is `<eos>`, `<turn|>` and `<|tool_response>`; recommended "
                           "T 1.0, top-p 0.95, top-k 64."},
                    {"zh": "`model.safetensors`（10.2 GB）：2,011 個 tensor、5.12B 個參數，全部 bfloat16。最大的是 Per-Layer Embeddings（`embed_tokens_per_layer`，2.35B，占 46%）："
                           "35 層各替 262,144 個 token 存一個 256 維的向量，只查表、不做矩陣乘法，所以名字裡算的是有效參數（E2B）。"
                           "其餘是 35 層本身 1.88B、輸入的 embedding 0.40B、音訊編碼器 0.30B、視覺編碼器 0.17B。",
                     "en": "`model.safetensors` (10.2 GB): 2,011 tensors, 5.12B parameters, all bfloat16. The largest part is the Per-Layer Embeddings "
                           "(`embed_tokens_per_layer`, 2.35B, 46%): each of the 35 layers keeps a 256-wide vector for each of the 262,144 tokens, only looked up "
                           "and never multiplied, which is why the name counts effective parameters (E2B). The rest is the 35 layers themselves at 1.88B, "
                           "the input embedding at 0.40B, the audio encoder at 0.30B and the vision encoder at 0.17B."},
                    {"zh": "`tokenizer.json`（32 MB）：SentencePiece 風格的 BPE，262,144 個 token、514,906 條 merges；空白換成 `▁` 接在下一個詞前面，詞彙表沒有的字元拆成 UTF-8 byte。"
                           "`tokenizer_config.json`（3 kB）替控制 token 取角色名稱（`sot_token` 是 `<|turn>`、`stc_token` 是 `<|tool_call>` 等等）。",
                     "en": "`tokenizer.json` (32 MB): a SentencePiece-style BPE with 262,144 tokens and 514,906 merges; a space becomes `▁` in front of the next word, "
                           "and a character the vocabulary lacks splits into UTF-8 bytes. `tokenizer_config.json` (3 kB) gives the control tokens their roles "
                           "(`sot_token` is `<|turn>`, `stc_token` `<|tool_call>` and so on)."},
                    {"zh": "`chat_template.jinja`（19 kB）：寫出 `<|turn>system` / `user` / `model` 回合、思考的 `<|think|>`、工具的 `<|tool>declaration:…<tool|>`，以及附件的佔位 token（{{context}}）。",
                     "en": "`chat_template.jinja` (19 kB): writes the `<|turn>system` / `user` / `model` turns, the `<|think|>` for thinking, the tools as "
                           "`<|tool>declaration:…<tool|>` and the attachments' placeholders ({{context}})."},
                    {"zh": "`processor_config.json`（1.7 kB）：一個檔案管三種輸入。圖片切成 16 像素的 patch、每 3×3 合併，最多 280 個 token；聲音取樣 16 kHz、128 個 mel 頻帶，"
                           "每 40 ms 一個 token，最多 750 個（30 秒）；影片最多 32 格，每格最多 70 個 token（{{media}}）。",
                     "en": "`processor_config.json` (1.7 kB): one file for all three inputs. An image is cut into 16-pixel patches pooled 3×3, at most 280 tokens; "
                           "sound is sampled at 16 kHz into 128 mel bands, one token per 40 ms and at most 750 (30 seconds); a video takes at most 32 frames of "
                           "at most 70 tokens each ({{media}})."},
                ),
            ),
            "qwen3_5": (
                {"zh": "**Qwen3.5 0.8B**（`Qwen/Qwen3.5-0.8B`，13 個檔案，約 1.8 GB）：",
                 "en": "**Qwen3.5 0.8B** (`Qwen/Qwen3.5-0.8B`, 13 files, about 1.8 GB):"},
                (
                    {"zh": "`config.json`（2.9 kB）：`Qwen3_5ForConditionalGeneration`。`text_config` 是 24 層的語言模型，18 層 Gated DeltaNet（`linear_attention`）、"
                           "6 層 full attention（每 4 層一層），隱藏層維度 1024，8 個注意力頭配 2 組 key / value，脈絡上限 262,144；`vision_config` 是 12 層的視覺編碼器，"
                           "圖片和影片共用；沒有聲音的編碼器。`mtp_num_hidden_layers: 1` 說權重裡附了一層 MTP。",
                     "en": "`config.json` (2.9 kB): `Qwen3_5ForConditionalGeneration`. `text_config` is the 24-layer language model, 18 Gated DeltaNet layers "
                           "(`linear_attention`) and 6 full attention layers (one in every 4), hidden size 1024, 8 heads with 2 sets of keys and values, a context "
                           "limit of 262,144; `vision_config` is a 12-layer vision encoder that images and video share; there is no sound encoder. "
                           "`mtp_num_hidden_layers: 1` says the weights come with an MTP layer."},
                    {"zh": "沒有 `generation_config.json`：停止 token 只剩 `config.json` 的 `<|endoftext|>`，所以 family 補上 `<|im_end|>`；"
                           "建議的抽樣設定（T 1.0、top-p 1.0、top-k 20）也由 family 提供（[[agent/model_families/qwen3_5.py|sampling={]]）。",
                     "en": "No `generation_config.json`: the only stop token left is `config.json`'s `<|endoftext|>`, so the family adds `<|im_end|>`; the "
                           "recommended sampling settings (T 1.0, top-p 1.0, top-k 20) come from the family too ([[agent/model_families/qwen3_5.py|sampling={]])."},
                    {"zh": "`model.safetensors-00001-of-00001.safetensors`（1.7 GB）：只有一個分片，檔名照分片的格式取；`model.safetensors.index.json` 記下 488 個 tensor 各在哪個分片。"
                           "0.87B 個參數：24 層 0.50B、embedding 0.25B、視覺編碼器 0.10B，還有 20M 的 MTP 層（`mtp.`），transformers 載入時跳過。"
                           "大多是 bfloat16，Gated DeltaNet 層的 `A_log` 和 norm 以 float32 存。",
                     "en": "`model.safetensors-00001-of-00001.safetensors` (1.7 GB): a single shard, named in the sharded format, with "
                           "`model.safetensors.index.json` recording which shard each of the 488 tensors is in. 0.87B parameters: the 24 layers at 0.50B, the "
                           "embedding at 0.25B, the vision encoder at 0.10B, and a 20M MTP layer (`mtp.`) that transformers skips when loading. Most are "
                           "bfloat16; the Gated DeltaNet layers' `A_log` and norms are stored in float32."},
                    {"zh": "`tokenizer.json`（12.8 MB）：GPT-2 式的 byte-level BPE，248,044 個 token 加上 26 個 added tokens、247,587 條 merges，先做 NFC 正規化；每個 byte 寫成一個可印出的字元，空白是 `Ġ`。"
                           "`vocab.json`（6.7 MB）和 `merges.txt`（3.4 MB）是同一份詞彙和 merges 的舊格式。"
                           "`tokenizer_config.json`（16.7 kB）把 `<|im_end|>` 當 `eos_token`，裡面還有一份和 `chat_template.jinja` 一樣的 chat template。",
                     "en": "`tokenizer.json` (12.8 MB): a byte-level BPE as GPT-2's, 248,044 tokens plus 26 added ones and 247,587 merges after NFC normalization; every byte is "
                           "spelled as one printable character, a space as `Ġ`. `vocab.json` (6.7 MB) and `merges.txt` (3.4 MB) are the same vocabulary and "
                           "merges in the older format. `tokenizer_config.json` (16.7 kB) takes `<|im_end|>` as its `eos_token` and also holds a chat template "
                           "identical to `chat_template.jinja`."},
                    {"zh": "`chat_template.jinja`（7.8 kB）：ChatML 格式，工具 schema 放在 `<tools>…</tools>`，工具結果是一則 `user` 訊息，思考寫在 `<think>` 裡（{{context}}）。",
                     "en": "`chat_template.jinja` (7.8 kB): ChatML, with the tool schemas in `<tools>…</tools>`, a tool result as a `user` message and the "
                           "thinking inside `<think>` ({{context}})."},
                    {"zh": "`preprocessor_config.json` 和 `video_preprocessor_config.json`：圖片切成 16 像素的 patch，每 2×2 合成一個 token，所以一個 token 涵蓋 32×32 像素，"
                           "一張圖介於 65,536 和 16,777,216 像素之間；影片每 2 格畫面合成一組 token（{{media}}）。",
                     "en": "`preprocessor_config.json` and `video_preprocessor_config.json`: an image is cut into 16-pixel patches, 2×2 of them merged into one "
                           "token, so a token covers 32×32 pixels, and an image is kept between 65,536 and 16,777,216 pixels; in a video, every 2 frames make "
                           "one set of tokens ({{media}})."},
                ),
            ),
            "gemma4-drafter": (
                {"zh": "**Gemma 4 E2B 的 MTP 草稿模型**（`google/gemma-4-E2B-it-assistant`，7 個檔案，約 190 MB）：",
                 "en": "**Gemma 4 E2B's MTP drafter** (`google/gemma-4-E2B-it-assistant`, 7 files, about 190 MB):"},
                (
                    {"zh": "`config.json`：`Gemma4AssistantForCausalLM`，4 層（3 層滑動視窗、1 層 full attention），隱藏層維度 256，主模型是 1536（`backbone_hidden_size`）；"
                           "4 層都讀主模型的 key / value（`num_kv_shared_layers: 4`）。輸出用分群：262,144 個 token 分成 2,048 群、每群 128 個，先替各群打分，"
                           "只對最高的 32 群、共 4,096 個 token 算 logits（{{mtp}}）。",
                     "en": "`config.json`: `Gemma4AssistantForCausalLM`, 4 layers (3 sliding window, 1 full attention), hidden size 256 next to the model's 1536 "
                           "(`backbone_hidden_size`); all 4 read the model's keys and values (`num_kv_shared_layers: 4`). Its output is clustered: the 262,144 "
                           "tokens are sorted into 2,048 clusters of 128, the clusters are scored first, and logits are computed only for the 4,096 tokens of "
                           "the best 32 ({{mtp}})."},
                    {"zh": "`generation_config.json`：`num_assistant_tokens: 6`，每次猜 6 個 token（[[agent/model_backends/simple_backend.py|self.draft_tokens = self.drafter.generation_config.num_assistant_tokens]]）。",
                     "en": "`generation_config.json`: `num_assistant_tokens: 6`, six guesses at a time "
                           "([[agent/model_backends/simple_backend.py|self.draft_tokens = self.drafter.generation_config.num_assistant_tokens]])."},
                    {"zh": "`model.safetensors`（158 MB）：77.7M 個參數。67M 是 `embed_tokens`（262,144 個 token × 256 維），當作輸出的表；"
                           "`pre_projection` 把主模型的 token embedding 接上 hidden state（3,072 維）壓成 256 維，`post_projection` 再放大回 1,536；4 層本身只有約 9M。"
                           "`masked_embedding.token_ordering` 是 262,144 個整數，依群列出 token。",
                     "en": "`model.safetensors` (158 MB): 77.7M parameters. 67M of them are `embed_tokens` (262,144 tokens × 256), used as the output table; "
                           "`pre_projection` narrows the model's token embedding next to its hidden state (3,072 wide) to 256, and `post_projection` widens it "
                           "back to 1,536; the 4 layers themselves are only about 9M. `masked_embedding.token_ordering` is 262,144 integers listing the tokens "
                           "cluster by cluster."},
                    {"zh": "`tokenizer.json`、`tokenizer_config.json`：和主模型同一套詞彙，app 不載入，草稿模型讀寫的都是主模型的 token id。",
                     "en": "`tokenizer.json` and `tokenizer_config.json`: the same vocabulary as the model's; the app does not load them, since the drafter reads "
                           "and writes the model's token ids."},
                ),
            ),
        }),
        {"zh": "模型分頁最下面的「模型資料夾」列出目前載入的模型的這些檔案：`describe_folder()` 讀出每個檔案的大小，以及 family 的說明提到的欄位在檔案裡的值"
               "（[[agent/model_backends/simple_backend.py|def describe_folder(]]）。說明本身寫在 family 裡（Gemma：[[agent/model_families/gemma4.py|file_notes=FILES,]]；Qwen：[[agent/model_families/qwen3_5.py|file_notes=FILES,]]），"
               "所以換成同一家族的另一個 checkpoint，數字會跟著它的檔案變。模型在 vLLM 上跑時，列的是網頁讀的那個資料夾：權重不在裡面時（`VLLM_URL` 的伺服器），"
               "寫成「只在伺服器上」（[[app/render.py|elif served and kind == \"model.safetensors\":]]）。",
         "en": "The model folder card at the bottom of the Model tab lists these files for the model that is loaded: `describe_folder()` reads each "
               "file's size and the values of the fields the family's notes mention ([[agent/model_backends/simple_backend.py|def describe_folder(]]). The notes themselves are on the "
               "family (Gemma: [[agent/model_families/gemma4.py|file_notes=FILES,]]; Qwen: [[agent/model_families/qwen3_5.py|file_notes=FILES,]]), so another checkpoint of "
               "the family shows its own numbers. With the model on vLLM, it lists the folder the page reads, and weights that are not in it (with a server at `VLLM_URL`) "
               "are on the server only ([[app/render.py|elif served and kind == \"model.safetensors\":]])."},
        Code("agent/model_families/__init__.py", "class FileNote:"),
    )),
    Section("submit", "ui",
            {"zh": "送出問題：載入模型、開一條 thread 跑 agent", "en": "Sending a question: load the model, start a thread for the agent"},
            {"zh": "輸入框和範例按鈕都走 `start_run()`：拿鎖、凍結設定、開 thread。",
             "en": "The input box and the example buttons both go through `start_run()`: take the lock, freeze the settings, start a thread."}, (
        {"zh": "畫面每次重跑，先由 `current_llm()` 照側欄選的模型拿到一個 `Backend`，畫面的其他地方只把它當成 `Backend` 用（{{backends}}）。"
               "`load_llm()` 用 `st.cache_resource(max_entries=1)` 快取，記憶體裡同時只留一個模型。換模型或換「用什麼跑模型」時，"
               "`release_model()` 先清掉快取、把記憶體還給系統，下一個模型才載入，兩個不會同時佔著記憶體。Streamlit 的 fast reruns 也關掉了（`.streamlit/config.toml`）：重跑會等正在跑的那一次到下一個 Streamlit 呼叫，而不是在新的 thread 裡和它同時跑，所以模型載入到一半時切換的設定，要等載入完才生效，不會在放掉之後才又載入完、留在記憶體裡。"
               "第一次用到某個模型時，把它建議的 temperature、top-p、top-k 寫進側欄；Qwen3.5 的 checkpoint 沒附 generation_config.json，建議值來自 family"
               "（[[agent/model_backends/simple_backend.py|def _sampling_defaults(self)]]）。側欄的模型選單換了之後會開新對話（[[app/page.py|key=\"model_id\", on_change=switch_model]]）。",
         "en": "On every rerun, `current_llm()` first gets a `Backend` for the model picked in the sidebar, and the rest of the page takes it for just that ({{backends}}). "
               "`load_llm()` is cached with `st.cache_resource(max_entries=1)`, so only one model stays in memory. Switching models, or what runs them, first has "
               "`release_model()` clear that cache and hand the memory back, and only then does the next model load, so the two never hold memory at once. "
               "Streamlit's fast reruns are off too (`.streamlit/config.toml`): a rerun waits for the run in progress to reach its next Streamlit call rather than running beside it in a new thread, so a switch made while a model loads takes effect once it has loaded, and the model does not finish loading after it was let go. "
               "The first time a model is used, its recommended temperature, top-p and top-k go into the sidebar; the Qwen3.5 checkpoint ships no "
               "generation_config.json, so its recommendations come from the family ([[agent/model_backends/simple_backend.py|def _sampling_defaults(self)]]). "
               "Switching models in the sidebar's menu starts a new conversation ([[app/page.py|key=\"model_id\", on_change=switch_model]])."},
        Code("app/page.py", "@st.cache_resource(max_entries=1", end="def release_model():"),
        Tabs({
            "simple": (
                {"zh": "內建 harness 跑的就是這個模型：側欄最上面選模型，再選用什麼跑它。",
                 "en": "The built-in harness runs this very model: the sidebar picks the model at its top, then what runs it."},
            ),
            "pi": (
                {"zh": "選了 pi 時，`pi_picker()` 先啟動一次 pi，問它有哪些模型（[[agent/harnesses/pi_harness.py|def probe()]]），最多每分鐘問一次。"
                       "「pi 問的模型」選這個網頁的模型時，接著照內建 harness 的方式選模型、載入它；選 pi 自己的模型時，列出 pi 有登入資訊的那些，這台電腦什麼都不載入。",
                 "en": "With pi picked, `pi_picker()` first starts pi once to ask which models it has ([[agent/harnesses/pi_harness.py|def probe()]]), at most once a minute. "
                       "With This page's model picked under The model pi asks, it then picks and loads the model as the built-in harness does; with pi's own models, it "
                       "lists those pi has credentials for, and nothing loads on this machine."},
                Code("app/page.py", 'if st.session_state.pi_source == "page":', end="llm=current_llm())"),
            ),
        }),
        {"zh": "載入這個網頁的模型時，其他的看是用什麼跑它（內建 harness 和 pi 問這個網頁的模型時都一樣）：",
         "en": "Loading the page's model, the rest depends on what runs it (for the built-in harness and for pi on this page's model alike):"},
        Tabs({
            "transformers": (
                {"zh": "`load_llm()` 建一個 `SimpleBackend`，把模型載入這台電腦（{{files}}）。剛從 vLLM (Docker) 換回 transformers 時，`current_llm()` 先停掉 vLLM 的容器，再載入模型。"
                       "在 CUDA 上，記憶體還給系統之前要先放掉 cuBLAS 的工作區：它從第一次矩陣乘法起就一直留著，而且是切在 transformers 替權重預留的那一大塊記憶體裡，"
                       "不放掉的話，那一整塊都還不回去（[[agent/model_backends/simple_backend.py|torch._C._cuda_clearCublasWorkspaces()]]）。"
                       "換模型時 MTP 回到關閉；打開「MTP 推測解碼」時，`current_llm()` 第一次會順便載入草稿模型（[[agent/model_backends/simple_backend.py|def load_drafter(self):]]）。",
                 "en": "`load_llm()` makes a `SimpleBackend`, which loads the model onto this machine ({{files}}). Right after transformers is picked again over vLLM (Docker), "
                       "`current_llm()` first stops vLLM's container, and only then loads the model. On CUDA, cuBLAS's workspaces go before the memory is handed back: "
                       "cuBLAS keeps one from the first matmul on, carved out of the segment transformers reserved for the weights, and while it lives none of that segment "
                       "can go back ([[agent/model_backends/simple_backend.py|torch._C._cuda_clearCublasWorkspaces()]]). Switching models turns MTP back off; the first time "
                       "MTP speculative decoding is switched on, `current_llm()` also loads the drafter ([[agent/model_backends/simple_backend.py|def load_drafter(self):]])."},
            ),
            "vllm-docker": (
                {"zh": "`current_llm()` 先確定 Docker 裡的 vLLM 伺服器已經啟動，沒有就啟動它，拿到它的網址（{{backends}}）；`load_llm()` 再為它建一個 `VLLMBackend`。"
                       "「用什麼跑模型」也和記憶體裡的模型一樣，是所有分頁共用的一個選擇：另一個分頁啟動或停掉了 vLLM 的容器，這個分頁就跟著換，不會在旁邊再載入一份模型（[[app/page.py|if not st.session_state.get(\"engine_picked\"):]]）；"
                       "新開的分頁或重新整理，也會選容器裡跑的那個模型（[[app/page.py|st.session_state.model_id = on_vllm[\"model\"] if on_vllm else default_model()]]）。"
                       "MTP 開或關也跟著伺服器，不會用另一種方式把它重新啟動；換模型時 MTP 也維持原樣，因為那是伺服器啟動的方式（{{mtp}}）。"
                       "只有剛選過的分頁例外，直到 `current_llm()` 啟動或停掉 vLLM 為止。",
                 "en": "`current_llm()` first makes sure the vLLM server in Docker is up, starting it if it is not, and takes its URL ({{backends}}); `load_llm()` then makes a "
                       "`VLLMBackend` for it. What runs the model is, like the model in memory, one choice for every tab: when another tab has started or stopped vLLM's container, this one "
                       "follows rather than load the model beside it ([[app/page.py|if not st.session_state.get(\"engine_picked\"):]]), and a new tab or a reload "
                       "also takes the model the container runs ([[app/page.py|st.session_state.model_id = on_vllm[\"model\"] if on_vllm else default_model()]]). "
                       "MTP follows the server too, on or off, rather than start it again the other way, and switching models leaves MTP as it is, since it is how the "
                       "server was started ({{mtp}}). Only the tab that just picked leads, until `current_llm()` has started or stopped vLLM."},
            ),
            "vllm-url": (
                {"zh": "`load_llm()` 為 `VLLM_URL` 的伺服器建一個 `VLLMBackend`。模型選單列的是那台伺服器提供的模型，側欄沒有「用什麼跑模型」可以選，"
                       "網頁也不會啟動或停掉那台伺服器；MTP 是伺服器的事，側欄的開關不能按（{{mtp}}）。",
                 "en": "`load_llm()` makes a `VLLMBackend` for the server at `VLLM_URL`. The model menu lists the models that server serves, the sidebar has no "
                       "What runs the model to pick, and the page never starts or stops the server; MTP is the server's, and the sidebar's switch is disabled ({{mtp}})."},
            ),
        }),
        Code("app/page.py", "def current_llm() -> Backend:"),
        {"zh": "輸入框（[[app/page.py|sent = st.chat_input(]]）和範例按鈕都會呼叫 `start_run()`。輸入框可以附上檔案，模型讀得懂聲音時還能錄音；"
               "有些範例也會附上 `app/assets/examples/` 裡的範例檔，輸入框上方也列出這些檔案，選了的會附在你自己打的問題裡。`start_run()` 先用不等待的方式拿一把全域的鎖"
               "（[[app/page.py|if not slot.lock.acquire(blocking=False):]]）："
               "模型只有一份，所以不管開幾個分頁，同一時間只能有一個 agent 在跑，拿不到鎖就跳出提示。拿到鎖之後，才把檔案存進這段對話的資料夾（{{media}}）。",
         "en": "The input box ([[app/page.py|sent = st.chat_input(]]) and the example buttons both call `start_run()`. Files can be attached "
               "in the input box, and a model that hears can be sent a recording; some examples attach a sample file from `app/assets/examples/`, and the same files are listed above the input box, to go with a question you type yourself. "
               "`start_run()` first tries to take a global lock without waiting ([[app/page.py|if not slot.lock.acquire(blocking=False):]]): "
               "there is one copy of the model, so however many tabs are open, only one agent can run at a time, and without the lock the tab shows a notice. "
               "Only with the lock does it save the files in the conversation's folder ({{media}})."},
        Code("app/page.py", "def start_run(",
             caption={"zh": "鎖是 `generation_slot()` 裡的 `threading.Lock`，用 `st.cache_resource` 讓所有分頁共用。",
                      "en": "The lock is the `threading.Lock` in `generation_slot()`, shared by every tab through `st.cache_resource`."}),
        {"zh": "`current_settings()` 把側欄的值打包成 `RunSettings`，解碼參數（包括要不要用 MTP）包在裡面的 `Sampling`。這份設定這一輪從頭用到尾，"
               "執行中改側欄只會影響下一輪。它也記下頁面現在的語言：harness 和工具回給模型的錯誤，這一輪都用這個語言寫。"
               "`tools` 是這一輪提供給模型的工具：打開「讓模型寫程式」而且 Docker 準備好了，才有 `run_python`（{{execute}}）。",
         "en": "`current_settings()` packs the sidebar's values into `RunSettings`, with the decoding parameters (MTP included) in its `Sampling`. "
               "These settings hold for the whole turn; changing the sidebar during a run only affects the next one. It also records the page's "
               "language: the errors that the harness and the tools send the model are worded in it for the whole turn. `tools` are the tools "
               "the model is offered: `run_python` only when Let the model write code is on and Docker is ready ({{execute}})."},
        Code("app/page.py", "def current_settings("),
        {"zh": "`harness_for()` 決定這一輪由誰跑：`SimpleHarness` 跑在記憶體裡的模型上，或是 `PiHarness`；pi 問這個網頁的模型時，"
               "`PiHarness` 也拿到那個模型和側欄的解碼設定（`PageModel`）。兩種都帶著這段對話的歷史、附件和資料夾，設定在這時候就固定下來。",
         "en": "`harness_for()` decides what runs the turn: a `SimpleHarness` on the model in memory, or a `PiHarness`, which, when pi asks this page's model, "
               "gets that model and the sidebar's decoding too (`PageModel`). Either gets the conversation so far, the attachments and the conversation's folder, "
               "and its settings are fixed now."},
        Code("app/page.py", "def harness_for("),
        {"zh": "`LiveRun` 拿 `self.events.append` 當成 `emit` 建出這個 harness，然後開背景 thread。"
               "thread 跑完 `run.run()` 之後才在 `finally` 裡放開鎖，所以就算 agent 出錯，鎖也一定會還回去。",
         "en": "`LiveRun` makes that harness with `self.events.append` as its `emit`, and starts the background thread. "
               "The thread releases the lock in a `finally` only after `run.run()` returns, so the lock always comes back, even when the agent fails."},
        Code("app/page.py", "class LiveRun:",
             caption={"zh": "`daemon=True`：關掉 server 時不用等 agent 跑完。", "en": "`daemon=True`: stopping the server does not wait for the agent to finish."}),
    )),
    Section("prepare", "harness",
            {"zh": "準備一輪：訊息、工具、system prompt", "en": "Preparing a run: messages, tools, system prompt"},
            {"zh": "harness 準備這一輪模型看得到的東西：對話、工具和 system prompt。",
             "en": "The harness gathers what the model will see this turn: the conversation, the tools and the system prompt."}, (
        Tabs({
            "simple": (
            {"zh": "`SimpleHarness.__init__` 把之前的對話歷史接上這次的問題，存成 `self.messages`（[[agent/harnesses/simple_harness.py|self.messages = [*history,]]）。"
                   "之後的工具呼叫和工具結果都 append 在後面；模型本身不記得任何事，每一步看到的都是這份 list 重新 render 出來的字串。"
                   "訊息用的是 Hugging Face 的通用格式（`role`、`content`、`tool_calls`），兩個家族都吃得下，格式差異交給各自的 chat template；"
                   "附了檔案時，這則訊息的 `content` 是一串 part（{{media}}）。`self.tools` 是這一輪模型拿得到的工具，"
                   "`self.files` 是這段對話的資料夾。`_stop` 和 `_decided` 是 UI thread 用來通知 agent 的兩個同步工具。",
             "en": "`SimpleHarness.__init__` appends this question to the conversation so far and keeps it as `self.messages` ([[agent/harnesses/simple_harness.py|self.messages = [*history,]]). "
                   "Tool calls and tool results are appended after it; the model itself remembers nothing, and what it sees in each step is this "
                   "list rendered into a string again. The messages use Hugging Face's common format (`role`, `content`, `tool_calls`), which "
                   "both families accept; each family's chat template takes care of its own format. With files attached, that message's `content` "
                   "is a list of parts ({{media}}). `self.tools` are the tools the model gets this turn, and `self.files` is the conversation's folder. "
                   "`_stop` and `_decided` are the two synchronization tools the UI thread signals the agent with."},
            Code("agent/harnesses/simple_harness.py", "def __init__(self, llm: Backend, history"),
            {"zh": "`run()` 一開始拿出這個模型的 family，決定這一輪模型看得到什麼：收集 `self.tools` 裡每個工具的 `schema()` 給 chat template"
                   "（關掉「使用工具」時一個都沒有），"
                   "另外留一份「工具名稱 → 參數 schema」給 parser（Qwen 的參數值是純文字，要靠 schema 才知道型別）；"
                   "system prompt 也分有工具（[[agent/harnesses/simple_harness.py|SYSTEM_WITH_TOOLS = ]]）和沒工具兩個版本。"
                   "第一個事件 `run_start` 帶著模型和 family，畫面就知道這一輪要用哪個模型的格式來畫；它也說這一輪是哪個 harness 跑的（`simple`，或 pi 的 `pi`）。",
             "en": "`run()` starts by taking the model's family and deciding what the model gets to see this turn: it collects the `schema()` "
                   "of every tool in `self.tools` for the chat template (none with Use tools off), and keeps a map from tool name to parameters schema for the parser (Qwen's "
                   "argument values are plain text, and only the schema says their types); the system prompt comes in two versions, with tools "
                   "([[agent/harnesses/simple_harness.py|SYSTEM_WITH_TOOLS = ]]) and without. The first event, `run_start`, carries the model and the family, so the page "
                   "knows whose format to draw the turn in; it also says which harness ran the turn (`simple`, or pi's `pi`)."},
            Code("agent/harnesses/simple_harness.py", "def run(self):", end='stop_reason, final = "max_steps", ""',
                 caption={"zh": "`stop_reason` 的預設值是 `max_steps`：迴圈跑完所有步數都沒有 break，停止原因就是步數上限。",
                          "en": "`stop_reason` starts out as `max_steps`: when the loop uses up its steps without a break, the step limit is why it stopped."}),
            {"zh": "工具的 schema 就是一份 JSON Schema，包成 Hugging Face chat template 認得的 function 格式：",
             "en": "A tool's schema is a JSON Schema, wrapped in the function format that Hugging Face chat templates know:"},
            Code("agent/tools.py", "def schema(self) -> dict:"),
            ),
            "pi": (
                {"zh": "`PiHarness.run()` 啟動 pi，先選好模型和思考等級、關掉 pi 自己的自動重試（重試會變成另一個 step，所以請求失敗就當成錯誤顯示），再把問題當成 prompt 送出；"
                       "每一個指令都等到 pi 回應才送下一個，因為 pi 是同時處理指令的，和 `set_model` 一起送出的 prompt 會送到原本的模型。",
                 "en": "`PiHarness.run()` starts pi, picks the model and the thinking level and turns pi's own retries off (a retry would be a step of its own, so a "
                       "failed request is shown as an error instead), then sends the question as a prompt, each command only after pi has answered the last, since pi "
                       "handles commands concurrently and a prompt sent beside `set_model` would go to the old model."},
                Code("agent/harnesses/pi_harness.py", "self._pi = Process(self._args()", end="self._loop()"),
                {"zh": "模型看得到什麼是 pi 的事：它自己的 system prompt、它的工具（側欄選模型看得到哪幾個），和它存在這段對話資料夾裡的 session；下一個問題用同一個 session 啟動 pi，"
                       "模型就讀得到前面的全部。網頁啟動 pi 時不載入你自己的 extension、skill、prompt 範本和 AGENTS.md（[[agent/harnesses/pi_harness.py|FLAGS = ]]），畫出來的就是 pi 本身的迴圈；"
                       "唯一多載入的是網頁的 extension `pi_extension.ts`（{{context}}、{{check}}）。問題附上的檔案存在 pi 的工作資料夾裡，問題寫出它們的名字，讀得懂圖片的模型也直接拿到圖片"
                       "（[[agent/harnesses/pi_harness.py|def _message(self)]]）。`run_start` 和內建 harness 的一樣，pi 問這個網頁的模型時帶著它的 family，問 pi 自己的模型時沒有。",
                 "en": "What the model gets to see is pi's: its own system prompt, its tools (of which the sidebar picks the ones the model sees) and the session it keeps in "
                       "the conversation's folder, on which the next question starts pi again, so the model reads all of it. The page starts pi without your own extensions, "
                       "skills, prompt templates and AGENTS.md ([[agent/harnesses/pi_harness.py|FLAGS = ]]), so what it draws is pi's own loop; the one thing it adds is its "
                       "extension, `pi_extension.ts` ({{context}}, {{check}}). Files attached to the question are saved in pi's working folder and named in the question, and a "
                       "model that reads images gets them too ([[agent/harnesses/pi_harness.py|def _message(self)]]). `run_start` is as the built-in harness's; it carries the "
                       "family when pi asks this page's model, and none when pi asks one of its own."},
            ),
        }),
    )),
    Section("context", "context",
            {"zh": "脈絡：把整段對話 render 成一個字串", "en": "Context: rendering the whole conversation into one string"},
            {"zh": "chat template 把 system、工具、對話接成一長串；模型這一步能看到的就只有它。",
             "en": "The chat template joins the system prompt, the tools and the conversation into one long string; it is all the model can see in this step."}, (
        Tabs({
            "simple": (
                {"zh": "每一圈從 `for step in range(1, s.max_steps + 1)` 開始，側欄的「步數上限」就是這個 range 的上界。第一件事是 render：",
                 "en": "Each step starts at `for step in range(1, s.max_steps + 1)`; the sidebar's Max steps is the top of that range. The first thing is to render:"},
                Code("agent/harnesses/simple_harness.py", "for step in range(1, s.max_steps + 1):", end="roles=_role_counts([system, *self.messages])",
                     caption={"zh": "`step_start` 讓畫面開一張新的 step 卡片；`context` 帶著模型實際讀到的完整字串送出。",
                              "en": "`step_start` has the page open a new step card; `context` goes out with the whole string the model really reads."}),
            ),
            "pi-page": (
                {"zh": "pi 開始一個 turn，把請求送給 provider `agent-loop`，也就是 `model_endpoint.py`：pi 的 system prompt、工具和對話，寫成 OpenAI 的 JSON。"
                       "和問 pi 自己的模型時一樣，extension 先回報這個請求，「脈絡」分頁把它收在脈絡下面。端點把它換回 `SimpleHarness` 用的那種對話"
                       "（[[agent/harnesses/model_endpoint.py|def conversation(]]）：工具呼叫的參數變回 dict，思考放進 family 的欄位，工具結果寫上它的工具名稱；"
                       "再和 `SimpleHarness` 一樣 render、encode，送出 `context`。這一步新加入的部分，從模型上一則訊息為止的對話算起（[[agent/harnesses/model_endpoint.py|def seen_before(]]）。",
                 "en": "pi starts a turn and sends its request to the provider `agent-loop`, which is `model_endpoint.py`: pi's system prompt, its tools and the "
                       "conversation, as OpenAI's JSON. As on pi's own models, the extension reports the request first, and the Context tab keeps it under the context. "
                       "The endpoint turns it back into the conversation `SimpleHarness` keeps ([[agent/harnesses/model_endpoint.py|def conversation(]]): a tool call's "
                       "arguments a dict again, the thinking under the family's key, a tool result with its tool's name. Then, as `SimpleHarness` does, it renders and "
                       "encodes it and sends `context`; what is new in the step is counted from the conversation as of the model's last message "
                       "([[agent/harnesses/model_endpoint.py|def seen_before(]])."},
                Code("agent/harnesses/model_endpoint.py", "messages = conversation(body, llm, serving.folder)", end='emit("context",'),
            ),
            "pi-provider": (
                {"zh": "網頁啟動的每個 pi 都載入一個小 extension，`pi_extension.ts`。pi 每次要向模型送出請求前，它把請求的內容當成一則通知送出；"
                       "在 RPC 模式下，通知是 stdout 上的一筆 `extension_ui_request`，`PiHarness` 把它變成 `request` 事件（[[agent/harnesses/pi_harness.py|def _request(self, payload)]]）："
                       "step 卡片的脈絡段和「脈絡」分頁畫出 pi 的 system prompt、工具和對話，寫成 JSON，這一步新加的部分（上一步的回覆和觀察）另外標出來。"
                       "provider 的 API 怎麼把這個請求寫成模型讀的字串，pi 和 provider 都不說，所以這一段下面講的 chat template，在這裡看不到。",
                 "en": "Every pi the page starts loads one small extension, `pi_extension.ts`. Before each request pi sends the model, it sends what the request holds as "
                       "a notification; in RPC mode a notification is an `extension_ui_request` on stdout, which `PiHarness` turns into a `request` event "
                       "([[agent/harnesses/pi_harness.py|def _request(self, payload)]]): the step card's context section and the Context tab show pi's system prompt, its "
                       "tools and the conversation as JSON, with what this step added (the last step's reply and observation) marked. How the provider's API writes the "
                       "request out as the string its model reads, neither pi nor the provider says, so the chat template the rest of this section is about is not to be seen here."},
                Code("agent/harnesses/pi_extension.ts", 'pi.on("before_provider_request"', end="});"),
            ),
        }),
        {"zh": "`render()` 呼叫 tokenizer 自帶的 chat template，模型在 vLLM 上跑時、pi 問這個網頁的模型時也都是在這裡 render（{{files}}）。格式是模型自己的，harness 不用知道：",
         "en": "`render()` calls the chat template that comes with the tokenizer, here even when the model runs on vLLM, and when pi asks this page's model ({{files}}). "
               "The format is the model's own, and the harness does not need to know it:"},
        Tabs({
            "gemma4": (
                {"zh": "`<bos><|turn>system` 開頭；開了思考模式，system turn 最前面多一個 `<|think|>`；"
                       "每個工具寫成 `<|tool>declaration:名稱{…}<tool|>`；每則訊息包在 `<|turn>角色` … `<turn|>` 裡。",
                 "en": "It starts with `<bos><|turn>system`; with thinking mode on, a `<|think|>` comes first in the system turn; "
                       "each tool is written as `<|tool>declaration:name{…}<tool|>`; each message is wrapped in `<|turn>role` … `<turn|>`."},
                {"zh": "`generation_prompt=True`（`add_generation_prompt`）在最後寫上 `<|turn>model`，打開模型的回合，請模型接著寫。",
                 "en": "`generation_prompt=True` (`add_generation_prompt`) ends it with `<|turn>model`, opening the model's turn for it to write next."},
            ),
            "qwen3_5": (
                {"zh": "沒有 `<bos>`，`<|im_start|>system` 開頭；工具 schema 原樣以 JSON 放進 `<tools>` … `</tools>`，"
                       "後面附上 XML 呼叫格式的說明；每則訊息包在 `<|im_start|>角色` … `<|im_end|>` 裡。",
                 "en": "No `<bos>`; it starts with `<|im_start|>system`; the tool schemas go in as JSON, as they are, between `<tools>` and "
                       "`</tools>`, followed by how to write a call in XML; each message is wrapped in `<|im_start|>role` … `<|im_end|>`."},
                {"zh": "`generation_prompt=True`（`add_generation_prompt`）在最後寫上 `<|im_start|>assistant`，打開模型的回合，請模型接著寫；"
                       "後面再加上 `<think>`（開思考模式）或一段空的 `<think></think>`（不開）。",
                 "en": "`generation_prompt=True` (`add_generation_prompt`) ends it with `<|im_start|>assistant`, opening the model's turn for it to write next, "
                       "followed by `<think>` (thinking mode on) or an empty `<think></think>` (off)."},
            ),
        }),
        Code("agent/model_backends/simple_backend.py", "def render(self, messages"),
        {"zh": "接著 `encode()` 把字串變成模型真正讀的 token（[[agent/harnesses/simple_harness.py|encoded = self.llm.encode(]]）。只有文字時，就是 tokenizer 切出來的 token；"
               "對話裡有圖片、聲音或影片時，chat template 在它們的位置只寫了一個佔位 token，`encode()` 把它展開成那個檔案佔的幾十到幾千個 token"
               "（{{media}}），所以 `context` 送出的是展開後的字串。`n_tokens` 是它的長度，畫面拿它和脈絡上限比，顯示用掉幾 %。"
               "`added_from` 是這一步新加入的內容從哪裡開始：先算 render 出來的字串和 `prev_sequence` 共同開頭的長度（{{record}}會說 `prev_sequence` 是什麼），"
               "再加上媒體展開多出來的長度：媒體都在使用者訊息裡，在新加入的內容前面。第 2 步以後，新加入的正好是工具結果。",
         "en": "Next `encode()` turns the string into the tokens the model really reads ([[agent/harnesses/simple_harness.py|encoded = self.llm.encode(]]). With text "
               "alone, they are the tokenizer's tokens; when the conversation has images, sound or video, the chat template wrote a single "
               "placeholder token where each goes, and `encode()` expands it into the tens to thousands of tokens that file takes ({{media}}), "
               "so `context` goes out with the expanded string. `n_tokens` is its length, which the page compares with the context limit to "
               "show how much of it is used. `added_from` is where what is new in this step begins: the length of the start that the rendered "
               "string shares with `prev_sequence` ({{record}} says what `prev_sequence` is), plus however much longer the media made it, since "
               "the media are all in user messages, before what is new. From step 2 on, what is new is exactly the tool result."},
        Code("agent/harnesses/simple_harness.py", "def common_prefix(",
             caption={"zh": "逐字元比較，回傳兩個字串共同開頭的長度。", "en": "Compares character by character and returns the length of the start the two strings share."}),
        {"zh": "畫面上的「脈絡」分頁把 `context` 事件裡的字串原樣印出來（[[app/render.py|def context_html(]]），後面接著模型這一步寫的輸出；"
               "`added_from` 之後的部分標成「這一步新加入」（第 1 步沒有上一步，從最新一則使用者訊息開始標），family 列出的特殊 token 換成可以滑過看說明的深色標籤；"
               "一長串同一個特殊 token，例如一張圖片展開成的幾百個 `<|image|>`，畫成一個標籤加上個數。pi 跑的 step 還有它送出的請求："
               "問 pi 自己的模型時畫在脈絡的位置，問這個網頁的模型時收在脈絡下面（[[app/render.py|def request_html(]]）。",
         "en": "The page's Context tab prints the string in the `context` event as it is ([[app/render.py|def context_html(]]), followed by what "
               "the model writes in this step; it marks everything after `added_from` as new in this step (in step 1, which has no step before it, "
               "from the newest user message on), and turns the special tokens the family lists into dark chips you can hover over for a note; "
               "a long run of one special token, like the hundreds of `<|image|>` an image expands into, is one chip with the count. A step pi ran also has the request "
               "it sent, in place of the context on pi's own models, and folded under it on this page's model ([[app/render.py|def request_html(]])."},
    )),
    Section("media", "context",
            {"zh": "圖片、聲音和影片：變成脈絡裡的 token", "en": "Images, sound and video: tokens in the context"},
            {"zh": "附上的檔案存進對話的資料夾；模型讀得懂的，由 processor 展開成一串 token，prefill 時再換成編碼器的輸出。",
             "en": "Attached files go into the conversation's folder; those the model reads are expanded by its processor into a run of tokens, "
                   "which the prefill replaces with the encoders' output."}, (
        {"zh": "一個模型讀得懂哪些東西，是載入時從 checkpoint 看出來的：config 裡要有那種編碼器，processor 也要有對應的部分。"
               "Gemma 4 E2B 有視覺和音訊編碼器，看得懂圖片、聽得懂聲音，也看得懂影片（一張張畫格）；Qwen3.5 0.8B 只有視覺編碼器，看得懂圖片和影片。"
               "輸入框只收模型讀得懂的檔案，打開「讓模型寫程式」時再加上 CSV、Excel 這類資料檔。",
         "en": "Which things a model reads is found out from the checkpoint when it loads: the config needs that kind of encoder, and the processor "
               "the part that feeds it. Gemma 4 E2B has a vision and an audio encoder, so it sees images, hears sound and watches video (frame by "
               "frame); Qwen3.5 0.8B has only a vision encoder and reads images and video. The input box takes only files the model reads, plus "
               "data files such as CSV and Excel once Let the model write code is on."},
        Tabs({
            "transformers": (
                Code("agent/model_backends/simple_backend.py", "def _modalities(self)"),
            ),
            "vllm": (
                {"zh": "在 vLLM 伺服器上，是這些再扣掉伺服器不收的：載入時，backend 對 checkpoint 讀得懂的每一種媒體，各送一個只生成一個 token 的請求問伺服器，"
                       "模型讀得懂的就是伺服器收的那幾種（[[agent/model_backends/vllm_backend.py|self.modalities = tuple(kind for kind in readable if kind not in refused)]]）。只有 `/inference/v1/generate` 收媒體，所以沒加 `--enable-scale-out` 的伺服器一種都不收。"
                       "不收的那幾種是給 `run_python` 的一般檔案；模型分頁列出伺服器不收哪幾種、為什麼（[[app/render.py|def refused_html(]]），輸入框上方也說哪些不能附上。",
                 "en": "On a vLLM server, it is those less what the server does not take: at load, the backend asks the server with a one-token request for each kind of "
                       "media the checkpoint reads, and the model reads those it takes ([[agent/model_backends/vllm_backend.py|self.modalities = tuple(kind for kind in readable if kind not in refused)]]). Only `/inference/v1/generate` takes media, so a server started without "
                       "`--enable-scale-out` takes none. A kind it refuses is a plain file for `run_python`; the Model tab lists the kinds the server refused and why "
                       "([[app/render.py|def refused_html(]]), and above the input box the page says which cannot be attached."},
                Code("agent/model_backends/vllm_backend.py", "def _probe("),
            ),
        }),
        {"zh": "`start_run()` 把每個檔案存進這段對話的資料夾，名字只留一般字元，因為模型可能要在程式裡打出它（[[agent/media.py|def save(]]）。"
               "使用者訊息裡，模型讀得懂的檔案各是一個 part，排在文字前面；其他檔案模型看不到內容，只在文字後面多一段說明："
               "`run_python` 在 `/data` 底下找得到它們，CSV、TSV、純文字和 Markdown 檔再附上第一行（CSV 的話就是欄位名稱；JSON 不附，它的第一行多半只是 `[`）。少了這一行，小模型寫程式時會亂猜欄位，"
               "例如不存在的 `Date`，然後出錯；多給幾列資料也不行，Gemma 會把那幾列直接抄進程式，只算它們，不去讀檔案。"
               "這段說明是寫給模型看的，所以和 system prompt 一樣用英文。",
         "en": "`start_run()` saves each file in the conversation's folder, keeping its name to plain characters, since the model may have to "
               "type it into code ([[agent/media.py|def save(]]). In the user's message, each file the model reads is a part of its own, before the "
               "text; the model cannot see into other files, which get a note after the text: `run_python` finds them under `/data`, and a CSV, "
               "TSV, plain-text or Markdown file comes with its first line (for a CSV, the column names; not a JSON file, whose first line is mostly a `[`). Without it a small model guesses the columns "
               "in its code, a `Date` that is not there, say, and fails; a few rows of data are too many, as Gemma copies them into its code and "
               "works out the answer from them alone instead of reading the file. The note is for the model, so it is in English, like the system prompt."},
        Code("agent/media.py", "def user_content("),
        {"zh": "chat template 在每個 part 的位置寫一個佔位 token，模型的 processor 再把每個佔位 token 展開成那個檔案佔的 token：",
         "en": "The chat template writes a placeholder token where each part goes, and the model's processor expands each placeholder into the tokens that file takes:"},
        Tabs({
            "gemma4": (
                {"zh": "佔位 token 是 `<|image|>`、`<|audio|>`、`<|video|>`。一張圖片最多 280 個 `<|image|>`，包在 `<|image>` 和 `<image|>` 之間；"
                       "影片平均取 32 張畫格（比這短的影片全部都用），每張前面寫上時間 `mm:ss`，最多 70 個 `<|video|>`；"
                       "聲音每 40 ms 一個 `<|audio|>`，最多 750 個，所以只聽得到前 30 秒。",
                 "en": "The placeholders are `<|image|>`, `<|audio|>` and `<|video|>`. An image is up to 280 `<|image|>` between `<|image>` and `<image|>`; "
                       "a video is 32 frames taken evenly (all of them, for a shorter one), each after its time as `mm:ss` and up to 70 `<|video|>`; "
                       "sound is one `<|audio|>` per 40 ms, at most 750, so only the first 30 seconds are heard."},
            ),
            "qwen3_5": (
                {"zh": "佔位 token 是圖片的 `<|vision_start|><|image_pad|><|vision_end|>` 和影片的 `<|video_pad|>`。圖片越大 token 越多"
                       "（640×480 約 300 個 `<|image_pad|>`）；影片每秒取 2 張畫格，每 2 張一組，前面寫上時間，例如 `<0.3 seconds>`。",
                 "en": "The placeholders are `<|vision_start|><|image_pad|><|vision_end|>` for an image and `<|video_pad|>` for a video. The bigger the image, "
                       "the more tokens (about 300 `<|image_pad|>` for 640×480); a video gives 2 frames a second, in pairs, each pair after its time, "
                       "such as `<0.3 seconds>`."},
            ),
        }),
        {"zh": "誰來展開、編碼器在哪裡跑，看是用什麼跑模型：", "en": "What expands them, and where the encoders run, depends on what runs the model:"},
        Tabs({
            "transformers": (
                {"zh": "`encode()` 把對話裡所有媒體依出現的順序交給 processor，由它展開：", "en": "`encode()` hands the processor all the media in the conversation, in the order they come, to expand:"},
                Code("agent/model_backends/simple_backend.py", "def encode(self, prompt",
                     caption={"zh": "`media_tokens` 數每種媒體佔了幾個 token，step 卡片的脈絡段會寫出來。",
                              "en": "`media_tokens` counts how many tokens each kind of media takes; the step card's context section shows it."}),
                {"zh": "processor 要的是解碼好的資料，由 `media.py` 準備：圖片轉成 RGB，照片照 EXIF 轉正；聲音用 PyAV 解碼成單聲道，"
                       "取樣率改成模型的 feature extractor 要的 16 kHz；影片平均取最多 64 張畫格、縮小到長邊 768 像素，"
                       "再附上取完之後的畫格率，processor 寫出的時間才對（[[agent/media.py|def load_video(]]）。每個檔案只解碼一次，之後每一步都重複使用。",
                 "en": "The processor wants decoded data, which `media.py` prepares: images become RGB, and photos are turned upright by their EXIF; "
                       "sound is decoded with PyAV into mono at the 16 kHz the model's feature extractor expects; a video gives at most 64 frames taken "
                       "evenly, shrunk to 768 pixels on the longer side, with the frame rate of what was kept, so the times the processor writes are "
                       "right ([[agent/media.py|def load_video(]]). Each file is decoded once and reused in every step after."},
                {"zh": "prefill 時，編碼器的輸出換掉這些佔位 token 的 embedding。圖片和影片先跑一次視覺編碼器再交給 prefill，和 `generate()` 的做法一樣；"
                       "直接交給 forward 的話，transformers 5.20 的 Gemma 4 會漏掉影片的畫格數。Qwen3.5 的位置編碼（M-RoPE）要知道圖片的長寬，"
                       "所以除了像素以外的輸入都留著；它也會把上一個 prompt 的位置偏移記在模型上，每次 prefill 前要清掉。"
                       "prefill 之後，媒體都已經在 KV cache 裡，解碼和純文字完全一樣。`tests/test_real_model.py` 確認附上圖片、影片和錄音時，greedy 的輸出也和 `generate()` 一致。",
                 "en": "In the prefill, the encoders' output replaces the embeddings of those placeholder tokens. Images and videos go through the "
                       "vision encoder first and its output goes to the prefill, as `generate()` does it; handed to the forward pass as they are, "
                       "videos lose their frame counts in transformers 5.20's Gemma 4. Qwen3.5's position encoding (M-RoPE) needs the images' sizes, "
                       "so every input but the pixels stays; it also keeps the last prompt's position offset on the model, which is cleared before "
                       "every prefill. After the prefill, the media are in the KV cache, and decoding is the same as for text. "
                       "`tests/test_real_model.py` checks that with an image, a video and a recording, too, greedy output matches `generate()`."},
                Code("agent/model_backends/simple_backend.py", "def _encoded_media("),
            ),
            "vllm": (
                {"zh": "檔案跟著 chat template 寫的佔位 token 一起送到伺服器（圖片轉正存成 PNG、聲音解碼成 WAV、影片送原檔，"
                       "[[agent/model_backends/vllm_backend.py|def content_part(]]），由伺服器展開每個佔位 token，編碼器也在伺服器的 prefill 裡跑。"
                       "`encode()` 先用一個只生成一個 token 的請求問伺服器怎麼展開，所以脈絡分頁上是伺服器讀到的 prompt；生成時伺服器若展開得不一樣，就當成錯誤。",
                 "en": "The files go to the server next to the placeholders the chat template wrote (an image upright as PNG, sound decoded to WAV, a video as "
                       "the file, [[agent/model_backends/vllm_backend.py|def content_part(]]), the server expands each placeholder, and the encoders run in its prefill. "
                       "`encode()` asks the server how first, with a request that generates one token, so the Context tab shows the prompt as the server reads it; "
                       "a different expansion at generation time is an error."},
                Code("agent/model_backends/vllm_backend.py", "def encode(self, prompt"),
            ),
        }),
    )),
    Section("model", "model",
            {"zh": "模型：prefill，然後一次一個 token", "en": "Model: prefill, then one token at a time"},
            {"zh": "一次 forward 讀完整段 prompt，之後每生成一個 token 再 forward 一次。",
             "en": "One forward pass reads the whole prompt; after that, each new token takes another forward pass."}, (
        Tabs({
            "simple": (
                {"zh": "harness 用一個 for 迴圈消費 `generate()` 產生的事件：`prefill` 和每個 `token` 都加上 step 編號轉送出去，"
                       "最後的 `done` 留下來，改名成 `model_end` 送出。",
                 "en": "The harness consumes the events that `generate()` yields in a for loop: `prefill` and each `token` are passed on with the step "
                       "number, and the final `done` is kept and sent on as `model_end`."},
                Code("agent/harnesses/simple_harness.py", "done = None", end='self.emit("model_end"',
                     caption={"zh": "傳進去的 `self._stop.is_set` 是一個函式：模型每生成一個 token 前都會呼叫它，問「要停了嗎？」。",
                              "en": "The `self._stop.is_set` passed in is a function: the model calls it before every token to ask whether to stop."}),
            ),
            "pi-page": (
                {"zh": "端點用同樣的方式消費 `generate()` 的事件，每一個都送進 pi 的那一步。pi 先開始一個 turn 才送出請求，兩件事從不同的路到頁面，"
                       "所以請求先等頁面開好那一步（[[agent/harnesses/pi_harness.py|def _serving_step(self)]]）。思考和回覆的每個 token 也一邊生成一邊傳回 pi，"
                       "分別是 `reasoning_content` 和 `content`。模型每生成一個 token 前都問一次要不要停：這一輪被停止了，或是 pi 已經不收了（它被中止時）。",
                 "en": "The endpoint consumes `generate()`'s events in the same way, sending each into pi's step. pi starts a turn and then sends its request, and the two "
                       "reach the page by different ways, so the request first waits until the page has opened that step ([[agent/harnesses/pi_harness.py|def _serving_step(self)]]). "
                       "Each token of the thinking and of the reply also goes back to pi as it comes, as `reasoning_content` and `content`. Before every token the model asks "
                       "whether to stop: whether the turn is being stopped, or pi has stopped listening (as it does when it is aborted)."},
                Code("agent/harnesses/model_endpoint.py", "for ev in llm.generate(encoded, serving.sampling"),
            ),
            "pi-provider": (
                {"zh": "這裡沒有 backend：provider 的 API 把文字一段一段串流回 pi，pi 把每一段當成 `message_update` 紀錄送出，`PiHarness` 轉成 `delta`，帶著它是哪一種 channel"
                       "（[[agent/harnesses/pi_harness.py|def _update(self, e: dict)]]）；step 卡片照 channel 畫出這些文字，沒有 token 的分界（[[app/render.py|def streamed_html(]]）。"
                       "訊息結束時，`model_end` 帶著 provider 報的 token 數、第一段輸出花了多久，和照 pi 的價目估的費用（[[agent/harnesses/pi_harness.py|def _message_end(self, message: dict)]]）。"
                       "下面的頁籤講的是這個網頁的模型。",
                 "en": "No backend runs here: the provider's API streams text back to pi, pi sends each piece on as a `message_update` record, and `PiHarness` turns it into "
                       "`delta`, with its channel ([[agent/harnesses/pi_harness.py|def _update(self, e: dict)]]); the step card draws the pieces by channel, without token "
                       "boundaries ([[app/render.py|def streamed_html(]]). When the message ends, `model_end` carries the token counts the provider reported, how long the "
                       "first output took and the cost at pi's prices ([[agent/harnesses/pi_harness.py|def _message_end(self, message: dict)]]). The tabs below are about "
                       "this page's model."},
                Code("agent/harnesses/pi_harness.py", "def _update(self, e: dict):"),
            ),
        }),
        {"zh": "這個網頁的模型怎麼生成（內建 harness 和 pi 問這個網頁的模型時都一樣），看是用什麼跑它：",
         "en": "How the page's model generates (for the built-in harness and for pi on this page's model alike) depends on what runs it:"},
        Tabs({
            "transformers": (
                {"zh": "**prefill**：`encode()` 已經把整個 prompt 轉成 token id，一次 forward 算完；有媒體時，這次 forward 也把編碼器的輸出放進它們的位置（{{media}}）。"
                       "`use_cache=True` 留下每一層的狀態（一般 attention 層是 key / value，"
                       "Qwen3.5 的 Gated DeltaNet 層是一個固定大小的狀態），之後每個新 token 只需要算自己那一個位置；"
                       "`logits_to_keep=1` 只取最後一個位置的 logits，因為只需要「下一個 token」的分布（[[agent/model_backends/simple_backend.py|def _forward(]]）。"
                       "prefill 的時間主要看 prompt 有多長，模型分頁的表格裡有每一步的 prefill ms。",
                 "en": "**prefill**: the whole prompt, already token ids from `encode()`, goes through one forward pass, which also puts the encoders' "
                       "output in place of any media ({{media}}). `use_cache=True` keeps each layer's state "
                       "(the keys and values in ordinary attention layers, a fixed-size state in Qwen3.5's Gated DeltaNet layers), so each new token "
                       "needs only its own position computed; `logits_to_keep=1` returns the logits for the last position only, since only the "
                       "distribution of the next token is needed ([[agent/model_backends/simple_backend.py|def _forward(]]). Prefill time depends mostly on the prompt's length; "
                       "the table in the Model tab has each step's prefill ms."},
                Code("agent/model_backends/simple_backend.py", "def generate(self, prompt", end='yield {"type": "prefill"',
                     caption={"zh": "prompt 如果以 family 的 `thought_prefix` 結尾（Gemma 的 `<|channel>thought`、Qwen 的 `<think>`），模型一開始就在思考裡。",
                              "en": "If the prompt ends with the family's `thought_prefix` (`<|channel>thought` for Gemma, `<think>` for Qwen), the model starts out inside its thinking."}),
                {"zh": "**解碼迴圈**從 `tokens` 一個一個拿抽好的 token：平常由 `_one_at_a_time()` 產生，打開 MTP 時改由 `_drafted()`（{{mtp}}），"
                       "迴圈本身不用知道是哪一種。每一圈：檢查停止 → 拿到一個 token → "
                       "判斷它是不是特殊 token、有沒有切換 channel：哪些 token 切到思考、工具呼叫或回覆，由 family 的 `channel_switch` 決定"
                       "（Gemma：[[agent/model_families/gemma4.py|channel_switch={]]；Qwen：[[agent/model_families/qwen3_5.py|channel_switch={]]）→ "
                       "把目前為止的所有 token 整段 decode 再取差值，這樣被拆成好幾個 byte token 的中文字會等到湊齊才輸出 → 送出 `token` 事件。",
                 "en": "**The decoding loop** takes sampled tokens from `tokens` one at a time: they normally come from `_one_at_a_time()`, and from "
                       "`_drafted()` with MTP on ({{mtp}}); the loop itself does not need to know which. Each time around: check for stop → get a token → "
                       "work out whether it is a special token and whether it switches channel, where the family's `channel_switch` says which tokens "
                       "switch to thinking, a tool call or the reply (Gemma: [[agent/model_families/gemma4.py|channel_switch={]]; Qwen: "
                       "[[agent/model_families/qwen3_5.py|channel_switch={]]) → decode all the tokens so far and take the difference, so that a Chinese character "
                       "split across several byte tokens comes out only once it is whole → send a `token` event."},
                Code("agent/model_backends/simple_backend.py", "decode_start = time.perf_counter()", end="**details,"),
                {"zh": "生成到停止 token，或到了 `max_new_tokens`（側欄的「每步 token 上限」），就跳出迴圈。",
                 "en": "A stop token, or reaching `max_new_tokens` (the sidebar's Max tokens per step), ends the loop."},
                Code("agent/model_backends/simple_backend.py", "if token_id in self.eos_ids:", end="if len(generated) == sampling.max_new_tokens:"),
                {"zh": "還沒停，迴圈就再跟 `tokens` 要下一個。`_one_at_a_time()` 這時才把剛抽到的 token 餵回模型，帶著快取算出下一個位置的 logits；"
                       "它是一個 generator，迴圈跳出之後就不會再多跑一次 forward。",
                 "en": "If it has not stopped, the loop asks `tokens` for the next one. Only then does `_one_at_a_time()` feed the token it just sampled "
                       "back into the model, with the cache, for the next position's logits; it is a generator, so once the loop ends, no extra forward pass runs."},
                Code("agent/model_backends/simple_backend.py", "def _one_at_a_time("),
                {"zh": "最後的 `done` 告訴 harness 為什麼停（`finish`：`stop_token`、`max_new_tokens` 或 `user_stopped`）、停在哪個 token、"
                       "生成速度，以及整段原始輸出 `raw`（保留特殊 token，因為 harness 要靠它們解析）。",
                 "en": "The final `done` tells the harness why it stopped (`finish`: `stop_token`, `max_new_tokens` or `user_stopped`), at which token, "
                       "how fast it generated, and the whole raw output, `raw` (special tokens included, because the harness parses by them)."},
                Code("agent/model_backends/simple_backend.py", "decode_s = time.perf_counter() - decode_start", end='"raw": self.tokenizer.decode(generated'),
            ),
            "vllm": (
                {"zh": "`VLLMBackend.generate()` 送出一樣的事件，內容來自伺服器用 server-sent events 一段一段送回的 token（每個 token 帶了什麼，見{{sampling}}）。"
                       "伺服器第一次回 token 時送出 `prefill`，所以它的時間是等到第一個 token 的時間，包含網路：",
                 "en": "`VLLMBackend.generate()` yields the same events, from the tokens the server streams back as server-sent events (what each token comes with is in "
                       "{{sampling}}). `prefill` goes out once the first token is back, so its time is the time to the first token, the network included:"},
                Code("agent/model_backends/vllm_backend.py", "def generate(self, prompt", end='yield {"type": "prefill"'),
                {"zh": "vLLM 會留著和上一個 prompt 相同開頭的 KV cache（prefix caching），所以第 2 步開始只讀和上一步不一樣的部分："
                       "chat template 重新寫過的模型回合，加上新的觀察；`SimpleBackend` 每一步都從頭讀完整個 prompt。",
                 "en": "vLLM keeps the KV cache of what a prompt shares with the last one (prefix caching), so from step 2 on it reads only what differs from the last step: "
                       "the model's turn as the chat template rewrote it, and the new observation; `SimpleBackend` reads the whole prompt from the start every step."},
                {"zh": "拿到 token 之後和 `SimpleBackend` 一樣：family 的 `channel_switch` 決定切到哪個 channel，整段 decode 再取差值，送出 `token` 事件"
                       "（[[agent/model_backends/vllm_backend.py|# From here on, as in SimpleBackend.generate.]]）。什麼時候停由伺服器決定：請求帶著停止 token（`stop_token_ids`）"
                       "和 `max_tokens`，也就是側欄的「每步 token 上限」，超過伺服器的脈絡上限就再縮短（{{sampling}}）。按下停止時，網頁關掉這條串流，伺服器就不再生成"
                       "（[[agent/model_backends/vllm_backend.py|tokens.close()]]）。",
                 "en": "From the token on it is as in `SimpleBackend`: the family's `channel_switch` says which channel, decoding all the tokens so far gives the text, and a "
                       "`token` event goes out ([[agent/model_backends/vllm_backend.py|# From here on, as in SimpleBackend.generate.]]). The server decides when to stop: the request "
                       "carries the stop tokens (`stop_token_ids`) and `max_tokens`, the sidebar's Max tokens per step, cut to what fits in the server's context ({{sampling}}). "
                       "Pressing stop closes the stream, which tells the server to stop generating ([[agent/model_backends/vllm_backend.py|tokens.close()]])."},
                {"zh": "最後的 `done` 和 `SimpleBackend` 的一樣：伺服器的 `finish_reason` 是 `stop` 就是 `stop_token`、`length` 就是 `max_new_tokens`，其他都當成錯誤；"
                       "伺服器推測解碼時還帶著它的統計（{{mtp}}）。",
                 "en": "The final `done` is the same as `SimpleBackend`'s: the server's `finish_reason` `stop` becomes `stop_token` and `length` `max_new_tokens`, and anything "
                       "else is an error; when the server decodes speculatively, it also carries the server's counts ({{mtp}})."},
                Code("agent/model_backends/vllm_backend.py", "if finish is None:", end='"raw": self.tokenizer.decode(generated'),
                {"zh": "greedy 的輸出不一定和 `generate()` 逐 token 相同：vLLM 會把請求合成一批跑，也有自己的 kernel。和 `generate()` 完全一樣，是 `SimpleBackend` 才有的保證。",
                 "en": "Greedy output need not match `generate()` token for token: vLLM batches requests and has kernels of its own. Matching `generate()` exactly is "
                       "`SimpleBackend`'s promise only."},
            ),
        }),
    )),
    Section("sampling", "model",
            {"zh": "抽樣：每個 token 是怎麼選出來的", "en": "Sampling: how each token is picked"},
            {"zh": "temperature → top-k → top-p → 抽一個；同時記下模型原本的分布，給浮窗用。",
             "en": "temperature → top-k → top-p → draw one; and keep the model's own distribution for the popover."}, (
        Tabs({
            "transformers": (
                {"zh": "`_sample()` 每個 token 跑一次。先對原始 logits 做 softmax，得到模型自己的分布（T = 1、什麼都沒剃掉）："
                       "token 浮窗裡的「模型給的機率」、排名和前 5 名候選都來自這個分布。",
                 "en": "`_sample()` runs once per token. It first takes the softmax of the raw logits, the model's own distribution (T = 1, nothing cut): "
                       "the model probability, the rank and the top 5 candidates in a token's popover all come from it."},
                {"zh": "temperature ≤ 0 是 greedy，直接拿機率最高的；兩個 token 機率一樣時拿 id 小的，和 `generate()` 的 argmax 一樣"
                       "（bfloat16 的 logits 打平比想像中常見）。否則依序：logits ÷ T → 只留前 k 個（top-k，0 = 不限）→ "
                       "重新 softmax → 從機率最高的往下累加，累積機率達到 top-p 為止（nucleus，至少留一個）→ 重新正規化 → "
                       "`torch.multinomial` 抽一個。這段跟模型無關，兩個家族共用。",
                 "en": "A temperature of 0 or less is greedy: it takes the most likely token, and of two equally likely ones the lower id, as "
                       "`generate()`'s argmax does (in bfloat16, logits tie more often than one would think). Otherwise, in order: logits ÷ T → keep only the top k "
                       "(top-k; 0 = no limit) → softmax again → add up from the most likely down until the cumulative probability reaches top-p "
                       "(nucleus, keeping at least one) → renormalize → draw one with `torch.multinomial`. None of this depends on the model; both families share it."},
                Code("agent/model_backends/simple_backend.py", "def _sample(self, logits"),
            ),
            "vllm": (
                {"zh": "抽樣在伺服器上做。側欄的每個設定都照送，其他取樣設定都設成不起作用，免得伺服器從 checkpoint 帶來的預設值改變分布：",
                 "en": "The server does the sampling. Every sidebar setting is sent, and every other sampling setting is set to do nothing, so that defaults the server "
                       "took from the checkpoint cannot change the distribution:"},
                Code("agent/model_backends/vllm_backend.py", "def _sampling_params("),
                {"zh": "一般的 `vllm serve` 只有 `/v1/completions`：每個 token 帶著它的 log 機率，"
                       "以及那個位置機率最高的 5 個 token 和它們的 log 機率，key 寫成 `token_id:<id>`；排名只有在 token 落在前 5 名裡時才算得出來。"
                       "vLLM 預設回傳的是模型原本的分布（`--logprobs-mode raw_logprobs`），和 `SimpleBackend` 的 `p` 意思一樣。",
                 "en": "A plain `vllm serve` has only `/v1/completions`: each token comes with its log probability "
                       "and the 5 most likely tokens at its position with theirs, keyed `token_id:<id>`, so the rank is known only when the token is among those 5. "
                       "By default vLLM sends the model's own distribution (`--logprobs-mode raw_logprobs`), which means the same as `SimpleBackend`'s `p`."},
                Code("agent/model_backends/vllm_backend.py", "def _details("),
                {"zh": "可是伺服器也可能用別的模式啟動，而沒有哪條路會說是哪一種，所以載入時 backend 先問一個 token：temperature 1、top-k 2、"
                       "要最可能的 5 個（[[agent/model_backends/vllm_backend.py|def logprobs_mode(]]）。processed 模式下，被 top-k 剃掉的 3 個會回 -9999；"
                       "log 機率加起來最多是 1，logits 不會。`processed_logprobs` 是 token 被抽出來的那個分布，套過 temperature、top-k、top-p，"
                       "所以選中的 token 的就是取樣時的機率，送成 `p_sampled`，不送 `p` 和前 5 名；列出來的有幾個被剃掉時，也知道剩下幾個。logits 不是機率，就不當成機率送。"
                       "排名每種模式都一樣：temperature 不改順序，top-k、top-p 留下的是最可能的那幾個。",
                 "en": "Yet a server may run another mode, and no route says which, so at load the backend asks for one token at temperature 1 with top-k 2 and the 5 "
                       "most likely tokens ([[agent/model_backends/vllm_backend.py|def logprobs_mode(]]). In a processed mode, the 3 that top-k cut come back as -9999; log probabilities "
                       "add up to at most 1, and logits do not. `processed_logprobs` is the distribution the token was drawn from, after temperature, top-k and top-p, "
                       "so the chosen token's is its sampling probability, sent as `p_sampled`, and no `p` or top 5 is sent; when some of those listed were cut, how many "
                       "were left is known too. Logits are no probabilities, and none is sent as one. The rank is the same in every mode: temperature keeps the order, and top-k and top-p keep the most likely tokens."},
                Code("agent/model_backends/vllm_backend.py", "def _token_details("),
                {"zh": "伺服器加上 `--enable-scale-out` 時，還有一條 `/inference/v1/generate`：一樣送 token id 進去、token id 出來，但說得更多。"
                       "載入時 backend 用幾個只生成一個 token 的請求問伺服器能做什麼：有沒有這條路、有沒有送 sampling mask、收哪些媒體（{{media}}）。這條路帶來的：",
                 "en": "Started with `--enable-scale-out`, the server also has `/inference/v1/generate`: token ids in and token ids out as well, but it tells more. At load "
                       "the backend asks the server what it can do, with requests that generate one token each: whether it has that route, whether it sends sampling masks, "
                       "and which media it takes ({{media}}). What that route brings:"},
                (
                    {"zh": "**每個 token 的排名**，就算不在前 5 名也有。它的清單把選中的 token 放第一個，所以多要一個，再照排名排出前 5 名。"
                           "vLLM 的排名把和選中的 token 同分的也算在前面，推測解碼時比的還是 logits，列出來的 log 機率可能在 logits 沒打平的地方打平。"
                           "所以 backend 照清單上的 log 機率算排名，只數比選中的更可能的：同分第一就是 #1，和 `SimpleBackend` 一樣。",
                     "en": "**Every token's rank**, outside the top 5 as well. Its list puts the chosen token first, so the backend asks for one more and sorts the top 5 by rank. "
                           "vLLM's rank counts the tokens tied with the chosen one as ahead of it, and with speculative decoding it compares the logits, while the log probabilities "
                           "listed can tie where the logits do not. So the backend counts the rank from the listed log probabilities, only those more likely: a token tied "
                           "for first is #1, as with `SimpleBackend`."},
                    {"zh": "**`--return-sampling-mask`**：每個位置 top-k、top-p 留下了哪些 token，由此知道剩下幾個候選。vLLM 只在 `--logprobs-mode processed_logprobs` "
                           "下開它，所以有 mask 的伺服器送回的是抽樣分布：看得到取樣時的機率和剩下幾個候選，看不到模型給的機率、前 5 名，也看不到哪幾個被剃掉。"
                           "這時它也不接受 temperature 0 和 top-k 0（mask 的大小要靠 top-k 限制），所以側欄的滑桿從 0.05 和 1 開始。"
                           "沒有 `--enable-scale-out` 的伺服器看不到 mask，所以載入時也要一個 greedy 的 token，被拒絕就知道它開了 mask（[[agent/model_backends/vllm_backend.py|def _refuses_greedy(]]）。"
                           "greedy 不用問伺服器：只剩一個候選，抽中的機率是 1（[[agent/model_backends/vllm_backend.py|def _greedy(]]）。",
                     "en": "**`--return-sampling-mask`**: the tokens top-k and top-p left at each position, from which comes how many candidates were left. vLLM allows it "
                           "only with `--logprobs-mode processed_logprobs`, so a server with masks sends the sampling distribution: the sampling probability and the "
                           "candidates left show, and the model probability, the top 5 and which of them were cut do not. Such a server also refuses temperature 0 and "
                           "top-k 0 (top-k bounds the mask), so the sidebar's sliders start at 0.05 and 1. A server without `--enable-scale-out` shows no mask, so at "
                           "load it is also asked for a greedy token, and a refusal says it has masks on ([[agent/model_backends/vllm_backend.py|def _refuses_greedy(]]). Greedy decoding "
                           "needs no server: one candidate left, drawn with probability 1 ([[agent/model_backends/vllm_backend.py|def _greedy(]])."},
                ),
                {"zh": "vLLM (Docker) 的伺服器加了 `--enable-scale-out`、用預設的 `raw_logprobs`，所以看得到每個 token 的排名、模型給的機率和前 5 名。",
                 "en": "vLLM (Docker) starts its server with `--enable-scale-out` and the default `raw_logprobs`, so every token's rank, the model probability and the top 5 show."},
            ),
        }),
        {"zh": "浮窗裡的數字：`p` 是被選中的 token 在原始分布的機率；`rank` 是它的名次；"
               "`kept` 是剃完還剩幾個候選；`p_sampled` 是它在剃完、重新正規化後的分布裡的機率。"
               "step 卡片上 token 的底色深淺就是用 `p` 算的（[[app/render.py|def heat(p: float)]]）：越深，代表模型原本越不看好它，卻被抽中了。"
               "這些數字列在 `backend.py` 的 `TOKEN_DETAILS`（[[agent/model_backends/backend.py|TOKEN_DETAILS = (]]）：transformers 全都知道，vLLM 伺服器知道哪些看它怎麼啟動，"
               "不知道的就不畫，不會用別的數字代替（{{backends}}）。",
         "en": "The numbers in the popover: `p` is the chosen token's probability in the model's own distribution; `rank` "
               "is its place; `kept` is how many candidates were left after the cuts; `p_sampled` is its probability in the cut and "
               "renormalized distribution. The shade behind a token in a step card comes from `p` ([[app/render.py|def heat(p: float)]]): "
               "the darker it is, the less the model expected that token, and yet it was drawn. These numbers are listed in `backend.py`'s "
               "`TOKEN_DETAILS` ([[agent/model_backends/backend.py|TOKEN_DETAILS = (]]): transformers knows them all, a vLLM server what it was started to tell, and what "
               "a backend does not know is left out, never drawn from some other number ({{backends}})."},
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
        Tabs({
            "transformers": (
                {"zh": "猜的是草稿模型 `google/gemma-4-E2B-it-assistant`（[[agent/model_families/__init__.py|DRAFTERS = {]]）：約 78M 參數、4 層、隱藏層維度 256，"
                       "主模型是 5.1B。它沒有自己的 KV cache：Gemma 4 E2B 的 35 層裡，後 20 層本來就不自己算 key / value，而是共用前面兩層"
                       "（最後一個滑動視窗層和最後一個 full attention 層）算好的，草稿模型也讀這一份。第一次打開 MTP 時才載入，"
                       "每次猜幾個 token 寫在它的 generation config（`num_assistant_tokens` = 6）。",
                 "en": "The guesses come from the drafter `google/gemma-4-E2B-it-assistant` ([[agent/model_families/__init__.py|DRAFTERS = {]]): about 78M "
                       "parameters, 4 layers, hidden size 256, next to the model's 5.1B. It has no KV cache of its own: of Gemma 4 E2B's 35 layers, "
                       "the last 20 do not compute their own keys and values but share those of two earlier layers (the last sliding window layer "
                       "and the last full attention layer), and the drafter reads the same ones. It loads the first time MTP is switched on, and how "
                       "many tokens it guesses at a time is in its generation config (`num_assistant_tokens` = 6)."},
                Code("agent/model_backends/simple_backend.py", "def load_drafter(self):"),
                {"zh": "**猜**：輸入是「上一個 token 的 embedding」接上「主模型在上一個位置最後一層的 hidden state」，輸出下一個 token 的 logits，"
                       "取機率最高的當作猜測。接著把猜到的 token 和草稿模型自己的 hidden state 當作下一次的輸入，再猜一個，最多 6 個。"
                       "6 次都站在同一個位置，因為它看到的 key / value 只到主模型真正讀過的地方。",
                 "en": "**Guess**: the input is the last token's embedding next to the model's last-layer hidden state at the last position; the output "
                       "is the logits for the next token, and the most likely one is the guess. The guessed token and the drafter's own hidden state "
                       "are then the input for another guess, up to 6. All 6 sit at the same position, because the keys and values it sees go only "
                       "as far as the model has really read."},
                Code("agent/model_backends/simple_backend.py", "def _draft("),
                {"zh": "**驗證**：主模型對「剛確定的 token + 6 個猜測」做一次 forward，`logits_to_keep=7` 拿回 7 個位置的分布："
                       "第 i 個位置的分布，就是讀到第 i 個猜測之前時，下一個 token 的分布。從第一個猜測開始依序交給 `_sample()` 檢查："
                       "收下就是一個不用另外 forward 的 token；第一個被退回的猜測，由主模型自己在那個位置選的 token 取代，後面的猜測全部作廢；"
                       "6 個都收下時，最後一個位置的分布還能多給一個 token。所以一次 forward 產生 1 到 7 個 token。",
                 "en": "**Check**: the model runs one forward pass over the token just settled plus the 6 guesses, and `logits_to_keep=7` returns the "
                       "distributions at 7 positions: the one at position i is the distribution of the next token before reading guess i. Starting "
                       "from the first guess, `_sample()` checks them in order: a guess it keeps is a token without a forward pass of its own; the "
                       "first guess it turns down is replaced by the model's own pick at that position, and the guesses after it are dropped; when "
                       "all 6 hold, the last position's distribution gives one more token. So one forward pass produces between 1 and 7 tokens."},
                Code("agent/model_backends/simple_backend.py", "def _drafted("),
                {"zh": "作廢的猜測已經寫進了 KV cache，要用 `past.crop()` 拿掉；滑動視窗層平常會馬上丟掉太舊的 key / value，"
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
                Code("agent/model_backends/simple_backend.py", "at = (cand_ids == draft)", end="pick = int(torch.multinomial(weights, 1))"),
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
            ),
            "vllm": (
                {"zh": "推測解碼是伺服器啟動時決定的（`--speculative-config`），不是每個請求各自決定：",
                 "en": "Speculative decoding is decided when the server starts (`--speculative-config`), not request by request:"},
                (
                    {"zh": "vLLM (Docker) 在 GPU 上用 `model_families.VLLM_SPECULATIVE` 裡這個模型的設定（[[agent/model_families/__init__.py|VLLM_SPECULATIVE = {]]）："
                           "Gemma 4 用它的草稿模型 `google/gemma-4-E2B-it-assistant`，Qwen3.5 用它自己 checkpoint 裡的 MTP 層（transformers 用不了）；預設開，切換開關就用另一種方式重新啟動伺服器"
                           "（[[agent/model_backends/vllm_docker.py|def speculative(model_id]]）。vLLM 0.32 的 CPU 版 image 兩個模型都不能推測解碼，所以在 CPU 上開關不能按。",
                     "en": "vLLM (Docker) on a GPU takes the model's setting in `model_families.VLLM_SPECULATIVE` ([[agent/model_families/__init__.py|VLLM_SPECULATIVE = {]]): "
                           "Gemma 4 with its drafter, `google/gemma-4-E2B-it-assistant`, Qwen3.5 with the MTP layer in its own checkpoint, which transformers cannot use. It is on by default, and the switch "
                           "starts the server again the other way ([[agent/model_backends/vllm_docker.py|def speculative(model_id]]). vLLM 0.32's CPU image cannot decode "
                           "either model speculatively, so on a CPU the switch is disabled."},
                    {"zh": "`VLLM_URL` 的伺服器由啟動它的人決定，側欄的 MTP 開關不能按。",
                     "en": "A server at `VLLM_URL` decides as whoever started it did, and the sidebar's MTP switch is disabled."},
                ),
                {"zh": "伺服器加上 `--per-request-spec-decode-metrics summary`（要和 `--speculative-config` 一起用）時，"
                       "最後一段回應帶著這一步的統計，變成 step 卡片上的「草稿收下幾個、forward 幾次」。設成 `detailed`（vLLM (Docker) 就是）還有每一次驗證收下幾個猜測："
                       "第一個 token 來自 prefill，之後每次驗證是收下的猜測，再加一個主模型自己的 token；收下的比猜的少，那個 token 就取代了被退回的猜測。"
                       "伺服器不記沒有猜測的那幾步，所以只有這些數字剛好對得上每個 token 時，才標得出哪些是猜中的；對不上就只顯示統計。對得上時，收下的猜測只算到停止 token 為止：vLLM 是在停止 token 截斷那次驗證之前算的，停止 token 後面的猜測也會算進去。"
                       "這要等這一步結束才知道，所以底線在 step 結束時才出現（[[app/render.py|def with_drafts(]]），而被退回的猜測原本是什麼，伺服器不說。",
                 "en": "With `--per-request-spec-decode-metrics summary` (which needs `--speculative-config`), the last part of the response carries the step's counts, which "
                       "become the step card's guesses kept and forward passes. Set to `detailed`, as vLLM (Docker) has it, it also says how many guesses each verify step kept: the first "
                       "token comes from the prefill, and each verify step after it gives the guesses it kept and then one token of the model's own, which replaced a guess "
                       "turned down when fewer were kept than guessed. The server does not count a step without guesses, so only when these numbers account for every token "
                       "can the page mark which were guesses; otherwise it shows the counts alone. When they do, only the guesses up to the stop token count as kept: vLLM "
                       "counts them before it cuts the step at the stop token, so the guesses after it would count too. That is known only once the step ends, so the "
                       "underlines appear then ([[app/render.py|def with_drafts(]]), and what a turned-down guess was, the server does not say."},
                Code("agent/model_backends/vllm_backend.py", "def drafted_tokens("),
            ),
        }),
    )),
    Section("backends", "model",
            {"zh": "換一層跑模型：Backend 介面、本機 transformers 和 vLLM", "en": "What runs the model: the Backend interface, transformers here or a vLLM server"},
            {"zh": "harness 和畫面只認得 `Backend`：主要是 render、encode、generate 三個方法，和它送出的事件。前面幾段在頁籤裡分開說明本機的 transformers 和 vLLM 伺服器；"
                   "這一段是它們共用的介面，以及 vLLM 伺服器從哪裡來。",
             "en": "The harness and the page know only a `Backend`: mainly three methods, render, encode and generate, and the events it yields. The sections above tell "
                   "transformers here and a vLLM server apart in their tabs; this one is about the interface they share, and where a vLLM server comes from."}, (
        {"zh": "前面幾段的 prefill、解碼迴圈、抽樣和 MTP，transformers 頁籤裡是 `simple_backend.py` 的 `SimpleBackend`，vLLM 頁籤裡是 `vllm_backend.py` 的 `VLLMBackend`。"
               "可是 harness 和畫面從來不直接找它們，只透過 `Backend` 用模型（[[agent/model_backends/backend.py|class Backend(Protocol):]]）：`render()` 把對話套進 chat template，`encode()` 把 prompt 變成模型讀的 token（媒體展開），"
               "`generate()` 依序送出 `prefill`、每個 token 一個 `token`、最後的 `done`；再加上畫面會讀的幾個屬性：family、模型讀得懂什麼、脈絡上限、模型分頁用的 `info`。"
               "誰有這些，誰就能跑模型。",
         "en": "The prefill, the decoding loop, the sampling and MTP in the sections above are `SimpleBackend`'s in `simple_backend.py` in their transformers tabs, and "
               "`VLLMBackend`'s in `vllm_backend.py` in their vLLM tabs. Yet the harness and the page never ask for either; they use the model only through a `Backend` "
               "([[agent/model_backends/backend.py|class Backend(Protocol):]]): `render()` puts the conversation through "
               "the chat template, `encode()` turns the prompt into the tokens the model reads (media expanded), and `generate()` yields `prefill`, then a `token` "
               "for each token, then `done`; and the page reads a few attributes: the family, what the model reads, the context limit and the Model tab's `info`. "
               "Whatever has these can run the model."},
        Code("agent/model_backends/backend.py", "class Backend(Protocol):",
             caption={"zh": "`Protocol` 只是一個型別：`SimpleBackend` 不必繼承它，有這些方法和屬性就算數。測試檢查 `SimpleBackend`、`VLLMBackend` 和測試用的 `FakeLLM` 都是 `Backend`。",
                      "en": "A `Protocol` is only a type: `SimpleBackend` need not inherit from it, and having these methods and attributes is enough. "
                            "The tests check that `SimpleBackend`, `VLLMBackend` and the tests' `FakeLLM` are each a `Backend`."}),
        {"zh": "畫面只在一個地方決定用哪一個：`load_llm()` 拿到伺服器的網址（`VLLM_URL`，或畫面在 Docker 裡啟動的伺服器）就用 vLLM，沒有就在這台電腦上跑"
               "（[[app/page.py|llm = vllm_backend.VLLMBackend(model_id, url) if url else SimpleBackend(model_id)]]）。"
               "之後 `current_llm()`、側欄和 `SimpleHarness`（{{prepare}}）拿到的都只是一個 `Backend`；它放在 `st.cache_resource` 裡，換模型或換「用什麼跑模型」才換掉（{{submit}}）。",
         "en": "The page decides which one in a single place: given a server's URL (`VLLM_URL`, or that of the server the page started in Docker), `load_llm()` "
               "uses vLLM, and without one runs the model here ([[app/page.py|llm = vllm_backend.VLLMBackend(model_id, url) if url else SimpleBackend(model_id)]]). "
               "From then on `current_llm()`, the sidebar and `SimpleHarness` ({{prepare}}) get just a `Backend`; it stays in `st.cache_resource` until the model, or what "
               "runs it, changes ({{submit}})."},
        {"zh": "兩種 backend 對模型知道的不一樣多，差別全在 `token` 事件裡。`TOKEN_FIELDS` 每個 backend 都要送：位置、token 的 id 和 piece、它加上的文字、"
               "是不是特殊 token、正在寫思考、工具呼叫還是回覆。`TOKEN_DETAILS` 則是知道才送：機率、排名、取樣時的機率、剩下的候選、前 5 名，以及 MTP 草稿猜了什麼。",
         "en": "The two backends know different amounts about the model, and the difference is all in the `token` event. Every backend sends `TOKEN_FIELDS`: "
               "the place, the token's id and piece, the text it adds, whether it is special, and whether it is thinking, a tool call or the reply. `TOKEN_DETAILS` "
               "it sends only when it knows them: the probability, the rank, the sampling probability, the candidates left, the top 5, and what the MTP drafter guessed."},
        Code("agent/model_backends/backend.py", "TOKEN_FIELDS = (", end="TOKEN_DETAILS = ("),
        {"zh": "畫面拿到什麼就畫什麼（[[app/render.py|def known(]]）：沒回報的數字整列不畫，沒有機率就沒有熱度底色，沒說哪些候選被剃掉，前 5 名就不會變灰"
               "（[[app/render.py|def candidates_html(]]）。畫面寧可少畫，也不拿別的數字湊。",
         "en": "The page draws what it gets ([[app/render.py|def known(]]): a number nobody reported is not a row, a token without a probability gets no heat, and when "
               "nobody said which candidates were cut, none of the top 5 is grayed out ([[app/render.py|def candidates_html(]]). The page would rather show less than "
               "fill in a number from something else."},
        {"zh": "三個 backend：", "en": "The three backends:"},
        Tabs({
            "transformers": (
                {"zh": "`SimpleBackend`（`simple_backend.py`）用 transformers 把權重載到這台電腦，自己寫的解碼迴圈每個 token 都看得到完整的分布，所以 `TOKEN_DETAILS` 每一項都有；"
                       "MTP 的每個猜測收下或退回都知道（{{mtp}}）；greedy 的輸出和 `generate()` 一樣；checkpoint 有編碼器的圖片、聲音、影片都讀得懂。",
                 "en": "`SimpleBackend` (`simple_backend.py`) loads the weights onto this machine with transformers, and its hand-written decoding loop sees the whole distribution at "
                       "every token, so it sends every one of `TOKEN_DETAILS`; it knows each MTP guess it kept or turned down ({{mtp}}); its greedy output equals "
                       "`generate()`'s; it reads the images, sound and video the checkpoint has encoders for."},
            ),
            "vllm": (
                {"zh": "`VLLMBackend`（`vllm_backend.py`）的權重在 vLLM 伺服器上。每個 token 回來時帶著它的 log 機率和最可能的幾個 token，其他的看伺服器怎麼啟動（{{sampling}}）；"
                       "模型讀得懂的媒體，是 checkpoint 有編碼器、而且伺服器收的那幾種（{{media}}）；推測解碼由伺服器決定（{{mtp}}）。",
                 "en": "`VLLMBackend` (`vllm_backend.py`) has the weights on a vLLM server. Each token comes back with its log probability and those of the most likely tokens, "
                       "and more depending on how the server was started ({{sampling}}); the media the model reads are those the checkpoint has encoders for and the "
                       "server takes ({{media}}); speculative decoding is the server's ({{mtp}})."},
            ),
            "fake": (
                {"zh": "`FakeLLM`（`tests/fakes.py`）照劇本一個 token 一個 token 吐出寫好的輸出，不載入權重；大部分測試用它跑完整個 harness 和網頁"
                       "（[[tests/fakes.py|class FakeLLM:]]）。",
                 "en": "`FakeLLM` (`tests/fakes.py`) replays scripted output token by token and loads no weights; most tests run the whole harness and page "
                       "with it ([[tests/fakes.py|class FakeLLM:]])."},
            ),
        }),
        {"zh": "vLLM 伺服器有兩種來處：", "en": "A vLLM server comes from one of two places:"},
        Tabs({
            "vllm-docker": (
                {"zh": "沒設 `VLLM_URL` 時，側欄「用什麼跑模型」選 vLLM (Docker)，"
                       "`agent/model_backends/vllm_docker.py` 就照 `docker-compose/compose.yaml` 用 Docker Compose 啟動它（[[agent/model_backends/vllm_docker.py|def start(model_id]]）。"
                       "權重先下載到這台電腦的 `models/`，容器唯讀掛上、自己不下載，所以伺服器載入的正是 `REVISIONS` 固定的那個 commit（{{files}}）。"
                       "Docker 有 NVIDIA 的 runtime 時，再疊上 `compose.gpu.yaml` 改在 GPU 上跑（KV cache 和 CPU 上一樣是 2 GiB，不是 vLLM 預設的 GPU 記憶體的 92%），否則在 CPU 上跑（[[agent/model_backends/vllm_docker.py|def device()]]），"
                       "Mac 也一樣，在 Docker 的 Linux 虛擬機裡；CPU 版的 image 由 `docker-compose/vllm-cpu/` 建，補上 vLLM 0.32.0 在 CPU 上缺的兩個修正。換到 vLLM 之前，畫面先放掉這台電腦上的模型（[[app/page.py|def release_model():]]）；"
                       "換回 transformers 或結束這個程式時，停掉容器（[[agent/model_backends/vllm_docker.py|def stop():]]、[[agent/model_backends/vllm_docker.py|def _stop_at_exit():]]）。",
                 "en": "Without `VLLM_URL`, when the sidebar's What runs the model picks vLLM (Docker), "
                       "`agent/model_backends/vllm_docker.py` starts it with Docker Compose from `docker-compose/compose.yaml` ([[agent/model_backends/vllm_docker.py|def start(model_id]]). "
                       "The weights are downloaded into this machine's `models/` first, and the container mounts that read-only and downloads nothing itself, so the "
                       "server loads exactly the commit `REVISIONS` pins ({{files}}). Where Docker has NVIDIA's runtime, `compose.gpu.yaml` goes on top and the server runs on the GPU, with a 2 GiB KV cache as on the CPU rather than vLLM's 92% of the GPU's memory; "
                       "elsewhere it runs on the CPU ([[agent/model_backends/vllm_docker.py|def device()]]), a Mac's included, in Docker's Linux VM, from an image "
                       "`docker-compose/vllm-cpu/` builds with two fixes vLLM 0.32.0 lacks on a CPU. Before vLLM takes over, the page "
                       "lets go of the model on this machine ([[app/page.py|def release_model():]]); picking transformers again, or quitting the app, stops the container "
                       "([[agent/model_backends/vllm_docker.py|def stop():]], [[agent/model_backends/vllm_docker.py|def _stop_at_exit():]])."},
            ),
            "vllm-url": (
                {"zh": "任何一台 vLLM 伺服器，在這台電腦上或別台都可以，用 `vllm serve <模型> --revision <它在 REVISIONS 的 commit>` 啟動，再用 `VLLM_URL` 告訴網頁它在哪裡。"
                       "模型選單列出它提供的模型（`/v1/models`），每個用它載入時的名字，不列 LoRA adapter；網頁還不支援的家族標上 ⚠，選了會說明原因，"
                       "也可以直接到 GitHub 提 issue（[[app/page.py|def model_menu()]]）。每個 token 看得到多少、模型讀得懂哪些媒體，看伺服器啟動時加了哪些參數"
                       "（{{sampling}}、{{media}}、{{mtp}}）。prompt 送到那台伺服器，網頁本身還是只在這台電腦上。",
                 "en": "Any vLLM server, on this machine or another, started with `vllm serve <model> --revision <its commit in REVISIONS>`, and `VLLM_URL` tells the page where it is. "
                       "The model menu lists the models it serves (`/v1/models`), each by what it was loaded from, without the LoRA adapters; one of a family the page does "
                       "not support yet is marked ⚠, and picking it says why and offers a GitHub issue asking for it ([[app/page.py|def model_menu()]]). How much the page sees "
                       "of each token, and which media the model reads, depends on the flags the server was started with ({{sampling}}, {{media}}, {{mtp}}). The prompts go "
                       "to that server; the page itself stays on this machine."},
            ),
        }),
        {"zh": "`VLLMBackend` 的 prompt 一樣精確：它讀的 tokenizer 和 chat template 跟 `SimpleBackend` 是 `models/` 裡同一個 commit 的同一份（{{files}}），"
               "prompt 在這裡 render、切成 token，送到伺服器的是 token id 而不是文字，所以伺服器讀到的就是脈絡分頁上那一串（{{context}}）。",
         "en": "`VLLMBackend`'s prompt is just as exact: it reads the very tokenizer and chat template `SimpleBackend` does, from the same commit in `models/` ({{files}}), "
               "renders and tokenizes the prompt here, and sends the server token ids, not text, so the server reads exactly the string in the Context tab ({{context}})."},
        {"zh": "模型分頁寫出伺服器和它的版本，不寫裝置、精度和參數量；checkpoint 讀得懂但伺服器不收的媒體，連同原因列在那裡（{{media}}）；"
               "模型資料夾列出網頁讀的那個資料夾（{{files}}）。",
         "en": "The Model tab names the server and its version and leaves out the device, the precision and the parameter count; the media the checkpoint reads but "
               "the server does not take are listed there with why ({{media}}), and the model folder card lists the folder the page reads ({{files}})."},
        {"zh": "要接 SGLang、Ollama 之類的另一層，就是再寫一個有這些方法和屬性的類別，送出一樣的事件，在 `load_llm()` 多一個選擇；"
               "`TOKEN_DETAILS` 裡它知道的才送。`tests/test_vllm_backend.py` 裡的假伺服器可以當寫測試的樣板。",
         "en": "To use another serving layer, SGLang or Ollama say, write one more class with these methods and attributes that yields the same events, and give "
               "`load_llm()` one more choice; it sends only the `TOKEN_DETAILS` it knows. The fake server in `tests/test_vllm_backend.py` is a pattern for its tests."},
    )),
    Section("parse", "harness",
            {"zh": "Harness：把輸出拆成思考、工具呼叫、回覆", "en": "Harness: splitting the output into thinking, tool calls and the reply"},
            {"zh": "模型只會吐文字；哪段是工具呼叫，要有人讀出來：內建 harness 和模型端點用 family 的 parser，provider 的 API 則自己讀好再傳回來。",
             "en": "The model only writes text; which part is a tool call, someone has to read out: the built-in harness and the model endpoint with the family's parser, a provider's API before it sends the reply back."}, (
        Tabs({
            "simple": (
                {"zh": "chat template 有時會在 prompt 結尾先打開思考（family 的 `thought_prefix`）；這段在 prompt 裡、不在 `raw` 裡，"
                       "所以先補回來，再交給 family 的 `parse`，`s.lang` 是它寫錯誤訊息用的語言。關掉「使用工具」時，就算模型寫了工具呼叫也當作沒有。",
                 "en": "The chat template sometimes opens the thinking at the end of the prompt (the family's `thought_prefix`); that part is in the "
                       "prompt, not in `raw`, so it is put back before the output goes to the family's `parse`, with `s.lang` as the language for its "
                       "error messages. With Use tools off, a tool call the model writes anyway counts as none."},
                Code("agent/harnesses/simple_harness.py", "prefix = family.thought_prefix if", end='self.emit("parsed"'),
            ),
            "pi-page": (
                {"zh": "端點用同一個 family 的 parser 讀模型的輸出，chat template 先打開的思考也一樣先補回來；讀得出來的呼叫以 `tool_calls` 傳回 pi，由 pi 去執行。"
                       "讀不出來的呼叫不送，pi 的這一輪就停在那裡：不像內建 harness（{{check}}），沒有人告訴模型哪裡寫錯。pi 收到的訊息結束時，`PiHarness` 把它讀到的思考、回覆和工具呼叫送成 `parsed`。",
                 "en": "The endpoint reads the model's output with the same family's parser, putting back first the thinking the chat template opened; the calls it can read "
                       "go back to pi as `tool_calls`, for pi to run. A call it cannot read is not sent, and pi's turn ends there: unlike the built-in harness ({{check}}), "
                       "nothing tells the model what it wrote wrong. Once the message pi got ends, `PiHarness` sends the thinking, the reply and the calls in it as `parsed`."},
                Code("agent/harnesses/model_endpoint.py", "prefix = family.thought_prefix if", end="send(finish=finish)"),
            ),
            "pi-provider": (
                {"zh": "provider 的 API 把工具呼叫和文字分開傳回來，已經是 JSON；模型自己的格式留在 provider 那邊。pi 從結束的訊息裡讀出它們，`PiHarness` 連同思考和回覆送成 `parsed`。",
                 "en": "A provider's API sends the tool calls apart from the text, already as JSON; the model's own format stays at the provider. pi reads them out of the "
                       "finished message, and `PiHarness` sends them on as `parsed`, with the thinking and the reply."},
                Code("agent/harnesses/pi_harness.py", 'blocks = message.get("content") or []', end='self.emit("parsed"'),
            ),
        }),
        {"zh": "內建 harness 和 pi 問這個網頁的模型時，讀輸出的都是 family 的 parser。兩個 parser 回傳一樣的結構：`thinking`、`tool_calls`（每個有 `name`、`arguments`、`error`）、`content`，所以後面的 harness 不用管是哪個模型。",
         "en": "For the built-in harness and for pi on this page's model, the family's parser reads the output. Both parsers return the same structure, "
               "`thinking`, `tool_calls` (each with `name`, `arguments` and `error`) and `content`, "
               "so the rest of the harness does not care which model it was."},
        Tabs({
            "gemma4": (
                {"zh": "Gemma 4 的 parser 分三步：(1) 切出 `<|channel>thought` 到 `<channel|>` 之間的思考；(2) 用正規表達式找出每一個 `<|tool_call>call:名稱{…}<tool_call|>`"
                       "（[[agent/model_families/gemma4.py|CALL_RE = re.compile]]），參數交給小 parser；(3) 有開頭、沒有結尾的工具呼叫（例如被 token 上限截斷）記成格式錯誤。"
                       "剩下的文字去掉停止 token，就是給使用者的回覆。",
                 "en": "Gemma 4's parser works in three steps: (1) cut out the thinking between `<|channel>thought` and `<channel|>`; (2) find every "
                       "`<|tool_call>call:name{…}<tool_call|>` with a regular expression ([[agent/model_families/gemma4.py|CALL_RE = re.compile]]) and hand the "
                       "arguments to a small parser; (3) record a tool call that opens but never closes (cut off at the token limit, say) as a format "
                       "error. The text left over, without stop tokens, is the reply for the user."},
                Code("agent/model_families/gemma4.py", "def parse_output("),
                {"zh": "Gemma 的參數不是 JSON：key 不加引號，字串用 `<|\"|>` 這個特殊 token 包起來，例如 `{timezone:<|\"|>Asia/Taipei<|\"|>}`。"
                       "`_Parser` 是遞迴下降 parser：看到 `<|\"|>` 就一路讀到下一個 `<|\"|>`，`{` 讀物件、`[` 讀陣列、`\"` 讀 JSON 字串，"
                       "其他的當作數字、`true` / `false` 或 `null`，都不是就當成字串。",
                 "en": "Gemma's arguments are not JSON: keys have no quotes, and strings are wrapped in the special token `<|\"|>`, as in "
                       "`{timezone:<|\"|>Asia/Taipei<|\"|>}`. `_Parser` is a recursive descent parser: at `<|\"|>` it reads up to the next `<|\"|>`, "
                       "at `{` an object, at `[` an array, at `\"` a JSON string, and anything else is a number, `true` / `false` or `null`, or else a string."},
                Code("agent/model_families/gemma4.py", "def value(self):"),
            ),
            "qwen3_5": (
                {"zh": "Qwen3.5 的 parser：思考在 `</think>` 之前；每個 `<tool_call>` 區塊裡找 `<function=名稱>` 和一段段 `<parameter=名稱>`。"
                       "值都是純文字，所以要照工具的 schema 轉型別：schema 說是字串就保持原樣，說是數字、陣列或物件就用 JSON 讀，"
                       "說是布林就看是不是 `true` / `false`；轉不過去就留著文字，讓下一關的 schema 驗證把型別錯誤回報給模型。",
                 "en": "Qwen3.5's parser: the thinking is before `</think>`; in each `<tool_call>` block it finds `<function=name>` and the "
                       "`<parameter=name>` sections. Every value is plain text, so its type comes from the tool's schema: a string stays as it is, "
                       "a number, array or object is read as JSON, and a boolean is whether it says `true` or `false`; text that does not convert "
                       "stays text, and the schema check that follows reports the wrong type to the model."},
                Code("agent/model_families/qwen3_5.py", "def parse_output("),
                Code("agent/model_families/qwen3_5.py", "def convert(value: str, kind"),
            ),
        }),
    )),
    Section("stop", "stop",
            {"zh": "要不要停：由 harness 決定", "en": "Whether to stop: the harness decides"},
            {"zh": "模型只是停止生成；這一輪要不要結束，是 harness 看完輸出才決定的。",
             "en": "The model only stops generating; whether the turn ends is up to the harness, once it has read the output."}, (
        Tabs({
            "simple": (
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
            Code("agent/harnesses/simple_harness.py", 'if done["finish"] == "user_stopped":', end="if not calls:"),
            {"zh": "其他停止條件分散在迴圈各處：", "en": "The other ways to stop are spread around the loop:"},
            (
                {"zh": "for 迴圈跑完所有步數都沒 break → `max_steps`（{{prepare}}設定的預設值）。",
                 "en": "The for loop uses up its steps without a break → `max_steps` (the default set in {{prepare}})."},
                {"zh": "等待確認時被停止或逾時 → `user_stopped` 或 `approval_timeout`（[[agent/harnesses/simple_harness.py|stop_reason = \"approval_timeout\" if self._timed_out]]）。",
                 "en": "Stopped or timed out while waiting for a confirmation → `user_stopped` or `approval_timeout` "
                       "([[agent/harnesses/simple_harness.py|stop_reason = \"approval_timeout\" if self._timed_out]])."},
                {"zh": "連續 `MAX_FORMAT_ERRORS`（3）次寫出無法解析的工具呼叫 → `format_errors`。",
                 "en": "`MAX_FORMAT_ERRORS` (3) tool calls in a row that cannot be parsed → `format_errors`."},
                {"zh": "backend 或 harness 本身丟出任何例外 → `error`，錯誤訊息會顯示在畫面上，而不是讓 thread 默默死掉；backend 的錯誤（例如 vLLM 伺服器的）兩種語言都有，畫面顯示目前語言的那一個。"
                       "工具丟的例外不會結束這一輪，而是變成觀察（{{execute}}）。",
                 "en": "Any exception from the backend or the harness itself → `error`; the message shows on the page instead of the thread dying silently, and a backend's error "
                       "(a vLLM server's, say) comes in both languages, of which the page shows the one it is in. A tool's exception does not end the turn; it "
                       "becomes an observation ({{execute}})."},
            ),
            Code("agent/harnesses/simple_harness.py", "if format_errors >= MAX_FORMAT_ERRORS:", end='self.emit("run_end"',
                 caption={"zh": "不管怎麼停，最後一定送出 `run_end`，帶著停止原因、最終回覆和整份訊息列表。",
                          "en": "However it stops, `run_end` always goes out last, with the stop reason, the final reply and the whole message list."}),
            ),
            "pi": (
                {"zh": "迴圈什麼時候結束由 pi 決定：一則沒有工具呼叫的回覆就結束它，pi 再用 `agent_settled` 說它沒有別的事要做了。`PiHarness` 只守著頁面的上限："
                       "pi 自己沒有步數上限，所以 pi 開始超過「步數上限」的那個 turn 時，`PiHarness` 請 pi 中止，不再畫下一步；按停止或確認逾時也會中止它。",
                 "en": "pi decides when its loop ends: a reply without a tool call ends it, and pi then says with `agent_settled` that it has nothing more to do. `PiHarness` "
                       "only keeps the page's limits: pi has no step limit of its own, so when pi starts the turn past Max steps, `PiHarness` asks pi to abort and draws no "
                       "further step; pressing stop or an unanswered confirmation aborts it too."},
                Code("agent/harnesses/pi_harness.py", "def _turn_start(self):"),
                {"zh": "停止原因照為什麼停來定：被中止就是中止的原因，pi 回報請求失敗就是 `error`（pi 的自動重試關掉了，失敗不會變成另一個 step），回覆被 token 上限截斷就是 `max_new_tokens`，其他都是 `no_tool_call`。",
                 "en": "The stop reason follows why it stopped: an abort's reason, `error` when pi reports a failed request (pi's retries are off, so a failure does not become "
                       "another step), `max_new_tokens` for a reply cut off at the token limit, and `no_tool_call` otherwise."},
                Code("agent/harnesses/pi_harness.py", "def _stop_reason(self) -> str:"),
            ),
        }),
    )),
    Section("record", "harness",
            {"zh": "把工具呼叫寫回對話", "en": "Writing the tool call back into the conversation"},
            {"zh": "工具呼叫先變成一則 assistant 訊息，下一步由 chat template 寫回模型自己的格式。",
             "en": "A tool call first becomes an assistant message, which the chat template writes back in the model's own format in the next step."}, (
        Tabs({
            "simple": (
                {"zh": "有工具呼叫時，harness 給每個呼叫一個 id（`call_{step}_{i}`），寫成一則 assistant 訊息，"
                       "格式和 OpenAI / Hugging Face 的 `tool_calls` 一樣；下一步 chat template 會把它寫回 Gemma 的 `<|tool_call>…<tool_call|>` 或 Qwen 的 `<tool_call>…</tool_call>`。",
                 "en": "With tool calls, the harness gives each one an id (`call_{step}_{i}`) and writes them into an assistant message in the format "
                       "of OpenAI's and Hugging Face's `tool_calls`; in the next step the chat template writes it back as Gemma's "
                       "`<|tool_call>…<tool_call|>` or Qwen's `<tool_call>…</tool_call>`."},
                {"zh": "接著先 render 一次「加上這則 assistant 訊息、但還沒有工具結果」的字串，存成 `prev_sequence`；`generation_prompt=False`，"
                       "因為這時還輪不到模型寫。下一步的 prompt 和它比，多出來的就是工具結果。直接拿「上一步 prompt + 模型輸出」來比會不準，"
                       "因為模板重新 render 時會改寫模型那一段（排序參數、換掉思考的標記）。",
                 "en": "Then it renders the conversation with this assistant message but without the tool results yet, and keeps that as "
                       "`prev_sequence`; `generation_prompt=False`, since it is not the model's turn to write yet. Compared with it, what the next "
                       "step's prompt has in addition is the tool result. Comparing with the last prompt plus the model's output would be off, "
                       "because the template rewrites the model's part when it renders it again (it sorts the arguments and replaces the thinking markup)."},
                Code("agent/harnesses/simple_harness.py", 'ids = [f"call_{step}_{i}"', end="prev_sequence = self.llm.render("),
            ),
            "pi": (
                {"zh": "pi 自己把這一步寫進它的 session：模型那則訊息（連同工具呼叫），接著每個工具結果，存在這段對話資料夾裡的 session 檔，下一個請求就帶著這一切。"
                       "pi 問這個網頁的模型時，端點把請求換回對話（{{context}}），chat template 再把工具呼叫寫回模型自己的格式，和下面說的一樣；"
                       "問 pi 自己的模型時，請求裡的工具呼叫是 JSON，provider 怎麼寫給模型就看不到了。",
                 "en": "pi writes the step into its session itself: the model's message, tool calls included, then each tool result, in the session file in the "
                       "conversation's folder, and its next request carries all of it. On this page's model, the endpoint turns the request back into the conversation "
                       "({{context}}), and the chat template writes the calls back in the model's own format, as below; on pi's own models, the calls in the request "
                       "are JSON, and how the provider writes them for its model is not to be seen."},
            ),
        }),
        {"zh": "chat template 把工具呼叫寫回模型自己的格式時（內建 harness 和端點都一樣），兩個家族在這裡有兩點不同，都由 family 決定：模型在工具呼叫前寫的文字要不要留（`text_with_calls`），以及思考內容存在哪個欄位，模板才讀得到（`reasoning_key`）。",
         "en": "Where the chat template writes the calls back in the model's own format (for the built-in harness and for the endpoint alike), the two families differ in two ways, both decided by the family: whether the text a model writes before a tool call is kept "
               "(`text_with_calls`), and the field the thinking goes in, for the template to read it from (`reasoning_key`)."},
        Tabs({
            "gemma4": (
                {"zh": "Gemma 的模板會把工具呼叫前的文字挪到工具結果後面，並在那裡關掉回合，所以這段文字丟掉；思考存在 `reasoning`。",
                 "en": "Gemma's template would move the text before a tool call after the tool results and close the turn there, so that text is dropped; "
                       "the thinking goes in `reasoning`."},
            ),
            "qwen3_5": (
                {"zh": "Qwen 的模板會保留工具呼叫前的文字，所以照樣留著；思考存在 `reasoning_content`。",
                 "en": "Qwen's template keeps the text before a tool call, and so does the harness; the thinking goes in `reasoning_content`."},
            ),
        }),
    )),
    Section("check", "harness",
            {"zh": "驗證與權限：模型從不直接碰環境", "en": "Validation and permission: the model never touches the environment"},
            {"zh": "每個呼叫要過三關：格式和 schema、重複檢查、（開著時）你的確認。",
             "en": "Each call has three checks to pass: format and schema, repeats, and (when it is on) your confirmation."}, (
        Tabs({
            "simple": (
            {"zh": "每個呼叫交給 `_handle_call()`。第一關是驗證：解析失敗、沒有這個工具（這一輪沒提供的也算，例如沒打開「讓模型寫程式」時的 `run_python`）、"
                   "參數不符合 schema，都不會執行。",
             "en": "Each call goes to `_handle_call()`. The first check is validation: a call that failed to parse, names no tool offered this turn "
                   "(`run_python` without Let the model write code, say) or has arguments that do not fit the schema never runs."},
            Code("agent/harnesses/simple_harness.py", "def _handle_call(", end="args, errors = tool.validate(args, lang)"),
            {"zh": "`Tool.validate()` 檢查必要參數有沒有給、有沒有多出不認得的參數、型別和 enum 對不對，然後補上預設值。"
                   "錯誤訊息是完整的句子，用這一輪開始時頁面的語言寫，因為它會原封不動變成模型下一步讀到的觀察。",
             "en": "`Tool.validate()` checks that the required arguments are there, that there are no unknown ones, and that the types and enums "
                   "are right, then fills in the defaults. Each error is a full sentence in the language the page was in when the turn started, "
                   "because it becomes, word for word, the observation the model reads in the next step."},
            Code("agent/tools.py", "def validate(self, args: dict"),
            {"zh": "第二關記錄重複：把「工具名稱 + 排序後的參數 JSON」當成 key，跟這一輪之前的呼叫比；一模一樣就在 step 卡片標出"
                   "「迴圈可能在打轉」。這只是提醒，不會擋。然後送出 `tool_call` 事件；有錯誤就直接把錯誤包成觀察回報，不進環境。",
             "en": "The second check records repeats: the tool name plus the arguments as sorted JSON is the key, compared with the earlier calls "
                   "in this turn; an exact repeat is flagged in the step card as the loop maybe going in circles. It is only a warning and blocks "
                   "nothing. Then the `tool_call` event goes out; with errors, the errors go straight back as the observation, without reaching the environment."},
            Code("agent/harnesses/simple_harness.py", 'key = (call["name"], json.dumps', end='{"error": "; ".join(errors)}'),
            {"zh": "第三關是你。開著「執行工具前先問我」時，harness 送出 `approval_request`，然後停在 `_wait_for_decision()` 等。"
                   "拒絕、停止、逾時都會變成一則錯誤觀察，模型下一步只看得到這段文字。",
             "en": "The third check is you. With Ask me before running a tool on, the harness sends `approval_request` and waits in "
                   "`_wait_for_decision()`. A rejection, a stop or a timeout each becomes an error observation, and that text is all the model sees in the next step."},
            Code("agent/harnesses/simple_harness.py", "if self.settings.approve_tools:", end='return self._observe(step, cid, call["name"], {"error": error}'),
            {"zh": "等待用的是 `threading.Condition`：每 0.5 秒醒來一次，直到你按下按鈕（`decide()` 會 `notify_all`）、按了停止，"
                   "或超過 `APPROVAL_TIMEOUT`（[[agent/harnesses/simple_harness.py|APPROVAL_TIMEOUT = ]]，5 分鐘）。逾時就當作拒絕並停止整輪，免得一個關掉的分頁把模型佔住。",
             "en": "The wait uses a `threading.Condition`: it wakes every 0.5 s until you press a button (`decide()` calls `notify_all`), press stop, "
                   "or `APPROVAL_TIMEOUT` passes ([[agent/harnesses/simple_harness.py|APPROVAL_TIMEOUT = ]], 5 minutes). A timeout counts as a rejection and stops the "
                   "whole turn, so a closed tab cannot hold on to the model."},
            Code("agent/harnesses/simple_harness.py", "def _wait_for_decision("),
            ),
            "pi": (
                {"zh": "pi 執行工具前自己用工具的 schema 檢查參數；不符合的呼叫不會執行，錯誤就是模型讀到的工具結果。重複的呼叫，`PiHarness` 用和內建 harness 一樣的方式標出來"
                       "（[[agent/harnesses/pi_harness.py|def _tool_start(self]]）。",
                 "en": "pi checks a call's arguments against the tool's schema itself before running it; a call that does not fit does not run, and the error is the result "
                       "the model reads. `PiHarness` flags repeats as the built-in harness does ([[agent/harnesses/pi_harness.py|def _tool_start(self]])."},
                {"zh": "pi 執行工具前不會問人。打開「執行工具前先問我」時，網頁的 extension 在 pi 每次要執行工具前開一個確認對話框。"
                       "在 RPC 模式下，這個對話框是 stdout 上的一筆 `extension_ui_request`，`PiHarness` 把它變成 `approval_request`（[[agent/harnesses/pi_harness.py|def _dialog(self, rec: dict)]]），"
                       "你按的「允許」或「拒絕」再從 stdin 送回去（[[agent/harnesses/pi_harness.py|def _decided(self]]）；被拒絕的呼叫不執行，模型讀到的結果就是拒絕的原因，用這一輪開始時頁面的語言寫。"
                       "沒有人回答的話，和內建 harness 一樣，`APPROVAL_TIMEOUT` 之後停掉 pi。",
                 "en": "pi does not ask anyone before it runs a tool. With Ask me before running a tool on, the page's extension opens a confirm dialog every time pi is "
                       "about to run a tool. In RPC mode the dialog is an `extension_ui_request` on stdout, which `PiHarness` turns into an `approval_request` "
                       "([[agent/harnesses/pi_harness.py|def _dialog(self, rec: dict)]]), and your Allow or Reject goes back on stdin ([[agent/harnesses/pi_harness.py|def _decided(self]]); "
                       "a rejected call does not run, and the result the model reads is the refusal, in the language the page was in when the turn started. Unanswered, "
                       "the dialog stops pi after `APPROVAL_TIMEOUT`, as with the built-in harness."},
                Code("agent/harnesses/pi_extension.ts", 'pi.on("tool_call"', end="return undefined;"),
            ),
        }),
        {"zh": "畫面這邊，等待中的呼叫會在聊天區顯示「允許／拒絕」按鈕，按下去就呼叫 `run.run.decide()`；"
               "像 `run_python` 的程式碼這種好幾行的參數，另外用程式碼區塊顯示，看得清楚再決定。"
               "「先問我」寫在 harness 的程式裡，而不是寫在 prompt 裡請模型遵守，所以模型沒有辦法繞過。",
         "en": "On the page, a waiting call shows Allow and Reject buttons in the chat, and pressing one calls `run.run.decide()`; an argument "
               "over several lines, like `run_python`'s code, is shown in a code block of its own, so you can read it before you decide. Asking first "
               "is written into the harness's code, not into the prompt as a request to the model, so the model has no way around it."},
        Code("app/page.py", "for call in render.pending_approvals(data):"),
    )),
    Section("execute", "env",
            {"zh": "環境：真的去執行工具", "en": "Environment: really running the tool"},
            {"zh": "通過檢查才會執行工具；錯誤也會變成觀察，不會讓迴圈當掉。",
             "en": "Only after the checks does the tool run; its errors become observations too, and never crash the loop."}, (
        Tabs({
            "simple": (
            {"zh": "送出 `tool_start` 後呼叫工具的 Python 函式；`takes_files` 的工具（`run_python`）另外拿到這段對話的資料夾。"
                   "工具自己丟的 `ToolError` 是可以預期的錯誤（時區打錯、找不到頁面、程式出錯），"
                   "訊息用這一輪的語言直接給模型看；其他例外也會被接住，變成 `{\"error\": …}`。",
             "en": "After `tool_start` goes out, the tool's Python function is called; a tool that `takes_files` (`run_python`) also gets the "
                   "conversation's folder. A `ToolError` from the tool is an expected failure (a mistyped timezone, a page that does not exist, "
                   "code that fails), and its message goes to the model as it is, in the turn's language; any other exception is caught too and "
                   "becomes `{\"error\": …}`."},
            Code("agent/harnesses/simple_harness.py", 'self.emit("tool_start"', end="return self._observe(step, cid, call[\"name\"], result, ok=ok"),
            {"zh": "五個工具各自連到不同的環境：", "en": "The five tools reach different environments:"},
            (
                {"zh": "`get_current_time`：讀系統時鐘，用 IANA 時區名稱換算（[[agent/tools.py|def get_current_time(]]）。",
                 "en": "`get_current_time`: reads the system clock and converts with an IANA timezone name ([[agent/tools.py|def get_current_time(]])."},
                {"zh": "`calculator`：在 AST 白名單上計算，從不呼叫 `eval()`。只允許數字、四則運算、次方和少數數學函式；"
                       "算次方前先估計位數，`round` 的位數參數也有上限（它在內部也會算次方），避免 `10**10**10` 這種算式把 thread 卡死。",
                 "en": "`calculator`: computes over an AST allowlist and never calls `eval()`. Only numbers, arithmetic, powers and a few math functions "
                       "are allowed; before a power it estimates the number of digits, and it caps the digits argument of `round`, which computes a power "
                       "inside, so an expression like `10**10**10` cannot hang the thread."},
                {"zh": "`search_wikipedia` / `read_wikipedia`：用 httpx 呼叫 Wikipedia API（[[agent/tools.py|def search_wikipedia(]]），"
                       "仿照 ReAct 論文在 HotpotQA 上給模型的 Wikipedia 動作（search、lookup、finish），這裡是 search 和 read 兩個。",
                 "en": "`search_wikipedia` / `read_wikipedia`: call the Wikipedia API with httpx ([[agent/tools.py|def search_wikipedia(]]), modeled "
                       "on the Wikipedia actions the ReAct paper gave its model on HotpotQA (search, lookup and finish), here as search and read."},
                {"zh": "`run_python`：在 Docker 容器裡跑模型寫的程式，把印出來的東西和畫出來的圖傳回來（[[agent/tools.py|def run_python(]]）。",
                 "en": "`run_python`: runs the code the model writes in a Docker container and returns what it prints and the charts it draws ([[agent/tools.py|def run_python(]])."},
            ),
            Code("agent/tools.py", "def _eval(node):",
                 caption={"zh": "遞迴走過語法樹，遇到白名單以外的節點就丟 `ToolError`。",
                          "en": "Walks the syntax tree recursively and raises `ToolError` at any node outside the allowlist."}),
            {"zh": "`calculator` 只算得了一行算式；`run_python` 則讓模型寫完整的程式，所以環境要關得更緊。每次呼叫都從 `agent/sandbox/Dockerfile` 建的 image"
                   "（Python 3.12 加上 numpy、pandas，讀 Excel 的 openpyxl、印表格的 tabulate，還有畫圖的 matplotlib 和一套中文字型）開一個新容器：沒有網路，檔案系統唯讀、只有 `/tmp` 能寫，記憶體 512 MB、一顆 CPU、最多 64 個 process，"
                   "拿掉所有 Linux capability，用你自己的使用者執行、絕不用 root（[[agent/tools.py|def sandbox_user(]]：在 Linux 上，掛進容器的資料夾保留原本的權限，換成別的使用者可能進不去）；"
                   "這段對話的資料夾唯讀掛在 `/data`。程式從 stdin 送進去，跑完容器就刪掉。",
             "en": "`calculator` can only work out one expression; `run_python` lets the model write whole programs, so its environment is shut "
                   "tighter. Every call starts a new container from the image `agent/sandbox/Dockerfile` builds (Python 3.12 with numpy and pandas, openpyxl to read Excel, tabulate to print tables, and matplotlib to draw charts, with a font for Chinese labels): no "
                   "network, a read-only file system where only `/tmp` can be written, 512 MB of memory, one CPU, at most 64 processes, no Linux "
                   "capabilities, and your own user, never root ([[agent/tools.py|def sandbox_user(]]: on Linux a folder mounted in the container keeps its "
                   "permissions, which might not let another user in); the conversation's folder is mounted read-only at `/data`. The code goes in on stdin, "
                   "and the container is deleted when it is done."},
            Code("agent/tools.py", "def sandbox_command("),
            {"zh": "容器寫不進任何頁面讀得到的地方，所以圖跟著印出來的東西從 stdout 出來：image 裡 matplotlib 的 backend（`agent/sandbox/agent_loop_figures.py`）"
                   "在 `plt.show()` 或程式結束時把每張圖轉成 PNG，程式結束前接在印出來的東西後面：先一行 `run_python` 用 `AGENT_LOOP_FIGURES` 傳進去的記號，再每行一張 base64 的 PNG。"
                   "`run_python` 把它們從模型要讀的輸出切掉，存進這段對話的資料夾，叫 `figure-1.png`、`figure-2.png`…（[[agent/tools.py|def save_figures(]]），"
                   "回傳一個 `ToolResult`：模型讀到的是 `{\"stdout\": …, \"figures\": [\"/data/figure-1.png\"]}`，圖本身只給頁面，畫在聊天區和這一步的卡片上。",
             "en": "The container cannot write anywhere the page can read, so the charts come out on stdout with what the code prints: matplotlib's "
                   "backend in the image (`agent/sandbox/agent_loop_figures.py`) turns each figure into a PNG at `plt.show()` or when the program ends, and as it "
                   "exits writes them after the printed text: a line with the marker `run_python` passes in `AGENT_LOOP_FIGURES`, then one base64 PNG per line. "
                   "`run_python` cuts them off what the model reads, saves them in the conversation's folder as `figure-1.png`, `figure-2.png`… "
                   "([[agent/tools.py|def save_figures(]]) and returns a `ToolResult`: the model reads `{\"stdout\": …, \"figures\": [\"/data/figure-1.png\"]}`, "
                   "and the images themselves are for the page alone, drawn in the chat and on this step's card."},
            {"zh": "跑超過 30 秒就把容器停掉；結束代碼不是 0（例如丟出例外）、記憶體用太多或什麼都沒印也沒畫，都當成 `ToolError`，模型下一步讀到 traceback 或提醒，可以改了再跑。"
                   "Docker 不是必要的：只有側欄打開「讓模型寫程式」時才會提供 `run_python`，頁面也是那時候才去找 Docker（[[agent/tools.py|def sandbox_problem(]]），"
                   "沒裝、沒啟動、還沒建 image，或 image 比 `agent/sandbox/Dockerfile` 舊（看它的 `agent-loop-sandbox.version` label），就在側欄說明怎麼做，這一輪也不把 `run_python` 給模型。",
             "en": "Code that runs past 30 seconds has its container stopped; a nonzero exit code (an exception, say), too much memory or neither output nor a chart is a "
                   "`ToolError`, so the model reads the traceback or a reminder in the next step and can fix the code and run it again. Docker is "
                   "not required: `run_python` is offered only when Let the model write code is on in the sidebar, and only then does the page look "
                   "for Docker ([[agent/tools.py|def sandbox_problem(]]); if it is not installed, not running, or the image is not built or is older than "
                   "`agent/sandbox/Dockerfile` (going by its `agent-loop-sandbox.version` label), the sidebar says "
                   "what to do, and the turn goes on without `run_python`."},
            ),
            "pi": (
                {"zh": "pi 執行它自己的工具：read、bash、edit、write、grep、find、ls，側欄選模型看得到哪幾個（預設 read、bash、edit、write，和 pi 一樣）。"
                       "它們在這台電腦上、這段對話的資料夾裡執行，那也是 pi 的工作資料夾，沒有沙箱：不像 `run_python` 在沒有網路、唯讀的 Docker 容器裡，bash 碰得到你碰得到的一切。"
                       "pi 開始執行一個工具時，`PiHarness` 送出 `tool_call` 和 `tool_start`（要先問你時，等你允許才送 `tool_start`）；工具結束時送出 `tool_result`，帶著 pi 的工具傳回的文字。",
                 "en": "pi runs its own tools: read, bash, edit, write, grep, find and ls, of which the sidebar picks the ones the model sees (read, bash, edit and write by "
                       "default, as pi offers). They run on this computer, in the conversation's folder, which is pi's working folder, without a sandbox: unlike `run_python` "
                       "in a Docker container with no network and read-only files, bash reaches whatever you can. When pi starts a tool, `PiHarness` sends `tool_call` and "
                       "`tool_start` (with asking first, `tool_start` only once you allow it); when the tool ends, `tool_result`, with the text pi's tool returned."},
                Code("agent/harnesses/pi_harness.py", "def _tool_start(self"),
            ),
        }),
    )),
    Section("observe", "context",
            {"zh": "觀察接回脈絡，進下一圈", "en": "The observation goes back into the context for the next step"},
            {"zh": "工具結果變成一則 `tool` 訊息；下一圈 render 時，它就出現在模型的工具呼叫後面。",
             "en": "The tool result becomes a `tool` message; when the next step renders, it appears after the model's tool call."}, (
        Tabs({
            "simple": (
                {"zh": "工具結果（或錯誤）在 `_observe()` 裡轉成 JSON 字串（帶著圖的 `ToolResult` 先拆開：模型只讀它的資料，圖只放進事件給頁面），超過 `MAX_RESULT_CHARS`（2000 字元）就截斷並加上 `…(truncated)`，"
                   "免得一個長網頁塞爆脈絡；送出 `tool_result` 事件，再把字串回傳。",
             "en": "In `_observe()` the tool result (or error) becomes a JSON string (a `ToolResult` with charts is split first: the model reads only "
                   "its data, and the images go only into the event, for the page), cut at `MAX_RESULT_CHARS` (2000 characters) with "
                   "`…(truncated)` added, so one long web page cannot flood the context; the `tool_result` event goes out, and the string is returned."},
            Code("agent/harnesses/simple_harness.py", "def _observe("),
            {"zh": "回傳的字串變成一則 `role: tool` 的訊息 append 到 `self.messages`。同時更新連續格式錯誤的次數：這次是解析錯誤就加一，正常就歸零。",
             "en": "The returned string is appended to `self.messages` as a `role: tool` message. The count of format errors in a row is updated at "
                   "the same time: one more for a parse error, back to zero otherwise."},
            Code("agent/harnesses/simple_harness.py", "for cid, call in zip(ids, calls):", end="if format_errors >= MAX_FORMAT_ERRORS:"),
            ),
            "pi": (
                {"zh": "pi 把工具的結果接在它的呼叫後面，開始下一個 turn；`PiHarness` 在工具結束時送出 `tool_result`，帶著 pi 的工具傳回的文字（[[agent/harnesses/pi_harness.py|def _tool_end(self]]）。"
                       "下一個請求就帶著它，step 的請求把它標成新加入的部分。pi 問這個網頁的模型時，端點把它換成一則帶著工具名稱的 `tool` 訊息，交給 chat template 寫進脈絡，和下面說的一樣。",
                 "en": "pi puts the tool's result after its call and starts the next turn; `PiHarness` sends `tool_result` when the tool ends, with the text pi's tool "
                       "returned ([[agent/harnesses/pi_harness.py|def _tool_end(self]]). The next request carries it, and the step's request marks it as new. On this page's "
                       "model, the endpoint makes it a `tool` message with its tool's name, which the chat template writes into the context, as below."},
            ),
        }),
        {"zh": "然後回到迴圈開頭，下一圈 render 時（內建 harness 和 pi 問這個網頁的模型時都一樣）chat template 把工具結果寫進脈絡，這就是 step 卡片上「上一步的觀察接回脈絡」那一段：",
         "en": "Then it is back to the top of the loop, and when the next step renders (for the built-in harness and for pi on this page's model alike), the chat "
               "template writes the tool result into the context; this is what the step card shows as the last step's observation, back in the context:"},
        Tabs({
            "gemma4": (
                {"zh": "`<|tool_response>response:名稱{value:<|\"|>…<|\"|>}<tool_response|>`，緊接在模型的工具呼叫後面，"
                       "還在同一個 model turn 裡，模型讀到結果後接著寫。",
                 "en": "`<|tool_response>response:name{value:<|\"|>…<|\"|>}<tool_response|>`, right after the model's tool call and "
                       "still in the same model turn; the model reads the result and carries on writing."},
            ),
            "qwen3_5": (
                {"zh": "`<|im_start|>user` 加上 `<tool_response>` … `</tool_response>`，結果算是一則 user 訊息；"
                       "後面再開一個新的 `<|im_start|>assistant`，模型從新的回合開始寫。",
                 "en": "`<|im_start|>user` followed by `<tool_response>` … `</tool_response>`, so the result is a user message; a new "
                       "`<|im_start|>assistant` follows, and the model writes in a new turn."},
            ),
        }),
        {"zh": "不管哪一種，模型下一步可能再呼叫一次工具，也可能直接回答。",
         "en": "Either way, in the next step the model may call a tool again or answer right away."},
    )),
    Section("ui", "ui",
            {"zh": "畫面：事件怎麼變成你看到的東西", "en": "The page: how events become what you see"},
            {"zh": "背景 thread 只管 append 事件；畫面每 0.5 秒把整張列表重畫一次。",
             "en": "The background thread only appends events; the page redraws the whole list every 0.5 s."}, (
        {"zh": "`workspace` 是一個 fragment：有 agent 在跑時，它每 0.5 秒自己重跑一次，只重畫聊天和右邊的檢視，不用整頁重跑。"
               "跑完的那一刻再整頁重跑一次，然後停止輪詢。Streamlit 會留著這個 fragment，好讓它自己重跑，所以它只用弱參照拿著模型："
               "換模型或換「用什麼跑模型」時，`release_model()` 才放得掉它。",
         "en": "`workspace` is a fragment: while an agent runs, it reruns itself every 0.5 s and redraws only the chat and the views on the "
               "right, not the whole page. The moment the run ends, the whole page reruns once, and the polling stops. Streamlit keeps the "
               "fragment to rerun it on its own, so it holds the model only weakly: when another model, or what runs it, is picked, "
               "`release_model()` can let this one go."},
        Code("app/page.py", "@st.fragment(run_every=", end="viz_panel(llm)"),
        {"zh": "每次重畫，`LiveRun.data()` 複製一份目前的事件列表交給 `render.build()`，按順序折成一個個 step。"
               "`run_start` 告訴它這一輪是哪個模型、哪個 family，之後畫特殊 token、停止 token 和使用者訊息的位置都問這個 family；"
               "`step_start` 開一個新的 step，`context`、`prefill`、`token`、`parsed` 填進去（pi 跑的一輪還有 pi 送出的請求 `request`；pi 問它自己的模型時，沒有 family 也沒有前三個，換成串流進來的文字 `delta`），`model_end` 帶來的推測解碼結果標出哪些 token 是猜中的"
               "（伺服器到這時才說，{{mtp}}），工具相關的事件依 `call_id` 歸到各自的呼叫。"
               "之後所有 HTML 都是這個結構（和頁面語言）的純函式，所以執行到一半的任何時刻都畫得出來；"
               "切換語言也只是用另一種語言重畫同一份事件，跑完的對話也會跟著換。",
         "en": "On every redraw, `LiveRun.data()` copies the event list so far and hands it to `render.build()`, which folds it into steps in "
               "order. `run_start` says which model and family the turn uses, and from then on the special tokens, the stop tokens and where "
               "the user's message starts are all asked of that family; `step_start` opens a new step, `context`, `prefill`, `token` and "
               "`parsed` fill it in (a turn pi ran also has the request pi sent, `request`, and when pi asks one of its own models, no family and none of the first three, but `delta`, the text as it streams), `model_end` marks the tokens that were guesses when a server says so only then ({{mtp}}), and the tool events go to their calls by `call_id`. All the HTML after that is a pure function of this "
               "structure (and of the page's language), so it can be drawn at any moment of a run; switching the language just redraws the "
               "same events in the other one, finished turns included."},
        Code("app/render.py", "def build(events: list[dict])"),
        {"zh": "迴圈圖哪一格亮、哪條箭頭在流動，由最後一個事件決定：",
         "en": "Which box of the loop diagram lights up and which arrow flows is decided by the last event:"},
        Code("app/render.py", 'state["active"], state["edge"] = {'),
        {"zh": "一輪結束後，`absorb_finished()` 把 `run_end` 帶回來的完整訊息列表存成 `history`。下一個問題就接在它後面，"
               "所以 agent 記得前面聊過什麼，包括之前的工具結果。",
         "en": "When a turn ends, `absorb_finished()` keeps the full message list from `run_end` as `history`. The next question is appended "
               "to it, so the agent remembers what was said before, earlier tool results included."},
        Code("app/page.py", "def absorb_finished("),
        Tabs({
            "simple": (
                {"zh": "停止按鈕呼叫 `SimpleHarness.stop()`：設定 `_stop` 並叫醒正在等確認的 thread。解碼迴圈每個 token 前都會檢查 `_stop`，"
                       "所以按下去之後最多再生成一個 token 就停。",
                 "en": "The stop button calls `SimpleHarness.stop()`: it sets `_stop` and wakes a thread that is waiting for a confirmation. The decoding loop "
                       "checks `_stop` before every token, so after the press at most one more token is generated."},
                Code("agent/harnesses/simple_harness.py", "def stop(self):", end="def decide(self, call_id"),
            ),
            "pi": (
                {"zh": "停止按鈕呼叫 `PiHarness.stop()`，它把停止放進 `run()` 讀 pi 紀錄的那個佇列；`run()` 接著送給 pi `abort`，等著的確認一律當成被停止，"
                       "還在等自己那一步的請求也叫醒（[[agent/harnesses/pi_harness.py|def _abort(self, reason: str)]]）。pi 停下模型的請求和正在執行的工具；"
                       "pi 問這個網頁的模型時，端點在下一個 token 前就停，不屬於任何一步的請求則直接拒絕，不再 prefill。"
                       "pi 超過 `STOP_GRACE`（10 秒）還沒停下，就直接結束它。",
                 "en": "The stop button calls `PiHarness.stop()`, which puts the stop in the queue `run()` reads pi's records from; `run()` then sends pi `abort`, "
                       "takes every confirmation still waiting as stopped, and wakes a request still waiting for its step ([[agent/harnesses/pi_harness.py|def _abort(self, reason: str)]]). "
                       "pi stops the request to the model and the tool that runs; on this page's model, the endpoint stops before the next token, and turns away a request "
                       "that belongs to no step instead of prefilling it. A pi that has not settled after `STOP_GRACE` (10 s) is killed."},
                Code("agent/harnesses/pi_harness.py", "def _abort(self, reason: str):"),
            ),
        }),
    )),
]

# (event, file, text on the line that sends it, what it carries, what the page does with it)
EVENTS = [
    ("run_start", "agent/harnesses/simple_harness.py", '"run_start", run_id=',
     {"zh": "哪個 harness 跑的、模型和 family（pi 沒有）、這一輪的設定、可用的工具", "en": "which harness ran it, the model and family (none from pi), the turn's settings, the available tools"},
     {"zh": "用哪個模型的格式畫這一輪；步數上限、解碼說明", "en": "whose format to draw the turn in; the step limit, the decoding notes"}),
    ("step_start", "agent/harnesses/simple_harness.py", 'self.emit("step_start"',
     {"zh": "step 編號", "en": "the step number"},
     {"zh": "開一張新的 step 卡片；迴圈圖亮起「脈絡」", "en": "opens a new step card; Context lights up in the loop diagram"}),
    ("request", "agent/harnesses/pi_harness.py", 'self.emit("request"',
     {"zh": "pi 送給模型的請求，寫成 JSON（圖片只留類型）、這一步新加的部分在哪、各角色訊息數；只有 pi 送",
      "en": "the request pi sends the model, as JSON (an image as its type alone), where this step's new part is, the number of messages per role; only from pi"},
     {"zh": "pi 問 provider 的模型時，step 卡片的脈絡段；「脈絡」分頁", "en": "the step card's context section when pi asks a provider's model; the Context tab"}),
    ("context", "agent/harnesses/simple_harness.py", '"context",',
     {"zh": "模型讀到的完整 prompt（媒體已展開）、token 數、`added_from`、每種媒體佔的 token 數、脈絡上限、各角色訊息數",
      "en": "the whole prompt as the model reads it (media expanded), its token count, `added_from`, the tokens each kind of media takes, "
            "the context limit, the number of messages per role"},
     {"zh": "step 卡片的脈絡段、「脈絡」分頁", "en": "the step card's context section, the Context tab"}),
    ("prefill", "agent/model_backends/simple_backend.py", 'yield {"type": "prefill"',
     {"zh": "prompt 的 token 數、prefill 耗時", "en": "the prompt's token count, the prefill time"},
     {"zh": "迴圈圖「模型」格、模型分頁的 prefill ms", "en": "the Model box in the loop diagram, prefill ms in the Model tab"}),
    ("token", "agent/model_backends/simple_backend.py", '"type": "token",',
     {"zh": "id、piece、文字、channel、是否為特殊 token、機率、排名、前 5 名候選；開 MTP 時還有 `draft`（草稿猜了什麼、有沒有收下）。"
            "機率以後的欄位要 backend 知道才有：`simple_backend.py` 全都有，vLLM 伺服器有機率和前 5 名，其他看它加了哪些參數",
      "en": "id, piece, text, channel, whether it is special, probability, rank, the top 5 candidates; with MTP, also `draft` (what the drafter guessed and whether it was kept). "
            "The fields from the probability on are there only when the backend knows them: `simple_backend.py` knows them all, a vLLM server the probability and the top 5, and more as its flags allow"},
     {"zh": "token 的熱度顏色、底線與浮窗、左邊逐字出現的回覆", "en": "the token's heat color, underline and popover; the reply appearing bit by bit on the left"}),
    ("delta", "agent/harnesses/pi_harness.py", 'self.emit("delta"',
     {"zh": "模型輸出的一段文字和它的 channel（思考、回覆、工具呼叫）；只有看不到 token 的 harness 送，也就是 pi",
      "en": "a piece of the model's output and its channel (thinking, reply, tool call); only from a harness that sees no tokens, which is pi"},
     {"zh": "step 卡片模型段一段段出現的文字、左邊逐字出現的回覆", "en": "the text appearing in the step card's model section, the reply appearing bit by bit on the left"}),
    ("model_end", "agent/harnesses/simple_harness.py", 'self.emit("model_end"',
     {"zh": "停止原因、停止 token、token 數、tok/s、MTP 統計（vLLM 伺服器還可能說哪些 token 是猜中的）、原始輸出",
      "en": "why it stopped, the stop token, the token count, tok/s, the MTP counts (from a vLLM server, perhaps which tokens were guesses), the raw output"},
     {"zh": "「停在 …」、tok/s、MTP 收下幾個、模型分頁的表格", "en": "“stopped at …”, tok/s, how many MTP guesses were kept, the table in the Model tab"}),
    ("parsed", "agent/harnesses/simple_harness.py", 'self.emit("parsed"',
     {"zh": "思考、回覆、工具呼叫", "en": "the thinking, the reply, the tool calls"},
     {"zh": "Harness 段開始出現", "en": "the Harness section appears"}),
    ("tool_call", "agent/harnesses/simple_harness.py", '"tool_call",',
     {"zh": "名稱、驗證後的參數、錯誤、是否重複、環境", "en": "the name, the validated arguments, errors, whether it repeats, the environment"},
     {"zh": "Harness 段的呼叫、驗證結果、重複警告", "en": "the call, the validation result and the repeat warning in the Harness section"}),
    ("approval_request", "agent/harnesses/simple_harness.py", 'self.emit("approval_request"',
     {"zh": "等待確認的呼叫", "en": "the call waiting for a confirmation"},
     {"zh": "聊天區的「允許／拒絕」按鈕；迴圈圖閃爍", "en": "the Allow and Reject buttons in the chat; the loop diagram pulses"}),
    ("approval", "agent/harnesses/simple_harness.py", 'self.emit("approval",',
     {"zh": "允許、拒絕、逾時或被停止", "en": "allowed, rejected, timed out or stopped"},
     {"zh": "Harness 段顯示你的決定", "en": "the Harness section shows your decision"}),
    ("tool_start", "agent/harnesses/simple_harness.py", 'self.emit("tool_start"',
     {"zh": "工具名稱、環境", "en": "the tool name, the environment"},
     {"zh": "迴圈圖亮起「環境」", "en": "Environment lights up in the loop diagram"}),
    ("tool_result", "agent/harnesses/simple_harness.py", 'self.emit("tool_result"',
     {"zh": "結果字串、成功與否、是否截斷、耗時、給你看的圖", "en": "the result string, whether it succeeded, whether it was cut, the time it took, the images for you"},
     {"zh": "環境段的結果、聊天區的工具標籤；圖畫在兩邊", "en": "the result in the Environment section, the tool chip in the chat; the charts in both"}),
    ("error", "agent/harnesses/simple_harness.py", 'self.emit("error"',
     {"zh": "例外的類型和訊息；backend 的錯誤兩種語言都有（`text`）", "en": "the exception's type and message, and both languages of a backend's error (`text`)"},
     {"zh": "聊天區和停止卡片的錯誤訊息", "en": "the error message in the chat and on the stop card"}),
    ("run_end", "agent/harnesses/simple_harness.py", 'self.emit("run_end"',
     {"zh": "停止原因、最終回覆、整份訊息列表", "en": "the stop reason, the final reply, the whole message list"},
     {"zh": "停止卡片；訊息列表成為下一輪的 `history`", "en": "the stop card; the message list becomes the next turn's `history`"}),
]


if __name__ == "__main__":
    print(*build(Path(sys.argv[1] if len(sys.argv) > 1 else "_site")), sep="\n")
