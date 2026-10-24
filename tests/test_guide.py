"""The 運作說明 page. Its line numbers are looked up from the code by text, so these tests are
what notice when an edit moves or removes a line the guide talks about. It comes in Chinese and
in English; both are written side by side, and these tests also check they say the same things."""

import ast
import re
import subprocess

import pytest

import guide
from guide import Code, span, statement_end
from i18n import LANGS
from test_render import assert_balanced

CODES = [item for sec in guide.SECTIONS for item in sec.body if isinstance(item, Code)]
PROSE = [guide.INTRO] + [text for sec in guide.SECTIONS for text in (sec.title, sec.lead, *sec.body) if not isinstance(text, Code)]
CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")


def bilingual():
    """Every piece of text on the page, as its {"zh": ..., "en": ...} pair."""
    for item in PROSE:
        yield from (item if isinstance(item, tuple) else (item,))
    yield from (code.caption for code in CODES if code.caption)
    for _, _, _, carries, shows in guide.EVENTS:
        yield from (carries, shows)


def paragraphs():
    for pair in bilingual():
        yield from pair.values()


@pytest.mark.parametrize("code", CODES, ids=lambda c: f"{c.file}:{c.start[:30]}")
def test_every_excerpt_starts_and_ends_where_it_says(code):
    lines, _ = guide.source(code.file)
    lo, hi = span(code)
    assert code.start in lines[lo - 1]
    assert lo <= hi
    if code.end:
        assert any(code.end in line for line in lines[lo - 1:hi])
    excerpt = "\n".join(lines[lo - 1:hi])
    html = guide.code_html(code, "zh")
    assert html.count('class="ln"') == hi - lo + 1
    assert f'<span class="ln">{lo}</span>' in html and f'<span class="ln">{hi}</span>' in html
    assert len(excerpt) > 0
    assert_balanced(html)


def test_every_inline_reference_resolves():
    refs = [m.groups() for text in paragraphs() for m in guide.REF_RE.finditer(text)]
    assert len(refs) > 20
    for file, text in refs:
        line = guide.find_line(file, text)
        assert text in guide.source(file)[0][line - 1]


@pytest.mark.parametrize("kind, file, text, _carries, _shows", guide.EVENTS, ids=[e[0] for e in guide.EVENTS])
def test_event_table_points_at_the_line_that_sends_it(kind, file, text, _carries, _shows):
    line = guide.source(file)[0][guide.find_line(file, text) - 1]
    assert f'"{kind}"' in line


def test_event_table_covers_every_event_the_harness_sends():
    events = set()
    for file in ("agent.py", "llm.py"):
        for node in ast.walk(guide.source(file)[1]):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "emit" and isinstance(node.args[0], ast.Constant):
                events.add(node.args[0].value)
    events |= {"prefill", "token"}  # yielded by llm.generate and forwarded as they are
    assert events == {e[0] for e in guide.EVENTS}


def test_section_links_point_at_sections():
    keys = [sec.key for sec in guide.SECTIONS]
    assert len(keys) == len(set(keys))
    for text in paragraphs():
        for key in guide.re.findall(r"\{\{(\w+)\}\}", text):
            assert key in keys, key
    n = [sec.key for sec in guide.SECTIONS].index("record") + 1
    assert guide.xref_html("record", "zh").startswith(f'<a class="g-xref" href="#g-record">第 {n} 段「')
    assert guide.xref_html("record", "en").startswith(f'<a class="g-xref" href="#g-record">§{n} “Writing the tool call')


def test_both_languages_say_the_same_things():
    """A paragraph points at the same lines and the same sections in either language."""
    pairs = list(bilingual())
    assert len(pairs) > 150
    for pair in pairs:
        assert set(pair) == set(LANGS) and all(pair.values()), pair
        zh, en = pair["zh"], pair["en"]
        assert guide.REF_RE.findall(zh) == guide.REF_RE.findall(en), en
        assert re.findall(r"\{\{(\w+)\}\}", zh) == re.findall(r"\{\{(\w+)\}\}", en), en
        assert not CJK.search(en), en


@pytest.mark.parametrize("lang", LANGS)
def test_page_html(lang):
    html = guide.page_html("0123456789abcdef", lang)
    assert_balanced(html)
    assert html.count('class="g-sec"') == len(guide.SECTIONS) + 1  # plus the event table
    for sec in guide.SECTIONS:
        assert f'id="g-{sec.key}"' in html and f'href="#g-{sec.key}"' in html  # the same anchors, so a switch keeps your place
    contents = {"zh": "目錄", "en": "Contents"}[lang]
    assert f'<nav class="g-toc" aria-label="{contents}"><details><summary>' in html  # closed until you open it
    assert html.count("<li ") == len(guide.SECTIONS) + 1
    assert html.count('<img class="diagram"') == 2  # one per theme
    assert f'href="{guide.REPO}/tree/0123456789abcdef"><code>0123456</code>' in html
    assert '<a href="../zh/" hreflang="zh-Hant" lang="zh-Hant" data-lang="zh"' in html and 'data-lang="en"' in html
    assert html.count('aria-current="page"') == 1 and f'data-lang="{lang}" aria-current="page"' in html
    # Code excerpts and the line shown on hovering a chip may contain [[ or ** themselves.
    prose = re.sub(r"<[^>]+>", "", re.sub(r'<figure class="src">.*?</figure>', "", html, flags=re.S))
    assert not guide.REF_RE.search(prose) and "{{" not in prose and "**" not in prose
    if lang == "en":  # the code is the same on both pages; everything around it is in English
        words = prose.replace("中文", "").replace("＃", "")  # the switch's other language, the events' number mark
        assert not CJK.search(words), words[CJK.search(words).start() - 80:][:160]


def test_stylesheet_has_both_palettes():
    css = guide.stylesheet()
    assert css.startswith(":root {") and "--c-harness: #6b4c8a;" in css  # light, from style.css
    dark = css[css.index("@media (prefers-color-scheme: dark)"):]
    assert "--c-harness: #b9a0e0;" in dark  # dark, from dark.css
    assert ".g-toc" in css and "stMainBlockContainer" not in css  # the page's rules, none of the app's
    assert css.count("{") == css.count("}")


def test_build(tmp_path):
    site = tmp_path / "site"
    assert guide.build(site) == [site / "index.html", site / "zh" / "index.html", site / "en" / "index.html"]
    head = '<!doctype html><html lang="{}">'
    for lang, title in (("zh", "運作說明 · Agent Loop 可視化"), ("en", "How it works · Agent Loop Visualizer")):
        doc = (site / lang / "index.html").read_text(encoding="utf-8")
        assert doc.startswith(head.format(guide.HTML_LANG[lang])) and f"<title>{title}</title>" in doc
        assert_balanced(doc)
        assert doc.count("<script>") == 1 and doc.index("<script>") > doc.index('id="g-events"')  # guide.js, after what it works on
        assert 'localStorage.setItem("lang", a.dataset.lang)' in doc  # the switch remembers the pick for index.html
        commit = re.search(r'/tree/([0-9a-f]{40})"', doc).group(1)
        assert commit == subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def test_the_root_page_sends_a_reader_to_their_language(tmp_path):
    guide.build(tmp_path)
    root = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert_balanced(root)
    assert 'localStorage.getItem("lang")' in root and "navigator.language" in root  # their last pick, else their browser's
    assert 'location.replace(lang + "/" + location.hash)' in root  # a link to a section keeps working
    assert '<a href="zh/">' in root and '<a href="en/">' in root  # without JavaScript, both are a click away


def test_prose_markup_is_escaped():
    html = guide.prose_html("a <b> `x<y>` **粗** [[agent.py|def run(self):]] {{stop}}", "zh")
    assert html.startswith("a &lt;b&gt; <code>x&lt;y&gt;</code> <b>粗</b> ")
    assert f'>agent.py:{guide.find_line("agent.py", "def run(self):")}</code>' in html
    assert 'title="def run(self):"' in html and 'href="#g-stop"' in html


def test_find_line_wants_exactly_one_match():
    with pytest.raises(LookupError, match="0 lines"):
        guide.find_line("agent.py", "no such text anywhere")
    with pytest.raises(LookupError, match="lines, expected exactly 1"):
        guide.find_line("agent.py", "self.emit(")
    with pytest.raises(LookupError):
        span(Code("agent.py", "def run(self):", end="no such text anywhere"))


def test_statement_end():
    tree = ast.parse("def f():\n    x = g(\n        1,\n    )\n    return x\ny = 2\n")
    assert statement_end(tree, 1) == 5  # a block: through its last line
    assert statement_end(tree, 2) == 4  # a call over several lines
    assert statement_end(tree, 3) == 4  # the middle of that call
    assert statement_end(tree, 6) == 6
