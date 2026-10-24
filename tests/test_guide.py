"""The 運作說明 page. Its line numbers are looked up from the code by text, so these tests are
what notice when an edit moves or removes a line the guide talks about."""

import ast
import re

import pytest

import guide
import render
from guide import Code, span, statement_end
from test_render import assert_balanced

CODES = [item for sec in guide.SECTIONS for item in sec.body if isinstance(item, Code)]
PROSE = [guide.INTRO] + [text for sec in guide.SECTIONS for text in (sec.lead, *sec.body) if not isinstance(text, Code)]


def paragraphs():
    for item in PROSE:
        yield from (item if isinstance(item, tuple) else (item,))


@pytest.mark.parametrize("code", CODES, ids=lambda c: f"{c.file}:{c.start[:30]}")
def test_every_excerpt_starts_and_ends_where_it_says(code):
    lines, _ = guide.source(code.file)
    lo, hi = span(code)
    assert code.start in lines[lo - 1]
    assert lo <= hi
    if code.end:
        assert any(code.end in line for line in lines[lo - 1:hi])
    excerpt = "\n".join(lines[lo - 1:hi])
    html = guide.code_html(code)
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
    assert guide.xref_html("record").startswith('<a class="g-xref" href="#g-record">第 ')


def test_page_html():
    html = guide.page_html(render.diagram_html(None, dark=False, counter="每一圈 = 一個 step"))
    assert_balanced(html)
    assert html.count('class="g-sec"') == len(guide.SECTIONS) + 1  # plus the event table
    for sec in guide.SECTIONS:
        assert f'id="g-{sec.key}"' in html and f'href="#g-{sec.key}"' in html
    # Code excerpts and the line shown on hovering a chip may contain [[ or ** themselves.
    prose = re.sub(r"<[^>]+>", "", re.sub(r'<figure class="src">.*?</figure>', "", html, flags=re.S))
    assert not guide.REF_RE.search(prose) and "{{" not in prose and "**" not in prose


def test_prose_markup_is_escaped():
    html = guide.prose_html("a <b> `x<y>` **粗** [[agent.py|def run(self):]] {{stop}}")
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
