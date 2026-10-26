"""The 運作說明 page. Its line numbers are looked up from the code by text, so these tests are
what notice when an edit moves or removes a line the guide talks about; guide_excerpts.txt keeps
the code the page quotes, so they also notice an edit to that code, which may make the text around
it untrue. It comes in Chinese and in English; both are written side by side, and these tests also
check they say the same things. They need pygments and pytest and nothing else of the app's, since
the Pages workflow runs them before it builds the page."""

import ast
import difflib
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from app import guide
from app.guide import Code, Tabs, span, statement_end
from harness.i18n import LANGS
from html_check import assert_balanced

CODES = [item for sec in guide.SECTIONS for item in guide.parts(sec.body) if isinstance(item, Code)]
PROSE = [guide.INTRO] + [text for sec in guide.SECTIONS for text in (sec.title, sec.lead, *guide.parts(sec.body)) if not isinstance(text, Code)]
TAB_SETS = [item for sec in guide.SECTIONS for item in sec.body if isinstance(item, Tabs)]
CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")


def bilingual():
    """Every piece of text on the page, as its {"zh": ..., "en": ...} pair."""
    for item in PROSE:
        yield from (item if isinstance(item, tuple) else (item,))
    yield from (code.caption for code in CODES if code.caption)
    for _, _, _, carries, shows in guide.EVENTS:
        yield from (carries, shows)
    for title, tabs in guide.TAB_KINDS.values():
        yield title
        yield from (label for label, _ in tabs.values())


def paragraphs():
    for pair in bilingual():
        yield from pair.values()


SNAPSHOT = Path(__file__).with_name("guide_excerpts.txt")
SNAPSHOT_HEAD = """\
# The code the 運作說明 page quotes, section by section: every excerpt as the page shows it and the
# whole line behind every [[file|text]] chip, without line numbers. tests/test_guide.py compares it
# with the code, so an edit to code the page quotes fails until someone has reread what the page
# says about it. Reread the sections the failure names, fix their text in guide.py if the behavior
# changed, then write this file again with
#     UPDATE_GUIDE_SNAPSHOT=1 uv run pytest tests/test_guide.py
"""


def chip(file: str, text: str) -> str:
    return f"[[{file}|{text}]]\n    {guide.source(file)[0][guide.find_line(file, text) - 1].strip()}"


def quoted(items) -> str:
    """The code that `items` (a section's paragraphs, lists and excerpts) quote, in reading order."""
    entries = []
    for item in items:
        if isinstance(item, Code):
            lo, hi = span(item)
            excerpt = textwrap.dedent("\n".join(guide.source(item.file)[0][lo - 1:hi]))
            entries.append(f"{item.file}: {item.start}{f' … {item.end}' if item.end else ''}\n{textwrap.indent(excerpt, '    ')}")
            item = item.caption or ()
        for text in item if isinstance(item, tuple) else (item,):
            entries += [chip(file, ref) for file, ref in guide.REF_RE.findall(text["en"])]
    return "\n\n".join(dict.fromkeys(entries))


def quoted_sections() -> dict[str, str]:
    """What each part of the page quotes, by section key; "intro" and "events" are the introduction
    and the event table. Line numbers are left out, so an edit elsewhere in a file changes nothing."""
    blocks = {"intro": quoted([guide.INTRO])}
    blocks |= {sec.key: quoted([sec.title, sec.lead, *guide.parts(sec.body)]) for sec in guide.SECTIONS}
    blocks["events"] = "\n\n".join(chip(file, text) for _, file, text, _, _ in guide.EVENTS)
    return {key: block for key, block in blocks.items() if block}


def snapshot_text(blocks: dict[str, str]) -> str:
    return SNAPSHOT_HEAD + "".join(f"\n== {key} ==\n\n{block}\n" for key, block in blocks.items())


def changed_sections(was: str, now: dict[str, str]) -> list[str]:
    parts = re.split(r"^== (\w+) ==$", was, flags=re.M)[1:]
    before = {key: block.strip("\n") for key, block in zip(parts[::2], parts[1::2])}
    return [key for key in dict.fromkeys([*before, *now]) if before.get(key) != now.get(key)]


def section_name(key: str) -> str:
    numbered = {sec.key: (n, sec) for n, sec in enumerate(guide.SECTIONS, 1)}
    if key in numbered:
        n, sec = numbered[key]
        return f'§{n} “{sec.title["en"]}” (Section("{key}", …) in guide.py)'
    return {"intro": "the introduction (INTRO in guide.py)", "events": "the event table (EVENTS in guide.py)"}.get(key, key)


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


def test_the_quoted_code_has_not_changed_since_the_guide_was_reread():
    now = quoted_sections()
    text = snapshot_text(now)
    if os.environ.get("UPDATE_GUIDE_SNAPSHOT") == "1":
        SNAPSHOT.write_text(text, encoding="utf-8")
    was = SNAPSHOT.read_text(encoding="utf-8") if SNAPSHOT.exists() else ""
    if was != text:
        names = "\n".join(f"    {section_name(key)}" for key in changed_sections(was, now)) or "    the file's header"
        diff = "".join(difflib.unified_diff(was.splitlines(True), text.splitlines(True), "tests/guide_excerpts.txt", "the code now"))
        pytest.fail(
            f"The code the guide quotes has changed in\n{names}\n"
            "Reread what those sections say about it and fix their text in guide.py if the behavior changed, then run\n"
            "    UPDATE_GUIDE_SNAPSHOT=1 uv run pytest tests/test_guide.py\n"
            f"and commit tests/guide_excerpts.txt with the change.\n\n{diff}",
            pytrace=False,
        )


def test_the_snapshot_follows_the_quoted_code_not_its_line_numbers(tmp_path, monkeypatch):
    files = {code.file for code in CODES} | {e[1] for e in guide.EVENTS}
    files |= {file for text in paragraphs() for file, _ in guide.REF_RE.findall(text)}
    for file in files:
        (tmp_path / file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(guide.ROOT / file, tmp_path / file)
    monkeypatch.setattr(guide, "ROOT", tmp_path)
    before = quoted_sections()
    agent_py = tmp_path / "harness/agent.py"
    agent_py.write_text("# a line that moves every other one down\n" + agent_py.read_text(encoding="utf-8"), encoding="utf-8")
    assert quoted_sections() == before
    code = next(code for code in CODES if code.file == "harness/agent.py")
    lines = agent_py.read_text(encoding="utf-8").splitlines()
    lines[span(code)[0] - 1] += "  # edited"
    agent_py.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sec = next(sec for sec in guide.SECTIONS if code in guide.parts(sec.body))
    assert sec.key in changed_sections(snapshot_text(before), quoted_sections())


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
    for file in ("harness/agent.py", "harness/backends/simple_backend.py"):
        for node in ast.walk(guide.source(file)[1]):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "emit" and isinstance(node.args[0], ast.Constant):
                events.add(node.args[0].value)
    events |= {"prefill", "token"}  # yielded by SimpleBackend.generate and forwarded as they are
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


def test_every_set_of_tabs_is_one_tab_per_backend_or_per_model():
    kinds = [set(tabs) for _, tabs in guide.TAB_KINDS.values()]
    assert not set.intersection(*kinds)  # a key says which kind its set of tabs is
    assert {tabs.kind for tabs in TAB_SETS} == set(guide.TAB_KINDS)
    for tabs in TAB_SETS:
        assert 2 <= len(tabs.panels) <= guide.MAX_TABS, list(tabs.panels)  # guide.css hides the panels not picked up to MAX_TABS
        assert set(tabs.panels) <= set(guide.TAB_KINDS[tabs.kind][1]), list(tabs.panels)
        for body in tabs.panels.values():
            assert body and not any(isinstance(item, Tabs) for item in body)
    files = next(sec for sec in guide.SECTIONS if sec.key == "files")
    assert [list(t.panels) for t in files.body if isinstance(t, Tabs)] == [
        ["transformers", "vllm-docker", "vllm-url"], ["transformers", "vllm"], ["gemma4", "qwen3_5", "gemma4-drafter"]]


@pytest.mark.parametrize("lang", LANGS)
def test_tabs_are_radio_buttons_that_switch_without_a_script(lang):
    html = guide.page_html("0123456789abcdef", lang)
    sets = re.findall(r'<div class="g-tabs" data-kind="(\w+)">(.*?)</div><div class="g-panel">', html, re.S)
    assert len(sets) == len(TAB_SETS)
    names = []
    for (kind, bar), tabs in zip(sets, TAB_SETS):
        title, labels = guide.TAB_KINDS[kind]
        assert kind == tabs.kind and f'role="radiogroup" aria-label="{title[lang]}"' in bar
        radios = re.findall(r'<input type="radio" name="([\w-]+)" value="([\w-]+)"( checked)?>', bar)
        assert [value for _, value, _ in radios] == list(tabs.panels)
        assert [checked for _, _, checked in radios] == [" checked"] + [""] * (len(radios) - 1)  # the first tab shows until one is picked
        assert len({name for name, _, _ in radios}) == 1
        names.append(radios[0][0])
        for key in tabs.panels:
            label, small = labels[key]
            assert f"<span>{guide.esc(label[lang])}</span><small>{guide.esc(small)}</small>" in bar
    assert len(set(names)) == len(names)  # each set is a radio group of its own
    assert html.count('class="g-panel"') == sum(len(t.panels) for t in TAB_SETS)
    css = guide.stylesheet()
    for n in range(1, guide.MAX_TABS + 1):  # the rule that shows only the picked tab's panel
        assert f"label:nth-child({n}) > input:not(:checked)) > .g-panel:nth-child({n + 1})" in css
    js = (guide.ASSETS / "guide.js").read_text(encoding="utf-8")
    assert 'localStorage.setItem("tab-" + kind' in js and 'localStorage.getItem("tab-" + kind)' in js  # each kind's pick holds across the page and visits


def test_an_excerpt_of_a_file_other_than_python_ends_where_it_says():
    code = Code("docker-compose/compose.yaml", "- ${AGENT_LOOP_MODELS:-../models}:/models:ro", end='HF_HUB_OFFLINE: "1"')
    lo, hi = span(code)
    lines = guide.source(code.file)[0]
    assert guide.source(code.file)[1] is None
    assert 'HF_HUB_OFFLINE: "1"' in lines[hi - 1] and hi - lo < 10
    assert span(Code(code.file, code.start)) == (lo, lo)
    assert '<span class="nt">' in guide.code_html(code, "en")  # highlighted as YAML


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
    html = guide.prose_html("a <b> `x<y>` **粗** [[harness/agent.py|def run(self):]] {{stop}}", "zh")
    assert html.startswith("a &lt;b&gt; <code>x&lt;y&gt;</code> <b>粗</b> ")
    assert f'>harness/agent.py:{guide.find_line("harness/agent.py", "def run(self):")}</code>' in html
    assert 'title="def run(self):"' in html and 'href="#g-stop"' in html


def test_find_line_wants_exactly_one_match():
    with pytest.raises(LookupError, match="0 lines"):
        guide.find_line("harness/agent.py", "no such text anywhere")
    with pytest.raises(LookupError, match="lines, expected exactly 1"):
        guide.find_line("harness/agent.py", "self.emit(")
    with pytest.raises(LookupError):
        span(Code("harness/agent.py", "def run(self):", end="no such text anywhere"))


def test_statement_end():
    tree = ast.parse("def f():\n    x = g(\n        1,\n    )\n    return x\ny = 2\n")
    assert statement_end(tree, 1) == 5  # a block: through its last line
    assert statement_end(tree, 2) == 4  # a call over several lines
    assert statement_end(tree, 3) == 4  # the middle of that call
    assert statement_end(tree, 6) == 6
