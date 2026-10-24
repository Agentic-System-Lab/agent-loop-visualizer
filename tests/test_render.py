"""render.py turns the event list into HTML. The page redraws every half second while
a run is live, so every function here must work on any prefix of a run's events."""

import base64
import re
import threading
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import pytest

import agent
import families
import render
from agent import Run, RunSettings
from families import GEMMA4, QWEN3_5
from fakes import DRAFTER, INFO, QWEN_INFO, FakeLLM, qwen_tool_call, record_run, tool_call
from i18n import LANGS
from llm import Sampling
from tools import TOOLS

ANSWER = "現在是下午三點。<turn|>"
VOID = {"br", "img", "hr", "input", "meta", "link", "wbr"}


class TagChecker(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.problems = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.problems.append(f"</{tag}> while open: {self.stack}")
        else:
            self.stack.pop()


def assert_balanced(markup: str):
    checker = TagChecker()
    checker.feed(markup)
    checker.close()
    assert checker.problems == [] and checker.stack == [], (checker.problems, checker.stack)


def svg_of(diagram: str) -> str:
    src = re.search(r'src="data:image/svg\+xml;base64,([^"]+)"', diagram).group(1)
    return base64.b64decode(src).decode()


def approved_run_events():
    """A thinking run that asks for approval, gets it, calls a tool and answers."""
    fake = FakeLLM(["要查時間<channel|>" + tool_call("get_current_time"), "好<channel|>" + ANSWER])
    events = []
    run = Run(fake, [], "台北現在幾點？", RunSettings(thinking=True, approve_tools=True), events.append)
    thread = threading.Thread(target=run.run, daemon=True)
    thread.start()
    for _ in range(500):
        if any(e["type"] == "approval_request" for e in events):
            break
        threading.Event().wait(0.01)
    run.decide("call_1_0", True)
    thread.join(5)
    return events


SCENARIOS = {
    "reply": lambda: record_run([ANSWER])[1],
    "tool": lambda: record_run([tool_call("calculator", expression="6*7"), "答案是 42。<turn|>"])[1],
    "approved": approved_run_events,
    "invalid_and_repeated": lambda: record_run([tool_call("calculator"), tool_call("calculator"), ANSWER])[1],
    "max_steps": lambda: record_run([tool_call("get_current_time")] * 3, max_steps=2)[1],
    "token_limit": lambda: record_run(["很長的回覆" * 5], sampling=Sampling(max_new_tokens=6))[1],
    "error": lambda: record_run([RuntimeError("boom")])[1],
    "mtp": lambda: record_run([tool_call("calculator", expression="6*7"), "答案是 42。<turn|>"], sampling=Sampling(mtp=True))[1],
    "qwen_tool": lambda: record_run(
        [qwen_tool_call("calculator", expression="6*7"), "答案是 42。<|im_end|>"], family=QWEN3_5)[1],
    "qwen_thinking": lambda: record_run(
        ["要算一下\n</think>\n\n" + qwen_tool_call("calculator", expression="1"), "想好了\n</think>\n\n好。<|im_end|>"],
        family=QWEN3_5, thinking=True)[1],
}


def render_everything(data: dict, live: bool, lang: str = "zh") -> list[str]:
    """The HTML the Streamlit app draws for one refresh (st.html). The plain-text
    pieces (status, draft, footer, captions) go through st.caption / st.markdown."""
    out = [
        render.diagram_html(data, dark=False, lang=lang),
        render.diagram_html(data, dark=True, lang=lang),
        render.trace_html(data, lang=lang),
    ]
    render.draft(data), render.pending_approvals(data)
    for step in data["steps"]:
        out.append(render.step_html(step, data, live=live and step is data["steps"][-1], lang=lang))
        if step["context"]:
            out.append(render.context_html(step, data["family"], lang=lang))
    if data["end"]:
        out.append(render.stop_html(data, lang=lang))
    if data["settings"]:
        out.append(render.knobs_html(data["settings"], INFO["defaults"], lang=lang))
        out.append(render.tools_html(TOOLS, data["settings"]["use_tools"], data["family"], lang=lang))
    return [piece for piece in out if piece]


def captions_of(data: dict, lang: str) -> list[str]:
    """The plain text the app shows through st.caption, and the column names of the step table."""
    texts = [render.run_stats(data, lang=lang), render.status(data, lang=lang)]
    texts += [render.context_summary(step, lang=lang) for step in data["steps"] if step["context"]]
    texts += [render.footer(data, lang=lang)] if data["end"] else []
    return texts + [key for row in render.step_rows(data, lang=lang) for key in row]


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_prefix_of_a_run_renders(scenario, lang):
    events = SCENARIOS[scenario]()
    assert events[-1]["type"] == "run_end"
    for n in range(len(events) + 1):
        data = render.build(events[:n])
        markup = render_everything(data, live=n < len(events), lang=lang)
        captions_of(data, lang)
        for piece in markup:
            assert_balanced(piece)
        ET.fromstring(svg_of(markup[0]))  # the diagram is valid SVG at every moment
        ET.fromstring(svg_of(markup[1]))


# ---------- the page in English ----------

CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")
EN_SCENARIOS = {
    "tool": lambda: record_run([tool_call("calculator", expression="6*7"), "It is 42.<turn|>"], user="What is 6 times 7?", lang="en")[1],
    "invalid_and_repeated": lambda: record_run(
        [tool_call("calculator"), tool_call("calculator"), "Done.<turn|>"], user="Add it up.", lang="en")[1],
    "max_steps": lambda: record_run([tool_call("get_current_time")] * 3, user="Time?", max_steps=2, lang="en")[1],
    "token_limit": lambda: record_run(["A long reply"], user="Hi", sampling=Sampling(max_new_tokens=6), lang="en")[1],
    "error": lambda: record_run([RuntimeError("boom")], user="Hi", lang="en")[1],
    "mtp": lambda: record_run([tool_call("calculator", expression="6*7"), "It is 42.<turn|>"], user="6*7?",
                              sampling=Sampling(mtp=True), lang="en")[1],
    "qwen_tool": lambda: record_run(
        [qwen_tool_call("calculator", expression="6*7"), "It is 42.<|im_end|>"], user="6*7?", family=QWEN3_5, lang="en")[1],
}


@pytest.mark.parametrize("scenario", EN_SCENARIOS)
def test_an_english_page_has_no_chinese_left(scenario):
    """Every word on an English page is English. The full prompt in the context tab is the model's own
    text (its system prompt asks for Traditional Chinese replies in Chinese), so it is left out."""
    events = EN_SCENARIOS[scenario]()
    for n in range(len(events) + 1):
        data = render.build(events[:n])
        for piece in render_everything(data, live=n < len(events), lang="en") + captions_of(data, "en"):
            text = re.sub(r'<pre class="ctx-view">.*?</pre>', "", piece, flags=re.S)
            assert not CJK.search(text), CJK.search(text).string[max(CJK.search(text).start() - 60, 0):CJK.search(text).end() + 20]


@pytest.mark.parametrize("info", [INFO, {**INFO, "drafter_id": DRAFTER["model_id"]}, {**INFO, "drafter_id": DRAFTER["model_id"], "drafter": DRAFTER}, QWEN_INFO])
def test_the_model_tab_in_english_has_no_chinese(info):
    family = families.get(info["family"])
    pieces = [
        render.model_html(info, lang="en"), render.special_tokens_html(info, lang="en"), render.INTRO["en"], render.CONTEXT_LEGEND["en"],
        render.tools_html(TOOLS, True, family, lang="en"), render.tools_html(TOOLS, False, family, lang="en"),
        render.knobs_html({"temperature": 0, "top_k": 0, "top_p": 0.9, "max_new_tokens": 512, "max_steps": 4, "mtp": True}, INFO["defaults"], lang="en"),
        render.diagram_html(None, dark=False, lang="en"),
    ]
    for piece in pieces:
        assert not CJK.search(piece), piece[:200]
    assert "<b>Start of turn</b>" in render.special_tokens_html(INFO, lang="en")
    assert "Special tokens (24)" in render.special_tokens_html(INFO, lang="en")


def test_switching_language_redraws_a_finished_run():
    """The events do not depend on the page's language; only the drawing does."""
    data = render.build(SCENARIOS["tool"]())
    zh, en = (render.stop_html(data, lang=lang) for lang in ("zh", "en"))
    assert "停止：模型沒有再呼叫工具" in zh and "Stop: no more tool calls" in en
    assert "It stopped at " + render.chip("<turn|>", GEMMA4, lang="en") + " (end of turn, like end_turn)." in en
    assert re.fullmatch(r"2 steps · .+ · stop: no more tool calls", render.footer(data, lang="en"))
    assert render.loop_state(data, lang="en")["counter"] == "done · 2 steps"


# ---------- build: events -> steps ----------

def test_build_of_nothing():
    data = render.build([])
    assert data == {"model": None, "family": None, "settings": {}, "steps": [], "end": None, "error": None, "last": None, "t": 0}


def test_build_folds_a_tool_run_into_steps():
    _, events, _ = record_run([tool_call("calculator", expression="6*7"), "答案是 42。<turn|>"])
    data = render.build(events)
    assert data["settings"]["max_steps"] == 6
    assert [s["n"] for s in data["steps"]] == [1, 2]
    one, two = data["steps"]
    assert one["context"]["step"] == 1 and one["prefill"] and one["done"] and one["parsed"]
    assert len(one["tokens"]) == one["done"]["n_tokens"]
    [call] = one["calls"].values()
    assert call["name"] == "calculator" and call["started"] is True and call["approval"] is None
    assert call["result"]["ok"] is True
    assert two["calls"] == {}
    assert one["t0"] <= one["t1"] <= two["t0"]
    assert data["end"]["stop_reason"] == "no_tool_call"
    assert data["last"] == "run_end" and data["t"] == events[-1]["t"]


def test_build_tracks_approval():
    events = approved_run_events()
    i = next(i for i, e in enumerate(events) if e["type"] == "approval_request")
    waiting = render.build(events[: i + 1])
    [pending] = render.pending_approvals(waiting)
    assert pending["call_id"] == "call_1_0" and pending["step"] == 1
    done = render.build(events)
    assert render.pending_approvals(done) == []
    assert list(done["steps"][0]["calls"].values())[0]["approval"] == "approved"


def test_build_keeps_the_error():
    data = render.build(record_run([RuntimeError("boom")])[1])
    assert data["error"] == "RuntimeError: boom"
    assert data["end"]["stop_reason"] == "error"


# ---------- chat side ----------

def test_draft_shows_only_reply_tokens():
    _, events, _ = record_run(["想一想<channel|>答案" + "<turn|>"], thinking=True)
    data = render.build(events)
    assert render.draft(data) == "答案"
    assert render.draft(render.build([])) == ""


def test_draft_skips_tool_call_tokens():
    data = render.build(record_run([tool_call("calculator", expression="1")])[1][:20])
    assert "calculator" not in render.draft(data)


def prefix_until(events, kind, nth=1):
    seen = 0
    for i, e in enumerate(events):
        if e["type"] == kind:
            seen += 1
            if seen == nth:
                return render.build(events[: i + 1])
    raise AssertionError(kind)


def test_status_follows_the_loop():
    events = approved_run_events()
    assert render.status(render.build(events[:1]), lang="zh") == "啟動中…"
    assert render.status(prefix_until(events, "context"), lang="zh") == "Step 1 · 把脈絡送進模型"
    assert render.status(prefix_until(events, "prefill"), lang="zh").startswith("Step 1 · 模型讀完")
    assert render.status(prefix_until(events, "token"), lang="zh") == "Step 1 · 模型思考中…"
    assert render.status(prefix_until(events, "parsed"), lang="zh") == "Step 1 · harness 解析與驗證"
    assert render.status(prefix_until(events, "approval_request"), lang="zh") == "Step 1 · 等你確認要不要執行工具"
    assert render.status(prefix_until(events, "tool_start"), lang="zh") == "Step 1 · 在環境執行 get_current_time"
    assert render.status(prefix_until(events, "tool_result"), lang="zh") == "Step 1 · 觀察接回脈絡"
    assert render.status(prefix_until(events, "context", 2), lang="zh") == "Step 2 · 把脈絡送進模型"


def test_status_while_writing_a_tool_call():
    events = record_run([tool_call("calculator", expression="1")])[1]
    i = next(i for i, e in enumerate(events) if e["type"] == "token" and e["channel"] == "tool_call")
    assert render.status(render.build(events[: i + 1]), lang="zh") == "Step 1 · 模型正在寫工具呼叫…"


def test_trace_chips():
    assert render.trace_html(render.build([]), lang="zh") == ""
    _, events, _ = record_run([tool_call("calculator", expression="1/0"), tool_call("calculator"), ANSWER])
    html = render.trace_html(render.build(events), lang="zh")
    assert html.count('class="trace-chip"') == 2
    assert 'data-ok="false"' in html and "Step 1 · calculator" in html
    assert "未執行" in html  # the invalid call never reached the environment


def test_footer_and_run_stats():
    _, events, _ = record_run([tool_call("calculator", expression="1"), ANSWER])
    data = render.build(events)
    assert re.fullmatch(r"2 步 · .+ · 停止：模型沒有再呼叫工具", render.footer(data, lang="zh"))
    stats = render.run_stats(data, lang="zh")
    assert stats.startswith("Step 2 / 6 · 脈絡 ")
    assert f"共生成 {sum(len(s['tokens']) for s in data['steps']):,} tokens" in stats


# ---------- the loop diagram ----------

STAGES = [
    ("context", "context", "context-model"),
    ("prefill", "model", "context-model"),
    ("token", "model", None),
    ("parsed", "harness", "model-harness"),
    ("approval_request", "harness", None),
    ("tool_start", "env", "harness-env"),
    ("tool_result", "env", "env-context"),
    ("run_end", "stop", "harness-stop"),
]


@pytest.mark.parametrize("kind, active, edge", STAGES)
def test_loop_state_lights_up_the_current_stage(kind, active, edge):
    state = render.loop_state(prefix_until(approved_run_events(), kind), lang="zh")
    assert (state["active"], state["edge"]) == (active, edge)
    assert state["waiting"] == (kind == "approval_request")


def test_loop_state_before_any_run():
    state = render.loop_state(None, lang="zh")
    assert state["active"] is None and state["counter"] == "等待問題" and not state["ended"]


def test_second_step_starts_from_the_observation():
    events = record_run([tool_call("calculator", expression="1"), ANSWER])[1]
    state = render.loop_state(prefix_until(events, "step_start", 2), lang="zh")
    assert (state["active"], state["edge"]) == ("context", "env-context")
    assert state["visited"] == {"context", "model", "harness", "env"}


def test_rejected_call_never_lights_the_environment():
    state = render.loop_state(render.build(record_run([tool_call("calculator"), ANSWER])[1]), lang="zh")
    assert "env" not in state["visited"]
    assert state["ended"] and state["captions"]["stop"] == "模型沒有再呼叫工具"
    assert state["counter"] == "完成 · 2 步"


def test_diagram_marks_active_node_and_edge():
    data = prefix_until(approved_run_events(), "approval_request")
    svg = svg_of(render.diagram_html(data, dark=False, lang="zh"))
    root = ET.fromstring(svg)
    groups = {g.get("class"): g for g in root.iter("{http://www.w3.org/2000/svg}g")}
    assert "node visited on waiting" in groups
    assert "等你確認" in svg and "Step 1 / 6" in svg
    assert render.PALETTES["light"]["accent"] in svg
    assert render.PALETTES["dark"]["accent"] in svg_of(render.diagram_html(data, dark=True, lang="zh"))


def test_diagram_alt_text_names_the_state():
    html = render.diagram_html(None, dark=False, lang="zh")
    assert 'alt="Agent 迴圈' in html and "等待問題" in html


# ---------- one step ----------

def test_step_card_for_a_tool_call():
    _, events, _ = record_run([tool_call("calculator", expression="6*7"), ANSWER])
    data = render.build(events)
    html = render.step_html(data["steps"][0], data, live=False, lang="zh")
    assert "STEP 1" in html and "呼叫工具 ×1" in html
    assert "最新的使用者訊息" in html
    assert '<span class="fn">calculator</span>' in html
    assert "符合工具 schema" in html
    assert "停在 " + render.chip("<|tool_response>", GEMMA4, lang="zh") + " 等工具結果" in html
    assert "Python 沙箱（AST 白名單）" in html and "&quot;result&quot;: 42" in html
    assert 'class="tip"' in html  # finished step: tooltips are on

    second = render.step_html(data["steps"][1], data, live=False, lang="zh")
    assert "回覆" in second and "上一步的觀察接回脈絡" in second and "沒有工具呼叫" in second


def test_live_step_has_no_tooltips():
    events = record_run([ANSWER])[1]
    data = render.build(events[:8])
    html = render.step_html(data["steps"][0], data, live=True, lang="zh")
    assert "進行中" in html and 'class="tip"' not in html
    assert "生成中，完成後可以滑過 token 看候選" in html


def test_step_card_shows_validation_repeat_and_rejection():
    _, events, _ = record_run([tool_call("calculator"), tool_call("calculator"), ANSWER])
    data = render.build(events)
    first, second = (render.step_html(s, data, live=False, lang="zh") for s in data["steps"][:2])
    assert "缺少必要參數 expression" in first and "沒有執行：harness 直接回報錯誤" in first
    assert "跟之前一模一樣的呼叫" not in first
    assert "跟之前一模一樣的呼叫" in second


def test_step_card_approval_states():
    events = approved_run_events()
    waiting = prefix_until(events, "approval_request")
    assert "等你在左邊按「允許」或「拒絕」" in render.step_html(waiting["steps"][0], waiting, live=True, lang="zh")
    done = render.build(events)
    assert "你允許執行" in render.step_html(done["steps"][0], done, live=False, lang="zh")


def test_approval_timeout_is_shown(monkeypatch):
    monkeypatch.setattr(agent, "APPROVAL_TIMEOUT", 0.05)
    events = []
    Run(FakeLLM([tool_call("get_current_time"), ANSWER]), [], "幾點？", RunSettings(approve_tools=True), events.append).run()
    data = render.build(events)
    assert list(data["steps"][0]["calls"].values())[0]["approval"] == "timeout"
    assert "等待確認逾時，harness 當作拒絕並停止迴圈" in render.step_html(data["steps"][0], data, live=False, lang="zh")
    assert "停止：等待確認逾時" in render.stop_html(data, lang="zh")
    assert render.footer(data, lang="zh").endswith("停止：等待確認逾時")


@pytest.mark.parametrize("answer, label, not_label", [
    (None, "⏹ 執行被停止，這個呼叫沒有執行", "你拒絕了"),
    (False, "你拒絕了，harness 把拒絕當作觀察送回模型", "執行被停止"),
])
def test_stopping_is_not_shown_as_rejecting(answer, label, not_label):
    fake = FakeLLM([tool_call("get_current_time"), ANSWER])
    events = []
    run = Run(fake, [], "幾點？", RunSettings(approve_tools=True), events.append)
    thread = threading.Thread(target=run.run, daemon=True)
    thread.start()
    for _ in range(500):
        if any(e["type"] == "approval_request" for e in events):
            break
        threading.Event().wait(0.01)
    run.stop() if answer is None else run.decide("call_1_0", answer)
    thread.join(5)
    data = render.build(events)
    html = render.step_html(data["steps"][0], data, live=False, lang="zh")
    assert label in html and not_label not in html


def test_step_card_marks_truncated_results(monkeypatch):
    monkeypatch.setattr(TOOLS["calculator"], "fn", lambda expression: {"r": "x" * 5000})
    data = render.build(record_run([tool_call("calculator", expression="1"), ANSWER])[1])
    assert "已截斷" in render.step_html(data["steps"][0], data, live=False, lang="zh")


def test_step_card_for_token_limit():
    data = render.build(record_run(["很長的回覆" * 5], sampling=Sampling(max_new_tokens=6))[1])
    html = render.step_html(data["steps"][0], data, live=False, lang="zh")
    assert "到達 token 上限" in html and "被 token 上限截斷" in html


def test_model_output_is_escaped_everywhere():
    evil = "<script>alert(1)</script><img src=x onerror=alert(2)>"
    _, events, _ = record_run([tool_call("search_wikipedia", query=evil), evil + "<turn|>"], user=f"<marquee>{evil}</marquee>")
    data = render.build(events)
    for piece in render_everything(data, live=False):
        assert "<script" not in piece and "<img src=x" not in piece and "<marquee" not in piece


def test_token_tooltip():
    tok = {
        "id": 7, "piece": "▁Hi\n", "index": 0, "p": 0.02, "rank": 9, "p_sampled": 0.3, "kept": 12,
        "top": [{"id": 1, "piece": "a", "p": 0.5, "kept": True}, {"id": 2, "piece": "b", "p": 0.2, "kept": False}],
    }
    html = render.tip_html(tok, GEMMA4, lang="zh")
    assert "␣Hi↵" in html and "#9" in html and "不在前 5 名" in html
    assert 'class="c cut"' in html
    tok["top"].append({"id": 7, "piece": "▁Hi\n", "p": 0.02, "kept": True})
    html = render.tip_html(tok, GEMMA4, lang="zh")
    assert "不在前 5 名" not in html and 'class="c on"' in html


def test_tokens_html_heat_and_newlines():
    toks = [
        {"id": 1, "piece": "a", "text": "a\nb", "special": False, "channel": "content", "p": 1.0},
        {"id": 2, "piece": "<turn|>", "text": "<turn|>", "special": True, "channel": "control", "p": 0.1},
    ]
    html = render.tokens_html(toks, False, GEMMA4, lang="zh")
    assert '--h:0.000' in html and '<span class="nl">↵</span>' in html
    assert 'class="tok ch-control special">&lt;turn|&gt;<span class="tip">' in html


def test_special_tokens_explain_themselves_while_generating():
    """While a step is live only special tokens get a popover, and it says what the token is for."""
    toks = [
        {"id": 1, "piece": "a", "text": "a", "special": False, "channel": "content", "p": 1.0},
        {"id": 2, "piece": "<|tool_call>", "text": "<|tool_call>", "special": True, "channel": "control", "p": 0.9},
    ]
    html = render.tokens_html(toks, False, GEMMA4, lang="zh")
    assert html.count('class="tip"') == 1 and "工具呼叫開始" in html
    assert "模型給的機率" not in html


def test_special_token_tooltip_adds_the_meaning():
    tok = {
        "id": 106, "piece": "<turn|>", "special": True, "index": 3, "p": 0.9, "rank": 1, "p_sampled": 0.95, "kept": 2,
        "top": [{"id": 106, "piece": "<turn|>", "p": 0.9, "kept": True}],
    }
    html = render.tip_html(tok, GEMMA4, lang="zh")
    assert '<span class="tip-what"><b>回合結束（end of turn）</b>' in html
    assert '<span class="tk">stop_reason: end_turn</span>' in html
    assert "模型給的機率" in html
    assert "tip-what" not in render.tip_html({**tok, "piece": "▁Hi", "special": False}, GEMMA4, lang="zh")


def test_token_legend_explains_the_sampling():
    assert "greedy" in render.token_legend({"temperature": 0}, live=False, lang="zh")
    legend = render.token_legend({"temperature": 0.7, "top_k": 0, "top_p": 0.9}, live=False, lang="zh")
    assert "logits ÷ T 0.7" in legend and "留前 全部 個" in legend and "0.9" in legend


# ---------- MTP ----------

def test_drafted_tokens_are_underlined_and_explained():
    data = render.build(SCENARIOS["mtp"]())
    step = data["steps"][0]
    kept = [t for t in step["tokens"] if t.get("draft", {}).get("accepted")]
    html = render.step_html(step, data, live=False, lang="zh")
    assert html.split('<div class="tokens">')[1].count(' drafted"') == len(kept) > 0
    assert "MTP：草稿猜中" in html and "MTP：草稿被退回" in html and '<span class="tk">␣guess</span>' in html
    assert "LLM · 一次推論；草稿模型先猜" in html
    mtp = step["done"]["mtp"]
    assert f'MTP 草稿收下 <b>{mtp["accepted"]}</b> / {mtp["drafted"]}' in html
    assert f'解碼 forward {mtp["forwards"]} 次（不開 MTP 要 {step["done"]["n_tokens"] - 1} 次）' in html


def test_drafted_tokens_are_underlined_while_generating():
    events = SCENARIOS["mtp"]()
    data = render.build(events[: next(i for i, e in enumerate(events) if e["type"] == "model_end")])
    html = render.step_html(data["steps"][0], data, live=True, lang="zh")
    assert " drafted\"" in html and "MTP：草稿猜中" not in html  # no popovers yet
    assert "MTP 草稿收下" not in html


def test_without_mtp_nothing_mentions_it():
    data = render.build(SCENARIOS["tool"]())
    html = render.step_html(data["steps"][0], data, live=False, lang="zh")
    assert "drafted" not in html and "MTP" not in html


def test_token_legend_explains_mtp_only_when_it_is_on():
    on = render.token_legend({"temperature": 1, "mtp": True}, live=False, lang="zh")
    assert '<span class="tok drafted">底線</span>' in on and "同一次 forward" in on
    assert "MTP" not in render.token_legend({"temperature": 1}, live=False, lang="zh")


def test_knobs_say_whether_mtp_is_on():
    s = {"temperature": 0.7, "top_k": 0, "top_p": 0.9, "max_new_tokens": 512, "max_steps": 4}
    assert "MTP 推測解碼 <b>關</b>" in render.knobs_html(s, INFO["defaults"], lang="zh")
    assert "MTP 推測解碼 <b>開</b>" in render.knobs_html({**s, "mtp": True}, INFO["defaults"], lang="zh")


def test_model_card_names_the_drafter():
    assert "MTP 草稿模型" not in render.model_html(INFO, lang="zh")
    waiting = render.model_html({**INFO, "drafter_id": DRAFTER["model_id"]}, lang="zh")
    assert "huggingface.co/fake/gemma-test-assistant" in waiting and "打開側欄的「MTP 推測解碼」就會載入" in waiting
    loaded = render.model_html({**INFO, "drafter_id": DRAFTER["model_id"], "drafter": DRAFTER}, lang="zh")
    assert "77.7M 參數 · 4 層" in loaded and "每次猜 6 個 token" in loaded
    assert_balanced(waiting)
    assert_balanced(loaded)


def test_step_rows_with_mtp():
    rows = render.step_rows(render.build(SCENARIOS["mtp"]()), lang="zh")
    assert rows[0]["MTP 收下"].count("/") == 1 and rows[0]["解碼 forward"] > 0


# ---------- stop card ----------

@pytest.mark.parametrize("scenario, title", [
    ("reply", "模型沒有再呼叫工具"),
    ("max_steps", "達到步數上限"),
    ("token_limit", "達到單步 token 上限"),
    ("error", "執行錯誤"),
])
def test_stop_card(scenario, title):
    data = render.build(SCENARIOS[scenario]())
    html = render.stop_html(data, lang="zh")
    assert f"停止：{title}" in html
    assert ("is-error" in html) == (scenario == "error")


def test_stop_card_names_the_stop_token():
    html = render.stop_html(render.build(record_run([ANSWER])[1]), lang="zh")
    assert "最後停在" in html and "相當於 end_turn" in html
    assert render.chip("<turn|>", GEMMA4, lang="zh") in html


def test_stop_card_shows_the_error_message():
    assert "RuntimeError: boom" in render.stop_html(render.build(SCENARIOS["error"]()), lang="zh")


# ---------- context tab ----------

def test_context_highlights_what_is_new():
    _, events, _ = record_run([tool_call("calculator", expression="1"), ANSWER])
    data = render.build(events)
    one, two = data["steps"]
    html1 = render.context_html(one, GEMMA4, lang="zh")
    assert '<mark class="added">' + render.chip("<|turn>", GEMMA4, lang="zh") + "user" in html1
    assert '<span class="gen">' in html1 and "calculator" in html1.split('<span class="gen">')[1]
    html2 = render.context_html(two, GEMMA4, lang="zh")
    assert '<mark class="added">response:calculator' in html2
    summary = render.context_summary(one, lang="zh")
    assert f"/ {INFO['context_length']:,} tokens" in summary and "system 1 · user 1" in summary


def test_context_of_a_step_still_generating():
    events = record_run([ANSWER])[1]
    data = render.build(events[:7])
    assert data["steps"][0]["done"] is None
    assert '<span class="gen">' in render.context_html(data["steps"][0], GEMMA4, lang="zh")


# ---------- model tab ----------

def test_model_card():
    html = render.model_html(INFO, lang="zh")
    assert "https://huggingface.co/fake/gemma-test" in html
    assert "5.10B" in html and "語言模型 4.60B" in html and "音訊編碼器 0.30B" in html and "視覺編碼器 0.20B" in html
    assert "8 Q / 1 KV" in html and "131,072 tokens" in html
    assert html.count('<li><span class="sp">') == 3
    assert_balanced(html)


def test_model_card_groups_unknown_parts():
    info = {**INFO, "parameter_parts": {"language_model": 4_000_000_000, "embed_a": 5_000_000, "embed_b": 7_000_000}}
    html = render.model_html(info, lang="zh")
    assert "其他 12.0M" in html and html.count("其他") == 2  # one bar segment, one legend entry


def test_knobs():
    html = render.knobs_html({"temperature": 0.7, "top_k": 0, "top_p": 0.9, "max_new_tokens": 512, "max_steps": 4}, INFO["defaults"], lang="zh")
    assert "top-k <b>不限</b>" in html and "temperature <b>0.7</b>" in html
    assert "模型建議值：T 1.0、top-p 0.95、top-k 64" in html


def test_step_rows():
    _, events, _ = record_run([tool_call("calculator", expression="1"), ANSWER])
    rows = render.step_rows(render.build(events), lang="zh")
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[0]["停止"] == "<|tool_response>" and rows[1]["停止"] == "<turn|>"
    assert rows[0]["平均機率"] == 0.9 and rows[0]["prefill ms"] == 12.5
    assert set(rows[0]) == {"step", "脈絡 tokens", "prefill ms", "生成 tokens", "tok/s", "平均機率", "停止"}


def test_tools_card():
    html = render.tools_html(TOOLS, True, GEMMA4, lang="zh")
    for name in TOOLS:
        assert f'<span class="fn">{name}</span>' in html
    assert "timezone?: string" in html and "expression: string" in html and "lang?: string" in html
    assert "目前關閉了「使用工具」" not in html
    assert "目前關閉了「使用工具」" in render.tools_html(TOOLS, False, GEMMA4, lang="zh")


def test_special_tokens_card_lists_every_token():
    html = render.special_tokens_html(INFO, lang="zh")
    assert "特殊 token（24 個）" in html
    for piece, token_id in INFO["special_tokens"].items():
        name, _ = GEMMA4.special_tokens[piece]["zh"]
        assert f'<span class="sp">{render.esc(piece)}</span><small>id {token_id}</small>' in html
        assert f"<b>{render.esc(name)}</b>" in html
    assert html.count('class="stop-tag"') == len(INFO["stop_tokens"])
    assert [title for title, _, _ in GEMMA4.special_groups["zh"]] == ["序列", "回合", "思考", "工具", "多模態", "保留"]
    assert "其他" not in html
    assert_balanced(html)


def test_special_tokens_card_keeps_tokens_it_cannot_explain():
    info = {**INFO, "special_tokens": {"<bos>": 2, "<|new|>": 9}, "stop_tokens": []}
    html = render.special_tokens_html(info, lang="zh")
    assert "特殊 token（2 個）" in html and "回合" not in html
    assert "其他" in html and "&lt;|new|&gt;" in html and "沒有說明" in html
    assert_balanced(html)


# ---------- formatting ----------

@pytest.mark.parametrize("ms, text", [(0.4, "<1 ms"), (12.3, "12 ms"), (999, "999 ms"), (1500, "1.5 s")])
def test_fmt_ms(ms, text):
    assert render.fmt_ms(ms) == text


@pytest.mark.parametrize("p, text", [(1.0, "100%"), (0.9996, "100%"), (0.5, "50.0%"), (0.0123, "1.23%"), (0.00001, "1.0e-03%")])
def test_pct(p, text):
    assert render.pct(p) == text


def test_fmt_params():
    assert render.fmt_params(5_100_000_000) == "5.10B"
    assert render.fmt_params(300_000_000) == "0.30B"
    assert render.fmt_params(42_000_000) == "42.0M"


def test_heat_is_clamped():
    assert render.heat(1.0) == "0.000" and render.heat(0.0) == "0.600" and render.heat(1.5) == "0.000"


def test_special_html_chips_and_escapes():
    html = render.special_html('<|turn>user\na < b & "c"<turn|><|"|>x<|"|><bos><eos>', GEMMA4, lang="zh")
    assert html.count('<span class="sp">') == 6
    assert html.count('<span class="sp-tip">') == 6  # every chip explains itself on hover
    assert "a &lt; b &amp; &quot;c&quot;" in html
    assert_balanced(html)


@pytest.mark.parametrize("piece", GEMMA4.special_tokens)
def test_every_explained_token_becomes_a_chip_with_a_tip(piece):
    assert GEMMA4.special_re.fullmatch(piece)
    html = render.special_html(f"x{piece}y", GEMMA4, lang="zh")
    name, _ = GEMMA4.special_tokens[piece]["zh"]
    assert html == f"x{render.chip(piece, GEMMA4, lang="zh")}y"
    assert f'<span class="sp-tip"><b>{render.esc(name)}</b>' in html
    assert_balanced(html)


def test_chip_without_a_note_has_no_tip():
    assert render.chip("<|new|>", GEMMA4, lang="zh") == '<span class="sp">&lt;|new|&gt;</span>'


def test_prose_marks_literal_text():
    assert render.prose_html("寫 `<|turn>model` 再 `a&b`") == '寫 <span class="tk">&lt;|turn&gt;model</span> 再 <span class="tk">a&amp;b</span>'


def test_pretty_json():
    assert render.pretty('{"a": 1}') == '{\n  "a": 1\n}'
    assert render.pretty("not json…(truncated)") == "not json…(truncated)"
    assert render.pretty('{"中": "文"}') == '{\n  "中": "文"\n}'


# ---------- another family ----------

def test_a_qwen_run_draws_in_qwen_terms():
    _, events, _ = record_run([qwen_tool_call("calculator", expression="6*7"), "答案是 42。<|im_end|>"], family=QWEN3_5)
    data = render.build(events)
    assert data["family"] is QWEN3_5 and data["model"] == "fake/qwen-test"
    one, two = data["steps"]
    html = render.step_html(one, data, live=False, lang="zh")
    assert "最新的使用者訊息" in html and render.chip("<|im_start|>", QWEN3_5, lang="zh") + "user" in html
    assert "停在 " + render.chip("<|im_end|>", QWEN3_5, lang="zh") + " 訊息結束；回覆和工具呼叫都停在這裡" in html
    assert '<span class="fn">calculator</span>' in html and "符合工具 schema" in html
    assert 'class="tok ch-control special">&lt;tool_call&gt;' in html  # an added token Qwen does not mark special
    context = render.context_html(two, QWEN3_5, lang="zh")
    assert '<mark class="added">' + render.chip("<|im_start|>", QWEN3_5, lang="zh") + "user\n" + render.chip("<tool_response>", QWEN3_5, lang="zh") in context
    assert "最後停在 " + render.chip("<|im_end|>", QWEN3_5, lang="zh") in render.stop_html(data, lang="zh")
    assert "gemma" not in render.diagram_html(data, dark=False, lang="zh").lower()


def test_qwen_thinking_tokens_are_italic_thoughts():
    _, events, _ = record_run(["想一下\n</think>\n\n好。<|im_end|>"], family=QWEN3_5, thinking=True)
    tokens = render.build(events)["steps"][0]["tokens"]
    assert {t["channel"] for t in tokens if t["text"] in "想一下"} == {"thought"}
    assert next(t for t in tokens if t["piece"] == "</think>")["channel"] == "control"
    assert {t["channel"] for t in tokens if t["text"] == "好"} == {"content"}


def test_qwen_model_card():
    html = render.model_html(QWEN_INFO, lang="zh")
    assert "Qwen3.5 家族" in html and "Gated DeltaNet" in html
    assert "sliding window" not in html  # Qwen3.5 has none
    assert "線性 attention 18 · full attention 6" in html and "視覺編碼器 0.10B" in html
    assert html.count('<li><span class="sp">') == 2
    assert_balanced(html)


def test_qwen_special_tokens_card():
    html = render.special_tokens_html(QWEN_INFO, lang="zh")
    assert "特殊 token（33 個）" in html and "其他" not in html
    assert html.count('class="stop-tag"') == 2
    assert "&lt;think&gt;" in html and "推理寫在 &lt;think&gt; 和 &lt;/think&gt; 之間" in html
    assert_balanced(html)


def test_tools_card_says_how_each_family_writes_the_schemas():
    assert "&lt;|tool&gt;declaration" in render.tools_html(TOOLS, True, GEMMA4, lang="zh")
    assert "&lt;tools&gt;…&lt;/tools&gt;" in render.tools_html(TOOLS, True, QWEN3_5, lang="zh")


def test_diagram_names_the_model():
    assert "Qwen3.5-0.8B" in svg_of(render.diagram_html(None, dark=False, model="Qwen/Qwen3.5-0.8B", lang="zh"))
    started = render.build(record_run([ANSWER])[1][:2])  # before the model box shows live numbers
    assert "gemma-test" in svg_of(render.diagram_html(started, dark=False, model="Qwen/Qwen3.5-0.8B", lang="zh"))  # the run's model wins
