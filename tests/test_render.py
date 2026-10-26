"""render.py turns the event list into HTML. The page redraws every half second while
a run is live, so every function here must work on any prefix of a run's events."""

import base64
import dataclasses
import json
import re
import tempfile
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from app import render
from fake_pi import FakePi
from fakes import DRAFTER, INFO, MEDIA_TOKENS, QWEN_INFO, FakeLLM, qwen_tool_call, record_run, tool_call
from agent import model_families
from agent.model_backends.backend import TOKEN_DETAILS, Sampling
from agent.model_families import GEMMA4, QWEN3_5
from agent.harnesses import model_endpoint, pi_harness, simple_harness
from agent.harnesses.simple_harness import RunSettings, SimpleHarness
from agent.i18n import LANGS
from agent.media import Attachment
from agent.tools import TOOLS, ToolResult
from html_check import assert_balanced

ANSWER = "現在是下午三點。<turn|>"


def svg_of(diagram: str) -> str:
    src = re.search(r'src="data:image/svg\+xml;base64,([^"]+)"', diagram).group(1)
    return base64.b64decode(src).decode()


def approved_run_events():
    """A thinking run that asks for approval, gets it, calls a tool and answers."""
    fake = FakeLLM(["要查時間<channel|>" + tool_call("get_current_time"), "好<channel|>" + ANSWER])
    events = []
    run = SimpleHarness(fake, [], "台北現在幾點？", RunSettings(thinking=True, approve_tools=True), events.append)
    thread = threading.Thread(target=run.run, daemon=True)
    thread.start()
    for _ in range(500):
        if any(e["type"] == "approval_request" for e in events):
            break
        threading.Event().wait(0.01)
    run.decide("call_1_0", True)
    thread.join(5)
    return events


def as_reported(events: list[dict], details: tuple[str, ...] = ()) -> list[dict]:
    """A run's events as a backend would send them that knows only `details` (of backend.TOKEN_DETAILS) about each
    token, such as a serving engine that reports no log probabilities, or only the token's and the top few."""
    def token(event):
        out = {k: v for k, v in event.items() if k not in TOKEN_DETAILS or k in details}
        if "top" in out and "kept" not in details:  # nor which candidates top-k and top-p cut
            out["top"] = [{k: v for k, v in c.items() if k != "kept"} for c in out["top"]]
        return out
    return [token(e) if e["type"] == "token" else e for e in events]


LOGPROBS = ("p", "logprob", "top")  # what an OpenAI-style API's log probabilities give
BASH = {"tool": "bash", "args": {"command": "date"}, "result": "Mon Oct 26 14:00:00 CST 2026\n"}


def pi_page_events(outputs: list, text: str = "現在幾點？", **settings) -> list[dict]:
    """A turn pi (its stand-in) runs on the page's own model (a FakeLLM) through model_endpoint, as its events."""
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory() as folder:
        FakePi(mp, Path(folder)).say(results={"bash": BASH["result"]})
        llm, events = FakeLLM(list(outputs)), []
        pi_harness.PiHarness(pi_harness.PiSettings(model_endpoint.PROVIDER, llm.model_id, **settings), [], text, events.append,
                             folder=Path(folder) / "chat", session="c", page=pi_harness.PageModel(llm, Sampling())).run()
    return events


def pi_events(*turns, approve: bool | None = None, text: str = "現在幾點？", **settings) -> list[dict]:
    """A turn run by pi (its stand-in, tests/fake_pi.py), as its events; with `approve`, the user answers the first call so."""
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory() as folder:
        FakePi(mp, Path(folder)).say(*turns)
        events = []
        run = pi_harness.PiHarness(pi_harness.PiSettings("fake", "fake-model", approve_tools=approve is not None, **settings), [], text, events.append,
                       folder=Path(folder) / "chat", session="c")
        thread = threading.Thread(target=run.run, daemon=True)
        thread.start()
        if approve is not None:
            for _ in range(1000):
                if request := next((e for e in events if e["type"] == "approval_request"), None):
                    run.decide(request["call_id"], approve)
                    break
                threading.Event().wait(0.01)
        thread.join(20)
    return events
PHOTO = Attachment("photo.png", "image", "/chat/photo.png", 2048)
CLIP = Attachment("clip.mp4", "video", "/chat/clip.mp4", 90_000)
SALES = Attachment("sales.csv", "file", str(Path(__file__).resolve().parent.parent / "app/assets/examples/sales.csv"), 942)
CODE = "import math\nprint(math.pi)"
SCENARIOS = {
    "reply": lambda: record_run([ANSWER])[1],
    "media": lambda: record_run([tool_call("calculator", expression="6*7"), ANSWER], attachments=[PHOTO, CLIP, SALES])[1],
    "qwen_media": lambda: record_run([qwen_tool_call("calculator", expression="6*7"), "好。<|im_end|>"], family=QWEN3_5, attachments=[CLIP])[1],
    # run_python left out of the turn, so the call is refused without Docker; its code is still drawn
    "code": lambda: record_run([tool_call("run_python", code=CODE), ANSWER], tools=("calculator",))[1],
    "tool": lambda: record_run([tool_call("calculator", expression="6*7"), "答案是 42。<turn|>"])[1],
    "approved": approved_run_events,
    "invalid_and_repeated": lambda: record_run([tool_call("calculator"), tool_call("calculator"), ANSWER])[1],
    "max_steps": lambda: record_run([tool_call("get_current_time")] * 3, max_steps=2)[1],
    "token_limit": lambda: record_run(["很長的回覆" * 5], sampling=Sampling(max_new_tokens=6))[1],
    "error": lambda: record_run([RuntimeError("boom")])[1],
    "mtp": lambda: record_run([tool_call("calculator", expression="6*7"), "答案是 42。<turn|>"], sampling=Sampling(mtp=True))[1],
    "qwen_tool": lambda: record_run(
        [qwen_tool_call("calculator", expression="6*7"), "答案是 42。<|im_end|>"], family=QWEN3_5)[1],
    "no_details": lambda: as_reported(record_run([tool_call("calculator", expression="6*7"), "答案是 42。<turn|>"])[1]),
    "logprobs": lambda: as_reported(record_run([tool_call("calculator", expression="6*7"), "答案是 42。<turn|>"])[1], LOGPROBS),
    "qwen_thinking": lambda: record_run(
        ["要算一下\n</think>\n\n" + qwen_tool_call("calculator", expression="1"), "想好了\n</think>\n\n好。<|im_end|>"],
        family=QWEN3_5, thinking=True)[1],
    "pi": lambda: pi_events({"say": [{"thinking": "用 date。"}, {"text": "我查一下。"}, BASH]}, {"say": [{"text": "下午兩點。"}]}),
    "pi_rejected": lambda: pi_events({"say": [BASH]}, {"say": [{"text": "好吧。"}]}, approve=False),
    "pi_max_steps": lambda: pi_events(*[{"say": [BASH]}] * 3, max_steps=2),
    "pi_error": lambda: pi_events({"say": [], "stop": "error", "error": "Connection error."}),
    "pi_page_model": lambda: pi_page_events([tool_call("bash", command="date"), ANSWER]),
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
    if data["settings"] and data["harness"] == "pi":
        out.append(render.pi_settings_html(data["settings"], lang=lang))
        out.append(render.pi_tools_html(pi_harness.TOOLS, data["settings"]["tools"], lang=lang))
    elif data["settings"]:
        out.append(render.knobs_html(data["settings"], INFO["defaults"], lang=lang))
        out.append(render.tools_html(TOOLS, data["settings"]["use_tools"], data["family"], lang=lang, offered=data["settings"]["tools"]))
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
    "media": lambda: record_run([tool_call("calculator", expression="6*7"), "It is 42.<turn|>"], user="What is this?",
                                attachments=[PHOTO, CLIP, SALES], lang="en")[1],
    "code": lambda: record_run([tool_call("run_python", code=CODE), "Pi.<turn|>"], user="Pi?", tools=("calculator",), lang="en")[1],
    "mtp": lambda: record_run([tool_call("calculator", expression="6*7"), "It is 42.<turn|>"], user="6*7?",
                              sampling=Sampling(mtp=True), lang="en")[1],
    "qwen_tool": lambda: record_run(
        [qwen_tool_call("calculator", expression="6*7"), "It is 42.<|im_end|>"], user="6*7?", family=QWEN3_5, lang="en")[1],
    "no_details": lambda: as_reported(record_run([tool_call("calculator", expression="6*7"), "It is 42.<turn|>"], user="6*7?", lang="en")[1]),
    "logprobs": lambda: as_reported(
        record_run([tool_call("calculator", expression="6*7"), "It is 42.<turn|>"], user="6*7?", lang="en")[1], LOGPROBS),
    "pi": lambda: pi_events({"say": [{"thinking": "Use date."}, BASH]}, {"say": [{"text": "It is 2 pm."}]}, text="Time?", lang="en"),
    "pi_rejected": lambda: pi_events({"say": [BASH]}, {"say": [{"text": "OK."}]}, approve=False, text="Time?", lang="en"),
    "pi_max_steps": lambda: pi_events(*[{"say": [BASH]}] * 3, max_steps=2, text="Time?", lang="en"),
    "pi_error": lambda: pi_events({"say": [], "stop": "error", "error": "Connection error."}, text="Time?", lang="en"),
    "pi_page_model": lambda: pi_page_events([tool_call("bash", command="date"), "It is 2 pm.<turn|>"], text="Time?", lang="en"),
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
    family = model_families.get(info["family"])
    pieces = [
        render.model_html(info, lang="en"), render.special_tokens_html(info, lang="en"), render.folder_html(info, lang="en"),
        render.INTRO["en"], render.CONTEXT_LEGEND["en"],
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
    assert data == {"model": None, "family": None, "harness": None, "settings": {}, "steps": [], "end": None, "error": None, "last": None, "t": 0}


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


def test_images_are_the_charts_of_each_step_in_order(monkeypatch):
    charts = iter([["/u/figure-1.png", "/u/figure-2.png"], ["/u/figure-3.png"]])

    def run_python(code, files=None):
        images = next(charts)
        return ToolResult({"figures": [f"/data/{Path(p).name}" for p in images]}, images)

    monkeypatch.setitem(TOOLS, "run_python", dataclasses.replace(TOOLS["run_python"], fn=run_python))
    outputs = [tool_call("run_python", code="a"), tool_call("calculator", expression="1+1"), tool_call("run_python", code="b"), "好了。<turn|>"]
    _, events, _ = record_run(outputs)
    data = render.build(events)
    assert render.images(data["steps"]) == ["/u/figure-1.png", "/u/figure-2.png", "/u/figure-3.png"]
    assert [render.images([s]) for s in data["steps"]] == [["/u/figure-1.png", "/u/figure-2.png"], [], ["/u/figure-3.png"], []]


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


def test_a_live_steps_plain_tokens_have_no_popovers():
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
    monkeypatch.setattr(simple_harness, "APPROVAL_TIMEOUT", 0.05)
    events = []
    SimpleHarness(FakeLLM([tool_call("get_current_time"), ANSWER]), [], "幾點？", RunSettings(approve_tools=True), events.append).run()
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
    run = SimpleHarness(fake, [], "幾點？", RunSettings(approve_tools=True), events.append)
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


@pytest.mark.parametrize("lang", LANGS)
def test_a_token_without_details_shows_its_id_and_no_numbers(lang):
    tok = {"index": 2, "id": 7, "piece": "▁Hi", "text": " Hi", "special": False, "channel": "content"}
    html = render.tip_html(tok, GEMMA4, lang=lang)
    assert '<span class="tip-piece">␣Hi</span>' in html and "token id 7" in html
    assert "tip-grid" not in html and "tip-label" not in html and "None" not in html
    assert "--h:" not in render.tokens_html([tok], True, GEMMA4, lang=lang)  # no heat without a probability


def test_a_token_with_log_probabilities_shows_them_and_marks_no_candidate_cut():
    tok = {
        "index": 0, "id": 7, "piece": "▁Hi", "text": " Hi", "special": False, "channel": "content", "p": 0.02, "logprob": -3.9,
        "top": [{"id": 1, "piece": "a", "p": 0.5}, {"id": 2, "piece": "b", "p": 0.2}],
    }
    html = render.tip_html(tok, GEMMA4, lang="zh")
    assert "<span>模型給的機率</span><b>2.00%</b>" in html and '<span class="cp">a</span>' in html
    assert "排名" not in html and "取樣時的機率" not in html and "剩下的候選" not in html
    assert " cut" not in html and "灰色" not in html and "前 5 名候選（模型原始分布）" in html
    assert "抽到的 token 不在前 5 名。" in html
    assert "Drew a token outside the top 5." in render.tip_html(tok, GEMMA4, lang="en")
    assert "--h:0.588" in render.tokens_html([tok], True, GEMMA4, lang="zh")


def test_the_legend_mentions_heat_and_candidates_only_when_the_tokens_carry_them():
    def legend(scenario):
        data = render.build(SCENARIOS[scenario]())
        return render.step_html(data["steps"][0], data, lang="zh", live=False)

    assert "heat-scale" in legend("tool") and "滑過 token 看候選" in legend("tool")
    assert "heat-scale" in legend("logprobs") and "滑過 token 看候選" in legend("logprobs")
    bare = legend("no_details")
    assert "heat-scale" not in bare and "滑過 token 看它的 id" in bare and "--h:" not in bare
    # before the first token there is nothing to tell by, so the legend is the full one
    assert "heat-scale" in render.token_legend({"temperature": 1}, live=True, lang="zh", tokens=[])


def test_step_rows_have_no_mean_probability_when_the_tokens_carry_none():
    rows = render.step_rows(render.build(SCENARIOS["no_details"]()), lang="zh")
    assert [r["平均機率"] for r in rows] == [None, None]
    assert [r["平均機率"] for r in render.step_rows(render.build(SCENARIOS["logprobs"]()), lang="zh")] == [0.9, 0.9]


def test_qwen_tokens_show_as_text_not_as_their_byte_spelling():
    """Qwen's vocabulary writes 如果 as å¦Ĥæŀľ; the popover, its candidates and a byte token's body show the text."""
    tok = {
        "id": 96604, "piece": "å¦Ĥæŀľ", "text": "如果", "special": False, "channel": "content", "index": 456,
        "p": 0.378, "rank": 1, "p_sampled": 0.4, "kept": 20,
        "top": [{"id": 96604, "piece": "å¦Ĥæŀľ", "p": 0.378, "kept": True}, {"id": 23066, "piece": "Ġhello", "p": 0.1, "kept": True}],
    }
    byte = {**tok, "id": 397, "piece": "ð", "text": "", "top": []}
    html = render.tokens_html([tok, byte], True, QWEN3_5, lang="zh")
    assert '<span class="tip-piece">如果</span>' in html and '<span class="cp">␣hello</span>' in html
    assert '&lt;0xF0&gt;<span class="tip">' in html and "å" not in html and "Ġ" not in html


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
    assert "MTP speculative decoding <b>the server&#x27;s</b>" in render.knobs_html({**s, "mtp": None}, INFO["defaults"], lang="en")


def test_model_card_names_the_drafter():
    assert "MTP 草稿模型" not in render.model_html(INFO, lang="zh")
    waiting = render.model_html({**INFO, "drafter_id": DRAFTER["model_id"]}, lang="zh")
    assert "huggingface.co/fake/gemma-test-assistant" in waiting and "打開側欄的「MTP 推測解碼」就會載入" in waiting
    loaded = render.model_html({**INFO, "drafter_id": DRAFTER["model_id"], "drafter": DRAFTER}, lang="zh")
    assert "77.7M 參數 · 4 層 · 隱藏層維度 256" in loaded and "每次猜 6 個 token" in loaded
    assert_balanced(waiting)
    assert_balanced(loaded)


def test_the_model_folder_lists_each_file_with_its_size_what_it_is_and_what_it_holds():
    html = render.folder_html(INFO, lang="en")
    assert "models/models--fake--gemma-test/snapshots/0123456789abcdef" in html and "at revision 0123456, the commit" in html
    names = re.findall(r'<dt><span class="tk">(.*?)</span>', html)
    assert names == ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                     "chat_template.jinja", "processor_config.json", "README.md", ".gitattributes"]
    assert "<small>10.2 GB</small>" in html and "<small>208 B</small>" in html and "<small>5.0 kB</small>" in html
    assert "<b>architecture</b>" in html and "Gemma 4: Three parts in one file" in html
    assert '<li><span class="tk">text_config.sliding_window</span> 512</li>' in html
    assert '<li><span class="tk">architectures</span> [&quot;Gemma4ForConditionalGeneration&quot;]</li>' in html
    assert "<b>git settings</b>" in html and "Gemma 4: " not in html.split(".gitattributes")[1]  # no family note on it
    assert "gloss-head" not in html  # no drafter loaded, so one folder
    assert "a later download may get a newer one" in html  # fake/gemma-test is not pinned
    pinned = render.folder_html({**INFO, "model_id": "google/gemma-4-E2B-it", "folder": {**INFO["folder"], "revision": model_families.REVISIONS["google/gemma-4-E2B-it"]}}, "en")
    assert "revision 3e22461, the commit the app pins it to" in pinned
    assert_balanced(html)


def test_the_model_folder_lists_a_file_the_family_notes_but_the_checkpoint_lacks():
    html = render.folder_html(QWEN_INFO, lang="zh")
    names = re.findall(r'<dt><span class="tk">(.*?)</span>', html)
    assert names[:3] == ["config.json", "generation_config.json", "model.safetensors-00001-of-00001.safetensors"]
    assert "generation_config.json</span><small>這個 checkpoint 沒有</small>" in html
    assert "Qwen3.5：沒有這個檔案時" in html
    shard = html.split("model.safetensors-00001-of-00001.safetensors")[1]
    assert "<b>權重</b>" in shard and "1.7 GB" in shard and "transformers 載入時跳過" in shard  # a shard is read as the weights
    assert names[3:8] == ["model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    assert names[-3:] == ["README.md", "LICENSE", ".gitattributes"] and html.count("這個 checkpoint 沒有") == 1
    assert_balanced(html)


def test_the_model_folder_shows_the_drafter_once_it_is_loaded():
    assert "gloss-head" not in render.folder_html({**INFO, "drafter_id": DRAFTER["model_id"]}, lang="zh")
    html = render.folder_html({**INFO, "drafter_id": DRAFTER["model_id"], "drafter": DRAFTER}, lang="zh")
    drafter = html.split('<h5 class="gloss-head">MTP 草稿模型')[1]
    assert "models/models--fake--gemma-test-assistant/snapshots/0123456789abcdef" in drafter
    assert 'Gemma 4：<span class="tk">Gemma4AssistantForCausalLM</span>' in drafter
    assert "tokenizer.json</span><small>這個 checkpoint 沒有</small>" in drafter  # the fake drafter lists two of its files
    assert '<li><span class="tk">num_centroids</span> 2048</li>' in drafter
    assert_balanced(html)


def test_the_notes_on_the_weights_are_said_once_for_all_their_shards():
    shards = [{"name": f"model-0000{i}-of-00002.safetensors", "bytes": 5_000_000_000, "fields": {}} for i in (1, 2)]
    html = render.folder_html({**QWEN_INFO, "folder": {**QWEN_INFO["folder"], "files": shards}}, lang="en")
    assert html.count("The weights: a header") == 1 and html.count("Qwen3.5: The weights") == 1
    assert "Another shard of the weights." in html.split("model-00002-of-00002.safetensors")[1]


def test_sizes_round_up_to_the_next_unit():
    assert [render.fmt_bytes(n) for n in (208, 999, 999_999, 1_746_942_600, 10_246_621_918)] == ["208 B", "1.0 kB", "1.0 MB", "1.7 GB", "10.2 GB"]


def test_the_model_folder_escapes_what_the_files_hold():
    files = [{"name": "<b>x</b>.json", "bytes": 1, "fields": {}}, {"name": "config.json", "bytes": 1, "fields": {"architectures": "<script>"}}]
    html = render.folder_html({**INFO, "folder": {"path": "<i>models</i>", "revision": None, "files": files}}, lang="en")
    assert "<script>" not in html and "<b>x</b>" not in html and "<i>models</i>" not in html
    assert "&lt;script&gt;" in html and "A checkpoint on disk, loaded in place." in html
    assert "<b>other</b>" in html
    assert_balanced(html)


def test_every_file_a_family_notes_is_a_kind_the_page_knows():
    for family in model_families.FAMILIES.values():
        for name in (*family.file_notes, *family.drafter_file_notes):
            assert render.file_kind(name) == name


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


def test_model_card_names_its_specs_in_the_page_language():
    zh, en = render.model_html(INFO, lang="zh"), render.model_html(INFO, lang="en")
    for label in ("隱藏層維度", "注意力頭數", "滑動視窗"):
        assert f"<dt>{label}</dt>" in zh
    assert "hidden size" not in zh and "attention heads" not in zh and "sliding window" not in zh
    for label in ("hidden size", "attention heads", "sliding window"):
        assert f"<dt>{label}</dt>" in en


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
    assert "滑動視窗" not in html  # Qwen3.5 has none
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


# ---------- media and code ----------

def test_a_run_of_one_special_token_is_one_chip_with_its_count():
    text = "<|turn>user\n<|image>" + "<|image|>" * 266 + "<image|>看<|\"|><|\"|>"
    out = render.special_html(text, GEMMA4, "zh")
    assert out.count('<span class="sp">&lt;|image|&gt;') == 1 and '<span class="sp-n">×266</span>' in out
    assert out.count('<span class="sp">&lt;|&quot;|&gt;') == 2  # a short run stays as it is
    assert_balanced(out)


@pytest.mark.parametrize("last", range(0, 26, 3))
def test_a_cut_snippet_never_splits_a_token_or_a_run(last):
    text = "前文<|turn>user\n" + "<|image|>" * 266 + "這張圖"  # 26 characters drawn, the run counting as one token
    out = render.special_html(text, GEMMA4, "zh", last=last)
    assert out.startswith("…") and out.endswith("這張圖"[-last:] if last else "…")
    rest = out[1:]
    plain = ("前文", "user\n", "這張圖")
    assert not rest or rest.startswith('<span class="sp">') or any(rest.startswith(t[-k:]) for t in plain for k in range(1, len(t) + 1))
    assert_balanced(out)


def test_a_snippet_short_enough_is_not_cut():
    text = "前文<|turn>user\n" + "<|image|>" * 266 + "這張圖"
    assert render.special_html(text, GEMMA4, "zh", last=26) == render.special_html(text, GEMMA4, "zh")


def test_the_step_card_shows_how_many_tokens_the_media_took():
    data = render.build(SCENARIOS["media"]())
    first = render.step_html(data["steps"][0], data, "zh", live=False)
    assert f"其中 圖片 {MEDIA_TOKENS} · 影片 {MEDIA_TOKENS} tokens" in first
    assert f'<span class="sp-n">×{MEDIA_TOKENS}</span>' in first  # the newest user message, with its media expanded
    assert f"of which image {MEDIA_TOKENS} · video {MEDIA_TOKENS} tokens" in render.context_summary(data["steps"][0], "en")


def test_code_in_a_call_is_written_out_below_it():
    data = render.build(SCENARIOS["code"]())
    card = render.step_html(data["steps"][0], data, "en", live=False)
    assert '<span class="fn">run_python</span>({&quot;code&quot;: &quot;…&quot;})' in card
    assert '<pre class="snippet arg">import math\nprint(math.pi)</pre>' in card


def test_a_result_over_several_lines_is_written_out_line_by_line():
    text = json.dumps({"stdout": "December 244320\nYear 2131200\n", "n": 2})
    assert render.pretty(text) == '{\n  "stdout": |\n    December 244320\n    Year 2131200\n  "n": 2\n}'
    assert render.pretty(json.dumps({"result": 42, "list": [1]})) == '{\n  "result": 42,\n  "list": [\n    1\n  ]\n}'
    assert render.pretty('{"stdout": "cut off…(truncated)') == '{"stdout": "cut off…(truncated)'  # no longer JSON: as it is


def test_the_model_tab_lists_what_the_model_reads():
    assert "<dt>讀得懂</dt><dd>文字 · 圖片 · 聲音 · 影片</dd>" in render.model_html(INFO, "zh")
    assert "<dt>reads</dt><dd>text · image · video</dd>" in render.model_html(QWEN_INFO, "en")


def test_a_tool_left_out_of_the_turn_is_marked():
    html = render.tools_html(TOOLS, True, GEMMA4, "en", offered=["calculator"])
    assert html.count("not offered to the model in this turn") == len(TOOLS) - 1


# ---------- a run pi ran ----------

def test_a_pi_step_shows_the_text_it_streamed_by_channel_and_no_token_numbers():
    data = render.build(SCENARIOS["pi"]())
    assert data["harness"] == "pi" and data["family"] is None
    step = data["steps"][0]
    assert step["text"] == [{"channel": "thought", "text": "用 date。"}, {"channel": "content", "text": "我查一下。"},
                            {"channel": "tool_call", "text": '\nbash {"command": "date"}'}]
    card = render.step_html(step, data, lang="en", live=False)
    assert '<span class="run ch-thought">用 date。</span>' in card and 'class="run ch-tool_call"' in card
    assert "wrote <b>20</b> tokens" in card and "read <b>1,200</b> tokens, 200 from the cache" in card and "at pi&#x27;s prices" not in card
    assert "about $0.0021 at pi's prices" in card
    assert "the API returned it as JSON" in card and "shell on this computer · returned" in card
    assert "Context <small>the request pi sends the API</small>" in card and "system 1 · user 1" in card  # the request, not a context
    assert 'class="tok ' not in card and "token id" not in card  # and no tokens


def test_what_pi_streams_is_escaped():
    data = render.build(pi_events({"say": [{"text": "<script>alert(1)</script>"}]}))
    card = render.step_html(data["steps"][0], data, lang="en", live=False)
    assert "<script>" not in card and "&lt;script&gt;alert(1)&lt;/script&gt;" in card


def test_build_joins_what_pi_streams_into_one_run_per_channel_without_changing_the_events():
    events = SCENARIOS["pi"]()
    before = json.dumps(events)
    render.build(events)
    render.build(events)
    assert json.dumps(events) == before


def test_the_reply_streams_into_the_chat_as_pi_sends_it():
    events = SCENARIOS["pi"]()
    second = [i for i, e in enumerate(events) if e["type"] == "step_start"][1]
    first_delta = next(i for i, e in enumerate(events) if i > second and e["type"] == "delta")
    assert render.draft(render.build(events[:first_delta + 1])) == events[first_delta]["text"]
    assert render.status(render.build(events[:first_delta + 1]), lang="en") == "Step 2 · the model is generating…"


def test_the_diagram_says_pi_runs_the_loop_on_this_computer():
    svg = svg_of(render.diagram_html(None, dark=False, lang="en", harness="pi"))
    assert "pi · check · ask" in svg and "this computer" in svg and "text out" in svg and "tokens out" not in svg
    assert "parse · check · ask" in svg_of(render.diagram_html(None, dark=False, lang="en"))
    data = render.build(SCENARIOS["pi"]())  # a run says which harness ran it
    assert "text out" in svg_of(render.diagram_html(data, dark=False, lang="en"))


def test_a_pi_step_lights_the_way_into_the_model_while_pi_waits_for_it():
    events = SCENARIOS["pi"]()
    first = next(i for i, e in enumerate(events) if e["type"] == "step_start")
    state = render.loop_state(render.build(events[:first + 1]), lang="en")
    assert (state["active"], state["edge"]) == ("context", "context-model")


def test_the_stop_card_says_the_page_asked_pi_to_abort():
    data = render.build(SCENARIOS["pi_max_steps"]())
    card = render.stop_html(data, lang="en")
    assert "Stop: step limit reached" in card and "pi has no step limit of its own" in card
    assert len(data["steps"]) == 2 and render.run_stats(data, lang="en").startswith("Step 2 / 2 · context 1,200 tokens · 40 tokens generated")


def test_a_failed_request_to_pis_model_is_its_error():
    data = render.build(SCENARIOS["pi_error"]())
    assert render.error_text(data, "en") == "pi: Connection error."
    assert "pi or the model's API reported an error" in render.stop_html(data, lang="en")


def test_a_call_pi_did_not_run_because_it_was_rejected():
    data = render.build(EN_SCENARIOS["pi_rejected"]())
    card = render.step_html(data["steps"][0], data, lang="en", live=False)
    assert "you rejected it" in card and "not run: the harness reports the error itself" in card
    assert "The user refused to run this tool call" in card


def test_the_step_table_has_a_row_per_pi_request():
    rows = render.step_rows(render.build(SCENARIOS["pi"]()), lang="en")
    assert [(r["step"], r["context tokens"], r["generated tokens"], r["prefill ms"]) for r in rows] == [(1, 1200, 20, None), (2, 1200, 20, None)]


@pytest.mark.parametrize("lang", LANGS)
def test_pis_cards_in_each_language(lang):
    model = {"id": "claude-sonnet-5-5", "name": "Claude Sonnet 5.5", "provider": "anthropic", "api": "anthropic-messages",
             "reasoning": True, "input": ["text", "image"], "contextWindow": 200000, "maxTokens": 64000,
             "cost": {"input": 3, "output": 15, "cacheRead": 0.3, "cacheWrite": 3.75}}
    pieces = [render.pi_model_html(model, "1.4.0", lang), render.pi_settings_html({"thinking": "high", "approve_tools": True, "max_steps": 6}, lang),
              render.pi_tools_html(pi_harness.TOOLS, ["read", "bash"], lang), render.PI_INTRO["page"][lang], render.PI_INTRO["pi"][lang]]
    for piece in pieces:
        assert_balanced(piece)
        if lang == "en":
            assert not CJK.search(piece), piece[:200]
    assert "200,000 tokens" in pieces[0] and "anthropic/claude-sonnet-5-5" in pieces[0]
    assert pieces[2].count('class="check warn"') == len(pi_harness.TOOLS) - 2  # the tools not offered are marked


def test_pis_card_escapes_what_pi_says_of_its_model():
    model = {"id": "<b>x</b>", "name": "<img src=x>", "provider": "p<i>", "api": "a", "input": ["text"]}
    card = render.pi_model_html(model, "<1.4>", "en")
    assert "<img" not in card and "<b>x" not in card and "&lt;1.4&gt;" in card


def test_a_pi_step_on_the_pages_model_draws_its_tokens_and_its_context():
    data = render.build(EN_SCENARIOS["pi_page_model"]())
    assert data["harness"] == "pi" and data["family"] is GEMMA4
    step = data["steps"][0]
    card = render.step_html(step, data, lang="en", live=False)
    assert "Context <small>everything the model sees in this step</small>" in card and 'class="tok ' in card
    assert "the model endpoint parsed it from the Gemma 4 format into JSON for pi" in card
    svg = svg_of(render.diagram_html(data, dark=False, lang="en"))
    assert "tokens out" in svg and "text out" not in svg
    assert step["request"] and "operating inside pi" in render.request_html(step)


def test_the_whole_request_marks_what_the_step_added():
    data = render.build(EN_SCENARIOS["pi"]())
    first, second = data["steps"]
    assert "<mark" not in render.request_html(first)
    marked = re.search(r'<mark class="added">(.*?)</mark>', render.request_html(second), re.S).group(1)
    assert "&quot;tool&quot;" in marked and "Mon Oct 26" in marked
    assert render.request_summary(second, "en").startswith("1,200 tokens, as the provider counts them · ")
    assert not CJK.search(render.REQUEST_LEGEND["en"])
