import json
import math
import re
import time

import httpx
import pytest

import tools
from tools import TOOLS, ToolError, calculator, get_current_time, read_wikipedia, search_wikipedia

CJK = re.compile(r"[\u4e00-\u9fff\uff00-\uffef]")


# ---------- schema and validation ----------

def test_registry_names_match():
    assert list(TOOLS) == ["get_current_time", "calculator", "search_wikipedia", "read_wikipedia"]
    for name, tool in TOOLS.items():
        assert tool.name == name
        assert set(tool.environment) == {"zh", "en"} and all(tool.environment.values())


def test_schema_is_openai_function_format():
    for tool in TOOLS.values():
        schema = tool.schema()
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == tool.name and fn["description"]
        assert fn["parameters"]["type"] == "object"
        for key in fn["parameters"].get("required", []):
            assert key in fn["parameters"]["properties"]


def test_validate_fills_defaults():
    args, errors = TOOLS["get_current_time"].validate({})
    assert errors == [] and args == {"timezone": "Asia/Taipei"}
    args, errors = TOOLS["search_wikipedia"].validate({"query": "ReAct"})
    assert errors == [] and args == {"query": "ReAct", "lang": "en"}


def test_validate_keeps_given_values_over_defaults():
    args, errors = TOOLS["search_wikipedia"].validate({"query": "ReAct", "lang": "zh"})
    assert errors == [] and args["lang"] == "zh"


def test_validate_reports_missing_required():
    _, errors = TOOLS["calculator"].validate({})
    assert errors == ["缺少必要參數 expression"]


def test_validate_reports_unknown_parameter():
    _, errors = TOOLS["calculator"].validate({"expression": "1", "precision": 2})
    assert errors == ["沒有 precision 這個參數"]


def test_validate_reports_wrong_type():
    _, errors = TOOLS["calculator"].validate({"expression": 42})
    assert errors == ["expression 應該是 string，收到 int"]


def test_validate_reports_enum_violation():
    _, errors = TOOLS["read_wikipedia"].validate({"title": "Gemma", "lang": "fr"})
    assert len(errors) == 1 and errors[0].startswith("lang 只能是")


def test_validate_speaks_english_when_asked():
    tool = TOOLS["read_wikipedia"]
    assert tool.validate({}, "en")[1] == ["Missing required argument title"]
    assert tool.validate({"title": "x", "n": 1}, "en")[1] == ["There is no argument n"]
    assert tool.validate({"title": 1}, "en")[1] == ["title should be string, got int"]
    assert tool.validate({"title": "x", "lang": "fr"}, "en")[1] == ["lang must be one of ['en', 'zh']"]


def test_bool_is_not_a_number():
    assert tools.TYPE_CHECKS["integer"](True) is False
    assert tools.TYPE_CHECKS["number"](False) is False
    assert tools.TYPE_CHECKS["number"](1.5) and tools.TYPE_CHECKS["integer"](3)


# ---------- calculator ----------

@pytest.mark.parametrize("expr, expected", [
    ("1 + 2 * 3", 7),
    ("(1 + 2) * 3", 9),
    ("7 / 2", 3.5),
    ("7 // 2", 3),
    ("7 % 3", 1),
    ("2 ** 10", 1024),
    ("2 ^ 10", 1024),        # ^ means power, not xor
    ("6 × 7", 42),
    ("84 ÷ 2", 42),
    ("-3 + +5", 2),
    ("sqrt(16)", 4),          # whole floats come back as int
    ("factorial(5)", 120),
    ("abs(-2.5)", 2.5),
    ("round(2.567, 2)", 2.57),
    ("round(1234, -2)", 1200),
    ("round(5, -2000)", 0),
    ("log10(1000)", 3),
    ("floor(2.7) + ceil(2.1)", 5),
])
def test_calculator_arithmetic(expr, expected):
    out = calculator(expr)
    assert out == {"expression": expr, "result": expected}
    assert type(out["result"]) is type(expected)


def test_calculator_constants():
    assert calculator("pi")["result"] == pytest.approx(math.pi)
    assert calculator("e ** 2")["result"] == pytest.approx(math.e ** 2)


def test_calculator_readme_example():
    assert calculator("(37.5 * 12) ** 0.5 + 2 ** 20")["result"] == pytest.approx(450 ** 0.5 + 2 ** 20)


def test_calculator_keeps_huge_floats_as_float():
    assert isinstance(calculator("10.0 ** 20")["result"], float)


@pytest.mark.parametrize("expr, message", [
    ("1 +", "語法錯誤"),
    ("1 / 0", "division by zero"),
    ("sqrt(-1)", "math domain error"),
    ("2 ** 7000", "結果超過 2000 位數"),
    ("factorial(1000)", "factorial 的參數太大"),
    ("round(5, 2001)", "round 的位數參數太大"),
    ("10.0 ** 400", "超出浮點數範圍"),
    ("exp(1000)", "超出浮點數範圍"),
    ("(10 ** 400) ** 0.5", "超出浮點數範圍"),
    ("factorial()", "argument"),
    ("sqrt(1, 2)", "argument"),
    ("factorial(2.5)", "integer"),
    ("(-8) ** (1/3)", "複數"),
    ("10**999 * 10**999 * 10**999", "結果超過 2000 位數"),
    ("1e308 * 10", "超出浮點數範圍"),
    ("1e308*10 - 1e308*10", "超出浮點數範圍"),  # inf - inf is nan
    ("(1e308*10) * 0", "超出浮點數範圍"),
])
def test_calculator_errors_are_tool_errors(expr, message):
    with pytest.raises(ToolError, match=message) as error:
        calculator(expr)
    assert not CJK.search(error.value.text["en"])  # the model of an English page reads it in English


def test_calculator_refuses_nested_powers_without_computing_them():
    start = time.perf_counter()
    with pytest.raises(ToolError, match="結果超過 2000 位數"):
        calculator("((10**999)**999)**999")
    assert time.perf_counter() - start < 1


def test_calculator_refuses_round_to_huge_negative_digits_without_computing_them():
    start = time.perf_counter()
    with pytest.raises(ToolError, match="round 的位數參數太大"):
        calculator("round(5, -10**9)")  # int.__round__ would compute 10 ** 10**9 first
    assert time.perf_counter() - start < 1


@pytest.mark.parametrize("expr, digits", [("2 ** 3000", 904), ("factorial(500)", 1135), ("10 ** 1999", 2000)])
def test_calculator_allows_big_but_printable_integers(expr, digits):
    result = calculator(expr)["result"]
    assert len(str(result)) == digits
    json.dumps(result)  # the harness can turn it into an observation


def test_calculator_small_powers_of_big_numbers_are_fine():
    assert calculator("10 ** -5000")["result"] == 0
    assert calculator("0.5 ** 5000")["result"] == 0
    assert calculator("(10 ** 1500) ** 1")["result"] == 10 ** 1500


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')",
    "open('/etc/passwd')",
    "(1).__class__",
    "x + 1",
    "[1, 2]",
    "'a' * 3",
    "True + 1",
    "sqrt(x=4)",
    "lambda: 1",
    "1 if 1 else 2",
    "1 < 2",
])
def test_calculator_refuses_anything_but_arithmetic(expr):
    with pytest.raises(ToolError, match="不支援的運算"):
        calculator(expr)


# ---------- clock ----------

def test_get_current_time_formats_fields():
    out = get_current_time("Asia/Taipei")
    assert out["timezone"] == "Asia/Taipei"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", out["datetime"])
    assert out["utc_offset"] == "+0800"
    assert out["weekday"] in {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}


def test_get_current_time_default_is_taipei():
    assert get_current_time()["timezone"] == "Asia/Taipei"


@pytest.mark.parametrize("tz", ["Taipei", "Mars/Olympus_Mons", "../etc"])
def test_get_current_time_rejects_unknown_timezone(tz):
    with pytest.raises(ToolError, match="不認得時區") as error:
        get_current_time(tz)
    assert error.value.text["en"].startswith(f"Unknown timezone {tz}; use an IANA name")


# ---------- Wikipedia (HTTP mocked, no network) ----------

class FakeHTTP:
    def __init__(self, status=200, payload=None):
        self.status, self.payload, self.calls = status, payload or {}, []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return httpx.Response(self.status, json=self.payload, request=httpx.Request("GET", url))


def test_search_wikipedia_strips_highlight_markup(monkeypatch):
    http = FakeHTTP(payload={"query": {"search": [
        {"title": "ReAct", "snippet": '<span class="searchmatch">ReAct</span> is a prompting method'},
        {"title": "React (software)", "snippet": "A JavaScript library"},
    ]}})
    monkeypatch.setattr(tools.httpx, "get", http)
    out = search_wikipedia("ReAct", lang="zh")
    assert out == {"query": "ReAct", "results": [
        {"title": "ReAct", "snippet": "ReAct is a prompting method"},
        {"title": "React (software)", "snippet": "A JavaScript library"},
    ]}
    [call] = http.calls
    assert call["url"] == "https://zh.wikipedia.org/w/api.php"
    assert call["params"]["srsearch"] == "ReAct" and call["params"]["srlimit"] == 5
    assert "User-Agent" in call["headers"]


def test_search_wikipedia_with_no_hits(monkeypatch):
    monkeypatch.setattr(tools.httpx, "get", FakeHTTP(payload={}))
    assert search_wikipedia("qwertyuiop") == {"query": "qwertyuiop", "results": []}


def test_search_wikipedia_raises_on_server_error(monkeypatch):
    monkeypatch.setattr(tools.httpx, "get", FakeHTTP(status=503))
    with pytest.raises(httpx.HTTPStatusError):
        search_wikipedia("ReAct")


def test_read_wikipedia_returns_summary_and_quotes_title(monkeypatch):
    http = FakeHTTP(payload={
        "title": "Gemma (language model)",
        "extract": "Gemma is a family of open models by Google DeepMind.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Gemma_(language_model)"}},
    })
    monkeypatch.setattr(tools.httpx, "get", http)
    out = read_wikipedia("Gemma (language model)")
    assert out == {
        "title": "Gemma (language model)",
        "summary": "Gemma is a family of open models by Google DeepMind.",
        "url": "https://en.wikipedia.org/wiki/Gemma_(language_model)",
    }
    assert http.calls[0]["url"] == "https://en.wikipedia.org/api/rest_v1/page/summary/Gemma_%28language_model%29"
    assert http.calls[0]["follow_redirects"] is True


def test_read_wikipedia_quotes_slashes_and_unicode(monkeypatch):
    http = FakeHTTP(payload={})
    monkeypatch.setattr(tools.httpx, "get", http)
    out = read_wikipedia("AC/DC 樂團", lang="zh")
    assert http.calls[0]["url"].endswith("/page/summary/AC%2FDC_%E6%A8%82%E5%9C%98")
    assert out == {"title": "AC/DC 樂團", "summary": "", "url": ""}


def test_read_wikipedia_404_suggests_search(monkeypatch):
    monkeypatch.setattr(tools.httpx, "get", FakeHTTP(status=404))
    with pytest.raises(ToolError, match="search_wikipedia"):
        read_wikipedia("No Such Page")
