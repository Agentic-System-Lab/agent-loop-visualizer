import base64
import io
import json
import math
import re
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import samples
from harness import tools
from harness.tools import TOOLS, ToolError, ToolResult, calculator, get_current_time, read_wikipedia, run_python, search_wikipedia

CJK = re.compile(r"[\u4e00-\u9fff\uff00-\uffef]")
SANDBOX = Path(__file__).resolve().parent.parent / "harness" / "sandbox"


# ---------- schema and validation ----------

def test_registry_names_match():
    assert list(TOOLS) == ["get_current_time", "calculator", "search_wikipedia", "read_wikipedia", "run_python"]
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


# ---------- run_python: the model's code in Docker ----------

class FakeDocker:
    """Stands in for subprocess.run: records each command and answers with a scripted result."""

    def __init__(self, returncode=0, stdout="", stderr="", raises=None):
        self.result = SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
        self.raises, self.calls = raises, []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self.raises and cmd[:2] == ["docker", "run"]:
            raise self.raises
        return self.result


def test_run_python_runs_the_code_in_a_locked_down_container(monkeypatch, tmp_path):
    docker = FakeDocker(stdout="42\n")
    monkeypatch.setattr(tools.subprocess, "run", docker)
    assert run_python("print(6 * 7)", tmp_path) == {"stdout": "42\n"}
    [(cmd, kwargs)] = docker.calls
    assert cmd[:3] == ["docker", "run", "--rm"] and cmd[-4:] == [tools.SANDBOX_IMAGE, "python", "-I", "-"]
    flags = " ".join(cmd)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges", f"--user {tools.sandbox_user()}", "--memory 512m"):
        assert flag in flags
    assert f"-v {tmp_path.resolve()}:/data:ro" in flags  # the conversation's files, read-only
    assert f"-e AGENT_LOOP_FIGURES={tools.FIGURES_MARK}" in flags  # where the charts start in what it prints
    assert kwargs["input"] == "print(6 * 7)" and kwargs["timeout"] == tools.SANDBOX_TIMEOUT


def test_the_sandbox_runs_as_the_host_user_and_never_as_root(monkeypatch):
    monkeypatch.setattr(tools.os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(tools.os, "getgid", lambda: 1001, raising=False)
    assert tools.sandbox_user() == "1000:1001"
    monkeypatch.setattr(tools.os, "getuid", lambda: 0)
    assert tools.sandbox_user() == "65534:65534"
    monkeypatch.delattr(tools.os, "getuid")  # Windows
    assert tools.sandbox_user() == "65534:65534"


def test_run_python_without_files_mounts_nothing(monkeypatch):
    docker = FakeDocker(stdout="ok\n")
    monkeypatch.setattr(tools.subprocess, "run", docker)
    run_python("print('ok')")
    assert "-v" not in docker.calls[0][0]


@pytest.mark.parametrize("result, zh, en", [
    (dict(returncode=1, stdout="half\n", stderr="Traceback …\nZeroDivisionError: division by zero\n"),
     "程式以結束代碼 1 結束：\nhalf\nTraceback …\nZeroDivisionError: division by zero",
     "The code exited with code 1:\nhalf\nTraceback …\nZeroDivisionError: division by zero"),
    (dict(returncode=0, stdout="  \n"), "程式沒有印出任何東西；用 print() 印出想看的結果", "The code printed nothing; print() what you want to see"),
    (dict(returncode=137), "程式用掉超過 512 MB 記憶體，被停止了", "The code used more than 512 MB of memory and was stopped"),
    (dict(returncode=125, stderr="Unable to find image\n"), "Docker 無法啟動沙箱：Unable to find image", "Docker could not start the sandbox: Unable to find image"),
])
def test_run_python_reports_what_went_wrong_in_both_languages(monkeypatch, result, zh, en):
    monkeypatch.setattr(tools.subprocess, "run", FakeDocker(**result))
    with pytest.raises(ToolError) as e:
        run_python("x")
    assert e.value.text == {"zh": zh, "en": en}


def test_run_python_stops_code_that_runs_too_long(monkeypatch):
    docker = FakeDocker(raises=subprocess.TimeoutExpired("docker", tools.SANDBOX_TIMEOUT))
    monkeypatch.setattr(tools.subprocess, "run", docker)
    with pytest.raises(ToolError, match=f"超過 {tools.SANDBOX_TIMEOUT} 秒"):
        run_python("while True: pass")
    name = docker.calls[0][0][docker.calls[0][0].index("--name") + 1]
    assert docker.calls[1][0] == ["docker", "rm", "-f", name]  # killing the client leaves the container running


def test_run_python_keeps_the_start_of_long_output(monkeypatch):
    monkeypatch.setattr(tools.subprocess, "run", FakeDocker(stdout="x" * 5000))
    out = run_python("print('x' * 5000)")["stdout"]
    assert out == "x" * tools.MAX_OUTPUT_CHARS + "…(truncated)"


def png(width=1) -> str:
    """A tiny real PNG, base64, as the sandbox writes each chart after FIGURES_MARK."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, 1)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_run_python_hands_the_charts_to_the_page_and_tells_the_model_where_they_are(monkeypatch, tmp_path):
    monkeypatch.setattr(tools.subprocess, "run", FakeDocker(stdout=f"2025-12 244320\n\n{tools.FIGURES_MARK}\n{png(1)}\n{png(2)}\n"))
    result = run_python("…", tmp_path)
    assert result == ToolResult({"stdout": "2025-12 244320\n", "figures": ["/data/figure-1.png", "/data/figure-2.png"]},
                                [str(tmp_path / "figure-1.png"), str(tmp_path / "figure-2.png")])
    assert (tmp_path / "figure-2.png").read_bytes() == base64.b64decode(png(2))


def test_a_chart_alone_is_an_answer(monkeypatch, tmp_path):
    monkeypatch.setattr(tools.subprocess, "run", FakeDocker(stdout=f"\n{tools.FIGURES_MARK}\n{png()}\n"))
    assert run_python("plt.show()", tmp_path).data == {"figures": ["/data/figure-1.png"]}  # not "printed nothing"


def test_charts_never_overwrite_a_file_and_only_pngs_are_kept(monkeypatch, tmp_path):
    (tmp_path / "figure-1.png").write_bytes(b"the user's own")
    lines = ["not-base64!", base64.b64encode(b"GIF89a...").decode(), *(png(n) for n in range(1, 6))]
    monkeypatch.setattr(tools.subprocess, "run", FakeDocker(stdout=f"ok\n\n{tools.FIGURES_MARK}\n" + "\n".join(lines) + "\n"))
    result = run_python("…", tmp_path)
    assert result.data["figures"] == ["/data/figure-2.png", "/data/figure-3.png"]  # of the first MAX_FIGURES lines, two are PNGs
    assert (tmp_path / "figure-1.png").read_bytes() == b"the user's own"


def test_failing_code_drops_its_charts(monkeypatch, tmp_path):
    monkeypatch.setattr(tools.subprocess, "run", FakeDocker(returncode=1, stdout=f"half\n\n{tools.FIGURES_MARK}\n{png()}\n", stderr="KeyError: 'Month'\n"))
    with pytest.raises(ToolError) as e:
        run_python("…", tmp_path)
    assert e.value.text["en"] == "The code exited with code 1:\nhalf\nKeyError: 'Month'"  # no base64 for the model to read
    assert list(tmp_path.iterdir()) == []


def test_without_a_folder_the_charts_have_nowhere_to_go(monkeypatch):
    monkeypatch.setattr(tools.subprocess, "run", FakeDocker(stdout=f"ok\n\n{tools.FIGURES_MARK}\n{png()}\n"))
    assert run_python("…") == {"stdout": "ok\n"}


def test_the_sandbox_image_agrees_with_tools():
    dockerfile = (SANDBOX / "Dockerfile").read_text()
    assert f'LABEL agent-loop-sandbox.version="{tools.SANDBOX_VERSION}"' in dockerfile
    assert f"MAX_FIGURES = {tools.MAX_FIGURES}" in (SANDBOX / "agent_loop_figures.py").read_text()
    assert "backend: module://agent_loop_figures" in (SANDBOX / "matplotlibrc").read_text()


def test_the_sandbox_image_installs_a_pip_without_the_revoked_certificates_by_hash():
    """pip below 26.2.2 also trusts its own copy of certifi, which still has eMudhra's compromised emSign certificates."""
    [line] = [l for l in (SANDBOX / "Dockerfile").read_text().splitlines() if '"pip @ ' in l]
    version = re.search(r"/pip-([\d.]+)-py3-none-any\.whl#sha256=[0-9a-f]{64}\"", line).group(1)
    assert tuple(map(int, version.split("."))) >= (26, 2, 2)


@pytest.mark.parametrize("docker, words", [
    (FakeDocker(returncode=0, stdout=f"{tools.SANDBOX_VERSION}\n"), None),
    (FakeDocker(returncode=0, stdout="\n"), ("舊版", "out of date")),  # built before the image had a version
    (FakeDocker(returncode=0, stdout="1\n"), ("舊版", "out of date")),
    (FakeDocker(returncode=1, stderr="Error: No such image: agent-loop-sandbox"), ("docker build -t agent-loop-sandbox harness/sandbox",) * 2),
    (FakeDocker(returncode=1, stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock."), ("沒有在執行", "not running")),
])
def test_sandbox_problem_says_what_is_missing(monkeypatch, docker, words):
    monkeypatch.setattr(tools.subprocess, "run", docker)
    problem = tools.sandbox_problem()
    if words is None:
        assert problem is None
    else:
        assert words[0] in problem["zh"] and words[1] in problem["en"]
        assert not CJK.search(problem["en"])


def test_sandbox_problem_without_docker_links_to_its_installation(monkeypatch):
    def missing(cmd, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(tools.subprocess, "run", missing)
    problem = tools.sandbox_problem()
    for lang in ("zh", "en"):
        assert f"]({tools.DOCKER_INSTALL})" in problem[lang] and "docker build -t agent-loop-sandbox harness/sandbox" in problem[lang]


@pytest.mark.skipif(tools.sandbox_problem() is not None, reason="needs Docker and the agent-loop-sandbox image")
def test_the_real_sandbox_reads_the_files_but_cannot_change_them_or_reach_the_network(tmp_path):
    (tmp_path / "sales.csv").write_text("month,revenue\nJan,10\nFeb,32\n")
    code = """
import pandas as pd, socket
print(pd.read_csv("/data/sales.csv").revenue.sum())
try:
    open("/data/new.txt", "w")
except OSError as e:
    print("read-only", e.errno)
try:
    socket.create_connection(("1.1.1.1", 53), timeout=2)
except OSError:
    print("no network")
"""
    assert run_python(code, tmp_path)["stdout"].split("\n")[:3] == ["42", "read-only 30", "no network"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sales.csv"]


READERS = {  # code that reads each data format and prints the year's revenue
    "csv": 'import pandas as pd; print(pd.read_csv("/data/sample.csv").revenue.sum())',
    "tsv": 'import pandas as pd; print(pd.read_csv("/data/sample.tsv", sep="\\t").revenue.sum())',
    "json": 'import pandas as pd; print(pd.read_json("/data/sample.json").revenue.sum())',
    "xlsx": 'import pandas as pd; print(pd.read_excel("/data/sample.xlsx").revenue.sum())',
    "txt": 'import re; print(sum(int(n) for n in re.findall(r"revenue (\\d+)", open("/data/sample.txt").read())))',
    "md": 'print(sum(int(l.split("|")[4]) for l in open("/data/sample.md") if l.startswith("| 2025")))',
}


def test_there_is_a_reader_for_every_data_format():
    assert sorted(READERS) == sorted(samples.DATA)


@pytest.mark.skipif(tools.sandbox_problem() is not None, reason="needs Docker and the agent-loop-sandbox image")
@pytest.mark.parametrize("ext", samples.DATA)
def test_the_real_sandbox_reads_every_data_format(tmp_path, ext):
    samples.make(tmp_path, ext)
    assert run_python(READERS[ext], tmp_path)["stdout"].strip() == str(samples.YEAR_REVENUE)


@pytest.mark.skipif(tools.sandbox_problem() is not None, reason="needs Docker and the agent-loop-sandbox image")
def test_the_real_sandbox_sends_back_its_charts_with_chinese_labels(tmp_path):
    code = """
import warnings
warnings.simplefilter("error")  # a character missing from every font is a warning, and so an error here
import matplotlib.pyplot as plt
plt.bar(["一月", "二月"], [3, -5])
plt.title("每月營收")
plt.show()
plt.plot([1, 2, 3])  # left open: sent when the program ends
print("done")
"""
    result = run_python(code, tmp_path)
    assert result.data == {"stdout": "done\n", "figures": ["/data/figure-1.png", "/data/figure-2.png"]}
    from PIL import Image

    for path in result.images:
        with Image.open(path) as chart:
            assert chart.format == "PNG" and chart.width > 300


@pytest.mark.skipif(tools.sandbox_problem() is not None, reason="needs Docker and the agent-loop-sandbox image")
def test_the_real_sandbox_has_a_pip_that_no_longer_trusts_the_revoked_certificates():
    code = "import pip; from pip._vendor import certifi; print(pip.__version__, certifi.__version__)"
    pip, certifi = run_python(code)["stdout"].split()
    assert tuple(map(int, pip.split("."))) >= (26, 2, 2)
    assert tuple(map(int, certifi.split("."))) >= (2026, 10, 19)
