import pytest

from agent.model_families.gemma4 import ParseError, parse_args, parse_output

S = '<|"|>'


# ---------- parse_output: splitting one model turn ----------

def test_plain_reply_strips_stop_marker():
    out = parse_output("台北現在是下午三點。<turn|>")
    assert out == {"thinking": None, "tool_calls": [], "content": "台北現在是下午三點。"}


@pytest.mark.parametrize("marker", ["<turn|>", "<|tool_response>", "<eos>"])
def test_every_end_marker_is_removed(marker):
    assert parse_output(f"hi{marker}")["content"] == "hi"


def test_tool_call_is_parsed_and_removed_from_content():
    raw = f"<|tool_call>call:get_current_time{{timezone:{S}Asia/Taipei{S}}}<tool_call|><|tool_response>"
    out = parse_output(raw)
    assert out["content"] == ""
    assert out["tool_calls"] == [{
        "name": "get_current_time",
        "raw_args": f"{{timezone:{S}Asia/Taipei{S}}}",
        "arguments": {"timezone": "Asia/Taipei"},
        "error": None,
    }]


def test_text_before_a_tool_call_stays_as_content():
    raw = f"我來算一下。<|tool_call>call:calculator{{expression:{S}1+1{S}}}<tool_call|><|tool_response>"
    out = parse_output(raw)
    assert out["content"] == "我來算一下。"
    assert out["tool_calls"][0]["arguments"] == {"expression": "1+1"}


def test_several_tool_calls_keep_their_order():
    raw = (
        f"<|tool_call>call:get_current_time{{timezone:{S}Asia/Taipei{S}}}<tool_call|>"
        f"<|tool_call>call:get_current_time{{timezone:{S}America/New_York{S}}}<tool_call|><|tool_response>"
    )
    calls = parse_output(raw)["tool_calls"]
    assert [c["arguments"]["timezone"] for c in calls] == ["Asia/Taipei", "America/New_York"]


def test_thinking_channel_is_split_out():
    raw = "<|channel>thought\n要查時間。\n<channel|>好的。<turn|>"
    out = parse_output(raw)
    assert out["thinking"] == "要查時間。"
    assert out["content"] == "好的。"


def test_unclosed_thinking_hides_the_rest():
    out = parse_output("<|channel>thought\n還在想，被 token 上限截斷")
    assert out["thinking"] == "還在想，被 token 上限截斷"
    assert out["content"] == ""
    assert out["tool_calls"] == []


def test_tool_call_after_thinking():
    raw = f"<|channel>thought\n用計算機<channel|><|tool_call>call:calculator{{expression:{S}2**10{S}}}<tool_call|>"
    out = parse_output(raw)
    assert out["thinking"] == "用計算機"
    assert out["tool_calls"][0]["arguments"] == {"expression": "2**10"}


def test_unterminated_tool_call_is_reported():
    out = parse_output(f"先查一下<|tool_call>call:search_wikipedia{{query:{S}ReAct")
    assert out["content"] == "先查一下"
    [call] = out["tool_calls"]
    assert call["name"] == "search_wikipedia"
    assert call["arguments"] is None
    assert "<tool_call|>" in call["error"]


def test_unterminated_tool_call_without_name():
    [call] = parse_output("<|tool_call>")["tool_calls"]
    assert call["name"] == "?"
    assert call["error"]


def test_malformed_arguments_keep_the_call_with_an_error():
    [call] = parse_output("<|tool_call>call:calculator{expression}<tool_call|>")["tool_calls"]
    assert call["name"] == "calculator"
    assert call["arguments"] is None
    assert call["error"]


# ---------- parse_args: Gemma's not-quite-JSON ----------

def test_empty_object():
    assert parse_args("{}") == {}
    assert parse_args("  { }  ") == {}


def test_delimited_strings_may_hold_anything():
    value = 'a, b: {c} "quoted" [x]'
    assert parse_args(f"{{q:{S}{value}{S}}}") == {"q": value}


def test_scalars():
    args = parse_args("{a:1,b:-2.5,c:1e3,d:true,e:false,f:null,g:Asia/Taipei}")
    assert args == {"a": 1, "b": -2.5, "c": 1000.0, "d": True, "e": False, "f": None, "g": "Asia/Taipei"}
    assert isinstance(args["a"], int) and isinstance(args["c"], float)


def test_nested_objects_and_arrays():
    text = f"{{filter:{{lang:{S}en{S},tags:[{S}a{S},2,[true]]}},empty:[]}}"
    assert parse_args(text) == {"filter": {"lang": "en", "tags": ["a", 2, [True]]}, "empty": []}


def test_json_style_strings_and_keys_are_accepted():
    assert parse_args('{"query": "ReAct paper", n: 3}') == {"query": "ReAct paper", "n": 3}


def test_delimited_keys():
    assert parse_args(f"{{{S}my key{S}:1}}") == {"my key": 1}


def test_whitespace_between_tokens():
    assert parse_args(f"{{ a : 1 ,\n b : {S}x{S} }}") == {"a": 1, "b": "x"}


@pytest.mark.parametrize("text", [
    "",                       # nothing
    "[1,2]",                  # not an object
    "{a:1} extra",            # trailing characters
    "{a:1",                   # unclosed object
    "{a 1}",                  # missing colon
    "{a:1 b:2}",              # missing comma
    f"{{a:{S}unclosed}}",     # unclosed <|"|> string
    '{a:"bad \\q escape"}',   # broken JSON string
    "{1a:2}",                 # key cannot start with a digit
    "{a:[1,2}",               # unclosed array
])
def test_malformed_arguments_raise(text):
    with pytest.raises(ParseError) as error:
        parse_args(text)
    zh, en = error.value.text["zh"], error.value.text["en"]
    assert zh and en and en.isascii()


def test_errors_are_worded_in_the_language_asked_for():
    [call] = parse_output("<|tool_call>call:calculator{expression}<tool_call|>", lang="en")["tool_calls"]
    assert call["error"].startswith("Expected ':' at character ")
    [cut] = parse_output("<|tool_call>call:calculator{", lang="en")["tool_calls"]
    assert cut["error"] == "The tool call has no closing <tool_call|>"
