"""Read Gemma 4 output: the thinking channel, tool calls, and the reply.

Gemma 4 writes a tool call as `<|tool_call>call:name{key:<|"|>text<|"|>,n:3}<tool_call|>`.
Keys are bare words and strings are wrapped in the `<|"|>` token instead of quotes,
so the arguments are not JSON and need their own small parser.
"""

import json
import re

STR = '<|"|>'
THOUGHT_OPEN = "<|channel>thought\n"
THOUGHT_CLOSE = "<channel|>"
CALL_OPEN = "<|tool_call>"
CALL_RE = re.compile(r"<\|tool_call>call:(\w+)(.*?)<tool_call\|>", re.S)
END_MARKERS = ("<turn|>", "<|tool_response>", "<eos>")
KEY_RE = re.compile(r"[A-Za-z_][\w.-]*")
SCALAR_RE = re.compile(r"[^,}\]\s]+")


class ParseError(ValueError):
    pass


def parse_output(raw: str) -> dict:
    """Split one model turn into thinking, tool calls and visible content."""
    text = raw
    thinking = None
    if THOUGHT_OPEN in text:
        before, _, rest = text.partition(THOUGHT_OPEN)
        thought, closed, after = rest.partition(THOUGHT_CLOSE)
        thinking = thought.strip()
        text = before + (after if closed else "")

    calls = []
    for m in CALL_RE.finditer(text):
        call = {"name": m.group(1), "raw_args": m.group(2), "arguments": None, "error": None}
        try:
            call["arguments"] = parse_args(m.group(2))
        except ParseError as e:
            call["error"] = str(e)
        calls.append(call)

    rest = CALL_RE.sub("", text)
    if CALL_OPEN in rest:
        head, _, tail = rest.partition(CALL_OPEN)
        name = re.match(r"call:(\w+)", tail)
        calls.append({
            "name": name.group(1) if name else "?",
            "raw_args": tail,
            "arguments": None,
            "error": "工具呼叫沒有結束標記 <tool_call|>",
        })
        rest = head

    for marker in END_MARKERS:
        rest = rest.replace(marker, "")
    return {"thinking": thinking, "tool_calls": calls, "content": rest.strip()}


def parse_args(text: str) -> dict:
    parser = _Parser(text.strip())
    value = parser.value()
    parser.skip_ws()
    if parser.pos != len(parser.text):
        raise ParseError(f"參數後面還有多餘的字元：{parser.text[parser.pos:][:30]!r}")
    if not isinstance(value, dict):
        raise ParseError("參數必須是 {...} 物件")
    return value


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def skip_ws(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def expect(self, ch: str):
        self.skip_ws()
        if self.peek() != ch:
            raise ParseError(f"第 {self.pos} 個字元應該是 {ch!r}")
        self.pos += 1

    def value(self):
        self.skip_ws()
        if self.text.startswith(STR, self.pos):
            return self.delimited()
        ch = self.peek()
        if ch == "{":
            return self.obj()
        if ch == "[":
            return self.arr()
        if ch == '"':
            return self.json_string()
        return self.scalar()

    def delimited(self) -> str:
        start = self.pos + len(STR)
        end = self.text.find(STR, start)
        if end < 0:
            raise ParseError('字串缺少結尾的 <|"|>')
        self.pos = end + len(STR)
        return self.text[start:end]

    def json_string(self) -> str:
        try:
            value, self.pos = json.JSONDecoder().raw_decode(self.text, self.pos)
        except json.JSONDecodeError as e:
            raise ParseError(f"字串格式錯誤：{e.msg}") from e
        return value

    def key(self) -> str:
        self.skip_ws()
        if self.text.startswith(STR, self.pos):
            return self.delimited()
        if self.peek() == '"':
            return self.json_string()
        m = KEY_RE.match(self.text, self.pos)
        if not m:
            raise ParseError(f"第 {self.pos} 個字元應該是參數名稱")
        self.pos = m.end()
        return m.group(0)

    def obj(self) -> dict:
        self.pos += 1
        out = {}
        self.skip_ws()
        if self.peek() == "}":
            self.pos += 1
            return out
        while True:
            key = self.key()
            self.expect(":")
            out[key] = self.value()
            self.skip_ws()
            ch = self.peek()
            self.pos += 1
            if ch == "}":
                return out
            if ch != ",":
                raise ParseError(f"物件裡第 {self.pos - 1} 個字元應該是 ',' 或 '}}'")

    def arr(self) -> list:
        self.pos += 1
        out = []
        self.skip_ws()
        if self.peek() == "]":
            self.pos += 1
            return out
        while True:
            out.append(self.value())
            self.skip_ws()
            ch = self.peek()
            self.pos += 1
            if ch == "]":
                return out
            if ch != ",":
                raise ParseError(f"陣列裡第 {self.pos - 1} 個字元應該是 ',' 或 ']'")

    def scalar(self):
        m = SCALAR_RE.match(self.text, self.pos)
        if not m:
            raise ParseError(f"第 {self.pos} 個字元應該是一個值")
        self.pos = m.end()
        token = m.group(0)
        if token in ("true", "false"):
            return token == "true"
        if token == "null":
            return None
        for cast in (int, float):
            try:
                return cast(token)
            except ValueError:
                pass
        return token
