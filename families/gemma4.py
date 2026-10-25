"""The Gemma 4 family: reading its output, and what its control tokens mean.

Gemma 4 writes a tool call as `<|tool_call>call:name{key:<|"|>text<|"|>,n:3}<tool_call|>`.
Keys are bare words and strings are wrapped in the `<|"|>` token instead of quotes,
so the arguments are not JSON and need their own small parser.
"""

import json
import re

from families import Family, FileNote
from i18n import pick

STR = '<|"|>'
THOUGHT_OPEN = "<|channel>thought\n"
THOUGHT_CLOSE = "<channel|>"
CALL_OPEN = "<|tool_call>"
CALL_RE = re.compile(r"<\|tool_call>call:(\w+)(.*?)<tool_call\|>", re.S)
END_MARKERS = ("<turn|>", "<|tool_response>", "<eos>")
KEY_RE = re.compile(r"[A-Za-z_][\w.-]*")
SCALAR_RE = re.compile(r"[^,}\]\s]+")


class ParseError(ValueError):
    """What is wrong with the arguments, in both languages; the model reads it in the run's."""

    def __init__(self, zh: str, en: str):
        super().__init__(zh)
        self.text = {"zh": zh, "en": en}


def parse_output(raw: str, schemas: dict | None = None, lang: str = "zh") -> dict:
    """Split one model turn into thinking, tool calls and visible content. Gemma marks each value's
    type itself, so the tool schemas are not needed. Errors are worded in `lang`."""
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
            call["error"] = e.text[lang]
        calls.append(call)

    rest = CALL_RE.sub("", text)
    if CALL_OPEN in rest:
        head, _, tail = rest.partition(CALL_OPEN)
        name = re.match(r"call:(\w+)", tail)
        calls.append({
            "name": name.group(1) if name else "?",
            "raw_args": tail,
            "arguments": None,
            "error": pick(lang, "工具呼叫沒有結束標記 <tool_call|>", "The tool call has no closing <tool_call|>"),
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
        rest = parser.text[parser.pos:][:30]
        raise ParseError(f"參數後面還有多餘的字元：{rest!r}", f"Extra characters after the arguments: {rest!r}")
    if not isinstance(value, dict):
        raise ParseError("參數必須是 {...} 物件", "The arguments must be a {...} object")
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
            raise ParseError(f"第 {self.pos} 個字元應該是 {ch!r}", f"Expected {ch!r} at character {self.pos}")
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
            raise ParseError('字串缺少結尾的 <|"|>', 'The string has no closing <|"|>')
        self.pos = end + len(STR)
        return self.text[start:end]

    def json_string(self) -> str:
        try:
            value, self.pos = json.JSONDecoder().raw_decode(self.text, self.pos)
        except json.JSONDecodeError as e:
            raise ParseError(f"字串格式錯誤：{e.msg}", f"Malformed string: {e.msg}") from e
        return value

    def key(self) -> str:
        self.skip_ws()
        if self.text.startswith(STR, self.pos):
            return self.delimited()
        if self.peek() == '"':
            return self.json_string()
        m = KEY_RE.match(self.text, self.pos)
        if not m:
            raise ParseError(f"第 {self.pos} 個字元應該是參數名稱", f"Expected an argument name at character {self.pos}")
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
                raise ParseError(f"物件裡第 {self.pos - 1} 個字元應該是 ',' 或 '}}'", f"Expected ',' or '}}' at character {self.pos - 1} of an object")

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
                raise ParseError(f"陣列裡第 {self.pos - 1} 個字元應該是 ',' 或 ']'", f"Expected ',' or ']' at character {self.pos - 1} of an array")

    def scalar(self):
        m = SCALAR_RE.match(self.text, self.pos)
        if not m:
            raise ParseError(f"第 {self.pos} 個字元應該是一個值", f"Expected a value at character {self.pos}")
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


# What each control token is for, grouped as the model tab lists them, in each language; `backticks` mark literal text.
GROUPS = {"zh": [
    ("序列", "整段文字的頭尾", [
        ("<bos>", "序列開頭（beginning of sequence）", "整段脈絡的第一個 token，chat template 一定會把它放在最前面。"),
        ("<eos>", "序列結束（end of sequence）", "整段文字結束。它也是停止 token，不過對話中模型通常會先停在 `<turn|>`。"),
    ]),
    ("回合", "把對話切成一則一則訊息", [
        ("<|turn>", "回合開始（start of turn）",
         "後面接角色名稱 `system`、`user` 或 `model` 和換行，開始一則訊息。脈絡最後的 `<|turn>model` 就是請模型接著寫。"),
        ("<turn|>", "回合結束（end of turn）",
         "一則訊息到這裡結束。模型寫出它就停下來，代表回覆寫完了，相當於 Messages API 的 `stop_reason: end_turn`。"),
    ]),
    ("思考", "思考模式下，推理寫在 thought channel 裡", [
        ("<|think|>", "開啟思考模式（think）", "放在 system turn 的最前面，模型就會先寫推理再行動。側欄的「思考模式」就是加上這個 token。"),
        ("<|channel>", "channel 開始（start of channel）", "後面接 channel 名稱 `thought` 和換行，接著是模型的推理（step 卡片裡的斜體 token）。"),
        ("<channel|>", "channel 結束（end of channel）", "推理到這裡結束，後面才是給使用者的回覆或工具呼叫。"),
    ]),
    ("工具", "工具的宣告、呼叫與結果", [
        ("<|tool>", "工具宣告開始（start of tool declaration）",
         "system turn 裡每個工具寫成 `<|tool>declaration:名稱{…}<tool|>`，內容來自工具的 JSON Schema。模型就是從這裡知道有哪些工具。"),
        ("<tool|>", "工具宣告結束（end of tool declaration）", "一個工具的宣告到這裡結束。"),
        ("<|tool_call>", "工具呼叫開始（start of tool call）", "模型要呼叫工具了，後面寫 `call:名稱{參數}`。harness 從這裡開始解析。"),
        ("<tool_call|>", "工具呼叫結束（end of tool call）", "一個工具呼叫寫完，後面通常緊接著 `<|tool_response>`。"),
        ("<|tool_response>", "工具結果開始（start of tool response）",
         "模型寫出它就停下來等結果，相當於 `stop_reason: tool_use`。harness 執行工具後，把 `response:名稱{value:…}` 接在它後面。"),
        ("<tool_response|>", "工具結果結束（end of tool response）", "工具結果到這裡結束。模型還在同一個 model turn 裡，讀完結果接著寫。"),
        ('<|"|>', "字串引號（escape）",
         "工具宣告、呼叫和結果裡的字串前後都包這個 token，取代一般的引號，所以字串裡的引號和括號都不用跳脫。"),
    ]),
    ("多模態", "圖片、聲音和影片的位置；附上這類檔案時才會出現", [
        ("<|image|>", "圖片（image）",
         "chat template 在圖片的位置寫入它，處理器再展開成 `<|image>`、一串 `<|image|>`、`<image|>`，這些位置的 embedding 換成視覺編碼器的輸出。"),
        ("<|image>", "圖片開始（begin of image）", "一張圖片的 token 從這裡開始。"),
        ("<image|>", "圖片結束（end of image）", "一張圖片的 token 到這裡結束。"),
        ("<|audio|>", "聲音（audio）",
         "和 `<|image|>` 一樣是佔位 token：展開成 `<|audio>`、一串 `<|audio|>`、`<audio|>`，換成音訊編碼器的輸出。每 40 ms 一個 token，最多 750 個，所以只聽得到前 30 秒。"),
        ("<|audio>", "聲音開始（begin of audio）", "一段聲音的 token 從這裡開始。"),
        ("<audio|>", "聲音結束（end of audio）", "一段聲音的 token 到這裡結束。"),
        ("<|video|>", "影片（video）",
         "影片平均取 32 張畫格（不到 32 張就全部用），每張前面寫上時間 `mm:ss`，接著 `<|image>`、一串 `<|video|>`、`<image|>`，換成視覺編碼器的輸出。沒有聲音。"),
    ]),
    ("保留", "tokenizer 預留的，對話格式用不到", [
        ("<pad>", "填充（padding）", "把長短不同的序列補成一樣長，才能放進同一批推論，attention 會略過它。這裡一次只跑一條序列，用不到。"),
        ("<unk>", "未知（unknown）", "代替 tokenizer 切不出來的字元。Gemma 能把任何字元拆成 byte token，所以幾乎不會出現。"),
        ("<mask>", "遮罩（mask）", "保留的遮罩 token，chat template 不會用到。"),
    ]),
], "en": [
    ("Sequence", "where the whole text starts and ends", [
        ("<bos>", "Beginning of sequence", "The first token of the whole context; the chat template always puts it in front."),
        ("<eos>", "End of sequence", "The end of the whole text. It is also a stop token, though in a conversation the model usually stops at `<turn|>` first."),
    ]),
    ("Turns", "split the conversation into messages", [
        ("<|turn>", "Start of turn",
         "Followed by a role, `system`, `user` or `model`, and a newline, it starts a message. The `<|turn>model` at the end of the context asks the model to write next."),
        ("<turn|>", "End of turn",
         "A message ends here. When the model writes it, it stops because its reply is done, like `stop_reason: end_turn` in the Messages API."),
    ]),
    ("Thinking", "in thinking mode, the reasoning goes in the thought channel", [
        ("<|think|>", "Thinking on", "At the very start of the system turn, it makes the model reason before it acts. The sidebar's Thinking mode adds this token."),
        ("<|channel>", "Start of channel", "Followed by the channel name `thought` and a newline, then the model's reasoning (the italic tokens in a step card)."),
        ("<channel|>", "End of channel", "The reasoning ends here; the reply to the user or a tool call comes after it."),
    ]),
    ("Tools", "declaring tools, calling them and their results", [
        ("<|tool>", "Start of tool declaration",
         "In the system turn each tool is written as `<|tool>declaration:name{…}<tool|>`, from the tool's JSON Schema. This is how the model learns which tools it has."),
        ("<tool|>", "End of tool declaration", "One tool's declaration ends here."),
        ("<|tool_call>", "Start of tool call", "The model is calling a tool and writes `call:name{arguments}` next. The harness starts parsing here."),
        ("<tool_call|>", "End of tool call", "A tool call is complete; `<|tool_response>` usually follows right after."),
        ("<|tool_response>", "Start of tool response",
         "When the model writes it, it stops to wait for the result, like `stop_reason: tool_use`. The harness runs the tool and appends `response:name{value:…}` after it."),
        ("<tool_response|>", "End of tool response", "The tool result ends here. The model is still in the same model turn and goes on writing after reading it."),
        ('<|"|>', "String quote (escape)",
         "Wraps every string in tool declarations, calls and results in place of ordinary quotes, so quotes and brackets inside a string need no escaping."),
    ]),
    ("Multimodal", "where images, audio and video go; they appear when you attach one", [
        ("<|image|>", "Image",
         "The chat template writes it where an image goes; the processor expands it into `<|image>`, a run of `<|image|>` and `<image|>`, and the embeddings at those positions become the vision encoder's output."),
        ("<|image>", "Begin of image", "An image's tokens start here."),
        ("<image|>", "End of image", "An image's tokens end here."),
        ("<|audio|>", "Audio",
         "A placeholder like `<|image|>`: it expands into `<|audio>`, a run of `<|audio|>` and `<audio|>`, which become the audio encoder's output. One token per 40 ms, at most 750, so only the first 30 seconds are heard."),
        ("<|audio>", "Begin of audio", "A clip's audio tokens start here."),
        ("<audio|>", "End of audio", "A clip's audio tokens end here."),
        ("<|video|>", "Video",
         "32 frames are taken evenly from a video (all of them, if it has fewer), each after its time as `mm:ss`, then `<|image>`, a run of `<|video|>` and `<image|>`, which become the vision encoder's output. There is no sound."),
    ]),
    ("Reserved", "kept by the tokenizer; the chat format does not use them", [
        ("<pad>", "Padding", "Pads sequences of different lengths to one length so they fit in one batch; attention skips it. This app runs one sequence at a time and never needs it."),
        ("<unk>", "Unknown", "Stands in for characters the tokenizer cannot split. Gemma can break any character into byte tokens, so it hardly ever appears."),
        ("<mask>", "Mask", "A reserved mask token; the chat template never uses it."),
    ]),
]}

# What each file in a Gemma 4 checkpoint's folder holds, for the Model tab; render.MODEL_FILES says what each kind of
# file is in any checkpoint. A note points at fields by name and the tab shows their values, so it fits every Gemma 4.
FILES = {
    "config.json": FileNote({
        "zh": "一個檔案分三部分：`text_config` 是語言模型，`vision_config` 和 `audio_config` 是圖片和聲音的編碼器；"
              "`image_token_id`、`audio_token_id`、`video_token_id` 是佔位 token，編碼器的輸出會放進它們的位置。"
              "`text_config` 的 `layer_types` 混用滑動視窗層（只看最後 `sliding_window` 個 token）和 full attention 層；"
              "最後 `num_kv_shared_layers` 層不自己算 key / value，直接用前面的層算好的，MTP 草稿模型讀的也是這一份；"
              "`hidden_size_per_layer_input` 是 Per-Layer Embeddings 每個向量的長度。",
        "en": "Three parts in one file: `text_config` is the language model, `vision_config` and `audio_config` are the image and sound encoders, "
              "and `image_token_id`, `audio_token_id` and `video_token_id` are the placeholder tokens whose places the encoders' outputs take. "
              "In `text_config`, `layer_types` mixes sliding window layers, which see only the last `sliding_window` tokens, with full attention layers; "
              "the last `num_kv_shared_layers` layers compute no keys and values of their own but use those of earlier layers, which the MTP drafter reads too; "
              "and `hidden_size_per_layer_input` is the length of each vector of the Per-Layer Embeddings.",
    }, ("architectures", "text_config.sliding_window", "text_config.num_kv_shared_layers", "text_config.hidden_size_per_layer_input",
        "vision_soft_tokens_per_image")),
    "generation_config.json": FileNote({
        "zh": "`eos_token_id` 列了三個停止 token：`<eos>`、結束回答的 `<turn|>`，以及 `<|tool_response>`：模型寫完工具呼叫就自己停下，"
              "等 harness 把結果接上。`temperature`、`top_p`、`top_k` 是 Google 建議的抽樣設定，也是側欄的預設值。",
        "en": "`eos_token_id` lists three stop tokens: `<eos>`, `<turn|>`, which ends an answer, and `<|tool_response>`, so the model stops by itself "
              "after a tool call and waits for the harness to add the result. `temperature`, `top_p` and `top_k` are the sampling settings Google "
              "recommends, and the sidebar's defaults.",
    }, ("eos_token_id", "temperature", "top_p", "top_k")),
    "model.safetensors": FileNote({
        "zh": "權重，bfloat16：語言模型、視覺編碼器（`vision_tower`）和音訊編碼器（`audio_tower`）。"
              "很大一部分是 `embed_tokens_per_layer`，也就是 Per-Layer Embeddings：每一層替每個 token 各存一個小向量。"
              "這些向量只查表、不參與矩陣乘法，所以 Gemma 4 的名字算的是有效參數，E2B 的 E 就是 effective。",
        "en": "The weights, in bfloat16: the language model, the vision encoder (`vision_tower`) and the audio encoder (`audio_tower`). "
              "A large share of them is `embed_tokens_per_layer`, the Per-Layer Embeddings: every layer keeps a small vector for every token. "
              "Those vectors are only looked up, never multiplied, so Gemma 4's names count effective parameters: the E in E2B stands for effective.",
    }),
    "tokenizer.json": FileNote({
        "zh": "SentencePiece 風格的 BPE，262,144 個 token：空白先換成 `▁`，接在下一個詞的開頭（`▁world`）；"
              "詞彙表裡沒有的字元拆回 UTF-8 的 byte（`<0xE5>` 這種），不會變成 unknown token。"
              "`<|turn>`、`<|tool_call>`、`<|channel>` 這些控制 token 是 added tokens，各佔一個 id。",
        "en": "A BPE vocabulary of 262,144 tokens in SentencePiece's style: a space becomes `▁` at the start of the next word (`▁world`), "
              "and a character the vocabulary lacks falls back to its UTF-8 bytes (like `<0xE5>`) instead of an unknown token. "
              "Control tokens such as `<|turn>`, `<|tool_call>` and `<|channel>` are added tokens, one id each.",
    }),
    "tokenizer_config.json": FileNote({
        "zh": "替每個控制 token 取了角色名稱：`sot_token` 是 `<|turn>`、`eot_token` 是 `<turn|>`、`stc_token` 是 `<|tool_call>`、"
              "`soc_token` 是 `<|channel>`、`think_token` 是 `<|think|>`。`response_schema` 描述怎麼把回覆讀回思考、內容和工具呼叫；"
              "這個 app 不用它，而是用 family 自己的 `parse` 讀回覆（見運作說明）。",
        "en": "Gives each control token a role: `sot_token` is `<|turn>`, `eot_token` `<turn|>`, `stc_token` `<|tool_call>`, "
              "`soc_token` `<|channel>` and `think_token` `<|think|>`. `response_schema` describes how to read a reply back into thinking, "
              "content and tool calls; this app does not use it, and reads replies with the family's own `parse` (see How it works).",
    }, ("tokenizer_class", "backend")),
    "chat_template.jinja": FileNote({
        "zh": "把對話寫成 `<|turn>system`、`<|turn>user`、`<|turn>model` 三種回合，每個以 `<turn|>` 結尾。打開思考時在 system turn 放 `<|think|>`；"
              "工具寫成 `<|tool>declaration:…<tool|>` 放進 system turn；之前的工具呼叫和結果寫回 model turn；"
              "附件寫成 `<|image|>`、`<|audio|>`、`<|video|>`，再由 `encode()` 展開成媒體的 token。",
        "en": "Writes the conversation as `<|turn>system`, `<|turn>user` and `<|turn>model` turns, each ending in `<turn|>`. With thinking on, "
              "it puts `<|think|>` in the system turn; it writes the tools as `<|tool>declaration:…<tool|>` in the system turn, earlier tool calls "
              "and their results back into the model turn, and an attachment as `<|image|>`, `<|audio|>` or `<|video|>`, which `encode()` "
              "expands into the media's tokens.",
    }),
    "processor_config.json": FileNote({
        "zh": "三個編碼器的輸入設定都在這裡：`image_processor` 把圖片切成 `patch_size` 像素的 patch 再合併，一張圖最多 `max_soft_tokens` 個 token；"
              "`feature_extractor` 把 `sampling_rate` 的聲音轉成 mel 特徵，每 `audio_ms_per_token` 毫秒一個 token；"
              "`video_processor` 最多取 `num_frames` 格畫面，每格的 token 比一張圖少。",
        "en": "The inputs of all three encoders are set here: `image_processor` cuts an image into patches of `patch_size` pixels and pools them, "
              "at most `max_soft_tokens` tokens per image; `feature_extractor` turns sound at `sampling_rate` into mel features, one token per "
              "`audio_ms_per_token` ms; `video_processor` takes at most `num_frames` frames, each with fewer tokens than an image.",
    }, ("processor_class", "image_processor.patch_size", "image_processor.max_soft_tokens", "feature_extractor.sampling_rate",
        "audio_ms_per_token", "video_processor.num_frames", "video_processor.max_soft_tokens")),
}

# The MTP drafter's own files (DRAFTERS); llm.LocalLLM.load_drafter loads its config and weights, not its tokenizer.
SAME_VOCABULARY = FileNote({
    "zh": "和主模型同一套詞彙。這個 app 不載入它：草稿模型讀寫的都是主模型的 token id。",
    "en": "The same vocabulary as the model's. The app does not load it: the drafter reads and writes the model's token ids.",
})
DRAFTER_FILES = {
    "config.json": FileNote({
        "zh": "`Gemma4AssistantForCausalLM`：只有幾層，`hidden_size` 比主模型的（`backbone_hidden_size`）小得多，"
              "而且每一層都讀主模型的 key / value（`num_kv_shared_layers`），自己不存。輸出也不替每個 token 打分："
              "詞彙表事先分成 `num_centroids` 群，先替各群打分，只對最高的 `centroid_intermediate_top_k` 群裡的 token 算 logits。",
        "en": "`Gemma4AssistantForCausalLM`: only a few layers, with a `hidden_size` far smaller than the model's (`backbone_hidden_size`), "
              "every one of which reads the model's keys and values (`num_kv_shared_layers`) instead of keeping its own. Its output does not score "
              "every token either: the vocabulary is sorted into `num_centroids` clusters, the clusters are scored first, and logits are computed "
              "only for the tokens of the best `centroid_intermediate_top_k`.",
    }, ("text_config.num_hidden_layers", "text_config.hidden_size", "backbone_hidden_size", "text_config.num_kv_shared_layers",
        "num_centroids", "centroid_intermediate_top_k")),
    "generation_config.json": FileNote({
        "zh": "`num_assistant_tokens` 是草稿模型每次先猜幾個 token 再交給主模型驗證，`load_drafter()` 讀的就是它；`is_assistant` 標明這是草稿模型。"
              "抽樣設定和停止 token 跟主模型一樣，這個 app 用的是主模型的。",
        "en": "`num_assistant_tokens` is how many tokens the drafter guesses before the model checks them, which is what `load_drafter()` reads; "
              "`is_assistant` marks it as a drafter. The sampling settings and stop tokens are the model's, and the app uses the model's.",
    }, ("num_assistant_tokens", "num_assistant_tokens_schedule", "is_assistant")),
    "model.safetensors": FileNote({
        "zh": "大部分是 `embed_tokens`：整個詞彙表在草稿模型窄窄的寬度下的向量，當作輸出的表（`lm_head` 跟它共用），因為輸入用的是主模型自己的 embedding。"
              "`pre_projection` 把輸入（主模型的 token embedding 接上它的 hidden state）壓到草稿模型的寬度，`post_projection` 再把結果放大回主模型的寬度，"
              "給下一次猜測用。`masked_embedding` 是分群的輸出：`centroids` 替各群打分，`token_ordering` 依群列出 token。",
        "en": "Most of it is `embed_tokens`, the vocabulary at the drafter's narrow width, which serves as the output table (`lm_head` is tied to it), "
              "since the input is the model's own embedding. `pre_projection` narrows that input, the model's token embedding next to its hidden state, "
              "to the drafter's width, and `post_projection` widens the result back to the model's for the next guess. `masked_embedding` is the "
              "clustered output: `centroids` scores the clusters and `token_ordering` lists the tokens cluster by cluster.",
    }),
    "tokenizer.json": SAME_VOCABULARY,
    "tokenizer_config.json": SAME_VOCABULARY,
}

GEMMA4 = Family(
    key="gemma4",
    name="Gemma 4",
    model_types=("gemma4",),
    parse=parse_output,
    thought_prefix=THOUGHT_OPEN,
    channel_switch={"<|channel>": "thought", "<channel|>": "content", "<|tool_call>": "tool_call", "<tool_call|>": "content"},
    stop_tokens={
        "<turn|>": {"zh": "回合結束，相當於 end_turn", "en": "end of turn, like end_turn"},
        "<|tool_response>": {"zh": "等工具結果，相當於 tool_use", "en": "waiting for a tool result, like tool_use"},
        "<eos>": {"zh": "序列結束", "en": "end of sequence"},
    },
    reasoning_key="reasoning",
    # The template writes a message's text after its tool responses and closes the turn there,
    # which would leave the next prompt outside the model's turn; so text before a call is dropped.
    text_with_calls=False,
    user_turn="<|turn>user",
    special_groups=GROUPS,
    thinking_help={
        "zh": "在 system prompt 開頭加上 `<|think|>`，模型會先在 thought channel 寫推理再行動。",
        "en": "Puts `<|think|>` at the start of the system prompt, so the model writes its reasoning in the thought channel before it acts.",
    },
    tools_note={
        "zh": "chat template 把每個工具的 JSON Schema 寫成 `<|tool>declaration:名稱{…}<tool|>`，放進 system turn。",
        "en": "The chat template writes each tool's JSON Schema as `<|tool>declaration:name{…}<tool|>` in the system turn.",
    },
    model_note={
        "zh": "官方標示 2.3B effective：每層各自的 embedding 表（Per-Layer Embeddings）只做查表，不算進有效參數。",
        "en": "Officially 2.3B effective: each layer's own embedding table (Per-Layer Embeddings) is only looked up, so it does not count as effective parameters.",
    },
    file_notes=FILES,
    drafter_file_notes=DRAFTER_FILES,
)
