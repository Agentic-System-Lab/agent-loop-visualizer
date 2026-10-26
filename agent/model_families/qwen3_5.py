"""The Qwen3.5 family: reading its output, and what its control tokens mean.

Qwen3.5 writes a tool call in XML between two control tokens:

    <tool_call>
    <function=get_current_time>
    <parameter=timezone>
    Asia/Taipei
    </parameter>
    </function>
    </tool_call>

Every value is plain text, so its type comes from the tool's schema (the same thing vLLM's
qwen3_coder parser does). Thinking sits between <think> and </think>.
"""

import json
import re

from agent.model_families import Family, FileNote
from agent.i18n import pick

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
CALL_OPEN = "<tool_call>"
CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
FUNCTION_RE = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.S)
END_MARKERS = ("<|im_end|>", "<|endoftext|>")


def parse_output(raw: str, schemas: dict | None = None, lang: str = "zh") -> dict:
    """Split one model turn into thinking, tool calls and visible content. `schemas` maps a tool
    name to its parameters schema, which says what type each plain-text value is. Errors are worded in `lang`."""
    text = raw
    thinking = None
    if THINK_CLOSE in text:
        thought, _, text = text.partition(THINK_CLOSE)
        thinking = thought.split(THINK_OPEN)[-1].strip()
    elif THINK_OPEN in text:  # cut off while still thinking
        text, _, thought = text.partition(THINK_OPEN)
        thinking = thought.strip()

    calls = [parse_call(m.group(1), schemas or {}, lang) for m in CALL_RE.finditer(text)]
    rest = CALL_RE.sub("", text)
    if CALL_OPEN in rest:
        rest, _, tail = rest.partition(CALL_OPEN)
        call = parse_call(tail, schemas or {}, lang)
        call["arguments"], call["error"] = None, pick(lang, "工具呼叫沒有結束標記 </tool_call>", "The tool call has no closing </tool_call>")
        calls.append(call)

    for marker in END_MARKERS:
        rest = rest.replace(marker, "")
    return {"thinking": thinking, "tool_calls": calls, "content": rest.strip()}


def parse_call(body: str, schemas: dict, lang: str = "zh") -> dict:
    m = FUNCTION_RE.search(body)
    if not m:
        error = pick(lang, "工具呼叫裡沒有 <function=名稱>", "The tool call has no <function=name>")
        return {"name": "?", "raw_args": body, "arguments": None, "error": error}
    name = m.group(1).strip()
    props = schemas.get(name, {}).get("properties", {})
    args = {}
    for p in PARAM_RE.finditer(m.group(2)):
        key = p.group(1).strip()
        args[key] = convert(strip_newline(p.group(2)), props.get(key, {}).get("type"))
    return {"name": name, "raw_args": m.group(2), "arguments": args, "error": None}


def strip_newline(value: str) -> str:
    """The template puts the value on its own line; drop that one newline on each side."""
    value = value[1:] if value.startswith("\n") else value
    return value[:-1] if value.endswith("\n") else value


def convert(value: str, kind: str | None):
    """Turn a plain-text value into the schema's type. If it does not fit, keep the text and let
    the schema check report the wrong type to the model."""
    if kind in (None, "string"):
        return value
    if kind == "boolean":
        word = value.strip().lower()
        return word == "true" if word in ("true", "false") else value
    try:
        return json.loads(value)  # integer, number, object, array
    except ValueError:
        return value


# What each control token is for, grouped as the model tab lists them, in each language; `backticks` mark literal text.
GROUPS = {"zh": [
    ("訊息", "ChatML 格式：把對話切成一則一則訊息", [
        ("<|im_start|>", "訊息開始",
         "後面接角色名稱 `system`、`user` 或 `assistant` 和換行，開始一則訊息。脈絡最後的 `<|im_start|>assistant` 就是請模型接著寫。"),
        ("<|im_end|>", "訊息結束",
         "一則訊息到這裡結束，也是停止 token。回覆寫完或工具呼叫寫完，模型都停在這裡；"
         "不像 Gemma 分成兩種停止 token，harness 要看輸出裡有沒有 `<tool_call>` 才知道是 end_turn 還是 tool_use。"),
        ("<|endoftext|>", "文字結束（end of text）",
         "整段文字結束。它也是停止 token，不過對話中模型通常先停在 `<|im_end|>`；tokenizer 也拿它當填充用的 pad token。"),
    ]),
    ("思考", "推理寫在 <think> 和 </think> 之間", [
        ("<think>", "思考開始",
         "開著思考模式時，chat template 在 `<|im_start|>assistant` 後面先寫好 `<think>` 和換行，模型接著寫推理（step 卡片裡的斜體 token）。"
         "關掉時，模板寫一段空的 `<think>` `</think>`，模型直接回答。"),
        ("</think>", "思考結束", "推理到這裡結束，後面才是給使用者的回覆或工具呼叫。"),
    ]),
    ("工具", "工具的呼叫與結果", [
        ("<tool_call>", "工具呼叫開始",
         "模型要呼叫工具了，裡面用 XML 寫：`<function=名稱>`，每個參數一段 `<parameter=名稱>`。harness 從這裡開始解析。"),
        ("</tool_call>", "工具呼叫結束", "一個工具呼叫寫完，後面接 `<|im_end|>` 結束這則訊息，等 harness 執行工具。"),
        ("<tool_response>", "工具結果開始",
         "工具結果包在 `<tool_response>` 和 `</tool_response>` 之間，放在一則 `user` 訊息裡送回模型。"),
        ("</tool_response>", "工具結果結束", "工具結果到這裡結束。"),
    ]),
    ("視覺", "圖片和影片；附上這類檔案時才會出現", [
        ("<|vision_start|>", "視覺內容開始", "一張圖片或一段影片的 token 從這裡開始。"),
        ("<|vision_end|>", "視覺內容結束", "一張圖片或一段影片的 token 到這裡結束。"),
        ("<|image_pad|>", "圖片佔位",
         "chat template 在圖片的位置寫 `<|vision_start|><|image_pad|><|vision_end|>`；處理器把 `<|image_pad|>` 展開成一串，這些位置的 embedding 換成視覺編碼器的輸出。"),
        ("<|video_pad|>", "影片佔位", "和 `<|image_pad|>` 一樣，用在影片：每秒取 2 張畫格，每 2 張一組，前面寫上時間，例如 `<0.3 seconds>`。沒有聲音。"),
        ("<|vision_pad|>", "視覺填充", "保留的視覺填充 token，chat template 不會用到。"),
        ("<|object_ref_start|>", "物件名稱開始", "Qwen-VL 系列做視覺定位（grounding）時，標出指涉的物件名稱。chat template 不會用到。"),
        ("<|object_ref_end|>", "物件名稱結束", "和 `<|object_ref_start|>` 成對。"),
        ("<|box_start|>", "方框座標開始", "視覺定位時，包住物件方框的座標。chat template 不會用到。"),
        ("<|box_end|>", "方框座標結束", "和 `<|box_start|>` 成對。"),
        ("<|quad_start|>", "四邊形座標開始", "視覺定位時，包住四個角的座標，例如圖片裡傾斜的文字區塊。chat template 不會用到。"),
        ("<|quad_end|>", "四邊形座標結束", "和 `<|quad_start|>` 成對。"),
    ]),
    ("聲音", "這個模型沒有音訊編碼器；tokenizer 保留了這些 token，不會出現", [
        ("<|audio_start|>", "聲音開始", "一段聲音的 token 從這裡開始。"),
        ("<|audio_end|>", "聲音結束", "一段聲音的 token 到這裡結束。"),
        ("<|audio_pad|>", "聲音佔位", "聲音的位置，展開後換成音訊編碼器的輸出。"),
        ("<tts_pad>", "語音合成填充", "保留給語音合成（TTS）的 token。"),
        ("<tts_text_bos>", "語音合成文字開始", "保留給語音合成的 token。"),
        ("<tts_text_eod>", "語音合成文字結束", "保留給語音合成的 token。"),
        ("<tts_text_bos_single>", "語音合成單段文字開始", "保留給語音合成的 token。"),
    ]),
    ("程式碼", "程式碼補全的格式；對話用不到", [
        ("<|fim_prefix|>", "填空：前文（fill in the middle）",
         "程式碼補全的填空格式：`<|fim_prefix|>` 前文 `<|fim_suffix|>` 後文 `<|fim_middle|>`，模型接著寫中間缺的部分。"),
        ("<|fim_suffix|>", "填空：後文", "和 `<|fim_prefix|>` 搭配，標出缺口後面的程式碼。"),
        ("<|fim_middle|>", "填空：中間", "從這裡開始寫缺口裡的程式碼。"),
        ("<|fim_pad|>", "填空：填充", "填空格式用的填充 token。"),
        ("<|repo_name|>", "程式庫名稱", "把整個程式庫當脈絡時，標出 repo 的名稱。"),
        ("<|file_sep|>", "檔案分隔", "把多個檔案接在一起當脈絡時，分隔每個檔案。"),
    ]),
], "en": [
    ("Messages", "the ChatML format: splits the conversation into messages", [
        ("<|im_start|>", "Start of message",
         "Followed by a role, `system`, `user` or `assistant`, and a newline, it starts a message. The `<|im_start|>assistant` at the end of the context asks the model to write next."),
        ("<|im_end|>", "End of message",
         "A message ends here, and it is a stop token. The model stops here after a reply and after a tool call alike; "
         "unlike Gemma, with its two stop tokens, the harness has to look for a `<tool_call>` in the output to tell end_turn from tool_use."),
        ("<|endoftext|>", "End of text",
         "The end of the whole text. It is also a stop token, though in a conversation the model usually stops at `<|im_end|>` first; the tokenizer also uses it as its pad token."),
    ]),
    ("Thinking", "the reasoning goes between <think> and </think>", [
        ("<think>", "Start of thinking",
         "With thinking mode on, the chat template writes `<think>` and a newline after `<|im_start|>assistant`, and the model goes on with its reasoning (the italic tokens in a step card). "
         "With it off, the template writes an empty `<think>` `</think>` and the model answers right away."),
        ("</think>", "End of thinking", "The reasoning ends here; the reply to the user or a tool call comes after it."),
    ]),
    ("Tools", "calling tools and their results", [
        ("<tool_call>", "Start of tool call",
         "The model is calling a tool, in XML: `<function=name>`, then one `<parameter=name>` per argument. The harness starts parsing here."),
        ("</tool_call>", "End of tool call", "A tool call is complete; `<|im_end|>` follows to end the message, and the harness runs the tool."),
        ("<tool_response>", "Start of tool response",
         "A tool result goes between `<tool_response>` and `</tool_response>`, in a `user` message sent back to the model."),
        ("</tool_response>", "End of tool response", "The tool result ends here."),
    ]),
    ("Vision", "images and video; they appear when you attach one", [
        ("<|vision_start|>", "Start of vision", "An image's or a video's tokens start here."),
        ("<|vision_end|>", "End of vision", "An image's or a video's tokens end here."),
        ("<|image_pad|>", "Image placeholder",
         "The chat template writes `<|vision_start|><|image_pad|><|vision_end|>` where an image goes; the processor expands `<|image_pad|>` into a run of them, and the embeddings at those positions become the vision encoder's output."),
        ("<|video_pad|>", "Video placeholder", "Like `<|image_pad|>`, for video: 2 frames a second, in pairs, each pair after its time, such as `<0.3 seconds>`. There is no sound."),
        ("<|vision_pad|>", "Vision padding", "A reserved vision padding token; the chat template never uses it."),
        ("<|object_ref_start|>", "Start of object reference", "Marks the name of the object referred to when Qwen-VL models do visual grounding. The chat template never uses it."),
        ("<|object_ref_end|>", "End of object reference", "Pairs with `<|object_ref_start|>`."),
        ("<|box_start|>", "Start of box", "Wraps the coordinates of an object's bounding box in visual grounding. The chat template never uses it."),
        ("<|box_end|>", "End of box", "Pairs with `<|box_start|>`."),
        ("<|quad_start|>", "Start of quad", "Wraps the coordinates of four corners in visual grounding, such as a tilted block of text in an image. The chat template never uses it."),
        ("<|quad_end|>", "End of quad", "Pairs with `<|quad_start|>`."),
    ]),
    ("Audio", "this model has no audio encoder; the tokenizer keeps these tokens, but they never appear", [
        ("<|audio_start|>", "Start of audio", "A clip's audio tokens start here."),
        ("<|audio_end|>", "End of audio", "A clip's audio tokens end here."),
        ("<|audio_pad|>", "Audio placeholder", "Where audio goes; it is expanded and becomes the audio encoder's output."),
        ("<tts_pad>", "TTS padding", "Reserved for text-to-speech (TTS)."),
        ("<tts_text_bos>", "TTS text start", "Reserved for text-to-speech."),
        ("<tts_text_eod>", "TTS text end", "Reserved for text-to-speech."),
        ("<tts_text_bos_single>", "TTS single text start", "Reserved for text-to-speech."),
    ]),
    ("Code", "code completion formats; conversations do not use them", [
        ("<|fim_prefix|>", "Fill in the middle: prefix",
         "The fill-in-the-middle format for code completion: `<|fim_prefix|>` code before `<|fim_suffix|>` code after `<|fim_middle|>`, and the model writes the missing middle."),
        ("<|fim_suffix|>", "Fill in the middle: suffix", "Goes with `<|fim_prefix|>` and marks the code after the gap."),
        ("<|fim_middle|>", "Fill in the middle: middle", "The code for the gap is written from here."),
        ("<|fim_pad|>", "Fill in the middle: padding", "Padding for the fill-in-the-middle format."),
        ("<|repo_name|>", "Repository name", "Marks the repository's name when a whole repository is the context."),
        ("<|file_sep|>", "File separator", "Separates the files when several are joined into one context."),
    ]),
]}

# What each file in a Qwen3.5 checkpoint's folder holds, for the Model tab; render.MODEL_FILES says what each kind of
# file is in any checkpoint. A note points at fields by name and the tab shows their values, so it fits every Qwen3.5.
FILES = {
    "config.json": FileNote({
        "zh": "`text_config` 是語言模型，`vision_config` 是圖片和影片共用的編碼器；沒有聲音的編碼器，所以聲音檔對這個模型只是給 `run_python` 的普通檔案。"
              "`text_config` 的 `layer_types` 交錯 `linear_attention`（Gated DeltaNet）和 `full_attention`，每 `full_attention_interval` 層有一層 full attention。"
              "`mtp_num_hidden_layers` 描述權重裡附帶的一層 MTP，但 transformers 不載入它。`vocab_size` 比 tokenizer 的詞彙多一點，是 embedding 表補齊的空位。",
        "en": "`text_config` is the language model and `vision_config` the encoder that images and video share; there is no sound encoder, so to this "
              "model a sound file is just a file for `run_python`. In `text_config`, `layer_types` interleaves `linear_attention` (Gated DeltaNet) with "
              "`full_attention`, one full attention layer every `full_attention_interval`. `mtp_num_hidden_layers` describes an MTP layer that comes "
              "with the weights, which transformers does not load. `vocab_size` is a little more than the tokenizer's vocabulary: the embedding table is padded.",
    }, ("architectures", "text_config.full_attention_interval", "text_config.mtp_num_hidden_layers", "text_config.vocab_size", "vision_config.depth")),
    "generation_config.json": FileNote({
        "zh": "沒有這個檔案時，停止 token 是 `config.json` 寫的 `<|endoftext|>`，加上 family 補的 `<|im_end|>`（每個回合的結尾）；"
              "建議的 temperature、top-p、top-k 來自 family，抄自模型卡裡不思考時的設定。",
        "en": "Without it, the stop tokens are the `<|endoftext|>` that `config.json` names plus the `<|im_end|>` the family adds, which ends every turn; "
              "the recommended temperature, top-p and top-k are the family's, taken from the model card's settings for answers without thinking.",
    }),
    "model.safetensors": FileNote({
        "zh": "權重，大多是 bfloat16；就算只有一個檔案，檔名也照分片的格式取。除了語言模型和視覺編碼器（`visual`），還有 `config.json` 說的那層 MTP（`mtp.`），"
              "transformers 載入時跳過，所以用 transformers 時沒有 Qwen 的草稿模型，側欄的 MTP 只給 Gemma 用；有 GPU 的 vLLM (Docker) 則用這層推測解碼。"
              "Gated DeltaNet 層的 `A_log`（狀態衰減的速度）和它們的 norm 以 float32 存。",
        "en": "The weights, mostly in bfloat16, in files named in the sharded format even when there is only one. Besides the language model and the "
              "vision encoder (`visual`), they hold the MTP layer `config.json` mentions (`mtp.`), which transformers skips when loading, so with "
              "transformers there is no drafter for Qwen and MTP in the sidebar is for Gemma only; vLLM (Docker) on a GPU decodes speculatively with it. "
              "The Gated DeltaNet layers' `A_log` (how fast their state decays) and their norms are stored in float32.",
    }),
    "tokenizer.json": FileNote({
        "zh": "跟 GPT-2 一樣的 byte-level BPE：先用正規表示式切出詞、數字和標點，每個 byte 寫成一個可印出的字元"
              "（空白是 `Ġ`，「如果」兩個字的 6 個 byte 是 `å¦Ĥæŀľ`），再照 merges 合併；每個 byte 都有自己的字元，所以不會有 unknown token。"
              "token 浮窗顯示的是它代表的文字（family 的 `byte_level`）。",
        "en": "A byte-level BPE as in GPT-2: a regular expression first splits out words, numbers and punctuation, every byte is spelled as one "
              "printable character (a space is `Ġ`, and the six bytes of a two-character Chinese word are `å¦Ĥæŀľ`), and the merges join them; "
              "every byte has its own character, so nothing is ever unknown. A token's popover shows the text it stands for (the family's `byte_level`).",
    }),
    "tokenizer_config.json": FileNote({
        "zh": "`eos_token` 是結束每個回合的 `<|im_end|>`，`pad_token` 是 `<|endoftext|>`。added tokens 裡有 `<think>` 和 `<tool_call>`，"
              "但沒有標成 special，所以由 family 補上（見「特殊 token」）。裡面還有一份 chat template，和 `chat_template.jinja` 一模一樣。",
        "en": "`eos_token` is the `<|im_end|>` that ends every turn, and `pad_token` is `<|endoftext|>`. The added tokens include `<think>` and "
              "`<tool_call>` but do not mark them special, so the family does (see Special tokens). It also holds a copy of the chat template, "
              "the same as `chat_template.jinja`.",
    }, ("tokenizer_class", "eos_token", "pad_token", "model_max_length")),
    "chat_template.jinja": FileNote({
        "zh": "寫成 ChatML：`<|im_start|>system`、`user`、`assistant` 訊息，各以 `<|im_end|>` 結尾。工具的 schema 放在 system 訊息的 `<tools>…</tools>` 裡，"
              "後面附上怎麼寫呼叫；工具結果是一則新的 `user` 訊息，包在 `<tool_response>` 裡；思考模式在 assistant 訊息開頭先寫 `<think>`（關掉時寫一段空的）；"
              "圖片和影片寫成 `<|vision_start|><|image_pad|><|vision_end|>` 這樣，再由 `encode()` 展開。",
        "en": "Writes ChatML: `<|im_start|>system`, `user` and `assistant` messages, each ending in `<|im_end|>`. The tool schemas go in the system "
              "message inside `<tools>…</tools>`, followed by how to write a call; a tool result is a new `user` message wrapped in `<tool_response>`; "
              "thinking opens the assistant message with `<think>` (an empty one when it is off); an image or a video is written like "
              "`<|vision_start|><|image_pad|><|vision_end|>`, which `encode()` expands.",
    }),
    "preprocessor_config.json": FileNote({
        "zh": "圖片的設定：切成 `patch_size` 像素的 patch，每 `merge_size` × `merge_size` 個合成一個 token；"
              "`size` 的兩個值是一張圖最少和最多的像素數（名字雖然叫 edge）。",
        "en": "For images: cut into patches of `patch_size` pixels, every `merge_size` × `merge_size` of them merged into one token; "
              "the two values in `size` are the fewest and most pixels an image may have (though they are named edges).",
    }, ("processor_class", "patch_size", "merge_size", "size.shortest_edge", "size.longest_edge")),
    "video_preprocessor_config.json": FileNote({
        "zh": "影片畫面的設定，跟圖片一樣切 patch，再加上時間：每 `temporal_patch_size` 格畫面合成一組 token。family 另外限制每格畫面的像素（`processor_kwargs`）。",
        "en": "For video frames: patches as for images, and in time too, `temporal_patch_size` frames making one set of tokens. "
              "The family also caps each frame's pixels (`processor_kwargs`).",
    }, ("video_processor_type", "temporal_patch_size", "size.longest_edge")),
}

QWEN3_5 = Family(
    key="qwen3_5",
    name="Qwen3.5",
    model_types=("qwen3_5",),
    parse=parse_output,
    thought_prefix="<think>\n",
    channel_switch={"<think>": "thought", "</think>": "content", "<tool_call>": "tool_call", "</tool_call>": "content"},
    stop_tokens={
        "<|im_end|>": {"zh": "訊息結束；回覆和工具呼叫都停在這裡", "en": "end of message; replies and tool calls both stop here"},
        "<|endoftext|>": {"zh": "文字結束", "en": "end of text"},
    },
    reasoning_key="reasoning_content",
    text_with_calls=True,  # the template keeps a message's text in front of its <tool_call>
    user_turn="<|im_start|>user",
    special_groups=GROUPS,
    thinking_help={
        "zh": "chat template 在 assistant 訊息開頭先寫好 `<think>`，模型先寫推理；關掉時寫一段空的 `<think></think>`。"
        "0.8B 官方預設不開，開了容易一直想下去停不下來。",
        "en": "The chat template opens the assistant message with `<think>`, so the model reasons first; off, it writes an empty `<think></think>`. "
        "Off by default for the 0.8B model, which once it starts thinking often cannot stop.",
    },
    tools_note={
        "zh": "chat template 把每個工具的 JSON Schema 原樣放進 system turn 的 `<tools>…</tools>`，後面附上 XML 呼叫格式的說明。",
        "en": "The chat template puts each tool's JSON Schema as it is in `<tools>…</tools>` in the system turn, followed by how to write a call in XML.",
    },
    model_note={
        "zh": "混合架構：24 層裡每 4 層有 3 層是 Gated DeltaNet（線性 attention，狀態大小固定，不隨脈絡變長），1 層是 gated attention。"
        "在 transformers 裡沒裝 flash-linear-attention 時，它們用 PyTorch 參考實作，prefill 比較慢。",
        "en": "A hybrid: of every 4 of its 24 layers, 3 are Gated DeltaNet (linear attention, whose state has a fixed size and does not grow with the context) and 1 is gated attention. "
        "In transformers, without flash-linear-attention installed, they run the PyTorch reference implementation, so prefill is slower.",
    },
    # The checkpoint ships no generation_config.json; these are the model card's non-thinking text settings.
    sampling={"temperature": 1.0, "top_p": 1.0, "top_k": 20},
    byte_level=True,
    # Cap each video frame at the pixels the reference implementation (qwen-vl-utils) allows; transformers asks
    # for this to be said outright until it becomes the default.
    processor_kwargs={"cap_pixels_per_frame": True},
    file_notes=FILES,
)
