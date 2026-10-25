"""Model families: everything that changes when the model changes.

The harness (agent.py), the decoding loop (simple_backend.py) and the page (render.py) are written once.
What differs between Gemma 4 and Qwen3.5 is how the model writes its turns, its thinking and its
tool calls, which tokens stop it, and what its control tokens mean. A family module describes all
of that in one Family, and everything else asks the family.

To run another checkpoint of a family that is already here, add its Hugging Face id to MODELS, and the
revision to download it at to REVISIONS (its MTP drafter's too, if it has one in DRAFTERS).
To support a new family, write a module like qwen3_5.py (a parser for the model's output and
notes on its control tokens and on the files of its checkpoints) and add its Family to FAMILIES; the
`model_type` in the checkpoint's config.json decides which family a model belongs to.
"""

import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

# The models in the page's model menu: Hugging Face id -> label. A label is the same in both of the page's
# languages: Streamlit's selectbox sends its label back, and one that changed with the language would become the value.
MODELS = {
    "google/gemma-4-E2B-it": "Gemma 4 E2B (Google)",
    "Qwen/Qwen3.5-0.8B": "Qwen3.5 0.8B (Alibaba)",
}

# MTP drafters: model id -> a small model that guesses the next few tokens from the model's own last hidden
# state and keys and values, so one forward pass of the model can check several tokens (simple_backend.LocalLLM.load_drafter).
# transformers runs this for Gemma 4's assistant checkpoints only.
DRAFTERS = {
    "google/gemma-4-E2B-it": "google/gemma-4-E2B-it-assistant",
}

# The Hub revision (a commit) each model and drafter downloads at, so that whenever someone downloads one they get
# the files the page was written and tested against, not whatever `main` has become. An id missing here downloads
# at `main`. Moving one to a newer revision means rerunning test_chat_template.py and test_real_model.py, and
# rereading the guide's section on the model folder, whose numbers come from these files.
REVISIONS = {
    "google/gemma-4-E2B-it": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    "google/gemma-4-E2B-it-assistant": "2d874ef7d29f9a30599a1e4b3c1cbc9595f005df",
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
}

# GPT-2's byte-level BPE, which Qwen's vocabulary uses, spells every byte as one printable character: printable
# Latin-1 bytes stand for themselves and the rest move up to U+0100 and beyond, so a space is Ġ, a newline Ċ and
# 如果 (six bytes of UTF-8) is å¦Ĥæŀľ. This maps each of those characters back to its byte.
_PRINTABLE = [*range(0x21, 0x7F), *range(0xA1, 0xAD), *range(0xAE, 0x100)]
BYTE_OF = {chr(b): b for b in _PRINTABLE} | {chr(0x100 + n): b for n, b in enumerate(b for b in range(256) if b not in _PRINTABLE)}


@dataclass(frozen=True)
class FileNote:
    """What one file in a checkpoint's folder holds in this family. The Model tab shows the note next to the file,
    with the values in this checkpoint of `fields`, dotted paths into a JSON file (text_config.sliding_window)."""

    text: dict[str, str]  # language -> the note
    fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class Family:
    key: str
    name: str
    model_types: tuple[str, ...]  # config.json `model_type` values of this family

    # Reading the model's output.
    parse: Callable[..., dict]  # (raw output, {tool name: parameters schema}, language of its errors) -> thinking, tool_calls, content
    thought_prefix: str  # a prompt that ends with this has the model start inside its thinking
    channel_switch: dict[str, str]  # tokens that switch what the model is writing: thought, tool_call, content
    stop_tokens: dict[str, dict[str, str]]  # tokens that end a generation -> language -> what stopping there means

    # Writing the conversation back for the chat template.
    reasoning_key: str  # the assistant-message field the chat template reads earlier thinking from
    text_with_calls: bool  # whether text written before a tool call survives re-rendering the history

    # Showing it on the page; the text is in both of the page's languages (i18n.py).
    user_turn: str  # where a user message starts in a prompt; the step card shows the newest one
    special_groups: dict[str, list]  # language -> (title, subtitle, ((token, name, what it is for), ...)) for each group of control tokens
    thinking_help: dict[str, str]  # what the thinking toggle does to the prompt
    tools_note: dict[str, str]  # how the tool schemas appear in the prompt
    model_note: dict[str, str] = field(default_factory=dict)
    sampling: dict | None = None  # recommended temperature / top_p / top_k when a checkpoint ships none
    byte_level: bool = False  # the vocabulary spells bytes as characters, GPT-2 style (BYTE_OF); otherwise it is SentencePiece's
    processor_kwargs: dict = field(default_factory=dict)  # passed to the processor along with images, audio and video
    file_notes: dict[str, FileNote] = field(default_factory=dict)  # a file in the checkpoint's folder -> what it holds in this family
    drafter_file_notes: dict[str, FileNote] = field(default_factory=dict)  # the same for the folder of its MTP drafter (DRAFTERS)

    def show_piece(self, piece: str) -> str:
        """A vocabulary entry as the text it stands for, which is what a token's popover shows: ␣ for a space,
        ↵ for a newline, and a byte that is only part of a character as <0xE5>, the way SentencePiece writes one."""
        if self.byte_level and all(c in BYTE_OF for c in piece):
            # surrogateescape keeps each byte that is not whole UTF-8 as U+DC80–U+DCFF, so it can be written in hex.
            text = bytes(BYTE_OF[c] for c in piece).decode("utf-8", errors="surrogateescape")
            text = "".join(f"<0x{ord(c) - 0xDC00:02X}>" if "\udc80" <= c <= "\udcff" else c for c in text)
        else:
            text = piece.replace("▁", " ")  # SentencePiece marks a leading space with ▁
        return text.replace(" ", "␣").replace("\n", "↵")

    @cached_property
    def special_tokens(self) -> dict[str, dict[str, tuple[str, str]]]:
        """Control token -> language -> (name, what it is for)."""
        out: dict[str, dict[str, tuple[str, str]]] = {}
        for lang, groups in self.special_groups.items():
            for _, _, rows in groups:
                for piece, name, desc in rows:
                    out.setdefault(piece, {})[lang] = (name, desc)
        return out

    @cached_property
    def special_re(self) -> re.Pattern:
        """Matches the control tokens in a prompt, longest first so <|image|> is not read as <|image>."""
        pieces = sorted(self.special_tokens, key=len, reverse=True)
        return re.compile("|".join(re.escape(p) for p in pieces))


from families.gemma4 import GEMMA4  # noqa: E402  (the family modules import Family from here)
from families.qwen3_5 import QWEN3_5  # noqa: E402

FAMILIES = {f.key: f for f in (GEMMA4, QWEN3_5)}


def get(key: str) -> Family:
    return FAMILIES[key]


def for_model_type(model_type: str) -> Family:
    for family in FAMILIES.values():
        if model_type in family.model_types:
            return family
    known = ", ".join(t for f in FAMILIES.values() for t in f.model_types)
    raise ValueError(
        f"Models of model_type {model_type!r} are not supported; supported: {known}. families/__init__.py says how to add a family. "
        f"（不支援 model_type {model_type!r} 的模型；目前支援：{known}。新增模型家族的方法見 families/__init__.py。）"
    )
