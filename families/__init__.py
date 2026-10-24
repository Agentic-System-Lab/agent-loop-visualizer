"""Model families: everything that changes when the model changes.

The harness (agent.py), the decoding loop (llm.py) and the page (render.py) are written once.
What differs between Gemma 4 and Qwen3.5 is how the model writes its turns, its thinking and its
tool calls, which tokens stop it, and what its control tokens mean. A family module describes all
of that in one Family, and everything else asks the family.

To run another checkpoint of a family that is already here, add its Hugging Face id to MODELS.
To support a new family, write a module like qwen3_5.py (a parser for the model's output and
notes on its control tokens) and add its Family to FAMILIES; the `model_type` in the checkpoint's
config.json decides which family a model belongs to.
"""

import re
from dataclasses import dataclass
from functools import cached_property
from typing import Callable

# The models in the page's model menu: Hugging Face id -> label.
MODELS = {
    "google/gemma-4-E2B-it": "Gemma 4 E2B（Google）",
    "Qwen/Qwen3.5-0.8B": "Qwen3.5 0.8B（Alibaba）",
}


@dataclass(frozen=True)
class Family:
    key: str
    name: str
    model_types: tuple[str, ...]  # config.json `model_type` values of this family

    # Reading the model's output.
    parse: Callable[..., dict]  # (raw output, {tool name: parameters schema}) -> thinking, tool_calls, content
    thought_prefix: str  # a prompt that ends with this has the model start inside its thinking
    channel_switch: dict[str, str]  # tokens that switch what the model is writing: thought, tool_call, content
    stop_tokens: dict[str, str]  # tokens that end a generation -> what stopping there means

    # Writing the conversation back for the chat template.
    reasoning_key: str  # the assistant-message field the chat template reads earlier thinking from
    text_with_calls: bool  # whether text written before a tool call survives re-rendering the history

    # Showing it on the page.
    user_turn: str  # where a user message starts in a prompt; the step card shows the newest one
    special_groups: list  # (title, subtitle, ((token, name, what it is for), ...)) for each group of control tokens
    thinking_help: str  # what the thinking toggle does to the prompt
    tools_note: str  # how the tool schemas appear in the prompt
    model_note: str = ""
    sampling: dict | None = None  # recommended temperature / top_p / top_k when a checkpoint ships none

    @cached_property
    def special_tokens(self) -> dict[str, tuple[str, str]]:
        """Control token -> (name, what it is for)."""
        return {piece: (name, desc) for _, _, rows in self.special_groups for piece, name, desc in rows}

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
    raise ValueError(f"不支援 model_type {model_type!r} 的模型；目前支援：{known}。新增模型家族的方法見 families/__init__.py。")
