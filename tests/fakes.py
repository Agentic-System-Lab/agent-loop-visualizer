"""A stand-in for llm.LocalLLM that replays scripted model outputs.

It speaks the same interface as the real thing (render, tokenizer, generate, family,
info), renders prompts in a simplified form of its family's chat format (Gemma 4 or
Qwen3.5) and streams each scripted reply token by token, so the harness and the UI run
end to end without loading any weights.
"""

import json
import re
import time
from types import SimpleNamespace

import families
from families import GEMMA4, QWEN3_5

STR = '<|"|>'

INFO = {
    "model_id": "fake/gemma-test",
    "architecture": "FakeForTests",
    "device": "cpu",
    "dtype": "float32",
    "load_seconds": 0.0,
    "parameters": 5_100_000_000,
    "parameter_parts": {"language_model": 4_600_000_000, "audio_tower": 300_000_000, "vision_tower": 200_000_000},
    "layers": 35,
    "hidden_size": 1536,
    "attention_heads": 8,
    "kv_heads": 1,
    "sliding_window": 512,
    "context_length": 131072,
    "vocab_size": 262144,
    "layer_types": {"sliding_attention": 28, "full_attention": 7},
    "family": "gemma4",
    "stop_tokens": ["<turn|>", "<|tool_response>", "<eos>"],
    "special_tokens": {
        "<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4, "<|tool>": 46, "<tool|>": 47, "<|tool_call>": 48,
        "<tool_call|>": 49, "<|tool_response>": 50, "<tool_response|>": 51, '<|"|>': 52, "<|think|>": 98, "<|channel>": 100,
        "<channel|>": 101, "<|turn>": 105, "<turn|>": 106, "<|image>": 255999, "<|audio>": 256000, "<|image|>": 258880,
        "<|audio|>": 258881, "<image|>": 258882, "<audio|>": 258883, "<|video|>": 258884,
    },
    "defaults": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
}
QWEN_INFO = {
    **INFO,
    "model_id": "fake/qwen-test",
    "parameters": 853_000_000,
    "parameter_parts": {"language_model": 752_000_000, "visual": 101_000_000},
    "layers": 24,
    "hidden_size": 1024,
    "attention_heads": 8,
    "kv_heads": 2,
    "sliding_window": None,
    "layer_types": {"linear_attention": 18, "full_attention": 6},
    "context_length": 262144,
    "vocab_size": 248320,
    "family": "qwen3_5",
    "stop_tokens": ["<|endoftext|>", "<|im_end|>"],
    "special_tokens": {piece: 248044 + i for i, piece in enumerate(QWEN3_5.special_tokens)},
    "defaults": {"temperature": 1.0, "top_p": 1.0, "top_k": 20},
}


def family_of(model_id: str) -> families.Family:
    return QWEN3_5 if "qwen" in model_id.lower() else GEMMA4


def pieces(text: str, family: families.Family = GEMMA4) -> list[str]:
    """Control tokens stay whole; everything else becomes one token per character."""
    out = []
    for part in re.split(f"({family.special_re.pattern})", text):
        if part in family.special_tokens:
            out.append(part)
        else:
            out.extend(part)
    return out


def gemma_value(value) -> str:
    if isinstance(value, str):
        return f"{STR}{value}{STR}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def gemma_call(name: str, args: dict) -> str:
    body = ",".join(f"{k}:{gemma_value(v)}" for k, v in args.items())
    return f"<|tool_call>call:{name}{{{body}}}<tool_call|>"


def qwen_call(name: str, args: dict) -> str:
    def value(v):
        return v if isinstance(v, str) else json.dumps(v) if isinstance(v, (dict, list)) else str(v)

    params = "".join(f"<parameter={k}>\n{value(v)}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


class FakeTokenizer:
    def __init__(self, family: families.Family = GEMMA4):
        self.family = family

    def __call__(self, text: str, add_special_tokens: bool = False):
        return SimpleNamespace(input_ids=list(range(len(pieces(text, self.family)))))


class FakeLLM:
    def __init__(self, outputs: list[str] | None = None, model_id: str | None = None, family: families.Family | None = None):
        self.family = family or family_of(model_id or "")
        info = QWEN_INFO if self.family is QWEN3_5 else INFO
        self.model_id = model_id or info["model_id"]
        self.outputs = outputs if outputs is not None else []  # shared on purpose: tests append to it
        self.tokenizer = FakeTokenizer(self.family)
        self.context_length = info["context_length"]
        self.info = {**info, "model_id": self.model_id}
        self.renders: list[dict] = []
        self.prompts: list[str] = []
        self.samplings: list = []
        self.gate = None  # a threading.Event the next generate() waits on, to hold a run open
        self._ids: dict[str, int] = {}

    def warmup(self):
        pass

    def render(self, messages: list[dict], tools: list[dict] | None, thinking: bool, generation_prompt: bool = True) -> str:
        self.renders.append({"messages": [dict(m) for m in messages], "tools": tools, "thinking": thinking})
        if self.family is QWEN3_5:
            return self._render_qwen(messages, tools, thinking, generation_prompt)
        out = "<bos>"
        for m in messages:
            role = m["role"]
            if role == "system":
                decls = "".join(f"<|tool>declaration:{t['function']['name']}<tool|>" for t in tools or [])
                out += "<|turn>system\n" + ("<|think|>\n" if thinking else "") + m["content"] + decls + "<turn|>\n"
            elif role == "user":
                out += f"<|turn>user\n{m['content']}<turn|>\n"
            elif role == "assistant":
                calls = "".join(gemma_call(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls", []))
                out += f"<|turn>model\n{m['content']}{calls}"
                out += "<|tool_response>" if calls else "<turn|>\n"
            elif role == "tool":
                out += f"response:{m['name']}{{value:{STR}{m['content']}{STR}}}<tool_response|>"
        if generation_prompt and messages[-1]["role"] == "user":
            out += "<|turn>model\n"
        if generation_prompt and thinking and messages[-1]["role"] != "assistant":
            out += GEMMA4.thought_prefix
        return out

    def _render_qwen(self, messages, tools, thinking, generation_prompt) -> str:
        out = ""
        for i, m in enumerate(messages):
            role = m["role"]
            if role == "system":
                decls = "".join(f"\n{json.dumps(t)}" for t in tools or [])
                head = f"# Tools\n\n<tools>{decls}\n</tools>\n\n" if tools else ""
                out += f"<|im_start|>system\n{head}{m['content']}<|im_end|>\n"
            elif role == "user":
                out += f"<|im_start|>user\n{m['content']}<|im_end|>\n"
            elif role == "assistant":
                calls = "".join(qwen_call(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls", []))
                think = m.get("reasoning_content", "")
                out += f"<|im_start|>assistant\n<think>\n{think}\n</think>\n\n{m['content']}{calls}<|im_end|>\n"
            elif role == "tool":
                first = messages[i - 1]["role"] != "tool"
                last = i == len(messages) - 1 or messages[i + 1]["role"] != "tool"
                out += ("<|im_start|>user" if first else "") + f"\n<tool_response>\n{m['content']}\n</tool_response>"
                out += "<|im_end|>\n" if last else ""
        if generation_prompt:
            out += "<|im_start|>assistant\n" + (QWEN3_5.thought_prefix if thinking else "<think>\n\n</think>\n\n")
        return out

    def _id(self, piece: str) -> int:
        return self._ids.setdefault(piece, len(self._ids) + 10)

    def generate(self, prompt, sampling, should_stop):
        self.prompts.append(prompt)
        self.samplings.append(sampling)
        if self.gate is not None:
            self.gate.wait(timeout=10)
        raw = self.outputs.pop(0) if self.outputs else "好的。" + next(iter(self.family.stop_tokens))
        if isinstance(raw, BaseException):
            raise raw
        yield {"type": "prefill", "n_tokens": len(self.tokenizer(prompt).input_ids), "ms": 12.5}

        family = self.family
        channel = "thought" if prompt.endswith(family.thought_prefix) else "content"
        emitted, finish = [], "max_new_tokens"
        start = time.perf_counter()
        for piece in pieces(raw, family)[: sampling.max_new_tokens]:
            if should_stop():
                finish = "user_stopped"
                break
            special = piece in family.special_tokens
            if piece in family.channel_switch:
                channel, token_channel = family.channel_switch[piece], "control"
            else:
                token_channel = "control" if special else channel
            emitted.append(piece)
            tid = self._id(piece)
            yield {
                "type": "token",
                "index": len(emitted) - 1,
                "id": tid,
                "piece": piece,
                "text": piece,
                "special": special,
                "channel": token_channel,
                "p": 0.9,
                "logprob": -0.105,
                "rank": 1,
                "p_sampled": 0.95,
                "kept": 2,
                "top": [
                    {"id": tid, "piece": piece, "p": 0.9, "kept": True},
                    {"id": 1, "piece": "▁alt", "p": 0.05, "kept": True},
                    {"id": 2, "piece": "▁other", "p": 0.01, "kept": False},
                ],
            }
            if piece in family.stop_tokens:
                finish = "stop_token"
                break
        decode_s = max(time.perf_counter() - start, 1e-6)
        yield {
            "type": "done",
            "finish": finish,
            "stop_piece": emitted[-1] if finish == "stop_token" else None,
            "n_tokens": len(emitted),
            "decode_ms": round(decode_s * 1000, 1),
            "tok_per_s": round(len(emitted) / decode_s, 1),
            "raw": "".join(emitted),
        }


def tool_call(name: str, **args) -> str:
    """A Gemma turn that calls one tool and stops to wait for its result."""
    return gemma_call(name, args) + "<|tool_response>"


def qwen_tool_call(name: str, **args) -> str:
    """A Qwen3.5 turn that calls one tool; it ends its message like any other."""
    return qwen_call(name, args) + "<|im_end|>"


def record_run(outputs: list, user: str = "台北現在幾點？", history: list | None = None, sampling=None, family=None, **settings):
    """Run the harness to the end on the calling thread; return the run, its events and the fake model."""
    from agent import Run, RunSettings
    from llm import Sampling

    fake = FakeLLM(list(outputs), family=family)
    events: list[dict] = []
    run = Run(fake, history or [], user, RunSettings(sampling=sampling or Sampling(), **settings), events.append)
    run.run()
    return run, events, fake
