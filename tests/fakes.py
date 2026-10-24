"""A stand-in for llm.LocalLLM that replays scripted model outputs.

It speaks the same interface as the real thing (render, tokenizer, generate, info),
renders prompts in a Gemma-like format and streams each scripted reply token by
token, so the harness and the UI run end to end without loading any weights.
"""

import re
import time
from types import SimpleNamespace

from llm import CHANNEL_SWITCH, THOUGHT_PREFIX

STR = '<|"|>'
STOP_PIECES = ("<turn|>", "<|tool_response>", "<eos>")
SPECIAL_PIECES = ("<|tool_call>", "<tool_call|>", "<|channel>", "<channel|>", STR, *STOP_PIECES)
PIECE_RE = re.compile("(" + "|".join(re.escape(p) for p in SPECIAL_PIECES) + ")")

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
    "stop_tokens": ["<turn|>", "<|tool_response>", "<eos>"],
    "defaults": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
}


def pieces(text: str) -> list[str]:
    """Special tokens stay whole; everything else becomes one token per character."""
    out = []
    for part in PIECE_RE.split(text):
        if part in SPECIAL_PIECES:
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


class FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        return SimpleNamespace(input_ids=list(range(len(pieces(text)))))


class FakeLLM:
    def __init__(self, outputs: list[str] | None = None, model_id: str = INFO["model_id"]):
        self.model_id = model_id
        self.outputs = outputs if outputs is not None else []  # shared on purpose: tests append to it
        self.tokenizer = FakeTokenizer()
        self.context_length = INFO["context_length"]
        self.info = {**INFO, "model_id": model_id}
        self.renders: list[dict] = []
        self.prompts: list[str] = []
        self.samplings: list = []
        self.gate = None  # a threading.Event the next generate() waits on, to hold a run open
        self._ids: dict[str, int] = {}

    def warmup(self):
        pass

    def render(self, messages: list[dict], tools: list[dict] | None, thinking: bool) -> str:
        self.renders.append({"messages": [dict(m) for m in messages], "tools": tools, "thinking": thinking})
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
        if messages[-1]["role"] == "user":
            out += "<|turn>model\n"
        if thinking and messages[-1]["role"] != "assistant":
            out += THOUGHT_PREFIX
        return out

    def _id(self, piece: str) -> int:
        return self._ids.setdefault(piece, len(self._ids) + 10)

    def generate(self, prompt, sampling, should_stop):
        self.prompts.append(prompt)
        self.samplings.append(sampling)
        if self.gate is not None:
            self.gate.wait(timeout=10)
        raw = self.outputs.pop(0) if self.outputs else "好的。<turn|>"
        if isinstance(raw, BaseException):
            raise raw
        yield {"type": "prefill", "n_tokens": len(self.tokenizer(prompt).input_ids), "ms": 12.5}

        channel = "thought" if prompt.endswith(THOUGHT_PREFIX) else "content"
        emitted, finish = [], "max_new_tokens"
        start = time.perf_counter()
        for piece in pieces(raw)[: sampling.max_new_tokens]:
            if should_stop():
                finish = "user_stopped"
                break
            special = piece in SPECIAL_PIECES
            if piece in CHANNEL_SWITCH:
                channel, token_channel = CHANNEL_SWITCH[piece], "control"
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
            if piece in STOP_PIECES:
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
    """A model turn that calls one tool and stops to wait for its result."""
    return gemma_call(name, args) + "<|tool_response>"


def record_run(outputs: list, user: str = "台北現在幾點？", history: list | None = None, sampling=None, **settings):
    """Run the harness to the end on the calling thread; return the run, its events and the fake model."""
    from agent import Run, RunSettings
    from llm import Sampling

    fake = FakeLLM(list(outputs))
    events: list[dict] = []
    run = Run(fake, history or [], user, RunSettings(sampling=sampling or Sampling(), **settings), events.append)
    run.run()
    return run, events, fake
