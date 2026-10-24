"""Local Gemma 4 inference with a hand-written decoding loop.

transformers' generate() hides the distribution at each step, so this module runs
the loop itself: one forward pass per token, then temperature, top-k and top-p,
then a sample. Every token comes back with its probability, its rank and the
strongest alternatives, which is what the visualization shows.
"""

import time
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer

TOP_N = 5
THOUGHT_PREFIX = "<|channel>thought\n"
# Special tokens that switch which part of the turn the model is writing.
CHANNEL_SWITCH = {"<|channel>": "thought", "<channel|>": "content", "<|tool_call>": "tool_call", "<tool_call|>": "content"}


@dataclass
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_new_tokens: int = 1024


def _load(cls, model_id: str, **kwargs):
    """Use the local Hugging Face cache first, so a downloaded model starts offline."""
    try:
        return cls.from_pretrained(model_id, local_files_only=True, **kwargs)
    except OSError:
        return cls.from_pretrained(model_id, **kwargs)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalLLM:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = pick_device()
        self.dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        start = time.perf_counter()
        self.tokenizer = _load(AutoTokenizer, model_id)
        self.model = _load(AutoModelForMultimodalLM, model_id, dtype=self.dtype, device_map=self.device).eval()
        self.load_seconds = time.perf_counter() - start

        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.special_ids = set(self.tokenizer.all_special_ids) | self.eos_ids
        text_cfg = getattr(self.model.config, "text_config", self.model.config)
        self.context_length = text_cfg.max_position_embeddings
        self.info = self._describe(text_cfg)

    def _describe(self, cfg) -> dict:
        parts = {}
        for name, child in self.model.model.named_children():
            parts[name] = sum(p.numel() for p in child.parameters())
        return {
            "model_id": self.model_id,
            "architecture": type(self.model).__name__,
            "device": self.device,
            "dtype": str(self.dtype).removeprefix("torch."),
            "load_seconds": round(self.load_seconds, 1),
            "parameters": self.model.num_parameters(),
            "parameter_parts": parts,
            "layers": cfg.num_hidden_layers,
            "hidden_size": cfg.hidden_size,
            "attention_heads": cfg.num_attention_heads,
            "kv_heads": cfg.num_key_value_heads,
            "sliding_window": getattr(cfg, "sliding_window", None),
            "context_length": self.context_length,
            "vocab_size": cfg.vocab_size,
            "stop_tokens": [self.piece(i) for i in sorted(self.eos_ids)],
            "defaults": {k: getattr(self.model.generation_config, k) for k in ("temperature", "top_p", "top_k")},
        }

    def warmup(self):
        """The first forward passes on MPS compile kernels and take seconds; pay that at load time."""
        prompt = self.render([{"role": "user", "content": "Hi"}], None, False)
        for _ in self.generate(prompt, Sampling(max_new_tokens=2), lambda: False):
            pass

    def piece(self, token_id: int) -> str:
        return self.tokenizer.convert_ids_to_tokens(token_id)

    def render(self, messages: list[dict], tools: list[dict] | None, thinking: bool) -> str:
        """The chat template turns messages and tool schemas into the one string the model reads."""
        return self.tokenizer.apply_chat_template(
            messages, tools=tools or None, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )

    def generate(self, prompt: str, sampling: Sampling, should_stop: Callable[[], bool]) -> Iterator[dict]:
        """Yield a prefill event, one event per token, then a done event."""
        ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        channel = "thought" if prompt.endswith(THOUGHT_PREFIX) else "content"
        generated: list[int] = []
        text = ""
        finish = "max_new_tokens"

        with torch.inference_mode():
            start = time.perf_counter()
            out = self.model(input_ids=ids, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            logits = out.logits[0, -1].float().cpu()
            prefill_ms = (time.perf_counter() - start) * 1000
            yield {"type": "prefill", "n_tokens": ids.shape[-1], "ms": round(prefill_ms, 1)}

            decode_start = time.perf_counter()
            for _ in range(sampling.max_new_tokens):
                if should_stop():
                    finish = "user_stopped"
                    break
                token_id, details = self._sample(logits, sampling)
                generated.append(token_id)

                piece = self.piece(token_id)
                special = token_id in self.special_ids
                if piece in CHANNEL_SWITCH:
                    channel = CHANNEL_SWITCH[piece]
                    token_channel = "control"
                else:
                    token_channel = "control" if special else channel

                # Decode the whole sequence and diff it, so characters split across byte tokens come out whole.
                full = self.tokenizer.decode(generated, skip_special_tokens=False)
                delta = "" if full.endswith("�") else full[len(text):]
                if delta:
                    text = full
                yield {
                    "type": "token",
                    "index": len(generated) - 1,
                    "id": token_id,
                    "piece": piece,
                    "text": delta,
                    "special": special,
                    "channel": token_channel,
                    **details,
                }

                if token_id in self.eos_ids:
                    finish = "stop_token"
                    break
                out = self.model(
                    input_ids=torch.tensor([[token_id]], device=self.device), past_key_values=past, use_cache=True, logits_to_keep=1
                )
                past = out.past_key_values
                logits = out.logits[0, -1].float().cpu()

        decode_s = time.perf_counter() - decode_start
        yield {
            "type": "done",
            "finish": finish,
            "stop_piece": self.piece(generated[-1]) if finish == "stop_token" else None,
            "n_tokens": len(generated),
            "decode_ms": round(decode_s * 1000, 1),
            "tok_per_s": round(len(generated) / decode_s, 1) if decode_s > 0 else None,
            "raw": self.tokenizer.decode(generated, skip_special_tokens=False),
        }

    def _sample(self, logits: torch.Tensor, s: Sampling) -> tuple[int, dict]:
        """Pick the next token and report the model's distribution and what the settings did to it."""
        probs = torch.softmax(logits, -1)  # the model's own distribution (T = 1, nothing cut off)
        top_p_vals, top_ids = probs.topk(TOP_N)

        if s.temperature <= 0:  # T -> 0 is greedy decoding
            chosen = int(top_ids[0])
            kept = 1
            p_sampled = 1.0
        else:
            scaled = logits / s.temperature
            if s.top_k > 0:
                cand_logits, cand_ids = scaled.topk(min(s.top_k, scaled.numel()))
            else:
                cand_logits, cand_ids = scaled.sort(descending=True)
            cand_probs = torch.softmax(cand_logits, -1)
            # Nucleus: keep the smallest prefix whose probability reaches top_p (always at least one token).
            kept = int(((cand_probs.cumsum(-1) - cand_probs) < s.top_p).sum())
            cand_ids, cand_probs = cand_ids[:kept], cand_probs[:kept] / cand_probs[:kept].sum()
            pick = int(torch.multinomial(cand_probs, 1))
            chosen = int(cand_ids[pick])
            p_sampled = float(cand_probs[pick])

        p = float(probs[chosen])
        # Candidates are a prefix of the tokens sorted by probability, so rank decides who survived.
        return chosen, {
            "p": p,
            "logprob": float(torch.log(probs[chosen])) if p > 0 else None,
            "rank": int((probs > probs[chosen]).sum()) + 1,
            "p_sampled": p_sampled,
            "kept": kept,
            "top": [
                {"id": int(i), "piece": self.piece(int(i)), "p": float(v), "kept": rank < kept}
                for rank, (v, i) in enumerate(zip(top_p_vals, top_ids))
            ],
        }
