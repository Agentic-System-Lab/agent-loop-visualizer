"""Local inference with a hand-written decoding loop.

transformers' generate() hides the distribution at each step, so this module runs
the loop itself: one forward pass per token, then temperature, top-k and top-p,
then a sample. Every token comes back with its probability, its rank and the
strongest alternatives, which is what the visualization shows.

Nothing here is specific to one model: which tokens stop a generation and which
ones switch between thinking, tool calls and the reply come from the model's family.
"""

import logging
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
from transformers import AutoConfig, AutoModelForMultimodalLM, AutoTokenizer

import families

TOP_N = 5


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


def _not_a_kernel_notice(record: logging.LogRecord) -> bool:
    """transformers notes, once per function, that Qwen3.5's Gated DeltaNet layers run in plain PyTorch because
    flash-linear-attention and causal-conv1d are missing. Both need CUDA (Triton ships Linux wheels only), so
    on MPS or CPU there is nothing to install and the notice is dropped."""
    return "falling back to its reference PyTorch implementation" not in record.getMessage()


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
        if self.device != "cuda":
            logging.getLogger("transformers.integrations.hub_kernels").addFilter(_not_a_kernel_notice)
        if self.device == "mps":
            # Loading weights onto MPS from several threads can hang (seen with Qwen3.5); one thread is fast enough.
            os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        start = time.perf_counter()
        # config.json alone says which family the model belongs to; an unsupported one fails before the weights load.
        self.family = families.for_model_type(_load(AutoConfig, model_id).model_type)
        self.tokenizer = _load(AutoTokenizer, model_id)
        self.model = _load(AutoModelForMultimodalLM, model_id, dtype=self.dtype, device_map=self.device).eval()
        self.load_seconds = time.perf_counter() - start

        # The generation config's stop tokens, plus the family's: a checkpoint may list only some of them.
        # Qwen3.5 ships no generation_config.json, and its config.json names <|endoftext|> but not <|im_end|>,
        # which ends every chat turn.
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) | self.token_ids(self.family.stop_tokens)
        # Control tokens count as special even when the tokenizer does not mark them (Qwen's <think>).
        self.special_ids = set(self.tokenizer.all_special_ids) | self.eos_ids | self.token_ids(self.family.special_tokens)
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
            "layer_types": dict(Counter(getattr(cfg, "layer_types", None) or [])),
            "context_length": self.context_length,
            "vocab_size": cfg.vocab_size,
            "family": self.family.key,
            "stop_tokens": [self.piece(i) for i in sorted(self.eos_ids)],
            "special_tokens": {self.piece(i): i for i in sorted(self.special_ids)},
            "defaults": self._sampling_defaults(),
        }

    def _sampling_defaults(self) -> dict:
        """The checkpoint's generation_config.json, or the family's recommendation when it ships none."""
        config = self.model.generation_config
        if getattr(config, "_from_model_config", False) and self.family.sampling:
            return dict(self.family.sampling)
        return {k: getattr(config, k) for k in ("temperature", "top_p", "top_k")}

    def token_ids(self, pieces) -> set[int]:
        """Ids of the pieces that are single tokens of this tokenizer."""
        ids = ((p, self.tokenizer.convert_tokens_to_ids(p)) for p in pieces)
        return {i for p, i in ids if isinstance(i, int) and self.piece(i) == p}

    def warmup(self):
        """The first forward passes on MPS compile kernels and take seconds; pay that at load time."""
        prompt = self.render([{"role": "user", "content": "Hi"}], None, False)
        for _ in self.generate(prompt, Sampling(max_new_tokens=2), lambda: False):
            pass

    def piece(self, token_id: int) -> str:
        return self.tokenizer.convert_ids_to_tokens(token_id)

    def render(self, messages: list[dict], tools: list[dict] | None, thinking: bool, generation_prompt: bool = True) -> str:
        """The chat template turns messages and tool schemas into the one string the model reads.
        With `generation_prompt` it also opens the model's turn, so the model writes next."""
        return self.tokenizer.apply_chat_template(
            messages, tools=tools or None, tokenize=False, add_generation_prompt=generation_prompt, enable_thinking=thinking
        )

    def generate(self, prompt: str, sampling: Sampling, should_stop: Callable[[], bool]) -> Iterator[dict]:
        """Yield a prefill event, one event per token, then a done event."""
        ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        channel = "thought" if prompt.endswith(self.family.thought_prefix) else "content"
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
                if piece in self.family.channel_switch:
                    channel = self.family.channel_switch[piece]
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
