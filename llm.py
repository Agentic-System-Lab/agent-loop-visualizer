"""Local inference with a hand-written decoding loop.

transformers' generate() hides the distribution at each step, so this module runs
the loop itself: one forward pass per token, then temperature, top-k and top-p,
then a sample. Every token comes back with its probability, its rank and the
strongest alternatives, which is what the visualization shows.

Nothing here is specific to one model: which tokens stop a generation and which
ones switch between thinking, tool calls and the reply come from the model's family.

With MTP switched on, a small drafter guesses the next few tokens and one forward pass
checks them all (_drafted); every token still comes from the model's distribution.
"""

import logging
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer

import families

TOP_N = 5


@dataclass
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_new_tokens: int = 1024
    mtp: bool = False  # let the MTP drafter guess ahead, once it is loaded (LocalLLM.load_drafter)


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
        self.drafter_id = families.DRAFTERS.get(model_id)
        self.drafter = None  # loaded the first time MTP is switched on
        self._drafter_lock = threading.Lock()
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
            "drafter_id": self.drafter_id,
            "drafter": None,
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

    def warmup(self, mtp: bool = False):
        """The first forward passes on MPS compile kernels and take seconds; pay that at load time."""
        prompt = self.render([{"role": "user", "content": "Hi"}], None, False)
        for _ in self.generate(prompt, Sampling(max_new_tokens=16 if mtp else 2, mtp=mtp), lambda: False):
            pass

    def load_drafter(self):
        """Load the MTP drafter that goes with this model (families.DRAFTERS), if any, and warm it up."""
        with self._drafter_lock:
            if self.drafter is not None or self.drafter_id is None:
                return
            start = time.perf_counter()
            self.drafter = _load(AutoModelForCausalLM, self.drafter_id, dtype=self.dtype, device_map=self.device).eval()
            self.draft_tokens = self.drafter.generation_config.num_assistant_tokens  # guesses per forward pass
            self.warmup(mtp=True)
            cfg = self.drafter.config.get_text_config()
            self.info["drafter"] = {
                "model_id": self.drafter_id,
                "parameters": self.drafter.num_parameters(),
                "layers": cfg.num_hidden_layers,
                "hidden_size": cfg.hidden_size,
                "draft_tokens": self.draft_tokens,
                "load_seconds": round(time.perf_counter() - start, 1),
            }

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
        drafting = sampling.mtp and self.drafter is not None
        mtp = {"drafted": 0, "accepted": 0, "forwards": 0} if drafting else None
        generated: list[int] = []
        text = ""
        finish = "max_new_tokens"

        with torch.inference_mode():
            start = time.perf_counter()
            out = self._forward(ids, None, 1, drafting)
            prefill_ms = (time.perf_counter() - start) * 1000
            yield {"type": "prefill", "n_tokens": ids.shape[-1], "ms": round(prefill_ms, 1)}

            decode_start = time.perf_counter()
            tokens = self._drafted(out, sampling, mtp) if drafting else self._one_at_a_time(out, sampling)
            for token_id, details in tokens:
                if should_stop():
                    finish = "user_stopped"
                    break
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
                if len(generated) == sampling.max_new_tokens:
                    break

        decode_s = time.perf_counter() - decode_start
        yield {
            "type": "done",
            "finish": finish,
            "stop_piece": self.piece(generated[-1]) if finish == "stop_token" else None,
            "n_tokens": len(generated),
            "decode_ms": round(decode_s * 1000, 1),
            "tok_per_s": round(len(generated) / decode_s, 1) if decode_s > 0 else None,
            "mtp": mtp,
            "raw": self.tokenizer.decode(generated, skip_special_tokens=False),
        }

    def _forward(self, ids: torch.Tensor, past, keep: int, drafting: bool = False):
        """One forward pass over `ids`, with the cache of everything before them; logits for the last `keep` positions.
        The drafter also reads the model's last hidden state and the keys and values its later layers share."""
        extra = {"output_hidden_states": True, "return_shared_kv_states": True} if drafting else {}
        return self.model(input_ids=ids, past_key_values=past, use_cache=True, logits_to_keep=keep, **extra)

    def _one_at_a_time(self, out, s: Sampling) -> Iterator[tuple[int, dict]]:
        """Plain decoding: sample a token, feed it back with the cache for the next position's logits, repeat."""
        while True:
            token_id, details = self._sample(out.logits[0, -1].float().cpu(), s)
            yield token_id, details
            out = self._forward(torch.tensor([[token_id]], device=self.device), out.past_key_values, 1)

    def _drafted(self, out, s: Sampling, stats: dict) -> Iterator[tuple[int, dict]]:
        """MTP decoding. After each token the drafter guesses the next few, and one forward pass over the token and the
        guesses gives the model's distribution at every guessed position. _sample checks the guesses in order: each
        one it keeps is a token for free, the first one it turns down is replaced by the model's own pick, and the
        rest are dropped. When every guess holds, the last position gives one more token."""
        past = out.past_key_values
        past.activate_past_recording()  # so the sliding-window layers can still drop guesses that were turned down
        hidden, shared = out.hidden_states[-1][:, -1:], out.shared_kv_states
        token_id, details = self._sample(out.logits[0, -1].float().cpu(), s)
        while True:
            yield token_id, details
            guesses = self._draft(token_id, hidden, shared, past.get_seq_length())
            out = self._forward(torch.tensor([[token_id, *guesses]], device=self.device), past, len(guesses) + 1, True)
            logits = out.logits[0].float().cpu()
            stats["forwards"] += 1
            stats["drafted"] += len(guesses)
            for n, guess in enumerate(guesses):
                token_id, details = self._sample(logits[n], s, guess)
                if token_id != guess:
                    details["draft"] = {"piece": self.piece(guess), "accepted": False}
                    break
                stats["accepted"] += 1
                yield token_id, {**details, "draft": {"piece": self.piece(guess), "accepted": True}}
            else:
                n = len(guesses)
                token_id, details = self._sample(logits[n], s)
            # Keep the cache and the drafter's inputs up to the last kept guess (or the token, when none held).
            dropped = len(guesses) - n
            past.crop(-dropped)
            hidden = out.hidden_states[-1][:, n : n + 1]
            shared = {k: (key[:, :, : key.shape[2] - dropped], value[:, :, : value.shape[2] - dropped])
                      for k, (key, value) in out.shared_kv_states.items()}

    def _draft(self, token_id: int, hidden: torch.Tensor, shared: dict, position: int) -> list[int]:
        """The drafter's greedy guesses for the tokens after `token_id`. It has no cache of its own: it attends to the
        model's keys and values, so every guess sits at the position of `token_id`, and what it has guessed so far
        comes in through its input, the embedding of the last guess next to the drafter's last hidden state."""
        embed = self.model.get_input_embeddings()
        positions = torch.tensor([[position]], device=self.device)
        guesses = []
        while len(guesses) < self.draft_tokens and token_id not in self.eos_ids:
            x = torch.cat([embed(torch.tensor([[token_id]], device=self.device)), hidden], -1)
            out = self.drafter(inputs_embeds=x, position_ids=positions, shared_kv_states=shared)
            token_id, hidden = int(out.logits[0, -1].argmax()), out.last_hidden_state
            guesses.append(token_id)
        return guesses

    def _sample(self, logits: torch.Tensor, s: Sampling, draft: int | None = None) -> tuple[int, dict]:
        """Pick the next token and report the model's distribution and what the settings did to it.
        A `draft` (the MTP drafter's guess) is kept with the probability the settings give it, which in greedy
        decoding means when it is the top token; otherwise the pick comes from the other candidates. Either way
        each token comes out exactly as often as without a drafter."""
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
            weights = cand_probs
            at = (cand_ids == draft).nonzero().flatten() if draft is not None else []
            if len(at) and float(torch.rand(())) < float(cand_probs[at[0]]):
                pick = int(at[0])
            else:
                if len(at):  # the guess was turned down: draw from the other candidates
                    weights = cand_probs.clone()
                    weights[at[0]] = 0
                pick = int(torch.multinomial(weights, 1))
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
