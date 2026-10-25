"""Running the model on a vLLM server instead of on this machine.

With VLLM_URL set (say http://localhost:8000, where `vllm serve Qwen/Qwen3.5-0.8B` listens), the page runs the
model through VLLMBackend, a Backend (backend.py) like llm.LocalLLM. The harness, the tools and the page stay the
same; what changes is how much of the model the page can see.

The prompt stays exact. Everything of the checkpoint but its weights (the config, the tokenizer, the chat
template) is downloaded into models/ at the revision families.REVISIONS pins, the prompt is rendered and
tokenized here as LocalLLM does it, and the token ids go to vLLM's /v1/completions, so the server reads exactly
what the Context tab shows. Each token comes back as its id with its log probability and the 5 most likely tokens
at its position; from those come p, logprob, the top 5 candidates and, when the token is one of them, its rank
(backend.TOKEN_DETAILS). vLLM's log probabilities are the model's own distribution unless the server was started
with another --logprobs-mode. What the sampler did (p_sampled, kept, which candidates top-k and top-p cut) stays
on the server, so the page leaves it out.

vLLM runs the model its own way (batches, its own kernels, a KV cache that keeps what one step's prompt shares
with the last), so greedy decoding need not match transformers' generate() token for token, and prefill is the
time to the first token, the network included. This backend sends vLLM text only: an image, a clip or a video
becomes a plain file for run_python. Speculative decoding is set when the server starts (--speculative-config),
not in the sidebar; with --per-request-spec-decode-metrics, each step says how many guesses it kept.
"""

import json
import math
import os
import time
from collections import Counter
from typing import Callable, Iterator

import httpx
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoProcessor, GenerationConfig

import families
from backend import Encoded, Sampling
from llm import MODELS_DIR, TOP_N, describe_folder

# The weights stay on the server; downloading them here as well would only take up space.
WEIGHTS = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf"]
# Reading a stream waits for the next chunk this long, which covers a long prefill on a busy server.
TIMEOUT = httpx.Timeout(10.0, read=600.0)


class ServerError(Exception):
    """The vLLM server could not be reached, or refused a request; `text` says so in both of the page's languages."""

    def __init__(self, zh: str, en: str):
        super().__init__(en)
        self.text = {"zh": zh, "en": en}


def connect(url: str, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """A client for the server at `url`, with the key in VLLM_API_KEY if there is one: the same variable `vllm serve`
    reads its --api-key from, so a server started with it in the environment and the page agree."""
    key = os.environ.get("VLLM_API_KEY")
    return httpx.Client(base_url=url.rstrip("/"), headers={"Authorization": f"Bearer {key}"} if key else {},
                        timeout=TIMEOUT, transport=transport)


def served_models(url: str, transport: httpx.BaseTransport | None = None) -> list[dict]:
    """The models the server at `url` serves: for each, the name requests use (id), what it was loaded from (root,
    the Hugging Face id when the server was started with one) and the most tokens it reads at once (max_model_len)."""
    with connect(url, transport) as client:
        return _get(client, "/v1/models")["data"]


def _get(client: httpx.Client, path: str) -> dict:
    try:
        response = client.get(path)
    except httpx.HTTPError as e:
        raise _unreachable(client, e) from e
    _check(response)
    return response.json()


def _check(response: httpx.Response):
    """Raise the server's own message when it refused a request."""
    if response.status_code == 200:
        return
    response.read()
    try:
        body = response.json()
    except ValueError:
        body = {}
    error = body.get("error") if isinstance(body.get("error"), dict) else body  # {"error": {"message": …}}, or older {"message": …}
    message = (error.get("message") if isinstance(error, dict) else None) or response.text or response.reason_phrase
    code = response.status_code
    raise ServerError(f"vLLM 伺服器拒絕了這個請求（HTTP {code}）：{message}", f"The vLLM server refused the request (HTTP {code}): {message}")


def _unreachable(client: httpx.Client, error: Exception) -> ServerError:
    url = str(client.base_url).rstrip("/")
    return ServerError(
        f"連不上 {url} 的 vLLM 伺服器（{error}）。先用 `vllm serve <模型>` 啟動它，或把 VLLM_URL 改成它的位址。",
        f"Could not reach the vLLM server at {url} ({error}). Start it with `vllm serve <model>`, or set VLLM_URL to where it listens.",
    )


def _checkpoint_files(model_id: str) -> str:
    """The model's folder in MODELS_DIR with everything but its weights, at the revision families.REVISIONS pins.
    The server should load the same revision (vllm serve <model> --revision <commit>) so that both read the same
    tokenizer and chat template."""
    if os.path.isdir(model_id):  # a checkpoint on disk rather than a Hugging Face id
        return model_id
    return snapshot_download(model_id, revision=families.REVISIONS.get(model_id), cache_dir=MODELS_DIR, ignore_patterns=WEIGHTS)


def _generation_config(folder: str, config) -> GenerationConfig:
    """The checkpoint's generation_config.json, or, when it ships none (Qwen3.5), one made from its config.json, as
    transformers makes it for the model LocalLLM loads."""
    try:
        return GenerationConfig.from_pretrained(folder)
    except OSError:
        return GenerationConfig.from_model_config(config)


class VLLMBackend:
    def __init__(self, model_id: str, url: str, transport: httpx.BaseTransport | None = None):
        self.model_id = model_id
        self.client = connect(url, transport)
        self.url = str(self.client.base_url).rstrip("/")
        cards = _get(self.client, "/v1/models")["data"]
        card = next((c for c in cards if model_id in (c.get("root"), c["id"])), None)
        if card is None:
            served = ", ".join(c.get("root") or c["id"] for c in cards) or "-"
            raise ServerError(f"{self.url} 的 vLLM 伺服器沒有提供 {model_id}，它提供的是：{served}。",
                              f"The vLLM server at {self.url} does not serve {model_id}; it serves {served}.")
        self.served_as = card["id"]  # what requests name the model, which --served-model-name may have changed
        files = self.folder = _checkpoint_files(model_id)
        config = AutoConfig.from_pretrained(files)
        self.family = families.for_model_type(config.model_type)
        # The same tokenizer and chat template LocalLLM loads, so the prompt is the same string and the same tokens.
        self.processor = AutoProcessor.from_pretrained(files)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.generation_config = _generation_config(files, config)
        eos = self.generation_config.eos_token_id
        self.eos_ids = {i for i in (eos if isinstance(eos, list) else [eos]) if i is not None} | self.token_ids(self.family.stop_tokens)
        self.special_ids = set(self.tokenizer.all_special_ids) | self.eos_ids | self.token_ids(self.family.special_tokens)
        text_cfg = getattr(config, "text_config", config)
        self.context_length = card.get("max_model_len") or text_cfg.max_position_embeddings  # the server's limit (--max-model-len)
        self.modalities = ()  # text only: /v1/completions takes token ids and nothing else
        self.drafter_id = None  # speculative decoding is the server's (--speculative-config): nothing to switch here
        self.drafter = None
        self.info = {
            "model_id": model_id,
            "architecture": (getattr(config, "architectures", None) or [type(config).__name__])[0],
            "layers": text_cfg.num_hidden_layers,
            "hidden_size": text_cfg.hidden_size,
            "attention_heads": text_cfg.num_attention_heads,
            "kv_heads": text_cfg.num_key_value_heads,
            "sliding_window": getattr(text_cfg, "sliding_window", None),
            "layer_types": dict(Counter(getattr(text_cfg, "layer_types", None) or [])),
            "context_length": self.context_length,
            "vocab_size": text_cfg.vocab_size,
            "modalities": [],
            "family": self.family.key,
            "stop_tokens": [self.piece(i) for i in sorted(self.eos_ids)],
            "special_tokens": {self.piece(i): i for i in sorted(self.special_ids)},
            "defaults": self._sampling_defaults(),
            "drafter_id": None,
            "drafter": None,
            "folder": describe_folder(files, self.family.file_notes),
            "server": {"engine": "vLLM", "version": self._version(), "url": self.url, "served_as": self.served_as},
        }

    def _version(self) -> str | None:
        try:
            return _get(self.client, "/version").get("version")
        except ServerError:  # only for the Model tab; a server without the route still runs the model
            return None

    def _sampling_defaults(self) -> dict:
        """The checkpoint's generation_config.json, or the family's recommendation when it ships none."""
        config = self.generation_config
        if getattr(config, "_from_model_config", False) and self.family.sampling:
            return dict(self.family.sampling)
        return {k: getattr(config, k) for k in ("temperature", "top_p", "top_k")}

    def warmup(self):
        """One short request, so that a server that cannot run the model's prompts fails now, not at the first question."""
        prompt = self.encode(self.render([{"role": "user", "content": "Hi"}], None, False), [])
        for _ in self.generate(prompt, Sampling(max_new_tokens=2), lambda: False):
            pass

    def load_drafter(self):
        """Nothing to load: the server drafts, if it was started with --speculative-config."""

    def token_ids(self, pieces) -> set[int]:
        """Ids of the pieces that are single tokens of this tokenizer."""
        ids = ((p, self.tokenizer.convert_tokens_to_ids(p)) for p in pieces)
        return {i for p, i in ids if isinstance(i, int) and self.piece(i) == p}

    def piece(self, token_id: int) -> str:
        return self.tokenizer.convert_ids_to_tokens(token_id)

    def render(self, messages: list[dict], tools: list[dict] | None, thinking: bool, generation_prompt: bool = True) -> str:
        """The chat template's string for the messages and tool schemas, rendered here, as LocalLLM renders it."""
        return self.tokenizer.apply_chat_template(
            messages, tools=tools or None, tokenize=False, add_generation_prompt=generation_prompt, enable_thinking=thinking
        )

    def encode(self, prompt: str, parts: list[dict]) -> Encoded:
        """The prompt's token ids, which are what the server reads. The model reads no media here (modalities is
        empty), so a conversation has none to expand."""
        if parts:
            raise ValueError("VLLMBackend sends vLLM text only; media.save keeps every attachment a plain file for it.")
        ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        return Encoded(prompt, ids, len(ids))

    def generate(self, prompt: Encoded, sampling: Sampling, should_stop: Callable[[], bool]) -> Iterator[dict]:
        """Yield the events LocalLLM.generate yields, from the server's stream: prefill once the first token is back
        (its time is the time to the first token), one event per token, then done."""
        ids = list(prompt.ids)
        channel = "thought" if prompt.text.endswith(self.family.thought_prefix) else "content"
        generated: list[int] = []
        text = ""
        finish = None
        end: dict = {}  # the finish reason and the speculative decoding counts, once the stream has them
        start = time.perf_counter()
        decode_start = None

        tokens = self._stream(self._request(ids, sampling), end)
        try:
            for token_id, details in tokens:
                if decode_start is None:
                    decode_start = time.perf_counter()
                    yield {"type": "prefill", "n_tokens": len(ids), "ms": round((decode_start - start) * 1000, 1)}
                if should_stop():
                    finish = "user_stopped"
                    break
                generated.append(token_id)

                # From here on, as in LocalLLM.generate.
                piece = self.piece(token_id)
                special = token_id in self.special_ids
                if piece in self.family.channel_switch:
                    channel = self.family.channel_switch[piece]
                    token_channel = "control"
                else:
                    token_channel = "control" if special else channel

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
        finally:
            tokens.close()  # closing the stream early tells the server to stop generating

        if decode_start is None:  # no token came back at all
            decode_start = time.perf_counter()
            yield {"type": "prefill", "n_tokens": len(ids), "ms": round((decode_start - start) * 1000, 1)}
        if finish is None:
            reason = end.get("finish_reason")
            if reason not in ("stop", "length"):
                raise ServerError(f"vLLM 伺服器沒有正常結束這一步（finish_reason：{reason}）。",
                                  f"The vLLM server did not finish this step normally (finish_reason: {reason}).")
            finish = "stop_token" if reason == "stop" else "max_new_tokens"
        decode_s = time.perf_counter() - decode_start
        yield {
            "type": "done",
            "finish": finish,
            "stop_piece": self.piece(generated[-1]) if finish == "stop_token" and generated else None,
            "n_tokens": len(generated),
            "decode_ms": round(decode_s * 1000, 1),
            "tok_per_s": round(len(generated) / decode_s, 1) if decode_s > 0 else None,
            "mtp": end.get("mtp"),
            "raw": self.tokenizer.decode(generated, skip_special_tokens=False),
        }

    def _request(self, ids: list[int], s: Sampling) -> dict:
        """The /v1/completions request for one step. Each sidebar setting is sent, and every other sampling setting is
        set to do nothing, so that a default the server took from the checkpoint (a repetition penalty, say) cannot
        change the distribution the page shows. The text comes from decoding the ids here, so the server's is unused."""
        return {
            "model": self.served_as,
            "prompt": ids,
            "max_tokens": max(1, min(s.max_new_tokens, self.context_length - len(ids))),  # vLLM refuses a request that cannot fit
            "temperature": max(s.temperature, 0.0),  # 0 is greedy there as here
            "top_p": s.top_p,
            "top_k": s.top_k,  # 0 is no limit there as here
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stop_token_ids": sorted(self.eos_ids),
            "logprobs": TOP_N,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "stream": True,
            "stream_options": {"include_usage": True},  # the last chunk then carries the speculative decoding counts
        }

    def _stream(self, body: dict, end: dict) -> Iterator[tuple[int, dict]]:
        """Each token the server sends for the request, with what it says about the token. `end` gets the finish
        reason and, from a server started with --per-request-spec-decode-metrics, the speculative decoding counts."""
        try:
            with self.client.stream("POST", "/v1/completions", json=body) as response:
                _check(response)
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ")
                    if data == "[DONE]":
                        return
                    chunk = json.loads(data)
                    if "error" in chunk:
                        message = chunk["error"].get("message") if isinstance(chunk["error"], dict) else chunk["error"]
                        raise ServerError(f"vLLM 伺服器在生成時出錯：{message}", f"The vLLM server failed while generating: {message}")
                    for choice in chunk.get("choices") or []:
                        logprobs = choice.get("logprobs") or {}
                        for n, token_id in enumerate(choice.get("token_ids") or []):
                            yield token_id, self._details(token_id, logprobs, n)
                        if choice.get("finish_reason"):
                            end["finish_reason"] = choice["finish_reason"]
                    if spec := (chunk.get("metrics") or {}).get("speculative_decoding"):
                        # Each verify step is one forward pass of the model over the token before and the guesses.
                        end["mtp"] = {"drafted": spec["num_draft_tokens"], "accepted": spec["num_accepted_draft_tokens"],
                                      "forwards": spec["num_spec_steps"]}
        except httpx.HTTPError as e:
            raise _unreachable(self.client, e) from e

    def _details(self, token_id: int, logprobs: dict, n: int) -> dict:
        """What the server said about the n-th token of a chunk: its log probability, and the 5 most likely tokens
        with theirs, keyed "token_id:<id>". The token's rank is known only when it is one of those 5."""
        token_logprobs, top_logprobs = logprobs.get("token_logprobs") or [], logprobs.get("top_logprobs") or []
        if n >= len(token_logprobs) or token_logprobs[n] is None:
            return {}
        logprob = token_logprobs[n]
        listed = top_logprobs[n] if n < len(top_logprobs) and top_logprobs[n] else {}
        # The chosen token is listed too, even when it is not among the 5; equally likely tokens go lowest id first, as in LocalLLM.
        top = sorted(((int(k.removeprefix("token_id:")), v) for k, v in listed.items() if k.startswith("token_id:")),
                     key=lambda c: (-c[1], c[0]))[:TOP_N]
        details = {
            "p": math.exp(logprob),
            "logprob": logprob,
            "top": [{"id": i, "piece": self.piece(i), "p": math.exp(v)} for i, v in top],
        }
        if any(i == token_id for i, _ in top):
            details["rank"] = 1 + sum(v > logprob for _, v in top)
        return details
