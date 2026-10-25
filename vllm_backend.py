"""Running the model on a vLLM server instead of on this machine.

With VLLM_URL set (say http://localhost:8000, where `vllm serve Qwen/Qwen3.5-0.8B` listens), the page runs the
model through VLLMBackend, a Backend (backend.py) like simple_backend.LocalLLM. The harness, the tools and the page stay the
same; what changes is how much of the model the page can see, which depends on how the server was started.

The prompt stays exact. Everything of the checkpoint but its weights (the config, the tokenizer, the chat
template) is downloaded into models/ at the revision families.REVISIONS pins, the prompt is rendered and
tokenized here as LocalLLM does it, and the server gets the token ids, so it reads exactly what the Context tab
shows. Each token comes back as its id with its log probability and those of the most likely tokens at its
position, from which come p, logprob and the top 5 candidates (backend.TOKEN_DETAILS). vLLM's log probabilities
are the model's own distribution unless the server was started with another --logprobs-mode.

A plain `vllm serve` has /v1/completions: text only, a token's rank only when it is one of the top 5, and nothing
of what the sampler did. Started with --enable-scale-out it also has /inference/v1/generate, which takes the same
token ids and gives more: every token's rank; with --return-sampling-mask, the tokens top-k and top-p left at each
position (so how many candidates were left, which of the top 5 were cut and, when every one left has its
probability in the response, the sampling probability); and images, sound and video. Those go in as they are
(content_parts) next to the placeholders the chat template wrote, the server expands each placeholder into the
tokens its file takes, and encode() asks it for that expansion first (a one-token request), so the Context tab
shows the prompt as the server reads it. At load, the backend asks the server which of the checkpoint's media it
takes; a kind it does not take is a plain file for run_python, and the Model tab says why.

vLLM runs the model its own way (batches, its own kernels, a KV cache that keeps what one step's prompt shares
with the last), so greedy decoding need not match transformers' generate() token for token, and prefill is the
time to the first token, the network included. Speculative decoding is set when the server starts
(--speculative-config), not in the sidebar. With --per-request-spec-decode-metrics, each step says how many
guesses were kept; with `detailed`, it says how many in each verify step, from which the page underlines the
guesses once the step is done (drafted_tokens).
"""

import base64
import io
import json
import math
import os
import time
import wave
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator

import av
import httpx
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoConfig, AutoProcessor, GenerationConfig

import families
import media
from backend import Encoded, Sampling
from simple_backend import MODELS_DIR, TOP_N, describe_folder

# The weights stay on the server; downloading them here as well would only take up space.
WEIGHTS = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf"]
# Reading a stream waits for the next chunk this long, which covers a long prefill on a busy server.
TIMEOUT = httpx.Timeout(10.0, read=600.0)
# vLLM's token-in, token-out route, there when the server was started with --enable-scale-out.
GENERATE = "/inference/v1/generate"
# Log probabilities asked for with sampling masks on: vLLM's default --max-logprobs. With top-k at 20 or less, or a
# nucleus that small, every token left has its probability in the response, and so the sampling probability is known.
MASKED_LOGPROBS = 20
AUDIO_RATE = 16000  # a clip is sent at the rate of the feature extractor, when the processor has none of its own


class ServerError(Exception):
    """The vLLM server could not be reached, or refused a request; `text` says so in both of the page's languages,
    and `status` is the HTTP status of a refusal."""

    def __init__(self, zh: str, en: str, status: int | None = None):
        super().__init__(en)
        self.text = {"zh": zh, "en": en}
        self.status = status


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


def _post(client: httpx.Client, path: str, body: dict) -> dict:
    try:
        response = client.post(path, json=body)
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
    message = (error.get("message") or error.get("detail") if isinstance(error, dict) else None) or response.text or response.reason_phrase
    code = response.status_code
    raise ServerError(f"vLLM 伺服器拒絕了這個請求（HTTP {code}）：{message}", f"The vLLM server refused the request (HTTP {code}): {message}", code)


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


def checkpoint_modalities(config, processor) -> tuple[str, ...]:
    """What the checkpoint reads besides text, worked out as LocalLLM._modalities does: an encoder in the config and a
    part of the processor to feed it. Whether the server takes each one is asked separately (VLLMBackend._probe)."""
    reads = {
        "image": getattr(config, "vision_config", None) is not None and hasattr(processor, "image_processor"),
        "audio": getattr(config, "audio_config", None) is not None and hasattr(processor, "feature_extractor"),
        "video": getattr(config, "video_token_id", None) is not None and hasattr(processor, "video_processor"),
    }
    return tuple(kind for kind, ok in reads.items() if ok)


def _data_url(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _wav(samples: np.ndarray, rate: int) -> bytes:
    """Mono float samples as 16-bit WAV, which every audio decoder reads."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def content_part(part: dict, audio_rate: int) -> dict:
    """One image, clip or video of the conversation as /inference/v1/generate takes it, a data URL. An image goes as
    PNG, turned upright by its EXIF, and a clip as WAV, both decoded here as LocalLLM decodes them (media.py); a video
    goes as the file itself, and the server picks its frames."""
    kind, path = part["type"], part["path"]
    if kind == "image":
        return {"type": "image_url", "url": _data_url("image/png", _png(media.load_image(path)))}
    if kind == "audio":
        return {"type": "audio_url", "url": _data_url("audio/wav", _wav(media.load_audio(path, audio_rate), audio_rate))}
    return {"type": "video_url", "url": _data_url(media.mime(path) or "video/x-matroska", Path(path).read_bytes())}


def probe_part(kind: str, audio_rate: int = AUDIO_RATE) -> dict:
    """A tiny image, clip or video made here, to ask the server whether it takes that kind at all."""
    if kind == "image":
        return {"type": "image_url", "url": _data_url("image/png", _png(Image.new("RGB", (64, 64), (128, 128, 128))))}
    if kind == "audio":
        return {"type": "audio_url", "url": _data_url("audio/wav", _wav(np.zeros(audio_rate // 2, dtype=np.float32), audio_rate))}
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:  # 40 frames, as many as any processor here samples
        stream = container.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for n in range(40):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), n * 6, dtype=np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return {"type": "video_url", "url": _data_url("video/mp4", buffer.getvalue())}


def drafted_tokens(accepted: list[int], drafted: list[int], n_tokens: int) -> dict | None:
    """Which of a step's tokens were guesses the model kept (kept_at) and which replace a guess it turned down
    (turned_down_at), from vLLM's per-step counts (--per-request-spec-decode-metrics detailed). The first token comes
    from the prefill; after it, each verify step gives the guesses it kept and then one token of the model's own,
    which replaces the next guess when it kept fewer than were drafted. vLLM does not count a step without guesses,
    so when the steps do not account for every token, nobody can tell which tokens were guessed: None."""
    kept, turned_down, at = [], [], 1
    for n_kept, n_drafted in zip(accepted, drafted):
        kept += range(at, at + n_kept)
        if n_kept < n_drafted:
            turned_down.append(at + n_kept)
        at += n_kept + 1
    last = accepted[-1] + 1 if accepted else 1
    if not at - last < n_tokens <= at:  # only a stop token or the token limit may cut the last step short
        return None
    return {"kept_at": [i for i in kept if i < n_tokens], "turned_down_at": [i for i in turned_down if i < n_tokens]}


def mtp_counts(spec: dict | None, n_tokens: int) -> dict | None:
    """A step's speculative decoding counts as the page shows MTP's (drafted, accepted, forwards), plus which tokens
    were guesses when the server counted each verify step."""
    if not spec:
        return None
    counts = {"drafted": spec["num_draft_tokens"], "accepted": spec["num_accepted_draft_tokens"], "forwards": spec["num_spec_steps"]}
    accepted, drafted = spec.get("per_step_accepted"), spec.get("per_step_drafted")
    if accepted is not None and drafted is not None and len(accepted) == len(drafted):
        counts.update(drafted_tokens(accepted, drafted, n_tokens) or {})
    return counts


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
        self.audio_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", AUDIO_RATE)
        self.media_token_ids = {kind: getattr(config, f"{kind}_token_id", None) for kind in media.MEDIA}
        self.drafter_id = None  # speculative decoding is the server's (--speculative-config): nothing to switch here
        self.drafter = None
        # What the server does beyond /v1/completions, and so which of the checkpoint's media the model reads here.
        readable = checkpoint_modalities(config, self.processor)
        self.generate_api, self.masks, refused = self._probe(readable)
        self.modalities = tuple(kind for kind in readable if kind not in refused)
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
            "modalities": list(self.modalities),
            "family": self.family.key,
            "stop_tokens": [self.piece(i) for i in sorted(self.eos_ids)],
            "special_tokens": {self.piece(i): i for i in sorted(self.special_ids)},
            "defaults": self._sampling_defaults(),
            "drafter_id": None,
            "drafter": None,
            "folder": describe_folder(files, self.family.file_notes),
            "server": {"engine": "vLLM", "version": self._version(), "url": self.url, "served_as": self.served_as,
                       "generate_api": self.generate_api, "sampling_mask": self.masks, "refused": refused},
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

    def _probe(self, kinds: tuple[str, ...]) -> tuple[bool, bool, dict]:
        """Ask the server, with one-token requests, whether it has /inference/v1/generate (--enable-scale-out), whether
        it sends sampling masks (--return-sampling-mask), and which of `kinds` (the checkpoint's media) it takes. A kind
        it does not take comes back with why, in both of the page's languages."""
        hi = self.encode(self.render([{"role": "user", "content": "Hi"}], None, False), []).ids
        try:
            out = self._generate_once(hi, [], top_k=2)
        except ServerError as e:
            if e.status is None:  # unreachable, not refused
                raise
            if e.status == 403:  # what vLLM answers on that route without --enable-scale-out
                why = {"zh": "伺服器啟動時沒有加 `--enable-scale-out`，只有 /v1/completions 可用，而它只收文字。",
                       "en": "the server was started without `--enable-scale-out`, so only /v1/completions is there, which takes text only."}
            elif e.status == 404:  # a vLLM without the route
                why = {"zh": "這個版本的 vLLM 沒有 /inference/v1/generate，只有 /v1/completions 可用，而它只收文字；新版 vLLM 加上 `--enable-scale-out` 才有。",
                       "en": "this vLLM has no /inference/v1/generate, only /v1/completions, which takes text only; a newer vLLM started with `--enable-scale-out` has it."}
            else:
                why = {"zh": f"伺服器的 /inference/v1/generate 拒絕了請求，所以只用 /v1/completions，它只收文字（{e.text['en']}）",
                       "en": f"the server's /inference/v1/generate refused a request, so only /v1/completions is used, which takes text only ({e.text['en']})"}
            return False, False, {kind: why for kind in kinds}
        masks = any(choice.get("sampling_mask") is not None for choice in out.get("choices") or [])
        refused = {}
        for kind in kinds:
            prompt = self.render([{"role": "user", "content": [{"type": kind}, {"type": "text", "text": "Hi"}]}], None, False)
            try:
                self._generate_once(self.tokenizer(prompt, add_special_tokens=False).input_ids, [probe_part(kind, self.audio_rate)])
            except ServerError as e:
                refused[kind] = {"zh": f"伺服器不收：{e.text['en']}", "en": f"the server refused it: {e.text['en']}"}
        return True, masks, refused

    def _generate_once(self, ids: list[int], parts: list[dict], top_k: int = 1, return_token_ids: bool = False) -> dict:
        """A one-token request to /inference/v1/generate, without streaming. Top-k keeps the sampling mask, if the server
        sends one, to `top_k` tokens."""
        body = {"model": self.served_as, "token_ids": ids, "stream": False, "return_token_ids": return_token_ids,
                "sampling_params": {"max_tokens": 1, "temperature": 1.0, "top_k": top_k, "detokenize": False}}
        if parts:
            body["content_parts"] = parts
        return _post(self.client, GENERATE, body)

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
        """The prompt's token ids, which are what the server reads. With media (`parts`, in the order of their
        placeholders), the server expands each placeholder into the tokens its file takes, so it is asked how first, with
        a one-token request; the text and the count are of the prompt as the server reads it, and the ids stay the ones
        the chat template wrote, which generate() sends again with the files."""
        ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        if not parts:
            return Encoded(prompt, ids, len(ids))
        if unread := {p["type"] for p in parts} - set(self.modalities):  # media.save makes those plain files, so this is a bug
            raise ValueError(f"The vLLM server takes no {', '.join(sorted(unread))} here, only {', '.join(self.modalities) or 'text'}.")
        files = [content_part(p, self.audio_rate) for p in parts]
        expanded = self._generate_once(ids, files, return_token_ids=True)["prompt_token_ids"]
        counts = Counter(kind for token in expanded for kind, media_id in self.media_token_ids.items() if media_id is not None and token == media_id)
        return Encoded(self.tokenizer.decode(expanded, skip_special_tokens=False), ids, len(expanded),
                       {"content_parts": files, "prompt_token_ids": expanded}, dict(counts))

    def generate(self, prompt: Encoded, sampling: Sampling, should_stop: Callable[[], bool]) -> Iterator[dict]:
        """Yield the events LocalLLM.generate yields, from the server's stream: prefill once the first token is back
        (its time is the time to the first token), one event per token, then done."""
        channel = "thought" if prompt.text.endswith(self.family.thought_prefix) else "content"
        generated: list[int] = []
        text = ""
        finish = None
        end: dict = {}  # the finish reason and the speculative decoding counts, once the stream has them
        start = time.perf_counter()
        decode_start = None

        # /inference/v1/generate when the server has it, except where its sampling mask would list the whole vocabulary
        # (greedy, or top-k off) and tell nothing: then /v1/completions, unless media need the other.
        bounded = sampling.temperature > 0 and sampling.top_k > 0
        if self.generate_api and (prompt.media or not self.masks or bounded):
            tokens = self._stream_generate(prompt, sampling, end)
        else:
            tokens = self._stream(self._request(prompt, sampling), sampling, end)
        try:
            for token_id, details in tokens:
                if decode_start is None:
                    decode_start = time.perf_counter()
                    yield {"type": "prefill", "n_tokens": prompt.n_tokens, "ms": round((decode_start - start) * 1000, 1)}
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
            yield {"type": "prefill", "n_tokens": prompt.n_tokens, "ms": round((decode_start - start) * 1000, 1)}
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
            "mtp": mtp_counts(end.get("spec"), len(generated)) if finish != "user_stopped" else None,
            "raw": self.tokenizer.decode(generated, skip_special_tokens=False),
        }

    def _sampling_params(self, prompt: Encoded, s: Sampling) -> dict:
        """Each sidebar setting, and every other sampling setting set to do nothing, so that a default the server took
        from the checkpoint (a repetition penalty, say) cannot change the distribution the page shows."""
        return {
            "max_tokens": max(1, min(s.max_new_tokens, self.context_length - prompt.n_tokens)),  # vLLM refuses a request that cannot fit
            "temperature": max(s.temperature, 0.0),  # 0 is greedy there as here
            "top_p": s.top_p,
            "top_k": s.top_k,  # 0 is no limit there as here
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stop_token_ids": sorted(self.eos_ids),
        }

    def _request(self, prompt: Encoded, s: Sampling) -> dict:
        """The /v1/completions request for one step. The text comes from decoding the ids here, so the server's is unused."""
        return {
            "model": self.served_as,
            "prompt": list(prompt.ids),
            **self._sampling_params(prompt, s),
            "logprobs": TOP_N,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "stream": True,
            "stream_options": {"include_usage": True},  # the last chunk then carries the speculative decoding counts
        }

    def _lines(self, path: str, body: dict) -> Iterator[dict]:
        """The chunks of a streamed response, as dicts; an error chunk is raised."""
        try:
            with self.client.stream("POST", path, json=body) as response:
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
                    yield chunk
        except httpx.HTTPError as e:
            raise _unreachable(self.client, e) from e

    def _stream(self, body: dict, s: Sampling, end: dict) -> Iterator[tuple[int, dict]]:
        """Each token /v1/completions sends, with what it says about the token. `end` gets the finish reason and, from a
        server started with --per-request-spec-decode-metrics, the speculative decoding counts."""
        for chunk in self._lines("/v1/completions", body):
            for choice in chunk.get("choices") or []:
                logprobs = choice.get("logprobs") or {}
                for n, token_id in enumerate(choice.get("token_ids") or []):
                    yield token_id, self._details(token_id, logprobs, n, s)
                if choice.get("finish_reason"):
                    end["finish_reason"] = choice["finish_reason"]
            if spec := (chunk.get("metrics") or {}).get("speculative_decoding"):
                end["spec"] = spec

    def _details(self, token_id: int, logprobs: dict, n: int, s: Sampling) -> dict:
        """What /v1/completions said about the n-th token of a chunk: its log probability, and those of the 5 most likely
        tokens, keyed "token_id:<id>". The token's rank is known only when it is one of those 5."""
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
        return self._greedy(details, token_id, s)

    def _stream_generate(self, prompt: Encoded, s: Sampling, end: dict) -> Iterator[tuple[int, dict]]:
        """Each token /inference/v1/generate sends for the step, with what it says about the token, and `end` as in
        _stream. With media, the server's first chunk has the prompt as it expanded it, which must be the one encode()
        was told and the Context tab shows."""
        body = {
            "model": self.served_as,
            "token_ids": list(prompt.ids),
            "sampling_params": {
                **self._sampling_params(prompt, s),
                # The chosen token comes first in the list, so one more than the top 5; with masks, as many as are
                # needed to know the probability of every token top-k and top-p leave.
                "logprobs": MASKED_LOGPROBS if self.masks else TOP_N + 1,
                "detokenize": False,
            },
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if prompt.media:
            body.update(content_parts=prompt.media["content_parts"], return_token_ids=True)
        for chunk in self._lines(GENERATE, body):
            if prompt.media and "prompt_token_ids" in chunk and chunk["prompt_token_ids"] != prompt.media["prompt_token_ids"]:
                raise ServerError("vLLM 伺服器這次展開媒體的方式和剛才不一樣，脈絡分頁上的 prompt 不是它讀到的。",
                                  "The vLLM server expanded the media differently this time, so the prompt in the Context tab is not the one it read.")
            for choice in chunk.get("choices") or []:
                content = (choice.get("logprobs") or {}).get("content") or []
                masks = choice.get("sampling_mask") or []
                for n, token_id in enumerate(choice.get("token_ids") or []):
                    entry = content[n] if n < len(content) else None
                    yield token_id, self._generate_details(token_id, entry, masks[n] if n < len(masks) else None, s)
                if choice.get("finish_reason"):
                    end["finish_reason"] = choice["finish_reason"]
            if spec := (chunk.get("metrics") or {}).get("speculative_decoding"):
                end["spec"] = spec

    def _generate_details(self, token_id: int, entry: dict | None, mask: list[int] | None, s: Sampling) -> dict:
        """What /inference/v1/generate said about a token: its log probability and rank, the most likely tokens with
        theirs (the chosen one first), and, with sampling masks, the tokens top-k and top-p left. From the mask come how
        many candidates were left and which of the top 5 were cut; when every token left has its probability in the
        list, the probability the token was drawn with is known too: their probabilities at the temperature, renormalized."""
        if entry is None or (entry["logprob"] <= -9999.0 and not entry.get("top_logprobs")):  # vLLM's mark for "not reported"
            return {}
        logprob = entry["logprob"]
        listed = {e["token_id"]: e["logprob"] for e in entry.get("top_logprobs") or []}
        rank = entry.get("rank")
        ranked = sorted(listed.items(), key=lambda c: (-c[1], c[0]))
        if rank is not None and rank > TOP_N:  # listed only because it was chosen
            ranked = [c for c in ranked if c[0] != token_id]
        top = [{"id": i, "piece": self.piece(i), "p": math.exp(v)} for i, v in ranked[:TOP_N]]
        details = {"p": math.exp(logprob), "logprob": logprob, "top": top, **({"rank": rank} if rank else {})}
        if s.temperature > 0 and mask is not None:
            left = set(mask)
            details["kept"] = len(left)
            for c in top:
                c["kept"] = c["id"] in left
            if token_id in left and left <= listed.keys():
                scaled = {i: listed[i] / s.temperature for i in left}
                peak = max(scaled.values())
                details["p_sampled"] = math.exp(scaled[token_id] - peak) / sum(math.exp(v - peak) for v in scaled.values())
        return self._greedy(details, token_id, s)

    def _greedy(self, details: dict, token_id: int, s: Sampling) -> dict:
        """In greedy decoding (temperature 0) only the most likely token can be chosen: one candidate left, drawn with
        probability 1, and the rest of the top 5 cut. That follows from the settings, not from the server."""
        if s.temperature > 0 or not details:
            return details
        for c in details["top"]:
            c["kept"] = c["id"] == token_id
        return {**details, "kept": 1, "p_sampled": 1.0}
