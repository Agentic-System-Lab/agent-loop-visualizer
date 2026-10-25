"""VLLMBackend against a fake vLLM server that speaks the streams of vLLM 0.32: /v1/completions
(entrypoints/openai/completion: token ids with their log probabilities, the top 5 keyed "token_id:<id>", the
finish reason, a last chunk with the usage and the speculative decoding counts) and, on a server started with
--enable-scale-out, /inference/v1/generate (entrypoints/scale_out/token_in_token_out: the same with each token's
rank, the chosen token first in its list, sampling masks with --return-sampling-mask, and media sent next to the
placeholders, which the server expands and, asked to, sends back), in each --logprobs-mode, and refusing what vLLM
refuses once it sends sampling masks.

Most tests use a tokenizer in the fakes' simplified chat format, so they need neither the Hub nor the
weights. The tests with the real tokenizer and chat template run when a model's files are in models/
(test_chat_template.py downloads nothing either). With VLLM_TEST_URL naming a running server, the last
tests ask it for real:  VLLM_TEST_URL=http://localhost:8000 uv run pytest tests/test_vllm_backend.py -k real_server
"""

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub import try_to_load_from_cache
from transformers import AutoProcessor

import families
import media
import render
import vllm_backend
from agent import Run, RunSettings
from backend import TOKEN_DETAILS, TOKEN_FIELDS, Backend, Sampling
from families import GEMMA4, QWEN3_5
from fakes import MEDIA_TOKENS, PLACEHOLDERS, FakeLLM, fake_folder, pieces, qwen_tool_call, tool_call
from html_check import assert_balanced
from i18n import LANGS, pick
from llm import MODELS_DIR
from test_render import render_everything
from vllm_backend import GENERATE, UNREPORTED, ServerError, VLLMBackend

URL = "http://vllm.test:8000"
ANSWER = "答案是 42。<turn|>"
EXAMPLES = Path(__file__).resolve().parent.parent / "assets" / "examples"
KIND_OF_PART = {"image_url": "image", "audio_url": "audio", "video_url": "video"}
LOGIT = 20.0  # how far a fake logit is above the log probability it gives


class Tokenizer:
    """A tokenizer for the fakes' chat format (FakeLLM.render): a control token is one token, any other character
    one token each. Ids are handed out as pieces are first seen, the family's control tokens first."""

    def __init__(self, family: families.Family):
        self.family = family
        self.vocab: dict[str, int] = {}
        self.fake = FakeLLM(family=family)
        for piece in family.special_tokens:
            self.id(piece)
        self.all_special_ids = list(self.vocab.values())

    def id(self, piece: str) -> int:
        return self.vocab.setdefault(piece, len(self.vocab))

    def __call__(self, text: str, add_special_tokens: bool = False):
        return SimpleNamespace(input_ids=[self.id(p) for p in pieces(text, self.family)])

    def convert_tokens_to_ids(self, piece: str):
        return self.vocab.get(piece)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return next(p for p, i in self.vocab.items() if i == token_id)

    def decode(self, ids, skip_special_tokens=False) -> str:
        return "".join(self.convert_ids_to_tokens(i) for i in ids)

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return self.fake.render(messages, tools, enable_thinking, add_generation_prompt)


class Stream(httpx.SyncByteStream):
    """A response body sent line by line as the client reads it, which notes how far it got and whether it was closed."""

    def __init__(self, lines: list[str], server: "Server"):
        self.lines, self.server = lines, server

    def __iter__(self):
        for line in self.lines:
            self.server.sent += 1
            yield line.encode()

    def close(self):
        self.server.closed = True


def sse(chunk: dict) -> str:
    return f"data: {json.dumps(chunk)}\n\n"


class Server:
    """vLLM's server, as far as VLLMBackend asks it anything: /v1/models, /version, streamed /v1/completions and, with
    `scale_out`, /inference/v1/generate, which otherwise answers 403 as vLLM's does. Each streamed answer is the next
    of `outputs`: its tokens until one of the request's stop_token_ids or max_tokens, `per_chunk` to a chunk. The
    chosen token has probability `p` (a dict maps a token's index to another) and the others listed are other tokens;
    a chosen token below 0.01 is outside the top 5. Top-k and top-p leave the chosen token and the others listed at 0.02
    or more, at most top-k of them. What the server lists for each token depends on `logprobs_mode` (values). With
    `masks` (--return-sampling-mask, which vLLM allows only with processed_logprobs), a token's mask is the tokens
    left, and the server refuses temperature 0 and top-k 0 on every route, as vLLM's engine does. The server takes the
    media in `takes`, expanding each placeholder into `expand` tokens as FakeLLM.encode does, and refuses other kinds.
    `spec` is the speculative decoding counts the last chunk carries."""

    tokenizers: dict = {}  # family key -> the tokenizer the hub fixture hands the backend, which this server shares

    def __init__(self, outputs: list[str], root="google/gemma-4-E2B-it", served_as=None, max_model_len=4096, per_chunk=2,
                 scale_out=False, masks=False, takes=media.MEDIA, logprobs_mode=None):
        self.outputs, self.root, self.served_as, self.max_model_len, self.per_chunk = list(outputs), root, served_as or root, max_model_len, per_chunk
        self.scale_out, self.masks, self.takes, self.expand = scale_out, masks, takes, MEDIA_TOKENS
        self.mode = logprobs_mode or ("processed_logprobs" if masks else "raw_logprobs")
        assert self.mode == "processed_logprobs" or not masks, "vLLM does not start with sampling masks in another --logprobs-mode"
        self._tokenizer = None
        self.requests: list[dict] = []  # the streamed /v1/completions bodies
        self.probes: list[dict] = []  # the one-token /v1/completions bodies that are not streamed
        self.generates: list[dict] = []  # the /inference/v1/generate bodies, one-token ones (probes, expansions) included
        self.headers: list[httpx.Headers] = []
        self.p: dict[int, float] = {}
        self.spec = None
        self.finish = None  # a finish reason to end with instead of vLLM's own
        self.fail = None  # (status, body) to answer the next streamed request with
        self.stream_error = None  # an error chunk to send instead of the tokens
        self.sent = 0
        self.closed = False

    @property
    def family(self) -> str:
        return "qwen3_5" if "qwen" in self.root.lower() else "gemma4"

    @property
    def tokenizer(self):
        return self._tokenizer or self.tokenizers[self.family]

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.headers.append(request.headers)
        if request.url.path == "/v1/models":
            card = {"id": self.served_as, "object": "model", "owned_by": "vllm", "root": self.root, "max_model_len": self.max_model_len}
            return httpx.Response(200, json={"object": "list", "data": [card]})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.32.0"})
        body = json.loads(request.content)
        if request.url.path == GENERATE:
            return self.generate(body)
        if not body.get("stream"):
            self.probes.append(body)
            return self.refused(body) or httpx.Response(200, json=self.completion(body))
        self.requests.append(body)
        if refused := self.refused(body):
            return refused
        if self.fail:
            status, error = self.fail
            return httpx.Response(status, json=error)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream(self.lines(body), self))

    # ---------- /inference/v1/generate ----------

    def generate(self, body: dict) -> httpx.Response:
        self.generates.append(body)
        if not self.scale_out:
            message = "Scale-out endpoints are disabled. Restart vllm serve with --enable-scale-out to enable them."
            return httpx.Response(403, json={"error": {"message": message, "type": "PermissionDeniedError", "param": None, "code": 403}})
        if refused := self.refused(body["sampling_params"]):
            return refused
        if self.fail and body.get("stream"):
            status, error = self.fail
            return httpx.Response(status, json=error)
        try:
            prompt = self.expanded(body["token_ids"], body.get("content_parts") or [])
        except ValueError as e:
            return httpx.Response(400, json={"error": {"message": str(e), "type": "BadRequestError", "param": None, "code": 400}})
        if not body.get("stream"):
            choice = {"index": 0, "token_ids": [prompt[-1]], "finish_reason": "length", "stop_reason": None}
            if self.masks:
                choice["sampling_mask"] = [self.kept(prompt[-1], 0.9, body["sampling_params"]["top_k"])]
            reply = {"request_id": "generate-tokens-1", "choices": [choice]}
            if body.get("return_token_ids"):
                reply["prompt_token_ids"] = prompt
            return httpx.Response(200, json=reply)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream(self.generate_lines(body, prompt), self))

    def expanded(self, ids: list[int], parts: list[dict]) -> list[int]:
        """The prompt with each placeholder the chat template wrote expanded for its file, in order."""
        out, at = list(ids), 0
        for part in parts:
            kind = KIND_OF_PART[part["type"]]
            if kind not in self.takes:
                raise ValueError(f"At most 0 {kind}(s) may be provided in one prompt.")
            assert part["url"].startswith("data:"), part["url"][:30]
            mark, start, end = (self.tokenizer.convert_tokens_to_ids(t) for t in PLACEHOLDERS[self.family][kind])
            at = out.index(mark, at)
            tokens = [start, *[mark] * self.expand, end] if self.family == "gemma4" else [mark] * self.expand
            out[at : at + 1] = tokens
            at += len(tokens)
        return out

    def ranked(self, token_id: int, p: float) -> list[tuple[int, float]]:
        """The chosen token and the others listed with it, most likely first."""
        # other tokens: new pieces of the fake tokenizer, or any five ids of a real one
        others = [self.tokenizer.id(f"▁alt{k}") if isinstance(self.tokenizer, Tokenizer) else 100 + k for k in range(5)]
        probs = [0.05, 0.02, 0.01, 0.005] if p >= 0.01 else [0.3, 0.2, 0.1, 0.05, 0.02]
        return sorted([(token_id, p), *zip(others, probs)], key=lambda c: (-c[1], c[0]))

    def kept(self, token_id: int, p: float, top_k: int) -> list[int]:
        """The tokens top-k and top-p leave: the chosen one and the others at 0.02 or more, at most top-k of them."""
        left = [i for i, q in self.ranked(token_id, p) if q >= 0.02 or i == token_id]
        return left[:top_k] if top_k > 0 else left

    def values(self, token_id: int, p: float, params: dict) -> dict[int, float]:
        """What the server lists for the chosen token and the others, most likely first, in its --logprobs-mode: the
        model's log probabilities (raw_logprobs) or logits (raw_logits); or, processed, the logits at the temperature
        over the tokens left, renormalized into log probabilities (processed_logprobs) or not (processed_logits), with
        the tokens cut at -inf, which vLLM sends as UNREPORTED."""
        ranked = self.ranked(token_id, p)
        if self.mode.startswith("raw"):
            return {i: math.log(q) + (LOGIT if self.mode == "raw_logits" else 0.0) for i, q in ranked}
        left = self.kept(token_id, p, params.get("top_k") or 0)
        t = params.get("temperature") or 1.0  # greedy leaves the logits as they are
        scaled = {i: (math.log(q) + LOGIT) / t for i, q in ranked if i in left}
        if self.mode == "processed_logprobs":
            peak = max(scaled.values())
            norm = peak + math.log(sum(math.exp(v - peak) for v in scaled.values()))
            scaled = {i: v - norm for i, v in scaled.items()}
        return {i: scaled.get(i, UNREPORTED) for i, _ in ranked}

    def refused(self, params: dict) -> httpx.Response | None:
        """What vLLM's engine answers, on every route, a request it cannot send a sampling mask for (InputProcessor)."""
        if not self.masks:
            return None
        if params.get("temperature", 1.0) <= 0:
            message = "sampling distribution replay requires temperature > 0"
        elif (params.get("top_k") or 0) <= 0:
            message = "sampling distribution replay requires top_k > 0 to bound sampling mask size, reduce transfer overhead, and avoid potential OOMs"
        else:
            return None
        return httpx.Response(400, json={"error": {"message": message, "type": "BadRequestError", "param": None, "code": 400}})

    def entry(self, token_id: int, p: float, params: dict) -> dict:
        """A token's logprobs as /inference/v1/generate sends them: its own and rank, then the chosen token first and
        the most likely after it, as many as the request's logprobs."""
        values = self.values(token_id, p, params)
        ranks = {i: r for r, i in enumerate(values, 1)}
        listed = [token_id] + [i for i in values if i != token_id]
        return {"token_id": token_id, "logprob": values[token_id], "rank": ranks[token_id],
                "top_logprobs": [{"token_id": i, "logprob": values[i], "rank": ranks[i]} for i in listed[: max(params.get("logprobs") or 0, 1)]]}

    def generate_lines(self, body: dict, prompt: list[int]) -> list[str]:
        params = body["sampling_params"]
        lines = []
        for n, (chunk_ids, probs, finish, stop_reason) in enumerate(self.chunks(params["stop_token_ids"], params["max_tokens"])):
            choice = {"index": 0, "token_ids": chunk_ids, "finish_reason": finish, "stop_reason": stop_reason,
                      "logprobs": {"content": [self.entry(i, p, params) for i, p in zip(chunk_ids, probs)]}}
            if self.masks:
                choice["sampling_mask"] = [self.kept(i, p, params["top_k"]) for i, p in zip(chunk_ids, probs)]
            chunk = {"request_id": "generate-tokens-1", "output_mode": "tokens", "choices": [choice]}
            if n == 0 and body.get("return_token_ids"):
                chunk.update(prompt_token_ids=prompt, mm_placeholders={})
            lines.append(sse(chunk))
        return lines + self.usage(body, len(prompt), {"request_id": "generate-tokens-1", "output_mode": "tokens"})

    # ---------- /v1/completions ----------

    def top(self, token_id: int, p: float, params: dict) -> dict[str, float]:
        """The chosen token's and the most likely tokens' values, as many as the request's logprobs, keyed "token_id:<id>"."""
        values = self.values(token_id, p, params)
        return {f"token_id:{i}": values[i] for i in dict.fromkeys([token_id, *list(values)[: params.get("logprobs") or 0]])}

    def completion(self, body: dict) -> dict:
        """A one-token answer, not streamed: the prompt's last token again, as likely as a token usually is here."""
        token_id = body["prompt"][-1]
        choice = {"index": 0, "text": "", "finish_reason": "length", "stop_reason": None, "token_ids": [token_id],
                  "logprobs": {"text_offset": [0], "token_logprobs": [self.values(token_id, 0.9, body)[token_id]],
                               "tokens": [f"token_id:{token_id}"], "top_logprobs": [self.top(token_id, 0.9, body)]}}
        return {"id": "cmpl-1", "object": "text_completion", "model": self.served_as, "choices": [choice]}

    def lines(self, body: dict) -> list[str]:
        lines = []
        for chunk_ids, probs, finish, stop_reason in self.chunks(body["stop_token_ids"], body["max_tokens"]):
            choice = {
                "index": 0,
                "text": "",  # VLLMBackend decodes the ids itself
                "logprobs": {
                    "text_offset": [0] * len(chunk_ids),
                    "token_logprobs": [self.values(i, p, body)[i] for i, p in zip(chunk_ids, probs)],
                    "tokens": [f"token_id:{i}" for i in chunk_ids],
                    "top_logprobs": [self.top(i, p, body) for i, p in zip(chunk_ids, probs)],
                },
                "finish_reason": finish,
                "stop_reason": stop_reason,
                "token_ids": chunk_ids,
            }
            lines.append(sse({"id": "cmpl-1", "object": "text_completion", "model": self.served_as, "choices": [choice]}))
        return lines + self.usage(body, len(body["prompt"]), {"id": "cmpl-1", "object": "text_completion", "model": self.served_as})

    # ---------- both ----------

    def chunks(self, stop_token_ids: list[int], max_tokens: int):
        """The next output's tokens until a stop token or max_tokens, `per_chunk` at a time, with their probabilities and,
        on the last, the finish reason; or the error chunk a test asked for."""
        if self.stream_error:  # usage() sends the error chunk in their place
            return
        ids = self.tokenizer(self.outputs.pop(0), add_special_tokens=False).input_ids
        out, finish, stop_reason = [], "length", None
        for token_id in ids:
            out.append(token_id)
            if token_id in stop_token_ids:
                finish, stop_reason = "stop", token_id
                break
            if len(out) == max_tokens:
                break
        for start in range(0, len(out), self.per_chunk):
            chunk_ids = out[start : start + self.per_chunk]
            last = start + self.per_chunk >= len(out)
            yield chunk_ids, [self.p.get(start + n, 0.9) for n in range(len(chunk_ids))], (self.finish or finish) if last else None, stop_reason if last else None

    def usage(self, body: dict, prompt_tokens: int, head: dict) -> list[str]:
        if self.stream_error:
            return [sse({"error": self.stream_error}), "data: [DONE]\n\n"]
        lines = []
        if body.get("stream_options", {}).get("include_usage"):
            usage = {**head, "choices": [], "usage": {"prompt_tokens": prompt_tokens}}
            if self.spec:
                usage["metrics"] = {"speculative_decoding": self.spec}
            lines.append(sse(usage))
        return lines + ["data: [DONE]\n\n"]


def configs(tokenizers: dict) -> dict:
    """Each family's config.json as the backend reads it, with an encoder for each kind of media its checkpoint reads."""
    def ids(key, kind):
        return tokenizers[key].vocab[PLACEHOLDERS[key][kind][0]]

    return {
        "gemma4": SimpleNamespace(
            model_type="gemma4", architectures=["Gemma4ForConditionalGeneration"], vision_config={}, audio_config={},
            image_token_id=ids("gemma4", "image"), audio_token_id=ids("gemma4", "audio"), video_token_id=ids("gemma4", "video"),
            text_config=SimpleNamespace(num_hidden_layers=35, hidden_size=1536, num_attention_heads=8, num_key_value_heads=1, sliding_window=512,
                                        layer_types=["sliding_attention"] * 28 + ["full_attention"] * 7, max_position_embeddings=131072, vocab_size=262144)),
        "qwen3_5": SimpleNamespace(
            model_type="qwen3_5", architectures=["Qwen3_5ForConditionalGeneration"], vision_config={},
            image_token_id=ids("qwen3_5", "image"), video_token_id=ids("qwen3_5", "video"),
            text_config=SimpleNamespace(num_hidden_layers=24, hidden_size=1024, num_attention_heads=8, num_key_value_heads=2,
                                        layer_types=["linear_attention"] * 18 + ["full_attention"] * 6, max_position_embeddings=262144, vocab_size=248320)),
    }


@pytest.fixture
def hub(monkeypatch):
    """The checkpoint's files without the Hub: the downloads it was asked for, a config, a processor whose tokenizer is
    in the fakes' format (shared with the fake server), and Gemma's generation config (Qwen3.5 ships none, so its comes
    from the config). Gemma's checkpoint reads images, sound and video; Qwen's images and video."""
    tokenizers = {"gemma4": Tokenizer(GEMMA4), "qwen3_5": Tokenizer(QWEN3_5)}
    seen = {"downloads": [], "tokenizers": tokenizers}
    monkeypatch.setattr(Server, "tokenizers", tokenizers)

    def key(folder: str) -> str:
        return "qwen3_5" if "qwen" in folder.lower() else "gemma4"

    def snapshot_download(model_id, revision=None, cache_dir=None, ignore_patterns=None):
        seen["downloads"].append({"model_id": model_id, "revision": revision, "cache_dir": cache_dir, "ignore": ignore_patterns})
        return f"/hub/{model_id}"

    def processor(folder):
        seen["tokenizer"] = tokenizers[key(folder)]
        hears = {"feature_extractor": SimpleNamespace(sampling_rate=16000)} if key(folder) == "gemma4" else {}
        return SimpleNamespace(tokenizer=seen["tokenizer"], image_processor=object(), video_processor=object(), **hears)

    def generation(folder):
        if key(folder) == "qwen3_5":
            raise OSError("no generation_config.json")
        vocab = tokenizers["gemma4"].vocab
        return SimpleNamespace(eos_token_id=[vocab["<eos>"], vocab["<turn|>"]], temperature=1.0, top_p=0.95, top_k=64)

    def from_model_config(config):
        return SimpleNamespace(eos_token_id=tokenizers["qwen3_5"].vocab["<|endoftext|>"], temperature=None, top_p=None, top_k=None, _from_model_config=True)

    monkeypatch.setattr(vllm_backend, "snapshot_download", snapshot_download)
    monkeypatch.setattr(vllm_backend, "AutoConfig", SimpleNamespace(from_pretrained=lambda folder: configs(tokenizers)[key(folder)]))
    monkeypatch.setattr(vllm_backend, "AutoProcessor", SimpleNamespace(from_pretrained=processor))
    monkeypatch.setattr(vllm_backend, "GenerationConfig", SimpleNamespace(from_pretrained=generation, from_model_config=from_model_config))
    monkeypatch.setattr(vllm_backend, "describe_folder", lambda folder, notes: fake_folder("fake", {"config.json": 4954, "tokenizer.json": 32_169_626}, {}))
    return seen


def connect(server: Server, model_id="google/gemma-4-E2B-it") -> VLLMBackend:
    return VLLMBackend(model_id, URL, transport=server.transport())


def generate(llm: VLLMBackend, prompt="Hi", sampling=None, should_stop=lambda: False):
    encoded = llm.encode(llm.render([{"role": "user", "content": prompt}], None, False), [])
    events = list(llm.generate(encoded, sampling or Sampling(), should_stop))
    return events[0], events[1:-1], events[-1]


# ---------- what goes to the server ----------

def test_the_backend_is_a_backend_that_reads_text_only(hub):
    llm = connect(Server([]))
    assert isinstance(llm, Backend)
    assert llm.modalities == () and llm.info["modalities"] == []
    assert llm.drafter_id is None and llm.drafter is None
    with pytest.raises(ValueError, match="takes no image here, only text"):
        llm.encode("<|image|>", [{"type": "image", "path": "/chat/photo.png"}])


def test_the_prompt_goes_to_the_server_as_the_token_ids_of_the_rendered_prompt(hub):
    server = Server([ANSWER])
    llm = connect(server)
    prompt = llm.render([{"role": "user", "content": "台北現在幾點？"}], None, False)
    generate(llm, "台北現在幾點？")
    body = server.requests[0]
    assert body["prompt"] == hub["tokenizer"](prompt).input_ids
    assert body["model"] == "google/gemma-4-E2B-it" and body["stream"] is True
    # each token's id and its log probability with the top 5, the ids as ids rather than text
    assert body["logprobs"] == 5 and body["return_tokens_as_token_ids"] and body["return_token_ids"]


def test_every_sampling_setting_is_sent_and_every_other_one_does_nothing(hub):
    server = Server([ANSWER])
    llm = connect(server)
    generate(llm, sampling=Sampling(temperature=0.7, top_p=0.9, top_k=0, max_new_tokens=100))
    body = server.requests[0]
    assert (body["temperature"], body["top_p"], body["top_k"], body["max_tokens"]) == (0.7, 0.9, 0, 100)
    assert (body["min_p"], body["repetition_penalty"], body["presence_penalty"], body["frequency_penalty"]) == (0.0, 1.0, 0.0, 0.0)
    assert set(body["stop_token_ids"]) == llm.eos_ids
    assert {llm.piece(i) for i in body["stop_token_ids"]} >= set(GEMMA4.stop_tokens)


def test_greedy_is_temperature_zero(hub):
    server = Server([ANSWER])
    generate(connect(server), sampling=Sampling(temperature=0))
    assert server.requests[0]["temperature"] == 0.0


def test_max_tokens_is_cut_to_what_fits_in_the_servers_context(hub):
    server = Server([ANSWER], max_model_len=40)
    llm = connect(server)
    generate(llm, sampling=Sampling(max_new_tokens=1024))
    assert server.requests[0]["max_tokens"] == 40 - len(server.requests[0]["prompt"])


def test_the_name_the_server_was_given_is_what_requests_use(hub):
    server = Server([ANSWER], served_as="gemma")
    llm = connect(server)
    generate(llm)
    assert server.requests[0]["model"] == "gemma" and llm.model_id == "google/gemma-4-E2B-it"


def test_the_api_key_goes_with_every_request(hub, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    server = Server([ANSWER])
    generate(connect(server))
    assert server.headers and all(h["authorization"] == "Bearer secret" for h in server.headers)


def test_the_checkpoint_downloads_without_its_weights_at_the_pinned_revision(hub):
    connect(Server([]))
    assert hub["downloads"] == [{"model_id": "google/gemma-4-E2B-it", "revision": families.REVISIONS["google/gemma-4-E2B-it"],
                                 "cache_dir": MODELS_DIR, "ignore": vllm_backend.WEIGHTS}]
    assert "*.safetensors" in vllm_backend.WEIGHTS


# ---------- what comes back ----------

def test_events_are_prefill_tokens_done_as_with_local_inference(hub):
    server = Server(["好<channel|>" + ANSWER])
    llm = connect(server)
    encoded = llm.encode(llm.render([{"role": "user", "content": "Hi"}], None, True), [])  # thinking: the prompt ends inside a thought
    events = list(llm.generate(encoded, Sampling(), lambda: False))
    prefill, tokens, done = events[0], events[1:-1], events[-1]
    assert prefill == {"type": "prefill", "n_tokens": encoded.n_tokens, "ms": prefill["ms"]} and prefill["ms"] >= 0
    assert [t["piece"] for t in tokens] == pieces("好<channel|>" + ANSWER, GEMMA4)
    assert [t["index"] for t in tokens] == list(range(len(tokens)))
    assert [t["channel"] for t in tokens][:2] == ["thought", "control"] and tokens[2]["channel"] == "content"
    assert tokens[-1]["special"] and tokens[-1]["channel"] == "control"
    assert "".join(t["text"] for t in tokens) == done["raw"] == "好<channel|>" + ANSWER
    assert done["finish"] == "stop_token" and done["stop_piece"] == "<turn|>" and done["n_tokens"] == len(tokens)
    assert done["mtp"] is None
    assert done["tok_per_s"] is None or done["tok_per_s"] > 0


def test_each_token_carries_its_probability_and_the_top_candidates_but_not_what_the_sampler_did(hub):
    _, tokens, _ = generate(connect(Server([ANSWER])))
    tok = tokens[0]
    assert set(tok) == {"type", *TOKEN_FIELDS, "p", "logprob", "rank", "top"}
    assert not {"p_sampled", "kept", "draft"} & set(tok)  # the sampler's side stays on the server
    assert set(tok) - {"type"} <= {*TOKEN_FIELDS, *TOKEN_DETAILS}
    assert tok["p"] == pytest.approx(0.9) and tok["logprob"] == pytest.approx(math.log(0.9)) and tok["rank"] == 1
    assert [c["p"] for c in tok["top"]] == pytest.approx([0.9, 0.05, 0.02, 0.01, 0.005])
    assert tok["top"][0] == {"id": tok["id"], "piece": tok["piece"], "p": pytest.approx(0.9)} and tok["top"][1]["piece"] == "▁alt0"
    assert all(set(c) == {"id", "piece", "p"} for c in tok["top"])  # no kept: nobody said what top-k and top-p cut


def test_a_token_in_the_top_five_has_its_rank_and_one_outside_has_none(hub):
    server = Server([ANSWER])
    server.p = {1: 0.03, 2: 0.001}
    _, tokens, _ = generate(connect(server))
    assert tokens[1]["rank"] == 2  # only the 0.05 one is more likely
    assert "rank" not in tokens[2] and len(tokens[2]["top"]) == 5 and all(c["id"] != tokens[2]["id"] for c in tokens[2]["top"])


def test_equally_likely_candidates_go_lowest_id_first_as_with_local_inference(hub):
    server = Server([ANSWER])
    llm = connect(server)
    a, b = server.tokenizer.id("▁a"), server.tokenizer.id("▁b")
    details = llm._details(b, {"token_logprobs": [math.log(0.4)], "top_logprobs": [{f"token_id:{b}": math.log(0.4), f"token_id:{a}": math.log(0.4)}]}, 0, Sampling())
    assert [c["id"] for c in details["top"]] == [a, b] and details["rank"] == 1


def test_a_token_the_server_sent_no_log_probability_for_carries_no_details(hub):
    llm = connect(Server([]))
    assert llm._details(5, {}, 0, Sampling()) == {} and llm._details(5, {"token_logprobs": [None], "top_logprobs": [None]}, 0, Sampling()) == {}


def test_reaching_max_tokens_is_the_token_limit(hub):
    _, tokens, done = generate(connect(Server(["很長的回覆" * 5])), sampling=Sampling(max_new_tokens=6))
    assert len(tokens) == 6 and done["finish"] == "max_new_tokens" and done["stop_piece"] is None


def test_stopping_closes_the_stream_before_the_rest_is_sent(hub):
    server = Server(["很長的回覆" * 20 + "<turn|>"], per_chunk=1)
    llm = connect(server)
    seen = []
    _, tokens, done = generate(llm, should_stop=lambda: len(seen.append(1) or seen) > 3)
    assert done["finish"] == "user_stopped" and len(tokens) == 3
    assert server.closed and server.sent < 10  # the server is not read to the end: vLLM stops a request whose client has gone


def test_speculative_decoding_counts_become_the_steps_mtp_counts(hub):
    server = Server([ANSWER])
    server.spec = {"mean_acceptance_length": 2.5, "draft_acceptance_rate": 0.75, "acceptance_histogram": [0, 1, 1],
                   "num_spec_steps": 2, "num_accepted_draft_tokens": 3, "num_draft_tokens": 4, "num_spec_tokens": 2}
    _, tokens, done = generate(connect(server))
    assert done["mtp"] == {"drafted": 4, "accepted": 3, "forwards": 2}
    assert not any("draft" in t for t in tokens)  # vLLM counts the guesses but does not say which tokens they were
    assert server.requests[0]["stream_options"] == {"include_usage": True}


# ---------- when the server says no ----------

def test_a_server_that_cannot_be_reached_says_so_in_both_languages(hub):
    def refuse(request):
        raise httpx.ConnectError("Connection refused")

    with pytest.raises(ServerError) as error:
        VLLMBackend("google/gemma-4-E2B-it", URL, transport=httpx.MockTransport(refuse))
    assert URL in error.value.text["en"] and "vllm serve" in error.value.text["en"]
    assert "連不上" in error.value.text["zh"] and URL in error.value.text["zh"]


def test_a_model_the_server_does_not_serve_is_refused_naming_what_it_serves(hub):
    with pytest.raises(ServerError) as error:
        VLLMBackend("Qwen/Qwen3.5-0.8B", URL, transport=Server([]).transport())
    assert "does not serve Qwen/Qwen3.5-0.8B; it serves google/gemma-4-E2B-it" in error.value.text["en"]
    assert "google/gemma-4-E2B-it" in error.value.text["zh"]


def test_the_servers_own_message_is_kept_when_it_refuses_a_request(hub):
    server = Server([ANSWER])
    llm = connect(server)
    server.fail = (400, {"error": {"message": "This model's maximum context length is 4096 tokens.", "type": "BadRequestError", "code": 400}})
    with pytest.raises(ServerError) as error:
        generate(llm)
    assert error.value.text["en"] == "The vLLM server refused the request (HTTP 400): This model's maximum context length is 4096 tokens."
    assert "HTTP 400" in error.value.text["zh"]


def test_an_error_in_the_middle_of_the_stream_is_raised(hub):
    server = Server([ANSWER])
    llm = connect(server)
    server.stream_error = {"message": "out of memory", "type": "InternalServerError", "code": 500}
    with pytest.raises(ServerError, match="failed while generating: out of memory"):
        generate(llm)


def test_a_step_the_server_ends_some_other_way_is_an_error(hub):
    server = Server([ANSWER])
    llm = connect(server)
    server.finish = "abort"
    with pytest.raises(ServerError, match="finish_reason: abort"):
        generate(llm)


# ---------- the model card ----------

def test_the_context_limit_and_the_specs_come_from_the_server_and_the_config(hub):
    llm = connect(Server([], max_model_len=8192))
    info = llm.info
    assert llm.context_length == info["context_length"] == 8192  # the server's --max-model-len, not the config's 131072
    assert (info["architecture"], info["layers"], info["kv_heads"], info["sliding_window"]) == ("Gemma4ForConditionalGeneration", 35, 1, 512)
    assert info["layer_types"] == {"sliding_attention": 28, "full_attention": 7}
    server = info["server"]
    assert {k: server[k] for k in ("engine", "version", "url", "served_as")} == {"engine": "vLLM", "version": "0.32.0", "url": URL, "served_as": "google/gemma-4-E2B-it"}
    assert not server["generate_api"] and not server["sampling_mask"] and set(server["refused"]) == {"image", "audio", "video"}
    assert info["defaults"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    assert not {"device", "dtype", "parameters", "parameter_parts", "load_seconds"} & set(info)  # nothing here knows them
    assert set(info["stop_tokens"]) >= set(GEMMA4.stop_tokens) and set(info["special_tokens"]) == set(GEMMA4.special_tokens)


def test_a_checkpoint_without_a_generation_config_takes_the_familys_sampling(hub):
    llm = connect(Server([], root="Qwen/Qwen3.5-0.8B"), "Qwen/Qwen3.5-0.8B")
    assert llm.family is QWEN3_5 and llm.info["defaults"] == QWEN3_5.sampling
    assert {"<|endoftext|>", "<|im_end|>"} <= set(llm.info["stop_tokens"])


@pytest.mark.parametrize("lang", LANGS)
def test_the_model_card_says_where_the_model_runs_and_leaves_out_what_is_not_known_here(hub, lang):
    info = connect(Server([])).info
    card, folder = render.model_html(info, lang), render.folder_html(info, lang)
    assert_balanced(card)
    assert_balanced(folder)
    assert URL in card and "vLLM 0.32.0" in card and "params-bar" not in card
    assert ("transformers" not in card) and ("None" not in card)
    assert pick(lang, "只在伺服器上", "on the server only") in folder  # the weights the family notes
    assert pick(lang, "伺服器自己讀權重", "server reads the weights itself") in folder
    # A plain `vllm serve` takes no media, and the card says which and why; the cut candidates stay on the server.
    assert card.count("<li><b>") == 1 and card.count("--enable-scale-out") == 1 and pick(lang, "留在伺服器上", "stays on the server") in card
    assert pick(lang, "<b>圖片、聲音、影片</b>", "<b>image, audio, video</b>") in card


@pytest.mark.parametrize("lang", LANGS)
def test_the_model_card_says_what_a_server_with_scale_out_and_masks_shows_and_which_media_it_refused(hub, lang):
    info = connect(Server([], scale_out=True, masks=True, takes=("image", "video"))).info
    card = render.model_html(info, lang)
    assert_balanced(card)
    assert pick(lang, "不是模型原本的機率，所以看不到模型給的機率和前 5 名", "so the model probability and the top 5 do not show") in card
    assert pick(lang, "看得到剩下幾個候選；這時它不接受 temperature 0 和 top-k 0", "how many candidates were left shows too; it then refuses temperature 0 and top-k 0") in card
    assert card.count("<li><b>") == 1 and pick(lang, "<b>聲音</b>", "<b>audio</b>") in card and "At most 0 audio(s)" in card
    assert pick(lang, "圖片 · 影片", "image · video") in card  # what the model reads here


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize(("mode", "says"), [
    ("processed_logprobs", ("看不到模型給的機率和前 5 名", "the model probability and the top 5 do not show")),
    ("raw_logits", ("回來的是 logits 而不是機率", "what comes back is logits, not probabilities")),
])
def test_the_model_card_says_what_the_numbers_are_in_each_logprobs_mode(hub, lang, mode, says):
    card = render.model_html(connect(Server([], logprobs_mode=mode)).info, lang)
    assert_balanced(card)
    assert pick(lang, *says) in card and f'<span class="tk">--logprobs-mode {mode}</span>' in card
    assert pick(lang, "sampling mask", "sampling masks") not in card and pick(lang, "前 5 名候選；", "and the top 5 candidates;") not in card


# ---------- a whole turn ----------

def test_the_harness_calls_a_tool_and_answers_over_vllm_and_every_moment_of_it_renders(hub):
    server = Server([tool_call("calculator", expression="6*7"), ANSWER])
    llm = connect(server)
    events = []
    Run(llm, [], "6 乘以 7 是多少？", RunSettings(), events.append).run()
    end = events[-1]
    assert end["stop_reason"] == "no_tool_call" and end["final"] == "答案是 42。"
    results = [e for e in events if e["type"] == "tool_result"]
    assert len(results) == 1 and results[0]["ok"] and "42" in results[0]["result"]
    contexts = [e for e in events if e["type"] == "context"]
    assert [r["prompt"] for r in server.requests] == [hub["tokenizer"](c["prompt"]).input_ids for c in contexts]  # what it read is what the page shows
    assert "42" in contexts[1]["prompt"][contexts[1]["added_from"]:]
    for lang in LANGS:
        for n in range(len(events) + 1):
            data = render.build(events[:n])
            for piece in render_everything(data, live=n < len(events), lang=lang):
                assert_balanced(piece)


def test_a_server_that_drafts_shows_its_counts_in_the_step_without_underlines(hub):
    server = Server([ANSWER])
    server.spec = {"num_spec_steps": 3, "num_accepted_draft_tokens": 4, "num_draft_tokens": 6, "acceptance_histogram": [1, 0, 2],
                   "mean_acceptance_length": 2.3, "draft_acceptance_rate": 0.67, "num_spec_tokens": 2}
    events = []
    Run(connect(server), [], "Hi", RunSettings(), events.append).run()
    data = render.build(events)
    html = render.step_html(data["steps"][0], data, lang="en", live=False)
    assert "MTP kept <b>4</b> / 6 guesses" in html and "3 decoding forward passes" in html
    assert "LLM · one inference; the drafter guesses" in html and " drafted\"" not in html and "underline" not in html
    assert "MTP kept" in render.step_rows(data, lang="en")[0]


# ---------- what the server can do: /inference/v1/generate ----------

def test_without_scale_out_steps_go_to_completions_and_no_media_can_go(hub):
    server = Server([ANSWER])
    llm = connect(server)
    assert not llm.generate_api and llm.modalities == () and llm.info["modalities"] == []
    refused = llm.info["server"]["refused"]
    assert set(refused) == {"image", "audio", "video"}
    assert all("--enable-scale-out" in why["en"] and "--enable-scale-out" in why["zh"] for why in refused.values())
    generate(llm)
    assert len(server.requests) == 1 and len(server.generates) == 1  # the one probe, which vLLM answered with 403


def test_with_scale_out_each_step_sends_the_token_ids_to_inference_generate(hub):
    server = Server([ANSWER], scale_out=True)
    llm = connect(server)
    assert llm.generate_api and not llm.masks and llm.modalities == ("image", "audio", "video")
    prompt = llm.render([{"role": "user", "content": "Hi"}], None, False)
    generate(llm, sampling=Sampling(temperature=0.7, top_p=0.9, top_k=40, max_new_tokens=100))
    body = server.generates[-1]
    assert server.requests == [] and body["token_ids"] == hub["tokenizer"](prompt).input_ids and body["stream"] and body["model"] == "google/gemma-4-E2B-it"
    params = body["sampling_params"]
    assert (params["temperature"], params["top_p"], params["top_k"], params["max_tokens"]) == (0.7, 0.9, 40, 100)
    assert (params["min_p"], params["repetition_penalty"], params["presence_penalty"], params["frequency_penalty"]) == (0.0, 1.0, 0.0, 0.0)
    assert params["logprobs"] == 6 and params["detokenize"] is False and set(params["stop_token_ids"]) == llm.eos_ids
    assert "content_parts" not in body and body["stream_options"] == {"include_usage": True}


def test_inference_generate_gives_every_tokens_rank_and_the_top_five_by_rank(hub):
    server = Server([ANSWER], scale_out=True)
    server.p = {1: 0.03, 2: 0.001}
    _, tokens, _ = generate(connect(server))
    assert [t["rank"] for t in tokens[:3]] == [1, 2, 6]  # the third is outside the top 5, and its rank is known all the same
    assert [c["p"] for c in tokens[1]["top"]] == pytest.approx([0.05, 0.03, 0.02, 0.01, 0.005])  # vLLM listed the chosen one first
    assert [c["p"] for c in tokens[2]["top"]] == pytest.approx([0.3, 0.2, 0.1, 0.05, 0.02]) and all(c["id"] != tokens[2]["id"] for c in tokens[2]["top"])
    assert not {"kept", "p_sampled"} & set(tokens[0])  # no sampling masks from this server
    html = render.tip_html({**tokens[2], "index": 2}, GEMMA4, lang="en")
    assert "Drew the token ranked #6, outside the top 5." in html


@pytest.mark.parametrize("temperature", [1.0, 0.5])
def test_sampling_masks_give_the_candidates_left_and_the_sampling_probability_but_not_the_models_own(hub, temperature):
    server = Server([ANSWER], scale_out=True, masks=True)
    llm = connect(server)
    assert llm.masks and llm.info["server"]["sampling_mask"] and llm.logprobs_mode == "processed_logprobs"
    _, tokens, _ = generate(llm, sampling=Sampling(temperature=temperature, top_k=64, top_p=0.95))
    tok = tokens[0]
    assert server.generates[-1]["sampling_params"]["logprobs"] == 6 and server.requests == []
    # vLLM sends the distribution the token was drawn from: the model's at the temperature, over the tokens left.
    weights = [q ** (1 / temperature) for q in (0.9, 0.05, 0.02)]
    assert tok["p_sampled"] == pytest.approx(weights[0] / sum(weights)) and tok["kept"] == 3 and tok["rank"] == 1
    assert not {"p", "logprob", "top"} & set(tok)  # the model's own probabilities stay on the server
    html = render.tip_html(tok, GEMMA4, lang="en")
    assert "candidates left</span><b>3</b>" in html and "sampling probability" in html
    assert "model probability" not in html and "top 5" not in html


@pytest.mark.parametrize("scale_out", [False, True])
def test_a_server_with_masks_refuses_greedy_and_no_top_k_as_vllm_does(hub, scale_out):
    server = Server([ANSWER] * 2, scale_out=True, masks=True)
    llm = connect(server)
    if not scale_out:  # what /v1/completions would answer too: the check is vLLM's engine's, not the route's
        llm.generate_api = False
    with pytest.raises(ServerError, match="requires temperature > 0") as refused:
        generate(llm, sampling=Sampling(temperature=0))
    assert refused.value.status == 400
    with pytest.raises(ServerError, match="requires top_k > 0"):
        generate(llm, sampling=Sampling(top_k=0))


def test_a_server_with_processed_logprobs_but_no_masks_shows_how_many_were_left_when_some_of_the_top_were_cut(hub):
    server = Server([ANSWER], logprobs_mode="processed_logprobs")
    server.p = {1: 0.001}  # every token listed is left
    llm = connect(server)
    assert llm.logprobs_mode == "processed_logprobs" and not llm.masks
    _, tokens, _ = generate(llm, sampling=Sampling(temperature=1.0, top_k=64, top_p=0.95))
    assert tokens[0]["kept"] == 3 and tokens[0]["p_sampled"] == pytest.approx(0.9 / 0.97) and tokens[0]["rank"] == 1
    assert "kept" not in tokens[1] and "rank" not in tokens[1]  # outside the top 5, where /v1/completions gives no rank
    assert not {"p", "logprob", "top"} & set(tokens[0])
    _, tokens, _ = generate(connect(Server([ANSWER], logprobs_mode="processed_logprobs")), sampling=Sampling(temperature=0))
    assert tokens[0]["kept"] == 1 and tokens[0]["p_sampled"] == 1.0  # greedy, which such a server takes


@pytest.mark.parametrize("mode", ["raw_logits", "processed_logits"])
@pytest.mark.parametrize("scale_out", [False, True])
def test_logits_are_no_probabilities_and_only_the_rank_shows(hub, mode, scale_out):
    llm = connect(Server([ANSWER], scale_out=scale_out, logprobs_mode=mode))
    assert llm.logprobs_mode == mode
    _, tokens, _ = generate(llm)
    assert tokens[0]["rank"] == 1 and not {"p", "logprob", "top", "p_sampled"} & set(tokens[0])


@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs", "raw_logits", "processed_logits"])
def test_the_servers_logprobs_mode_is_worked_out_at_load_with_one_token(hub, mode):
    server = Server([], logprobs_mode=mode)
    llm = connect(server)
    assert llm.logprobs_mode == llm.info["server"]["logprobs_mode"] == mode
    probe = server.probes[0]
    assert len(server.probes) == 1 and (probe["max_tokens"], probe["temperature"], probe["top_k"], probe["logprobs"]) == (1, 1.0, 2, 5)


def test_logits_too_large_to_be_probabilities_do_not_overflow():
    assert vllm_backend.logprobs_mode([800.0, 790.0, 700.0, 650.0, 600.0]) == "raw_logits"
    assert vllm_backend.logprobs_mode([800.0, 790.0, UNREPORTED, UNREPORTED, UNREPORTED]) == "processed_logits"


@pytest.mark.parametrize("scale_out", [False, True])
def test_greedy_leaves_one_candidate_drawn_for_sure(hub, scale_out):
    server = Server([ANSWER], scale_out=scale_out)
    _, tokens, _ = generate(connect(server), sampling=Sampling(temperature=0))
    tok = tokens[0]
    assert tok["kept"] == 1 and tok["p_sampled"] == 1.0 and [c["kept"] for c in tok["top"]] == [True, False, False, False, False]


# ---------- images, sound and video ----------

def ask_with(llm: VLLMBackend, *names: str) -> tuple[list[dict], str]:
    """A question with example files attached, as the harness has it: its messages, and the prompt rendered from them."""
    kinds = {"png": "image", "flac": "audio", "mp4": "video"}
    attached = [media.Attachment(n, kinds[n.rsplit(".", 1)[1]], str(EXAMPLES / n), (EXAMPLES / n).stat().st_size) for n in names]
    messages = [{"role": "user", "content": media.user_content("這是什麼？", attached)}]
    return messages, llm.render(messages, None, False)


def test_the_server_is_asked_which_media_it_takes(hub):
    server = Server([], scale_out=True, takes=("image",))
    llm = connect(server)
    assert llm.modalities == ("image",) and llm.info["modalities"] == ["image"]
    refused = llm.info["server"]["refused"]
    assert set(refused) == {"audio", "video"}
    assert "At most 0 audio(s) may be provided in one prompt." in refused["audio"]["en"] and "At most 0 audio(s)" in refused["audio"]["zh"]
    probes = [b for b in server.generates if b.get("content_parts")]
    assert [b["content_parts"][0]["url"].split(";")[0] for b in probes] == ["data:image/png", "data:audio/wav", "data:video/mp4"]
    assert all(b["sampling_params"]["max_tokens"] == 1 for b in server.generates)  # one token each


def test_qwens_checkpoint_has_no_sound_to_ask_about(hub):
    server = Server([], root="Qwen/Qwen3.5-0.8B", scale_out=True)
    llm = connect(server, "Qwen/Qwen3.5-0.8B")
    assert llm.modalities == ("image", "video") and llm.info["server"]["refused"] == {}


def test_media_go_to_the_server_as_files_and_the_context_is_the_prompt_it_expanded(hub):
    server = Server([ANSWER], scale_out=True)
    llm = connect(server)
    messages, prompt = ask_with(llm, "receipt.png", "speech.flac", "ball.mp4")
    before = len(server.generates)
    encoded = llm.encode(prompt, media.parts(messages))
    expansion = server.generates[before]
    assert expansion["token_ids"] == hub["tokenizer"](prompt).input_ids and expansion["return_token_ids"] and not expansion["stream"]
    assert [p["url"].split(";")[0] for p in expansion["content_parts"]] == ["data:image/png", "data:audio/wav", "data:video/mp4"]
    assert expansion["content_parts"][2]["url"].endswith(vllm_backend.base64.b64encode((EXAMPLES / "ball.mp4").read_bytes()).decode())
    # The Context tab gets the prompt as the server read it: each placeholder expanded, as a processor expands it.
    assert encoded.text == FakeLLM(family=GEMMA4).encode(prompt, media.parts(messages)).text
    assert encoded.media_tokens == {"image": MEDIA_TOKENS, "audio": MEDIA_TOKENS, "video": MEDIA_TOKENS}
    assert encoded.n_tokens == len(hub["tokenizer"](encoded.text).input_ids)
    events = list(llm.generate(encoded, Sampling(), lambda: False))
    body = server.generates[-1]
    assert body["stream"] and body["content_parts"] == expansion["content_parts"] and body["token_ids"] == expansion["token_ids"] and body["return_token_ids"]
    assert events[0]["n_tokens"] == encoded.n_tokens and events[-1]["finish"] == "stop_token"


def test_a_server_that_expands_the_media_differently_the_second_time_is_an_error(hub):
    server = Server([ANSWER], scale_out=True)
    llm = connect(server)
    messages, prompt = ask_with(llm, "receipt.png")
    encoded = llm.encode(prompt, media.parts(messages))
    server.expand = MEDIA_TOKENS + 1
    with pytest.raises(ServerError, match="expanded the media differently"):
        list(llm.generate(encoded, Sampling(), lambda: False))


def test_a_question_with_an_image_runs_through_the_harness_and_every_moment_of_it_renders(hub, tmp_path):
    server = Server([ANSWER], scale_out=True, masks=True)
    llm = connect(server)
    photo = media.save(tmp_path, "receipt.png", (EXAMPLES / "receipt.png").read_bytes(), llm.modalities)
    events = []
    Run(llm, [], "這張收據的總計是多少？", RunSettings(), events.append, [photo], tmp_path).run()
    assert events[-1]["stop_reason"] == "no_tool_call" and events[-1]["final"] == "答案是 42。"
    context = next(e for e in events if e["type"] == "context")
    assert context["media_tokens"] == {"image": MEDIA_TOKENS} and "<|image|>" * MEDIA_TOKENS in context["prompt"]
    for lang in LANGS:
        for n in range(len(events) + 1):
            for piece in render_everything(render.build(events[:n]), live=n < len(events), lang=lang):
                assert_balanced(piece)


def test_a_kind_the_server_does_not_take_is_a_plain_file_for_run_python(hub, tmp_path):
    server = Server([ANSWER], scale_out=True, takes=("image",))
    llm = connect(server)
    clip = media.save(tmp_path, "speech.flac", (EXAMPLES / "speech.flac").read_bytes(), llm.modalities)
    assert clip.kind == "file"
    assert media.parts([{"role": "user", "content": media.user_content("?", [clip])}]) == []


# ---------- MTP: which tokens were guesses ----------

DETAILED = {"num_spec_steps": 3, "num_accepted_draft_tokens": 5, "num_draft_tokens": 9, "acceptance_histogram": [1, 0, 1, 1],
            "mean_acceptance_length": 2.67, "draft_acceptance_rate": 0.56, "num_spec_tokens": 3,
            "per_step_accepted": [2, 0, 3], "per_step_drafted": [3, 3, 3]}


def test_per_step_counts_mark_the_guesses_once_the_step_is_done(hub):
    server = Server([ANSWER])  # 8 tokens: one from the prefill, then 2 + 1, 0 + 1 and 3, where the stop token ends it
    server.spec = DETAILED
    events = []
    Run(connect(server), [], "Hi", RunSettings(), events.append).run()
    end = next(e for e in events if e["type"] == "model_end")
    assert end["mtp"] == {"drafted": 9, "accepted": 5, "forwards": 3, "kept_at": [1, 2, 5, 6, 7], "turned_down_at": [3, 4]}
    assert not any("draft" in e for e in events if e["type"] == "token")  # the events stay as the server sent them
    data = render.build(events)
    tokens = data["steps"][0]["tokens"]
    assert [t["index"] for t in tokens if (t.get("draft") or {}).get("accepted")] == [1, 2, 5, 6, 7]
    assert [t["draft"] for t in tokens if t["index"] in (3, 4)] == [{"piece": None, "accepted": False}] * 2
    html = render.step_html(data["steps"][0], data, lang="en", live=False)
    assert html.split('<div class="tokens">')[1].count(' drafted"') == 5
    assert "the server does not say which" in html and 'underline</span> = guessed by the MTP drafter' in html
    assert "MTP kept <b>5</b> / 9 guesses" in html
    live = render.build(events[: next(i for i, e in enumerate(events) if e["type"] == "model_end")])
    assert ' drafted"' not in render.step_html(live["steps"][0], live, lang="en", live=True)  # nobody knows while it generates


def test_per_step_counts_that_do_not_add_up_mark_no_token(hub):
    server = Server([ANSWER])
    server.spec = {**DETAILED, "num_spec_steps": 1, "num_accepted_draft_tokens": 1, "num_draft_tokens": 3, "per_step_accepted": [1], "per_step_drafted": [3]}
    _, _, done = generate(connect(server))
    assert done["mtp"] == {"drafted": 3, "accepted": 1, "forwards": 1}


@pytest.mark.parametrize(("accepted", "drafted", "n_tokens", "marks"), [
    ([2, 0, 3], [3, 3, 3], 9, {"kept_at": [1, 2, 5, 6, 7], "turned_down_at": [3, 4]}),  # every step whole, the last one's bonus token included
    ([2, 0, 3], [3, 3, 3], 6, {"kept_at": [1, 2, 5], "turned_down_at": [3, 4]}),  # a stop token among the last step's guesses
    ([2, 0, 3], [3, 3, 3], 10, None),  # one more token than the steps give: a step without guesses somewhere
    ([2, 0, 3], [3, 3, 3], 5, None),  # the last step gave nothing at all
    ([], [], 1, {"kept_at": [], "turned_down_at": []}),  # one token, from the prefill
    ([1, 1], [1, 2], 5, {"kept_at": [1, 3], "turned_down_at": [4]}),
])
def test_drafted_tokens_follow_the_verify_steps(accepted, drafted, n_tokens, marks):
    assert vllm_backend.drafted_tokens(accepted, drafted, n_tokens) == marks


# ---------- with the real tokenizer and chat template (no weights) ----------

def downloaded(model_id: str) -> str | None:
    """The model's folder in models/ when its tokenizer and chat template are there."""
    revision = families.REVISIONS.get(model_id)
    files = [try_to_load_from_cache(model_id, f, cache_dir=MODELS_DIR, revision=revision) for f in ("config.json", "tokenizer.json", "chat_template.jinja")]
    return os.path.dirname(files[0]) if all(isinstance(f, str) for f in files) else None


REAL_REPLIES = {
    "google/gemma-4-E2B-it": (tool_call("calculator", expression="6*7"), "答案是 42。<turn|>"),
    "Qwen/Qwen3.5-0.8B": (qwen_tool_call("calculator", expression="6*7"), "答案是 42。<|im_end|>"),
}


@pytest.mark.parametrize("model_id", list(families.MODELS))
def test_with_the_real_tokenizer_the_server_reads_the_prompt_the_page_shows(model_id, monkeypatch):
    folder = downloaded(model_id)
    if folder is None:
        pytest.skip(f"{model_id}'s tokenizer is not in models/")
    monkeypatch.setattr(vllm_backend, "snapshot_download", lambda *args, **kwargs: folder)  # the files are here: no Hub
    server = Server(list(REAL_REPLIES[model_id]), root=model_id, per_chunk=3)
    server.tokenizer = AutoProcessor.from_pretrained(folder).tokenizer  # the server answers with the real tokenizer's ids
    llm = connect(server, model_id)
    events = []
    Run(llm, [], "6 乘以 7 是多少？", RunSettings(), events.append).run()
    assert events[-1]["stop_reason"] == "no_tool_call" and events[-1]["final"] == "答案是 42。"
    contexts = [e for e in events if e["type"] == "context"]
    # The ids the server got are the real tokenizer's for the prompt the Context tab shows, special tokens whole.
    for request, context in zip(server.requests, contexts):
        assert request["prompt"] == llm.tokenizer(context["prompt"], add_special_tokens=False).input_ids
        assert llm.tokenizer.decode(request["prompt"], skip_special_tokens=False) == context["prompt"]
    assert "42" in contexts[1]["prompt"][contexts[1]["added_from"]:]
    assert set(llm.family.stop_tokens) <= set(llm.info["stop_tokens"])
    tokens = [e for e in events if e["type"] == "token"]
    assert any(t["special"] and t["channel"] == "control" for t in tokens)
    assert_balanced(render.model_html(llm.info, "zh"))
    assert_balanced(render.folder_html(llm.info, "en"))


# ---------- against a real server ----------

REAL_URL = os.environ.get("VLLM_TEST_URL")
real_server = pytest.mark.skipif(not REAL_URL, reason="set VLLM_TEST_URL to a running vLLM server to ask it for real")


@pytest.fixture(scope="module")
def served():
    card = vllm_backend.served_models(REAL_URL)[0]
    llm = VLLMBackend(card.get("root") or card["id"], REAL_URL)
    llm.warmup()
    return llm


def greedy(llm: VLLMBackend, max_new_tokens: int) -> Sampling:
    """Greedy decoding; or, from a server that sends sampling masks and so refuses temperature 0, top-k 1, which leaves
    the most likely token all the same."""
    if llm.masks:
        return Sampling(temperature=1.0, top_k=1, max_new_tokens=max_new_tokens)
    return Sampling(temperature=0, max_new_tokens=max_new_tokens)


@real_server
def test_real_server_greedy_tokens_carry_what_its_logprobs_mode_tells(served):
    _, tokens, done = generate(served, "用一句話介紹台北。", greedy(served, 24))
    assert tokens and done["n_tokens"] == len(tokens) and "".join(t["text"] for t in tokens) == done["raw"]
    assert all(t["rank"] == 1 for t in tokens)  # greedy takes the top token
    if served.logprobs_mode == "raw_logprobs":
        assert all(t["top"][0]["id"] == t["id"] and 0 < t["p"] <= 1 for t in tokens)
    elif served.logprobs_mode == "processed_logprobs":
        assert all(t["kept"] == 1 and t["p_sampled"] == pytest.approx(1.0) and "p" not in t for t in tokens)


@real_server
def test_real_server_with_scale_out_gives_every_rank_and_with_masks_how_many_were_left(served):
    if not served.generate_api:
        pytest.skip("the server was started without --enable-scale-out")
    _, tokens, _ = generate(served, "用一句話介紹台北。", Sampling(temperature=1.0, top_k=20, top_p=0.95, max_new_tokens=24))
    assert all(t.get("rank", 0) >= 1 for t in tokens)
    if served.masks:  # --return-sampling-mask, with --logprobs-mode processed_logprobs
        assert all(1 <= t["kept"] <= 20 and 0 < t["p_sampled"] <= 1 and not {"p", "top"} & set(t) for t in tokens)


@real_server
@pytest.mark.parametrize("name", ["receipt.png", "speech.flac", "ball.mp4"])
def test_real_server_reads_each_kind_of_media_it_takes(served, name):
    kind = {"png": "image", "flac": "audio", "mp4": "video"}[name.rsplit(".", 1)[1]]
    if kind not in served.modalities:
        pytest.skip(f"the server does not take {kind}: {served.info['server']['refused'].get(kind, {}).get('en', 'the checkpoint has no encoder for it')}")
    messages, prompt = ask_with(served, name)
    encoded = served.encode(prompt, media.parts(messages))
    assert encoded.media_tokens.get(kind, 0) > 0 and encoded.n_tokens > len(served.tokenizer(prompt, add_special_tokens=False).input_ids)
    events = list(served.generate(encoded, greedy(served, 16), lambda: False))
    assert events[-1]["n_tokens"] > 0


@real_server
def test_real_server_with_detailed_spec_counts_marks_guesses_that_add_up(served):
    _, tokens, done = generate(served, "Write a Python function that adds two numbers.", greedy(served, 64))
    if not (done["mtp"] or {}).get("kept_at") and not (done["mtp"] or {}).get("turned_down_at"):
        pytest.skip("the server does not decode speculatively with --per-request-spec-decode-metrics detailed, or its counts did not add up")
    marked = done["mtp"]["kept_at"] + done["mtp"]["turned_down_at"]
    assert all(0 < i < len(tokens) for i in marked) and len(done["mtp"]["kept_at"]) <= done["mtp"]["accepted"]


@real_server
def test_real_server_runs_the_calculator_and_answers(served):
    events = []
    Run(served, [], "What is 1234 * 5678? Use the calculator.", RunSettings(sampling=greedy(served, 256)), events.append).run()
    calls = [e for e in events if e["type"] == "tool_call"]
    assert calls and calls[0]["name"] == "calculator" and not calls[0]["errors"]
    assert events[-1]["stop_reason"] == "no_tool_call" and "7006652" in events[-1]["final"].replace(",", "")
