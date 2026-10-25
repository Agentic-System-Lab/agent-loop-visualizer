"""VLLMBackend against a fake vLLM server that speaks the stream of vLLM 0.32's /v1/completions
(entrypoints/openai/completion): token ids with their log probabilities, the top 5 keyed "token_id:<id>",
the finish reason, and a last chunk with the usage and the speculative decoding counts.

Most tests use a tokenizer in the fakes' simplified chat format, so they need neither the Hub nor the
weights. The tests with the real tokenizer and chat template run when a model's files are in models/
(test_chat_template.py downloads nothing either). With VLLM_TEST_URL naming a running server, the last
tests ask it for real:  VLLM_TEST_URL=http://localhost:8000 uv run pytest tests/test_vllm_backend.py -k real_server
"""

import json
import math
import os
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub import try_to_load_from_cache

import families
import render
import vllm_backend
from agent import Run, RunSettings
from backend import TOKEN_DETAILS, TOKEN_FIELDS, Backend, Sampling
from families import GEMMA4, QWEN3_5
from fakes import FakeLLM, fake_folder, pieces, qwen_tool_call, tool_call
from html_check import assert_balanced
from i18n import LANGS, pick
from llm import MODELS_DIR
from test_render import render_everything
from vllm_backend import ServerError, VLLMBackend

URL = "http://vllm.test:8000"
ANSWER = "答案是 42。<turn|>"


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


class Server:
    """vLLM's OpenAI-compatible server, as far as VLLMBackend asks it anything: /v1/models, /version and streamed
    /v1/completions. Each completion is the next of `outputs`: its tokens until one of the request's stop_token_ids
    or max_tokens, `per_chunk` to a chunk. The chosen token has probability `p` (a dict maps a token's index to
    another), and the rest of the top 5 are other tokens; a chosen token below 0.01 is outside the top 5."""

    def __init__(self, outputs: list[str], root="google/gemma-4-E2B-it", served_as=None, max_model_len=4096, per_chunk=2):
        self.outputs, self.root, self.served_as, self.max_model_len, self.per_chunk = list(outputs), root, served_as or root, max_model_len, per_chunk
        self.tokenizer = None  # the backend's, once it has one
        self.requests: list[dict] = []
        self.headers: list[httpx.Headers] = []
        self.p: dict[int, float] = {}
        self.spec = None  # the speculative decoding counts the last chunk carries
        self.finish = None  # a finish reason to end with instead of vLLM's own
        self.fail = None  # (status, body) to answer the next completion with
        self.stream_error = None  # an error chunk to send instead of the tokens
        self.sent = 0
        self.closed = False

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
        self.requests.append(body)
        if self.fail:
            status, error = self.fail
            return httpx.Response(status, json=error)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream(self.lines(body), self))

    def top(self, token_id: int, p: float) -> dict[str, float]:
        # other tokens: new pieces of the fake tokenizer, or any five ids of a real one
        others = [self.tokenizer.id(f"▁alt{k}") if isinstance(self.tokenizer, Tokenizer) else 100 + k for k in range(5)]
        probs = [0.05, 0.02, 0.01, 0.005] if p >= 0.01 else [0.3, 0.2, 0.1, 0.05, 0.02]
        listed = {f"token_id:{i}": math.log(q) for i, q in zip(others, probs)}
        return {f"token_id:{token_id}": math.log(p), **listed}

    def lines(self, body: dict) -> list[str]:
        if self.stream_error:
            return [f"data: {json.dumps({'error': self.stream_error})}\n\n", "data: [DONE]\n\n"]
        ids = self.tokenizer(self.outputs.pop(0), add_special_tokens=False).input_ids
        out, finish, stop_reason = [], "length", None
        for token_id in ids:
            out.append(token_id)
            if token_id in body["stop_token_ids"]:
                finish, stop_reason = "stop", token_id
                break
            if len(out) == body["max_tokens"]:
                break
        lines = []
        for start in range(0, len(out), self.per_chunk):
            chunk_ids = out[start : start + self.per_chunk]
            last = start + self.per_chunk >= len(out)
            probs = [self.p.get(start + n, 0.9) for n in range(len(chunk_ids))]
            choice = {
                "index": 0,
                "text": "",  # VLLMBackend decodes the ids itself
                "logprobs": {
                    "text_offset": [0] * len(chunk_ids),
                    "token_logprobs": [math.log(p) for p in probs],
                    "tokens": [f"token_id:{i}" for i in chunk_ids],
                    "top_logprobs": [self.top(i, p) for i, p in zip(chunk_ids, probs)],
                },
                "finish_reason": (self.finish or finish) if last else None,
                "stop_reason": stop_reason if last else None,
                "token_ids": chunk_ids,
            }
            lines.append(f"data: {json.dumps({'id': 'cmpl-1', 'object': 'text_completion', 'model': self.served_as, 'choices': [choice]})}\n\n")
        if body.get("stream_options", {}).get("include_usage"):
            usage = {"id": "cmpl-1", "object": "text_completion", "model": self.served_as, "choices": [],
                     "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": len(out), "total_tokens": len(body["prompt"]) + len(out)}}
            if self.spec:
                usage["metrics"] = {"speculative_decoding": self.spec}
            lines.append(f"data: {json.dumps(usage)}\n\n")
        return lines + ["data: [DONE]\n\n"]


CONFIGS = {
    "gemma4": SimpleNamespace(model_type="gemma4", architectures=["Gemma4ForConditionalGeneration"], text_config=SimpleNamespace(
        num_hidden_layers=35, hidden_size=1536, num_attention_heads=8, num_key_value_heads=1, sliding_window=512,
        layer_types=["sliding_attention"] * 28 + ["full_attention"] * 7, max_position_embeddings=131072, vocab_size=262144)),
    "qwen3_5": SimpleNamespace(model_type="qwen3_5", architectures=["Qwen3_5ForConditionalGeneration"], text_config=SimpleNamespace(
        num_hidden_layers=24, hidden_size=1024, num_attention_heads=8, num_key_value_heads=2,
        layer_types=["linear_attention"] * 18 + ["full_attention"] * 6, max_position_embeddings=262144, vocab_size=248320)),
}


@pytest.fixture
def hub(monkeypatch):
    """The checkpoint's files without the Hub: the downloads it was asked for, a config, a tokenizer in the fakes' format, and
    Gemma's generation config (Qwen3.5 ships none, so its comes from the config)."""
    seen = {"downloads": []}

    def snapshot_download(model_id, revision=None, cache_dir=None, ignore_patterns=None):
        seen["downloads"].append({"model_id": model_id, "revision": revision, "cache_dir": cache_dir, "ignore": ignore_patterns})
        return f"/hub/{model_id}"

    def config(folder):
        return CONFIGS["qwen3_5" if "qwen" in folder.lower() else "gemma4"]

    def processor(folder):
        seen["tokenizer"] = Tokenizer(QWEN3_5 if "qwen" in folder.lower() else GEMMA4)
        return SimpleNamespace(tokenizer=seen["tokenizer"])

    def generation(folder):
        if "qwen" in folder.lower():
            raise OSError("no generation_config.json")
        vocab = seen["tokenizer"].vocab
        return SimpleNamespace(eos_token_id=[vocab["<eos>"], vocab["<turn|>"]], temperature=1.0, top_p=0.95, top_k=64)

    def from_model_config(config):
        return SimpleNamespace(eos_token_id=seen["tokenizer"].vocab["<|endoftext|>"], temperature=None, top_p=None, top_k=None, _from_model_config=True)

    monkeypatch.setattr(vllm_backend, "snapshot_download", snapshot_download)
    monkeypatch.setattr(vllm_backend, "AutoConfig", SimpleNamespace(from_pretrained=config))
    monkeypatch.setattr(vllm_backend, "AutoProcessor", SimpleNamespace(from_pretrained=processor))
    monkeypatch.setattr(vllm_backend, "GenerationConfig", SimpleNamespace(from_pretrained=generation, from_model_config=from_model_config))
    monkeypatch.setattr(vllm_backend, "describe_folder", lambda folder, notes: fake_folder("fake", {"config.json": 4954, "tokenizer.json": 32_169_626}, {}))
    return seen


def connect(server: Server, model_id="google/gemma-4-E2B-it") -> VLLMBackend:
    llm = VLLMBackend(model_id, URL, transport=server.transport())
    server.tokenizer = llm.tokenizer
    return llm


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
    with pytest.raises(ValueError, match="text only"):
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
    details = llm._details(b, {"token_logprobs": [math.log(0.4)], "top_logprobs": [{f"token_id:{b}": math.log(0.4), f"token_id:{a}": math.log(0.4)}]}, 0)
    assert [c["id"] for c in details["top"]] == [a, b] and details["rank"] == 1


def test_a_token_the_server_sent_no_log_probability_for_carries_no_details(hub):
    llm = connect(Server([]))
    assert llm._details(5, {}, 0) == {} and llm._details(5, {"token_logprobs": [None], "top_logprobs": [None]}, 0) == {}


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
    assert info["server"] == {"engine": "vLLM", "version": "0.32.0", "url": URL, "served_as": "google/gemma-4-E2B-it"}
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


@real_server
def test_real_server_greedy_tokens_carry_their_probability_and_candidates(served):
    _, tokens, done = generate(served, "用一句話介紹台北。", Sampling(temperature=0, max_new_tokens=24))
    assert tokens and done["n_tokens"] == len(tokens) and "".join(t["text"] for t in tokens) == done["raw"]
    assert all(t["rank"] == 1 and t["top"][0]["id"] == t["id"] and 0 < t["p"] <= 1 for t in tokens)  # greedy takes the top token


@real_server
def test_real_server_runs_the_calculator_and_answers(served):
    events = []
    Run(served, [], "What is 1234 * 5678? Use the calculator.", RunSettings(sampling=Sampling(temperature=0, max_new_tokens=256)), events.append).run()
    calls = [e for e in events if e["type"] == "tool_call"]
    assert calls and calls[0]["name"] == "calculator" and not calls[0]["errors"]
    assert events[-1]["stop_reason"] == "no_tool_call" and "7006652" in events[-1]["final"].replace(",", "")
