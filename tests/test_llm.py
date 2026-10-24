"""The hand-written decoding loop, run against a tiny scripted model and tokenizer."""

from types import SimpleNamespace

import pytest
import torch

import llm
from llm import LocalLLM, Sampling

VOCAB = [
    "<pad>", "<eos>", "<turn|>", "<|tool_response>", "<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>",
    "▁Hi", "!", "call", "<0xE5>", "<0xA5>", "<0xBD>", "▁think",
]
ID = {piece: i for i, piece in enumerate(VOCAB)}
SPECIAL = {i for i, p in enumerate(VOCAB) if p.startswith("<") and not p.startswith("<0x")}


class Tokenizer:
    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return SimpleNamespace(input_ids=torch.tensor([[ID["▁Hi"]] * len(text)]))

    def convert_ids_to_tokens(self, token_id):
        return VOCAB[token_id]

    def decode(self, ids, skip_special_tokens=False):
        data = b""
        for i in ids:
            piece = VOCAB[i]
            if piece.startswith("<0x"):
                data += bytes([int(piece[3:5], 16)])
            else:
                data += piece.replace("▁", " ").encode()
        return data.decode("utf-8", errors="replace")


class Model:
    """Each forward pass puts nearly all probability on the next scripted token."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, input_ids, past_key_values=None, use_cache=True, logits_to_keep=1):
        self.calls.append({"input_ids": input_ids.tolist(), "past": past_key_values})
        logits = torch.zeros(len(VOCAB))
        logits[ID[self.script.pop(0)] if self.script else ID["<eos>"]] = 30.0
        return SimpleNamespace(logits=logits.view(1, 1, -1), past_key_values=f"kv{len(self.calls)}")


def make_llm(script=()):
    m = LocalLLM.__new__(LocalLLM)  # skip __init__: no weights
    m.device = "cpu"
    m.tokenizer = Tokenizer()
    m.model = Model(script)
    m.eos_ids = {ID["<eos>"], ID["<turn|>"], ID["<|tool_response>"]}
    m.special_ids = SPECIAL
    return m


def run(m, prompt="hi", sampling=None, should_stop=lambda: False):
    events = list(m.generate(prompt, sampling or Sampling(temperature=0), should_stop))
    return events[0], events[1:-1], events[-1]


GREEDY = Sampling(temperature=0)


# ---------- generate ----------

def test_events_are_prefill_tokens_done():
    m = make_llm(["▁Hi", "!", "<turn|>"])
    prefill, tokens, done = run(m, prompt="abcd")
    assert prefill["type"] == "prefill" and prefill["n_tokens"] == 4 and prefill["ms"] >= 0
    assert [t["type"] for t in tokens] == ["token"] * 3
    assert [t["piece"] for t in tokens] == ["▁Hi", "!", "<turn|>"]
    assert [t["index"] for t in tokens] == [0, 1, 2]
    assert done["type"] == "done"
    assert done["finish"] == "stop_token" and done["stop_piece"] == "<turn|>"
    assert done["n_tokens"] == 3
    assert done["raw"] == " Hi!<turn|>"


def test_text_deltas_add_up_to_raw():
    m = make_llm(["▁Hi", "!", "▁think", "<eos>"])
    _, tokens, done = run(m)
    assert "".join(t["text"] for t in tokens) == done["raw"]


def test_uses_kv_cache_after_prefill():
    m = make_llm(["▁Hi", "!", "<turn|>"])
    run(m, prompt="abc")
    calls = m.model.calls
    assert len(calls) == 3  # prefill + one pass per token, none after the stop token
    assert calls[0]["past"] is None and len(calls[0]["input_ids"][0]) == 3
    assert calls[1] == {"input_ids": [[ID["▁Hi"]]], "past": "kv1"}
    assert calls[2] == {"input_ids": [[ID["!"]]], "past": "kv2"}


@pytest.mark.parametrize("stop", ["<turn|>", "<|tool_response>", "<eos>"])
def test_every_eos_token_stops(stop):
    _, tokens, done = run(make_llm(["▁Hi", stop, "▁Hi"]))
    assert len(tokens) == 2
    assert done["finish"] == "stop_token" and done["stop_piece"] == stop


def test_max_new_tokens_cuts_off():
    _, tokens, done = run(make_llm(["▁Hi"] * 10), sampling=Sampling(temperature=0, max_new_tokens=4))
    assert len(tokens) == 4
    assert done["finish"] == "max_new_tokens" and done["stop_piece"] is None


def test_should_stop_is_checked_before_each_token():
    count = {"n": 0}

    def should_stop():
        count["n"] += 1
        return count["n"] > 2

    _, tokens, done = run(make_llm(["▁Hi"] * 10), should_stop=should_stop)
    assert len(tokens) == 2
    assert done["finish"] == "user_stopped" and done["n_tokens"] == 2


def test_stopped_before_first_token():
    _, tokens, done = run(make_llm(["▁Hi"]), should_stop=lambda: True)
    assert tokens == []
    assert done["finish"] == "user_stopped" and done["raw"] == "" and done["stop_piece"] is None


def test_channels_follow_the_switch_tokens():
    script = ["▁think", "<channel|>", "▁Hi", "<|tool_call>", "call", "<tool_call|>", "<|tool_response>"]
    _, tokens, _ = run(make_llm(script), prompt="x" + llm.THOUGHT_PREFIX)
    assert [t["channel"] for t in tokens] == ["thought", "control", "content", "control", "tool_call", "control", "control"]
    assert [t["special"] for t in tokens] == [False, True, False, True, False, True, True]


def test_prompt_without_thought_prefix_starts_in_content():
    _, tokens, _ = run(make_llm(["▁Hi", "<turn|>"]))
    assert tokens[0]["channel"] == "content"


def test_characters_split_across_byte_tokens_come_out_whole():
    _, tokens, done = run(make_llm(["<0xE5>", "<0xA5>", "<0xBD>", "<turn|>"]))
    assert [t["text"] for t in tokens] == ["", "", "好", "<turn|>"]
    assert done["raw"] == "好<turn|>"


def test_token_events_carry_sampling_details():
    _, tokens, done = run(make_llm(["▁Hi", "<turn|>"]))
    tok = tokens[0]
    assert tok["id"] == ID["▁Hi"] and tok["rank"] == 1 and tok["p"] > 0.99
    assert len(tok["top"]) == llm.TOP_N
    assert done["tok_per_s"] is None or done["tok_per_s"] > 0


# ---------- _sample ----------

def logits_for(probs):
    return torch.log(torch.tensor(probs, dtype=torch.float32))


PROBS = [0.5, 0.3, 0.12, 0.05, 0.02, 0.01]


def test_greedy_picks_the_argmax():
    m = make_llm()
    chosen, d = m._sample(logits_for([0.1, 0.6, 0.3, 0, 0, 0]), GREEDY)
    assert chosen == 1
    assert d["rank"] == 1 and d["kept"] == 1 and d["p_sampled"] == 1.0
    assert d["p"] == pytest.approx(0.6)
    assert d["logprob"] == pytest.approx(torch.log(torch.tensor(0.6)).item())
    assert [c["kept"] for c in d["top"]] == [True, False, False, False, False]


def test_top_candidates_are_the_model_distribution_sorted():
    _, d = make_llm()._sample(logits_for(PROBS), Sampling(temperature=2.0, top_k=0, top_p=1.0))
    assert [c["id"] for c in d["top"]] == [0, 1, 2, 3, 4]
    assert [c["p"] for c in d["top"]] == pytest.approx(PROBS[:5])  # T does not change the reported p
    assert d["top"][0]["piece"] == VOCAB[0]


def test_top_k_limits_candidates():
    torch.manual_seed(0)
    m = make_llm()
    picks = {m._sample(logits_for(PROBS), Sampling(temperature=1.0, top_k=2, top_p=1.0))[0] for _ in range(200)}
    assert picks <= {0, 1} and len(picks) == 2


def test_top_k_one_is_greedy():
    torch.manual_seed(0)
    chosen, d = make_llm()._sample(logits_for(PROBS), Sampling(temperature=1.5, top_k=1, top_p=1.0))
    assert chosen == 0 and d["kept"] == 1 and d["p_sampled"] == pytest.approx(1.0)


def test_top_p_keeps_the_smallest_prefix_reaching_p():
    # cumulative before each token: 0, .5, .8, ...  -> with p = .7 only the first two stay
    _, d = make_llm()._sample(logits_for(PROBS), Sampling(temperature=1.0, top_k=0, top_p=0.7))
    assert d["kept"] == 2
    assert [c["kept"] for c in d["top"]] == [True, True, False, False, False]


def test_top_p_always_keeps_one_token():
    chosen, d = make_llm()._sample(logits_for(PROBS), Sampling(temperature=1.0, top_k=0, top_p=0.01))
    assert chosen == 0 and d["kept"] == 1


def test_no_limits_keep_the_whole_vocabulary():
    _, d = make_llm()._sample(logits_for(PROBS + [1e-3] * 4), Sampling(temperature=1.0, top_k=0, top_p=1.0))
    assert d["kept"] == 10


def test_sampled_probability_is_renormalized(monkeypatch):
    monkeypatch.setattr(llm.torch, "multinomial", lambda probs, n: torch.tensor([1]))
    chosen, d = make_llm()._sample(logits_for(PROBS), Sampling(temperature=1.0, top_k=0, top_p=0.7))
    assert chosen == 1
    assert d["rank"] == 2
    assert d["p"] == pytest.approx(0.3)
    assert d["p_sampled"] == pytest.approx(0.3 / 0.8)


def test_temperature_reshapes_the_sampling_distribution(monkeypatch):
    monkeypatch.setattr(llm.torch, "multinomial", lambda probs, n: torch.tensor([0]))
    _, cold = make_llm()._sample(logits_for(PROBS), Sampling(temperature=0.5, top_k=0, top_p=1.0))
    _, hot = make_llm()._sample(logits_for(PROBS), Sampling(temperature=2.0, top_k=0, top_p=1.0))
    assert cold["p_sampled"] > 0.5 > hot["p_sampled"]
    assert cold["p"] == hot["p"] == pytest.approx(0.5)


def test_rank_counts_strictly_more_likely_tokens(monkeypatch):
    monkeypatch.setattr(llm.torch, "multinomial", lambda probs, n: torch.tensor([3]))
    chosen, d = make_llm()._sample(logits_for(PROBS), Sampling(temperature=1.0, top_k=0, top_p=1.0))
    assert chosen == 3 and d["rank"] == 4
    assert any(c["id"] == 3 for c in d["top"])


# ---------- loading helpers ----------

def test_load_prefers_local_cache():
    seen = []

    class Cls:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            seen.append(kwargs)
            return "model"

    assert llm._load(Cls, "m", dtype="bf16") == "model"
    assert seen == [{"local_files_only": True, "dtype": "bf16"}]


def test_load_falls_back_to_download():
    seen = []

    class Cls:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            seen.append(kwargs)
            if kwargs.get("local_files_only"):
                raise OSError("not cached")
            return "downloaded"

    assert llm._load(Cls, "m") == "downloaded"
    assert seen == [{"local_files_only": True}, {}]


@pytest.mark.parametrize("cuda, mps, expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_pick_device(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(llm.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(llm.torch.backends.mps, "is_available", lambda: mps)
    assert llm.pick_device() == expected
