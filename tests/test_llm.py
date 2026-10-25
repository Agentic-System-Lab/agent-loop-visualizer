"""The hand-written decoding loop, run against a tiny scripted model and tokenizer."""

import logging
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

import llm
from families import GEMMA4
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

    base_model = None  # no multimodal RoPE to reset

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
    m.family = GEMMA4
    m.eos_ids = {ID["<eos>"], ID["<turn|>"], ID["<|tool_response>"]}
    m.special_ids = SPECIAL
    return m


def run(m, prompt="hi", sampling=None, should_stop=lambda: False):
    events = list(m.generate(m.encode(prompt, []), sampling or Sampling(temperature=0), should_stop))
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
    _, tokens, _ = run(make_llm(script), prompt="x" + GEMMA4.thought_prefix)
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


# ---------- MTP ----------

class Cache:
    """Only the length matters: how many positions the model has read, and what was cropped."""

    def __init__(self):
        self.length, self.recording, self.crops = 0, False, []

    def activate_past_recording(self):
        self.recording = True

    def get_seq_length(self):
        return self.length

    def crop(self, n):
        self.crops.append(n)
        self.length += n


class Target:
    """At every position the model is sure of the next token of `script` (generated tokens start after the prompt).
    Hidden states and shared keys carry the position they belong to, so tests can see what the drafter was given."""

    base_model = None

    def __init__(self, script, prompt_len):
        self.script, self.prompt_len, self.calls = list(script), prompt_len, []

    def next_piece(self, position):
        i = position + 1 - self.prompt_len
        return self.script[i] if i < len(self.script) else "<eos>"

    def __call__(self, input_ids, past_key_values=None, use_cache=True, logits_to_keep=1, **extra):
        cache = self.cache = past_key_values or Cache()
        start, n = cache.length, input_ids.shape[-1]
        cache.length += n
        self.calls.append({"input_ids": input_ids.tolist(), "keep": logits_to_keep, **extra})
        logits = torch.zeros(n, len(VOCAB))
        for i in range(n):
            logits[i, ID[self.next_piece(start + i)]] = 30.0
        positions = torch.arange(start, start + n, dtype=torch.float32)
        keys = torch.arange(cache.length, dtype=torch.float32).view(1, 1, -1, 1)
        return SimpleNamespace(
            logits=logits[-logits_to_keep:].unsqueeze(0), past_key_values=cache,
            hidden_states=(positions.view(1, n, 1),), shared_kv_states={"full_attention": (keys, keys)},
        )

    def get_input_embeddings(self):
        return lambda ids: ids.float().view(1, 1, 1)


class Drafter:
    """Guesses the scripted tokens, except the wrong guesses given for some of them (by generated index)."""

    def __init__(self, target, wrong=None):
        self.target, self.wrong, self.calls = target, wrong or {}, []

    def __call__(self, inputs_embeds, position_ids, shared_kv_states):
        position = int(position_ids)
        nth = sum(1 for c in self.calls if c["position"] == position)  # guesses already made from this position
        self.calls.append({
            "position": position, "token": int(inputs_embeds[0, 0, 0]), "hidden": float(inputs_embeds[0, 0, 1]),
            "last_key": int(shared_kv_states["full_attention"][0][0, 0, -1, 0]),
        })
        index = position + 1 + nth - self.target.prompt_len
        piece = self.wrong.get(index) or self.target.next_piece(position + nth)
        logits = torch.zeros(len(VOCAB))
        logits[ID[piece]] = 1.0
        return SimpleNamespace(logits=logits.view(1, 1, -1), last_hidden_state=torch.full((1, 1, 1), -1.0))


def make_mtp_llm(script, prompt="abc", draft_tokens=3, wrong=None):
    m = make_llm()
    m.model = Target(script, len(prompt))
    m.drafter = Drafter(m.model, wrong)
    m.draft_tokens = draft_tokens
    return m


MTP = Sampling(temperature=0, mtp=True)
SCRIPT = ["▁Hi", "!", "▁think", "!", "▁Hi", "<turn|>"]


def marks(tokens):
    return [("kept" if t["draft"]["accepted"] else "turned down") if "draft" in t else None for t in tokens]


def test_mtp_checks_the_guesses_and_replaces_the_first_one_turned_down():
    m = make_mtp_llm(SCRIPT, wrong={2: "call"})
    _, tokens, done = run(m, prompt="abc", sampling=MTP)
    assert [t["piece"] for t in tokens] == SCRIPT  # the same tokens as decoding one at a time
    assert marks(tokens) == [None, "kept", "turned down", "kept", "kept", "kept"]
    assert tokens[2]["draft"]["piece"] == "call"
    assert done["mtp"] == {"drafted": 6, "accepted": 4, "forwards": 2}
    assert done["finish"] == "stop_token" and done["raw"] == " Hi! think! Hi<turn|>"

    calls = m.model.calls
    assert len(calls) == 3  # prefill and two checks
    assert calls[1]["input_ids"] == [[ID[p] for p in ["▁Hi", "!", "call", "!"]]] and calls[1]["keep"] == 4
    assert all(c["output_hidden_states"] and c["return_shared_kv_states"] for c in calls)
    # The two guesses after the one turned down leave the cache, and the drafter never sees their keys.
    assert m.model.cache.recording and m.model.cache.crops == [-2]
    assert [c["last_key"] for c in m.drafter.calls] == [c["position"] - 1 for c in m.drafter.calls]
    # The second round drafts from the replacement, with the hidden state of the last guess kept.
    second = next(c for c in m.drafter.calls if c["position"] == 5)
    assert second["token"] == ID["▁think"] and second["hidden"] == 4.0


def test_mtp_takes_one_more_token_when_every_guess_holds():
    m = make_mtp_llm(SCRIPT, draft_tokens=2)
    _, tokens, done = run(m, prompt="abc", sampling=MTP)
    assert [t["piece"] for t in tokens] == SCRIPT
    assert marks(tokens) == [None, "kept", "kept", None, "kept", "kept"]  # the fourth came free with the second guess
    assert done["mtp"] == {"drafted": 4, "accepted": 4, "forwards": 2}


def test_mtp_rounds_over_a_longer_reply():
    script = ["▁Hi", "!", "▁think", "!", "▁Hi", "!", "▁think", "<turn|>"]
    m = make_mtp_llm(script, wrong={2: "call"})
    _, tokens, done = run(m, prompt="abc", sampling=MTP)
    assert [t["piece"] for t in tokens] == script
    assert marks(tokens) == [None, "kept", "turned down", "kept", "kept", "kept", None, "kept"]
    assert m.model.cache.crops == [-2, 0]  # nothing to drop when every guess held
    assert [c["position"] for c in m.drafter.calls] == [3, 3, 3, 5, 5, 5, 9]  # drafting stops at a stop token
    assert done["mtp"] == {"drafted": 7, "accepted": 5, "forwards": 3}


def test_mtp_stops_at_the_token_limit():
    m = make_mtp_llm(["▁Hi", "!"] * 10)
    _, tokens, done = run(m, prompt="abc", sampling=Sampling(temperature=0, mtp=True, max_new_tokens=5))
    assert len(tokens) == 5 and done["finish"] == "max_new_tokens"


def test_mtp_is_off_without_a_drafter():
    m = make_llm(["▁Hi", "<turn|>"])
    m.drafter = None
    _, tokens, done = run(m, sampling=MTP)
    assert done["mtp"] is None and all("draft" not in t for t in tokens)


def test_greedy_keeps_a_guess_only_when_it_is_the_top_token():
    m = make_llm()
    assert m._sample(logits_for(PROBS), GREEDY, draft=0)[0] == 0
    assert m._sample(logits_for(PROBS), GREEDY, draft=1)[0] == 0


def test_a_guess_turned_down_is_never_drawn_again(monkeypatch):
    monkeypatch.setattr(llm.torch, "rand", lambda *a: torch.tensor(0.99))  # above p(guess): turned down
    seen = {}

    def multinomial(weights, n):
        seen["weights"] = weights.tolist()
        return torch.tensor([0])

    monkeypatch.setattr(llm.torch, "multinomial", multinomial)
    chosen, d = make_llm()._sample(logits_for(PROBS), Sampling(temperature=1.0, top_k=0, top_p=1.0), draft=1)
    assert seen["weights"][1] == 0 and seen["weights"][0] == pytest.approx(0.5)
    assert chosen == 0 and d["p_sampled"] == pytest.approx(0.5)  # its probability before the guess was removed


def test_a_guess_cut_by_top_k_is_never_kept():
    torch.manual_seed(0)
    m = make_llm()
    picks = {m._sample(logits_for(PROBS), Sampling(temperature=1.0, top_k=2, top_p=1.0), draft=3)[0] for _ in range(200)}
    assert picks == {0, 1}


@pytest.mark.parametrize("draft", [0, 1, 4])
def test_checking_a_guess_leaves_the_distribution_unchanged(draft):
    """Speculative sampling is exact: whatever the drafter guessed, each token comes out with its own probability."""
    torch.manual_seed(1)
    m = make_llm()
    s = Sampling(temperature=1.0, top_k=0, top_p=1.0)
    n = 20000
    counts = Counter(m._sample(logits_for(PROBS), s, draft=draft)[0] for _ in range(n))
    for token, p in enumerate(PROBS):
        assert counts[token] / n == pytest.approx(p, abs=0.012), token


# ---------- media ----------

class Processor:
    """Expands a placeholder ("<|tool_call>" here, standing in for an image's) into three of its tokens, as a real
    processor expands an image into its tokens, and returns made-up inputs for the encoders."""

    feature_extractor = SimpleNamespace(sampling_rate=16000)

    def __init__(self, num_frames=None):
        self.tokenizer, self.calls = Tokenizer(), []
        self.video_processor = SimpleNamespace(num_frames=num_frames)  # Gemma 4's takes 32 frames, Qwen3.5's a rate

    def __call__(self, text, return_tensors, **kwargs):
        from transformers import BatchFeature

        self.calls.append(kwargs)
        ids = [ID["▁Hi"]] + [ID["<|tool_call>"]] * 3 * text.count("<|tool_call>") + [ID["!"]]
        return BatchFeature({"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids)),
                             "pixel_values": torch.ones(2, 2), "image_grid_thw": torch.tensor([[1, 2, 2]])})


def media_llm(monkeypatch):
    m = make_llm()
    m.processor, m.dtype = Processor(), torch.bfloat16
    m.media_token_ids = {"image": ID["<|tool_call>"], "audio": ID["<|tool_response>"]}
    for kind in ("image", "audio", "video"):
        monkeypatch.setattr(llm.media, f"load_{kind}", lambda path, *rest, kind=kind: (f"{kind}:{path}", {"fps": 2.0}) if kind == "video" else f"{kind}:{path}")
    return m


def test_encode_hands_the_processor_each_kind_of_media_in_order(monkeypatch):
    m = media_llm(monkeypatch)
    m.family = llm.families.QWEN3_5
    parts = [{"type": "image", "path": "a.png"}, {"type": "video", "path": "v.mp4"}, {"type": "image", "path": "b.png"}]
    encoded = m.encode("look <|tool_call>", parts)
    [call] = m.processor.calls
    assert call["images"] == ["image:a.png", "image:b.png"]
    assert call["videos"] == ["video:v.mp4"] and call["videos_kwargs"] == {"video_metadata": [{"fps": 2.0}]}
    assert "audio" not in call
    assert call["cap_pixels_per_frame"] is True  # the family's own processor settings
    assert encoded.n_tokens == 5 and encoded.media_tokens == {"image": 3}
    assert encoded.text == " Hi" + "<|tool_call>" * 3 + "!"  # what the model reads: the placeholder expanded
    assert set(encoded.media) == {"pixel_values", "image_grid_thw"}  # no ids, no attention mask
    assert encoded.media["pixel_values"].dtype == torch.bfloat16 and encoded.media["image_grid_thw"].dtype == torch.int64


def test_a_video_shorter_than_the_frames_a_processor_takes_gives_it_all_it_has(monkeypatch):
    m = media_llm(monkeypatch)
    m.processor = Processor(num_frames=32)
    monkeypatch.setattr(llm.media, "load_video", lambda path: (torch.zeros(30 if "short" in path else 64, 2, 2, 3), {"fps": 10.0}))
    m.encode("x", [{"type": "video", "path": "short.mp4"}, {"type": "video", "path": "long.mp4"}])
    assert m.processor.calls[0]["videos_kwargs"]["num_frames"] == 30
    m.encode("x", [{"type": "video", "path": "long.mp4"}])
    assert "num_frames" not in m.processor.calls[1]["videos_kwargs"]  # enough frames: the processor's own number


def test_encode_without_media_is_the_prompt_tokens(monkeypatch):
    m = media_llm(monkeypatch)
    encoded = m.encode("abc", [])
    assert m.processor.calls == [] and encoded.text == "abc" and encoded.n_tokens == 3 and encoded.media == {}


def test_images_go_through_the_vision_encoder_before_the_prefill():
    seen = {}

    class Base:
        rope_deltas = "stale"

        def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
            return ("image features", pixel_values, image_grid_thw)

    class Multimodal(Model):
        base_model = Base()

        def forward(self, input_ids, mm_encoder_outputs=None, **kwargs):
            pass

        def __call__(self, input_ids, past_key_values=None, use_cache=True, logits_to_keep=1, **media):
            if past_key_values is None:
                seen.update(media, rope_deltas=self.base_model.rope_deltas)
            return super().__call__(input_ids, past_key_values, use_cache, logits_to_keep)

    m = make_llm()
    m.model = Multimodal(["▁Hi", "<turn|>"])
    prompt = llm.Encoded("x", torch.tensor([[ID["▁Hi"]]]), 1, {"pixel_values": "pixels", "image_grid_thw": "grid", "mm_token_type_ids": "types"})
    list(m.generate(prompt, GREEDY, lambda: False))
    assert seen["mm_encoder_outputs"] == {"image": ("image features", "pixels", "grid")}
    assert "pixel_values" not in seen  # the forward pass gets the encoder's output instead of the pixels
    assert seen["image_grid_thw"] == "grid" and seen["mm_token_type_ids"] == "types"  # Qwen3.5 computes positions from these
    assert seen["rope_deltas"] is None  # the last prompt's image positions do not carry over


# ---------- loading helpers ----------

class Loaded(Exception):
    """The fake weights end LocalLLM.__init__: everything after them needs a real model."""


def fake_hub(monkeypatch, model_type="gemma4"):
    """Records each download and each from_pretrained, in order, and where it loaded from."""
    seen = []

    def snapshot_download(model_id, allow_patterns=None):
        seen.append(("download", model_id, allow_patterns))
        return f"/hub/{model_id}"

    def loader(name, result):
        def from_pretrained(path, **kwargs):
            seen.append((name, path, kwargs))
            if result is Loaded:
                raise Loaded
            return result
        return SimpleNamespace(from_pretrained=from_pretrained)

    monkeypatch.setattr(llm, "snapshot_download", snapshot_download)
    monkeypatch.setattr(llm, "AutoConfig", loader("config", SimpleNamespace(model_type=model_type)))
    monkeypatch.setattr(llm, "AutoProcessor", loader("processor", SimpleNamespace(tokenizer="tokenizer")))
    monkeypatch.setattr(llm, "AutoModelForMultimodalLM", loader("model", Loaded))
    monkeypatch.setattr(llm, "AutoModelForCausalLM", loader("drafter", Loaded))
    monkeypatch.setattr(llm, "pick_device", lambda: "cuda")
    return seen


def test_the_processor_and_weights_load_from_the_whole_downloaded_model(monkeypatch):
    # A cache holding only config.json used to load Gemma's tokenizer locally and fail, and Qwen's empty.
    seen = fake_hub(monkeypatch)
    with pytest.raises(Loaded):
        LocalLLM("google/gemma-4-E2B-it")
    assert seen == [
        ("download", "google/gemma-4-E2B-it", "config.json"),
        ("config", "/hub/google/gemma-4-E2B-it", {}),
        ("download", "google/gemma-4-E2B-it", None),
        ("processor", "/hub/google/gemma-4-E2B-it", {}),
        ("model", "/hub/google/gemma-4-E2B-it", {"dtype": torch.bfloat16, "device_map": "cuda"}),
    ]


def test_an_unsupported_model_fails_before_its_weights_download(monkeypatch):
    seen = fake_hub(monkeypatch, model_type="llama")
    with pytest.raises(ValueError, match="'llama' are not supported"):
        LocalLLM("meta/llama")
    assert seen == [("download", "meta/llama", "config.json"), ("config", "/hub/meta/llama", {})]


def test_a_checkpoint_on_disk_loads_in_place(monkeypatch, tmp_path):
    seen = fake_hub(monkeypatch)
    with pytest.raises(Loaded):
        LocalLLM(str(tmp_path))
    assert [s for s in seen if s[0] == "download"] == []
    assert {path for _, path, _ in seen} == {str(tmp_path)}


def test_the_drafter_loads_from_its_downloaded_files(monkeypatch):
    seen = fake_hub(monkeypatch)
    m = make_llm()
    m.dtype, m.device = torch.bfloat16, "cuda"
    m.drafter, m.drafter_id = None, "google/gemma-4-E2B-it-assistant"
    m._drafter_lock = llm.threading.Lock()
    with pytest.raises(Loaded):
        m.load_drafter()
    assert seen == [
        ("download", "google/gemma-4-E2B-it-assistant", None),
        ("drafter", "/hub/google/gemma-4-E2B-it-assistant", {"dtype": torch.bfloat16, "device_map": "cuda"}),
    ]


@pytest.mark.parametrize("cuda, mps, expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_pick_device(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(llm.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(llm.torch.backends.mps, "is_available", lambda: mps)
    assert llm.pick_device() == expected


def test_the_kernel_notice_is_dropped_and_nothing_else():
    def record(msg):
        return logging.LogRecord("transformers.integrations.hub_kernels", logging.WARNING, __file__, 1, msg, None, None)

    notice = ("`chunk_gated_delta_rule` is falling back to its reference PyTorch implementation because "
              "`flash-linear-attention` is not installed. This is correct but much slower; install it for the optimized kernel.")
    assert not llm._not_a_kernel_notice(record(notice))
    assert llm._not_a_kernel_notice(record("Could not load the kernel from the hub"))


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_the_kernel_notice_is_dropped_on_every_device(monkeypatch, device):
    logger = logging.getLogger("transformers.integrations.hub_kernels")
    monkeypatch.setattr(logger, "filters", [])
    monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)  # set on MPS
    fake_hub(monkeypatch)
    monkeypatch.setattr(llm, "pick_device", lambda: device)
    with pytest.raises(Loaded):
        LocalLLM("google/gemma-4-E2B-it")
    assert llm._not_a_kernel_notice in logger.filters
