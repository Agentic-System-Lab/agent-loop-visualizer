"""Local inference with a hand-written decoding loop.

transformers' generate() hides the distribution at each step, so this module runs
the loop itself: one forward pass per token, then temperature, top-k and top-p,
then a sample. Every token comes back with its probability, its rank and the
strongest alternatives, which is what the visualization shows.

Nothing here is specific to one model: which tokens stop a generation and which
ones switch between thinking, tool calls and the reply come from the model's family.

With MTP switched on, a small drafter guesses the next few tokens and one forward pass
checks them all (_drafted); every token still comes from the model's distribution.

Images, sound and video go in through the model's processor: encode() expands each placeholder the
chat template wrote into the tokens of its file, and the prefill hands the encoders their inputs.
After the prefill, decoding is the same as for text.

The models download into MODELS_DIR (the project's models/) at the commits families.REVISIONS pins,
and describe_folder() lists what a model's folder holds for the Model tab.
"""

import inspect
import json
import logging
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor

import families
import media

TOP_N = 5
ROOT = Path(__file__).resolve().parent
# Models download into models/ next to this file, which git ignores, rather than into the shared
# ~/.cache/huggingface/hub, so they are easy to find and leave with the project. HF_HUB_CACHE puts them elsewhere.
MODELS_DIR = Path(os.environ.get("HF_HUB_CACHE") or ROOT / "models").expanduser()


@dataclass
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_new_tokens: int = 1024
    mtp: bool = False  # let the MTP drafter guess ahead, once it is loaded (LocalLLM.load_drafter)


@dataclass
class Encoded:
    """A prompt as the model reads it (LocalLLM.encode)."""

    text: str  # the prompt, with each placeholder of an image, clip or video expanded into the tokens it takes
    ids: object  # its token ids, a tensor of shape (1, n_tokens)
    n_tokens: int
    media: dict = field(default_factory=dict)  # the processor's inputs for the encoders (pixels, audio features), for the prefill
    media_tokens: dict = field(default_factory=dict)  # kind -> how many of the tokens are that kind of media


def _download(model_id: str, only: str | None = None) -> str:
    """The model's folder in MODELS_DIR, after fetching the files it lacks (only `only`, if given), at the revision
    families.REVISIONS pins it to.

    from_pretrained(local_files_only=True) is no test of the cache: once config.json is cached, a tokenizer whose
    own files are missing raises a ValueError (Gemma 4) or loads empty (Qwen3.5) rather than an OSError.
    snapshot_download lists the model's files on the Hub and fetches only those the cache lacks; when the Hub
    can't be reached it returns the cached folder, so a downloaded model still starts offline."""
    if os.path.isdir(model_id):  # a checkpoint on disk rather than a Hugging Face id
        return model_id
    return snapshot_download(model_id, revision=families.REVISIONS.get(model_id), allow_patterns=only, cache_dir=MODELS_DIR)


def describe_folder(folder: str, notes: dict) -> dict:
    """A checkpoint's folder as the Model tab lists it: where it is, the Hub revision it was downloaded at, and every
    file with its size and the values of the fields its family's note on it names (families.FileNote). In the cache a
    file is a link into blobs/, and the size is that of the file it links to."""
    path = Path(folder).absolute()
    files = []
    for file in sorted(f for f in path.rglob("*") if f.is_file()):
        name = file.relative_to(path).as_posix()
        if name.startswith(".cache/"):  # what `hf download --local-dir` keeps about the download, not part of the checkpoint
            continue
        note = notes.get(name)
        files.append({"name": name, "bytes": file.stat().st_size, "fields": _fields(file, note.fields) if note else {}})
    return {"path": _shown(path), "revision": path.name if path.parent.name == "snapshots" else None, "files": files}


def _fields(file: Path, fields: tuple[str, ...]) -> dict:
    """The values at dotted paths (text_config.sliding_window) in a JSON file; a path the file lacks is left out."""
    if not fields:
        return {}
    data, out = json.loads(file.read_text(encoding="utf-8")), {}
    for dotted in fields:
        value = data
        for key in dotted.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            out[dotted] = value
    return out


def _shown(path: Path) -> str:
    """A path as the page shows it: from the project's folder when it is inside, else from the home folder."""
    for base, prefix in ((ROOT, ""), (Path.home(), "~/")):
        if path.is_relative_to(base):
            return prefix + path.relative_to(base).as_posix()
    return path.as_posix()


def _not_a_kernel_notice(record: logging.LogRecord) -> bool:
    """transformers notes, once per function, that Qwen3.5's Gated DeltaNet layers run in plain PyTorch because
    flash-linear-attention and causal-conv1d are missing. Neither is a dependency, so the notice is dropped on every
    device. Both need a CUDA GPU: on MPS or CPU there is nothing to install, and fla installed on a Linux machine
    without a GPU still gets picked and fails in the first forward pass. With a GPU the gain is small for a 0.8B
    model (fla speeds up prefill only) and causal-conv1d has to be compiled; the README has the measurements."""
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
        logging.getLogger("transformers.integrations.hub_kernels").addFilter(_not_a_kernel_notice)
        if self.device == "mps":
            # Loading weights onto MPS from several threads can hang (seen with Qwen3.5); one thread is fast enough.
            os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        self.drafter_id = families.DRAFTERS.get(model_id)
        self.drafter = None  # loaded the first time MTP is switched on
        self._drafter_lock = threading.Lock()
        start = time.perf_counter()
        # config.json alone says which family the model belongs to; an unsupported one fails before the weights download.
        self.family = families.for_model_type(AutoConfig.from_pretrained(_download(model_id, "config.json")).model_type)
        files = self.folder = _download(model_id)
        # The processor holds the tokenizer and, for a model that reads more than text, what turns files into its inputs.
        self.processor = AutoProcessor.from_pretrained(files)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.model = AutoModelForMultimodalLM.from_pretrained(files, dtype=self.dtype, device_map=self.device).eval()
        self.modalities = self._modalities()
        self.media_token_ids = {kind: getattr(self.model.config, f"{kind}_token_id", None) for kind in self.modalities}
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
            "modalities": list(self.modalities),
            "family": self.family.key,
            "stop_tokens": [self.piece(i) for i in sorted(self.eos_ids)],
            "special_tokens": {self.piece(i): i for i in sorted(self.special_ids)},
            "defaults": self._sampling_defaults(),
            "drafter_id": self.drafter_id,
            "drafter": None,
            "folder": describe_folder(self.folder, self.family.file_notes),
        }

    def _modalities(self) -> tuple[str, ...]:
        """What the checkpoint reads besides text: each kind needs an encoder in the config and a part of the processor
        to feed it. It is asked of the checkpoint, not the family, since checkpoints of one family can differ."""
        config, processor = self.model.config, self.processor
        reads = {
            "image": getattr(config, "vision_config", None) is not None and hasattr(processor, "image_processor"),
            "audio": getattr(config, "audio_config", None) is not None and hasattr(processor, "feature_extractor"),
            "video": getattr(config, "video_token_id", None) is not None and hasattr(processor, "video_processor"),
        }
        return tuple(kind for kind, ok in reads.items() if ok)

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
        prompt = self.encode(self.render([{"role": "user", "content": "Hi"}], None, False), [])
        for _ in self.generate(prompt, Sampling(max_new_tokens=16 if mtp else 2, mtp=mtp), lambda: False):
            pass

    def load_drafter(self):
        """Load the MTP drafter that goes with this model (families.DRAFTERS), if any, and warm it up."""
        with self._drafter_lock:
            if self.drafter is not None or self.drafter_id is None:
                return
            start = time.perf_counter()
            folder = _download(self.drafter_id)
            self.drafter = AutoModelForCausalLM.from_pretrained(folder, dtype=self.dtype, device_map=self.device).eval()
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
                "folder": describe_folder(folder, self.family.drafter_file_notes),
            }

    def piece(self, token_id: int) -> str:
        return self.tokenizer.convert_ids_to_tokens(token_id)

    def render(self, messages: list[dict], tools: list[dict] | None, thinking: bool, generation_prompt: bool = True) -> str:
        """The chat template turns messages and tool schemas into the one string the model reads.
        With `generation_prompt` it also opens the model's turn, so the model writes next."""
        return self.tokenizer.apply_chat_template(
            messages, tools=tools or None, tokenize=False, add_generation_prompt=generation_prompt, enable_thinking=thinking
        )

    def encode(self, prompt: str, parts: list[dict]) -> Encoded:
        """The prompt as the model reads it. Without media that is the prompt's tokens. With media (`parts`, the
        conversation's media in the order of their placeholders), the processor replaces each placeholder with the
        tokens that image, clip or video takes, and makes the inputs of the encoders whose output goes there."""
        if not parts:
            ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
            return Encoded(prompt, ids, ids.shape[-1])
        files = {kind: [p["path"] for p in parts if p["type"] == kind] for kind in media.MEDIA}
        inputs = {}
        if files["image"]:
            inputs["images"] = [media.load_image(path) for path in files["image"]]
        if files["audio"]:
            inputs["audio"] = [media.load_audio(path, self.processor.feature_extractor.sampling_rate) for path in files["audio"]]
        if files["video"]:
            videos = [media.load_video(path) for path in files["video"]]
            inputs["videos"] = [frames for frames, _ in videos]
            inputs["videos_kwargs"] = {"video_metadata": [metadata for _, metadata in videos]}
            # A processor that takes a set number of frames (Gemma 4: 32) refuses a video with fewer; it gets all there are.
            wanted, fewest = getattr(self.processor.video_processor, "num_frames", None), min(len(frames) for frames, _ in videos)
            if wanted and fewest < wanted:
                inputs["videos_kwargs"]["num_frames"] = fewest
        out = self.processor(text=prompt, return_tensors="pt", **inputs, **self.family.processor_kwargs)
        ids = out.pop("input_ids")
        out.pop("attention_mask", None)  # one sequence, no padding: every position counts, as with text
        counts = {kind: int((ids == token).sum()) for kind, token in self.media_token_ids.items() if token is not None}
        text = self.tokenizer.decode(ids[0], skip_special_tokens=False)
        return Encoded(text, ids, ids.shape[-1], dict(out.to(self.device, dtype=self.dtype)), {k: n for k, n in counts.items() if n})

    def generate(self, prompt: Encoded, sampling: Sampling, should_stop: Callable[[], bool]) -> Iterator[dict]:
        """Yield a prefill event, one event per token, then a done event."""
        ids = prompt.ids.to(self.device)
        channel = "thought" if prompt.text.endswith(self.family.thought_prefix) else "content"
        drafting = sampling.mtp and self.drafter is not None
        mtp = {"drafted": 0, "accepted": 0, "forwards": 0} if drafting else None
        generated: list[int] = []
        text = ""
        finish = "max_new_tokens"

        with torch.inference_mode():
            # Models with multimodal RoPE (Qwen3.5) keep the position offset of the last prompt's images on the model
            # and add it in every later forward pass; each prompt starts from its own.
            if hasattr(self.model.base_model, "rope_deltas"):
                self.model.base_model.rope_deltas = None
            start = time.perf_counter()
            out = self._forward(ids, None, 1, drafting, **self._encoded_media(prompt.media))
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

    def _forward(self, ids: torch.Tensor, past, keep: int, drafting: bool = False, **media):
        """One forward pass over `ids`, with the cache of everything before them; logits for the last `keep` positions.
        The drafter also reads the model's last hidden state and the keys and values its later layers share.
        In the prefill, `media` are the encoders' inputs, whose outputs replace the embeddings of the media's tokens."""
        extra = {"output_hidden_states": True, "return_shared_kv_states": True} if drafting else {}
        return self.model(input_ids=ids, past_key_values=past, use_cache=True, logits_to_keep=keep, **extra, **media)

    def _encoded_media(self, inputs: dict) -> dict:
        """Run the vision encoder over the prompt's images and videos before the prefill, as generate() does, and pass
        its output in their place. Leaving it to the forward pass breaks videos on Gemma 4: transformers 5.20's
        Gemma4ForConditionalGeneration takes num_frames_per_video but does not hand it on to the video encoder."""
        if not inputs or "mm_encoder_outputs" not in inspect.signature(self.model.forward).parameters:
            return inputs
        base, encoded = self.model.base_model, {}
        for kind in ("image", "video"):
            encoder = getattr(base, f"get_{kind}_features", None)
            args = {k: inputs[k] for k in inspect.signature(encoder).parameters if k in inputs} if encoder else {}
            if any(k.startswith("pixel_values") for k in args):
                encoded[kind] = encoder(**args, return_dict=True)
        if not encoded:
            return inputs
        # The pixels go; the rest stays, such as the grid sizes Qwen3.5's positions are computed from.
        return {**{k: v for k, v in inputs.items() if not k.startswith("pixel_values")}, "mm_encoder_outputs": encoded}

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
        # Equally likely tokens go lowest id first, the one argmax and so generate() takes. topk leaves their order open
        # (on the CPU it can list the higher id first), and in bfloat16 two logits are equal more often than one would think.
        top = sorted(zip(*probs.topk(TOP_N)), key=lambda c: (-float(c[0]), int(c[1])))

        if s.temperature <= 0:  # T -> 0 is greedy decoding
            chosen = int(top[0][1])
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
                for rank, (v, i) in enumerate(top)
            ],
        }
