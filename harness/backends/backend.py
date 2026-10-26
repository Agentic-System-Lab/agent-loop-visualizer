"""The interface between the harness and whatever runs the model.

This project's harness (harnesses/simple_harness.SimpleHarness) and the page (app/page.py) reach the model only through
a Backend. It renders the conversation into a prompt with the model's chat template, encodes the prompt into the tokens
the model reads, and generates, yielding the events render.py draws. The app runs one of two: simple_backend.SimpleBackend, transformers
on this machine, with a hand-written decoding loop that sees the whole distribution at every token, or
vllm_backend.VLLMBackend, a vLLM server. tests/fakes.FakeLLM replays scripted replies. A backend that asks another
serving engine, such as SGLang or Ollama, would be one more; this module says what it has to provide.

Such an engine tells less about each token than SimpleBackend's own loop does: a few log probabilities, and only with
the flags for them what temperature, top-k and top-p left of the distribution or which tokens an MTP drafter guessed
(vllm_backend.py). So a token event
has the fields every backend sends (TOKEN_FIELDS) and the ones a backend sends only when it knows them
(TOKEN_DETAILS). render.py draws a token with whichever details it has and leaves the rest out, so the page
never shows a number nobody measured.

Nothing here imports torch, so a backend can be written, and the interface read, without it.
"""

from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol, runtime_checkable

from harness import families


@dataclass
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_new_tokens: int = 1024
    mtp: bool = False  # let the MTP drafter guess ahead, once it is loaded (Backend.load_drafter)


@dataclass
class Encoded:
    """A prompt as the model reads it (Backend.encode)."""

    text: str  # the prompt, with each placeholder of an image, clip or video expanded into the tokens it takes
    ids: object  # its token ids, in the form the backend's generate() takes (SimpleBackend: a tensor of shape (1, n_tokens))
    n_tokens: int
    # what generate() needs to read the media: the processor's inputs for the encoders (SimpleBackend), or the files
    # and the server's expansion of the ids (VLLMBackend)
    media: dict = field(default_factory=dict)
    media_tokens: dict = field(default_factory=dict)  # kind -> how many of the tokens are that kind of media


# What every token event carries: its place among the step's tokens, the token's id and piece (its entry in the
# vocabulary), the text it adds to the output, whether it is a special token, and what the model is writing
# there (thought, tool_call, content, or control for a special token).
TOKEN_FIELDS = ("index", "id", "piece", "text", "special", "channel")
# What a token event carries only when the backend knows it:
#   p, logprob  the token's probability in the model's own distribution (temperature 1, nothing cut off), and its log
#   rank        the token's place in that distribution, 1 for the most likely
#   p_sampled   its probability in the distribution it was drawn from, after temperature, top-k and top-p
#   kept        how many candidates top-k and top-p left
#   top         the 5 most likely tokens, each with id, piece, p, and kept (whether top-k and top-p left it) when known
#   draft       with MTP, what the drafter guessed here and whether the model kept it; absent where it guessed nothing.
#               A backend that learns only at the end of a step which tokens were guesses (vllm_backend.py) says so in
#               done's mtp instead (kept_at, turned_down_at), and render.py marks those tokens; it may not know what a
#               turned-down guess was (piece None).
TOKEN_DETAILS = ("p", "logprob", "rank", "p_sampled", "kept", "top", "draft")


@runtime_checkable
class Backend(Protocol):
    """What runs the model. The page holds one at a time (app.page.current_llm) and runs one turn on it at a time."""

    model_id: str
    family: families.Family  # how the model writes its turns, thinking and tool calls (families/)
    modalities: tuple[str, ...]  # what it reads besides text: "image", "audio", "video"
    context_length: int  # the most tokens it reads at once
    info: dict  # what the Model tab shows, and in "defaults" the recommended temperature, top_p and top_k
    drafter_id: str | None  # the MTP drafter that goes with the model (families.DRAFTERS), if any
    drafter: object | None  # that drafter once load_drafter() has loaded it; Sampling.mtp turns it on only then

    def warmup(self) -> None:
        """Pay what the first question would otherwise wait for, such as compiling kernels."""

    def load_drafter(self) -> None:
        """Load the MTP drafter, if the model has one and it is not loaded yet."""

    def render(self, messages: list[dict], tools: list[dict] | None, thinking: bool, generation_prompt: bool = True) -> str:
        """The chat template's string for the messages and tool schemas, which is what the model reads; with
        `generation_prompt` it also opens the model's turn. The harness finds what is new in a step by comparing
        this string with the last step's, so it has to be the prompt itself, not something like it."""

    def encode(self, prompt: str, parts: list[dict]) -> Encoded:
        """The prompt as the model reads it, with each image, clip or video in `parts` (media.parts) expanded into
        the tokens it takes."""

    def generate(self, prompt: Encoded, sampling: Sampling, should_stop: Callable[[], bool]) -> Iterator[dict]:
        """Generate the model's turn, asking should_stop() before every token. It yields, in order:

        - {"type": "prefill", "n_tokens", "ms"}: the prompt has been read, in `ms` milliseconds;
        - {"type": "token", ...} for each token, with TOKEN_FIELDS and the TOKEN_DETAILS the backend knows;
        - {"type": "done", "finish", "stop_piece", "n_tokens", "decode_ms", "tok_per_s", "mtp", "raw"}: `finish`
          is "stop_token", "max_new_tokens" or "user_stopped"; `stop_piece` is the stop token's piece; `mtp` is
          {"drafted", "accepted", "forwards"} with MTP on (and "kept_at", "turned_down_at" from a backend that
          learns the guesses only then), else None; and `raw` is the whole output, special tokens included, which
          the family parses."""
