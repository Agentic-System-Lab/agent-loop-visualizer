"""Attachments: the files a user adds to a question.

Each file is saved in the conversation's folder, which the run_python tool's sandbox sees, read-only,
as /data (tools.py). An image, a sound clip or a video that the model reads goes into the user's message
as a part of its own; the chat template writes a placeholder token there, and LocalLLM.encode() (llm.py)
has the model's processor replace it with the tokens the file takes, from what load() decodes here. Any
other file reaches the model only as a line in the message saying where run_python finds it.

Which kinds a model reads comes from the checkpoint (LocalLLM.modalities); this module only knows the
file extensions of each kind.
"""

import functools
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageOps

# File extension -> the kind of media it holds. A file whose kind the model does not read is a "file".
KINDS = {
    **dict.fromkeys(("png", "jpg", "jpeg", "webp", "gif", "bmp"), "image"),
    **dict.fromkeys(("wav", "mp3", "m4a", "flac", "ogg"), "audio"),
    **dict.fromkeys(("mp4", "mov", "webm", "mkv"), "video"),
}
DATA_TYPES = ("csv", "tsv", "json", "txt", "md", "xlsx")  # files only run_python reads
PREVIEW_CHARS = 300  # of a text file's first line, shown to the model so its code can use the right columns
MEDIA = ("image", "audio", "video")
MAX_FRAMES = 64  # frames decoded from a video, evenly spaced; the processor samples its own number from them
MAX_SIDE = 768  # pixels on a decoded frame's longer side; the processors shrink frames further anyway


@dataclass(frozen=True)
class Attachment:
    name: str  # its file name in the conversation's folder, and so under /data in the sandbox
    kind: str  # "image", "audio" or "video" when the model reads it, else "file"
    path: str  # where it is saved
    size: int  # in bytes

    def part(self) -> dict:
        """The part of the user's message the chat template writes a placeholder for."""
        return {"type": self.kind, **asdict(self)}


def kind_of(name: str, modalities: tuple[str, ...]) -> str:
    kind = KINDS.get(Path(name).suffix.lower().lstrip("."), "file")
    return kind if kind in modalities else "file"


def accepted(modalities: tuple[str, ...], python: bool) -> list[str]:
    """The file extensions the chat input takes: the media the model reads, and data files when run_python can open them."""
    return [ext for ext, kind in KINDS.items() if kind in modalities] + (list(DATA_TYPES) if python else [])


def save(folder: Path, name: str, data: bytes, modalities: tuple[str, ...]) -> Attachment:
    """Write an uploaded file into the conversation's folder, under a name no other file there has."""
    folder.mkdir(parents=True, exist_ok=True)
    stem, suffix = Path(name).stem, Path(name).suffix
    # The model types the name into its code, so keep it to plain characters.
    stem = re.sub(r"[^\w.-]+", "_", stem).strip("._") or "file"
    path, n = folder / f"{stem}{suffix}", 1
    while path.exists():
        n += 1
        path = folder / f"{stem}-{n}{suffix}"
    path.write_bytes(data)
    return Attachment(path.name, kind_of(path.name, modalities), str(path), len(data))


def fmt_size(n: int) -> str:
    if n < 1000:
        return f"{n} B"
    return f"{n / 1000:.1f} KB" if n < 1_000_000 else f"{n / 1_000_000:.1f} MB"


def preview(path: str) -> str:
    """The first line of a text file such as a CSV (its column names), or "" for other files. Without it a small
    model guesses the columns in its code (a "Date", a "month_revenue") and fails; given a few rows of data as well,
    Gemma copies them into its code and answers from those alone instead of reading the file."""
    if Path(path).suffix.lower().lstrip(".") not in ("csv", "tsv", "txt", "md"):  # a JSON file's first line is a "["
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.readline()[:PREVIEW_CHARS].rstrip("\n")


def user_content(text: str, attachments: list[Attachment]) -> str | list[dict]:
    """The user's message: the text alone, or the media first (as their models are trained to read them), then the
    text, and a note on the files only run_python can open: where they are and how a text file begins. The note is
    for the model, so it is in English."""
    if not attachments:
        return text
    files = [a for a in attachments if a.kind == "file"]
    described = [f"/data/{a.name} ({fmt_size(a.size)})" + (f", whose first line is: {head}" if (head := preview(a.path)) else "") for a in files]
    note = "Attached for run_python, which reads them:\n" + "\n\n".join(described) if files else ""
    words = "\n\n".join(t for t in (text, note) if t)
    return [a.part() for a in attachments if a.kind != "file"] + ([{"type": "text", "text": words}] if words else [])


def parts(messages: list[dict]) -> list[dict]:
    """The media in a conversation, in the order the chat template writes their placeholders."""
    return [p for m in messages if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") in MEDIA]


# ---------- decoding, for the processor ----------
# Each file is decoded once (the cache is keyed by path) and then read again in every step of every turn.

@functools.lru_cache(maxsize=16)
def load_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")  # a phone photo stores its rotation in EXIF


@functools.lru_cache(maxsize=16)
def load_audio(path: str, rate: int) -> np.ndarray:
    """The clip as mono float samples at `rate` Hz, the rate the model's feature extractor expects."""
    with av.open(path) as container:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        chunks = [r.to_ndarray() for frame in container.decode(audio=0) for r in resampler.resample(frame)]
        chunks += [r.to_ndarray() for r in resampler.resample(None)]
    return np.concatenate(chunks, axis=1)[0] if chunks else np.zeros(0, dtype=np.float32)


@functools.lru_cache(maxsize=4)
def load_video(path: str) -> tuple[np.ndarray, dict]:
    """Up to MAX_FRAMES frames spread evenly over the video, shrunk to MAX_SIDE, and the metadata the processor
    reads their times from. The processor picks its own frames from these (Gemma 4 takes 32, Qwen3.5 two a second)."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 24)
        total = stream.frames
        if not total and container.duration:  # not every container counts its frames; its duration says about how many
            total = round(container.duration / av.time_base * fps)
        total = total or MAX_FRAMES  # nothing to go by: the first frames, then
        keep = set(np.linspace(0, total - 1, min(total, MAX_FRAMES)).round().astype(int).tolist())
        scale = min(1.0, MAX_SIDE / max(stream.width, stream.height))
        size = {"width": max(2, round(stream.width * scale)), "height": max(2, round(stream.height * scale))}
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in keep:
                frames.append(frame.to_ndarray(format="rgb24", **size))
            if i >= max(keep):
                break
    if not frames:
        raise ValueError(f"{Path(path).name} has no frames to read")
    # Evenly spaced, the frames kept are a video of their own at a lower frame rate.
    duration = total / fps
    return np.stack(frames), {"fps": len(frames) / duration, "total_num_frames": len(frames), "duration": duration}
