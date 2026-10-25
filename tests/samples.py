"""A small file of every format the page accepts, made from the examples' files in assets/examples, so a
test can ask the same question of each format: the receipt in every image format, the recording in every
audio format, the rolling ball in every video container, and the sales data in every data format.

FORMATS lists them all; test_media.py checks it covers every extension media.py accepts, so a format
added there without a sample here fails the tests."""

import csv
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

EXAMPLES = Path(__file__).resolve().parent.parent / "assets" / "examples"
IMAGES = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP", "gif": "GIF", "bmp": "BMP"}
AUDIO = {"wav": ("wav", "pcm_s16le"), "mp3": ("mp3", "libmp3lame"), "m4a": ("ipod", "aac"), "flac": ("flac", "flac"), "ogg": ("ogg", "libopus")}
VIDEO = {"mp4": ("mp4", "libx264"), "mov": ("mov", "libx264"), "mkv": ("matroska", "libx264"), "webm": ("webm", "libvpx-vp9")}
DATA = ("csv", "tsv", "json", "txt", "md", "xlsx")
FORMATS = (*IMAGES, *AUDIO, *VIDEO, *DATA)

TOTAL_PRINTED = "370"  # on the receipt, whose items add up to 360
SPOKEN = "Mister Quilter is the apostle of the middle classes and we are glad to welcome his gospel"
YEAR_REVENUE = 2131200  # of the sales data, December the best month


def make(folder: Path, ext: str) -> Path:
    """Write sample.<ext> into `folder` and return its path."""
    path = folder / f"sample.{ext}"
    if ext in IMAGES:
        with Image.open(EXAMPLES / "receipt.png") as image:
            image.convert("RGB").save(path, IMAGES[ext])
    elif ext in AUDIO:
        _audio(path, *AUDIO[ext])
    elif ext in VIDEO:
        _video(path, *VIDEO[ext])
    elif ext in DATA:
        _data(path, ext)
    else:
        raise ValueError(f"no sample for .{ext}")
    return path


def _audio(path: Path, container_format: str, codec: str):
    """The recording, decoded and encoded again in another format."""
    rate = 48000 if codec == "libopus" else 16000  # Opus takes 48 kHz
    with av.open(str(EXAMPLES / "speech.flac")) as source:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        frames = [r for frame in source.decode(audio=0) for r in resampler.resample(frame)] + resampler.resample(None)
    with av.open(str(path), "w", format=container_format) as out:
        stream = out.add_stream(codec, rate=rate, layout="mono")
        for frame in frames:
            frame.pts = None
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def ball_frames(n: int = 32, size=(320, 240)):
    """A red ball rolling from the left edge to the right, as in ball.mp4: 32 frames, played at 4 a second. Gemma 4
    writes each frame's time in whole seconds, and over 8 seconds those times tell it the frames are a sequence; packed
    into 3 seconds, it took them for still pictures one time in four."""
    for i in range(n):
        frame = Image.new("RGB", size, "#f2efe6")
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, size[1] - 50, size[0], size[1]), fill="#9bb37a")
        x = 30 + i * (size[0] - 60) / (n - 1)
        draw.ellipse((x - 22, size[1] - 94, x + 22, size[1] - 50), fill="#d23a2a")
        yield np.asarray(frame)


def _video(path: Path, container_format: str, codec: str):
    with av.open(str(path), "w", format=container_format) as out:
        stream = out.add_stream(codec, rate=Fraction(4))
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for pixels in ball_frames():
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def _data(path: Path, ext: str):
    """The sales data: a table in each format, and for .txt a report with one line per row."""
    with open(EXAMPLES / "sales.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    columns = list(rows[0])
    if ext == "csv":
        path.write_bytes((EXAMPLES / "sales.csv").read_bytes())
    elif ext == "tsv":
        path.write_text("\n".join("\t".join(r) for r in [columns, *([row[c] for c in columns] for row in rows)]) + "\n")
    elif ext == "json":
        path.write_text(json.dumps([{**row, "units": int(row["units"]), "revenue": int(row["revenue"])} for row in rows], indent=1))
    elif ext == "txt":
        path.write_text("Sales report, 2025\n" + "".join(f"{r['month']} {r['region']}: {r['units']} units, revenue {r['revenue']}\n" for r in rows))
    elif ext == "md":
        table = [f"| {' | '.join(columns)} |", f"|{'---|' * len(columns)}"] + [f"| {' | '.join(row[c] for c in columns)} |" for row in rows]
        path.write_text("# Sales, 2025\n\n" + "\n".join(table) + "\n")
    elif ext == "xlsx":
        from openpyxl import Workbook

        book = Workbook()
        sheet = book.active
        sheet.append(columns)
        for row in rows:
            sheet.append([row["month"], row["region"], int(row["units"]), int(row["revenue"])])
        book.save(path)
