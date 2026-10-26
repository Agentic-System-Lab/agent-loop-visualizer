"""Attachments: saving them, what the model's message says about them, and decoding them for the processor.
The audio and video are made on the fly with PyAV, the same library that decodes them."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image

import samples
from agent import media
from agent.media import Attachment

GEMMA = ("image", "audio", "video")
QWEN = ("image", "video")
EXAMPLES = Path(__file__).resolve().parent.parent / "app" / "assets" / "examples"


def test_a_file_is_media_only_if_the_model_reads_its_kind():
    assert [media.kind_of(n, GEMMA) for n in ("a.PNG", "b.mp3", "c.mov", "d.csv", "e")] == ["image", "audio", "video", "file", "file"]
    assert media.kind_of("b.mp3", QWEN) == "file"  # Qwen3.5 has no audio encoder: only run_python could open it


def test_the_input_takes_data_files_only_when_code_may_read_them():
    assert "csv" not in media.accepted(QWEN, python=False) and "png" in media.accepted(QWEN, python=False)
    assert "mp3" not in media.accepted(QWEN, python=True) and {"csv", "xlsx"} <= set(media.accepted(QWEN, python=True))
    assert media.accepted((), python=False) == []


def test_saving_keeps_every_name_distinct_and_plain(tmp_path):
    first = media.save(tmp_path, "收據 (1).png", b"one", GEMMA)
    second = media.save(tmp_path, "收據 (1).png", b"two", GEMMA)
    data = media.save(tmp_path, "my data.csv", b"x,y\n", GEMMA)
    assert (first.name, second.name, data.name) == ("收據_1.png", "收據_1-2.png", "my_data.csv")
    assert (first.kind, data.kind, data.size) == ("image", "file", 4)
    assert Path(second.path).read_bytes() == b"two"


def test_the_question_is_plain_text_without_attachments():
    assert media.user_content("hi", []) == "hi"


def test_a_recording_alone_is_the_whole_question():
    clip = Attachment("recording.wav", "audio", "/c/recording.wav", 10)
    assert media.user_content("", [clip]) == [clip.part()]


def test_parts_are_the_conversations_media_in_order():
    a, b = Attachment("a.png", "image", "/a.png", 1), Attachment("b.mp4", "video", "/b.mp4", 1)
    messages = [
        {"role": "user", "content": media.user_content("one", [a])},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": media.user_content("two", [b, Attachment("sales.csv", "file", str(EXAMPLES / "sales.csv"), 942)])},
    ]
    assert [p["name"] for p in media.parts(messages)] == ["a.png", "b.mp4"]


def test_the_note_names_a_text_files_columns_but_shows_none_of_its_rows(tmp_path):
    table = tmp_path / "wide.csv"
    table.write_text(",".join(f"column_{n}" for n in range(60)) + "\n1,2,3\n4,5,6\n")
    sheet = tmp_path / "book.xlsx"
    sheet.write_bytes(b"PK\x03\x04")
    note = media.user_content("", [Attachment("wide.csv", "file", str(table), 2500), Attachment("book.xlsx", "file", str(sheet), 4)])[0]["text"]
    head, tail = note.split("\n\n")
    assert head.startswith("Attached for run_python, which reads them:\n/data/wide.csv (2.5 KB), whose first line is: column_0,column_1,")
    assert len(head.split("first line is: ")[1]) == media.PREVIEW_CHARS and "1,2,3" not in head  # no rows to copy into the code
    assert tail == "/data/book.xlsx (4 B)"  # not text: no first line


def test_an_image_comes_upright_and_in_rgb(tmp_path):
    path = tmp_path / "phone.jpg"
    image = Image.new("RGBA", (40, 20), (255, 0, 0, 255))
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways: turn 90° to show it
    image.convert("RGB").save(path, exif=exif)
    loaded = media.load_image(str(path))
    assert loaded.mode == "RGB" and loaded.size == (20, 40)


def write_audio(path: Path, rate: int, seconds: float, channels: int):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("pcm_s16le", rate=rate, layout="stereo" if channels == 2 else "mono")
        t = np.arange(int(rate * seconds)) / rate
        tone = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(np.tile(tone, (channels, 1)).reshape(1, -1) if channels == 2 else tone.reshape(1, -1),
                                           format="s16", layout="stereo" if channels == 2 else "mono")
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_audio_becomes_mono_samples_at_the_models_rate(tmp_path):
    path = tmp_path / "stereo.wav"
    write_audio(path, 44100, 1.5, channels=2)
    samples = media.load_audio(str(path), 16000)
    assert samples.ndim == 1 and samples.dtype == np.float32
    assert abs(len(samples) - 24000) < 200 and 0.1 < np.abs(samples).max() <= 1.0


def test_the_sample_recording_decodes():
    samples = media.load_audio(str(EXAMPLES / "speech.flac"), 16000)
    assert 5.5 < len(samples) / 16000 < 6.2


def write_video(path: Path, frames: int, size: tuple[int, int] = (64, 48), fps=10):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps))
        stream.width, stream.height, stream.pix_fmt = *size, "yuv420p"
        for i in range(frames):
            pixels = np.full((size[1], size[0], 3), i % 256, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_a_short_video_keeps_every_frame():
    frames, meta = media.load_video(str(EXAMPLES / "ball.mp4"))
    assert frames.shape == (32, 240, 320, 3)
    assert meta == {"fps": pytest.approx(4.0), "total_num_frames": 32, "duration": pytest.approx(8.0)}


def test_a_long_video_keeps_evenly_spaced_frames_at_their_own_rate(tmp_path):
    path = tmp_path / "long.mp4"
    write_video(path, frames=200, size=(1600, 900))
    frames, meta = media.load_video(str(path))
    assert frames.shape == (media.MAX_FRAMES, 432, 768, 3)  # shrunk to MAX_SIDE on the longer side
    assert meta["total_num_frames"] == media.MAX_FRAMES and meta["duration"] == pytest.approx(20.0)
    # Frame i of the kept ones is at i / fps seconds, so the processor's timestamps stay true.
    assert meta["fps"] == pytest.approx(media.MAX_FRAMES / 20.0)
    assert frames[-1].mean() > frames[0].mean() + 100  # the last kept frame is from the end


# ---------- every format the page accepts ----------

def test_every_format_the_page_accepts_has_a_sample():
    assert sorted(samples.FORMATS) == sorted([*media.KINDS, *media.DATA_TYPES])
    assert sorted(media.accepted(GEMMA, python=True)) == sorted(samples.FORMATS)
    assert sorted(media.accepted(QWEN, python=True)) == sorted(ext for ext in samples.FORMATS if media.KINDS.get(ext) != "audio")


@pytest.mark.parametrize("ext", samples.FORMATS)
def test_every_format_is_read_as_what_it_is(tmp_path, ext):
    path = str(samples.make(tmp_path, ext))
    kind = media.KINDS.get(ext, "file")
    assert media.kind_of(path, GEMMA) == kind
    assert media.kind_of(path, QWEN) == ("file" if kind == "audio" else kind)  # Qwen3.5 has no ears: run_python may open it
    if kind == "image":
        image = media.load_image(path)
        assert image.mode == "RGB" and image.size == (420, 560)
    elif kind == "audio":
        assert 5.7 < len(media.load_audio(path, 16000)) / 16000 < 6.1
    elif kind == "video":
        frames, meta = media.load_video(path)
        assert frames.shape == (32, 240, 320, 3) and meta["fps"] == pytest.approx(4.0)
        assert frames[0][190 - 22, 30].tolist() != frames[-1][190 - 22, 30].tolist()  # the ball has rolled away from the left
    else:
        first = {"csv": "month,region,units,revenue", "tsv": "month\tregion\tunits\trevenue", "txt": "Sales report, 2025", "md": "# Sales, 2025"}
        assert media.preview(path) == first.get(ext, "")


def test_the_chat_plays_back_what_browsers_play_and_shows_frames_of_the_rest(tmp_path):
    assert media.mime("a.MP3") == "audio/mpeg" and media.mime("b.m4a") == "audio/mp4" and media.mime("c.mov") == "video/quicktime"
    assert media.mime("d.mkv") is None  # Safari cannot play Matroska
    assert all(media.mime(f"x.{ext}") for ext, kind in media.KINDS.items() if kind == "audio")
    grid = media.frame_grid(str(samples.make(tmp_path, "mkv")))
    assert grid.shape == (2 * 120 + 6, 2 * 160 + 6, 4)  # four 320×240 frames at 120 pixels tall, two by two
    assert grid[120:126, :, 3].max() == 0 and grid[:, 160:166, 3].max() == 0  # see-through gaps between them
    red = lambda pixel: pixel[0] > 150 and pixel[1] < 100  # the ball's center, at half size: (15, 84) in the first frame
    assert red(grid[84, 15]) and not red(grid[126 + 84, 166 + 15])  # by the last frame it has rolled away from the left
