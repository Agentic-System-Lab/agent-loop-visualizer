"""The registry of model families, and what every family must provide."""

import re

import pytest

from agent import model_families
from agent.i18n import LANGS
from agent.model_families import GEMMA4, QWEN3_5

CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")


def test_model_type_picks_the_family():
    assert model_families.for_model_type("gemma4") is GEMMA4
    assert model_families.for_model_type("qwen3_5") is QWEN3_5
    with pytest.raises(ValueError, match="'llama' are not supported; supported: gemma4, qwen3_5"):
        model_families.for_model_type("llama")


@pytest.mark.parametrize("family", model_families.FAMILIES.values(), ids=list(model_families.FAMILIES))
def test_a_family_explains_every_token_it_relies_on(family):
    """Stop and channel tokens are drawn as chips with a note, so each needs one."""
    for piece in (*family.stop_tokens, *family.channel_switch, family.user_turn.split("\n")[0].removesuffix("user")):
        assert piece in family.special_tokens, piece
    assert set(family.channel_switch.values()) == {"thought", "content", "tool_call"}
    assert family.thought_prefix.startswith(next(p for p, c in family.channel_switch.items() if c == "thought"))


@pytest.mark.parametrize("family", model_families.FAMILIES.values(), ids=list(model_families.FAMILIES))
def test_special_re_prefers_the_longest_token(family):
    for piece in family.special_tokens:
        assert family.special_re.match(piece).group(0) == piece


def test_a_token_shows_as_the_text_it_stands_for():
    """Qwen's vocabulary spells bytes as characters (如果 is å¦Ĥæŀľ) and Gemma's marks a space with ▁; a popover shows the text."""
    assert QWEN3_5.show_piece("å¦Ĥæŀľ") == "如果"
    assert QWEN3_5.show_piece("ĠhelloĊ") == "␣hello↵"
    assert QWEN3_5.show_piece("<|im_end|>") == "<|im_end|>"
    assert GEMMA4.show_piece("▁Hi\n") == "␣Hi↵"


def test_a_byte_that_is_only_part_of_a_character_shows_in_hex():
    """Both vocabularies split a rare character into bytes, and both show such a byte the same way."""
    assert QWEN3_5.show_piece("ð") == "<0xF0>" and QWEN3_5.show_piece("łĢ") == "<0xA0><0x80>"
    assert QWEN3_5.show_piece("Ġð") == "␣<0xF0>"
    assert GEMMA4.show_piece("<0xF0>") == "<0xF0>"


def test_every_model_and_drafter_downloads_at_a_pinned_commit():
    assert set(model_families.REVISIONS) == set(model_families.MODELS) | set(model_families.DRAFTERS.values())
    assert all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in model_families.REVISIONS.values())


def test_vllm_drafts_a_listed_model_with_the_mtp_layer_in_its_checkpoint_or_the_drafter_transformers_uses():
    assert set(model_families.VLLM_SPECULATIVE) <= set(model_families.MODELS)
    for model_id, config in model_families.VLLM_SPECULATIVE.items():
        assert config["num_speculative_tokens"] > 0
        assert config.get("model") == model_families.DRAFTERS.get(model_id) or config == {"method": "mtp", "num_speculative_tokens": config["num_speculative_tokens"]}


def test_the_menu_lists_one_model_of_each_family():
    assert len(model_families.MODELS) == len(model_families.FAMILIES)


@pytest.mark.parametrize("family", model_families.FAMILIES.values(), ids=list(model_families.FAMILIES))
def test_a_family_explains_itself_in_both_languages(family):
    """The model tab lists the same tokens in the same groups whichever language the page is in."""
    groups = {lang: [[piece for piece, _, _ in rows] for _, _, rows in family.special_groups[lang]] for lang in LANGS}
    assert groups["zh"] == groups["en"]
    assert all(set(notes) == set(LANGS) for notes in family.special_tokens.values())
    for text in (*family.stop_tokens.values(), family.thinking_help, family.tools_note, family.model_note):
        assert set(text) == set(LANGS) and all(text.values())
    english = [family.thinking_help["en"], family.tools_note["en"], family.model_note["en"]]
    english += [meaning["en"] for meaning in family.stop_tokens.values()]
    english += [part for title, sub, rows in family.special_groups["en"] for part in (title, sub, *(n + d for _, n, d in rows))]
    assert not [text for text in english if CJK.search(text)]


@pytest.mark.parametrize("family", list(model_families.FAMILIES.values()), ids=lambda f: f.key)
def test_a_family_explains_its_files_in_both_languages(family):
    """A note on a file names the same fields and tokens in both languages, and the fields it shows are dotted names."""
    for name, note in {**family.file_notes, **family.drafter_file_notes}.items():
        assert set(note.text) == set(LANGS) and all(note.text.values()), name
        assert sorted(re.findall(r"`(.+?)`", note.text["zh"])) == sorted(re.findall(r"`(.+?)`", note.text["en"])), name
        assert not CJK.search(note.text["en"]), name
        assert all(re.fullmatch(r"\w+(\.\w+)*", f) for f in note.fields), name
