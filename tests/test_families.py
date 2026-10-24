"""The registry of model families, and what every family must provide."""

import re

import pytest

import families
from families import GEMMA4, QWEN3_5
from i18n import LANGS

CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")


def test_model_type_picks_the_family():
    assert families.for_model_type("gemma4") is GEMMA4
    assert families.for_model_type("qwen3_5") is QWEN3_5
    with pytest.raises(ValueError, match="'llama' are not supported; supported: gemma4, qwen3_5"):
        families.for_model_type("llama")


@pytest.mark.parametrize("family", families.FAMILIES.values(), ids=list(families.FAMILIES))
def test_a_family_explains_every_token_it_relies_on(family):
    """Stop and channel tokens are drawn as chips with a note, so each needs one."""
    for piece in (*family.stop_tokens, *family.channel_switch, family.user_turn.split("\n")[0].removesuffix("user")):
        assert piece in family.special_tokens, piece
    assert set(family.channel_switch.values()) == {"thought", "content", "tool_call"}
    assert family.thought_prefix.startswith(next(p for p, c in family.channel_switch.items() if c == "thought"))


@pytest.mark.parametrize("family", families.FAMILIES.values(), ids=list(families.FAMILIES))
def test_special_re_prefers_the_longest_token(family):
    for piece in family.special_tokens:
        assert family.special_re.match(piece).group(0) == piece


def test_the_menu_lists_one_model_of_each_family():
    assert len(families.MODELS) == len(families.FAMILIES)


@pytest.mark.parametrize("family", families.FAMILIES.values(), ids=list(families.FAMILIES))
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
