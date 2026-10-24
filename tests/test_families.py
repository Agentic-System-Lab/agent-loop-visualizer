"""The registry of model families, and what every family must provide."""

import pytest

import families
from families import GEMMA4, QWEN3_5


def test_model_type_picks_the_family():
    assert families.for_model_type("gemma4") is GEMMA4
    assert families.for_model_type("qwen3_5") is QWEN3_5
    with pytest.raises(ValueError, match="llama.*gemma4, qwen3_5"):
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
