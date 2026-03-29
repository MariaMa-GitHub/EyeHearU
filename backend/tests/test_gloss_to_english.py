"""Tests for gloss_sequence_to_english (gloss join + polish only)."""

from app.services.gloss_to_english import (
    _gloss_to_token,
    _polish_sentence,
    gloss_sequence_to_english,
)


def test_empty_glosses():
    assert gloss_sequence_to_english([]) == ""
    assert gloss_sequence_to_english(["", "  "]) == ""


def test_only_underscore_glosses_yield_empty():
    assert gloss_sequence_to_english(["___", "__"]) == ""


def test_polish_capitalizes_and_period():
    assert gloss_sequence_to_english(["hello", "world"]) == "Hello world."


def test_polish_replaces_lonely_i():
    assert "I" in gloss_sequence_to_english(["today", "i", "eat"])


def test_underscore_gloss():
    out = gloss_sequence_to_english(["some_gloss"])
    assert "some gloss" in out.lower()


def test_gloss_to_token_blank():
    assert _gloss_to_token("   ") == ""


def test_polish_whitespace_only():
    assert _polish_sentence("  \n ") == ""


def test_polish_single_character():
    assert _polish_sentence("i") == "I."


def test_polish_preserves_existing_terminal_punct():
    assert _polish_sentence("OK?") == "OK?"
