"""Tests for FLAN-T5 gloss→English (mocked transformers)."""

from unittest.mock import MagicMock, patch

import pytest

from app.services import gloss_to_english_t5 as t5m


@pytest.fixture(autouse=True)
def clear_t5_cache():
    t5m._load_t5.cache_clear()
    yield
    t5m._load_t5.cache_clear()


def test_empty_gloss_list():
    assert t5m.gloss_sequence_to_english_t5([]) == ""
    assert t5m.gloss_sequence_to_english_t5(["", "  "]) == ""


def test_looks_like_gloss_render_true_when_short_overlap():
    assert t5m._looks_like_gloss_render(["a", "b"], "a b")


def test_looks_like_gloss_render_false_when_long_extra_words():
    gloss = ["hello"]
    text = "hello there my friend today we go"
    assert t5m._looks_like_gloss_render(gloss, text) is False


def test_looks_like_gloss_render_false_empty():
    assert t5m._looks_like_gloss_render([], "x") is False
    assert t5m._looks_like_gloss_render(["a"], "") is False


@patch.object(t5m, "_load_t5")
def test_t5_single_pass_natural_output(mock_load):
    tok = MagicMock()
    model = MagicMock()
    mock_load.return_value = (model, tok)

    def decode_side_effect(seq, skip_special_tokens=True):
        return "Hello there."

    tok.decode.side_effect = decode_side_effect
    gen_out = MagicMock()
    gen_out.__getitem__ = lambda self, i: self if i == 0 else None
    model.generate.return_value = gen_out

    out = t5m.gloss_sequence_to_english_t5(["hello_there"])
    assert out == "Hello there."
    assert model.generate.call_count == 1


@patch.object(t5m, "_load_t5")
def test_t5_rewrite_pass_when_output_still_glossy(mock_load):
    tok = MagicMock()
    model = MagicMock()
    mock_load.return_value = (model, tok)

    decodes = ["hello thanks", "Hello, thanks."]

    def decode_side_effect(seq, skip_special_tokens=True):
        return decodes.pop(0)

    tok.decode.side_effect = decode_side_effect
    gen_out = MagicMock()
    gen_out.__getitem__ = lambda self, i: self if i == 0 else None
    model.generate.return_value = gen_out

    out = t5m.gloss_sequence_to_english_t5(["hello", "thanks"])
    assert out == "Hello, thanks."
    assert model.generate.call_count == 2


@patch.object(t5m, "_load_t5")
def test_t5_rewrite_empty_keeps_first_and_adds_period(mock_load):
    tok = MagicMock()
    model = MagicMock()
    mock_load.return_value = (model, tok)

    decodes = ["hello thanks", ""]

    def decode_side_effect(seq, skip_special_tokens=True):
        return decodes.pop(0)

    tok.decode.side_effect = decode_side_effect
    gen_out = MagicMock()
    gen_out.__getitem__ = lambda self, i: self if i == 0 else None
    model.generate.return_value = gen_out

    out = t5m.gloss_sequence_to_english_t5(["hello", "thanks"])
    assert out == "hello thanks."


@patch("transformers.T5ForConditionalGeneration.from_pretrained")
@patch("transformers.T5TokenizerFast.from_pretrained")
def test_load_t5_pulls_pretrained_models(mock_tokenizer_from_pretrained, mock_model_from_pretrained):
    t5m._load_t5.cache_clear()
    mock_model_from_pretrained.return_value = MagicMock()
    mock_tokenizer_from_pretrained.return_value = MagicMock()
    m1, t1 = t5m._load_t5()
    m2, t2 = t5m._load_t5()
    assert m1 is m2 and t1 is t2
    mock_model_from_pretrained.assert_called_once()
    mock_tokenizer_from_pretrained.assert_called_once()


@patch.object(t5m, "_load_t5")
def test_t5_appends_period_when_missing(mock_load):
    tok = MagicMock()
    model = MagicMock()
    mock_load.return_value = (model, tok)
    tok.decode.return_value = "No ending punctuation"
    gen_out = MagicMock()
    gen_out.__getitem__ = lambda self, i: self if i == 0 else None
    model.generate.return_value = gen_out

    out = t5m.gloss_sequence_to_english_t5(["x"])
    assert out.endswith(".")
    assert "No ending punctuation" in out
