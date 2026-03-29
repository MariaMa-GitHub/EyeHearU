"""Tests for Bedrock gloss→English (mocked boto3)."""

import io
import json
from unittest.mock import MagicMock, patch

from app.services.gloss_to_english_bedrock import (
    _normalize_sentence_case,
    gloss_sequence_to_english_bedrock,
)


def test_normalize_empty():
    assert _normalize_sentence_case("") == ""
    assert _normalize_sentence_case("   ") == ""


def test_normalize_already_mixed_case_unchanged_capitalize_first():
    assert _normalize_sentence_case("hello world") == "Hello world"


def test_normalize_mostly_uppercase_lowercases_first():
    out = _normalize_sentence_case("HELLO WORLD")
    assert out == "Hello world"


def test_normalize_pronoun_i():
    assert " I " in f" {_normalize_sentence_case('i am here')} "


def test_normalize_acronym_asl():
    s = _normalize_sentence_case("i use asl daily")
    assert "ASL" in s


def test_normalize_capitalize_first_alpha_after_digits():
    assert _normalize_sentence_case("123 starts") == "123 Starts"


def test_normalize_capitalize_first_alpha_after_punct():
    assert _normalize_sentence_case("!bang") == "!Bang"


def test_normalize_no_letters_skips_upper_ratio_block():
    assert _normalize_sentence_case("123") == "123"


def test_normalize_no_alphabetic_char_after_rules():
    assert _normalize_sentence_case("+++") == "+++"


def test_normalize_mixed_case_not_mostly_upper():
    """Upper ratio ≤ 0.7: do not force lowercase."""
    assert _normalize_sentence_case("Hello WORLD") == "Hello WORLD"


def test_gloss_sequence_empty():
    assert gloss_sequence_to_english_bedrock([], region="x", model_id="y") == ""


@patch("app.services.gloss_to_english_bedrock.boto3")
def test_bedrock_success_adds_period(mock_boto):
    body = io.BytesIO(
        json.dumps({"content": [{"text": "Hello from model"}]}).encode("utf-8")
    )
    mock_boto.client.return_value.invoke_model.return_value = {"body": body}

    out = gloss_sequence_to_english_bedrock(
        ["hello"], region="ca-central-1", model_id="anthropic.fake"
    )
    assert out == "Hello from model."
    mock_boto.client.assert_called_once()
    _, kwargs = mock_boto.client.call_args
    assert kwargs["region_name"] == "ca-central-1"


@patch("app.services.gloss_to_english_bedrock.boto3")
def test_bedrock_content_not_list_yields_empty_normalized(mock_boto):
    body = io.BytesIO(json.dumps({"content": {"not": "list"}}).encode("utf-8"))
    mock_boto.client.return_value.invoke_model.return_value = {"body": body}

    out = gloss_sequence_to_english_bedrock(
        ["a"], region="r", model_id="m", timeout_s=5.0
    )
    assert out == ""


@patch("app.services.gloss_to_english_bedrock.boto3")
def test_bedrock_empty_text_after_strip(mock_boto):
    body = io.BytesIO(json.dumps({"content": [{"text": "  "}]}).encode("utf-8"))
    mock_boto.client.return_value.invoke_model.return_value = {"body": body}

    out = gloss_sequence_to_english_bedrock(["b"], region="r", model_id="m")
    assert out == ""
