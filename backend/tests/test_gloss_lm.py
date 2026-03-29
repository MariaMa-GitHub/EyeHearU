"""Tests for gloss n-gram LM."""

import json

from app.services.gloss_lm import (
    GlossBeamLM,
    GlossBigramLM,
    START_TOKEN,
    _parse_trigram_counts,
    load_gloss_lm,
)


def test_uniform_log_p_constant():
    lm = GlossBigramLM.uniform_over_vocab({"a", "b"})
    p1 = lm.log_p(START_TOKEN, "a")
    p2 = lm.log_p("a", "b")
    assert p1 == p2


def test_load_gloss_lm_fallback_corrupt_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    lm = load_gloss_lm(bad, {0: "x", 1: "y"})
    assert isinstance(lm, GlossBeamLM)


def test_from_json_skips_keys_without_separator(tmp_path):
    path = tmp_path / "lm.json"
    path.write_text(
        json.dumps(
            {
                "bigram_counts": {"badkey": 9, "a|||b": 3},
                "unigram_counts": {"a": 3, "b": 3},
            }
        ),
        encoding="utf-8",
    )
    lm = GlossBigramLM.from_json_file(path, vocab_hint=None)
    assert lm.log_p("a", "b") <= 0


def test_from_json_file(tmp_path):
    path = tmp_path / "lm.json"
    path.write_text(
        json.dumps(
            {
                "alpha": 1.0,
                "bigram_counts": {"<s>|||hello": 10, "hello|||thanks": 5},
                "unigram_counts": {"<s>": 10, "hello": 15, "thanks": 5},
            }
        ),
        encoding="utf-8",
    )
    lm = GlossBigramLM.from_json_file(path, vocab_hint={"extra"})
    assert lm.log_p(START_TOKEN, "hello") < 0


def test_log_p_denom_zero_returns_neg_log_v():
    """Empty LM with alpha=0 yields denom 0 for unseen prev — hits fallback branch."""
    lm = GlossBigramLM({}, {}, alpha=0.0)
    # vocab_size 1, uni empty, bigram empty — denom for any prev is 0 + 0 = 0
    assert lm.log_p("unknown_prev", "y") == -__import__("math").log(1.0)


def test_load_missing_file_uniform():
    lm = load_gloss_lm(None, {0: "a", 1: "b"})
    assert lm.log_p(START_TOKEN, "a") == lm.log_p(START_TOKEN, "b")


def test_bigram_log_p_step_matches_first_and_rest():
    lm = GlossBigramLM(
        {(START_TOKEN, "a"): 5, ("a", "b"): 8},
        {START_TOKEN: 5, "a": 13, "b": 8},
        alpha=1.0,
    )
    assert lm.log_p_step(START_TOKEN, START_TOKEN, "a") == lm.log_p(START_TOKEN, "a")
    assert lm.log_p_step(START_TOKEN, "a", "b") == lm.log_p("a", "b")


def test_gloss_beam_lm_trigram_when_prefix_has_mass():
    bi = GlossBigramLM(
        {(START_TOKEN, "a"): 10, ("a", "b"): 10, ("b", "x"): 100, ("b", "y"): 1},
        {START_TOKEN: 10, "a": 20, "b": 102, "x": 50, "y": 50},
        alpha=1.0,
    )
    tri = {
        (START_TOKEN, START_TOKEN, "a"): 5,
        (START_TOKEN, "a", "b"): 5,
        ("a", "b", "x"): 80,
        ("a", "b", "y"): 1,
    }
    lm = GlossBeamLM(bi, tri)
    assert lm.log_p_step(START_TOKEN, START_TOKEN, "a") < 0
    lx = lm.log_p_step("a", "b", "x")
    ly = lm.log_p_step("a", "b", "y")
    assert lx > ly


def test_parse_trigram_skips_malformed_keys():
    out = _parse_trigram_counts(
        {"trigram_counts": {"only_two|||parts": 1, "a|||b|||c": 5}}
    )
    assert out == {("a", "b", "c"): 5}


def test_gloss_beam_from_json_file_with_trigrams(tmp_path):
    path = tmp_path / "lm.json"
    path.write_text(
        json.dumps(
            {
                "alpha": 1.0,
                "bigram_counts": {
                    "<s>|||a": 2,
                    "a|||b": 2,
                    "b|||c": 2,
                    "b|||d": 2,
                },
                "unigram_counts": {"<s>": 2, "a": 2, "b": 4, "c": 2, "d": 2},
                "trigram_counts": {"<s>|||a|||b": 3, "a|||b|||c": 50, "a|||b|||d": 1},
            }
        ),
        encoding="utf-8",
    )
    lm = GlossBeamLM.from_json_file(path)
    assert lm.log_p_step("a", "b", "c") > lm.log_p_step("a", "b", "d")
