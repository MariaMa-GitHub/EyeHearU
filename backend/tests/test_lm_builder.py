"""Tests for lm_builder (offline gloss LM generation)."""

import json
from pathlib import Path

from app.services.gloss_lm import START_TOKEN
from app.services.lm_builder import (
    build_lm_dict,
    load_glosses_from_label_map,
    parse_sequences_file,
)


def test_load_glosses_from_label_map_orders_by_index(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text(
        json.dumps({"gloss_to_index": {"zebra": 1, "apple": 0}}),
        encoding="utf-8",
    )
    assert load_glosses_from_label_map(p) == ["apple", "zebra"]


def test_parse_sequences_file_skips_comments_and_blank(tmp_path):
    p = tmp_path / "seq.txt"
    p.write_text(
        "# comment\n\nhello world\n  a  b c  \n",
        encoding="utf-8",
    )
    assert parse_sequences_file(p) == [["hello", "world"], ["a", "b", "c"]]


def test_build_lm_dict_floors_without_sequences():
    lm = build_lm_dict(["a", "b"], None, unigram_floor=2, start_bigram_floor=1)
    assert lm["alpha"] == 1.0
    assert lm["unigram_counts"]["a"] == 2
    assert lm["bigram_counts"][f"{START_TOKEN}|||a"] == 1
    assert lm["trigram_counts"] == {}


def test_build_lm_dict_adds_ngrams_from_sequences():
    lm = build_lm_dict(
        ["today", "i", "run"],
        [["today", "i"], ["i", "run", "today"]],
        unigram_floor=1,
        start_bigram_floor=1,
    )
    assert lm["bigram_counts"][f"{START_TOKEN}|||today"] >= 2
    assert lm["bigram_counts"]["today|||i"] >= 1
    assert lm["bigram_counts"]["i|||run"] >= 1
    assert lm["trigram_counts"][f"{START_TOKEN}|||{START_TOKEN}|||today"] >= 1
    assert f"{START_TOKEN}|||i|||run" in lm["trigram_counts"]
    assert "i|||run|||today" in lm["trigram_counts"]


def test_sequence_all_unknown_yields_no_extra_ngrams():
    lm = build_lm_dict(["a"], [["missing", "nope"]], unigram_floor=1, start_bigram_floor=1)
    assert lm["bigram_counts"] == {f"{START_TOKEN}|||a": 1}
    assert lm["trigram_counts"] == {}


def test_build_omits_unknown_tokens_in_sequence(tmp_path):
    lm = build_lm_dict(["x"], [["x", "missing", "y"]], unigram_floor=1, start_bigram_floor=1)
    assert "x|||y" not in lm["bigram_counts"]
