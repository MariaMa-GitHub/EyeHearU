"""Unit tests for beam_search."""

from app.services.beam_search import beam_search
from app.services.gloss_lm import GlossBeamLM, GlossBigramLM, START_TOKEN


def test_beam_prefers_lm_plausible_sequence():
    lm = GlossBigramLM(
        {
            (START_TOKEN, "a"): 10,
            ("a", "b"): 50,
            (START_TOKEN, "b"): 1,
            ("b", "a"): 1,
        },
        {START_TOKEN: 11, "a": 60, "b": 2},
        alpha=1.0,
    )
    candidates = [
        [
            {"sign": "a", "confidence": 0.45},
            {"sign": "b", "confidence": 0.55},
        ],
        [
            {"sign": "b", "confidence": 0.45},
            {"sign": "a", "confidence": 0.55},
        ],
    ]
    out = beam_search(candidates, lm, beam_size=4, lm_weight=2.0, top_sequences=4)
    assert out
    assert out[0].glosses == ("a", "b")


def test_beam_empty_input():
    lm = GlossBigramLM.uniform_over_vocab({"x"})
    assert beam_search([], lm) == []


def test_beam_single_clip():
    lm = GlossBigramLM.uniform_over_vocab({"hi"})
    out = beam_search(
        [[{"sign": "hi", "confidence": 0.99}]],
        lm,
        beam_size=2,
        lm_weight=1.0,
        top_sequences=3,
    )
    assert len(out) == 1
    assert out[0].glosses == ("hi",)


def test_beam_top_sequences_cap():
    lm = GlossBigramLM.uniform_over_vocab({"a", "b", "c"})
    candidates = [
        [{"sign": "a", "confidence": 0.4}, {"sign": "b", "confidence": 0.35}],
        [{"sign": "c", "confidence": 0.5}],
    ]
    out = beam_search(candidates, lm, beam_size=8, lm_weight=0.1, top_sequences=2)
    assert len(out) <= 2


def test_beam_trigram_prefers_plausible_third_gloss():
    """Bigram favors y after b; trigram mass on (a,b,x) should flip best path to a,b,x."""
    bi = GlossBigramLM(
        {
            (START_TOKEN, "a"): 20,
            ("a", "b"): 20,
            ("b", "x"): 2,
            ("b", "y"): 80,
        },
        {START_TOKEN: 20, "a": 40, "b": 82, "x": 10, "y": 80},
        alpha=1.0,
    )
    tri = {
        (START_TOKEN, START_TOKEN, "a"): 10,
        (START_TOKEN, "a", "b"): 10,
        ("a", "b", "x"): 100,
        ("a", "b", "y"): 1,
    }
    lm = GlossBeamLM(bi, tri)
    candidates = [
        [{"sign": "a", "confidence": 0.99}],
        [{"sign": "b", "confidence": 0.99}],
        [
            {"sign": "y", "confidence": 0.55},
            {"sign": "x", "confidence": 0.45},
        ],
    ]
    out = beam_search(candidates, lm, beam_size=8, lm_weight=4.0, top_sequences=3)
    assert out[0].glosses == ("a", "b", "x")
