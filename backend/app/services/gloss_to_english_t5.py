"""
Convert a gloss sequence into a natural English sentence using FLAN-T5-small.

This module uses a 2-pass strategy:
1) gloss -> sentence
2) if output still looks like gloss tokens, force a rewrite pass for natural English.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import T5ForConditionalGeneration, T5TokenizerFast

_MODEL_NAME = "google/flan-t5-small"
_PREFIX = (
    "Translate this ASL gloss sequence into ONE short, natural English sentence. "
    "Use normal grammar, function words, and punctuation. "
    "Do not copy gloss formatting. Glosses: "
)
_REWRITE_PREFIX = (
    "Rewrite the following rough phrase into ONE natural conversational English sentence. "
    "Keep the same core meaning, but make grammar fluent. Phrase: "
)
_MAX_INPUT_TOKENS = 128
_MAX_OUTPUT_TOKENS = 128


@functools.lru_cache(maxsize=1)
def _load_t5() -> tuple[T5ForConditionalGeneration, T5TokenizerFast]:
    from transformers import T5ForConditionalGeneration, T5TokenizerFast

    tokenizer = T5TokenizerFast.from_pretrained(_MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(_MODEL_NAME)
    model.eval()
    return model, tokenizer


def gloss_sequence_to_english_t5(glosses: list[str]) -> str:
    """Convert ordered gloss tokens into one natural English sentence via FLAN-T5-small."""
    clean = [str(g).strip().replace("_", " ") for g in glosses if str(g).strip()]
    if not clean:
        return ""

    prompt = _PREFIX + " ".join(clean)
    model, tokenizer = _load_t5()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=_MAX_INPUT_TOKENS,
        truncation=True,
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=_MAX_OUTPUT_TOKENS,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=0.9,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # If the output is still mostly copied gloss tokens, force a rewrite pass.
    if _looks_like_gloss_render(clean, text):
        rewrite_inputs = tokenizer(
            _REWRITE_PREFIX + text,
            return_tensors="pt",
            max_length=_MAX_INPUT_TOKENS,
            truncation=True,
        )
        rewrite_outputs = model.generate(
            **rewrite_inputs,
            max_new_tokens=_MAX_OUTPUT_TOKENS,
            num_beams=6,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
        rewritten = tokenizer.decode(rewrite_outputs[0], skip_special_tokens=True).strip()
        if rewritten:
            text = rewritten

    if text and text[-1] not in ".?!":
        text += "."
    return text


def _looks_like_gloss_render(gloss_tokens: list[str], text: str) -> bool:
    """Heuristic: generated text is mostly the same keywords in order."""
    out_tokens = [t.strip(".,!?;:").lower() for t in text.split() if t.strip()]
    gloss = [t.lower() for t in gloss_tokens]
    if not out_tokens or not gloss:
        return False

    # High overlap with little extra function words usually means "glossy" output.
    overlap = sum(1 for t in out_tokens if t in gloss)
    overlap_ratio = overlap / max(len(out_tokens), 1)
    short_and_overlapping = len(out_tokens) <= len(gloss) + 1 and overlap_ratio >= 0.75
    return short_and_overlapping
