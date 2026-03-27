"""
Format ordered classifier glosses for display as a single line of text (fully offline).

Gloss labels from the model are usually English-like lemmas. This module joins them
with light surface polish only: underscore → space, isolated ``i`` → ``I``, capitalize
first character, add a final period if missing.

This is **not** full grammatical English generation—there is no tense agreement, reordering,
or article insertion.
"""

from __future__ import annotations

import re


def _gloss_to_token(gloss: str) -> str:
    g = gloss.strip()
    if not g:
        return ""
    if "_" in g:
        return g.replace("_", " ").strip()
    return g


def _polish_sentence(text: str) -> str:
    s = " ".join(text.split())
    if not s:
        return ""
    s = re.sub(r"(?<!\w)i(?!\w)", "I", s, flags=re.IGNORECASE)
    s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
    if s[-1] not in ".?!":
        s += "."
    return s


def gloss_sequence_to_english(glosses: list[str]) -> str:
    """Join gloss strings into one line with light polish (see module docstring)."""
    glist = [str(x).strip() for x in glosses if str(x).strip()]
    if not glist:
        return ""
    tokens = [_gloss_to_token(x) for x in glist]
    tokens = [t for t in tokens if t]
    if not tokens:
        return ""
    return _polish_sentence(" ".join(tokens))
