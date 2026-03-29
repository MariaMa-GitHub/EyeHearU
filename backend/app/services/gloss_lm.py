"""
Gloss n-gram language models for multi-clip decoding.

- **GlossBigramLM**: Laplace-smoothed bigram P(g_j | g_{i-1}), with ``<s>`` start.
- **GlossBeamLM**: bigram + optional **trigram** P(g_k | g_{k-2}, g_{k-1}) with backoff to
  ``log_p_step`` on the bigram when the trigram context has no observed mass.

Beam search uses ``log_p_step(prev2, prev1, next)`` with ``(prev2, prev1) = (<s>, <s>)``
for the first gloss, then ``(<s>, g0)``, then ``(g_{t-2}, g_{t-1})``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

START_TOKEN = "<s>"


class GlossBigramLM:
    """Log-probability estimates from (sparse) bigram counts + smoothing."""

    __slots__ = (
        "_uniform",
        "_vocab_size_uniform",
        "_bigram",
        "_uni",
        "_alpha",
        "_vocab_size",
    )

    def __init__(
        self,
        bigram_counts: dict[tuple[str, str], int] | None = None,
        unigram_counts: dict[str, int] | None = None,
        *,
        alpha: float = 1.0,
        uniform: bool = False,
        vocab_size_uniform: int = 2,
    ):
        self._uniform = bool(uniform)
        self._vocab_size_uniform = max(int(vocab_size_uniform), 1)
        self._alpha = float(alpha)
        if self._uniform:
            self._bigram = {}
            self._uni = {}
            self._vocab_size = self._vocab_size_uniform
            return

        self._bigram = dict(bigram_counts or {})
        keys: set[str] = set()
        for (a, b) in self._bigram:
            keys.add(a)
            keys.add(b)
        if unigram_counts:
            keys |= set(unigram_counts.keys())
        self._vocab_size = max(len(keys), 1)

        self._uni = {g: 0 for g in keys}
        if unigram_counts:
            for g, c in unigram_counts.items():
                self._uni[g] = self._uni.get(g, 0) + int(c)
        for (prev, _), c in self._bigram.items():
            self._uni[prev] = self._uni.get(prev, 0) + int(c)

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def log_p(self, prev_gloss: str, next_gloss: str) -> float:
        """Log P(next | prev) with Laplace smoothing over the observed vocab."""
        if self._uniform:
            return -math.log(self._vocab_size_uniform)

        v = float(self._vocab_size)
        a = self._alpha
        c_bg = self._bigram.get((prev_gloss, next_gloss), 0)
        denom = self._uni.get(prev_gloss, 0) + a * v
        if denom <= 0:
            return -math.log(v)
        num = c_bg + a
        return math.log(num / denom)

    def log_p_step(self, prev2: str, prev1: str, next_gloss: str) -> float:
        """One step for beam search: first token uses P(next | <s>); else P(next | prev1)."""
        if prev2 == START_TOKEN and prev1 == START_TOKEN:
            return self.log_p(START_TOKEN, next_gloss)
        return self.log_p(prev1, next_gloss)

    @classmethod
    def uniform_over_vocab(cls, vocab: set[str]) -> GlossBigramLM:
        """O(1) storage — all transitions get the same log-prob (beam uses model scores)."""
        vs = len(vocab | {START_TOKEN})
        return cls(uniform=True, vocab_size_uniform=max(vs, 1))

    @classmethod
    def from_json_raw(
        cls,
        raw: dict,
        *,
        vocab_hint: set[str] | None = None,
    ) -> GlossBigramLM:
        bg_raw: dict[str, int] = raw.get("bigram_counts") or {}
        uni_raw: dict[str, int] = raw.get("unigram_counts") or {}
        bigram: dict[tuple[str, str], int] = {}
        for k, c in bg_raw.items():
            if "|||" in k:
                a, b = k.split("|||", 1)
                bigram[(a, b)] = int(c)
        uni: dict[str, int] = {str(g): int(c) for g, c in uni_raw.items()}
        if vocab_hint:
            for g in vocab_hint:
                uni.setdefault(g, 0)
        return cls(bigram, uni if uni else None, alpha=float(raw.get("alpha", 1.0)))

    @classmethod
    def from_json_file(cls, path: Path, *, vocab_hint: set[str] | None = None) -> GlossBigramLM:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_json_raw(raw, vocab_hint=vocab_hint)


def _parse_trigram_counts(raw: dict) -> dict[tuple[str, str, str], int]:
    tri_raw: dict[str, int] = raw.get("trigram_counts") or {}
    out: dict[tuple[str, str, str], int] = {}
    for k, c in tri_raw.items():
        parts = k.split("|||")
        if len(parts) == 3:
            out[(parts[0], parts[1], parts[2])] = int(c)
    return out


class GlossBeamLM:
    """
    Bigram base LM + optional trigram extension for beam decoding.
    If trigram context (prev2, prev1) has no observed count mass, falls back to bigram step.
    """

    __slots__ = ("_bi", "_tri", "_tri_prefix", "_alpha")

    def __init__(
        self,
        bigram: GlossBigramLM,
        trigram_counts: dict[tuple[str, str, str], int],
    ):
        self._bi = bigram
        self._tri = dict(trigram_counts)
        self._alpha = bigram.alpha
        self._tri_prefix: dict[tuple[str, str], int] = {}
        for (a, b, _), c in self._tri.items():
            self._tri_prefix[(a, b)] = self._tri_prefix.get((a, b), 0) + int(c)

    @classmethod
    def uniform_over_vocab(cls, vocab: set[str]) -> GlossBeamLM:
        return cls(GlossBigramLM.uniform_over_vocab(vocab), {})

    @classmethod
    def from_json_file(cls, path: Path, *, vocab_hint: set[str] | None = None) -> GlossBeamLM:
        raw = json.loads(path.read_text(encoding="utf-8"))
        bi = GlossBigramLM.from_json_raw(raw, vocab_hint=vocab_hint)
        tri = _parse_trigram_counts(raw)
        return cls(bi, tri)

    def log_p_step(self, prev2: str, prev1: str, next_gloss: str) -> float:
        tot = self._tri_prefix.get((prev2, prev1), 0)
        if tot > 0:
            c = self._tri.get((prev2, prev1, next_gloss), 0)
            v = float(self._bi.vocab_size)
            a = self._alpha
            denom = tot + a * v
            return math.log((c + a) / denom)
        return self._bi.log_p_step(prev2, prev1, next_gloss)

    # Delegate for tests / introspection that still read bigram mass
    def log_p(self, prev_gloss: str, next_gloss: str) -> float:
        return self._bi.log_p(prev_gloss, next_gloss)


def load_gloss_lm(path: Path | None, index_to_gloss: dict[int, str]) -> GlossBeamLM:
    """Load trigram+bigram LM from disk, or fall back to uniform over label-map glosses."""
    vocab = {g for g in index_to_gloss.values() if g}
    if path and path.is_file():
        try:
            return GlossBeamLM.from_json_file(path, vocab_hint=vocab)
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return GlossBeamLM.uniform_over_vocab(vocab)
