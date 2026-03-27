"""
Build ``gloss_lm.json`` from classifier label vocabulary + optional gloss sequences.

Used offline (no paid APIs) to scale LM coverage beyond hand-written examples.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from app.services.gloss_lm import START_TOKEN


def load_glosses_from_label_map(path: Path) -> list[str]:
    """Ordered gloss strings from ``i3d_label_map_*.json`` (by index)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    gti: dict[str, int] = raw.get("gloss_to_index") or {}
    return [g for g, _ in sorted(gti.items(), key=lambda kv: (kv[1], kv[0]))]


def parse_sequences_file(path: Path) -> list[list[str]]:
    """
    One sentence per line: whitespace-separated gloss tokens.

    Lines starting with ``#`` and empty lines are skipped.
    Unknown tokens (not in label map) are omitted when building counts.
    """
    sequences: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        sequences.append(s.split())
    return sequences


def build_lm_dict(
    glosses: list[str],
    sequences: list[list[str]] | None = None,
    *,
    alpha: float = 1.0,
    unigram_floor: int = 5,
    start_bigram_floor: int = 3,
) -> dict[str, float | dict[str, int]]:
    """
    Produce a JSON-serializable LM payload for :class:`GlossBeamLM`.

    Unigrams get a floor so every label has smoothing mass. Bigrams ``<s>|||g`` give a
    reasonable sentence-start distribution. If ``sequences`` is provided, consecutive gloss
    pairs and triples accrue extra counts from your training or silver data.
    """
    gloss_set = set(glosses)
    uni: dict[str, int] = defaultdict(int)
    uni[START_TOKEN] = max(unigram_floor, 1)
    for g in glosses:
        uni[g] += unigram_floor

    bi: dict[str, int] = defaultdict(int)
    tri: dict[str, int] = defaultdict(int)

    for g in glosses:
        bi[f"{START_TOKEN}|||{g}"] += start_bigram_floor

    if sequences:
        for seq in sequences:
            clean = [tok for tok in seq if tok in gloss_set]
            if not clean:
                continue
            g0 = clean[0]
            uni[g0] += 1
            bi[f"{START_TOKEN}|||{g0}"] += 1
            tri[f"{START_TOKEN}|||{START_TOKEN}|||{g0}"] += 1
            for i in range(1, len(clean)):
                a, b = clean[i - 1], clean[i]
                uni[b] += 1
                bi[f"{a}|||{b}"] += 1
                p2 = clean[i - 2] if i >= 2 else START_TOKEN
                tri[f"{p2}|||{a}|||{b}"] += 1

    return {
        "alpha": float(alpha),
        "unigram_counts": dict(uni),
        "bigram_counts": dict(bi),
        "trigram_counts": dict(tri),
    }
