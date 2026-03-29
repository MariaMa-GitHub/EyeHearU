"""
Beam search over per-clip top-k gloss hypotheses with a gloss n-gram LM.

Score(sequence) = sum_t log P_model(g_t | clip_t)
                  + lm_weight * sum_t log P_LM(g_t | g_{t-2}, g_{t-1})

First step: g_{-2} = g_{-1} = START_TOKEN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.services.gloss_lm import START_TOKEN, GlossBeamLM, GlossBigramLM

LOG_EPS = 1e-12


def _model_logp(confidence: float) -> float:
    c = max(float(confidence), LOG_EPS)
    return math.log(min(c, 1.0))


@dataclass(frozen=True)
class BeamPath:
    score: float
    glosses: tuple[str, ...]


def beam_search(
    candidates_per_clip: list[list[dict]],
    lm: GlossBeamLM | GlossBigramLM,
    *,
    beam_size: int = 8,
    lm_weight: float = 1.0,
    top_sequences: int = 5,
) -> list[BeamPath]:
    """
    Args:
        candidates_per_clip: For each clip, a list of dicts with keys "sign", "confidence"
        lm: Language model with ``log_p_step(prev2, prev1, next)``
        beam_size: Max partial hypotheses to keep after each clip
        lm_weight: Strength of LM vs model (typical 0.5–2.0)
        top_sequences: How many full sequences to return (sorted by score desc)

    Returns:
        Best full sequences with total log scores (higher is better).

    Raises:
        ValueError: If any clip has an empty candidate list.
    """
    if not candidates_per_clip:
        return []

    for clip_idx, cands in enumerate(candidates_per_clip):
        if not cands:
            raise ValueError(
                f"Empty top-k hypotheses for clip {clip_idx + 1} of {len(candidates_per_clip)}; "
                "cannot run beam search. Each clip must have at least one classifier candidate."
            )

    beam: list[tuple[float, tuple[str, ...], str, str]] = []
    # (total_score, path, prev2, prev1) — context for the *next* LM step
    for cand in candidates_per_clip[0]:
        g = cand["sign"]
        mlp = _model_logp(float(cand["confidence"]))
        lmp = lm.log_p_step(START_TOKEN, START_TOKEN, g)
        score = mlp + lm_weight * lmp
        beam.append((score, (g,), START_TOKEN, g))

    beam.sort(key=lambda x: -x[0])
    beam = beam[: max(beam_size, 1)]

    for step in range(1, len(candidates_per_clip)):
        next_beam: list[tuple[float, tuple[str, ...], str, str]] = []
        for prev_score, path, prev2, prev1 in beam:
            for cand in candidates_per_clip[step]:
                g = cand["sign"]
                mlp = _model_logp(float(cand["confidence"]))
                lmp = lm.log_p_step(prev2, prev1, g)
                score = prev_score + mlp + lm_weight * lmp
                next_beam.append((score, path + (g,), prev1, g))
        next_beam.sort(key=lambda x: -x[0])
        beam = next_beam[: max(beam_size, 1)]

    final = [BeamPath(score=s, glosses=p) for s, p, _, _ in beam]
    best: dict[tuple[str, ...], float] = {}
    for bp in final:
        best[bp.glosses] = max(best.get(bp.glosses, float("-inf")), bp.score)
    out = [BeamPath(score=v, glosses=k) for k, v in best.items()]
    out.sort(key=lambda x: -x.score)
    return out[: max(top_sequences, 1)]
