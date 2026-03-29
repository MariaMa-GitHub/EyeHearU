from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from metrics import (
    bleu_n_precision,
    capitalization_ok,
    exact_match,
    punctuation_ok,
    rouge_l_recall,
    sentence_bleu,
    token_precision_recall_f1,
)


SYSTEMS = (
    "greedy_rule",
    "greedy_t5",
    "greedy_bedrock",
    "beam_rule",
    "beam_t5",
    "beam_bedrock",
)


def _clean_candidates(raw: list) -> list[list[dict]]:
    out: list[list[dict]] = []
    for clip in raw:
        clip_out = []
        for c in clip:
            sign = str(c.get("sign", "")).strip()
            if not sign:
                continue
            conf = float(c.get("confidence", 0.0))
            clip_out.append({"sign": sign, "confidence": conf})
        if clip_out:
            clip_out.sort(key=lambda x: x["confidence"], reverse=True)
            out.append(clip_out)
    return out


def _greedy_glosses(candidates_per_clip: list[list[dict]]) -> list[str]:
    return [clip[0]["sign"] for clip in candidates_per_clip if clip]


def _beam_glosses(candidates_per_clip: list[list[dict]], lm, *, beam_size: int, lm_weight: float) -> list[str]:
    from app.services.beam_search import beam_search

    beams = beam_search(
        candidates_per_clip,
        lm,
        beam_size=beam_size,
        lm_weight=lm_weight,
        top_sequences=1,
    )
    if not beams:
        return _greedy_glosses(candidates_per_clip)
    return list(beams[0].glosses)


def _default_lm_path() -> Path:
    return Path(__file__).resolve().parent / "gloss_lm_ablation.json"


def generate_predictions(
    input_json: Path,
    output_json: Path,
    *,
    beam_size: int,
    lm_weight: float,
    lm_json: Path | None,
) -> None:
    from app.config import get_settings
    from app.services.gloss_lm import load_gloss_lm
    from app.services.gloss_to_english import gloss_sequence_to_english as rule_fn
    from app.services.gloss_to_english_bedrock import gloss_sequence_to_english_bedrock as bedrock_fn
    from app.services.gloss_to_english_t5 import gloss_sequence_to_english_t5 as t5_fn

    settings = get_settings()
    data = json.loads(input_json.read_text(encoding="utf-8"))
    out_rows = []

    for item in data:
        case_id = str(item.get("case_id", ""))
        reference = str(item.get("reference", ""))
        candidates = _clean_candidates(item.get("candidates", []))
        if not candidates:
            continue

        # Build vocab hint from candidate signs for safe LM fallback.
        vocab = {c["sign"] for clip in candidates for c in clip}
        fake_idx = {i: g for i, g in enumerate(sorted(vocab))}

        repo_backend = Path(__file__).resolve().parents[2] / "backend"
        lm_path = lm_json if lm_json is not None else _default_lm_path()
        if not lm_path.is_file():
            lm_path = repo_backend / settings.gloss_lm_path
        lm = load_gloss_lm(lm_path if lm_path.is_file() else None, fake_idx)

        greedy_glosses = _greedy_glosses(candidates)
        beam_glosses = _beam_glosses(candidates, lm, beam_size=beam_size, lm_weight=lm_weight)

        def bedrock_safe(glosses: list[str]) -> str:
            try:
                return bedrock_fn(
                    glosses,
                    region=settings.bedrock_region,
                    model_id=settings.bedrock_model_id,
                    timeout_s=settings.bedrock_timeout_s,
                )
            except Exception as e:
                return f"[BEDROCK_ERROR] {e}"

        row = {
            "case_id": case_id,
            "reference": reference,
            "greedy_glosses": greedy_glosses,
            "beam_glosses": beam_glosses,
            "greedy_rule": rule_fn(greedy_glosses),
            "greedy_t5": t5_fn(greedy_glosses),
            "greedy_bedrock": bedrock_safe(greedy_glosses),
            "beam_rule": rule_fn(beam_glosses),
            "beam_t5": t5_fn(beam_glosses),
            "beam_bedrock": bedrock_safe(beam_glosses),
        }
        out_rows.append(row)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out_rows, indent=2), encoding="utf-8")
    print(f"[ok] Wrote ablation predictions: {output_json}")


def score_predictions(pred_json: Path, out_json: Path) -> None:
    rows = json.loads(pred_json.read_text(encoding="utf-8"))
    agg = {s: [] for s in SYSTEMS}
    details = []

    for r in rows:
        ref = r.get("reference", "")
        detail = {
            "case_id": r.get("case_id", ""),
            "reference": ref,
            "greedy_glosses": r.get("greedy_glosses", []),
            "beam_glosses": r.get("beam_glosses", []),
            "systems": {},
        }
        for s in SYSTEMS:
            pred = r.get(s, "")
            p, rc, f1 = token_precision_recall_f1(pred, ref)
            m = {
                "exact_match": exact_match(pred, ref),
                "token_precision": p,
                "token_recall": rc,
                "token_f1": f1,
                "bleu1_precision": bleu_n_precision(pred, ref, 1),
                "bleu4_precision": bleu_n_precision(pred, ref, 4),
                "sentence_bleu": sentence_bleu(pred, ref, 4),
                "rougeL_recall": rouge_l_recall(pred, ref),
                "capitalization_ok": capitalization_ok(pred),
                "punctuation_ok": punctuation_ok(pred),
            }
            detail["systems"][s] = {"prediction": pred, "metrics": m}
            agg[s].append(m)
        details.append(detail)

    summary = {}
    keys = (
        "exact_match",
        "token_precision",
        "token_recall",
        "token_f1",
        "bleu1_precision",
        "bleu4_precision",
        "sentence_bleu",
        "rougeL_recall",
        "capitalization_ok",
        "punctuation_ok",
    )
    for s in SYSTEMS:
        items = agg[s]
        summary[s] = {k: round(mean([m[k] for m in items]), 4) if items else 0.0 for k in keys}

    out = {"count": len(rows), "summary": summary, "details": details}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[ok] Wrote ablation metrics: {out_json}")
    print("\n=== Ablation aggregate (higher is better) ===")
    for s in SYSTEMS:
        m = summary[s]
        print(
            f"{s:14} "
            f"F1={m['token_f1']:.3f} "
            f"BLEU={m['sentence_bleu']:.3f} "
            f"ROUGE-L={m['rougeL_recall']:.3f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Ablation: beam vs greedy + rewrite models")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate predictions for ablation")
    g.add_argument("--input", required=True, type=Path, help="JSON list with case_id,candidates,reference")
    g.add_argument("--output", required=True, type=Path, help="Output JSON predictions")
    g.add_argument("--beam-size", type=int, default=8)
    g.add_argument("--lm-weight", type=float, default=2.0)
    g.add_argument(
        "--lm-json",
        type=Path,
        default=None,
        help="Peaked gloss LM JSON for ablation (default: benchmark/sentence_quality/gloss_lm_ablation.json)",
    )

    s = sub.add_parser("score", help="Score ablation predictions")
    s.add_argument("--predictions", required=True, type=Path, help="Prediction JSON path")
    s.add_argument("--out", required=True, type=Path, help="Metrics JSON path")

    args = p.parse_args()
    if args.cmd == "generate":
        generate_predictions(
            args.input,
            args.output,
            beam_size=args.beam_size,
            lm_weight=args.lm_weight,
            lm_json=args.lm_json,
        )
    else:
        score_predictions(args.predictions, args.out)


if __name__ == "__main__":
    main()

