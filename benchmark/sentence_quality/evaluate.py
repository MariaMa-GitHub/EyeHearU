from __future__ import annotations

import argparse
import csv
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


def parse_glosses(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            data = json.loads(raw)
            return [str(x).strip() for x in data if str(x).strip()]
        except Exception:
            pass
    sep = "|" if "|" in raw else " "
    return [x.strip() for x in raw.split(sep) if x.strip()]


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            clean = dict(row)
            extra = clean.pop(None, None)
            # If a CSV row had unquoted commas, DictReader puts overflow here.
            # Stitch overflow back into reference so evaluation can proceed.
            if extra:
                ref = (clean.get("reference") or "").strip()
                suffix = ", ".join(str(x).strip() for x in extra if str(x).strip())
                clean["reference"] = f"{ref}, {suffix}".strip(", ").strip()
            rows.append(clean)
        return rows


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def generate_predictions(input_csv: Path, output_csv: Path) -> None:
    # Lazy imports so scoring-only runs do not require heavy deps.
    from app.config import get_settings
    from app.services.gloss_to_english import gloss_sequence_to_english as rule_fn
    from app.services.gloss_to_english_bedrock import (
        gloss_sequence_to_english_bedrock as bedrock_fn,
    )
    from app.services.gloss_to_english_t5 import gloss_sequence_to_english_t5 as t5_fn

    settings = get_settings()
    rows = load_rows(input_csv)
    out = []
    for r in rows:
        glosses = parse_glosses(r.get("glosses", ""))
        row = dict(r)
        row["rule"] = rule_fn(glosses)
        row["t5"] = t5_fn(glosses)
        try:
            row["bedrock"] = bedrock_fn(
                glosses,
                region=settings.bedrock_region,
                model_id=settings.bedrock_model_id,
                timeout_s=settings.bedrock_timeout_s,
            )
        except Exception as e:
            row["bedrock"] = f"[BEDROCK_ERROR] {e}"
        out.append(row)

    fields = list(out[0].keys()) if out else ["case_id", "glosses", "reference", "rule", "t5", "bedrock"]
    write_rows(output_csv, out, fields)
    print(f"[ok] Wrote predictions: {output_csv}")


def score_predictions(pred_csv: Path, out_json: Path) -> None:
    rows = load_rows(pred_csv)
    systems = ["rule", "t5", "bedrock"]
    details = []
    agg = {s: [] for s in systems}

    for r in rows:
        ref = r.get("reference", "")
        row_detail = {
            "case_id": r.get("case_id", ""),
            "glosses": r.get("glosses", ""),
            "reference": ref,
            "systems": {},
        }
        for s in systems:
            pred = r.get(s, "")
            p, rc, f1 = token_precision_recall_f1(pred, ref)
            metrics = {
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
            row_detail["systems"][s] = {"prediction": pred, "metrics": metrics}
            agg[s].append(metrics)
        details.append(row_detail)

    summary = {}
    for s in systems:
        items = agg[s]
        summary[s] = {
            k: round(mean([m[k] for m in items]), 4) if items else 0.0
            for k in (
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
        }

    out = {"count": len(rows), "summary": summary, "details": details}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[ok] Wrote metrics: {out_json}")
    print("\n=== Aggregate (higher is better) ===")
    for s in systems:
        m = summary[s]
        print(
            f"{s:8} "
            f"F1={m['token_f1']:.3f} "
            f"BLEU={m['sentence_bleu']:.3f} "
            f"ROUGE-L={m['rougeL_recall']:.3f} "
            f"Cap={m['capitalization_ok']:.3f} "
            f"Punct={m['punctuation_ok']:.3f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Sentence quality benchmark for rule/t5/bedrock")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Generate predictions for all systems")
    p_gen.add_argument("--input", required=True, type=Path, help="CSV with case_id,glosses,reference")
    p_gen.add_argument("--output", required=True, type=Path, help="CSV to write predictions")

    p_score = sub.add_parser("score", help="Score existing predictions")
    p_score.add_argument("--predictions", required=True, type=Path, help="CSV with rule,t5,bedrock,reference")
    p_score.add_argument("--out", required=True, type=Path, help="JSON report path")

    args = p.parse_args()
    if args.cmd == "generate":
        generate_predictions(args.input, args.output)
    elif args.cmd == "score":
        score_predictions(args.predictions, args.out)


if __name__ == "__main__":
    main()

