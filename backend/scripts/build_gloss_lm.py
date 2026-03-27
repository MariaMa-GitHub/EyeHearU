#!/usr/bin/env python3
"""
Build backend/data/gloss_lm.json from the I3D label map and optional gloss sequences.

Usage (from repository root)::

  cd backend
  PYTHONPATH=. python scripts/build_gloss_lm.py \\
    --label-map ../ml/i3d_label_map_mvp-sft-full-v1.json \\
    --out data/gloss_lm.json

  # Optional: add bigram/trigram stats from one gloss per line (whitespace-separated)
  PYTHONPATH=. python scripts/build_gloss_lm.py \\
    --label-map ../ml/i3d_label_map_mvp-sft-full-v1.json \\
    --sequences path/to/gloss_sentences.txt \\
    --out data/gloss_lm.json

The API ``english`` field is produced by joining ``best_glosses`` with light polish only
(see ``app.services.gloss_to_english``), not a separate lexicon file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.lm_builder import (
    build_lm_dict,
    load_glosses_from_label_map,
    parse_sequences_file,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build gloss LM JSON from label map + sequences.")
    ap.add_argument(
        "--label-map",
        type=Path,
        required=True,
        help="Path to i3d_label_map_*.json (gloss_to_index).",
    )
    ap.add_argument(
        "--sequences",
        type=Path,
        default=None,
        help="Optional text file: one sentence per line, whitespace-separated glosses.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output gloss_lm.json path.",
    )
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--unigram-floor", type=int, default=5)
    ap.add_argument("--start-bigram-floor", type=int, default=3)
    args = ap.parse_args()

    glosses = load_glosses_from_label_map(args.label_map)
    seqs = parse_sequences_file(args.sequences) if args.sequences else None
    payload = build_lm_dict(
        glosses,
        seqs,
        alpha=args.alpha,
        unigram_floor=args.unigram_floor,
        start_bigram_floor=args.start_bigram_floor,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[build_gloss_lm] Wrote {args.out} ({len(glosses)} labels, {len(seqs or [])} sequence lines).")


if __name__ == "__main__":
    main()
