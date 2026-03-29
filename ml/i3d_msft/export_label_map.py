"""
Export gloss <-> class index mapping for I3D inference (no training required).

The checkpoint does not store label names. Class order matches training:
  sorted(unique glosses from the train split CSV), lowercased — same as
  ASLCitizenI3DDataset and evaluate.py::_build_gloss_dict_from_csv.

Use the *same* CSV you used for training (usually filtered_splits/.../train.csv),
or the plan's train.csv if your run used the full split with no extra drops.

Example:
  python -m i3d_msft.export_label_map \\
    --csv path/to/train.csv \\
    --output i3d_label_map.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_gloss_dict_from_csv(path: Path) -> dict[str, int]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    glosses = sorted(
        {
            (r.get("gloss") or "").strip().lower()
            for r in rows
            if (r.get("gloss") or "").strip()
        }
    )
    return {g: i for i, g in enumerate(glosses)}


def main() -> None:
    p = argparse.ArgumentParser(description="Export I3D gloss->index JSON from train.csv")
    p.add_argument("--csv", required=True, type=Path, help="Train split CSV (user,filename,gloss)")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("i3d_label_map.json"),
        help="Output JSON path",
    )
    p.add_argument(
        "--inverse",
        action="store_true",
        help="Also write index_to_gloss list in the JSON",
    )
    args = p.parse_args()

    path = args.csv.resolve()
    gloss_to_idx = build_gloss_dict_from_csv(path)
    payload: dict = {
        "source_csv": str(path),
        "num_classes": len(gloss_to_idx),
        "gloss_to_index": gloss_to_idx,
    }
    if args.inverse:
        inv = [None] * len(gloss_to_idx)
        for g, i in gloss_to_idx.items():
            inv[i] = g
        payload["index_to_gloss"] = inv

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[export] wrote {len(gloss_to_idx)} classes -> {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
