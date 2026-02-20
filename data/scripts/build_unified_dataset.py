"""
Build the unified dataset from processed clips.

Reads  processed_clips.csv  (output of preprocess_clips.py) and produces:
  - label_map.json     gloss → integer index
  - dataset_stats.json per-split and per-class counts
  - Prints a summary table

Usage:
    python build_unified_dataset.py
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from pipeline_config import PROCESSED_DIR, LABEL_MAP_JSON, STATS_JSON, MIN_VIDEOS_PER_GLOSS


def load_processed_csv() -> list[dict]:
    csv_path = PROCESSED_DIR / "processed_clips.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run preprocess_clips.py first."
        )
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def filter_rare_glosses(records: list[dict]) -> list[dict]:
    """Remove glosses that have fewer than MIN_VIDEOS_PER_GLOSS training clips."""
    train_counts: Counter = Counter()
    for r in records:
        if r["split"] == "train":
            train_counts[r["gloss"]] += 1

    keep = {g for g, c in train_counts.items() if c >= MIN_VIDEOS_PER_GLOSS}
    before = len({r["gloss"] for r in records})
    filtered = [r for r in records if r["gloss"] in keep]
    after = len(keep)
    print(f"[build] Filtered glosses with <{MIN_VIDEOS_PER_GLOSS} training clips: "
          f"{before} → {after} glosses.")
    return filtered


def build_label_map(records: list[dict]) -> dict[str, int]:
    glosses = sorted({r["gloss"] for r in records})
    return {g: i for i, g in enumerate(glosses)}


def compute_stats(records: list[dict], label_map: dict) -> dict:
    split_counts: dict[str, int] = Counter()
    class_split_counts: dict[str, dict[str, int]] = defaultdict(Counter)
    source_counts: Counter = Counter()

    for r in records:
        split_counts[r["split"]] += 1
        class_split_counts[r["gloss"]][r["split"]] += 1
        source_counts[r["source"]] += 1

    return {
        "num_classes": len(label_map),
        "total_clips": len(records),
        "splits": dict(split_counts),
        "source_breakdown": dict(source_counts),
        "per_class": {
            g: dict(class_split_counts[g])
            for g in sorted(class_split_counts)
        },
    }


def main():
    print("=" * 60)
    print("Build Unified Dataset")
    print("=" * 60)

    records = load_processed_csv()
    print(f"  Loaded {len(records)} processed clips.")

    records = filter_rare_glosses(records)

    label_map = build_label_map(records)
    with open(LABEL_MAP_JSON, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"  Label map ({len(label_map)} classes) → {LABEL_MAP_JSON}")

    stats = compute_stats(records, label_map)
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Dataset stats → {STATS_JSON}")

    print(f"\n  Classes:  {stats['num_classes']}")
    print(f"  Total:    {stats['total_clips']}")
    for split, count in sorted(stats["splits"].items()):
        print(f"    {split:6s}: {count}")
    print(f"  Sources:")
    for src, count in sorted(stats["source_breakdown"].items()):
        print(f"    {src:15s}: {count}")


if __name__ == "__main__":
    main()
