"""
Build the unified dataset from processed clips.

Reads  processed_clips.csv  (output of preprocess_clips.py) and produces:
  - label_map.json     gloss -> integer index
  - dataset_stats.json per-split and per-class counts
  - Prints a summary table

With PIPELINE_ENV=dev and --mvp: reads processed_clips.csv from S3 processed/mvp/
and writes label_map.json and dataset_stats.json to S3 (so the pipeline can
continue entirely from S3).

Usage:
    python build_unified_dataset.py
    PIPELINE_ENV=dev python build_unified_dataset.py --mvp   # read/write S3
"""

import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from pipeline_config import (
    PROCESSED_DIR, MIN_VIDEOS_PER_GLOSS,
    get_processed_base, get_processed_prefix_s3,
    is_cloud, S3_BUCKET,
    read_text_from_s3, write_text_to_s3,
)


def load_processed_csv(processed_dir: Path | None = None) -> list[dict]:
    base = processed_dir or PROCESSED_DIR
    csv_path = base / "processed_clips.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run preprocess_clips.py first."
        )
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_processed_csv_from_s3(mvp: bool = True) -> list[dict]:
    """Load processed_clips.csv from S3 (e.g. processed/mvp/processed_clips.csv)."""
    prefix = get_processed_prefix_s3(mvp)
    key = f"{prefix}/processed_clips.csv"
    text = read_text_from_s3(key)
    return list(csv.DictReader(io.StringIO(text)))


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
          f"{before} -> {after} glosses.")
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
    import argparse
    parser = argparse.ArgumentParser(description="Build label map and dataset stats from processed clips.")
    parser.add_argument("--mvp", action="store_true", help="Read/write under processed/mvp/ (MVP-filtered run).")
    args = parser.parse_args()

    use_s3 = is_cloud() and args.mvp
    processed_dir = get_processed_base(args.mvp)
    prefix = get_processed_prefix_s3(True) if use_s3 else None

    if args.mvp:
        print(f"[build] MVP mode: using {processed_dir}" + (" (S3)" if use_s3 else ""))
    if use_s3:
        print(f"[build] Reading from s3://{S3_BUCKET}/{prefix}/processed_clips.csv")

    print("=" * 60)
    print("Build Unified Dataset")
    print("=" * 60)

    if use_s3:
        records = load_processed_csv_from_s3(mvp=True)
    else:
        records = load_processed_csv(processed_dir)
    print(f"  Loaded {len(records)} processed clips.")

    records = filter_rare_glosses(records)

    label_map = build_label_map(records)
    stats = compute_stats(records, label_map)

    if use_s3:
        write_text_to_s3(json.dumps(label_map, indent=2), f"{prefix}/label_map.json")
        print(f"  Label map ({len(label_map)} classes) -> s3://{S3_BUCKET}/{prefix}/label_map.json")
        write_text_to_s3(json.dumps(stats, indent=2), f"{prefix}/dataset_stats.json")
        print(f"  Dataset stats -> s3://{S3_BUCKET}/{prefix}/dataset_stats.json")
    else:
        label_path = processed_dir / "label_map.json"
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2)
        print(f"  Label map ({len(label_map)} classes) -> {label_path}")
        stats_path = processed_dir / "dataset_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"  Dataset stats -> {stats_path}")

    print(f"\n  Classes:  {stats['num_classes']}")
    print(f"  Total:    {stats['total_clips']}")
    for split, count in sorted(stats["splits"].items()):
        print(f"    {split:6s}: {count}")
    print(f"  Sources:")
    for src, count in sorted(stats["source_breakdown"].items()):
        print(f"    {src:15s}: {count}")


if __name__ == "__main__":
    main()
