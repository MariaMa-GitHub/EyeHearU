"""
Copy raw source videos into data/processed/clips/ (no decode/trim/resize).

Mode 1: From processed_clips.csv — copy each row's source file to its clip_path.
Mode 2: If no processed_clips.csv (or --from-ingested), build clip paths from
        ingested_*.csv and copy all. Use when preprocess_clips.py fails (e.g. OpenCV).

Usage:
    cd data/scripts
    python copy_raw_to_clips.py
    python copy_raw_to_clips.py --from-ingested   # use all ingested rows, ignore processed_clips
"""

import argparse
import csv
import shutil
from pathlib import Path

from pipeline_config import PROCESSED_DIR, CLIPS_DIR, SOURCES


def load_src_by_clip_id() -> dict[str, str]:
    """Build mapping clip_id -> src_path from all ingested_*.csv that exist."""
    out = {}
    for source in SOURCES:
        path = PROCESSED_DIR / f"ingested_{source}.csv"
        if not path.exists():
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                cid = row.get("clip_id", "")
                src = row.get("src_path", "")
                if cid and src and Path(src).exists():
                    out[cid] = src
    return out


def main():
    ap = argparse.ArgumentParser(description="Copy raw videos to data/processed/clips/")
    ap.add_argument("--from-ingested", action="store_true", help="Build list from ingested CSVs only (ignore processed_clips.csv)")
    args = ap.parse_args()

    print("=" * 60)
    print("Copy raw videos → data/processed/clips/")
    print("=" * 60)

    src_map = load_src_by_clip_id()
    print(f"  Loaded src_path for {len(src_map)} clip_ids from ingested CSVs.")

    if args.from_ingested or not (PROCESSED_DIR / "processed_clips.csv").exists():
        rows = []
        for source in SOURCES:
            path = PROCESSED_DIR / f"ingested_{source}.csv"
            if not path.exists():
                continue
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    clip_id = row.get("clip_id", "")
                    gloss = row.get("gloss", "").strip()
                    split = (row.get("split") or "train").strip().lower()
                    if not clip_id or not gloss:
                        continue
                    dest = CLIPS_DIR / split / gloss / f"{clip_id}.mp4"
                    rows.append({"clip_id": clip_id, "clip_path": str(dest.resolve())})
        print(f"  Using --from-ingested: {len(rows)} target paths from ingested CSVs.")
    else:
        with open(PROCESSED_DIR / "processed_clips.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        print(f"  Using processed_clips.csv: {len(rows)} rows.")

    ok, skip = 0, 0
    for r in rows:
        clip_path_str = r.get("clip_path", "")
        clip_id = r.get("clip_id", "")
        if not clip_path_str or not clip_id:
            skip += 1
            continue
        src = src_map.get(clip_id)
        if not src:
            skip += 1
            continue
        dest = Path(clip_path_str)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dest)
            ok += 1
        except Exception as e:
            print(f"  [fail] {clip_id}: {e}")
            skip += 1

    print(f"\n[copy] {ok} files copied to clips/, {skip} skipped.")
    print(f"  Clips dir: {PROCESSED_DIR / 'clips'}")


if __name__ == "__main__":
    main()
