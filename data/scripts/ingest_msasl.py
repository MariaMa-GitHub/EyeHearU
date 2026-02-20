"""
Ingestion script for MS-ASL — supplementary training data.

MS-ASL metadata: https://github.com/AlanTatworked/MS-ASL
Pre-downloaded videos should be placed in:
    data/raw/msasl/videos/       ← .mp4 files named by video filename or URL hash

This script:
  1. Downloads the MS-ASL class list and split JSONs (if not cached).
  2. Parses them into normalised records with temporal boundaries.
  3. Validates video availability.
  4. Writes  ingested_msasl.csv  for downstream processing.
"""

import csv
import json
import sys
from pathlib import Path

import requests

from pipeline_config import (
    MSASL_RAW, PROCESSED_DIR,
    MSASL_CLASSES_URL, MSASL_TRAIN_URL, MSASL_VAL_URL, MSASL_TEST_URL,
)


def _download_json(url: str, dest: Path) -> list:
    if not dest.exists():
        print(f"[msasl] Downloading {dest.name} …")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(dest, "w") as f:
            json.dump(resp.json(), f, indent=2)
    with open(dest) as f:
        return json.load(f)


def download_metadata():
    MSASL_RAW.mkdir(parents=True, exist_ok=True)
    classes = _download_json(MSASL_CLASSES_URL, MSASL_RAW / "classes.json")
    train = _download_json(MSASL_TRAIN_URL, MSASL_RAW / "train.json")
    val = _download_json(MSASL_VAL_URL, MSASL_RAW / "val.json")
    test = _download_json(MSASL_TEST_URL, MSASL_RAW / "test.json")
    return classes, train, val, test


def parse_records(classes: list, *splits_with_name: tuple) -> list[dict]:
    """
    Each MS-ASL split JSON entry:
      { "url": "...", "start_time": 0.5, "end_time": 2.1,
        "label": 42, "signer_id": 7, "text": "book", ... }
    """
    video_dir = MSASL_RAW / "videos"
    records = []

    for split_name, entries in splits_with_name:
        for entry in entries:
            label_idx = entry.get("label", -1)
            gloss = entry.get("text", "").strip().lower()
            if not gloss and 0 <= label_idx < len(classes):
                gloss = classes[label_idx].strip().lower()

            vid_file = entry.get("file", "") or f"{entry.get('url', '').split('=')[-1]}.mp4"
            src = video_dir / vid_file

            records.append({
                "clip_id": f"msasl_{split_name}_{len(records)}",
                "gloss": gloss,
                "signer_id": str(entry.get("signer_id", "")),
                "split": split_name,
                "start_time": entry.get("start_time", 0),
                "end_time": entry.get("end_time", -1),
                "src_path": str(src),
            })

    return records


def validate_videos(records: list[dict]) -> list[dict]:
    valid = [r for r in records if Path(r["src_path"]).exists()]
    print(f"[msasl] {len(valid)}/{len(records)} videos found on disk.")
    return valid


def write_ingested_csv(records: list[dict]):
    out_path = PROCESSED_DIR / "ingested_msasl.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "clip_id", "gloss", "signer_id", "split", "source",
        "start_time", "end_time", "src_path",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({**r, "source": "msasl"})

    print(f"[msasl] Wrote {len(records)} records → {out_path}")


def main():
    print("=" * 60)
    print("MS-ASL Ingestion")
    print("=" * 60)

    classes, train, val, test = download_metadata()
    print(f"  {len(classes)} classes in MS-ASL vocabulary.")

    records = parse_records(
        classes,
        ("train", train),
        ("val", val),
        ("test", test),
    )
    print(f"  Parsed {len(records)} total instances.")

    records = validate_videos(records)
    write_ingested_csv(records)


if __name__ == "__main__":
    main()
