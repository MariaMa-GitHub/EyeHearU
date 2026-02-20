"""
Ingestion script for WLASL — supplementary training data.

WLASL metadata: https://github.com/dxli94/WLASL
Pre-downloaded videos should be placed in:
    data/raw/wlasl/videos/       ← .mp4 files named by video_id

This script:
  1. Downloads the WLASL v0.3 JSON annotation file (if not cached).
  2. Parses it into normalised records with temporal boundaries.
  3. Validates video availability.
  4. Writes  ingested_wlasl.csv  for downstream processing.
"""

import csv
import json
import sys
from pathlib import Path

import requests

from pipeline_config import WLASL_RAW, WLASL_JSON_URL, PROCESSED_DIR


def download_metadata() -> list:
    WLASL_RAW.mkdir(parents=True, exist_ok=True)
    meta_path = WLASL_RAW / "WLASL_v0.3.json"

    if not meta_path.exists():
        print("[wlasl] Downloading annotation JSON …")
        resp = requests.get(WLASL_JSON_URL, timeout=30)
        resp.raise_for_status()
        with open(meta_path, "w") as f:
            json.dump(resp.json(), f, indent=2)

    with open(meta_path) as f:
        return json.load(f)


def parse_records(raw_meta: list) -> list[dict]:
    """
    WLASL JSON structure:
      [ { "gloss": "book",
          "instances": [ { "video_id": "69241", "split": "train",
                           "frame_start": 0, "frame_end": -1, ... }, ... ]
        }, ... ]
    """
    records = []
    video_dir = WLASL_RAW / "videos"

    for entry in raw_meta:
        gloss = entry.get("gloss", "").strip().lower()
        for inst in entry.get("instances", []):
            vid = inst.get("video_id", "")
            src = video_dir / f"{vid}.mp4"
            records.append({
                "clip_id": f"wlasl_{vid}",
                "gloss": gloss,
                "signer_id": str(inst.get("signer_id", "")),
                "split": inst.get("split", "train").strip().lower(),
                "frame_start": inst.get("frame_start", 0),
                "frame_end": inst.get("frame_end", -1),
                "src_path": str(src),
            })
    return records


def validate_videos(records: list[dict]) -> list[dict]:
    valid = [r for r in records if Path(r["src_path"]).exists()]
    print(f"[wlasl] {len(valid)}/{len(records)} videos found on disk.")
    return valid


def write_ingested_csv(records: list[dict]):
    out_path = PROCESSED_DIR / "ingested_wlasl.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "clip_id", "gloss", "signer_id", "split", "source",
        "frame_start", "frame_end", "src_path",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({**r, "source": "wlasl"})

    print(f"[wlasl] Wrote {len(records)} records → {out_path}")


def main():
    print("=" * 60)
    print("WLASL Ingestion")
    print("=" * 60)

    raw_meta = download_metadata()
    records = parse_records(raw_meta)
    print(f"  Parsed {len(records)} instances across {len(raw_meta)} glosses.")

    records = validate_videos(records)
    write_ingested_csv(records)


if __name__ == "__main__":
    main()
