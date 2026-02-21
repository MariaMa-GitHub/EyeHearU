"""
Ingestion script for WLASL — supplementary training data.

WLASL metadata: https://github.com/dxli94/WLASL
Pre-downloaded videos should be placed in:
    data/raw/wlasl/videos/       ← .mp4 files named by video_id

This script:
  1. Downloads the WLASL v0.3 JSON annotation file (if not cached).
  2. Parses it into normalised records with temporal boundaries (converted to
     start_time / end_time in seconds, same output format as ingest_msasl).
  3. Validates video availability.
  4. Writes  ingested_wlasl.csv  for downstream preprocess_clips (trim by time).
"""

import csv
import json
import sys
from pathlib import Path

import requests

from pipeline_config import WLASL_RAW, WLASL_JSON_URL, PROCESSED_DIR

# WLASL annotations use frame indices at 25 fps (per official README).
WLASL_FPS = 25.0


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

    We output the same schema as ingest_msasl: start_time, end_time (seconds)
    so preprocess_clips can trim by time like MS-ASL. Frames are converted
    using WLASL_FPS (25).
    """
    records = []
    video_dir = WLASL_RAW / "videos"

    for entry in raw_meta:
        gloss = entry.get("gloss", "").strip().lower()
        for inst in entry.get("instances", []):
            vid = inst.get("video_id", "")
            src = video_dir / f"{vid}.mp4"
            frame_start = int(inst.get("frame_start", 0) or 0)
            frame_end = int(inst.get("frame_end", -1) or -1)

            start_time = frame_start / WLASL_FPS
            end_time = (frame_end / WLASL_FPS) if frame_end >= 0 else -1.0

            records.append({
                "clip_id": f"wlasl_{vid}",
                "gloss": gloss,
                "signer_id": str(inst.get("signer_id", "")),
                "split": inst.get("split", "train").strip().lower(),
                "start_time": start_time,
                "end_time": end_time,
                "src_path": str(src),
            })
    return records


def validate_videos(records: list[dict]) -> list[dict]:
    valid = [r for r in records if Path(r["src_path"]).exists()]
    print(f"[wlasl] {len(valid)}/{len(records)} videos found on disk.")
    return valid


def write_ingested_csv(records: list[dict]):
    """Write CSV with same columns as ingested_msasl (start_time, end_time, src_path)."""
    out_path = PROCESSED_DIR / "ingested_wlasl.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "clip_id", "gloss", "signer_id", "split", "source",
        "start_time", "end_time", "src_path",
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
