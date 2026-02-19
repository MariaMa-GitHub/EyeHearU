"""
Ingestion script for ASL Citizen — the primary training dataset.

ASL Citizen is distributed by Microsoft Research:
  https://www.microsoft.com/en-us/research/project/asl-citizen/

Download steps (manual — requires acceptance of data-use agreement):
  1. Visit the link above and request access.
  2. Download the dataset archive.
  3. Extract into  data/raw/asl_citizen/  with the layout:
       data/raw/asl_citizen/
         videos/           ← all .mp4 clip files
         metadata.json     ← or .csv with columns: filename, gloss, split, user_id

This script:
  1. Reads the ASL Citizen metadata.
  2. Validates that expected video files exist on disk.
  3. Writes a normalised  ingested_asl_citizen.csv  ready for the build step.
"""

import csv
import json
import sys
from pathlib import Path

from pipeline_config import ASL_CITIZEN_RAW, PROCESSED_DIR


def load_metadata() -> list[dict]:
    """
    Parse ASL Citizen metadata into a list of records.

    The dataset ships with a JSON (or CSV) manifest.  We normalise into:
      {clip_id, gloss, signer_id, split, src_path}
    """
    json_path = ASL_CITIZEN_RAW / "metadata.json"
    csv_path = ASL_CITIZEN_RAW / "metadata.csv"

    records: list[dict] = []

    if json_path.exists():
        with open(json_path) as f:
            raw = json.load(f)
        for entry in raw:
            records.append({
                "clip_id": entry.get("filename", entry.get("video_id", "")),
                "gloss": entry.get("gloss", "").strip().lower(),
                "signer_id": str(entry.get("user_id", entry.get("signer_id", ""))),
                "split": entry.get("split", "train").strip().lower(),
                "src_path": str(ASL_CITIZEN_RAW / "videos" / entry.get("filename", "")),
            })

    elif csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("filename", row.get("video_id", ""))
                records.append({
                    "clip_id": fname,
                    "gloss": row.get("gloss", "").strip().lower(),
                    "signer_id": str(row.get("user_id", row.get("signer_id", ""))),
                    "split": row.get("split", "train").strip().lower(),
                    "src_path": str(ASL_CITIZEN_RAW / "videos" / fname),
                })
    else:
        print("[ERROR] No metadata.json or metadata.csv found in", ASL_CITIZEN_RAW)
        sys.exit(1)

    return records


def validate_videos(records: list[dict]) -> list[dict]:
    """Keep only records whose video file actually exists on disk."""
    valid = []
    missing = 0
    for r in records:
        if Path(r["src_path"]).exists():
            valid.append(r)
        else:
            missing += 1
    print(f"[asl_citizen] {len(valid)} videos found, {missing} missing on disk.")
    return valid


def write_ingested_csv(records: list[dict]):
    """Write a normalised CSV to the processed directory for downstream use."""
    out_path = PROCESSED_DIR / "ingested_asl_citizen.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = ["clip_id", "gloss", "signer_id", "split", "source", "src_path"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({**r, "source": "asl_citizen"})

    print(f"[asl_citizen] Wrote {len(records)} records → {out_path}")


def main():
    print("=" * 60)
    print("ASL Citizen Ingestion")
    print("=" * 60)

    records = load_metadata()
    print(f"  Parsed {len(records)} metadata entries.")
    print(f"  Unique glosses: {len({r['gloss'] for r in records})}")
    print(f"  Unique signers: {len({r['signer_id'] for r in records})}")

    records = validate_videos(records)
    write_ingested_csv(records)


if __name__ == "__main__":
    main()
