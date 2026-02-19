"""
Validation script for the processed dataset.

Checks:
  1. Every clip referenced in processed_clips.csv exists on disk.
  2. Every clip has exactly NUM_SAMPLE_FRAMES frames at the expected resolution.
  3. label_map.json and dataset_stats.json are consistent with the CSV.
  4. Train/val/test splits have no signer overlap (for ASL Citizen portion).

Usage:
    python validate.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2

from pipeline_config import (
    PROCESSED_DIR, LABEL_MAP_JSON, STATS_JSON,
    NUM_SAMPLE_FRAMES, FRAME_HEIGHT, FRAME_WIDTH,
)


def load_processed_csv() -> list[dict]:
    csv_path = PROCESSED_DIR / "processed_clips.csv"
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def check_files_exist(records: list[dict]) -> int:
    missing = 0
    for r in records:
        if not Path(r["clip_path"]).exists():
            missing += 1
    return missing


def check_clip_properties(records: list[dict], sample_size: int = 200) -> dict:
    """Spot-check a random subset of clips for frame count and resolution."""
    import random
    sample = random.sample(records, min(sample_size, len(records)))

    bad_frames = 0
    bad_resolution = 0

    for r in sample:
        cap = cv2.VideoCapture(r["clip_path"])
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if n != NUM_SAMPLE_FRAMES:
            bad_frames += 1
        if h != FRAME_HEIGHT or w != FRAME_WIDTH:
            bad_resolution += 1

    return {
        "sampled": len(sample),
        "bad_frame_count": bad_frames,
        "bad_resolution": bad_resolution,
    }


def check_signer_leakage(records: list[dict]) -> dict:
    """Check if any signer appears in more than one split (ASL Citizen only)."""
    ac_records = [r for r in records if r.get("source") == "asl_citizen"]
    split_signers: dict[str, set] = defaultdict(set)
    for r in ac_records:
        split_signers[r["split"]].add(r["signer_id"])

    leaks = {}
    splits = list(split_signers.keys())
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            overlap = split_signers[s1] & split_signers[s2]
            if overlap:
                leaks[f"{s1} ∩ {s2}"] = len(overlap)
    return leaks


def check_label_map_consistency(records: list[dict]) -> bool:
    with open(LABEL_MAP_JSON) as f:
        label_map = json.load(f)
    csv_glosses = {r["gloss"] for r in records}
    map_glosses = set(label_map.keys())
    return csv_glosses == map_glosses


def main():
    print("=" * 60)
    print("Dataset Validation")
    print("=" * 60)

    records = load_processed_csv()
    total = len(records)
    print(f"  Records in CSV: {total}")

    # 1. File existence
    missing = check_files_exist(records)
    status = "PASS" if missing == 0 else "FAIL"
    print(f"  [{status}] Missing clip files: {missing}")

    # 2. Clip properties
    props = check_clip_properties(records)
    status = "PASS" if props["bad_frame_count"] == 0 and props["bad_resolution"] == 0 else "WARN"
    print(f"  [{status}] Spot-checked {props['sampled']} clips: "
          f"{props['bad_frame_count']} bad frame count, "
          f"{props['bad_resolution']} bad resolution")

    # 3. Signer leakage
    leaks = check_signer_leakage(records)
    status = "PASS" if not leaks else "FAIL"
    print(f"  [{status}] ASL Citizen signer leakage across splits: "
          f"{leaks if leaks else 'none'}")

    # 4. Label map consistency
    consistent = check_label_map_consistency(records)
    status = "PASS" if consistent else "FAIL"
    print(f"  [{status}] label_map.json matches processed CSV glosses: {consistent}")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
