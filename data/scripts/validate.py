"""
Validation script for the processed dataset (Stage 4).

Checks:
  1. Every clip referenced in processed_clips.csv exists (local disk or S3).
  2. Spot-check frame count & resolution (local only; S3 skips or samples a few).
  3. label_map.json consistent with CSV glosses.
  4. Train/val/test signer-disjoint for ASL Citizen.
  5. Optionally publish pass/fail to CloudWatch (when PIPELINE_ENV set).

Usage:
    python validate.py
    python validate.py --mvp
    PIPELINE_ENV=dev python validate.py --mvp   # validate S3 processed/mvp/
"""

import argparse
import csv
import io
import json
import re
from collections import defaultdict
from pathlib import Path

from pipeline_config import (
    PROCESSED_DIR, NUM_SAMPLE_FRAMES, FRAME_HEIGHT, FRAME_WIDTH,
    AWS_REGION,
    get_processed_base, get_processed_prefix_s3,
    is_cloud, S3_BUCKET,
    get_s3_client, read_text_from_s3,
)


def _parse_s3_uri(clip_path: str) -> tuple[str, str] | None:
    m = re.match(r"s3://([^/]+)/(.+)", clip_path.strip())
    if m:
        return m.group(1), m.group(2)
    return None


def load_processed_csv(processed_dir: Path | None = None) -> list[dict]:
    base = processed_dir or PROCESSED_DIR
    csv_path = base / "processed_clips.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_processed_csv_from_s3(mvp: bool = True) -> list[dict]:
    prefix = get_processed_prefix_s3(mvp)
    key = f"{prefix}/processed_clips.csv"
    text = read_text_from_s3(key)
    return list(csv.DictReader(io.StringIO(text)))


def load_label_map(processed_dir: Path | None = None) -> dict:
    base = processed_dir or PROCESSED_DIR
    with open(base / "label_map.json", encoding="utf-8") as f:
        return json.load(f)


def load_label_map_from_s3(mvp: bool = True) -> dict:
    prefix = get_processed_prefix_s3(mvp)
    text = read_text_from_s3(f"{prefix}/label_map.json")
    return json.loads(text)


def check_files_exist_local(records: list[dict]) -> int:
    missing = 0
    for r in records:
        if not Path(r["clip_path"]).exists():
            missing += 1
    return missing


def check_files_exist_s3(records: list[dict]) -> int:
    s3 = get_s3_client()
    missing = 0
    for r in records:
        clip_path = r.get("clip_path", "")
        parsed = _parse_s3_uri(clip_path)
        if not parsed:
            missing += 1
            continue
        bucket, key = parsed
        try:
            s3.head_object(Bucket=bucket, Key=key)
        except Exception:
            missing += 1
    return missing


def check_clip_properties_local(records: list[dict], sample_size: int = 200) -> dict:
    import random
    import cv2
    sample = random.sample(records, min(sample_size, len(records)))
    bad_frames = bad_resolution = 0
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
    return {"sampled": len(sample), "bad_frame_count": bad_frames, "bad_resolution": bad_resolution}


def check_signer_leakage(records: list[dict]) -> dict:
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
                leaks[f"{s1} & {s2}"] = len(overlap)
    return leaks


def check_label_map_consistency(records: list[dict], label_map: dict) -> tuple[bool, str]:
    """
    Label map is consistent if every gloss in the map appears in the CSV.
    CSV may have extra glosses (rare ones dropped by build_unified_dataset's filter).
    Returns (pass, message).
    """
    csv_glosses = {r["gloss"] for r in records}
    map_glosses = set(label_map.keys())
    if not map_glosses:
        return False, "label_map is empty"
    extra_in_csv = csv_glosses - map_glosses
    missing_in_csv = map_glosses - csv_glosses
    if missing_in_csv:
        return False, f"label_map has {len(missing_in_csv)} glosses not in CSV"
    if extra_in_csv:
        return True, f"OK (CSV has {len(extra_in_csv)} extra glosses not in map, expected after rare-gloss filter)"
    return True, "OK (exact match)"


def publish_cloudwatch_metric(pass_value: bool):
    try:
        import boto3
        cw = boto3.client("cloudwatch", region_name=AWS_REGION)
        cw.put_metric_data(
            Namespace="EyeHearU/Pipeline",
            MetricData=[{
                "MetricName": "ValidationPass",
                "Value": 1 if pass_value else 0,
                "Unit": "None",
            }],
        )
        print("  [CloudWatch] Published ValidationPass =", 1 if pass_value else 0)
    except Exception as e:
        print("  [CloudWatch] Skip:", e)


def main():
    parser = argparse.ArgumentParser(description="Validate processed dataset (Stage 4).")
    parser.add_argument("--mvp", action="store_true", help="Use processed/mvp/ (local or S3).")
    args = parser.parse_args()

    use_s3 = is_cloud() and args.mvp
    processed_dir = get_processed_base(args.mvp)

    print("=" * 60)
    print("Dataset Validation (Stage 4)")
    print("=" * 60)
    if args.mvp:
        print(f"  MVP: {processed_dir}" + (" (S3)" if use_s3 else ""))

    if use_s3:
        records = load_processed_csv_from_s3(mvp=True)
        label_map = load_label_map_from_s3(mvp=True)
    else:
        records = load_processed_csv(processed_dir)
        label_map = load_label_map(processed_dir)

    total = len(records)
    print(f"  Records in CSV: {total}")

    # 1. File existence
    if use_s3:
        missing = check_files_exist_s3(records)
    else:
        missing = check_files_exist_local(records)
    status = "PASS" if missing == 0 else "FAIL"
    print(f"  [{status}] Missing clip files: {missing}")

    # 2. Spot-check frame count & resolution
    if use_s3:
        print("  [SKIP] Frame/resolution spot-check (S3: run locally with synced data to check)")
    else:
        props = check_clip_properties_local(records)
        status = "PASS" if props["bad_frame_count"] == 0 and props["bad_resolution"] == 0 else "WARN"
        print(f"  [{status}] Spot-checked {props['sampled']} clips: "
              f"{props['bad_frame_count']} bad frame count, "
              f"{props['bad_resolution']} bad resolution")

    # 3. Signer leakage
    leaks = check_signer_leakage(records)
    status = "PASS" if not leaks else "FAIL"
    print(f"  [{status}] ASL Citizen signer-disjoint splits: {leaks if leaks else 'none'}")

    # 4. Label map consistency
    consistent, msg = check_label_map_consistency(records, label_map)
    status = "PASS" if consistent else "FAIL"
    print(f"  [{status}] label_map vs CSV: {msg}")

    overall_pass = missing == 0 and not leaks and consistent
    if is_cloud():
        publish_cloudwatch_metric(overall_pass)

    print("\nValidation complete.", "PASS" if overall_pass else "FAIL")


if __name__ == "__main__":
    main()
