"""
S3 sync helpers for EyeHearU I3D training inputs.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def get_s3_client(region: str):
    return boto3.client("s3", region_name=region)


def get_active_plan_id(s3, bucket: str, mvp: bool = True) -> str:
    key = f"processed/{'mvp/' if mvp else ''}i3d/split_plans/ACTIVE_PLAN.json"
    obj = s3.get_object(Bucket=bucket, Key=key)
    payload = json.loads(obj["Body"].read().decode("utf-8"))
    active = payload.get("active_plan_id")
    if not active:
        raise RuntimeError(f"ACTIVE_PLAN.json missing active_plan_id at s3://{bucket}/{key}")
    return active


def download_splits(
    s3,
    bucket: str,
    plan_id: str,
    local_splits_dir: Path,
    mvp: bool = True,
) -> dict[str, Path]:
    base = f"processed/{'mvp/' if mvp else ''}i3d/split_plans/{plan_id}/splits"
    local_splits_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "val", "test"):
        key = f"{base}/{split}.csv"
        dst = local_splits_dir / f"{split}.csv"
        s3.download_file(bucket, key, str(dst))
        out[split] = dst
    return out


def _read_split_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_required_filenames(split_paths: list[Path], limit: int | None = None) -> list[str]:
    filenames = []
    seen = set()
    for p in split_paths:
        for row in _read_split_rows(p):
            name = (row.get("filename") or "").strip()
            if not name or name in seen:
                continue
            filenames.append(name)
            seen.add(name)
            if limit is not None and len(filenames) >= limit:
                return filenames
    return filenames


def download_clip_subset(
    s3,
    bucket: str,
    clip_filenames: list[str],
    local_clip_root: Path,
    mvp: bool = True,
) -> tuple[int, int]:
    """
    Download clip paths listed in split CSVs.
    filenames are expected relative to clips root, e.g. train/gloss/clip.mp4.
    """
    prefix = f"processed/{'mvp/' if mvp else ''}clips"
    local_clip_root.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0

    for name in clip_filenames:
        key = f"{prefix}/{name}"
        dst = local_clip_root / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            skipped += 1
            continue
        try:
            s3.download_file(bucket, key, str(dst))
            downloaded += 1
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey"}:
                print(f"[warn] Missing clip in S3: s3://{bucket}/{key}")
                continue
            raise
    return downloaded, skipped

