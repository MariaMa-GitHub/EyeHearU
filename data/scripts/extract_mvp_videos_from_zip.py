"""
Extract only MVP clip videos from ASL Citizen archive.zip in S3.

Reads processed/mvp/ingested_asl_citizen.csv from S3 to get the list of clip_ids,
opens raw/asl_citizen/archive.zip in S3 via range requests, and streams each
matching member to raw/asl_citizen/videos/{clip_id}. No full zip download;
extracts only the MVP subset so preprocess can run without 40 GB.

Requires PIPELINE_ENV=dev (or staging/prod) and the archive + MVP CSV in S3.

Usage:
  PIPELINE_ENV=dev python extract_mvp_videos_from_zip.py
  PIPELINE_ENV=dev python extract_mvp_videos_from_zip.py --skip-existing
  PIPELINE_ENV=dev python extract_mvp_videos_from_zip.py --dry-run
  PIPELINE_ENV=dev python extract_mvp_videos_from_zip.py --limit 10
"""

import argparse
import csv
import io
import zipfile

from pipeline_config import (
    is_cloud,
    S3_BUCKET,
    S3_RAW_PREFIX,
    get_processed_prefix_s3,
    get_s3_client,
    read_text_from_s3,
    s3_object_exists,
)
from ingest_asl_citizen import S3RangeReader

ARCHIVE_KEY = f"{S3_RAW_PREFIX}/asl_citizen/archive.zip"
VIDEOS_PREFIX = f"{S3_RAW_PREFIX}/asl_citizen/videos/"


def load_mvp_clip_ids_from_s3() -> list[str]:
    """Load MVP ingested CSV from S3 and return list of clip_id (unique, order preserved)."""
    key = f"{get_processed_prefix_s3(mvp=True)}/ingested_asl_citizen.csv"
    text = read_text_from_s3(key)
    rows = list(csv.DictReader(io.StringIO(text)))
    seen = set()
    clip_ids = []
    for r in rows:
        cid = (r.get("clip_id") or "").strip()
        if cid and cid not in seen:
            seen.add(cid)
            clip_ids.append(cid)
    return clip_ids


def find_zip_member_for_clip(zip_namelist: list[str], clip_id: str) -> str | None:
    """Return the zip member path that corresponds to this clip_id (match by basename)."""
    norm = clip_id.replace("\\", "/")
    for name in zip_namelist:
        if name.endswith("/"):
            continue
        parts = name.replace("\\", "/").split("/")
        base = parts[-1] if parts else name
        if base == norm or name == norm:
            return name
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract MVP videos from ASL Citizen archive.zip in S3 to raw/asl_citizen/videos/."
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip clips already in S3")
    parser.add_argument("--dry-run", action="store_true", help="Only list what would be extracted")
    parser.add_argument("--limit", type=int, default=0, help="Max number of clips to extract (0 = all)")
    args = parser.parse_args()

    if not is_cloud():
        print("[ERROR] This script requires cloud mode. Set PIPELINE_ENV=dev (or staging/prod).")
        exit(1)

    print("=" * 60)
    print("Extract MVP videos from archive.zip (S3)")
    print(f"  Bucket: {S3_BUCKET}")
    print("=" * 60)

    clip_ids = load_mvp_clip_ids_from_s3()
    print(f"  MVP ingested CSV: {len(clip_ids)} unique clip_ids")

    s3 = get_s3_client()
    try:
        head = s3.head_object(Bucket=S3_BUCKET, Key=ARCHIVE_KEY)
    except s3.exceptions.ClientError as e:
        print(f"[ERROR] Archive not found: s3://{S3_BUCKET}/{ARCHIVE_KEY}")
        print("  Run ingest first so the archive is in S3.")
        exit(1)

    size = head["ContentLength"]
    print(f"  Opening archive.zip ({size / (1024**3):.1f} GB) via range requests ...")
    reader = S3RangeReader(S3_BUCKET, ARCHIVE_KEY, size, s3)

    with zipfile.ZipFile(reader, "r") as zf:
        names = zf.namelist()
        print(f"  Zip has {len(names)} entries")

        # Build clip_id -> zip member name
        clip_to_member = {}
        for cid in clip_ids:
            member = find_zip_member_for_clip(names, cid)
            if member:
                clip_to_member[cid] = member
            else:
                print(f"  [skip] No zip member for clip_id: {cid}")

        to_extract = []
        for cid in clip_ids:
            if cid not in clip_to_member:
                continue
            key = f"{VIDEOS_PREFIX}{cid}"
            if args.skip_existing and s3_object_exists(key):
                continue
            to_extract.append((cid, clip_to_member[cid]))
            if args.limit and len(to_extract) >= args.limit:
                break

        print(f"  Clips to extract: {len(to_extract)}")
        if args.dry_run:
            for cid, member in to_extract[:20]:
                print(f"    {cid} <- {member}")
            if len(to_extract) > 20:
                print(f"    ... and {len(to_extract) - 20} more")
            print("  [dry-run] Done.")
            return

        for i, (cid, member_name) in enumerate(to_extract):
            key = f"{VIDEOS_PREFIX}{cid}"
            try:
                with zf.open(member_name) as f:
                    s3.upload_fileobj(f, S3_BUCKET, key)
            except Exception as e:
                print(f"  [FAIL] {cid}: {e}")
                continue
            if (i + 1) % 50 == 0 or i == len(to_extract) - 1:
                print(f"  Extracted {i + 1}/{len(to_extract)} -> s3://{S3_BUCKET}/{key}")

    print(f"  Done. {len(to_extract)} videos in s3://{S3_BUCKET}/{VIDEOS_PREFIX}")


if __name__ == "__main__":
    main()
