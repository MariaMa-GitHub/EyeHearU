"""
Ingestion script for ASL Citizen — the primary training dataset.

ASL Citizen is publicly available from Microsoft Download Center:
  Web: https://www.microsoft.com/en-us/download/details.aspx?id=105253
  Direct: https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip

Supports two modes:
  - local:  reads pre-downloaded files from data/raw/asl_citizen/
  - cloud:  streams the ZIP archive directly from Microsoft to S3 (no local download needed)

For cloud mode, the script will automatically download the archive from Microsoft
and stream it to S3. You can override the URL by setting ASL_CITIZEN_DOWNLOAD_URL env var.
"""

import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import requests

from pipeline_config import (
    ASL_CITIZEN_RAW, PROCESSED_DIR,
    ASL_CITIZEN_DOWNLOAD_URL, ASL_CITIZEN_DEMOGRAPHICS_URL,
    is_cloud, get_s3_client, s3_object_exists, list_s3_keys,
    write_text_to_s3, stream_url_to_s3, download_json_to_s3, read_json_from_s3,
    S3_BUCKET, S3_RAW_PREFIX, S3_PROCESSED_PREFIX,
    get_processed_base, get_processed_prefix_s3,
)


class S3RangeReader:
    """Read-only file-like object that fetches bytes from S3 via range requests (no local file)."""

    def __init__(self, bucket: str, key: str, size: int, s3_client):
        self.bucket = bucket
        self.key = key
        self.size = size
        self.s3 = s3_client
        self._pos = 0

    def seekable(self):
        return True

    def seek(self, offset: int, whence: int = 0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self.size + offset
        self._pos = max(0, min(self._pos, self.size))
        return self._pos

    def tell(self):
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self._pos
        if self._pos >= self.size or size == 0:
            return b""
        end = min(self._pos + size, self.size)
        resp = self.s3.get_object(
            Bucket=self.bucket, Key=self.key,
            Range=f"bytes={self._pos}-{end - 1}",
        )
        data = resp["Body"].read()
        self._pos += len(data)
        return data

    def __len__(self):
        return self.size


def load_metadata_local(mvp_glosses: set[str] | None = None) -> list[dict]:
    """Load metadata from local filesystem. If mvp_glosses is set, keep only those glosses."""
    json_path = ASL_CITIZEN_RAW / "metadata.json"
    csv_path = ASL_CITIZEN_RAW / "metadata.csv"

    records: list[dict] = []

    def keep(r: dict) -> bool:
        return mvp_glosses is None or (r.get("gloss") or "").strip().lower() in mvp_glosses

    if json_path.exists():
        with open(json_path) as f:
            raw = json.load(f)
        for entry in raw:
            r = _normalize_entry(entry)
            if keep(r):
                records.append(r)

    elif csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = _normalize_entry(row)
                if keep(r):
                    records.append(r)
    else:
        print("[ERROR] No metadata.json or metadata.csv found in", ASL_CITIZEN_RAW)
        sys.exit(1)

    return records


# Metadata file names to look for (order matters: try exact first, then flexible)
_METADATA_JSON_NAMES = (
    "metadata.json", "demographics.json", "labels.json", "annotations.json",
    "train.json", "data.json",
)
_METADATA_CSV_NAMES = (
    "metadata.csv", "demographics.csv", "labels.csv", "annotations.csv",
    "train.csv", "data.csv",
)


def _try_read_metadata_entry(
    zf: zipfile.ZipFile,
    name: str,
    s3_json_key: str,
    s3_csv_key: str,
    mvp_glosses: set[str] | None = None,
) -> list[dict] | None:
    """Try to read one entry as JSON or CSV metadata; return records or None. If mvp_glosses set, filter in-place."""
    try:
        content = zf.read(name).decode("utf-8")
    except Exception:
        return None

    def keep(r: dict) -> bool:
        return mvp_glosses is None or (r.get("gloss") or "").strip().lower() in mvp_glosses

    name_lower = name.lower()
    if name_lower.endswith(".json"):
        try:
            raw = json.loads(content)
            if isinstance(raw, list) and raw and isinstance(raw[0], dict):
                write_text_to_s3(content, s3_json_key)
                out = [r for r in (_normalize_entry(entry) for entry in raw) if keep(r)]
                return out
            if isinstance(raw, dict):
                write_text_to_s3(content, s3_json_key)
                out = [r for r in (_normalize_entry(entry) for entry in raw.get("entries", raw.get("videos", [raw]))) if keep(r)]
                return out
        except (json.JSONDecodeError, TypeError):
            pass
    if name_lower.endswith(".csv"):
        try:
            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)
            if rows:
                write_text_to_s3(content, s3_csv_key)
                out = [r for r in (_normalize_entry(row) for row in rows) if keep(r)]
                return out
        except Exception:
            pass
    return None


def extract_metadata_from_s3_zip(
    bucket: str,
    key: str,
    s3_json_key: str,
    s3_csv_key: str,
    mvp_glosses: set[str] | None = None,
) -> list[dict]:
    """Extract metadata from a ZIP in S3 using range requests only (no local file). If mvp_glosses set, keep only those glosses."""
    print("  Extracting metadata from ZIP in S3 (range requests only, no local download) ...")
    if mvp_glosses:
        print(f"  MVP filter: keeping only {len(mvp_glosses)} glosses")
    s3 = get_s3_client()
    head = s3.head_object(Bucket=bucket, Key=key)
    size = head["ContentLength"]
    reader = S3RangeReader(bucket, key, size, s3)

    with zipfile.ZipFile(reader, "r") as zf:
        names = zf.namelist()
        # Try exact and flexible names
        for name in names:
            base = name.split("/")[-1].lower()
            if base in _METADATA_JSON_NAMES or base in _METADATA_CSV_NAMES:
                print(f"    Trying {name} ...")
                result = _try_read_metadata_entry(zf, name, s3_json_key, s3_csv_key, mvp_glosses)
                if result is not None:
                    print(f"    Found metadata: {name} ({len(result)} entries)")
                    return result
            if "metadata" in base or "demographic" in base or "label" in base or "annotation" in base:
                if base.endswith(".json") or base.endswith(".csv"):
                    print(f"    Trying {name} ...")
                    result = _try_read_metadata_entry(zf, name, s3_json_key, s3_csv_key, mvp_glosses)
                    if result is not None:
                        print(f"    Found metadata: {name} ({len(result)} entries)")
                        return result

        # Not found: show sample of contents so we can fix the script
        sample = [n for n in names if not n.endswith("/")][:30]
        print("  No metadata file found. First 30 entries in ZIP:")
        for n in sample:
            print(f"    - {n}")
        if len(names) > 30:
            print(f"    ... and {len(names) - 30} more")
    raise FileNotFoundError(
        "No metadata file found in ZIP. ASL_Citizen.zip may contain only videos; "
        "metadata might be in Demographics.zip (5 KB). Set ASL_CITIZEN_DEMOGRAPHICS_URL to that download URL and re-run."
    )


def download_and_extract_zip(
    url: str,
    s3_json_key: str,
    s3_csv_key: str,
    mvp_glosses: set[str] | None = None,
) -> list[dict]:
    """Stream ZIP directly to S3, then extract metadata from S3 using range requests only (no local file)."""
    s3_archive_key = f"{S3_RAW_PREFIX}/asl_citizen/archive.zip"

    # Step 1: Stream ZIP directly to S3 (no local storage)
    if not s3_object_exists(s3_archive_key):
        print("  Streaming ASL Citizen archive to S3 (~42 GB; can take 2-4+ hours) ...")
        print("  Direct to S3 only; progress logged every 1 GB.")
        stream_url_to_s3(url, s3_archive_key)
        print(f"  Archive saved to s3://{S3_BUCKET}/{s3_archive_key}")
    else:
        print("  Archive already in S3, skipping download.")

    # Step 2: Extract metadata from ZIP in S3 using range requests (no local file)
    try:
        return extract_metadata_from_s3_zip(
            S3_BUCKET, s3_archive_key, s3_json_key, s3_csv_key, mvp_glosses
        )
    except FileNotFoundError:
        # Main zip may be videos only; try Demographics.zip if URL is set
        if not ASL_CITIZEN_DEMOGRAPHICS_URL:
            raise
        demo_key = f"{S3_RAW_PREFIX}/asl_citizen/demographics.zip"
        if not s3_object_exists(demo_key):
            print("  Metadata not in main archive; streaming Demographics.zip to S3 ...")
            stream_url_to_s3(ASL_CITIZEN_DEMOGRAPHICS_URL, demo_key)
        return extract_metadata_from_s3_zip(
            S3_BUCKET, demo_key, s3_json_key, s3_csv_key, mvp_glosses
        )


def load_metadata_cloud(mvp_glosses: set[str] | None = None) -> list[dict]:
    """Load metadata from S3, downloading and extracting from ZIP if needed. If mvp_glosses set, keep only those glosses."""
    s3_json_key = f"{S3_RAW_PREFIX}/asl_citizen/metadata.json"
    s3_csv_key = f"{S3_RAW_PREFIX}/asl_citizen/metadata.csv"

    def keep(r: dict) -> bool:
        return mvp_glosses is None or (r.get("gloss") or "").strip().lower() in mvp_glosses

    # Check if metadata already exists in S3
    if s3_object_exists(s3_json_key):
        print("  Reading metadata.json from S3 ...")
        raw = read_json_from_s3(s3_json_key)
        return [r for r in (_normalize_entry(entry) for entry in raw) if keep(r)]

    if s3_object_exists(s3_csv_key):
        print("  Reading metadata.csv from S3 ...")
        s3 = get_s3_client()
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_csv_key)
        text = obj["Body"].read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        return [r for r in (_normalize_entry(row) for row in reader) if keep(r)]

    # Metadata not in S3 - need to download and extract
    if ASL_CITIZEN_DOWNLOAD_URL:
        if ASL_CITIZEN_DOWNLOAD_URL.endswith(".json"):
            # Direct JSON URL (unlikely but handle it)
            print("  Downloading metadata.json directly from URL ...")
            raw = download_json_to_s3(ASL_CITIZEN_DOWNLOAD_URL, s3_json_key)
            return [r for r in (_normalize_entry(entry) for entry in raw) if keep(r)]
        else:
            # ZIP URL - download, extract, and process
            return download_and_extract_zip(ASL_CITIZEN_DOWNLOAD_URL, s3_json_key, s3_csv_key, mvp_glosses)

    # This should never happen now since we have a default URL
    print("[ERROR] No metadata found in S3 and ASL_CITIZEN_DOWNLOAD_URL not set.")
    print("  Upload metadata.json to s3://" + S3_BUCKET + "/" + s3_json_key)
    sys.exit(1)


def _normalize_entry(entry: dict) -> dict:
    # ASL Citizen CSV uses "Video file", "Participant ID", "Gloss"; support both naming conventions
    fname = (
        entry.get("filename")
        or entry.get("video_id")
        or entry.get("Video file")
        or ""
    )
    if isinstance(fname, str):
        fname = fname.strip()
    gloss = (
        entry.get("gloss")
        or entry.get("Gloss")
        or ""
    )
    if isinstance(gloss, str):
        gloss = gloss.strip().lower()
    signer_id = str(
        entry.get("user_id")
        or entry.get("signer_id")
        or entry.get("Participant ID")
        or ""
    )
    split = (
        entry.get("split")
        or entry.get("Split")
        or "train"
    )
    if isinstance(split, str):
        split = split.strip().lower()
    src_path = (
        f"raw/asl_citizen/videos/{fname}" if is_cloud()
        else str(ASL_CITIZEN_RAW / "videos" / fname)
    )
    return {
        "clip_id": fname,
        "gloss": gloss,
        "signer_id": signer_id,
        "split": split,
        "src_path": src_path,
    }


def validate_videos_local(records: list[dict]) -> list[dict]:
    valid = []
    missing = 0
    for r in records:
        if Path(r["src_path"]).exists():
            valid.append(r)
        else:
            missing += 1
    print(f"[asl_citizen] {len(valid)} videos found, {missing} missing on disk.")
    return valid


def validate_videos_cloud(records: list[dict]) -> list[dict]:
    print("  Listing videos in S3 ...")
    existing_keys = set(list_s3_keys(f"{S3_RAW_PREFIX}/asl_citizen/videos/"))
    if existing_keys:
        valid = [r for r in records if r["src_path"] in existing_keys]
        missing = len(records) - len(valid)
        print(f"[asl_citizen] {len(valid)} videos in S3, {missing} missing.")
        return valid
    # Videos are inside archive.zip, not extracted as separate S3 objects: keep all records
    print("  No separate video objects in S3 (videos are inside archive.zip); keeping all metadata rows.")
    return records


def write_ingested_csv(records: list[dict], mvp: bool = False):
    fieldnames = ["clip_id", "gloss", "signer_id", "split", "source", "src_path"]
    prefix = get_processed_prefix_s3(mvp) if is_cloud() else None
    base = get_processed_base(mvp)

    if is_cloud():
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({**r, "source": "asl_citizen"})
        key = f"{prefix}/ingested_asl_citizen.csv"
        write_text_to_s3(buf.getvalue(), key)
        print(f"[asl_citizen] Wrote {len(records)} records -> s3://.../{key}")
    else:
        out_path = base / "ingested_asl_citizen.csv"
        base.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow({**r, "source": "asl_citizen"})
        print(f"[asl_citizen] Wrote {len(records)} records -> {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest ASL Citizen metadata (and optionally filter to MVP vocabulary).")
    parser.add_argument("--mvp", action="store_true", help="Keep only MVP glosses (see mvp_glosses.txt)")
    args = parser.parse_args()

    mvp = None
    if args.mvp:
        from pipeline_config import load_mvp_glosses
        mvp = load_mvp_glosses()
        if not mvp:
            print("  [mvp] No mvp_glosses.txt found; skipping filter.")

    print("=" * 60)
    print("ASL Citizen Ingestion")
    print(f"  Mode: {'cloud (' + S3_BUCKET + ')' if is_cloud() else 'local'}")
    if mvp:
        print(f"  MVP: keeping only {len(mvp)} glosses")
    print("=" * 60)

    records = load_metadata_cloud(mvp) if is_cloud() else load_metadata_local(mvp)
    print(f"  Parsed {len(records)} metadata entries.")
    print(f"  Unique glosses: {len({r['gloss'] for r in records})}")
    print(f"  Unique signers: {len({r['signer_id'] for r in records})}")

    records = validate_videos_cloud(records) if is_cloud() else validate_videos_local(records)
    write_ingested_csv(records, mvp=args.mvp)


if __name__ == "__main__":
    main()
