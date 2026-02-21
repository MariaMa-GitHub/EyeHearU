"""
Shared configuration for the data-processing pipeline.

All paths, constants, and tunables for ingestion, preprocessing,
and dataset building live here so every script uses the same values.

Supports two storage backends:
  - local  (default): reads/writes to the local data/ directory
  - s3     (cloud) : reads/writes to an S3 bucket, selected by PIPELINE_ENV

Set PIPELINE_ENV to control the environment:
  PIPELINE_ENV=local       ->local filesystem (default)
  PIPELINE_ENV=dev         ->s3://eye-hear-u-dev-data/
  PIPELINE_ENV=staging     ->s3://eye-hear-u-staging-data/
  PIPELINE_ENV=prod        ->s3://eye-hear-u-prod-data/
"""

import os
from pathlib import Path

# ── Environment ──────────────────────────────────────────────────
PIPELINE_ENV = os.getenv("PIPELINE_ENV", "local")
AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")

# ── S3 bucket name (only used when PIPELINE_ENV != "local") ─────
S3_BUCKET = f"eye-hear-u-{PIPELINE_ENV}-data" if PIPELINE_ENV != "local" else None

# ── Local paths (used when PIPELINE_ENV == "local") ─────────────
_DATA_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = _DATA_ROOT / "raw"
PROCESSED_DIR = _DATA_ROOT / "processed"

# Per-source raw directories
ASL_CITIZEN_RAW = RAW_DIR / "asl_citizen"
WLASL_RAW = RAW_DIR / "wlasl"
MSASL_RAW = RAW_DIR / "msasl"

# Processed output
CLIPS_DIR = PROCESSED_DIR / "clips"
METADATA_CSV = PROCESSED_DIR / "metadata.csv"
LABEL_MAP_JSON = PROCESSED_DIR / "label_map.json"
STATS_JSON = PROCESSED_DIR / "dataset_stats.json"

# MVP-filtered outputs go under this subdir (processed/mvp/ in S3, processed/mvp/ locally)
MVP_PROCESSED_SUBDIR = "mvp"


def get_processed_base(mvp: bool = False) -> Path:
    """Local path for processed (or processed/mvp when mvp=True)."""
    return PROCESSED_DIR / MVP_PROCESSED_SUBDIR if mvp else PROCESSED_DIR


def get_processed_prefix_s3(mvp: bool = False) -> str:
    """S3 key prefix for processed (processed/ or processed/mvp/ when mvp=True)."""
    if not mvp:
        return S3_PROCESSED_PREFIX
    return f"{S3_PROCESSED_PREFIX}/{MVP_PROCESSED_SUBDIR}"

# ── S3 key prefixes (mirrors local layout) ──────────────────────
S3_RAW_PREFIX = "raw"
S3_PROCESSED_PREFIX = "processed"
S3_MODELS_PREFIX = "models"

# ── Remote metadata URLs ─────────────────────────────────────────
WLASL_JSON_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
)
MSASL_CLASSES_URL = (
    "https://raw.githubusercontent.com/iamgarcia/msasl-video-downloader/master/MSASL_classes.json"
)
MSASL_TRAIN_URL = (
    "https://raw.githubusercontent.com/iamgarcia/msasl-video-downloader/master/MSASL_train.json"
)
MSASL_VAL_URL = (
    "https://raw.githubusercontent.com/iamgarcia/msasl-video-downloader/master/MSASL_val.json"
)
MSASL_TEST_URL = (
    "https://raw.githubusercontent.com/iamgarcia/msasl-video-downloader/master/MSASL_test.json"
)

# ── Video preprocessing ──────────────────────────────────────────
NUM_SAMPLE_FRAMES = 16          # frames uniformly sampled per clip (I3D standard)
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
MIN_CLIP_FRAMES = 8             # discard clips shorter than this
MAX_CLIP_SECONDS = 4.0          # discard clips longer than this
VIDEO_FPS = 30                  # target fps when re-encoding

# ── Dataset building ─────────────────────────────────────────────
MIN_VIDEOS_PER_GLOSS = 5       # drop glosses with fewer training samples
SOURCES = ["asl_citizen", "wlasl", "msasl"]

# ── MVP vocabulary (single isolated signs: greetings, needs, restaurant, medical, A-Z, 1-10) ─
_SCRIPTS_DIR = Path(__file__).resolve().parent
MVP_GLOSSES_FILE = _SCRIPTS_DIR / "mvp_glosses.txt"


def load_mvp_glosses(path: Path | None = None) -> set[str]:
    """Load MVP gloss set from file (one gloss per line; # and blank lines ignored; lowercased)."""
    p = path or MVP_GLOSSES_FILE
    glosses = set()
    if not p.exists():
        return glosses
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if line and not line.startswith("#"):
                glosses.add(line)
    return glosses


# ── Remote dataset URLs ───────────────────────────────────────────
# ASL Citizen: Microsoft Download Center (publicly available)
# Web: https://www.microsoft.com/en-us/download/details.aspx?id=105253
# Two files: ASL_Citizen.zip (42.8 GB videos), Demographics.zip (5 KB - may contain metadata)
ASL_CITIZEN_DOWNLOAD_URL = os.getenv(
    "ASL_CITIZEN_DOWNLOAD_URL",
    "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
)
# Optional: if metadata is in a separate zip, set this (get exact URL from download page)
ASL_CITIZEN_DEMOGRAPHICS_URL = os.getenv("ASL_CITIZEN_DEMOGRAPHICS_URL", "")


# ── S3 helpers ───────────────────────────────────────────────────
def is_cloud() -> bool:
    return PIPELINE_ENV != "local" and S3_BUCKET is not None


def get_s3_client():
    """Lazy-import boto3 and return an S3 client."""
    import boto3
    return boto3.client("s3", region_name=AWS_REGION)


def s3_key(local_path: Path) -> str:
    """Convert a local data/ path to its S3 key equivalent."""
    rel = local_path.relative_to(_DATA_ROOT)
    return str(rel)


def s3_object_exists(key: str) -> bool:
    s3 = get_s3_client()
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def list_s3_keys(prefix: str) -> list[str]:
    s3 = get_s3_client()
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def write_text_to_s3(text: str, key: str):
    s3 = get_s3_client()
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=text.encode("utf-8"))
    print(f"  [s3] Wrote s3://{S3_BUCKET}/{key}")


def upload_file_to_s3(local_path: Path, key: str):
    s3 = get_s3_client()
    s3.upload_file(str(local_path), S3_BUCKET, key)
    print(f"  [s3] Uploaded {local_path.name} ->s3://{S3_BUCKET}/{key}")


def stream_url_to_s3(
    url: str, key: str,
    chunk_size: int = 10 * 1024 * 1024,  # 10MB parts (42GB = ~4300 parts, under 10k limit)
    progress_interval_gb: float = 1.0,
):
    """Stream a URL download directly to S3 via multipart upload (no local disk).
    For large files (e.g. 42GB) this can take 2-4+ hours; progress is logged to stdout."""
    import requests
    s3 = get_s3_client()

    # Long timeout for initial connection; streaming has no body timeout
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    mpu = s3.create_multipart_upload(Bucket=S3_BUCKET, Key=key)
    upload_id = mpu["UploadId"]
    parts = []
    part_num = 1
    buf = b""
    total_bytes = 0
    next_log_gb = progress_interval_gb

    try:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            buf += chunk
            if len(buf) >= chunk_size:
                part = s3.upload_part(
                    Bucket=S3_BUCKET, Key=key,
                    UploadId=upload_id, PartNumber=part_num,
                    Body=buf,
                )
                parts.append({"ETag": part["ETag"], "PartNumber": part_num})
                part_num += 1
                total_bytes += len(buf)
                buf = b""
                total_gb = total_bytes / (1024 ** 3)
                if total_gb >= next_log_gb:
                    print(f"  [s3] Streamed {total_gb:.1f} GB to S3 ...")
                    next_log_gb += progress_interval_gb

        if buf:
            part = s3.upload_part(
                Bucket=S3_BUCKET, Key=key,
                UploadId=upload_id, PartNumber=part_num,
                Body=buf,
            )
            parts.append({"ETag": part["ETag"], "PartNumber": part_num})
            total_bytes += len(buf)

        s3.complete_multipart_upload(
            Bucket=S3_BUCKET, Key=key, UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
        print(f"  [s3] Streamed {total_bytes / (1024**3):.2f} GB -> s3://{S3_BUCKET}/{key} ({part_num} parts)")
    except Exception:
        s3.abort_multipart_upload(Bucket=S3_BUCKET, Key=key, UploadId=upload_id)
        raise


def download_json_to_s3(url: str, s3_key_path: str) -> list | dict:
    """Download a JSON from a URL, store in S3, and return parsed content."""
    import requests
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    write_text_to_s3(resp.text, s3_key_path)
    return data


def read_text_from_s3(key: str) -> str:
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return obj["Body"].read().decode("utf-8")


def read_json_from_s3(key: str) -> list | dict:
    import json as _json
    return _json.loads(read_text_from_s3(key))
