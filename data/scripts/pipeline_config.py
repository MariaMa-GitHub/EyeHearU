"""
Shared configuration for the data-processing pipeline.

All paths, constants, and tunables for ingestion, preprocessing,
and dataset building live here so every script uses the same values.

Supports two storage backends:
  - local  (default): reads/writes to the local data/ directory
  - s3     (cloud) : reads/writes to an S3 bucket, selected by PIPELINE_ENV

Set PIPELINE_ENV to control the environment:
  PIPELINE_ENV=local       → local filesystem (default)
  PIPELINE_ENV=dev         → s3://eye-hear-u-dev-data/
  PIPELINE_ENV=staging     → s3://eye-hear-u-staging-data/
  PIPELINE_ENV=prod        → s3://eye-hear-u-prod-data/
"""

import os
from pathlib import Path

# ── Environment ──────────────────────────────────────────────────
PIPELINE_ENV = os.getenv("PIPELINE_ENV", "local")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

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

# ── S3 key prefixes (mirrors local layout) ──────────────────────
S3_RAW_PREFIX = "raw"
S3_PROCESSED_PREFIX = "processed"
S3_MODELS_PREFIX = "models"

# ── Remote metadata URLs ─────────────────────────────────────────
WLASL_JSON_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
)
MSASL_CLASSES_URL = (
    "https://raw.githubusercontent.com/AlanTatworked/MS-ASL/master/MSASL_classes.json"
)
MSASL_TRAIN_URL = (
    "https://raw.githubusercontent.com/AlanTatworked/MS-ASL/master/MSASL_train.json"
)
MSASL_VAL_URL = (
    "https://raw.githubusercontent.com/AlanTatworked/MS-ASL/master/MSASL_val.json"
)
MSASL_TEST_URL = (
    "https://raw.githubusercontent.com/AlanTatworked/MS-ASL/master/MSASL_test.json"
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
