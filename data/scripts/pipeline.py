"""
Main Data Processing Pipeline Orchestrator — Eye Hear U

This script orchestrates the complete end-to-end data processing pipeline:

  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │  INGEST     │ ──▶ │  CLEAN      │ ──▶ │  TRANSFORM  │ ──▶ │  LOAD       │
  │             │     │             │     │             │     │             │
  │ Download    │     │ Validate    │     │ Resize      │     │ Train/Val/  │
  │ ASL Citizen │     │ Filter      │     │ Crop hands  │     │ Test split  │
  │ + WLASL     │     │ Deduplicate │     │ Normalize   │     │ Label map   │
  │ metadata    │     │ Quality chk │     │ Augment     │     │ Stats       │
  └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘

Pipeline Stages:
  Stage 1: Data Ingestion      — Download raw datasets to data lake (S3 / local)
  Stage 2: Data Cleaning        — Validate, filter, deduplicate raw videos/frames
  Stage 3: Data Transformation  — Resize, crop, normalize, augment images
  Stage 4: Data Loading         — Split into train/val/test, generate metadata

AWS Service Mapping (for production deployment):
  - S3:          Raw data lake (videos) + processed data warehouse (images)
  - Lambda:      Lightweight frame extraction per video
  - Step Functions: Pipeline orchestration
  - SageMaker:   Model training on processed data
  - Glue:        ETL catalog and data quality checks
  - DynamoDB:    Pipeline run metadata and provenance tracking
  - CloudWatch:  Pipeline monitoring and alerting

Usage:
  python pipeline.py --stage all          # Run full pipeline
  python pipeline.py --stage ingest       # Only download/ingest
  python pipeline.py --stage clean        # Only clean/validate
  python pipeline.py --stage transform    # Only transform/preprocess
  python pipeline.py --stage load         # Only split and generate metadata
  python pipeline.py --stage report       # Only generate reports

Prerequisites:
  pip install opencv-python tqdm requests pandas mediapipe numpy
"""

import argparse
import hashlib
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent paths for config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ml"))
from config import Config

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

# ─── Paths ───────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
LOGS_DIR = DATA_DIR / "logs"

# ─── Logging ─────────────────────────────────────────────────────────────────

LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ─── Pipeline Run Metadata ───────────────────────────────────────────────────

class PipelineRun:
    """
    Tracks metadata for a single pipeline execution.

    In production, this would be stored in DynamoDB / a metadata database.
    Locally, we write it to JSON for provenance tracking.
    """

    def __init__(self):
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.stages_completed = []
        self.metrics = {}
        self.errors = []

    def log_stage(self, stage: str, metrics: dict):
        self.stages_completed.append(stage)
        self.metrics[stage] = metrics
        logger.info(f"Stage '{stage}' complete: {metrics}")

    def log_error(self, stage: str, error: str):
        self.errors.append({"stage": stage, "error": error, "time": time.time()})
        logger.error(f"[{stage}] {error}")

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": time.time(),
            "duration_seconds": time.time() - self.start_time,
            "stages_completed": self.stages_completed,
            "metrics": self.metrics,
            "errors": self.errors,
        }
        path = output_dir / f"pipeline_run_{self.run_id}.json"
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)
        logger.info(f"Pipeline run metadata saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: DATA INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

def stage_ingest(cfg: Config, run: PipelineRun) -> dict:
    """
    Data Ingestion Stage.

    Downloads raw datasets and organizes them in the raw data lake.

    AWS equivalent:
      - S3 bucket: s3://eyehearu-data-lake/raw/
      - Lambda trigger: On new dataset release, auto-ingest
      - Glue Crawler: Catalog new data partitions

    Local implementation:
      - data/raw/asl_citizen/  → ASL Citizen videos
      - data/raw/wlasl/        → WLASL videos
      - data/metadata/         → Ingestion logs
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA INGESTION")
    logger.info("=" * 60)

    metrics = {
        "datasets_ingested": [],
        "total_raw_files": 0,
    }

    # Ingest ASL Citizen
    asl_citizen_dir = RAW_DIR / "asl_citizen"
    if asl_citizen_dir.exists():
        video_count = sum(1 for _ in asl_citizen_dir.rglob("*.mp4"))
        logger.info(f"ASL Citizen: {video_count} videos found in {asl_citizen_dir}")
        metrics["datasets_ingested"].append("asl_citizen")
        metrics["asl_citizen_videos"] = video_count
        metrics["total_raw_files"] += video_count
    else:
        logger.warning(
            f"ASL Citizen not found at {asl_citizen_dir}. "
            "Run download_asl_citizen.py first."
        )

    # Ingest WLASL
    wlasl_dir = RAW_DIR / "wlasl"
    if wlasl_dir.exists():
        video_count = sum(1 for _ in wlasl_dir.rglob("*.mp4"))
        logger.info(f"WLASL: {video_count} videos found in {wlasl_dir}")
        metrics["datasets_ingested"].append("wlasl")
        metrics["wlasl_videos"] = video_count
        metrics["total_raw_files"] += video_count
    else:
        logger.warning(
            f"WLASL not found at {wlasl_dir}. "
            "Run download_wlasl.py first."
        )

    # Build raw data catalog
    catalog = _build_raw_catalog()
    catalog_path = METADATA_DIR / "raw_data_catalog.json"
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)
    metrics["catalog_entries"] = len(catalog.get("files", []))

    run.log_stage("ingest", metrics)
    return metrics


def _build_raw_catalog() -> dict:
    """
    Build a catalog of all raw data files.

    In production, this would be an AWS Glue Data Catalog.
    """
    catalog = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
        "files": [],
    }

    for dataset_dir in RAW_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        video_files = list(dataset_dir.rglob("*.mp4"))
        catalog["datasets"][dataset_name] = {
            "path": str(dataset_dir),
            "video_count": len(video_files),
            "total_size_mb": sum(f.stat().st_size for f in video_files) / (1024 ** 2),
        }

        for vf in video_files:
            catalog["files"].append({
                "dataset": dataset_name,
                "path": str(vf),
                "size_bytes": vf.stat().st_size,
            })

    return catalog


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def stage_clean(cfg: Config, run: PipelineRun) -> dict:
    """
    Data Cleaning Stage.

    Validates raw data, removes corrupted files, deduplicates, and flags
    quality issues.

    AWS equivalent:
      - Lambda: Per-file validation function
      - Glue Job: Batch cleaning and deduplication
      - S3: Move cleaned data to s3://eyehearu-data-lake/cleaned/
      - DynamoDB: Log cleaning decisions for each file

    Cleaning steps:
      1. Video validation (can be opened, has frames, meets min duration)
      2. Frame quality check (not black, not blurry, meets min resolution)
      3. Deduplication (hash-based duplicate detection)
      4. Class balance analysis
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: DATA CLEANING")
    logger.info("=" * 60)

    metrics = {
        "total_checked": 0,
        "valid": 0,
        "invalid": 0,
        "duplicates_removed": 0,
        "quality_issues": [],
    }

    images_dir = PROCESSED_DIR / "images" / "all"
    if not images_dir.exists():
        logger.warning(f"No extracted images found at {images_dir}. Run ingest first.")
        run.log_stage("clean", metrics)
        return metrics

    # ── Step 1: Validate extracted frames ────────────────────────────────
    logger.info("Validating extracted frames...")
    seen_hashes = set()

    for class_dir in sorted(images_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for img_path in class_dir.glob("*.jpg"):
            metrics["total_checked"] += 1

            # Check if image can be loaded
            img = cv2.imread(str(img_path))
            if img is None:
                logger.debug(f"  Invalid image (cannot load): {img_path}")
                img_path.unlink()
                metrics["invalid"] += 1
                continue

            h, w = img.shape[:2]

            # Check minimum dimensions
            if h < 64 or w < 64:
                logger.debug(f"  Too small ({w}x{h}): {img_path}")
                img_path.unlink()
                metrics["invalid"] += 1
                continue

            # Check for (near-)black frames
            if np.mean(img) < 10:
                logger.debug(f"  Black frame: {img_path}")
                img_path.unlink()
                metrics["invalid"] += 1
                continue

            # Check for extremely blurry frames (Laplacian variance)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 10:
                logger.debug(f"  Too blurry (var={laplacian_var:.1f}): {img_path}")
                img_path.unlink()
                metrics["invalid"] += 1
                continue

            # Deduplication via perceptual hash
            img_hash = _compute_image_hash(img)
            if img_hash in seen_hashes:
                logger.debug(f"  Duplicate: {img_path}")
                img_path.unlink()
                metrics["duplicates_removed"] += 1
                continue

            seen_hashes.add(img_hash)
            metrics["valid"] += 1

    # ── Step 2: Class balance analysis ───────────────────────────────────
    logger.info("Analyzing class balance...")
    class_counts = {}
    for class_dir in sorted(images_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
            class_counts[class_dir.name] = count

    if class_counts:
        counts = list(class_counts.values())
        metrics["class_balance"] = {
            "num_classes": len(class_counts),
            "min_samples": min(counts),
            "max_samples": max(counts),
            "mean_samples": np.mean(counts),
            "std_samples": np.std(counts),
            "min_class": min(class_counts, key=class_counts.get),
            "max_class": max(class_counts, key=class_counts.get),
        }

        # Flag severely imbalanced classes
        mean_count = np.mean(counts)
        for cls, cnt in class_counts.items():
            if cnt < mean_count * 0.2:
                metrics["quality_issues"].append(
                    f"Severely underrepresented class '{cls}': {cnt} samples "
                    f"(mean={mean_count:.0f})"
                )

    logger.info(f"Cleaning complete: {metrics['valid']} valid, "
                f"{metrics['invalid']} removed, "
                f"{metrics['duplicates_removed']} duplicates")

    run.log_stage("clean", metrics)
    return metrics


def _compute_image_hash(img: np.ndarray, hash_size: int = 8) -> str:
    """Compute a perceptual hash (average hash) for deduplication."""
    resized = cv2.resize(img, (hash_size, hash_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    bits = (gray > mean_val).flatten()
    hash_int = sum(bit << i for i, bit in enumerate(bits))
    return hex(hash_int)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: DATA TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

def stage_transform(cfg: Config, run: PipelineRun) -> dict:
    """
    Data Transformation Stage.

    Applies preprocessing transformations to cleaned images to prepare
    them for model training.

    AWS equivalent:
      - Lambda / Batch: Image processing at scale
      - S3: Store transformed images in s3://eyehearu-data-warehouse/processed/
      - Step Functions: Orchestrate parallel processing

    Transformation steps:
      1. Resize to consistent dimensions (224x224)
      2. Hand region detection and cropping (MediaPipe)
      3. Color normalization
      4. Data augmentation (rotations, color jitter) — applied during training
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: DATA TRANSFORMATION")
    logger.info("=" * 60)

    metrics = {
        "total_processed": 0,
        "hand_detected": 0,
        "hand_not_detected": 0,
        "resize_count": 0,
    }

    images_dir = PROCESSED_DIR / "images" / "all"
    if not images_dir.exists():
        logger.warning(f"No images found at {images_dir}")
        run.log_stage("transform", metrics)
        return metrics

    target_size = cfg.data.image_size  # 224

    # Initialize MediaPipe if available
    mp_hands = None
    hands_detector = None
    if HAS_MEDIAPIPE:
        mp_hands = mp.solutions.hands
        hands_detector = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
        )
        logger.info("MediaPipe Hands initialized for hand region detection")
    else:
        logger.warning("MediaPipe not available — skipping hand cropping")

    for class_dir in tqdm(sorted(images_dir.iterdir()), desc="Transforming"):
        if not class_dir.is_dir():
            continue

        for img_path in class_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            transformed = img

            # ── Hand region cropping ─────────────────────────────────
            if hands_detector is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands_detector.process(rgb)

                if results.multi_hand_landmarks:
                    # Get bounding box around all detected hands
                    h, w, _ = img.shape
                    all_x, all_y = [], []
                    for hand_landmarks in results.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            all_x.append(lm.x * w)
                            all_y.append(lm.y * h)

                    padding = 0.25
                    x_min, x_max = int(min(all_x)), int(max(all_x))
                    y_min, y_max = int(min(all_y)), int(max(all_y))
                    pad_x = int((x_max - x_min) * padding)
                    pad_y = int((y_max - y_min) * padding)

                    x_min = max(0, x_min - pad_x)
                    y_min = max(0, y_min - pad_y)
                    x_max = min(w, x_max + pad_x)
                    y_max = min(h, y_max + pad_y)

                    cropped = img[y_min:y_max, x_min:x_max]
                    if cropped.size > 0 and cropped.shape[0] > 32 and cropped.shape[1] > 32:
                        transformed = cropped
                        metrics["hand_detected"] += 1
                    else:
                        metrics["hand_not_detected"] += 1
                else:
                    metrics["hand_not_detected"] += 1

            # ── Resize to target dimensions ──────────────────────────
            transformed = cv2.resize(
                transformed, (target_size, target_size),
                interpolation=cv2.INTER_AREA
            )
            metrics["resize_count"] += 1

            # ── Save transformed image (overwrite original) ──────────
            cv2.imwrite(str(img_path), transformed, [cv2.IMWRITE_JPEG_QUALITY, 95])
            metrics["total_processed"] += 1

    if hands_detector is not None:
        hands_detector.close()

    logger.info(f"Transformation complete: {metrics['total_processed']} images processed")
    if HAS_MEDIAPIPE:
        logger.info(f"  Hand detected: {metrics['hand_detected']}, "
                     f"Not detected: {metrics['hand_not_detected']}")

    run.log_stage("transform", metrics)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: DATA LOADING (Split + Metadata)
# ═══════════════════════════════════════════════════════════════════════════════

def stage_load(cfg: Config, run: PipelineRun) -> dict:
    """
    Data Loading Stage.

    Splits processed data into train/val/test sets and generates
    final metadata for model training.

    AWS equivalent:
      - S3: Organize into s3://eyehearu-data-warehouse/train|val|test/
      - Glue: Update data catalog with new partitions
      - DynamoDB: Store dataset version metadata
      - SageMaker: Auto-detect new data for retraining trigger

    Loading steps:
      1. Stratified train/val/test split (80/10/10)
      2. Generate label_map.json
      3. Compute and save dataset statistics
      4. Generate dataset version manifest
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: DATA LOADING")
    logger.info("=" * 60)

    metrics = {
        "num_classes": 0,
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0,
        "total_samples": 0,
    }

    images_dir = PROCESSED_DIR / "images"
    all_dir = images_dir / "all"

    if not all_dir.exists():
        logger.warning(f"No images found at {all_dir}")
        run.log_stage("load", metrics)
        return metrics

    import random
    random.seed(42)

    # ── Step 1: Split into train/val/test ────────────────────────────────
    logger.info("Splitting data into train/val/test...")

    # Clean existing splits
    for split in ("train", "val", "test"):
        split_dir = images_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

    for class_dir in sorted(all_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(files)

        n = len(files)
        n_train = int(n * cfg.data.train_ratio)
        n_val = int(n * cfg.data.val_ratio)

        splits = {
            "train": files[:n_train],
            "val": files[n_train: n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            dest = images_dir / split_name / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dest / f.name)

        logger.info(
            f"  {class_dir.name}: {n} images → "
            f"train={len(splits['train'])}, "
            f"val={len(splits['val'])}, "
            f"test={len(splits['test'])}"
        )

        metrics["num_classes"] += 1
        metrics["train_samples"] += len(splits["train"])
        metrics["val_samples"] += len(splits["val"])
        metrics["test_samples"] += len(splits["test"])

    metrics["total_samples"] = (
        metrics["train_samples"] + metrics["val_samples"] + metrics["test_samples"]
    )

    # ── Step 2: Generate label map ───────────────────────────────────────
    logger.info("Generating label map...")
    classes = sorted(
        d.name for d in (images_dir / "train").iterdir() if d.is_dir()
    ) if (images_dir / "train").exists() else []

    label_map = {cls: idx for idx, cls in enumerate(classes)}
    label_map_path = PROCESSED_DIR / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    logger.info(f"Saved label map ({len(label_map)} classes) to {label_map_path}")

    # ── Step 3: Compute dataset statistics ───────────────────────────────
    logger.info("Computing dataset statistics...")
    stats = {"splits": {}, "label_map": label_map}

    for split in ("train", "val", "test"):
        split_dir = images_dir / split
        if not split_dir.exists():
            continue
        class_counts = {}
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                count = (
                    len(list(class_dir.glob("*.jpg")))
                    + len(list(class_dir.glob("*.png")))
                )
                class_counts[class_dir.name] = count
        stats["splits"][split] = {
            "total_images": sum(class_counts.values()),
            "num_classes": len(class_counts),
            "class_counts": class_counts,
        }

    stats_path = PROCESSED_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Dataset stats saved to {stats_path}")

    # ── Step 4: Generate dataset version manifest ────────────────────────
    manifest = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%d"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets_used": ["asl_citizen", "wlasl"],
        "pipeline_version": "1.0.0",
        "splits": {
            "train": metrics["train_samples"],
            "val": metrics["val_samples"],
            "test": metrics["test_samples"],
        },
        "num_classes": metrics["num_classes"],
        "image_size": cfg.data.image_size,
        "label_map_path": str(label_map_path),
    }
    manifest_path = METADATA_DIR / "dataset_manifest.json"
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Dataset manifest saved to {manifest_path}")

    run.log_stage("load", metrics)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5: REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def stage_report(cfg: Config, run: PipelineRun) -> dict:
    """
    Generate comprehensive pipeline report.

    AWS equivalent:
      - CloudWatch Dashboards: Real-time pipeline metrics
      - QuickSight: Visual reporting for stakeholders
      - SNS: Alert on quality issues
    """
    logger.info("=" * 60)
    logger.info("STAGE 5: REPORTING")
    logger.info("=" * 60)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_run_id": run.run_id,
        "stages": run.stages_completed,
        "metrics": run.metrics,
        "errors": run.errors,
        "data_summary": {},
    }

    # Summarize processed data
    images_dir = PROCESSED_DIR / "images"
    for split in ("train", "val", "test"):
        split_dir = images_dir / split
        if split_dir.exists():
            total = sum(
                len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
                for d in split_dir.iterdir()
                if d.is_dir()
            )
            classes = sum(1 for d in split_dir.iterdir() if d.is_dir())
            report["data_summary"][split] = {
                "total_images": total,
                "num_classes": classes,
            }

    report_path = METADATA_DIR / f"pipeline_report_{run.run_id}.json"
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE REPORT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Run ID: {run.run_id}")
    logger.info(f"Stages completed: {run.stages_completed}")
    for stage, stage_metrics in run.metrics.items():
        logger.info(f"\n  [{stage}]")
        for k, v in stage_metrics.items():
            if not isinstance(v, (dict, list)):
                logger.info(f"    {k}: {v}")
    if run.errors:
        logger.warning(f"\n  Errors: {len(run.errors)}")
        for err in run.errors:
            logger.warning(f"    - [{err['stage']}] {err['error']}")
    logger.info(f"\nFull report: {report_path}")

    run.log_stage("report", {"report_path": str(report_path)})
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

STAGES = {
    "ingest": stage_ingest,
    "clean": stage_clean,
    "transform": stage_transform,
    "load": stage_load,
    "report": stage_report,
}

STAGE_ORDER = ["ingest", "clean", "transform", "load", "report"]


def main():
    parser = argparse.ArgumentParser(
        description="Eye Hear U — Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --stage all          # Run full pipeline
  python pipeline.py --stage ingest       # Only download/ingest
  python pipeline.py --stage clean        # Only clean/validate
  python pipeline.py --stage transform    # Only transform images
  python pipeline.py --stage load         # Only split + generate metadata
  python pipeline.py --stage report       # Only generate reports
        """,
    )
    parser.add_argument(
        "--stage",
        choices=["all"] + STAGE_ORDER,
        default="all",
        help="Which pipeline stage to run (default: all)",
    )
    args = parser.parse_args()

    cfg = Config()
    run = PipelineRun()

    logger.info("=" * 60)
    logger.info("EYE HEAR U — DATA PROCESSING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Run ID:     {run.run_id}")
    logger.info(f"Stage:      {args.stage}")
    logger.info(f"Log file:   {log_file}")
    logger.info(f"Target vocab: {len(cfg.data.target_vocab)} signs")
    logger.info("")

    if args.stage == "all":
        stages_to_run = STAGE_ORDER
    else:
        stages_to_run = [args.stage]

    for stage_name in stages_to_run:
        try:
            logger.info(f"\n{'─'*60}")
            logger.info(f"Running stage: {stage_name}")
            logger.info(f"{'─'*60}")
            STAGES[stage_name](cfg, run)
        except Exception as e:
            run.log_error(stage_name, str(e))
            logger.exception(f"Stage '{stage_name}' failed")
            if args.stage == "all":
                logger.error("Stopping pipeline due to error.")
                break

    # Save run metadata
    run.save(METADATA_DIR)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Run ID:      {run.run_id}")
    logger.info(f"Duration:    {time.time() - run.start_time:.1f} seconds")
    logger.info(f"Stages:      {run.stages_completed}")
    logger.info(f"Errors:      {len(run.errors)}")
    logger.info(f"Log file:    {log_file}")


if __name__ == "__main__":
    main()
