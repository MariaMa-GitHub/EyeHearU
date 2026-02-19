"""
Main Data Processing Pipeline Orchestrator — Eye Hear U

This script orchestrates the complete data processing pipeline for the
pose-based ST-GCN sign language recognition system.  Multiple datasets
(ASL Citizen, WLASL, MS-ASL) are ingested, validated, and
combined into **one unified dataset** of pose-keypoint sequences.

  ASL Citizen ─┐
  WLASL ───────┼─▶ Combine ─▶ Validate ─▶ Pose Extract ─▶ Split ─▶ train/val/test
  MS-ASL ──────┘   metadata    videos      → .npy files           + label_map.json

Pipeline Stages:
  Stage 1: Data Ingestion       — Download datasets + build combined_metadata.csv
  Stage 2: Data Cleaning        — Validate raw videos (codec, duration, resolution)
  Stage 3: Pose Extraction      — MediaPipe Holistic → .npy pose keypoint files
  Stage 4: Data Splitting       — Video-level train/val/test CSV splits + label map
  Stage 5: Reporting            — Pipeline execution report and quality metrics

Datasets:
  - ASL Citizen  metadata.csv            (user, filename, gloss)
  - WLASL        WLASL_v0.3.json         (gloss → instances[])
  - MS-ASL       MSASL_{split}.json      (url, text, signer, label, ...)

All sources are normalised into combined_metadata.csv with columns:
  user, filename, gloss, dataset

Usage:
  python pipeline.py --stage all          # Run full pipeline
  python pipeline.py --stage ingest       # Only download/ingest + combine
  python pipeline.py --stage clean        # Only validate videos
  python pipeline.py --stage pose         # Only extract poses → .npy
  python pipeline.py --stage split        # Only split + generate metadata
  python pipeline.py --stage report       # Only generate reports

Prerequisites:
  pip install opencv-python tqdm requests pandas mediapipe numpy
"""

import argparse
import csv
import json
import logging
import random
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
      - data/raw/ms_asl/       → MS-ASL videos
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

    # Ingest MS-ASL
    ms_asl_dir = RAW_DIR / "ms_asl"
    if ms_asl_dir.exists():
        video_count = sum(1 for _ in ms_asl_dir.rglob("*.mp4"))
        logger.info(f"MS-ASL: {video_count} videos found in {ms_asl_dir}")
        metrics["datasets_ingested"].append("ms_asl")
        metrics["ms_asl_videos"] = video_count
        metrics["total_raw_files"] += video_count
    else:
        logger.warning(
            f"MS-ASL not found at {ms_asl_dir}. "
            "Run download_ms_asl.py first."
        )

    # Build raw data catalog
    catalog = _build_raw_catalog()
    catalog_path = METADATA_DIR / "raw_data_catalog.json"
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)
    metrics["catalog_entries"] = len(catalog.get("files", []))

    # Build combined metadata CSV across all datasets
    combined_df = _build_combined_metadata()
    if combined_df is not None:
        combined_path = PROCESSED_DIR / "combined_metadata.csv"
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(combined_path, index=False, header=False)
        metrics["combined_videos"] = len(combined_df)
        metrics["combined_glosses"] = int(combined_df["gloss"].nunique())
        logger.info(
            f"Combined metadata: {len(combined_df)} videos, "
            f"{combined_df['gloss'].nunique()} glosses → {combined_path}"
        )

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


def _build_combined_metadata() -> pd.DataFrame | None:
    """
    Build a single unified metadata DataFrame (user, filename, gloss, dataset)
    from all available dataset sources so that subsequent stages (pose extraction,
    splitting) operate on one combined dataset.

    Supported sources:
      - ASL Citizen: metadata.csv             (user, filename, gloss)
      - WLASL:       WLASL_v0.3.json          (gloss → instances[])
      - MS-ASL:      MSASL_{train,val,test}.json  (url, text, signer, ...)
    """
    frames = []

    # ── ASL Citizen ───────────────────────────────────────────────────────
    asl_csv = RAW_DIR / "asl_citizen" / "metadata.csv"
    if asl_csv.exists():
        df = pd.read_csv(asl_csv, header=None, names=["user", "filename", "gloss"])
        df["dataset"] = "asl_citizen"
        # Ensure filenames are relative to raw/asl_citizen/videos/
        frames.append(df)
        logger.info(f"ASL Citizen: loaded {len(df)} entries from {asl_csv}")

    # ── WLASL ─────────────────────────────────────────────────────────────
    wlasl_json = RAW_DIR / "wlasl" / "WLASL_v0.3.json"
    if wlasl_json.exists():
        with open(wlasl_json) as f:
            wlasl_data = json.load(f)

        rows = []
        videos_dir = RAW_DIR / "wlasl" / "videos"
        for entry in wlasl_data:
            gloss = entry.get("gloss", "")
            for inst in entry.get("instances", []):
                video_id = inst.get("video_id", "")
                video_file = f"{video_id}.mp4"
                if (videos_dir / video_file).exists():
                    signer = inst.get("signer_id", "unknown")
                    rows.append({
                        "user": str(signer),
                        "filename": video_file,
                        "gloss": gloss,
                        "dataset": "wlasl",
                    })
        if rows:
            wlasl_df = pd.DataFrame(rows)
            frames.append(wlasl_df)
            logger.info(f"WLASL: loaded {len(wlasl_df)} entries from {wlasl_json}")
    elif (RAW_DIR / "wlasl").exists():
        # Fallback: no JSON, but videos exist — build entries from filenames
        wlasl_vids = list((RAW_DIR / "wlasl").rglob("*.mp4"))
        if wlasl_vids:
            rows = []
            for vp in wlasl_vids:
                rows.append({
                    "user": "unknown",
                    "filename": vp.name,
                    "gloss": vp.stem,
                    "dataset": "wlasl",
                })
            wlasl_df = pd.DataFrame(rows)
            frames.append(wlasl_df)
            logger.info(f"WLASL (fallback glob): {len(wlasl_df)} videos")

    # ── MS-ASL ────────────────────────────────────────────────────────────
    # MS-ASL provides MSASL_train.json, MSASL_val.json, MSASL_test.json.
    # Each entry: {"url", "start_time", "end_time", "label", "text", "signer", ...}
    ms_asl_dir = RAW_DIR / "ms_asl"
    ms_asl_jsons = [
        ms_asl_dir / f"MSASL_{split}.json"
        for split in ("train", "val", "test")
    ]
    ms_asl_found = [p for p in ms_asl_jsons if p.exists()]

    if ms_asl_found:
        rows = []
        videos_dir = ms_asl_dir / "videos"
        for json_path in ms_asl_found:
            with open(json_path) as f:
                entries = json.load(f)
            for entry in entries:
                gloss = entry.get("text", entry.get("clean_text", ""))
                signer = entry.get("signer", "unknown")
                # Video filenames typically match the URL basename or a
                # sequential naming scheme; adapt to your download script.
                url = entry.get("url", "")
                video_file = entry.get("file", url.split("/")[-1] if url else "")
                if not video_file:
                    continue
                if not video_file.endswith(".mp4"):
                    video_file = f"{video_file}.mp4"
                if (videos_dir / video_file).exists():
                    rows.append({
                        "user": str(signer),
                        "filename": video_file,
                        "gloss": gloss,
                        "dataset": "ms_asl",
                    })
        if rows:
            ms_asl_df = pd.DataFrame(rows)
            frames.append(ms_asl_df)
            logger.info(f"MS-ASL: loaded {len(ms_asl_df)} entries from {len(ms_asl_found)} JSON files")
    elif ms_asl_dir.exists():
        ms_asl_vids = list(ms_asl_dir.rglob("*.mp4"))
        if ms_asl_vids:
            rows = []
            for vp in ms_asl_vids:
                rows.append({
                    "user": "unknown",
                    "filename": vp.name,
                    "gloss": vp.stem,
                    "dataset": "ms_asl",
                })
            ms_asl_df = pd.DataFrame(rows)
            frames.append(ms_asl_df)
            logger.info(f"MS-ASL (fallback glob): {len(ms_asl_df)} videos")

    if not frames:
        logger.warning("No metadata found for any dataset.")
        return None

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Combined dataset: {len(combined)} videos across "
        f"{combined['dataset'].nunique()} sources, "
        f"{combined['gloss'].nunique()} unique glosses"
    )
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

MIN_DURATION_SEC = 0.3
MIN_RESOLUTION = 64

def stage_clean(cfg: Config, run: PipelineRun) -> dict:
    """
    Data Cleaning Stage.

    Validates raw videos — removes corrupted or unusable files and flags
    quality issues.  Operates at the video level (no frame extraction).

    AWS equivalent:
      - Lambda: Per-file video validation
      - Glue Job: Batch validation
      - S3: Move validated videos to s3://eyehearu-data-lake/cleaned/
      - DynamoDB: Log validation decisions for each file

    Cleaning rules:
      1. Video must be decodable (codec / corruption check)
      2. Duration ≥ 0.3 seconds
      3. Resolution ≥ 64×64
      4. Class balance analysis (flag under-represented glosses)
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: DATA CLEANING")
    logger.info("=" * 60)

    metrics = {
        "total_checked": 0,
        "valid": 0,
        "invalid_corrupt": 0,
        "invalid_short": 0,
        "invalid_resolution": 0,
        "quality_issues": [],
    }

    valid_videos = []

    for dataset_dir in sorted(RAW_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue

        video_files = list(dataset_dir.rglob("*.mp4"))
        if not video_files:
            continue

        logger.info(f"Validating {len(video_files)} videos in {dataset_dir.name}...")

        for vpath in tqdm(video_files, desc=f"Cleaning {dataset_dir.name}"):
            metrics["total_checked"] += 1

            cap = cv2.VideoCapture(str(vpath))
            if not cap.isOpened():
                logger.debug(f"  Cannot open: {vpath}")
                metrics["invalid_corrupt"] += 1
                cap.release()
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            # Read one frame to verify the video is actually decodable
            ret, _ = cap.read()
            cap.release()

            if not ret:
                logger.debug(f"  Cannot decode frames: {vpath}")
                metrics["invalid_corrupt"] += 1
                continue

            if duration < MIN_DURATION_SEC:
                logger.debug(f"  Too short ({duration:.2f}s): {vpath}")
                metrics["invalid_short"] += 1
                continue

            if width < MIN_RESOLUTION or height < MIN_RESOLUTION:
                logger.debug(f"  Too small ({width}x{height}): {vpath}")
                metrics["invalid_resolution"] += 1
                continue

            metrics["valid"] += 1
            valid_videos.append({
                "path": str(vpath),
                "dataset": dataset_dir.name,
                "duration_sec": round(duration, 2),
                "fps": round(fps, 1),
                "num_frames": frame_count,
                "resolution": f"{width}x{height}",
            })

    # ── Class balance analysis ────────────────────────────────────────────
    combined_csv = PROCESSED_DIR / "combined_metadata.csv"
    if combined_csv.exists():
        logger.info("Analyzing class balance from combined metadata...")
        df = pd.read_csv(combined_csv, header=None, names=["user", "filename", "gloss", "dataset"])
        class_counts = df["gloss"].value_counts().to_dict()
        counts = list(class_counts.values())
        metrics["class_balance"] = {
            "num_classes": len(class_counts),
            "min_samples": int(min(counts)),
            "max_samples": int(max(counts)),
            "mean_samples": float(np.mean(counts)),
            "min_class": min(class_counts, key=class_counts.get),
            "max_class": max(class_counts, key=class_counts.get),
        }
        mean_count = np.mean(counts)
        for cls, cnt in class_counts.items():
            if cnt < 5:
                metrics["quality_issues"].append(
                    f"Class '{cls}' has only {cnt} videos (needs supplementation)"
                )

    # Save validated video catalog
    catalog_path = METADATA_DIR / "validated_video_catalog.json"
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(catalog_path, "w") as f:
        json.dump(valid_videos, f, indent=2)

    total_invalid = (
        metrics["invalid_corrupt"]
        + metrics["invalid_short"]
        + metrics["invalid_resolution"]
    )
    logger.info(
        f"Cleaning complete: {metrics['valid']} valid, {total_invalid} removed "
        f"(corrupt={metrics['invalid_corrupt']}, short={metrics['invalid_short']}, "
        f"small={metrics['invalid_resolution']})"
    )

    run.log_stage("clean", metrics)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: POSE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

NUM_POSE = 33
NUM_HAND = 21
NUM_FACE = 468
NUM_KEYPOINTS = NUM_POSE + NUM_HAND + NUM_HAND + NUM_FACE  # 543


def _extract_pose_from_video(video_path: str) -> np.ndarray:
    """
    Extract MediaPipe Holistic pose keypoints from a video file.

    Returns an ndarray of shape (T, 543, 2) where T is the frame count,
    543 = 33 pose + 21 right hand + 21 left hand + 468 face landmarks,
    and 2 = (x, y) normalised coordinates.
    """
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    feature = np.zeros((total_frames, NUM_KEYPOINTS, 2))

    with mp_holistic.Holistic(
        static_image_mode=False, min_detection_confidence=0.5
    ) as holistic:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                for i in range(NUM_POSE):
                    feature[idx][i][0] = results.pose_landmarks.landmark[i].x
                    feature[idx][i][1] = results.pose_landmarks.landmark[i].y

            offset = NUM_POSE
            if results.right_hand_landmarks:
                for i in range(NUM_HAND):
                    feature[idx][i + offset][0] = results.right_hand_landmarks.landmark[i].x
                    feature[idx][i + offset][1] = results.right_hand_landmarks.landmark[i].y

            offset = NUM_POSE + NUM_HAND
            if results.left_hand_landmarks:
                for i in range(NUM_HAND):
                    feature[idx][i + offset][0] = results.left_hand_landmarks.landmark[i].x
                    feature[idx][i + offset][1] = results.left_hand_landmarks.landmark[i].y

            offset = NUM_POSE + NUM_HAND + NUM_HAND
            if results.face_landmarks:
                for i in range(NUM_FACE):
                    feature[idx][i + offset][0] = results.face_landmarks.landmark[i].x
                    feature[idx][i + offset][1] = results.face_landmarks.landmark[i].y

            idx += 1

    cap.release()
    return feature[:idx]


def stage_pose(cfg: Config, run: PipelineRun) -> dict:
    """
    Pose Extraction Stage.

    Extracts MediaPipe Holistic pose keypoints from every validated video
    and saves them as .npy files.  Also generates pose_mapping.csv linking
    video filenames to their .npy paths.

    AWS equivalent:
      - AWS Batch (GPU): MediaPipe extraction at scale
      - S3: Store .npy files in s3://eyehearu-data-lake/processed/poses/

    Output per video:
      .npy file with shape (T, 543, 2)  — T = number of frames
        543 keypoints = 33 pose + 21 R hand + 21 L hand + 468 face
        2 channels   = (x, y) normalised coordinates
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: POSE EXTRACTION")
    logger.info("=" * 60)

    if not HAS_MEDIAPIPE:
        msg = "MediaPipe is required for pose extraction. pip install mediapipe"
        run.log_error("pose", msg)
        logger.error(msg)
        return {"error": msg}

    metrics = {
        "total_videos": 0,
        "poses_extracted": 0,
        "failed": 0,
    }

    poses_dir = PROCESSED_DIR / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    # Collect videos to process from the combined metadata
    video_entries = []
    combined_csv = PROCESSED_DIR / "combined_metadata.csv"

    DATASET_VIDEO_ROOTS = {
        "asl_citizen": RAW_DIR / "asl_citizen" / "videos",
        "wlasl": RAW_DIR / "wlasl" / "videos",
        "ms_asl": RAW_DIR / "ms_asl" / "videos",
    }

    if combined_csv.exists():
        df = pd.read_csv(
            combined_csv, header=None,
            names=["user", "filename", "gloss", "dataset"],
        )
        for _, row in df.iterrows():
            root = DATASET_VIDEO_ROOTS.get(row["dataset"], RAW_DIR / row["dataset"])
            vpath = root / row["filename"]
            if vpath.exists():
                # Use dataset/filename as the unique key
                unique_name = f"{row['dataset']}/{row['filename']}"
                video_entries.append((str(vpath), unique_name))
    else:
        logger.warning(
            "combined_metadata.csv not found — falling back to glob. "
            "Run the ingest stage first to build combined metadata."
        )
        for vpath in RAW_DIR.rglob("*.mp4"):
            rel = vpath.relative_to(RAW_DIR)
            video_entries.append((str(vpath), str(rel)))

    metrics["total_videos"] = len(video_entries)
    logger.info(f"Found {len(video_entries)} videos to process")

    # Pose extraction loop
    pose_mapping = []
    for video_path, video_name in tqdm(video_entries, desc="Extracting poses"):
        npy_name = video_name.replace("/", "_").replace("\\", "_")
        if npy_name.endswith(".mp4"):
            npy_name = npy_name[:-4]
        npy_path = poses_dir / f"{npy_name}.npy"

        if npy_path.exists():
            pose_mapping.append((video_name, str(npy_path.relative_to(PROCESSED_DIR))))
            metrics["poses_extracted"] += 1
            continue

        try:
            features = _extract_pose_from_video(video_path)
            np.save(str(npy_path), features)
            pose_mapping.append((video_name, str(npy_path.relative_to(PROCESSED_DIR))))
            metrics["poses_extracted"] += 1
        except Exception as e:
            logger.debug(f"  Failed to extract pose for {video_name}: {e}")
            metrics["failed"] += 1

    # Write pose_mapping.csv
    mapping_path = PROCESSED_DIR / "pose_mapping.csv"
    with open(mapping_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_filename", "pose_npy_path"])
        writer.writerows(pose_mapping)

    logger.info(
        f"Pose extraction complete: {metrics['poses_extracted']} extracted, "
        f"{metrics['failed']} failed"
    )
    run.log_stage("pose", metrics)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: DATA SPLITTING
# ═══════════════════════════════════════════════════════════════════════════════

def stage_split(cfg: Config, run: PipelineRun) -> dict:
    """
    Data Splitting Stage.

    Splits video metadata into train/val/test CSV files and generates
    a label map and dataset statistics.  Splitting happens at the **video
    level** — each row in the split CSVs references a video, not a frame.

    AWS equivalent:
      - Glue: Partition management
      - S3: Store split CSVs alongside raw metadata
      - DynamoDB: Store dataset version metadata
      - SageMaker: Auto-detect new data for retraining trigger

    Steps:
      1. Stratified train/val/test split (80/10/10, by gloss class)
      2. Write train.csv, val.csv, test.csv
      3. Generate label_map.json  (gloss → integer index)
      4. Compute and save dataset_stats.json
      5. Generate dataset version manifest
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: DATA SPLITTING")
    logger.info("=" * 60)

    metrics = {
        "num_classes": 0,
        "train_videos": 0,
        "val_videos": 0,
        "test_videos": 0,
        "total_videos": 0,
    }

    combined_csv = PROCESSED_DIR / "combined_metadata.csv"
    if not combined_csv.exists():
        logger.warning(
            f"Combined metadata not found at {combined_csv}. "
            "Run the ingest stage first to build combined metadata."
        )
        run.log_stage("split", metrics)
        return metrics

    df = pd.read_csv(
        combined_csv, header=None,
        names=["user", "filename", "gloss", "dataset"],
    )
    logger.info(
        f"Loaded {len(df)} video entries from {combined_csv} "
        f"({df['dataset'].value_counts().to_dict()})"
    )

    random.seed(42)

    # ── Step 1: Stratified split by gloss ─────────────────────────────────
    logger.info("Splitting videos into train/val/test (80/10/10, stratified by gloss)...")

    train_rows, val_rows, test_rows = [], [], []

    for gloss, group in df.groupby("gloss"):
        rows = group.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(rows)
        n_train = int(n * cfg.data.train_ratio)
        n_val = int(n * cfg.data.val_ratio)

        train_rows.append(rows.iloc[:n_train])
        val_rows.append(rows.iloc[n_train: n_train + n_val])
        test_rows.append(rows.iloc[n_train + n_val:])

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df = pd.concat(val_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    # ── Step 2: Write split CSVs ──────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False, header=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False, header=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False, header=False)

    metrics["train_videos"] = len(train_df)
    metrics["val_videos"] = len(val_df)
    metrics["test_videos"] = len(test_df)
    metrics["total_videos"] = len(df)

    logger.info(
        f"  Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    # ── Step 3: Generate label map ────────────────────────────────────────
    logger.info("Generating label map...")
    glosses = sorted(df["gloss"].unique())
    label_map = {gloss: idx for idx, gloss in enumerate(glosses)}
    metrics["num_classes"] = len(label_map)

    label_map_path = PROCESSED_DIR / "label_map.json"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    logger.info(f"Saved label map ({len(label_map)} classes) to {label_map_path}")

    # ── Step 4: Compute dataset statistics ────────────────────────────────
    logger.info("Computing dataset statistics...")
    stats = {"label_map": label_map, "splits": {}}

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        class_counts = split_df["gloss"].value_counts().to_dict()
        stats["splits"][split_name] = {
            "total_videos": len(split_df),
            "num_classes": len(class_counts),
            "class_counts": {k: int(v) for k, v in class_counts.items()},
        }

    stats_path = PROCESSED_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Dataset stats saved to {stats_path}")

    # ── Step 5: Generate dataset version manifest ─────────────────────────
    manifest = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%d"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets_used": ["asl_citizen", "wlasl", "ms_asl"],
        "pipeline_version": "2.0.0",
        "model_type": "st_gcn",
        "splits": {
            "train": metrics["train_videos"],
            "val": metrics["val_videos"],
            "test": metrics["test_videos"],
        },
        "num_classes": metrics["num_classes"],
        "pose_keypoints": NUM_KEYPOINTS,
        "label_map_path": str(label_map_path),
    }
    manifest_path = METADATA_DIR / "dataset_manifest.json"
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Dataset manifest saved to {manifest_path}")

    run.log_stage("split", metrics)
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

    # Summarize pose data
    poses_dir = PROCESSED_DIR / "poses"
    if poses_dir.exists():
        npy_files = list(poses_dir.glob("*.npy"))
        report["data_summary"]["poses"] = {
            "total_pose_files": len(npy_files),
            "total_size_mb": round(
                sum(f.stat().st_size for f in npy_files) / (1024 ** 2), 1
            ),
        }

    # Summarize split CSVs
    for split in ("train", "val", "test"):
        csv_path = PROCESSED_DIR / f"{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, header=None, names=["user", "filename", "gloss", "dataset"])
            report["data_summary"][split] = {
                "total_videos": len(df),
                "num_classes": df["gloss"].nunique(),
            }

    # Summarize label map
    label_map_path = PROCESSED_DIR / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            lm = json.load(f)
        report["data_summary"]["label_map_classes"] = len(lm)

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
    "pose": stage_pose,
    "split": stage_split,
    "report": stage_report,
}

STAGE_ORDER = ["ingest", "clean", "pose", "split", "report"]


def main():
    parser = argparse.ArgumentParser(
        description="Eye Hear U — Data Processing Pipeline (ST-GCN pose-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --stage all          # Run full pipeline
  python pipeline.py --stage ingest       # Only download/ingest
  python pipeline.py --stage clean        # Only validate videos
  python pipeline.py --stage pose         # Only extract poses → .npy
  python pipeline.py --stage split        # Only split + generate metadata
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
