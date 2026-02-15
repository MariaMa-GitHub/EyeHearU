"""
ASL Citizen dataset download and preprocessing script.

ASL Citizen is the largest crowdsourced isolated sign language dataset:
- Paper: https://arxiv.org/abs/2304.05934
- Project: https://www.microsoft.com/en-us/research/project/asl-citizen/
- Contains: ~84K videos, 2,731 distinct ASL signs, 52 signers
- License: Research use (see Microsoft's project page for details)

This script:
  1. Downloads the ASL Citizen dataset ZIP from Microsoft
  2. Extracts and organizes videos by gloss (sign label)
  3. Filters to our target vocabulary
  4. Extracts representative frames from each video
  5. Produces metadata CSV for tracking provenance

Prerequisites:
  pip install opencv-python tqdm requests pandas
"""

import csv
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Add parent paths for config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ml"))
from config import Config

# ─── Constants ───────────────────────────────────────────────────────────────

ASL_CITIZEN_URL = (
    "https://download.microsoft.com/download/b/8/8/"
    "b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
)

RAW_DIR = Path(__file__).parent.parent / "raw" / "asl_citizen"
PROCESSED_DIR = Path(__file__).parent.parent / "processed"
METADATA_DIR = Path(__file__).parent.parent / "metadata"

# Minimum video duration (seconds) to keep — filters out corrupt/tiny files
MIN_VIDEO_DURATION_SEC = 0.3
# Maximum frames to extract per video
MAX_FRAMES_PER_VIDEO = 5
# Minimum image dimension (pixels) — filters out too-small frames
MIN_FRAME_DIM = 64


# ─── Download ────────────────────────────────────────────────────────────────

def download_dataset(force: bool = False) -> Path:
    """
    Download the ASL Citizen ZIP file from Microsoft.

    Returns the path to the downloaded ZIP.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "ASL_Citizen.zip"

    if zip_path.exists() and not force:
        size_gb = zip_path.stat().st_size / (1024 ** 3)
        print(f"[SKIP] Dataset ZIP already exists at {zip_path} ({size_gb:.2f} GB)")
        return zip_path

    print(f"Downloading ASL Citizen dataset from Microsoft...")
    print(f"  URL: {ASL_CITIZEN_URL}")
    print(f"  Destination: {zip_path}")
    print("  (This is a large file — ~15 GB — it will take a while.)\n")

    # Stream download with progress bar
    resp = requests.get(ASL_CITIZEN_URL, stream=True)
    resp.raise_for_status()
    total_size = int(resp.headers.get("content-length", 0))

    with open(zip_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Downloading"
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"\nDownload complete: {zip_path}")
    return zip_path


def extract_dataset(zip_path: Path, force: bool = False) -> Path:
    """
    Extract the ASL Citizen ZIP file.

    Returns the path to the extracted directory.
    """
    extract_dir = RAW_DIR / "extracted"

    if extract_dir.exists() and not force:
        print(f"[SKIP] Already extracted to {extract_dir}")
        return extract_dir

    print(f"Extracting {zip_path}...")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="Extracting"):
            zf.extract(member, extract_dir)

    print(f"Extracted {len(members)} files to {extract_dir}")
    return extract_dir


# ─── Metadata Parsing ────────────────────────────────────────────────────────

def parse_metadata(extract_dir: Path) -> pd.DataFrame:
    """
    Parse ASL Citizen metadata to build a dataframe of all videos.

    ASL Citizen files are typically named: <signer_id>_<gloss>.mp4
    The dataset also includes CSV/JSON metadata files.

    Returns a DataFrame with columns: [video_path, gloss, signer_id, filename]
    """
    print("Scanning extracted files for videos and metadata...")

    # Find all video files
    video_files = []
    for ext in ("*.mp4", "*.mov", "*.avi", "*.webm"):
        video_files.extend(extract_dir.rglob(ext))

    print(f"  Found {len(video_files)} video files")

    # Check for CSV/JSON metadata
    csv_files = list(extract_dir.rglob("*.csv"))
    json_files = list(extract_dir.rglob("*.json"))

    records = []

    # Try to use official metadata if available
    if csv_files:
        print(f"  Found metadata CSV: {csv_files[0]}")
        try:
            meta_df = pd.read_csv(csv_files[0])
            print(f"  Metadata columns: {list(meta_df.columns)}")
            # Will merge with video paths below
        except Exception as e:
            print(f"  Warning: Could not parse CSV metadata: {e}")

    # Parse video filenames as fallback / primary source
    for vf in video_files:
        filename = vf.stem  # e.g., "signer01_hello" or "hello_001"
        parts = filename.split("_")

        # Try to extract gloss and signer ID from filename
        # ASL Citizen naming convention varies; handle common patterns
        gloss = parts[-1].lower() if len(parts) >= 2 else filename.lower()
        signer_id = parts[0] if len(parts) >= 2 else "unknown"

        records.append({
            "video_path": str(vf),
            "filename": vf.name,
            "gloss": gloss,
            "signer_id": signer_id,
            "file_size_bytes": vf.stat().st_size,
        })

    df = pd.DataFrame(records)
    print(f"  Built metadata for {len(df)} videos across {df['gloss'].nunique()} glosses")
    return df


def filter_target_glosses(df: pd.DataFrame, target_vocab: list[str]) -> pd.DataFrame:
    """
    Filter the video metadata to only include our target vocabulary.

    Returns filtered DataFrame + prints coverage report.
    """
    target_set = set(v.lower() for v in target_vocab)

    # Exact match first
    exact_match = df[df["gloss"].isin(target_set)]
    found_glosses = set(exact_match["gloss"].unique())
    missing = target_set - found_glosses

    print(f"\n{'='*60}")
    print(f"TARGET VOCABULARY COVERAGE REPORT")
    print(f"{'='*60}")
    print(f"  Target glosses:  {len(target_set)}")
    print(f"  Found (exact):   {len(found_glosses)}")
    print(f"  Missing:         {len(missing)}")

    if found_glosses:
        print(f"\n  Found glosses: {sorted(found_glosses)}")
    if missing:
        print(f"\n  Missing glosses (need alternative data): {sorted(missing)}")

    # Per-gloss video counts
    print(f"\n  Videos per found gloss:")
    for gloss in sorted(found_glosses):
        count = len(exact_match[exact_match["gloss"] == gloss])
        print(f"    {gloss:20s}: {count:5d} videos")

    return exact_match


# ─── Frame Extraction ────────────────────────────────────────────────────────

def validate_video(video_path: str) -> dict | None:
    """
    Validate a video file and return its properties.

    Returns dict with {fps, frame_count, duration, width, height} or None if invalid.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or frame_count <= 0:
        return None

    duration = frame_count / fps

    if duration < MIN_VIDEO_DURATION_SEC:
        return None
    if width < MIN_FRAME_DIM or height < MIN_FRAME_DIM:
        return None

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": width,
        "height": height,
    }


def extract_frames(
    video_path: str,
    output_dir: Path,
    max_frames: int = MAX_FRAMES_PER_VIDEO,
    prefix: str = "",
) -> int:
    """
    Extract evenly-spaced frames from a video file.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        max_frames: Maximum number of frames to extract
        prefix: Filename prefix for extracted frames

    Returns:
        Number of frames successfully saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0

    # Pick evenly-spaced frame indices (skip first/last 10% to avoid black frames)
    start_idx = max(1, int(total_frames * 0.1))
    end_idx = min(total_frames - 1, int(total_frames * 0.9))
    usable_frames = end_idx - start_idx

    if usable_frames <= 0:
        # Very short video — just grab middle frame
        indices = [total_frames // 2]
    elif usable_frames <= max_frames:
        indices = list(range(start_idx, end_idx + 1))
    else:
        indices = [
            start_idx + int(i * usable_frames / max_frames)
            for i in range(max_frames)
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            frame_path = output_dir / f"{prefix}frame_{idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

    cap.release()
    return saved


# ─── Label Map ───────────────────────────────────────────────────────────────

def build_label_map(glosses: list[str]) -> dict:
    """Create a gloss → integer label mapping (sorted alphabetically)."""
    return {gloss: idx for idx, gloss in enumerate(sorted(glosses))}


# ─── Data Quality Report ─────────────────────────────────────────────────────

def generate_quality_report(df: pd.DataFrame, output_path: Path):
    """
    Generate a data quality report as JSON.

    Includes:
    - Total videos, total glosses, total signers
    - Per-gloss statistics (count, avg file size)
    - Data quality flags (missing videos, small files, etc.)
    """
    report = {
        "dataset": "ASL Citizen",
        "total_videos": len(df),
        "total_glosses": df["gloss"].nunique(),
        "total_signers": df["signer_id"].nunique(),
        "glosses": {},
        "quality_flags": [],
    }

    for gloss, group in df.groupby("gloss"):
        report["glosses"][gloss] = {
            "video_count": len(group),
            "signer_count": group["signer_id"].nunique(),
            "avg_file_size_kb": group["file_size_bytes"].mean() / 1024,
        }

    # Quality flags
    for gloss, info in report["glosses"].items():
        if info["video_count"] < 5:
            report["quality_flags"].append(
                f"Low sample count for '{gloss}': only {info['video_count']} videos"
            )
        if info["signer_count"] < 2:
            report["quality_flags"].append(
                f"Low signer diversity for '{gloss}': only {info['signer_count']} signer(s)"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nData quality report saved to {output_path}")
    print(f"  Quality flags: {len(report['quality_flags'])}")
    return report


# ─── Provenance Tracking ─────────────────────────────────────────────────────

def save_provenance(df: pd.DataFrame, output_path: Path):
    """
    Save a provenance CSV tracking which videos were processed.

    Columns: video_path, gloss, signer_id, file_size_bytes, processed (bool)
    This supports data lineage requirements for the pipeline.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Provenance metadata saved to {output_path}")


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    cfg = Config()

    print("=" * 60)
    print("ASL CITIZEN DATASET PIPELINE")
    print("=" * 60)
    print(f"Target vocabulary: {len(cfg.data.target_vocab)} signs")
    print(f"Raw directory:     {RAW_DIR}")
    print(f"Processed dir:     {PROCESSED_DIR}")
    print()

    # ── Step 1: Download ─────────────────────────────────────────────────
    print("\n[STEP 1/6] Downloading ASL Citizen dataset...")
    zip_path = download_dataset()

    # ── Step 2: Extract ──────────────────────────────────────────────────
    print("\n[STEP 2/6] Extracting dataset...")
    extract_dir = extract_dataset(zip_path)

    # ── Step 3: Parse metadata ───────────────────────────────────────────
    print("\n[STEP 3/6] Parsing metadata...")
    all_videos_df = parse_metadata(extract_dir)

    if all_videos_df.empty:
        print("\nERROR: No videos found in extracted directory.")
        print(f"Please check the contents of: {extract_dir}")
        return

    # ── Step 4: Filter to target vocabulary ──────────────────────────────
    print("\n[STEP 4/6] Filtering to target vocabulary...")
    target_df = filter_target_glosses(all_videos_df, cfg.data.target_vocab)

    # ── Step 5: Extract frames ───────────────────────────────────────────
    print(f"\n[STEP 5/6] Extracting frames from {len(target_df)} videos...")
    images_dir = PROCESSED_DIR / "images"

    total_frames = 0
    skipped = 0
    errors = 0

    for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc="Extracting"):
        gloss = row["gloss"]
        video_path = row["video_path"]

        # Validate video first
        props = validate_video(video_path)
        if props is None:
            skipped += 1
            continue

        # Extract frames into gloss directory
        # (train/val/test split happens in preprocess.py)
        output_dir = images_dir / "all" / gloss
        prefix = f"{row['signer_id']}_"

        try:
            n_frames = extract_frames(video_path, output_dir, prefix=prefix)
            total_frames += n_frames
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error processing {video_path}: {e}")

    print(f"\nFrame extraction complete:")
    print(f"  Total frames saved: {total_frames}")
    print(f"  Videos skipped:     {skipped}")
    print(f"  Errors:             {errors}")

    # ── Step 6: Generate metadata & reports ──────────────────────────────
    print("\n[STEP 6/6] Generating metadata and quality reports...")

    # Label map
    found_glosses = list(target_df["gloss"].unique())
    label_map = build_label_map(found_glosses)
    label_map_path = PROCESSED_DIR / "label_map.json"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved label map ({len(label_map)} classes) to {label_map_path}")

    # Quality report
    report_path = METADATA_DIR / "asl_citizen_quality_report.json"
    generate_quality_report(target_df, report_path)

    # Provenance
    provenance_path = METADATA_DIR / "asl_citizen_provenance.csv"
    save_provenance(target_df, provenance_path)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Dataset:         ASL Citizen")
    print(f"  Total glosses:   {len(found_glosses)}")
    print(f"  Total videos:    {len(target_df)}")
    print(f"  Total frames:    {total_frames}")
    print(f"  Label map:       {label_map_path}")
    print(f"  Quality report:  {report_path}")
    print(f"  Provenance:      {provenance_path}")
    print(f"  Images dir:      {images_dir}")
    print("\nNext step: Run preprocess.py to split into train/val/test")


if __name__ == "__main__":
    main()
