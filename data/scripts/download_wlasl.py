"""
WLASL dataset download and preprocessing script.

WLASL (Word-Level American Sign Language) is the primary training dataset.
- Paper: https://arxiv.org/abs/1910.11006
- GitHub: https://github.com/dxli94/WLASL
- Contains: 2000 ASL glosses, ~21K video samples

This script:
  1. Downloads the WLASL metadata (JSON with video URLs)
  2. Downloads videos for our target vocabulary
  3. Extracts representative frames from each video
  4. Organizes into train/val/test splits

Prerequisites:
  pip install yt-dlp opencv-python tqdm requests
"""

import json
import os
import sys
from pathlib import Path

import cv2
import requests
from tqdm import tqdm

# Add parent paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ml"))
from config import Config

# Constants
WLASL_JSON_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
RAW_DIR = Path(__file__).parent.parent / "raw" / "wlasl"
PROCESSED_DIR = Path(__file__).parent.parent / "processed"


def download_metadata():
    """Download the WLASL annotation JSON."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = RAW_DIR / "WLASL_v0.3.json"

    if meta_path.exists():
        print(f"Metadata already exists at {meta_path}")
    else:
        print("Downloading WLASL metadata...")
        resp = requests.get(WLASL_JSON_URL)
        resp.raise_for_status()
        with open(meta_path, "w") as f:
            json.dump(resp.json(), f, indent=2)
        print(f"Saved to {meta_path}")

    with open(meta_path) as f:
        return json.load(f)


def filter_target_glosses(metadata: list, target_vocab: list[str]) -> dict:
    """
    Filter WLASL metadata to only include our target vocabulary.

    Returns: dict mapping gloss → list of video instances
    """
    target_set = set(v.lower() for v in target_vocab)
    filtered = {}

    for entry in metadata:
        gloss = entry.get("gloss", "").lower()
        if gloss in target_set:
            filtered[gloss] = entry.get("instances", [])

    found = set(filtered.keys())
    missing = target_set - found
    print(f"Found {len(found)}/{len(target_set)} target glosses in WLASL")
    if missing:
        print(f"Missing glosses (will need alternative data): {sorted(missing)}")

    return filtered


def extract_frames(video_path: str, output_dir: Path, max_frames: int = 5):
    """
    Extract evenly-spaced frames from a video file.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        max_frames: Maximum number of frames to extract per video
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0

    # Pick evenly-spaced frame indices
    if total_frames <= max_frames:
        indices = list(range(total_frames))
    else:
        indices = [int(i * total_frames / max_frames) for i in range(max_frames)]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_path = output_dir / f"frame_{idx:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved += 1

    cap.release()
    return saved


def build_label_map(glosses: list[str]) -> dict:
    """Create a gloss → integer label mapping."""
    return {gloss: idx for idx, gloss in enumerate(sorted(glosses))}


def main():
    cfg = Config()

    print("=" * 60)
    print("WLASL Dataset Pipeline")
    print("=" * 60)

    # Step 1: Download metadata
    metadata = download_metadata()

    # Step 2: Filter to target vocabulary
    target_data = filter_target_glosses(metadata, cfg.data.target_vocab)

    # Step 3: Create label map
    label_map = build_label_map(list(target_data.keys()))
    label_map_path = PROCESSED_DIR / "label_map.json"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved label map ({len(label_map)} classes) to {label_map_path}")

    # Step 4: Download videos and extract frames
    # NOTE: Many WLASL videos are sourced from YouTube and may be unavailable.
    # For the initial prototype, we recommend:
    #   - Using the WLASL subset that comes with pre-downloaded videos
    #   - Supplementing with ASL Citizen dataset
    #   - Recording custom samples for missing glosses
    print("\n[INFO] Video download step is not automated in this script.")
    print("Please download WLASL videos manually from the official repository:")
    print("  https://github.com/dxli94/WLASL")
    print("  Place videos in: data/raw/wlasl/videos/")
    print("\nThen run this script again to extract frames.")

    # Step 5: If videos exist, extract frames
    video_dir = RAW_DIR / "videos"
    if video_dir.exists():
        print(f"\nExtracting frames from {video_dir}...")
        for gloss, instances in tqdm(target_data.items()):
            for inst in instances:
                video_id = inst.get("video_id", "")
                video_path = video_dir / f"{video_id}.mp4"
                if video_path.exists():
                    output_dir = PROCESSED_DIR / "images" / "train" / gloss
                    extract_frames(video_path, output_dir)
    else:
        print(f"\n[SKIP] Video directory not found: {video_dir}")

    print("\nPipeline complete.")
    print(f"  Label map: {label_map_path}")
    print(f"  Processed images will be in: {PROCESSED_DIR}/images/")


if __name__ == "__main__":
    main()
