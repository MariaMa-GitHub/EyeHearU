"""
Preprocessing utilities for ASL datasets.

This script handles:
  - Resizing images to consistent dimensions
  - Hand region cropping using MediaPipe Hands
  - Frame quality validation (blur, black, resolution)
  - Deduplication via perceptual hashing
  - Train/val/test splitting (stratified by class)
  - Dataset statistics and validation
  - Data augmentation helpers for training-time use

AWS Equivalent:
  - AWS Batch / Lambda for image transformation at scale
  - S3 for storing processed images
  - Glue for data cataloging and quality checks
"""

import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("[WARNING] MediaPipe not installed. Hand cropping will be skipped.")
    print("  Install with: pip install mediapipe")


PROCESSED_DIR = Path(__file__).parent.parent / "processed"
IMAGE_SIZE = 224

# ─── Quality thresholds ──────────────────────────────────────────────────────
MIN_DIMENSION = 64              # Minimum width/height in pixels
MAX_BLACK_MEAN = 10             # Max mean pixel for "black frame" detection
MIN_LAPLACIAN_VAR = 10.0        # Min Laplacian variance for blur detection
PERCEPTUAL_HASH_SIZE = 8        # Size for average-hash deduplication


# ═══════════════════════════════════════════════════════════════════════════════
# HAND REGION DETECTION & CROPPING
# ═══════════════════════════════════════════════════════════════════════════════

def crop_hand_region(image: np.ndarray, padding: float = 0.25) -> np.ndarray | None:
    """
    Use MediaPipe Hands to detect and crop the hand region from an image.

    For ASL recognition, focusing on the hand region improves model accuracy
    by removing irrelevant background information.

    Args:
        image: BGR image (OpenCV format)
        padding: Fraction of extra space around the detected hand bounding box

    Returns:
        Cropped hand region (BGR), or None if no hand detected.
    """
    if not HAS_MEDIAPIPE:
        return None

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as hands:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        # Get bounding box around ALL detected hands (some signs use both)
        h, w, _ = image.shape
        all_x, all_y = [], []

        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                all_x.append(lm.x * w)
                all_y.append(lm.y * h)

        x_min, x_max = int(min(all_x)), int(max(all_x))
        y_min, y_max = int(min(all_y)), int(max(all_y))

        # Add padding
        pad_x = int((x_max - x_min) * padding)
        pad_y = int((y_max - y_min) * padding)
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)

        cropped = image[y_min:y_max, x_min:x_max]

        # Validate crop is not too small
        if cropped.shape[0] < 32 or cropped.shape[1] < 32:
            return None

        return cropped


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE QUALITY VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def is_black_frame(image: np.ndarray, threshold: float = MAX_BLACK_MEAN) -> bool:
    """Check if an image is a near-black frame (common at video start/end)."""
    return np.mean(image) < threshold


def is_blurry(image: np.ndarray, threshold: float = MIN_LAPLACIAN_VAR) -> bool:
    """
    Check if an image is too blurry using Laplacian variance.

    The Laplacian operator highlights rapid intensity changes (edges).
    Low variance = few edges = blurry image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


def is_too_small(image: np.ndarray, min_dim: int = MIN_DIMENSION) -> bool:
    """Check if image dimensions are below the minimum threshold."""
    h, w = image.shape[:2]
    return h < min_dim or w < min_dim


def compute_perceptual_hash(image: np.ndarray, hash_size: int = PERCEPTUAL_HASH_SIZE) -> str:
    """
    Compute a perceptual hash (average hash) for image deduplication.

    Two images with the same perceptual hash are visually very similar.
    """
    resized = cv2.resize(image, (hash_size, hash_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    bits = (gray > mean_val).flatten()
    hash_int = sum(bit << i for i, bit in enumerate(bits))
    return hex(hash_int)


def validate_image(image_path: Path) -> dict:
    """
    Run all quality checks on a single image.

    Returns a dict with validation results:
      {valid: bool, reason: str, metrics: {...}}
    """
    img = cv2.imread(str(image_path))

    if img is None:
        return {"valid": False, "reason": "cannot_load", "metrics": {}}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    metrics = {
        "width": w,
        "height": h,
        "mean_pixel": float(np.mean(img)),
        "laplacian_var": float(laplacian_var),
    }

    if is_too_small(img):
        return {"valid": False, "reason": "too_small", "metrics": metrics}

    if is_black_frame(img):
        return {"valid": False, "reason": "black_frame", "metrics": metrics}

    if is_blurry(img):
        return {"valid": False, "reason": "too_blurry", "metrics": metrics}

    return {"valid": True, "reason": "ok", "metrics": metrics}


def clean_directory(images_dir: Path, remove_invalid: bool = True) -> dict:
    """
    Validate and optionally remove invalid images from a directory tree.

    Returns statistics about the cleaning process.
    """
    stats = {
        "total": 0,
        "valid": 0,
        "removed": 0,
        "reasons": Counter(),
        "duplicates": 0,
    }

    seen_hashes = set()

    for class_dir in sorted(images_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for img_path in sorted(class_dir.glob("*.jpg")):
            stats["total"] += 1
            result = validate_image(img_path)

            if not result["valid"]:
                stats["reasons"][result["reason"]] += 1
                if remove_invalid:
                    img_path.unlink()
                    stats["removed"] += 1
                continue

            # Dedup check
            img = cv2.imread(str(img_path))
            if img is not None:
                phash = compute_perceptual_hash(img)
                if phash in seen_hashes:
                    stats["duplicates"] += 1
                    if remove_invalid:
                        img_path.unlink()
                        stats["removed"] += 1
                    continue
                seen_hashes.add(phash)

            stats["valid"] += 1

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

def resize_image(image: np.ndarray, target_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Resize an image to target_size x target_size.

    Uses INTER_AREA for downsampling (better quality than INTER_LINEAR).
    """
    return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)


def transform_image(
    image: np.ndarray,
    target_size: int = IMAGE_SIZE,
    crop_hands: bool = True,
) -> np.ndarray:
    """
    Apply the full transformation pipeline to a single image.

    Steps:
      1. Hand region cropping (if MediaPipe available and crop_hands=True)
      2. Resize to target dimensions
    """
    transformed = image

    # Optional hand cropping
    if crop_hands:
        cropped = crop_hand_region(image)
        if cropped is not None:
            transformed = cropped

    # Resize
    transformed = resize_image(transformed, target_size)

    return transformed


def transform_directory(
    images_dir: Path,
    target_size: int = IMAGE_SIZE,
    crop_hands: bool = True,
) -> dict:
    """
    Apply transformations to all images in a directory tree.

    Overwrites original files with transformed versions.
    """
    stats = {
        "total_processed": 0,
        "hand_detected": 0,
        "hand_not_detected": 0,
    }

    for class_dir in sorted(images_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for img_path in sorted(class_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            if crop_hands:
                cropped = crop_hand_region(img)
                if cropped is not None:
                    img = cropped
                    stats["hand_detected"] += 1
                else:
                    stats["hand_not_detected"] += 1

            resized = resize_image(img, target_size)
            cv2.imwrite(str(img_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            stats["total_processed"] += 1

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN / VAL / TEST SPLITTING
# ═══════════════════════════════════════════════════════════════════════════════

def split_dataset(
    images_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Split a flat directory of class folders into train/val/test.

    Expects:
        images_dir/
        ├── all/
        │   ├── hello/
        │   ├── goodbye/
        │   └── ...

    Creates:
        images_dir/train/hello/...
        images_dir/val/hello/...
        images_dir/test/hello/...
    """
    random.seed(seed)

    source_dir = images_dir / "all"
    if not source_dir.exists():
        # Fallback: try class dirs directly under images_dir
        source_dir = images_dir

    print(f"Splitting dataset from {source_dir}...")

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name in ("train", "val", "test", "all"):
            continue

        files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(files)

        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

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

        print(f"  {class_dir.name}: {n} images → train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dataset_stats(images_dir: Path) -> dict:
    """
    Compute and print comprehensive dataset statistics.

    Returns dict with per-split, per-class counts and overall summary.
    """
    stats = {"splits": {}, "total_images": 0}

    for split in ("train", "val", "test"):
        split_dir = images_dir / split
        if not split_dir.exists():
            continue
        class_counts = {}
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
                class_counts[class_dir.name] = count
                stats["total_images"] += count

        split_total = sum(class_counts.values())
        stats["splits"][split] = {
            "total": split_total,
            "num_classes": len(class_counts),
            "class_counts": class_counts,
        }
        print(f"\n[{split}] {split_total} images across {len(class_counts)} classes")

        # Show per-class breakdown
        if class_counts:
            for cls in sorted(class_counts.keys()):
                print(f"  {cls:20s}: {class_counts[cls]:5d}")

    print(f"\nTotal images: {stats['total_images']}")

    # Class balance analysis
    if "train" in stats["splits"]:
        counts = list(stats["splits"]["train"]["class_counts"].values())
        if counts:
            print(f"\nClass balance (train split):")
            print(f"  Min samples:  {min(counts)}")
            print(f"  Max samples:  {max(counts)}")
            print(f"  Mean samples: {np.mean(counts):.1f}")
            print(f"  Std samples:  {np.std(counts):.1f}")

    return stats


def save_dataset_stats(images_dir: Path, output_path: Path | None = None) -> dict:
    """Compute stats and save to JSON."""
    stats = compute_dataset_stats(images_dir)

    if output_path is None:
        output_path = images_dir.parent / "dataset_stats.json"

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {output_path}")

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    images_dir = PROCESSED_DIR / "images"

    if not images_dir.exists():
        print(f"No processed images found at {images_dir}")
        print("Run download_asl_citizen.py or download_wlasl.py first.")
        exit(1)

    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Clean
    all_dir = images_dir / "all"
    if all_dir.exists():
        print("\n[Step 1] Cleaning images...")
        clean_stats = clean_directory(all_dir, remove_invalid=True)
        print(f"  Checked: {clean_stats['total']}")
        print(f"  Valid:   {clean_stats['valid']}")
        print(f"  Removed: {clean_stats['removed']}")
        print(f"  Reasons: {dict(clean_stats['reasons'])}")
        print(f"  Dupes:   {clean_stats['duplicates']}")

        # Step 2: Transform
        print("\n[Step 2] Transforming images (resize + optional hand crop)...")
        transform_stats = transform_directory(all_dir, crop_hands=HAS_MEDIAPIPE)
        print(f"  Processed: {transform_stats['total_processed']}")
        if HAS_MEDIAPIPE:
            print(f"  Hand detected:     {transform_stats['hand_detected']}")
            print(f"  Hand not detected: {transform_stats['hand_not_detected']}")

        # Step 3: Split
        print("\n[Step 3] Splitting into train/val/test...")
        split_dataset(images_dir)

    # Step 4: Stats
    print("\n[Step 4] Computing dataset statistics...")
    save_dataset_stats(images_dir)

    print("\nPreprocessing complete.")
