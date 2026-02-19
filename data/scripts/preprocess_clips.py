"""
Video clip preprocessing.

Takes raw source videos and produces normalised clips:
  - Trim to sign boundaries (if annotations exist).
  - Uniformly sample to a fixed number of frames.
  - Resize each frame to (FRAME_HEIGHT, FRAME_WIDTH).
  - Write the processed clip as a short .mp4 to the output directory.

Usage:
    python preprocess_clips.py          # processes all ingested CSVs
    python preprocess_clips.py --source asl_citizen   # one source only
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from pipeline_config import (
    PROCESSED_DIR, CLIPS_DIR,
    NUM_SAMPLE_FRAMES, FRAME_HEIGHT, FRAME_WIDTH,
    MIN_CLIP_FRAMES, MAX_CLIP_SECONDS, VIDEO_FPS, SOURCES,
)


def load_ingested_records(source: str) -> list[dict]:
    csv_path = PROCESSED_DIR / f"ingested_{source}.csv"
    if not csv_path.exists():
        print(f"[preprocess] Skipping {source} — no ingested CSV found.")
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def read_video_frames(path: str, start: int = 0, end: int = -1) -> list[np.ndarray]:
    """Read frames from a video file, optionally trimming by frame range."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS

    if end <= 0 or end > total:
        end = total
    if start < 0:
        start = 0

    duration = (end - start) / fps
    if duration > MAX_CLIP_SECONDS:
        cap.release()
        return []

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(end - start):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def read_video_frames_by_time(path: str, start_sec: float, end_sec: float) -> list[np.ndarray]:
    """Read frames between two timestamps (seconds)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) if end_sec > 0 else -1

    cap.release()
    return read_video_frames(path, start_frame, end_frame)


def uniform_sample(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    """Uniformly sample exactly n frames from a list."""
    total = len(frames)
    if total == 0:
        return []
    if total <= n:
        # Repeat the last frame to pad
        return frames + [frames[-1]] * (n - total)
    indices = np.linspace(0, total - 1, n, dtype=int)
    return [frames[i] for i in indices]


def resize_frames(frames: list[np.ndarray], h: int, w: int) -> list[np.ndarray]:
    return [cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in frames]


def write_clip(frames: list[np.ndarray], dest: Path, fps: int = VIDEO_FPS):
    """Write a list of frames as a short .mp4 clip."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dest), fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def process_record(record: dict) -> dict | None:
    """
    Process a single ingested record:
      1. Read frames (optionally trimming to annotations).
      2. Validate minimum length.
      3. Uniformly sample to NUM_SAMPLE_FRAMES.
      4. Resize.
      5. Write processed clip.

    Returns an updated record dict with processed path, or None on failure.
    """
    src = record["src_path"]
    source = record.get("source", "unknown")

    # Determine trim boundaries based on dataset format
    if source == "wlasl":
        start = int(record.get("frame_start", 0) or 0)
        end = int(record.get("frame_end", -1) or -1)
        frames = read_video_frames(src, start, end)
    elif source == "msasl":
        start_t = float(record.get("start_time", 0) or 0)
        end_t = float(record.get("end_time", -1) or -1)
        if end_t > 0:
            frames = read_video_frames_by_time(src, start_t, end_t)
        else:
            frames = read_video_frames(src)
    else:
        frames = read_video_frames(src)

    if len(frames) < MIN_CLIP_FRAMES:
        return None

    frames = uniform_sample(frames, NUM_SAMPLE_FRAMES)
    frames = resize_frames(frames, FRAME_HEIGHT, FRAME_WIDTH)

    gloss = record["gloss"]
    split = record.get("split", "train")
    clip_id = record["clip_id"]
    dest = CLIPS_DIR / split / gloss / f"{clip_id}.mp4"

    write_clip(frames, dest)

    return {
        "clip_id": clip_id,
        "gloss": gloss,
        "signer_id": record.get("signer_id", ""),
        "split": split,
        "source": source,
        "num_frames": NUM_SAMPLE_FRAMES,
        "height": FRAME_HEIGHT,
        "width": FRAME_WIDTH,
        "clip_path": str(dest),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SOURCES, default=None,
                        help="Process a single source. Default: all.")
    args = parser.parse_args()
    sources = [args.source] if args.source else SOURCES

    all_processed: list[dict] = []

    for source in sources:
        records = load_ingested_records(source)
        if not records:
            continue
        print(f"\n[preprocess] Processing {len(records)} clips from {source} …")

        ok, fail = 0, 0
        for r in records:
            result = process_record(r)
            if result:
                all_processed.append(result)
                ok += 1
            else:
                fail += 1

        print(f"[preprocess] {source}: {ok} processed, {fail} skipped.")

    # Write master processed CSV
    if all_processed:
        out = PROCESSED_DIR / "processed_clips.csv"
        fieldnames = list(all_processed[0].keys())
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_processed)
        print(f"\n[preprocess] Total: {len(all_processed)} clips → {out}")


if __name__ == "__main__":
    main()
