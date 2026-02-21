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
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from pipeline_config import (
    PROCESSED_DIR, CLIPS_DIR,
    NUM_SAMPLE_FRAMES, FRAME_HEIGHT, FRAME_WIDTH,
    MIN_CLIP_FRAMES, MAX_CLIP_SECONDS, VIDEO_FPS, SOURCES,
)


def _ffprobe_size_fps(path: str) -> tuple[int, int, float] | None:
    """Return (width, height, fps) or None if ffprobe fails."""
    path = str(Path(path).resolve())
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        parts = out.stdout.strip().split(",")
        if len(parts) < 3:
            return None
        w, h = int(parts[0]), int(parts[1])
        rate = parts[2].strip()
        if "/" in rate:
            a, b = rate.split("/", 1)
            fps = float(a) / float(b) if float(b) else 30.0
        else:
            fps = float(rate) if rate else 30.0
        return (w, h, fps)
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        return None


def _read_frames_ffmpeg(path: str, start_sec: float = 0, end_sec: float = -1) -> list[np.ndarray]:
    """Fallback: ffmpeg decodes to a temp MJPEG mp4, then OpenCV reads it. Returns list of BGR frames."""
    path = str(Path(path).resolve())
    if not Path(path).exists():
        return []
    info = _ffprobe_size_fps(path)
    if not info:
        return []
    w, h, fps = info
    if end_sec <= 0:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            end_sec = float(out.stdout.strip())
        else:
            end_sec = 60.0
    duration = max(0.01, end_sec - start_sec)
    if duration > MAX_CLIP_SECONDS:
        return []
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="preprocess_") as f:
            tmp_path = f.name
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-ss", str(start_sec), "-i", path,
                "-t", str(duration), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an", tmp_path,
            ],
            capture_output=True,
            timeout=60,
        )
        if r.returncode != 0 or not Path(tmp_path).exists():
            return []
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return []
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return []
    finally:
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass


def load_ingested_records(source: str) -> list[dict]:
    csv_path = PROCESSED_DIR / f"ingested_{source}.csv"
    if not csv_path.exists():
        print(f"[preprocess] Skipping {source} — no ingested CSV found.")
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def read_video_frames(path: str, start: int = 0, end: int = -1) -> list[np.ndarray]:
    """Read frames from a video file, optionally trimming by frame range. Falls back to ffmpeg if OpenCV fails."""
    path = str(Path(path).resolve())
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if cap.isOpened():
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
        if end <= 0 or end > total:
            end = total
        if start < 0:
            start = 0
        duration = (end - start) / fps
        if duration <= MAX_CLIP_SECONDS:
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(end - start):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            if frames:
                return frames
    if cap.isOpened():
        cap.release()

    info = _ffprobe_size_fps(path)
    if not info:
        return []
    _, _, fps = info
    start_sec = start / fps
    end_sec = (end / fps) if end > 0 else -1.0
    return _read_frames_ffmpeg(path, start_sec, end_sec)


def read_video_frames_by_time(path: str, start_sec: float, end_sec: float) -> list[np.ndarray]:
    """Read frames between two timestamps (seconds). Falls back to ffmpeg if OpenCV fails."""
    path = str(Path(path).resolve())
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps) if end_sec > 0 else -1
        cap.release()
        frames = read_video_frames(path, start_frame, end_frame)
        if frames:
            return frames
    return _read_frames_ffmpeg(path, start_sec, end_sec)


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


def write_clip(frames: list[np.ndarray], dest: Path, fps: int = VIDEO_FPS) -> bool:
    """Write a list of frames as a short .mp4 clip. Returns True if file exists and has size > 0."""
    dest = Path(dest).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dest), fourcc, fps, (w, h))
    if not writer.isOpened():
        return False
    for f in frames:
        writer.write(f)
    writer.release()
    return dest.exists() and dest.stat().st_size > 0


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

    # Determine trim boundaries: WLASL and MS-ASL both use start_time / end_time (seconds)
    if source in ("wlasl", "msasl"):
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

    if not write_clip(frames, dest):
        return None

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
