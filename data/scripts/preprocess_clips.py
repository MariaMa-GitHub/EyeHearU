"""
Video clip preprocessing.

Takes raw source videos and produces normalised clips:
  - Trim to sign boundaries (if annotations exist).
  - Uniformly sample to a fixed number of frames.
  - Resize each frame to (FRAME_HEIGHT, FRAME_WIDTH).
  - Write the processed clip as a short .mp4 to the output directory.

With PIPELINE_ENV=dev and --mvp: reads MVP CSV and videos from S3, uploads
processed clips and processed_clips.csv to s3://.../processed/mvp/.

Usage:
    python preprocess_clips.py          # processes all ingested CSVs (local)
    python preprocess_clips.py --source asl_citizen --mvp   # one source, local MVP dir
    PIPELINE_ENV=dev python preprocess_clips.py --source asl_citizen --mvp  # S3 in/out
"""

import argparse
import csv
import io
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from pipeline_config import (
    PROCESSED_DIR, CLIPS_DIR,
    NUM_SAMPLE_FRAMES, FRAME_HEIGHT, FRAME_WIDTH,
    MIN_CLIP_FRAMES, MAX_CLIP_SECONDS, VIDEO_FPS, SOURCES,
    get_processed_base, get_processed_prefix_s3,
    is_cloud, S3_BUCKET,
    get_s3_client, read_text_from_s3, write_text_to_s3,
)


def load_ingested_records(source: str, processed_dir: Path | None = None) -> list[dict]:
    base = processed_dir or PROCESSED_DIR
    csv_path = base / f"ingested_{source}.csv"
    if not csv_path.exists():
        print(f"[preprocess] Skipping {source} — no ingested CSV at {csv_path}.")
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_ingested_records_from_s3(source: str, mvp: bool = False) -> list[dict]:
    """Load ingested CSV from S3. src_path in each record is the S3 key (e.g. raw/asl_citizen/videos/x.mp4)."""
    prefix = get_processed_prefix_s3(mvp)
    key = f"{prefix}/ingested_{source}.csv"
    try:
        text = read_text_from_s3(key)
    except Exception as e:
        print(f"[preprocess] No ingested CSV in S3 at {key}: {e}")
        return []
    return list(csv.DictReader(io.StringIO(text)))


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
      1. Read frames from local path or S3 (download to temp if S3).
      2. Validate minimum length.
      3. Uniformly sample to NUM_SAMPLE_FRAMES.
      4. Resize.
      5. Write processed clip (local or upload to S3).

    Returns an updated record dict with processed path, or None on failure.
    """
    source = record.get("source", "unknown")
    src = record["src_path"]
    s3_video_key = record.get("_s3_video_key")
    clips_s3_prefix = record.get("_clips_s3_prefix")

    # Resolve source video path: local or download from S3 to temp
    if s3_video_key:
        s3 = get_s3_client()
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_video_key)
            body = obj["Body"]
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(body.read())
                src = tmp.name
        except Exception as e:
            print(f"  [skip] S3 download failed {s3_video_key}: {e}")
            return None
    else:
        src = record["src_path"]

    try:
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
    finally:
        if s3_video_key and src and os.path.exists(src):
            try:
                os.unlink(src)
            except OSError:
                pass

    if len(frames) < MIN_CLIP_FRAMES:
        return None

    frames = uniform_sample(frames, NUM_SAMPLE_FRAMES)
    frames = resize_frames(frames, FRAME_HEIGHT, FRAME_WIDTH)

    gloss = record["gloss"]
    split = record.get("split", "train")
    clip_id = record["clip_id"]

    if clips_s3_prefix:
        # Write clip to temp file, upload to S3, then delete temp
        s3_key = f"{clips_s3_prefix}/{split}/{gloss}/{clip_id}.mp4"
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            write_clip(frames, Path(tmp_path))
            s3 = get_s3_client()
            s3.upload_file(tmp_path, S3_BUCKET, s3_key)
            clip_path = f"s3://{S3_BUCKET}/{s3_key}"
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    else:
        clips_dir = record.get("_clips_dir") or CLIPS_DIR
        dest = clips_dir / split / gloss / f"{clip_id}.mp4"
        write_clip(frames, dest)
        clip_path = str(dest)

    return {
        "clip_id": clip_id,
        "gloss": gloss,
        "signer_id": record.get("signer_id", ""),
        "split": split,
        "source": source,
        "num_frames": NUM_SAMPLE_FRAMES,
        "height": FRAME_HEIGHT,
        "width": FRAME_WIDTH,
        "clip_path": clip_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SOURCES, default=None,
                        help="Process a single source. Default: all.")
    parser.add_argument("--mvp", action="store_true",
                        help="Read/write under processed/mvp/ (MVP-filtered run).")
    args = parser.parse_args()
    sources = [args.source] if args.source else SOURCES

    use_s3 = is_cloud() and args.mvp
    processed_dir = get_processed_base(args.mvp)
    clips_dir = processed_dir / "clips"
    clips_s3_prefix = f"{get_processed_prefix_s3(True)}/clips" if use_s3 else None

    if args.mvp:
        print(f"[preprocess] MVP mode: using {processed_dir}" + (" (S3)" if use_s3 else ""))
    if use_s3:
        print(f"[preprocess] Reading CSV/videos from S3, writing clips to s3://{S3_BUCKET}/{clips_s3_prefix}/")

    all_processed: list[dict] = []

    for source in sources:
        if use_s3:
            records = load_ingested_records_from_s3(source, mvp=True)
            # src_path in CSV is the S3 key (e.g. raw/asl_citizen/videos/x.mp4)
            records = [
                {**r, "_s3_video_key": r["src_path"], "_clips_s3_prefix": clips_s3_prefix}
                for r in records
            ]
        else:
            records = load_ingested_records(source, processed_dir)
            records = [{**r, "_clips_dir": clips_dir} for r in records]

        if not records:
            continue
        print(f"\n[preprocess] Processing {len(records)} clips from {source} ...")

        ok, fail = 0, 0
        for r in records:
            result = process_record(r)
            if result:
                all_processed.append(result)
                ok += 1
            else:
                fail += 1

        print(f"[preprocess] {source}: {ok} processed, {fail} skipped.")

    # Write master processed CSV (local or S3)
    if all_processed:
        fieldnames = [k for k in all_processed[0].keys() if not k.startswith("_")]
        if use_s3:
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_processed)
            csv_key = f"{get_processed_prefix_s3(True)}/processed_clips.csv"
            write_text_to_s3(buf.getvalue(), csv_key)
            print(f"\n[preprocess] Total: {len(all_processed)} clips -> s3://{S3_BUCKET}/{csv_key}")
        else:
            out = processed_dir / "processed_clips.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_processed)
            print(f"\n[preprocess] Total: {len(all_processed)} clips -> {out}")


if __name__ == "__main__":
    main()
