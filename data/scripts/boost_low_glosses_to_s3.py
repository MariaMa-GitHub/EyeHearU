"""
Boost low-count MVP glosses in S3 from both MS-ASL and ASL Citizen.

What it does:
1) Reads current processed metadata from S3:
   - processed/mvp/metadata/processed_clips.csv
2) Finds low-count MS-ASL glosses (count < --low-threshold).
3) Pulls extra candidates from:
   - processed/mvp/metadata/ingested_msalmvp.csv
   - processed/mvp/metadata/ingested_asl_citizen.csv
4) Reprocesses clips to fixed-size MVP format and uploads to:
   - processed/mvp/clips/{split}/{gloss}/{clip_id}.mp4
5) Merges new rows back into:
   - processed/mvp/metadata/processed_clips.csv
   - processed/mvp/processed_clips.csv

Usage:
  AWS_PROFILE=public AWS_REGION=ca-central-1 python data/scripts/boost_low_glosses_to_s3.py \
    --bucket eye-hear-u-public-data-ca1
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import boto3

from preprocess_clips import (
    read_video_frames,
    read_video_frames_by_time,
    resize_frames,
    uniform_sample,
    write_clip,
)
from pipeline_config import FRAME_HEIGHT, FRAME_WIDTH, MIN_CLIP_FRAMES, NUM_SAMPLE_FRAMES, VIDEO_FPS


def _load_csv_from_s3(s3, bucket: str, key: str) -> list[dict]:
    text = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _download_msasl_video(video_id: str, out_path: Path) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "--no-warnings",
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(out_path),
        "--no-overwrites",
        url,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    except Exception:
        return False
    return out_path.exists() and out_path.stat().st_size > 0


def _process_frames(source: str, row: dict, local_src: Path):
    if source == "msasl":
        start_t = float(row.get("start_time") or 0)
        end_t = float(row.get("end_time") or -1)
        # Prefer timestamp boundaries, but fall back to full video if the
        # clipped segment is too short/corrupt so we can salvage more samples.
        frames = read_video_frames_by_time(str(local_src), start_t, end_t) if end_t > 0 else read_video_frames(str(local_src))
        if len(frames) < MIN_CLIP_FRAMES:
            frames = read_video_frames(str(local_src))
    else:
        frame_start = row.get("frame_start")
        frame_end = row.get("frame_end")
        if frame_start and frame_end:
            frames = read_video_frames(str(local_src), int(float(frame_start)), int(float(frame_end)))
        else:
            frames = read_video_frames(str(local_src))
    if len(frames) < MIN_CLIP_FRAMES:
        return None
    frames = uniform_sample(frames, NUM_SAMPLE_FRAMES)
    frames = resize_frames(frames, FRAME_HEIGHT, FRAME_WIDTH)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Boost low-count glosses from MS-ASL + ASL Citizen into S3 MVP clips.")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="ca-central-1")
    parser.add_argument("--low-threshold", type=int, default=10)
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    bucket = args.bucket

    proc_key = "processed/mvp/metadata/processed_clips.csv"
    ms_ing_key = "processed/mvp/metadata/ingested_msalmvp.csv"
    asl_ing_key = "processed/mvp/metadata/ingested_asl_citizen.csv"

    processed = _load_csv_from_s3(s3, bucket, proc_key)
    ms_ingested = _load_csv_from_s3(s3, bucket, ms_ing_key)
    asl_ingested = _load_csv_from_s3(s3, bucket, asl_ing_key)

    ms_rows = [r for r in processed if (r.get("source") or "").strip().lower() == "msasl"]
    ms_counts = Counter((r.get("gloss") or "").strip().lower() for r in ms_rows)
    low_glosses = [g for g, c in sorted(ms_counts.items(), key=lambda kv: (kv[1], kv[0])) if c < args.low_threshold]
    print(f"[boost] low glosses (<{args.low_threshold}): {low_glosses}")

    existing_paths = {(r.get("clip_path") or "").strip() for r in processed}
    existing_ids = {Path((r.get("clip_path") or "").strip()).stem for r in processed}

    ms_candidates = []
    for r in ms_ingested:
        gloss = (r.get("gloss") or "").strip().lower()
        clip_id = (r.get("clip_id") or "").strip()
        if gloss in low_glosses and clip_id and clip_id not in existing_ids:
            ms_candidates.append(r)

    asl_candidates = []
    for r in asl_ingested:
        gloss = (r.get("gloss") or "").strip().lower()
        clip_id = (r.get("clip_id") or "").strip()
        split = (r.get("split") or "train").strip().lower()
        s3_uri = f"s3://{bucket}/processed/mvp/clips/{split}/{gloss}/{clip_id}.mp4"
        if gloss in low_glosses and clip_id and s3_uri not in existing_paths:
            asl_candidates.append(r)

    print(f"[boost] candidates msasl={len(ms_candidates)} asl_citizen={len(asl_candidates)}")

    workdir = Path("/tmp/msasl_asl_boost")
    ms_video_dir = workdir / "ms_videos"
    out_dir = workdir / "clips"
    ms_video_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure MS-ASL source videos locally
    ms_video_ids = sorted({Path((r.get("src_path") or "").strip()).stem for r in ms_candidates if (r.get("src_path") or "").strip()})
    for i, vid in enumerate(ms_video_ids, 1):
        ok = _download_msasl_video(vid, ms_video_dir / f"{vid}.mp4")
        if i % 20 == 0:
            print(f"[boost] ms download {i}/{len(ms_video_ids)} ok={ok}")

    new_rows: list[dict] = []
    status = Counter()

    def save_row(source: str, row: dict, local_src: Path):
        frames = _process_frames(source, row, local_src)
        if frames is None:
            return "too_short"

        gloss = (row.get("gloss") or "").strip().lower()
        split = (row.get("split") or "train").strip().lower()
        clip_id = (row.get("clip_id") or "").strip()
        local_out = out_dir / split / gloss / f"{clip_id}.mp4"
        local_out.parent.mkdir(parents=True, exist_ok=True)
        write_clip(frames, local_out, fps=VIDEO_FPS)

        key = f"processed/mvp/clips/{split}/{gloss}/{clip_id}.mp4"
        clip_uri = f"s3://{bucket}/{key}"
        if clip_uri not in existing_paths:
            try:
                s3.head_object(Bucket=bucket, Key=key)
            except Exception:
                s3.upload_file(str(local_out), bucket, key)
            existing_paths.add(clip_uri)

        new_rows.append(
            {
                "clip_id": clip_id,
                "gloss": gloss,
                "signer_id": (row.get("signer_id") or "").strip(),
                "split": split,
                "source": source,
                "num_frames": str(NUM_SAMPLE_FRAMES),
                "height": str(FRAME_HEIGHT),
                "width": str(FRAME_WIDTH),
                "clip_path": clip_uri,
            }
        )
        return "ok"

    for r in ms_candidates:
        vid = Path((r.get("src_path") or "").strip()).stem
        src = ms_video_dir / f"{vid}.mp4"
        if not (src.exists() and src.stat().st_size > 0):
            status["missing_ms_video"] += 1
            continue
        status[save_row("msasl", r, src)] += 1

    for r in asl_candidates:
        src_key = (r.get("src_path") or "").strip()
        if not src_key:
            status["missing_asl_key"] += 1
            continue
        try:
            obj = s3.get_object(Bucket=bucket, Key=src_key)
        except Exception:
            status["missing_asl_object"] += 1
            continue
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tf.write(obj["Body"].read())
            tmp_path = Path(tf.name)
        try:
            status[save_row("asl_citizen", r, tmp_path)] += 1
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass

    merged = processed[:]
    seen_paths = {(r.get("clip_path") or "").strip() for r in merged}
    for r in new_rows:
        if r["clip_path"] not in seen_paths:
            merged.append(r)
            seen_paths.add(r["clip_path"])

    fields = ["clip_id", "gloss", "signer_id", "split", "source", "num_frames", "height", "width", "clip_path"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for r in merged:
        writer.writerow({k: r.get(k, "") for k in fields})
    body = buf.getvalue().encode("utf-8")

    s3.put_object(Bucket=bucket, Key="processed/mvp/metadata/processed_clips.csv", Body=body, ContentType="text/csv")
    s3.put_object(Bucket=bucket, Key="processed/mvp/processed_clips.csv", Body=body, ContentType="text/csv")

    print(f"[boost] status={dict(status)}")
    print(f"[boost] new_rows={len(new_rows)} merged_rows={len(merged)}")
    print(f"[boost] updated s3://{bucket}/processed/mvp/metadata/processed_clips.csv")


if __name__ == "__main__":
    main()
