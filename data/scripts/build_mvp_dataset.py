"""
Build MVP vocabulary dataset — filtered subset of processed clips.

Reads ingested_msasl.csv, filters to MVP glosses only, preprocesses to data/processed/mvp/

Usage:
    python build_mvp_dataset.py
"""

import csv
from pathlib import Path

from pipeline_config import (
    PROCESSED_DIR, MVP_DIR, MVP_CLIPS_DIR,
    NUM_SAMPLE_FRAMES, FRAME_HEIGHT, FRAME_WIDTH,
    MIN_CLIP_FRAMES, MAX_CLIP_SECONDS, VIDEO_FPS,
)

# Same aliases as download_msasl (MS-ASL synonym mapping)
GLOSS_ALIASES = {
    "thank-you": "thanks",
    "goodbye": "bye",
    "toilet": "bathroom",
    "good-morning": "good morning",
    "good-afternoon": "good afternoon",
    "good-night": "night",
    "nice-to-meet-you": "nice to meet you",
    "food": "eat",
    "pain": "hurt",
}

MVP_GLOSSES_FILE = Path(__file__).parent / "mvp_glosses.txt"


def load_mvp_glosses() -> set:
    allowed = set()
    if not MVP_GLOSSES_FILE.exists():
        return allowed
    for line in MVP_GLOSSES_FILE.read_text().splitlines():
        g = line.strip().lower()
        if not g or g.startswith("#"):
            continue
        key = g.replace(" ", "-")
        if key in GLOSS_ALIASES:
            allowed.add(GLOSS_ALIASES[key])
        allowed.add(g.replace("-", " "))
    return allowed


def process_record(record: dict) -> dict | None:
    """Process one record, writing to MVP_CLIPS_DIR. Reuses preprocess_clips logic."""
    import cv2
    import numpy as np

    from preprocess_clips import (
        read_video_frames,
        read_video_frames_by_time,
        uniform_sample,
        resize_frames,
        write_clip,
    )

    src = record["src_path"]
    if not Path(src).exists():
        return None

    start_t = float(record.get("start_time", 0) or 0)
    end_t = float(record.get("end_time", -1) or -1)
    if end_t > 0:
        frames = read_video_frames_by_time(src, start_t, end_t)
    else:
        frames = read_video_frames(src)

    if len(frames) < MIN_CLIP_FRAMES:
        return None

    frames = uniform_sample(frames, NUM_SAMPLE_FRAMES)
    frames = resize_frames(frames, FRAME_HEIGHT, FRAME_WIDTH)

    gloss = record["gloss"]
    split = record.get("split", "train")
    clip_id = record["clip_id"]
    dest = MVP_CLIPS_DIR / split / gloss / f"{clip_id}.mp4"

    write_clip(frames, dest)

    return {
        "clip_id": clip_id,
        "gloss": gloss,
        "signer_id": record.get("signer_id", ""),
        "split": split,
        "source": "msasl",
        "num_frames": NUM_SAMPLE_FRAMES,
        "height": FRAME_HEIGHT,
        "width": FRAME_WIDTH,
        "clip_path": str(dest),
    }


def main():
    print("=" * 60)
    print("MVP Dataset Build")
    print("=" * 60)

    allowed = load_mvp_glosses()
    print(f"  MVP glosses: {len(allowed)}")

    ingest_path = PROCESSED_DIR / "ingested_msasl.csv"
    if not ingest_path.exists():
        print(f"[mvp] No {ingest_path}. Run ingest_msasl.py first.")
        return

    with open(ingest_path, newline="") as f:
        records = list(csv.DictReader(f))

    mvp_records = [r for r in records if (r.get("gloss") or "").strip().lower() in allowed]
    print(f"  Filtered: {len(mvp_records)}/{len(records)} records")

    if not mvp_records:
        print("[mvp] No MVP records. Download with --glosses first.")
        return

    MVP_DIR.mkdir(parents=True, exist_ok=True)
    MVP_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # Write filtered ingest for MVP
    mvp_ingest = MVP_DIR / "ingested_msalmvp.csv"
    with open(mvp_ingest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mvp_records[0].keys()))
        writer.writeheader()
        writer.writerows(mvp_records)
    print(f"  Wrote {mvp_ingest}")

    # Preprocess
    all_processed = []
    ok, fail = 0, 0
    for r in mvp_records:
        result = process_record(r)
        if result:
            all_processed.append(result)
            ok += 1
        else:
            fail += 1

    if all_processed:
        out = MVP_DIR / "processed_clips.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_processed[0].keys()))
            writer.writeheader()
            writer.writerows(all_processed)
        print(f"\n[mvp] {ok} clips processed, {fail} skipped")
        print(f"  Output: {MVP_CLIPS_DIR}")
        print(f"  CSV: {out}")


if __name__ == "__main__":
    main()
