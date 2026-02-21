"""
Build MVP subset from WLASL only — writes to data/processed/mvp/clips/.

Does not modify pipeline_config, build_mvp_dataset.py, or mvp_glosses.txt.
Reads ingested_wlasl.csv, filters to MVP glosses (case-insensitive; Hello = hello),
and writes clips to data/processed/mvp/clips/ so layout matches server.

Usage:
    cd data/scripts
    python build_mvp_from_wlasl.py

Requires: run ingest_wlasl.py first so data/processed/ingested_wlasl.csv exists.
"""

import csv
from pathlib import Path

# Paths: do not touch pipeline_config
_DATA_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = _DATA_ROOT / "processed"
MVP_DIR = PROCESSED_DIR / "mvp"
MVP_CLIPS_DIR = MVP_DIR / "clips"

MVP_GLOSSES_FILE = Path(__file__).parent / "mvp_glosses_wlasl.txt"

# From pipeline_config (duplicated here to avoid importing)
NUM_SAMPLE_FRAMES = 16
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
MIN_CLIP_FRAMES = 8
VIDEO_FPS = 30


def normalize_gloss(gloss: str) -> str:
    """Canonical form: lower, strip, spaces and underscores → hyphen. Hello/good morning/thank_you → same as list."""
    s = (gloss or "").strip().lower()
    return s.replace("_", "-").replace(" ", "-")


def load_mvp_glosses() -> set[str]:
    """Load exactly the MVP list; same canonical form so we allow no extra words."""
    allowed = set()
    if not MVP_GLOSSES_FILE.exists():
        return allowed
    for line in MVP_GLOSSES_FILE.read_text().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        # same canonical form as normalize_gloss so matching is strict
        canonical = raw.lower().replace("_", "-").replace(" ", "-")
        allowed.add(canonical)
    return allowed


def process_record(record: dict, gloss_canonical: str) -> dict | None:
    from preprocess_clips import (
        read_video_frames,
        read_video_frames_by_time,
        uniform_sample,
        resize_frames,
        write_clip,
    )

    src = record.get("src_path", "")
    if not src or not Path(src).exists():
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

    split = (record.get("split") or "train").strip().lower()
    clip_id = record.get("clip_id", "")
    dest = MVP_CLIPS_DIR / split / gloss_canonical / f"{clip_id}.mp4"

    write_clip(frames, dest)

    return {
        "clip_id": clip_id,
        "gloss": gloss_canonical,
        "signer_id": record.get("signer_id", ""),
        "split": split,
        "source": "wlasl",
        "num_frames": NUM_SAMPLE_FRAMES,
        "height": FRAME_HEIGHT,
        "width": FRAME_WIDTH,
        "clip_path": str(dest),
    }


def main():
    print("=" * 60)
    print("MVP from WLASL → data/processed/mvp/clips/")
    print("=" * 60)

    mvp_glosses = load_mvp_glosses()
    print(f"  MVP glosses: {len(mvp_glosses)} (from {MVP_GLOSSES_FILE.name})")

    ingest_path = PROCESSED_DIR / "ingested_wlasl.csv"
    if not ingest_path.exists():
        print(f"[mvp] No {ingest_path}. Run ingest_wlasl.py first.")
        return

    with open(ingest_path, newline="") as f:
        records = list(csv.DictReader(f))

    mvp_records = []
    for r in records:
        g = normalize_gloss(r.get("gloss", ""))
        if g in mvp_glosses:
            mvp_records.append(r)

    print(f"  Ingested WLASL: {len(records)} → MVP filter: {len(mvp_records)}")

    if not mvp_records:
        print("[mvp] No records match MVP vocabulary.")
        return

    MVP_DIR.mkdir(parents=True, exist_ok=True)
    MVP_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    all_processed = []
    ok, fail = 0, 0
    for r in mvp_records:
        gloss_canonical = normalize_gloss(r.get("gloss", ""))
        result = process_record(r, gloss_canonical)
        if result:
            all_processed.append(result)
            ok += 1
        else:
            fail += 1

    if all_processed:
        out_csv = MVP_DIR / "processed_clips.csv"
        fieldnames = list(all_processed[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_processed)
        print(f"\n[mvp] {ok} clips processed, {fail} skipped")
        print(f"  Clips: {MVP_CLIPS_DIR}")
        print(f"  CSV:  {out_csv}")
    else:
        print("\n[mvp] No clips written. Check that src_path videos exist.")


if __name__ == "__main__":
    main()
