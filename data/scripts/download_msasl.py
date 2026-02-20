"""
Download MS-ASL videos from YouTube using URLs in the metadata JSON files.

MS-ASL metadata only contains JSON; videos must be downloaded from YouTube.
Place metadata in: data/raw/msasl/MS-ASL/
Output: data/raw/msasl/videos/{video_id}.mp4

Usage:
    python download_msasl.py                    # MVP glosses, train+val+test (default)
    python download_msasl.py --all             # Download all glosses (full dataset)
    python download_msasl.py --max-videos 200   # Limit for testing
    python download_msasl.py --glosses "hello,hi"  # Custom glosses

Requires: pip install yt-dlp
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from pipeline_config import MSASL_RAW

# MS-ASL metadata can be in MS-ASL/ subfolder (official download) or at root
MSASL_META = MSASL_RAW / "MS-ASL"
if not MSASL_META.exists():
    MSASL_META = MSASL_RAW

MVP_GLOSSES_FILE = Path(__file__).parent / "mvp_glosses.txt"


def _load_mvp_glosses() -> str:
    """Load MVP glosses from mvp_glosses.txt, return comma-separated for --glosses."""
    if not MVP_GLOSSES_FILE.exists():
        return ""
    lines = [
        line.strip() for line in MVP_GLOSSES_FILE.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return ",".join(lines)


def _video_id_from_url(url: str) -> str | None:
    """Extract YouTube video ID from URL."""
    if not url:
        return None
    # Handle youtube.com/watch?v=ID and youtu.be/ID
    m = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return m.group(1) if m else None


def load_metadata() -> tuple[list, list[dict], list[dict], list[dict]]:
    """Load classes and split JSONs from MS-ASL folder."""
    classes_path = MSASL_META / "MSASL_classes.json"
    train_path = MSASL_META / "MSASL_train.json"
    val_path = MSASL_META / "MSASL_val.json"
    test_path = MSASL_META / "MSASL_test.json"

    for p in [classes_path, train_path, val_path, test_path]:
        if not p.exists():
            sys.exit(f"[download_msasl] Missing {p}. Place MS-ASL JSON files in {MSASL_META}.")

    with open(classes_path) as f:
        classes = json.load(f)
    with open(train_path) as f:
        train = json.load(f)
    with open(val_path) as f:
        val = json.load(f)
    with open(test_path) as f:
        test = json.load(f)
    return classes, train, val, test


# Map user gloss variants to MS-ASL class names (synonyms / aliases)
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


def _build_allowed_glosses(gloss_arg: str) -> set:
    """Build set of allowed gloss strings (MS-ASL format) from user input."""
    allowed = set()
    for raw in gloss_arg.replace(",", "\n").replace(" ", "\n").split():
        g = raw.strip().lower()
        if not g:
            continue
        # Add alias if exists (thank-you->thanks, goodbye->bye, toilet->bathroom)
        key = g.replace(" ", "-")
        if key in GLOSS_ALIASES:
            allowed.add(GLOSS_ALIASES[key])
        allowed.add(g.replace("-", " "))
    return allowed


def collect_videos_to_download(
    train: list, val: list, test: list,
    subset: int | None = None,
    max_videos: int | None = None,
    glosses: str | None = None,
    classes: list | None = None,
    splits: list[str] | None = None,
) -> list[tuple[str, str]]:
    """
    Return [(video_id, url), ...] for videos to download.
    subset: if set, only include entries with label < subset (e.g. 100 for MS-ASL100)
    max_videos: cap total number of videos to download
    glosses: if set, only include entries whose text matches (faster, fewer videos)
    """
    seen = set()
    out = []
    allowed = _build_allowed_glosses(glosses) if glosses else None

    def matches(e: dict) -> bool:
        if allowed is None:
            return True
        t = (e.get("text") or "").strip().lower()
        if t in allowed:
            return True
        # Check label -> class name
        if classes and len(classes) > 0:
            lbl = e.get("label", -1)
            if 0 <= lbl < len(classes):
                c = (classes[lbl] or "").strip().lower()
                if c in allowed:
                    return True
        return False

    def add(entries: list, max_label: int | None = None):
        for e in entries:
            if max_label is not None and e.get("label", 0) >= max_label:
                continue
            if not matches(e):
                continue
            vid = _video_id_from_url(e.get("url", ""))
            if vid and vid not in seen:
                seen.add(vid)
                url = e.get("url", "")
                if not url.startswith("http"):
                    url = "https://" + url
                out.append((vid, url))

    # Val and test first so we get eval splits (train/val/test have disjoint videos)
    want = splits or ["val", "test", "train"]
    for s in want:
        if s == "val":
            add(val, subset)
        elif s == "test":
            add(test, subset)
        elif s == "train":
            add(train, subset)

    if max_videos is not None:
        return out[:max_videos]
    return out


def download_video(video_id: str, url: str, out_dir: Path) -> bool:
    """Download a single video with yt-dlp. Returns True if successful."""
    out_path = out_dir / f"{video_id}.mp4"
    if out_path.exists():
        return True

    cmd = [
        "yt-dlp",
        "--no-warnings",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(out_path),
        "--no-overwrites",
        url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return out_path.exists()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def main():
    parser = argparse.ArgumentParser(description="Download MS-ASL videos from YouTube")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Max number of unique videos to download (for testing)")
    parser.add_argument("--subset", type=int, default=None,
                        help="MS-ASL subset: only labels < N (e.g. 100 for MS-ASL100)")
    parser.add_argument("--glosses", type=str, default=None,
                        help="Comma-separated glosses to download only (e.g. hello,hi,thanks,water)")
    parser.add_argument("--split", type=str, default=None,
                        help="Only download from split(s): train, val, test (comma-separated)")
    parser.add_argument("--all", action="store_true",
                        help="Download full dataset (default: MVP only)")
    args = parser.parse_args()

    glosses = args.glosses
    if not args.all and not glosses:
        glosses = _load_mvp_glosses()
        if not glosses:
            sys.exit("[download_msasl] mvp_glosses.txt not found. Use --glosses or --all.")
    if args.all:
        glosses = None
    splits = [x.strip() for x in args.split.split(",")] if args.split else ["val", "test", "train"]

    print("=" * 60)
    print("MS-ASL Video Download")
    print("=" * 60)

    classes, train, val, test = load_metadata()
    print(f"  Metadata: {len(classes)} classes, {len(train)} train, {len(val)} val, {len(test)} test")
    if glosses:
        n = len(_build_allowed_glosses(glosses))
        print(f"  Filter: {n} glosses (MVP)")
    else:
        print("  Filter: all glosses")
    print("  Splits: train, val, test")
    to_download = collect_videos_to_download(
        train, val, test,
        subset=args.subset,
        max_videos=args.max_videos,
        glosses=glosses,
        classes=classes,
        splits=splits,
    )
    print(f"  Unique videos to fetch: {len(to_download)}")

    out_dir = MSASL_RAW / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for i, (vid, url) in enumerate(to_download):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(to_download)} (ok={ok}, fail={fail})")
        if download_video(vid, url, out_dir):
            ok += 1
        else:
            fail += 1

    print(f"\n  Done: {ok} downloaded, {fail} failed")
    print(f"  Videos saved to: {out_dir}")


if __name__ == "__main__":
    main()
