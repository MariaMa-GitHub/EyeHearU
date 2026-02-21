"""
Download WLASL videos from URLs in WLASL_v0.3.json.

Saves files as data/raw/wlasl/videos/{video_id}.mp4 so that main branch's
ingest_wlasl.py can find them (it only keeps records where the .mp4 exists).

- YouTube / youtu.be URLs: downloaded with yt-dlp (saved as .mp4).
- Direct .mp4 URLs: downloaded with requests.
- Other URLs (e.g. aslpro .swf): downloaded but saved with original extension;
  ingest_wlasl only looks for .mp4, so those will be skipped unless you convert.

By default only downloads videos for the MVP vocabulary (greetings, basic needs,
restaurant, medical, A–Z, numbers 1–10). Already-downloaded .mp4 files are
skipped. To download all WLASL glosses, use --all-glosses.

Usage:
  cd data/scripts
  pip install yt-dlp requests tqdm   # if not already
  python download_wlasl_videos.py

Options:
  --all-glosses Download all glosses (full WLASL; slow). Default: MVP only.
  --max N       Only download at most N videos (for testing).
  --skip-yt     Skip YouTube videos (only download direct URLs).

Where to run:
  Same script works locally or on a server. Wherever you run the pipeline
  (ingest_wlasl, preprocess, etc.), that machine must have the videos first—
  so you always need to run this download step on that machine (or copy the
  downloaded videos folder there). ~21K videos can be tens of GB; use a server
  if your connection or disk is limited.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

# Paths: same layout as pipeline
RAW_DIR = Path(__file__).resolve().parent.parent
WLASL_RAW = RAW_DIR / "raw" / "wlasl"
VIDEOS_DIR = WLASL_RAW / "videos"
META_PATH = WLASL_RAW / "WLASL_v0.3.json"

# MVP vocabulary: only download videos for these glosses (faster, smaller).
# Normalized to lowercase; WLASL JSON may use hyphens (e.g. thank-you).
MVP_GLOSSES = {
    "hello", "hi", "goodbye", "bye", "thanks", "thank-you", "thank_you",
    "please", "yes", "no", "nice-to-meet-you", "nice_to_meet_you",
    "good-morning", "good_morning", "good-afternoon", "good_afternoon",
    "good-night", "good_night",
    "water", "food", "eat", "drink", "help", "bathroom", "toilet",
    "rest", "sleep", "need", "want",
    "menu", "order", "bill", "check", "waiter", "table", "cup", "plate",
    "fork", "spoon", "knife", "coffee", "tea", "milk", "bread",
    "doctor", "nurse", "hospital", "medicine", "sick", "hurt", "pain", "emergency",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
}


def load_tasks(
    max_videos: int | None,
    skip_youtube: bool,
    all_glosses: bool = False,
) -> list[tuple[str, str, str]]:
    """Load (video_id, url, gloss) from JSON. Only MVP glosses unless all_glosses; skip existing .mp4."""
    if not META_PATH.exists():
        print(f"Error: {META_PATH} not found. Run from repo root or ensure WLASL_v0.3.json exists.")
        sys.exit(1)
    with open(META_PATH) as f:
        data = json.load(f)

    tasks = []
    for entry in data:
        gloss = (entry.get("gloss") or "").strip().lower()
        if not all_glosses and gloss not in MVP_GLOSSES:
            continue
        for inst in entry.get("instances", []):
            video_id = inst.get("video_id", "")
            url = (inst.get("url") or "").strip()
            if not video_id or not url:
                continue
            out_mp4 = VIDEOS_DIR / f"{video_id}.mp4"
            if out_mp4.exists():
                continue
            if skip_youtube and ("youtube.com" in url or "youtu.be" in url):
                continue
            tasks.append((video_id, url, gloss))
            if max_videos and len(tasks) >= max_videos:
                return tasks
    return tasks


def is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def download_direct(video_id: str, url: str, session) -> bool:
    """Download non-YouTube URL to VIDEOS_DIR / video_id.mp4 (or other ext)."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        r = session.get(url, timeout=60, stream=True)
        r.raise_for_status()
        # Use .mp4 if URL looks like mp4, else keep extension from Content-Type or URL
        path = VIDEOS_DIR / f"{video_id}.mp4"
        if "mp4" in (r.headers.get("Content-Type") or "").lower() or url.rstrip("/").lower().endswith(".mp4"):
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        else:
            ext = ".mp4"
            for e in (".mp4", ".webm", ".mkv", ".swf"):
                if e in url.lower():
                    ext = e
                    break
            path = VIDEOS_DIR / f"{video_id}{ext}"
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            if ext != ".mp4":
                print(f"  [skip .mp4] {video_id} saved as {ext} (ingest expects .mp4)")
        return True
    except Exception as e:
        print(f"  [fail] {video_id}: {e}")
        return False


def download_youtube_ytdlp(video_id: str, url: str) -> bool:
    """Download YouTube URL with yt-dlp to video_id.mp4."""
    import subprocess
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    out_tpl = str(VIDEOS_DIR / f"{video_id}.%(ext)s")
    try:
        r = subprocess.run(
            [
                "yt-dlp",
                "--no-warnings",
                "-q",
                "-f", "best[ext=mp4]/best",
                "-o", out_tpl,
                "--no-check-certificate",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if r.returncode != 0:
            print(f"  [fail] {video_id}: yt-dlp {r.stderr[:200] if r.stderr else r.stdout}")
            return False
        # yt-dlp may write video_id.extension; rename to .mp4 if needed
        for p in VIDEOS_DIR.glob(f"{video_id}.*"):
            if p.suffix.lower() != ".mp4":
                mp4_path = p.with_suffix(".mp4")
                if not mp4_path.exists():
                    p.rename(mp4_path)
                else:
                    p.unlink()
        return True
    except FileNotFoundError:
        print("  yt-dlp not found. Install with: pip install yt-dlp")
        return False
    except Exception as e:
        print(f"  [fail] {video_id}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Download WLASL videos to data/raw/wlasl/videos/")
    ap.add_argument("--all-glosses", action="store_true", help="Download all glosses (default: MVP only)")
    ap.add_argument("--max", type=int, default=None, help="Max number of videos to download (default: all)")
    ap.add_argument("--skip-yt", action="store_true", help="Skip YouTube videos")
    ap.add_argument("--workers", type=int, default=1, help="Not used for now (sequential only)")
    args = ap.parse_args()

    tasks = load_tasks(
        max_videos=args.max,
        skip_youtube=args.skip_yt,
        all_glosses=args.all_glosses,
    )
    if not tasks:
        print("Nothing to download (all target .mp4 files already exist or no tasks).")
        return

    scope = "all glosses" if args.all_glosses else "MVP vocabulary only"
    print(f"WLASL video download: {len(tasks)} videos to fetch ({scope}).")
    print(f"Output dir: {VIDEOS_DIR}")
    print()

    try:
        import requests
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; WLASL-download/1.0)"})
    except ImportError:
        session = None

    ok = 0
    fail = 0
    for video_id, url, gloss in tqdm(tasks, desc="Downloading"):
        if is_youtube(url):
            if download_youtube_ytdlp(video_id, url):
                ok += 1
            else:
                fail += 1
            time.sleep(random.uniform(1.0, 2.0))
        else:
            if session and download_direct(video_id, url, session):
                ok += 1
            else:
                fail += 1
            time.sleep(random.uniform(0.5, 1.0))

    print()
    print(f"Done. OK: {ok}, Failed: {fail}")
    print("Then run ingest_wlasl (from main branch) so pipeline sees the new videos.")


if __name__ == "__main__":
    main()
