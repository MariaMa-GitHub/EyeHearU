"""
Benchmark: Sign-Speak API vs EyeHearU backend on the same val clips.

Usage:
    python run_benchmark.py pick         # Step 1: download 10 val clips from S3
    python run_benchmark.py sign-speak   # Step 2: call Sign-Speak API
    python run_benchmark.py ours         # Step 3: call our backend /predict
    python run_benchmark.py compare      # Step 4: side-by-side comparison
    python run_benchmark.py all          # Steps 1–4 in sequence
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
CLIPS_DIR = SCRIPT_DIR / "clips"
SAMPLES_JSON = RESULTS_DIR / "samples.json"
SIGN_SPEAK_RESULTS = RESULTS_DIR / "sign_speak_results.json"
OURS_RESULTS = RESULTS_DIR / "ours_results.json"
COMPARISON_JSON = RESULTS_DIR / "comparison.json"

NUM_SAMPLES = 10


def _load_env():
    """Read .env in script dir (simple key=value, no quotes)."""
    env_path = SCRIPT_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# ── Step 1: pick clips ──────────────────────────────────────────

def step_pick():
    """Download val.csv from S3, pick NUM_SAMPLES clips, download them."""
    import boto3

    bucket = _env("AWS_S3_BUCKET", "eye-hear-u-public-data-ca1")
    region = _env("AWS_S3_REGION", "ca-central-1")
    s3 = boto3.client("s3", region_name=region)

    # Find active plan
    plan_key = "processed/mvp/i3d/split_plans/ACTIVE_PLAN.json"
    plan = json.loads(s3.get_object(Bucket=bucket, Key=plan_key)["Body"].read())
    plan_id = plan["active_plan_id"]
    print(f"[pick] Active plan: {plan_id}")

    # Download val.csv
    val_key = f"processed/mvp/i3d/split_plans/{plan_id}/splits/val.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    local_val = RESULTS_DIR / "val.csv"
    s3.download_file(bucket, val_key, str(local_val))

    with open(local_val, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"[pick] Val set has {len(rows)} clips")

    # Pick diverse glosses: one clip per unique gloss, then sample
    by_gloss: dict[str, list[dict]] = {}
    for r in rows:
        by_gloss.setdefault(r["gloss"], []).append(r)

    glosses = list(by_gloss.keys())
    random.seed(42)
    random.shuffle(glosses)
    samples = []
    for g in glosses:
        if len(samples) >= NUM_SAMPLES:
            break
        samples.append(random.choice(by_gloss[g]))

    # Download clips
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    clips_prefix = "processed/mvp/clips"
    for s in samples:
        fname = s["filename"]
        local = CLIPS_DIR / Path(fname).name
        s["local_path"] = str(local)
        if local.exists():
            print(f"  [skip] {local.name}")
            continue
        s3_key = f"{clips_prefix}/{fname}"
        print(f"  [download] {s3_key} -> {local.name}")
        try:
            s3.download_file(bucket, s3_key, str(local))
        except Exception as e:
            print(f"  [error] {e}")
            s["local_path"] = ""

    SAMPLES_JSON.write_text(json.dumps(samples, indent=2), encoding="utf-8")
    print(f"[pick] Saved {len(samples)} samples to {SAMPLES_JSON}")


# ── Step 2: Sign-Speak API ──────────────────────────────────────

def step_sign_speak():
    """Call Sign-Speak /recognize-sign for each clip."""
    api_key = _env("SIGN_SPEAK_API_KEY")
    if not api_key:
        print("[sign-speak] ERROR: SIGN_SPEAK_API_KEY not set in .env")
        sys.exit(1)

    samples = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    results = []

    for i, s in enumerate(samples):
        local = Path(s.get("local_path", ""))
        if not local.exists():
            print(f"  [{i+1}/{len(samples)}] SKIP {s['gloss']} — file missing")
            results.append({**s, "sign_speak_prediction": None, "sign_speak_raw": None})
            continue

        b64 = base64.standard_b64encode(local.read_bytes()).decode("ascii")
        body = {
            "payload": b64,
            "single_recognition_mode": True,
            "request_class": "BLOCKING",
            "model_version": "SLR.2.sm",
        }
        print(f"  [{i+1}/{len(samples)}] {s['gloss']} ({local.name}, {local.stat().st_size//1024} KB) ...", end=" ", flush=True)

        try:
            resp = requests.post(
                "https://api.sign-speak.com/recognize-sign",
                headers={
                    "X-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=60,
            )
            data = resp.json() if resp.status_code == 200 else {"error": resp.text, "status": resp.status_code}
            pred = data.get("prediction") or data.get("sign") or data.get("text") or data.get("result") or str(data)
            print(f"-> {pred}")
            results.append({**s, "sign_speak_prediction": pred, "sign_speak_raw": data})
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({**s, "sign_speak_prediction": None, "sign_speak_raw": str(e)})

        time.sleep(1)

    SIGN_SPEAK_RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[sign-speak] Saved {len(results)} results to {SIGN_SPEAK_RESULTS}")


# ── Step 3: Our backend API ─────────────────────────────────────

def step_ours():
    """Call our backend POST /api/v1/predict for each clip."""
    api_url = _env("EYEHEARU_API_URL", "http://localhost:8000").rstrip("/")
    samples = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    results = []

    for i, s in enumerate(samples):
        local = Path(s.get("local_path", ""))
        if not local.exists():
            print(f"  [{i+1}/{len(samples)}] SKIP {s['gloss']} — file missing")
            results.append({**s, "ours_prediction": None, "ours_raw": None})
            continue

        print(f"  [{i+1}/{len(samples)}] {s['gloss']} ({local.name}) ...", end=" ", flush=True)

        try:
            with open(local, "rb") as f:
                resp = requests.post(
                    f"{api_url}/api/v1/predict",
                    files={"file": (local.name, f, "video/mp4")},
                    headers={"bypass-tunnel-reminder": "true"},
                    timeout=120,
                )
            data = resp.json() if resp.status_code == 200 else {"error": resp.text, "status": resp.status_code}
            pred = data.get("sign", str(data))
            conf = data.get("confidence", 0)
            print(f"-> {pred} ({conf:.2f})")
            results.append({**s, "ours_prediction": pred, "ours_confidence": conf, "ours_raw": data})
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({**s, "ours_prediction": None, "ours_raw": str(e)})

    OURS_RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[ours] Saved {len(results)} results to {OURS_RESULTS}")


# ── Step 4: compare ─────────────────────────────────────────────

def _normalize(s: str | None) -> str:
    """Lowercase, strip, underscores to spaces for fair comparison."""
    if not s:
        return ""
    return s.lower().strip().replace("_", " ").replace("-", " ")


def step_compare():
    """Load both result files and compare to ground truth."""
    ss = json.loads(SIGN_SPEAK_RESULTS.read_text(encoding="utf-8"))
    ours = json.loads(OURS_RESULTS.read_text(encoding="utf-8"))

    rows = []
    ss_correct = 0
    ours_correct = 0
    total = 0

    for s_item, o_item in zip(ss, ours):
        gt = _normalize(s_item["gloss"])
        sp = _normalize(str(s_item.get("sign_speak_prediction") or ""))
        op = _normalize(str(o_item.get("ours_prediction") or ""))

        sp_match = gt == sp or gt in sp or sp in gt
        op_match = gt == op or gt in op or op in gt

        total += 1
        if sp_match:
            ss_correct += 1
        if op_match:
            ours_correct += 1

        row = {
            "ground_truth": gt,
            "sign_speak": sp,
            "sign_speak_match": sp_match,
            "ours": op,
            "ours_confidence": o_item.get("ours_confidence"),
            "ours_match": op_match,
        }
        rows.append(row)

    summary = {
        "total": total,
        "sign_speak_correct": ss_correct,
        "sign_speak_accuracy": round(ss_correct / max(total, 1), 4),
        "ours_correct": ours_correct,
        "ours_accuracy": round(ours_correct / max(total, 1), 4),
    }

    output = {"summary": summary, "details": rows}
    COMPARISON_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"{'Ground Truth':<20} {'Sign-Speak':<20} {'Ours':<20}")
    print("-" * 60)
    for r in rows:
        gt_mark = "✓" if r["sign_speak_match"] else "✗"
        our_mark = "✓" if r["ours_match"] else "✗"
        print(f"{r['ground_truth']:<20} {r['sign_speak']:<18}{gt_mark}  {r['ours']:<18}{our_mark}")
    print("-" * 60)
    print(f"Sign-Speak accuracy: {summary['sign_speak_correct']}/{total} = {summary['sign_speak_accuracy']:.0%}")
    print(f"EyeHearU accuracy:   {summary['ours_correct']}/{total} = {summary['ours_accuracy']:.0%}")
    print("=" * 60)
    print(f"\nFull results: {COMPARISON_JSON}")


# ── Main ─────────────────────────────────────────────────────────

STEPS = {
    "pick": step_pick,
    "sign-speak": step_sign_speak,
    "ours": step_ours,
    "compare": step_compare,
}


def main():
    parser = argparse.ArgumentParser(description="Sign-Speak vs EyeHearU benchmark")
    parser.add_argument("step", choices=[*STEPS, "all"], help="Which step to run")
    args = parser.parse_args()

    _load_env()

    if args.step == "all":
        for name, fn in STEPS.items():
            print(f"\n{'='*20} {name} {'='*20}")
            fn()
    else:
        STEPS[args.step]()


if __name__ == "__main__":
    main()
