"""
Create versioned, rollback-safe I3D split plans on S3.

Why this exists
---------------
- We want evaluation to be clean (no leakage) while still expanding training data.
- We also want safe rollout/rollback of split definitions.

Strategy (default)
------------------
1) ASL Citizen becomes the primary evaluation source:
   - Re-split ASL Citizen clips by signer into train/val/test (signer-disjoint).
2) Supplemental datasets (e.g. MS-ASL) are train-only.
3) Upload split CSVs under a versioned prefix:
   s3://<bucket>/processed[/mvp]/i3d/split_plans/<plan_id>/splits/{train,val,test}.csv
4) Upload a manifest with stats + quality checks.
5) Optionally update ACTIVE_PLAN.json pointer; rollback = point ACTIVE to old plan.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pipeline_config import (
    S3_BUCKET,
    get_processed_base,
    get_processed_prefix_s3,
    get_s3_client,
    is_cloud,
    read_text_from_s3,
    s3_object_exists,
    write_text_to_s3,
)


@dataclass
class ClipRow:
    source: str
    signer_id: str
    gloss: str
    assigned_split: str
    filename: str
    s3_key: str


def _load_processed_clips(mvp: bool) -> list[dict]:
    if is_cloud():
        prefix = get_processed_prefix_s3(mvp)
        candidates = [
            f"{prefix}/processed_clips.csv",
            f"{prefix}/metadata/processed_clips.csv",
        ]
        key = next((k for k in candidates if s3_object_exists(k)), None)
        if key is None:
            raise FileNotFoundError(
                "processed_clips.csv not found in expected S3 locations: "
                + ", ".join(f"s3://{S3_BUCKET}/{k}" for k in candidates)
            )
        text = read_text_from_s3(key)
        print(f"[plan] Loaded s3://{S3_BUCKET}/{key}")
        return list(csv.DictReader(io.StringIO(text)))

    local_candidates = [
        get_processed_base(mvp) / "processed_clips.csv",
        get_processed_base(mvp) / "metadata" / "processed_clips.csv",
    ]
    path = next((p for p in local_candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "processed_clips.csv not found in local candidates: "
            + ", ".join(str(p) for p in local_candidates)
        )
    with open(path, newline="", encoding="utf-8") as f:
        print(f"[plan] Loaded {path}")
        return list(csv.DictReader(f))


def _parse_s3_uri(uri: str) -> tuple[str, str] | None:
    uri = (uri or "").strip()
    if not uri.startswith("s3://"):
        return None
    rest = uri[5:]
    if "/" not in rest:
        return None
    bucket, key = rest.split("/", 1)
    return bucket, key


def _filename_from_clip_path(clip_path: str) -> tuple[str, str]:
    """
    Return (filename_for_i3d_csv, full_s3_key).
    filename_for_i3d_csv is relative under clips root if possible:
      e.g. 'train/hello/abc.mp4'
    """
    parsed = _parse_s3_uri(clip_path)
    if parsed:
        _, key = parsed
        marker = "/clips/"
        if marker in key:
            return key.split(marker, 1)[1], key
        return Path(key).name, key

    p = Path(clip_path)
    parts = p.parts
    if "clips" in parts:
        idx = parts.index("clips")
        rel = Path(*parts[idx + 1 :]).as_posix()
    else:
        rel = p.name
    return rel, rel


def _build_rows(raw_rows: list[dict]) -> list[ClipRow]:
    out: list[ClipRow] = []
    for r in raw_rows:
        source = (r.get("source") or "").strip().lower()
        signer_id = str(r.get("signer_id") or "").strip()
        gloss = (r.get("gloss") or "").strip().lower()
        split = (r.get("split") or "").strip().lower()
        clip_path = (r.get("clip_path") or "").strip()
        if not source or not gloss or split not in {"train", "val", "test"}:
            continue
        filename, s3_key = _filename_from_clip_path(clip_path)
        if not filename:
            continue
        out.append(
            ClipRow(
                source=source,
                signer_id=signer_id,
                gloss=gloss,
                assigned_split=split,
                filename=filename,
                s3_key=s3_key,
            )
        )
    return out


def _split_asl_citizen_by_signer(
    rows: list[ClipRow],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    signer_to_rows: dict[str, list[ClipRow]] = defaultdict(list)
    for r in rows:
        signer_to_rows[r.signer_id].append(r)

    signers = sorted(signer_to_rows.keys())
    if len(signers) < 3:
        raise RuntimeError("Need at least 3 ASL Citizen signers for signer-disjoint split.")

    counts = {s: len(signer_to_rows[s]) for s in signers}
    total = sum(counts.values())
    target_val = max(1, int(round(total * val_ratio)))
    target_test = max(1, int(round(total * test_ratio)))
    if target_val + target_test >= total:
        target_test = max(1, target_test // 2)
        target_val = max(1, min(target_val, total - target_test - 1))

    rng = random.Random(seed)
    ordered = signers[:]
    rng.shuffle(ordered)
    ordered.sort(key=lambda s: counts[s], reverse=True)

    signer_split: dict[str, str] = {}
    val_sum = 0
    test_sum = 0
    val_signers = 0
    test_signers = 0

    for s in ordered:
        c = counts[s]
        need_val = max(0, target_val - val_sum)
        need_test = max(0, target_test - test_sum)

        if need_val == 0 and need_test == 0:
            signer_split[s] = "train"
            continue

        if need_val > need_test:
            signer_split[s] = "val"
            val_sum += c
            val_signers += 1
        else:
            signer_split[s] = "test"
            test_sum += c
            test_signers += 1

    # Ensure both holdout splits have at least one signer.
    if val_signers == 0 or test_signers == 0:
        largest = ordered[0]
        smallest = ordered[-1]
        signer_split[largest] = "val"
        signer_split[smallest] = "test"
        for s in ordered[1:-1]:
            signer_split.setdefault(s, "train")
    else:
        for s in ordered:
            signer_split.setdefault(s, "train")

    return signer_split


def _assign_new_splits(
    rows: list[ClipRow],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    supplemental_train_only: bool,
) -> list[ClipRow]:
    ac_rows = [r for r in rows if r.source == "asl_citizen"]
    signer_split = _split_asl_citizen_by_signer(ac_rows, val_ratio, test_ratio, seed)

    out: list[ClipRow] = []
    for r in rows:
        split = r.assigned_split
        if r.source == "asl_citizen":
            split = signer_split[r.signer_id]
        elif supplemental_train_only:
            split = "train"

        out.append(
            ClipRow(
                source=r.source,
                signer_id=r.signer_id,
                gloss=r.gloss,
                assigned_split=split,
                filename=r.filename,
                s3_key=r.s3_key,
            )
        )
    return out


def _csv_text(rows: list[ClipRow], split: str) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["user", "filename", "gloss"])
    for r in rows:
        if r.assigned_split == split:
            # Prefix source to keep user ids unambiguous across datasets.
            w.writerow([f"{r.source}:{r.signer_id}", r.filename, r.gloss])
    return buf.getvalue()


def _split_counts(rows: list[ClipRow]) -> dict[str, int]:
    c = Counter(r.assigned_split for r in rows)
    return {"train": c["train"], "val": c["val"], "test": c["test"]}


def _source_split_counts(rows: list[ClipRow]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for r in rows:
        out[r.source][r.assigned_split] += 1
    return dict(out)


def _signer_overlap(rows: list[ClipRow], source: str) -> dict[str, int]:
    sets: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for r in rows:
        if r.source == source:
            sets[r.assigned_split].add(r.signer_id)
    return {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }


def _gloss_coverage(rows: list[ClipRow]) -> dict[str, int]:
    train = {r.gloss for r in rows if r.assigned_split == "train"}
    val = {r.gloss for r in rows if r.assigned_split == "val"}
    test = {r.gloss for r in rows if r.assigned_split == "test"}
    return {
        "train_classes": len(train),
        "val_classes": len(val),
        "test_classes": len(test),
        "val_not_in_train": len(val - train),
        "test_not_in_train": len(test - train),
    }


def _sample_s3_existence(rows: list[ClipRow], sample_size: int, seed: int) -> dict:
    if not is_cloud():
        return {"checked": 0, "missing": 0}
    rng = random.Random(seed)
    sample = rows if len(rows) <= sample_size else rng.sample(rows, sample_size)
    s3 = get_s3_client()
    missing = 0
    for r in sample:
        try:
            s3.head_object(Bucket=S3_BUCKET, Key=r.s3_key)
        except Exception:
            missing += 1
    return {"checked": len(sample), "missing": missing}


def _drop_missing_s3_rows(rows: list[ClipRow], clips_prefix: str) -> tuple[list[ClipRow], int]:
    """
    Drop rows whose referenced clip object is not present in S3.
    Uses one S3 listing pass for efficiency.
    """
    if not is_cloud():
        return rows, 0
    s3 = get_s3_client()
    existing: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=clips_prefix):
        for obj in page.get("Contents", []):
            existing.add(obj["Key"])
    kept = [r for r in rows if r.s3_key in existing]
    dropped = len(rows) - len(kept)
    return kept, dropped


def _plan_root_prefix(mvp: bool) -> str:
    base = get_processed_prefix_s3(mvp) if is_cloud() else "processed"
    return f"{base}/i3d/split_plans"


def _write_plan_to_s3(plan_prefix: str, rows: list[ClipRow], manifest: dict) -> None:
    for split in ("train", "val", "test"):
        key = f"{plan_prefix}/splits/{split}.csv"
        write_text_to_s3(_csv_text(rows, split), key)
        print(f"[plan] Uploaded s3://{S3_BUCKET}/{key}")
    manifest_key = f"{plan_prefix}/manifest.json"
    write_text_to_s3(json.dumps(manifest, indent=2), manifest_key)
    print(f"[plan] Uploaded s3://{S3_BUCKET}/{manifest_key}")


def _active_pointer_key(mvp: bool) -> str:
    return f"{_plan_root_prefix(mvp)}/ACTIVE_PLAN.json"


def _read_active_plan_id(mvp: bool) -> str | None:
    key = _active_pointer_key(mvp)
    if not (is_cloud() and s3_object_exists(key)):
        return None
    data = json.loads(read_text_from_s3(key))
    return data.get("active_plan_id")


def _activate_plan(mvp: bool, plan_id: str) -> None:
    if not is_cloud():
        raise RuntimeError("Plan activation is only supported in cloud mode.")
    root = _plan_root_prefix(mvp)
    manifest_key = f"{root}/{plan_id}/manifest.json"
    if not s3_object_exists(manifest_key):
        raise FileNotFoundError(f"Cannot activate missing plan: s3://{S3_BUCKET}/{manifest_key}")
    payload = {
        "active_plan_id": plan_id,
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }
    key = _active_pointer_key(mvp)
    write_text_to_s3(json.dumps(payload, indent=2), key)
    print(f"[plan] ACTIVE set to {plan_id} at s3://{S3_BUCKET}/{key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and manage rollback-safe I3D split plans.")
    parser.add_argument("--mvp", action="store_true", help="Use processed/mvp/ paths.")
    parser.add_argument("--plan-id", default=None, help="Plan ID. Default: timestamp-based.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="ASL Citizen signer val ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="ASL Citizen signer test ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--supplemental-train-only",
        action="store_true",
        default=True,
        help="Force non-ASL-Citizen sources to train-only (default on).",
    )
    parser.add_argument(
        "--sample-s3-check",
        type=int,
        default=50,
        help="Sample size for S3 object existence validation.",
    )
    parser.add_argument(
        "--drop-missing-s3",
        action="store_true",
        help="Drop rows referencing missing S3 clip objects before split generation.",
    )
    parser.add_argument(
        "--set-active",
        action="store_true",
        help="After creating plan, set ACTIVE_PLAN pointer to it.",
    )
    parser.add_argument(
        "--activate-plan",
        default=None,
        help="Only activate an existing plan ID (rollback switch).",
    )
    args = parser.parse_args()

    if args.activate_plan:
        _activate_plan(args.mvp, args.activate_plan)
        return

    if not is_cloud():
        raise RuntimeError("This script is intended for S3-backed cloud mode. Set PIPELINE_ENV=dev/staging/prod.")

    raw = _load_processed_clips(args.mvp)
    rows = _build_rows(raw)
    if not rows:
        raise RuntimeError("No valid rows found in processed_clips.csv.")

    dropped_missing = 0
    if args.drop_missing_s3:
        prefix = f"{get_processed_prefix_s3(args.mvp)}/clips/"
        rows, dropped_missing = _drop_missing_s3_rows(rows, prefix)
        print(f"[plan] Dropped missing S3 clip references: {dropped_missing}")

    planned = _assign_new_splits(
        rows=rows,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        supplemental_train_only=args.supplemental_train_only,
    )

    plan_id = args.plan_id or f"candidate-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    root = _plan_root_prefix(args.mvp)
    plan_prefix = f"{root}/{plan_id}"

    prev_active = _read_active_plan_id(args.mvp)
    counts = _split_counts(planned)
    source_counts = _source_split_counts(planned)
    overlap_ac = _signer_overlap(planned, "asl_citizen")
    overlap_msasl = _signer_overlap(planned, "msasl")
    coverage = _gloss_coverage(planned)
    s3_check = _sample_s3_existence(planned, args.sample_s3_check, args.seed)

    manifest = {
        "plan_id": plan_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "previous_active_plan_id": prev_active,
        "strategy": {
            "asl_citizen": "signer_disjoint_train_val_test",
            "supplemental_sources": "train_only" if args.supplemental_train_only else "keep_original",
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
        },
        "counts": counts,
        "source_split_counts": source_counts,
        "quality_checks": {
            "asl_citizen_signer_overlap": overlap_ac,
            "msasl_signer_overlap": overlap_msasl,
            "gloss_coverage": coverage,
            "s3_existence_sample": s3_check,
            "dropped_missing_rows": dropped_missing,
        },
        "artifacts": {
            "train_csv": f"s3://{S3_BUCKET}/{plan_prefix}/splits/train.csv",
            "val_csv": f"s3://{S3_BUCKET}/{plan_prefix}/splits/val.csv",
            "test_csv": f"s3://{S3_BUCKET}/{plan_prefix}/splits/test.csv",
            "manifest_json": f"s3://{S3_BUCKET}/{plan_prefix}/manifest.json",
            "active_pointer": f"s3://{S3_BUCKET}/{_active_pointer_key(args.mvp)}",
        },
        "rollback": {
            "how": "Run with --activate-plan <previous_active_plan_id> to switch back immediately.",
            "safe_publish_flow": [
                "Create candidate plan without --set-active.",
                "Run short training/eval on candidate.",
                "If metrics acceptable, rerun with --activate-plan <candidate_id>.",
                "If not acceptable, keep ACTIVE unchanged or point back to previous ID.",
            ],
        },
    }

    _write_plan_to_s3(plan_prefix, planned, manifest)

    print("[plan] Summary")
    print(f"  plan_id: {plan_id}")
    print(f"  previous_active: {prev_active}")
    print(f"  counts: {counts}")
    print(f"  ASL signer overlap: {overlap_ac}")
    print(f"  S3 existence sample: {s3_check}")

    if args.set_active:
        _activate_plan(args.mvp, plan_id)
    else:
        print("[plan] Candidate created; ACTIVE pointer unchanged.")


if __name__ == "__main__":
    main()
