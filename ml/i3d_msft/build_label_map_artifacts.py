"""
Rebuild the exact class mapping artifacts used for I3D inference.

Why
----
The label->index mapping used by training/eval is derived from the *filtered*
train split CSV (after dropping missing/unreadable clips). During Modal
training, only the checkpoint .pt is uploaded; the filtered split CSV is not.

This script reproduces the filtered train CSV and then exports:
  - `filtered_train.csv` (for `evaluate.py --gloss-dict-csv`)
  - `label_map.json` (convenience; not required by our evaluate script)

It mirrors the filtering logic in `ml/i3d_msft/train.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from i3d_msft.export_label_map import build_gloss_dict_from_csv
from i3d_msft.s3_data import download_clip_subset, download_splits, get_s3_client
from i3d_msft.train import (
    _is_readable_video,
    _select_filenames_with_val_coverage,
    _write_filtered_split,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild filtered train label mapping for I3D.")
    p.add_argument("--bucket", required=True, help="S3 bucket with processed data + split plans.")
    p.add_argument("--region", default="ca-central-1")
    p.add_argument("--plan-id", required=True, help="Split plan id (e.g. candidate-ac-eval-v4).")
    p.add_argument("--mvp", action="store_true", default=True, help="Use processed/mvp prefixes.")
    p.add_argument(
        "--clip-limit",
        type=int,
        default=0,
        help="Same semantics as train.py: 0 or negative means 'no limit' (None).",
    )
    p.add_argument(
        "--no-verify-readable",
        action="store_true",
        help="Disable OpenCV decode checks during filtering.",
    )
    p.add_argument(
        "--local-root",
        default="workdir/i3d_msft_label_map",
        help="Local workspace root for downloaded splits/clips.",
    )
    p.add_argument(
        "--upload-to-s3-base",
        default=None,
        help=(
            "Optional S3 base prefix to upload artifacts under, e.g. "
            "'models/i3d/modal/<plan>/<run_id>'."
        ),
    )
    args = p.parse_args()

    s3 = get_s3_client(args.region)

    local_root = Path(args.local_root).resolve()
    if local_root.exists():
        # Keep this workspace reproducible and avoid mixing runs.
        shutil.rmtree(local_root)
    local_root.mkdir(parents=True, exist_ok=True)

    splits_dir = local_root / "splits" / args.plan_id
    clips_root = local_root / "clips" / args.plan_id

    # 1) Download plan splits CSVs.
    split_paths = download_splits(s3, args.bucket, args.plan_id, splits_dir, mvp=args.mvp)
    train_csv = split_paths["train"]
    val_csv = split_paths["val"]

    # 2) Select which clip filenames were downloaded in train.py.
    clip_limit = None if args.clip_limit is None or args.clip_limit <= 0 else args.clip_limit
    needed = _select_filenames_with_val_coverage(
        train_csv=train_csv, val_csv=val_csv, limit=clip_limit
    )
    needed_set = set(needed) if clip_limit is not None else None

    # 3) Download exactly those clip objects.
    downloaded, skipped = download_clip_subset(
        s3, args.bucket, needed, clips_root, mvp=args.mvp
    )
    print(f"[label-map] clips downloaded={downloaded} skipped={skipped} requested={len(needed)}")

    # 4) Rebuild the filtered train CSV using the same logic as train.py.
    filtered_splits_dir = local_root / "filtered_splits" / args.plan_id
    filtered_train_csv = filtered_splits_dir / "train.csv"
    kept, dropped = _write_filtered_split(
        src_csv=train_csv,
        dst_csv=filtered_train_csv,
        clips_root=clips_root,
        allowed_filenames=needed_set,
        verify_readable=not args.no_verify_readable,
    )
    print(f"[label-map] filtered_train.csv kept={kept} dropped={dropped}")

    if kept <= 0:
        raise RuntimeError("Filtered train split is empty; cannot build label mapping.")

    # 5) Export gloss->index mapping.
    gloss_dict = build_gloss_dict_from_csv(filtered_train_csv)
    label_map_path = local_root / "label_map.json"
    _write_json(
        label_map_path,
        {
            "source_filtered_train_csv": str(filtered_train_csv),
            "num_classes": len(gloss_dict),
            "gloss_to_index": gloss_dict,
        },
    )
    print(f"[label-map] wrote {label_map_path} num_classes={len(gloss_dict)}")

    # 6) Optionally upload to S3 for your checkpoint folder.
    if args.upload_to_s3_base:
        base = args.upload_to_s3_base.strip("/").strip()
        upload_csv_key = f"{base}/filtered_train.csv"
        upload_label_key = f"{base}/label_map.json"

        s3.upload_file(str(filtered_train_csv), args.bucket, upload_csv_key)
        s3.upload_file(str(label_map_path), args.bucket, upload_label_key)
        print(f"[label-map] uploaded:")
        print(f"  - s3://{args.bucket}/{upload_csv_key}")
        print(f"  - s3://{args.bucket}/{upload_label_key}")


if __name__ == "__main__":  # pragma: no cover
    main()

