"""
Prepare local ASL Citizen data layout for microsoft/ASL-citizen-code I3D training.

Target layout:
  <output_root>/
    videos/
      *.mp4
    splits/
      train.csv
      val.csv
      test.csv

CSV format matches I3D loader expectation (first 3 columns used):
  user,filename,gloss

Usage examples:
  # MVP records from S3, write split CSVs only
  PIPELINE_ENV=dev python prepare_i3d_from_s3.py --mvp

  # MVP records from S3, also download referenced videos to local videos/
  PIPELINE_ENV=dev python prepare_i3d_from_s3.py --mvp --download-videos
"""

from __future__ import annotations

import argparse
import csv
import io
import random
from dataclasses import dataclass
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
class I3DRecord:
    user: str
    filename: str
    gloss: str
    split: str
    s3_key: str


def _parse_s3_uri(uri: str) -> tuple[str, str] | None:
    uri = (uri or "").strip()
    if not uri.startswith("s3://"):
        return None
    rest = uri[5:]
    if "/" not in rest:
        return None
    bucket, key = rest.split("/", 1)
    return bucket, key


def _load_ingested_asl_citizen(mvp: bool) -> list[dict]:
    if is_cloud():
        prefix = get_processed_prefix_s3(mvp)
        candidates = [
            f"{prefix}/ingested_asl_citizen.csv",
            f"{prefix}/metadata/ingested_asl_citizen.csv",
        ]
        key = next((k for k in candidates if s3_object_exists(k)), None)
        if key is None:
            raise FileNotFoundError(
                "Could not find ingested_asl_citizen.csv in any expected S3 location: "
                + ", ".join(f"s3://{S3_BUCKET}/{k}" for k in candidates)
            )
        text = read_text_from_s3(key)
        print(f"[i3d] Loaded s3://{S3_BUCKET}/{key}")
        return list(csv.DictReader(io.StringIO(text)))

    local_candidates = [
        get_processed_base(mvp) / "ingested_asl_citizen.csv",
        get_processed_base(mvp) / "metadata" / "ingested_asl_citizen.csv",
    ]
    local_path = next((p for p in local_candidates if p.exists()), None)
    if local_path is None:
        raise FileNotFoundError(
            "Could not find local ingested_asl_citizen.csv in: "
            + ", ".join(str(p) for p in local_candidates)
        )
    with open(local_path, newline="", encoding="utf-8") as f:
        print(f"[i3d] Loaded {local_path}")
        return list(csv.DictReader(f))


def _to_i3d_records(rows: list[dict]) -> list[I3DRecord]:
    out: list[I3DRecord] = []
    for row in rows:
        split = (row.get("split") or "").strip().lower()
        if split not in {"train", "val", "test"}:
            continue

        gloss = (row.get("gloss") or "").strip()
        if not gloss:
            continue

        filename = (row.get("clip_id") or "").strip()
        if not filename:
            continue

        signer_id = str(row.get("signer_id") or "").strip()
        src_path = (row.get("src_path") or "").strip()

        parsed = _parse_s3_uri(src_path)
        if parsed:
            _, s3_key = parsed
        else:
            s3_key = f"raw/asl_citizen/videos/{filename}"

        out.append(
            I3DRecord(
                user=signer_id,
                filename=filename,
                gloss=gloss,
                split=split,
                s3_key=s3_key,
            )
        )
    return out


def _write_split_csv(records: list[I3DRecord], split: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user", "filename", "gloss"])
        for r in records:
            if r.split == split:
                writer.writerow([r.user, r.filename, r.gloss])
    print(f"[i3d] Wrote {split}.csv -> {path}")


def _split_csv_text(records: list[I3DRecord], split: str) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["user", "filename", "gloss"])
    for r in records:
        if r.split == split:
            writer.writerow([r.user, r.filename, r.gloss])
    return buf.getvalue()


def _upload_split_csvs_to_s3(records: list[I3DRecord], prefix: str) -> None:
    if not is_cloud():
        raise RuntimeError("S3 upload requested but PIPELINE_ENV is local.")
    for split in ("train", "val", "test"):
        key = f"{prefix}/{split}.csv"
        write_text_to_s3(_split_csv_text(records, split), key)
        print(f"[i3d] Uploaded {split}.csv -> s3://{S3_BUCKET}/{key}")


def _count_splits(records: list[I3DRecord]) -> dict[str, int]:
    return {
        "train": sum(1 for r in records if r.split == "train"),
        "val": sum(1 for r in records if r.split == "val"),
        "test": sum(1 for r in records if r.split == "test"),
    }


def _auto_split_missing(
    records: list[I3DRecord],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[I3DRecord]:
    counts = _count_splits(records)
    needs_val = counts["val"] == 0
    needs_test = counts["test"] == 0
    if not (needs_val or needs_test):
        return records

    train_records = [r for r in records if r.split == "train"]
    if not train_records:
        raise RuntimeError("Cannot auto-split because no train records exist.")

    user_to_records: dict[str, list[I3DRecord]] = {}
    for r in train_records:
        user_to_records.setdefault(r.user, []).append(r)

    users = list(user_to_records.keys())
    if len(users) < 3:
        raise RuntimeError(
            "Need at least 3 distinct signers in train split to auto-create val/test."
        )

    rng = random.Random(seed)
    rng.shuffle(users)

    n_users = len(users)
    n_val = max(1, int(round(n_users * val_ratio))) if needs_val else 0
    n_test = max(1, int(round(n_users * test_ratio))) if needs_test else 0
    n_holdout = n_val + n_test
    if n_holdout >= n_users:
        n_holdout = n_users - 1
        if needs_val and needs_test:
            n_val = max(1, n_holdout // 2)
            n_test = max(1, n_holdout - n_val)
        elif needs_val:
            n_val = n_holdout
        else:
            n_test = n_holdout

    val_users = set(users[:n_val]) if needs_val else set()
    test_users = set(users[n_val : n_val + n_test]) if needs_test else set()

    remapped: list[I3DRecord] = []
    for r in records:
        new_split = r.split
        if r.split == "train":
            if r.user in val_users:
                new_split = "val"
            elif r.user in test_users:
                new_split = "test"
        remapped.append(
            I3DRecord(
                user=r.user,
                filename=r.filename,
                gloss=r.gloss,
                split=new_split,
                s3_key=r.s3_key,
            )
        )

    new_counts = _count_splits(remapped)
    print(
        "[i3d] Auto-split created missing splits by signer: "
        f"train={new_counts['train']}, val={new_counts['val']}, test={new_counts['test']}"
    )
    return remapped


def _download_videos(
    records: list[I3DRecord],
    videos_dir: Path,
    skip_existing: bool = True,
    max_videos: int | None = None,
) -> None:
    if not is_cloud():
        print("[i3d] Skipping S3 downloads in local mode.")
        return

    s3 = get_s3_client()
    videos_dir.mkdir(parents=True, exist_ok=True)

    unique: list[I3DRecord] = []
    seen: set[str] = set()
    for r in records:
        if r.filename not in seen:
            unique.append(r)
            seen.add(r.filename)

    if max_videos is not None:
        unique = unique[:max_videos]

    print(f"[i3d] Downloading up to {len(unique)} videos from s3://{S3_BUCKET}/...")
    downloaded = 0
    skipped = 0
    failed = 0

    for idx, r in enumerate(unique, start=1):
        dst = videos_dir / r.filename
        if skip_existing and dst.exists():
            skipped += 1
            continue
        try:
            s3.download_file(S3_BUCKET, r.s3_key, str(dst))
            downloaded += 1
        except Exception as exc:
            failed += 1
            print(f"[i3d] WARN failed {r.s3_key}: {exc}")

        if idx % 200 == 0:
            print(
                f"[i3d] Progress {idx}/{len(unique)} | "
                f"downloaded={downloaded}, skipped={skipped}, failed={failed}"
            )

    print(
        f"[i3d] Download complete | downloaded={downloaded}, "
        f"skipped={skipped}, failed={failed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create microsoft I3D-compatible ASL Citizen splits from current pipeline outputs."
    )
    parser.add_argument(
        "--mvp",
        action="store_true",
        help="Read processed/mvp/ingested_asl_citizen.csv (recommended first).",
    )
    parser.add_argument(
        "--output-root",
        default=str(
            Path(__file__).resolve().parent.parent / "i3d_data" / "ASL_Citizen"
        ),
        help="Local output root that will contain videos/ and splits/.",
    )
    parser.add_argument(
        "--download-videos",
        action="store_true",
        help="Download referenced videos from S3 into <output-root>/videos.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional cap for number of videos to download.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download videos even if local file exists.",
    )
    parser.add_argument(
        "--auto-split-missing",
        action="store_true",
        help="If val/test are empty, create signer-disjoint splits from train.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of train signers reassigned to val when auto-splitting.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Ratio of train signers reassigned to test when auto-splitting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for signer-level auto-splitting.",
    )
    parser.add_argument(
        "--upload-splits-to-s3",
        action="store_true",
        help="Upload generated train/val/test CSVs to S3 for AWS training jobs.",
    )
    parser.add_argument(
        "--s3-splits-prefix",
        default=None,
        help=(
            "S3 prefix to upload split CSVs (default: "
            "processed[/mvp]/i3d/splits based on --mvp)."
        ),
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    splits_dir = output_root / "splits"
    videos_dir = output_root / "videos"

    rows = _load_ingested_asl_citizen(args.mvp)
    records = _to_i3d_records(rows)
    if args.auto_split_missing:
        records = _auto_split_missing(
            records=records,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    if not records:
        raise RuntimeError("No ASL Citizen records found to export.")

    print(f"[i3d] Parsed {len(records)} records.")
    for split in ("train", "val", "test"):
        split_count = sum(1 for r in records if r.split == split)
        print(f"[i3d]   {split}: {split_count}")

    _write_split_csv(records, "train", splits_dir / "train.csv")
    _write_split_csv(records, "val", splits_dir / "val.csv")
    _write_split_csv(records, "test", splits_dir / "test.csv")

    if args.upload_splits_to_s3:
        if args.s3_splits_prefix:
            s3_splits_prefix = args.s3_splits_prefix.strip().strip("/")
        else:
            base = get_processed_prefix_s3(args.mvp) if is_cloud() else "processed"
            s3_splits_prefix = f"{base}/i3d/splits"
        _upload_split_csvs_to_s3(records, s3_splits_prefix)
        print(
            "[i3d] AWS training inputs:\n"
            f"       videos: s3://{S3_BUCKET}/raw/asl_citizen/videos/\n"
            f"       splits: s3://{S3_BUCKET}/{s3_splits_prefix}/"
        )

    if args.download_videos:
        _download_videos(
            records=records,
            videos_dir=videos_dir,
            skip_existing=not args.no_skip_existing,
            max_videos=args.max_videos,
        )
    else:
        print("[i3d] Split CSV export done. Use --download-videos to fetch local videos.")

    print("[i3d] Done.")


if __name__ == "__main__":
    main()
