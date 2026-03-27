"""
Evaluate vendored Microsoft I3D checkpoints on EyeHearU S3 split plans.

Usage examples:
  # Evaluate a checkpoint from S3 on test split
  python -m i3d_msft.evaluate \
    --bucket eye-hear-u-public-data-ca1 \
    --region ca-central-1 \
    --plan-id candidate-ac-eval-v4 \
    --split test \
    --checkpoint-s3-key models/i3d/pretrained/ASL_citizen_I3D_weights.pt

  # Evaluate a local checkpoint path
  python -m i3d_msft.evaluate \
    --bucket eye-hear-u-public-data-ca1 \
    --plan-id candidate-ac-eval-v4 \
    --split val \
    --checkpoint-local workdir/i3d_msft/checkpoints/candidate-ac-eval-v4/best_model.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from i3d_msft.dataset import ASLCitizenI3DDataset
    from i3d_msft.pytorch_i3d import InceptionI3d
    from i3d_msft.s3_data import download_clip_subset, download_splits, get_active_plan_id, get_s3_client
    from i3d_msft.videotransforms import CenterCrop
except ImportError:
    from ml.i3d_msft.dataset import ASLCitizenI3DDataset
    from ml.i3d_msft.pytorch_i3d import InceptionI3d
    from ml.i3d_msft.s3_data import download_clip_subset, download_splits, get_active_plan_id, get_s3_client
    from ml.i3d_msft.videotransforms import CenterCrop


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _read_split_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_gloss_dict_from_csv(path: Path) -> dict[str, int]:
    rows = _read_split_rows(path)
    glosses = sorted(
        {
            (r.get("gloss") or "").strip().lower()
            for r in rows
            if (r.get("gloss") or "").strip()
        }
    )
    return {g: i for i, g in enumerate(glosses)}


def _is_readable_video(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    cap = cv2.VideoCapture(str(path))
    ok, _ = cap.read()
    cap.release()
    return bool(ok)


def _write_filtered_split(
    src_csv: Path,
    dst_csv: Path,
    clips_root: Path,
    limit: int | None,
    verify_readable: bool,
) -> tuple[int, int]:
    rows = _read_split_rows(src_csv)
    kept = []
    dropped = 0
    kept_count = 0
    for row in rows:
        if limit is not None and kept_count >= limit:
            break
        name = (row.get("filename") or "").strip()
        gloss = (row.get("gloss") or "").strip()
        if not name or not gloss:
            dropped += 1
            continue
        clip_path = clips_root / name
        if not clip_path.exists() or clip_path.stat().st_size == 0:
            dropped += 1
            continue
        if verify_readable and not _is_readable_video(clip_path):
            dropped += 1
            continue
        kept.append(
            {
                "user": row.get("user", ""),
                "filename": name,
                "gloss": gloss,
            }
        )
        kept_count += 1

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user", "filename", "gloss"])
        writer.writeheader()
        for row in kept:
            writer.writerow(row)
    return len(kept), dropped


def _topk_hits(clip_logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    topk = clip_logits.topk(k=min(k, clip_logits.shape[1]), dim=1).indices
    hits = topk.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
    return int(hits)


def _compute_mrr_and_dcg(clip_logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    ranked = torch.argsort(clip_logits, dim=1, descending=True)
    labels_col = labels.unsqueeze(1)
    matches = ranked.eq(labels_col)
    rr_sum = 0.0
    dcg_sum = 0.0
    total = clip_logits.shape[0]
    for i in range(total):
        pos = torch.nonzero(matches[i], as_tuple=False)
        if pos.numel() == 0:
            continue  # pragma: no cover – label not in ranked list
        rank = int(pos[0].item()) + 1  # 1-based
        rr_sum += 1.0 / rank
        dcg_sum += 1.0 / np.log2(rank + 1.0)
    return rr_sum / max(total, 1), dcg_sum / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, topk: list[int]) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    loss_sum = 0.0
    topk_hits = {k: 0 for k in topk}
    correct = 0
    rr_total = 0.0
    dcg_total = 0.0
    confusion = defaultdict(int)

    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)
        logits_t = model(clips, pretrained=False)
        clip_logits = torch.max(logits_t, dim=2)[0]

        loss = criterion(clip_logits, labels)
        batch = labels.shape[0]
        total += batch
        loss_sum += loss.item() * batch

        preds = torch.argmax(clip_logits, dim=1)
        correct += (preds == labels).sum().item()

        for k in topk:
            topk_hits[k] += _topk_hits(clip_logits, labels, k)

        mrr, dcg = _compute_mrr_and_dcg(clip_logits, labels)
        rr_total += mrr * batch
        dcg_total += dcg * batch

        for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
            if t != p:
                confusion[(t, p)] += 1

    metrics = {
        "num_samples": total,
        "loss": loss_sum / max(total, 1),
        "top1_acc": correct / max(total, 1),
        "mrr": rr_total / max(total, 1),
        "dcg": dcg_total / max(total, 1),
    }
    for k in topk:
        metrics[f"top{k}_acc"] = topk_hits[k] / max(total, 1)

    top_confusions = sorted(confusion.items(), key=lambda kv: kv[1], reverse=True)[:20]
    metrics["top_confusions"] = [
        {"true_idx": t, "pred_idx": p, "count": c} for (t, p), c in top_confusions
    ]
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate vendored Microsoft I3D on EyeHearU S3 split plans.")
    parser.add_argument("--bucket", required=True, help="S3 bucket with processed data and split plans.")
    parser.add_argument("--region", default="ca-central-1")
    parser.add_argument("--plan-id", default=None, help="Split plan id. If omitted, use ACTIVE_PLAN.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--mvp", action="store_true", default=True, help="Use processed/mvp prefixes (default on).")
    parser.add_argument("--local-root", default="workdir/i3d_msft_eval", help="Local workspace for eval artifacts.")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--clip-limit", type=int, default=None, help="Optional cap on number of split rows to evaluate.")
    parser.add_argument("--no-verify-readable", action="store_true", help="Disable decode check for faster filtering.")
    parser.add_argument(
        "--checkpoint-s3-key",
        default=None,
        help="S3 key to model checkpoint, e.g. models/i3d/pretrained/ASL_citizen_I3D_weights.pt",
    )
    parser.add_argument("--checkpoint-local", default=None, help="Local checkpoint path alternative to --checkpoint-s3-key.")
    parser.add_argument(
        "--topk",
        default="1,5,15,20",
        help="Comma-separated k values for top-k accuracy (default: 1,5,15,20).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON path. Default: <local-root>/eval/<plan_id>/<split>_metrics.json",
    )
    parser.add_argument(
        "--gloss-dict-csv",
        default=None,
        help=(
            "Optional CSV used to build class mapping (gloss->index). "
            "Use training split CSV to keep eval label space aligned with a fine-tuned checkpoint."
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()
    if not args.checkpoint_s3_key and not args.checkpoint_local:
        raise ValueError("Pass either --checkpoint-s3-key or --checkpoint-local.")

    topk = sorted({int(x.strip()) for x in args.topk.split(",") if x.strip()})
    if not topk:
        raise ValueError("Invalid --topk values.")

    device = get_device(args.device)
    print(f"[eval] device={device}")

    s3 = get_s3_client(args.region)
    plan_id = args.plan_id or get_active_plan_id(s3, args.bucket, mvp=args.mvp)
    print(f"[eval] using plan_id={plan_id} split={args.split}")

    root = Path(args.local_root).resolve()
    splits_dir = root / "splits" / plan_id
    clips_root = root / "clips"
    eval_dir = root / "eval" / plan_id
    eval_dir.mkdir(parents=True, exist_ok=True)

    split_paths = download_splits(s3, args.bucket, plan_id, splits_dir, mvp=args.mvp)
    src_split = split_paths[args.split]

    split_rows = _read_split_rows(src_split)
    filenames = []
    seen = set()
    for row in split_rows:
        name = (row.get("filename") or "").strip()
        if not name or name in seen:
            continue  # pragma: no cover – dedup in main()
        seen.add(name)
        filenames.append(name)
        if args.clip_limit is not None and len(filenames) >= args.clip_limit:
            break

    downloaded, skipped = download_clip_subset(s3, args.bucket, filenames, clips_root, mvp=args.mvp)
    print(f"[eval] clips downloaded={downloaded}, skipped={skipped}, requested={len(filenames)}")

    filtered_split = eval_dir / f"{args.split}.filtered.csv"
    kept, dropped = _write_filtered_split(
        src_csv=src_split,
        dst_csv=filtered_split,
        clips_root=clips_root,
        limit=args.clip_limit,
        verify_readable=not args.no_verify_readable,
    )
    print(f"[eval] filtered split kept={kept}, dropped={dropped}")
    if kept == 0:
        raise RuntimeError("No evaluable rows after filtering.")  # pragma: no cover

    gloss_dict = None
    if args.gloss_dict_csv:
        gloss_dict = _build_gloss_dict_from_csv(Path(args.gloss_dict_csv).resolve())
        print(f"[eval] gloss_dict source={args.gloss_dict_csv} num_classes={len(gloss_dict)}")

    ds = ASLCitizenI3DDataset(
        video_root=clips_root,
        split_csv=filtered_split,
        transforms=transforms.Compose([CenterCrop(224)]),
        gloss_dict=gloss_dict,
        total_frames=64,
        require_existing=True,
    )
    if len(ds) == 0:
        raise RuntimeError("Dataset resolved to 0 samples.")  # pragma: no cover

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(len(ds.gloss_dict))
    model = model.to(device)

    if args.checkpoint_s3_key:
        ckpt_path = eval_dir / Path(args.checkpoint_s3_key).name
        s3.download_file(args.bucket, args.checkpoint_s3_key, str(ckpt_path))
        ckpt_source = f"s3://{args.bucket}/{args.checkpoint_s3_key}"
    else:
        ckpt_path = Path(args.checkpoint_local).resolve()
        ckpt_source = str(ckpt_path)
    print(f"[eval] loading checkpoint from {ckpt_source}")

    state = torch.load(str(ckpt_path), map_location="cpu")
    model_state = model.state_dict()
    compatible = {}
    skipped: list[tuple[str, tuple[int, ...] | None, tuple[int, ...] | None]] = []
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            compatible[k] = v
        else:
            ckpt_shape = tuple(v.shape) if hasattr(v, "shape") else None
            model_shape = tuple(model_state[k].shape) if k in model_state else None
            skipped.append((k, ckpt_shape, model_shape))
    model.load_state_dict(compatible, strict=False)
    print(f"[eval] loaded_keys={len(compatible)} skipped_keys={len(skipped)}")
    if skipped:
        print("[eval] skipped checkpoint keys:")
        for name, ckpt_shape, model_shape in skipped:
            if model_shape is None:
                print(f"[eval]   - {name}: missing in model (ckpt_shape={ckpt_shape})")  # pragma: no cover
            else:
                print(
                    f"[eval]   - {name}: shape mismatch "
                    f"(ckpt_shape={ckpt_shape}, model_shape={model_shape})"
                )

    metrics = evaluate(model, loader, device, topk=topk)
    metrics["plan_id"] = plan_id
    metrics["split"] = args.split
    metrics["bucket"] = args.bucket
    metrics["checkpoint_source"] = ckpt_source
    metrics["num_classes"] = len(ds.gloss_dict)
    metrics["topk"] = topk

    out_path = Path(args.output_json).resolve() if args.output_json else (eval_dir / f"{args.split}_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[eval] done")
    print(f"[eval] metrics_file={out_path}")
    print(f"[eval] top1={metrics['top1_acc']:.4f}")
    for k in topk:
        print(f"[eval] top{k}={metrics[f'top{k}_acc']:.4f}")
    print(f"[eval] mrr={metrics['mrr']:.4f} dcg={metrics['dcg']:.4f}")


if __name__ == "__main__":
    main()

