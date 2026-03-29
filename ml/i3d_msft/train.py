"""
Train Microsoft I3D (vendored) against EyeHearU S3 dataset layout.

Examples:
  # Auto-pick ACTIVE plan, sync train+val splits and clips from S3, then train
  python -m i3d_msft.train --bucket eye-hear-u-dev-data --region ca-central-1

  # Pin a specific plan and run a short smoke training
  python -m i3d_msft.train --bucket eye-hear-u-dev-data --region ca-central-1 \
    --plan-id candidate-ac-eval-v4 --epochs 3 --clip-limit 500
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from i3d_msft.dataset import ASLCitizenI3DDataset
from i3d_msft.pytorch_i3d import InceptionI3d
from i3d_msft.s3_data import (
    download_clip_subset,
    download_splits,
    get_active_plan_id,
    get_s3_client,
)
from i3d_msft.videotransforms import CenterCrop, RandomCrop, RandomHorizontalFlip


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()

        # I3D output: (B, C, T'). Max-over-time for clip-level prediction.
        logits_t = model(clips, pretrained=False)
        clip_logits = torch.max(logits_t, dim=2)[0]

        loss = criterion(clip_logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * clips.size(0)
        preds = clip_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += clips.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        logits_t = model(clips, pretrained=False)
        clip_logits = torch.max(logits_t, dim=2)[0]

        loss = criterion(clip_logits, labels)
        running_loss += loss.item() * clips.size(0)
        preds = clip_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += clips.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train vendored Microsoft I3D with EyeHearU S3 data.")
    parser.add_argument("--bucket", required=True, help="S3 bucket holding processed data and split plans.")
    parser.add_argument("--region", default="ca-central-1")
    parser.add_argument("--plan-id", default=None, help="Split plan id. If omitted, use ACTIVE_PLAN.")
    parser.add_argument("--mvp", action="store_true", default=True, help="Use processed/mvp prefixes (default on).")
    parser.add_argument("--local-root", default="workdir/i3d_msft", help="Local workspace for splits/clips/checkpoints.")
    parser.add_argument("--clip-limit", type=int, default=None, help="Optional limit on downloaded clips for smoke runs.")
    parser.add_argument(
        "--no-verify-readable",
        action="store_true",
        help="Disable decode check during split filtering (faster, less robust).",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="Backbone LR after unfreezing. Default: lr * 0.1",
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=None,
        help="Classifier head LR. Default: lr",
    )
    parser.add_argument(
        "--head-only-epochs",
        type=int,
        default=0,
        help="Train only logits head for first N epochs, then unfreeze full backbone.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--init-checkpoint-s3-key",
        default=None,
        help="Optional S3 key for warm-start checkpoint (SFT), e.g. models/i3d/modal/.../best_model.pt",
    )
    parser.add_argument(
        "--init-strict",
        action="store_true",
        help="Load warm-start checkpoint with strict=True (default is compatible partial load).",
    )
    parser.add_argument(
        "--s3-checkpoint-prefix",
        default=None,
        help="If set, upload checkpoints to s3://<bucket>/<prefix>/ after each save.",
    )
    return parser


def _read_split_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _select_filenames_with_val_coverage(
    train_csv: Path,
    val_csv: Path,
    limit: int | None,
) -> list[str]:
    train_rows = [
        {
            "filename": (r.get("filename") or "").strip(),
            "gloss": (r.get("gloss") or "").strip().lower(),
        }
        for r in _read_split_rows(train_csv)
    ]
    val_rows = [
        {
            "filename": (r.get("filename") or "").strip(),
            "gloss": (r.get("gloss") or "").strip().lower(),
        }
        for r in _read_split_rows(val_csv)
    ]
    train_rows = [r for r in train_rows if r["filename"] and r["gloss"]]
    val_rows = [r for r in val_rows if r["filename"] and r["gloss"]]

    if limit is None:
        seen = set()
        out = []
        for n in [r["filename"] for r in train_rows + val_rows]:
            if n in seen:
                continue
            seen.add(n)
            out.append(n)
        return out

    if limit <= 0:
        return []

    # Keep some budget for val so smoke runs don't end up val=0.
    val_budget = min(max(1, limit // 4), len(val_rows))
    train_budget = max(0, limit - val_budget)

    selected_train = train_rows[:train_budget]
    selected_train_gloss = {r["gloss"] for r in selected_train}
    val_candidates = [r for r in val_rows if r["gloss"] in selected_train_gloss]
    if not val_candidates:
        # Fallback when overlap is tiny: add at least one val gloss into train budget.
        seed_gloss = val_rows[0]["gloss"] if val_rows else None
        if seed_gloss:
            extra_train = [r for r in train_rows if r["gloss"] == seed_gloss]
            if extra_train and selected_train:
                selected_train[-1] = extra_train[0]
                selected_train_gloss = {r["gloss"] for r in selected_train}
                val_candidates = [r for r in val_rows if r["gloss"] in selected_train_gloss]

    out = []
    seen = set()
    for r in val_candidates:
        if len(out) >= min(val_budget, len(val_candidates)):
            break  # pragma: no cover – budget guard
        n = r["filename"]
        if n in seen:
            continue  # pragma: no cover – dedup guard
        seen.add(n)
        out.append(n)
    for r in selected_train:
        if len(out) >= min(val_budget, len(val_candidates)) + len(selected_train):
            break  # pragma: no cover – budget guard
        n = r["filename"]
        if n in seen:
            continue  # pragma: no cover – dedup guard
        seen.add(n)
        out.append(n)
    return out


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
    allowed_filenames: set[str] | None,
    verify_readable: bool,
) -> tuple[int, int]:
    rows = _read_split_rows(src_csv)
    kept = []
    dropped = 0
    for row in rows:
        name = (row.get("filename") or "").strip()
        gloss = (row.get("gloss") or "").strip()
        if not name or not gloss:
            dropped += 1
            continue
        if allowed_filenames is not None and name not in allowed_filenames:
            dropped += 1
            continue
        clip_path = clips_root / name
        if not clip_path.exists() or clip_path.stat().st_size == 0:
            dropped += 1
            continue
        if verify_readable and not _is_readable_video(clip_path):
            dropped += 1
            continue
        kept.append(row)

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user", "filename", "gloss"])
        writer.writeheader()
        for row in kept:
            writer.writerow(
                {
                    "user": row.get("user", ""),
                    "filename": row.get("filename", ""),
                    "gloss": row.get("gloss", ""),
                }
            )
    return len(kept), dropped


def _load_compatible_checkpoint(model: torch.nn.Module, ckpt_path: Path, strict: bool):
    state = torch.load(str(ckpt_path), map_location="cpu")
    if strict:
        model.load_state_dict(state, strict=True)
        return {"loaded": len(state), "skipped": 0}

    model_state = model.state_dict()
    compatible = {}
    skipped = 0
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped += 1
    model.load_state_dict(compatible, strict=False)
    return {"loaded": len(compatible), "skipped": skipped}


def _upload_checkpoint_to_s3(s3, bucket: str, local_path: Path, s3_key: str):
    try:
        s3.upload_file(str(local_path), bucket, s3_key)
        print(f"[i3d] uploaded {local_path.name} -> s3://{bucket}/{s3_key}")
    except Exception as exc:
        print(f"[i3d] WARNING: failed to upload {local_path.name}: {exc}")


def _set_backbone_trainable(model: torch.nn.Module, trainable: bool):
    for name, param in model.named_parameters():
        if name.startswith("logits."):
            continue
        param.requires_grad = trainable


def _build_optimizer(
    model: torch.nn.Module,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
    backbone_trainable: bool,
):
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("logits."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    if backbone_trainable and backbone_params:
        groups = [
            {"params": head_params, "lr": head_lr},
            {"params": backbone_params, "lr": backbone_lr},
        ]
    else:
        groups = [{"params": head_params, "lr": head_lr}]
    return torch.optim.Adam(groups, weight_decay=weight_decay)


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"[i3d] device={device}")

    s3 = get_s3_client(args.region)
    plan_id = args.plan_id or get_active_plan_id(s3, args.bucket, mvp=args.mvp)
    print(f"[i3d] using plan_id={plan_id}")

    root = Path(args.local_root).resolve()
    splits_dir = root / "splits" / plan_id
    clips_root = root / "clips"
    ckpt_dir = root / "checkpoints" / plan_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    split_paths = download_splits(s3, args.bucket, plan_id, splits_dir, mvp=args.mvp)
    needed = _select_filenames_with_val_coverage(
        train_csv=split_paths["train"],
        val_csv=split_paths["val"],
        limit=args.clip_limit,
    )
    needed_set = set(needed) if args.clip_limit is not None else None
    downloaded, skipped = download_clip_subset(s3, args.bucket, needed, clips_root, mvp=args.mvp)
    print(f"[i3d] clips downloaded={downloaded}, skipped={skipped}, requested={len(needed)}")

    filtered_splits_dir = root / "filtered_splits" / plan_id
    train_filtered = filtered_splits_dir / "train.csv"
    val_filtered = filtered_splits_dir / "val.csv"
    train_kept, train_dropped = _write_filtered_split(
        src_csv=split_paths["train"],
        dst_csv=train_filtered,
        clips_root=clips_root,
        allowed_filenames=needed_set,
        verify_readable=not args.no_verify_readable,
    )
    val_kept, val_dropped = _write_filtered_split(
        src_csv=split_paths["val"],
        dst_csv=val_filtered,
        clips_root=clips_root,
        allowed_filenames=needed_set,
        verify_readable=not args.no_verify_readable,
    )
    print(
        f"[i3d] filtered splits | train kept={train_kept} dropped={train_dropped} | "
        f"val kept={val_kept} dropped={val_dropped}"
    )

    train_transforms = transforms.Compose([RandomCrop(224), RandomHorizontalFlip()])
    val_transforms = transforms.Compose([CenterCrop(224)])

    train_ds = ASLCitizenI3DDataset(
        video_root=clips_root,
        split_csv=train_filtered,
        transforms=train_transforms,
        gloss_dict=None,
        total_frames=64,
        require_existing=True,
    )
    val_ds = ASLCitizenI3DDataset(
        video_root=clips_root,
        split_csv=val_filtered,
        transforms=val_transforms,
        gloss_dict=train_ds.gloss_dict,
        total_frames=64,
        require_existing=True,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"Dataset empty after sync. train={len(train_ds)}, val={len(val_ds)}. "
            "Check split CSVs and clip sync."
        )

    print(
        f"[i3d] train_samples={len(train_ds)}, val_samples={len(val_ds)}, "
        f"num_classes={len(train_ds.gloss_dict)}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(len(train_ds.gloss_dict))
    model = model.to(device)

    if args.init_checkpoint_s3_key:  # pragma: no cover – requires real S3 checkpoint
        init_dir = root / "init_ckpt"
        init_dir.mkdir(parents=True, exist_ok=True)
        init_path = init_dir / Path(args.init_checkpoint_s3_key).name
        s3.download_file(args.bucket, args.init_checkpoint_s3_key, str(init_path))
        stats = _load_compatible_checkpoint(model, init_path, strict=args.init_strict)
        print(
            f"[i3d] warm start from s3://{args.bucket}/{args.init_checkpoint_s3_key} | "
            f"loaded={stats['loaded']} skipped={stats['skipped']} strict={args.init_strict}"
        )

    if args.head_only_epochs < 0:
        raise ValueError("--head-only-epochs must be >= 0")  # pragma: no cover – CLI guard
    head_lr = args.head_lr if args.head_lr is not None else args.lr
    backbone_lr = args.backbone_lr if args.backbone_lr is not None else (args.lr * 0.1)

    backbone_trainable = args.head_only_epochs == 0
    _set_backbone_trainable(model, trainable=backbone_trainable)
    optimizer = _build_optimizer(
        model=model,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        weight_decay=args.weight_decay,
        backbone_trainable=backbone_trainable,
    )
    phase = "full-finetune" if backbone_trainable else "head-only"
    print(
        f"[i3d] optimization phase={phase} | head_lr={head_lr:g} "
        f"backbone_lr={backbone_lr:g} head_only_epochs={args.head_only_epochs}"
    )
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        if epoch == args.head_only_epochs + 1 and args.head_only_epochs > 0:  # pragma: no cover – requires multi-epoch run with full model
            _set_backbone_trainable(model, trainable=True)
            optimizer = _build_optimizer(
                model=model,
                head_lr=head_lr,
                backbone_lr=backbone_lr,
                weight_decay=args.weight_decay,
                backbone_trainable=True,
            )
            print(
                f"[i3d] optimization phase=full-finetune (unfrozen at epoch {epoch}) | "
                f"head_lr={head_lr:g} backbone_lr={backbone_lr:g}"
            )
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val:  # pragma: no cover – depends on random-init accuracy
            best_val = val_acc
            best_path = ckpt_dir / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            print(f"[i3d] new best -> {best_path} (val_acc={best_val:.4f})")
            if args.s3_checkpoint_prefix:
                _upload_checkpoint_to_s3(
                    s3, args.bucket, best_path,
                    f"{args.s3_checkpoint_prefix}/best_model.pt",
                )

        if epoch % 5 == 0:  # pragma: no cover – requires ≥5 epoch run with full model
            epoch_path = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save(model.state_dict(), epoch_path)
            if args.s3_checkpoint_prefix:
                _upload_checkpoint_to_s3(
                    s3, args.bucket, epoch_path,
                    f"{args.s3_checkpoint_prefix}/epoch_{epoch}.pt",
                )

    print(f"[i3d] done. best_val_acc={best_val:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()

