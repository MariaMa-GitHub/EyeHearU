"""
Training script for the ASL video classifier.

Usage:
    cd ml/
    python -m training.train

Expects the data pipeline to have been run first:
    data/processed/clips/{train,val}/{gloss}/*.mp4
    data/processed/label_map.json
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from models.classifier import ASLVideoClassifier
from training.dataset import ASLVideoDataset

SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * clips.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += clips.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)

        logits = model(clips)
        loss = criterion(logits, labels)

        total_loss += loss.item() * clips.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += clips.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def _worker_init_fn(worker_id: int):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)


def main():
    set_seed(SEED)
    cfg = Config()

    # Resolve num_classes from the label map produced by the data pipeline
    label_map_path = Path(cfg.data.processed_data_dir) / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            num_classes = len(json.load(f))
        print(f"Loaded label map: {num_classes} classes")
    else:
        num_classes = cfg.data.num_classes
        print(f"label_map.json not found — using default num_classes={num_classes}")

    device = torch.device(cfg.train.device)
    print(f"Device: {device}")

    # --- Data ---
    train_ds = ASLVideoDataset(
        data_dir=cfg.data.processed_data_dir,
        split="train",
        num_frames=cfg.data.num_frames,
        augment=True,
    )
    val_ds = ASLVideoDataset(
        data_dir=cfg.data.processed_data_dir,
        split="val",
        num_frames=cfg.data.num_frames,
        augment=False,
    )

    if len(train_ds) == 0:
        print("ERROR: No training data found. Run the data pipeline first.")
        print(f"  Expected clips at: {cfg.data.processed_data_dir}/clips/train/")
        sys.exit(1)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )

    # --- Model ---
    model = ASLVideoClassifier(
        num_classes=num_classes,
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        head_dropout=cfg.model.head_dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # --- Optimizer & Scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    # --- Training loop ---
    best_val_acc = 0.0
    patience_counter = 0
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()

        if epoch <= cfg.model.backbone_freeze_epochs:
            model.freeze_backbone()
        else:
            model.unfreeze_backbone()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{cfg.train.epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            print(f"  -> New best (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1

        if epoch % cfg.train.save_every_n_epochs == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")

        if patience_counter >= cfg.train.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nDone. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
