"""
Evaluation script for the ASL video classifier.

Generates:
  - Per-class accuracy
  - Top-k accuracy
  - Most confused pairs
  - Saves results to evaluation_results.json

Usage:
    cd ml/
    python -m evaluation.evaluate --checkpoint checkpoints/best_model.pt
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from models.classifier import ASLVideoClassifier
from training.dataset import ASLVideoDataset


def compute_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5):
    _, topk_preds = logits.topk(k, dim=-1)
    correct = topk_preds.eq(labels.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


@torch.no_grad()
def evaluate_model(model, loader, device, label_map_inv):
    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []

    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        logits = model(clips)

        all_logits.append(logits.cpu())
        all_preds.append(logits.argmax(dim=-1).cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    overall_acc = (all_preds == all_labels).float().mean().item()
    top5_acc = compute_topk_accuracy(all_logits, all_labels, k=5)

    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, label in zip(all_preds, all_labels):
        name = label_map_inv.get(label.item(), f"class_{label.item()}")
        per_class[name]["total"] += 1
        if pred == label:
            per_class[name]["correct"] += 1

    per_class_acc = {
        name: stats["correct"] / max(stats["total"], 1)
        for name, stats in per_class.items()
    }

    confusion_pairs = defaultdict(int)
    for pred, label in zip(all_preds, all_labels):
        if pred != label:
            pred_name = label_map_inv.get(pred.item(), f"class_{pred.item()}")
            label_name = label_map_inv.get(label.item(), f"class_{label.item()}")
            confusion_pairs[(label_name, pred_name)] += 1

    top_confusions = sorted(confusion_pairs.items(), key=lambda x: -x[1])[:20]

    return {
        "overall_accuracy": overall_acc,
        "top5_accuracy": top5_acc,
        "per_class_accuracy": per_class_acc,
        "top_confusions": [
            {"true": pair[0], "predicted": pair[1], "count": count}
            for pair, count in top_confusions
        ],
        "total_samples": len(all_labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(cfg.train.device)

    label_map_path = Path(cfg.data.processed_data_dir) / "label_map.json"
    with open(label_map_path) as f:
        label_map = json.load(f)
    label_map_inv = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    dataset = ASLVideoDataset(
        data_dir=cfg.data.processed_data_dir,
        split=args.split,
        num_frames=cfg.data.num_frames,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False)

    model = ASLVideoClassifier(
        num_classes=num_classes,
        backbone=cfg.model.backbone,
        pretrained=False,
        head_dropout=cfg.model.head_dropout,
    ).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    results = evaluate_model(model, loader, device, label_map_inv)

    print(f"\n{'=' * 50}")
    print(f"Evaluation — {args.split} set")
    print(f"{'=' * 50}")
    print(f"Overall Accuracy:  {results['overall_accuracy']:.4f}")
    print(f"Top-5 Accuracy:    {results['top5_accuracy']:.4f}")
    print(f"Total Samples:     {results['total_samples']}")

    print(f"\nPer-class accuracy:")
    for name, acc in sorted(results["per_class_accuracy"].items()):
        print(f"  {name:20s}: {acc:.4f}")

    if results["top_confusions"]:
        print(f"\nMost confused pairs:")
        for item in results["top_confusions"][:10]:
            print(f"  {item['true']:15s} -> {item['predicted']:15s}  ({item['count']})")

    output_path = Path("evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
