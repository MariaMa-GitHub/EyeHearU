"""
Evaluation script for the ASL video classifier.

Generates:
  - Per-class and overall accuracy (top-1 and top-5)
  - Precision, recall, F1 (macro and per-class)
  - Full confusion matrix (saved as JSON and PNG)
  - Most confused pairs
  - Inference latency statistics (p50 / p95 / p99)
  - Saves all results to evaluation_results.json

Usage:
    cd ml/
    python -m evaluation.evaluate --checkpoint checkpoints/best_model.pt
"""

import sys
import json
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

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


def compute_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5):
    _, topk_preds = logits.topk(k, dim=-1)
    correct = topk_preds.eq(labels.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


def compute_precision_recall_f1(all_preds: torch.Tensor, all_labels: torch.Tensor, num_classes: int):
    """Compute per-class and macro precision, recall, and F1."""
    per_class = {}
    precisions, recalls, f1s = [], [], []

    for c in range(num_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum().item()
        fp = ((all_preds == c) & (all_labels != c)).sum().item()
        fn = ((all_preds != c) & (all_labels == c)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[c] = {"precision": precision, "recall": recall, "f1": f1}
        if (tp + fn) > 0:
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return per_class, macro_precision, macro_recall, macro_f1


def build_confusion_matrix(all_preds: torch.Tensor, all_labels: torch.Tensor, num_classes: int):
    """Build a num_classes x num_classes confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(all_preds.numpy(), all_labels.numpy()):
        cm[label, pred] += 1
    return cm


@torch.no_grad()
def evaluate_model(model, loader, device, label_map_inv, num_classes: int):
    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []
    latencies_ms = []

    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)

        t0 = time.perf_counter()
        logits = model(clips)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        latencies_ms.append(elapsed_ms / clips.size(0))  # per-sample

        all_logits.append(logits.cpu())
        all_preds.append(logits.argmax(dim=-1).cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    overall_acc = (all_preds == all_labels).float().mean().item()
    top5_acc = compute_topk_accuracy(all_logits, all_labels, k=5)

    per_class_metrics, macro_precision, macro_recall, macro_f1 = compute_precision_recall_f1(
        all_preds, all_labels, num_classes
    )

    per_class_acc = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, label in zip(all_preds, all_labels):
        name = label_map_inv.get(label.item(), f"class_{label.item()}")
        per_class_acc[name]["total"] += 1
        if pred == label:
            per_class_acc[name]["correct"] += 1

    per_class_accuracy = {
        name: stats["correct"] / max(stats["total"], 1)
        for name, stats in per_class_acc.items()
    }

    per_class_detail = {}
    for c, metrics in per_class_metrics.items():
        name = label_map_inv.get(c, f"class_{c}")
        per_class_detail[name] = {
            "accuracy": per_class_accuracy.get(name, 0.0),
            **metrics,
        }

    confusion_pairs = defaultdict(int)
    for pred, label in zip(all_preds, all_labels):
        if pred != label:
            pred_name = label_map_inv.get(pred.item(), f"class_{pred.item()}")
            label_name = label_map_inv.get(label.item(), f"class_{label.item()}")
            confusion_pairs[(label_name, pred_name)] += 1

    top_confusions = sorted(confusion_pairs.items(), key=lambda x: -x[1])[:20]

    cm = build_confusion_matrix(all_preds, all_labels, num_classes)

    latencies = np.array(latencies_ms)
    latency_stats = {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }

    return {
        "overall_accuracy": overall_acc,
        "top5_accuracy": top5_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_detail": per_class_detail,
        "per_class_accuracy": per_class_accuracy,
        "top_confusions": [
            {"true": pair[0], "predicted": pair[1], "count": count}
            for pair, count in top_confusions
        ],
        "confusion_matrix": cm.tolist(),
        "inference_latency": latency_stats,
        "total_samples": len(all_labels),
    }


def save_confusion_matrix_plot(cm: np.ndarray, class_names: list[str], output_path: Path):
    """Save a confusion matrix heatmap as a PNG image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed — skipping confusion matrix plot")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.4),
                                     max(8, len(class_names) * 0.35)))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", fmt="d", annot=len(class_names) <= 30, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)

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

    results = evaluate_model(model, loader, device, label_map_inv, num_classes)

    print(f"\n{'=' * 60}")
    print(f"Evaluation — {args.split} set  (seed={args.seed})")
    print(f"{'=' * 60}")
    print(f"Overall Accuracy:  {results['overall_accuracy']:.4f}")
    print(f"Top-5 Accuracy:    {results['top5_accuracy']:.4f}")
    print(f"Macro Precision:   {results['macro_precision']:.4f}")
    print(f"Macro Recall:      {results['macro_recall']:.4f}")
    print(f"Macro F1:          {results['macro_f1']:.4f}")
    print(f"Total Samples:     {results['total_samples']}")

    lat = results["inference_latency"]
    print(f"\nInference Latency (per sample):")
    print(f"  Mean:  {lat['mean_ms']:.1f} ms")
    print(f"  p50:   {lat['p50_ms']:.1f} ms")
    print(f"  p95:   {lat['p95_ms']:.1f} ms")
    print(f"  p99:   {lat['p99_ms']:.1f} ms")

    print(f"\nPer-class detail:")
    for name in sorted(results["per_class_detail"]):
        d = results["per_class_detail"][name]
        print(
            f"  {name:20s}: acc={d['accuracy']:.3f}  "
            f"P={d['precision']:.3f}  R={d['recall']:.3f}  F1={d['f1']:.3f}"
        )

    if results["top_confusions"]:
        print(f"\nMost confused pairs:")
        for item in results["top_confusions"][:10]:
            print(f"  {item['true']:15s} -> {item['predicted']:15s}  ({item['count']})")

    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / "evaluation_results.json"
    serialisable = {k: v for k, v in results.items() if k != "confusion_matrix"}
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved to {json_path}")

    cm = np.array(results["confusion_matrix"])
    class_names = [label_map_inv.get(i, str(i)) for i in range(num_classes)]
    cm_json_path = output_dir / "confusion_matrix.json"
    with open(cm_json_path, "w") as f:
        json.dump({"class_names": class_names, "matrix": cm.tolist()}, f)
    print(f"Confusion matrix JSON saved to {cm_json_path}")

    save_confusion_matrix_plot(cm, class_names, output_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
