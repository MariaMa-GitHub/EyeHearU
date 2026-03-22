"""
Model loading and inference service for the I3D ASL video classifier.

Downloads the trained InceptionI3d checkpoint from S3 (if not cached locally),
loads the model with the label map from the repo, and runs top-k inference
on preprocessed video tensors.
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _download_model_from_s3(settings, dest: Path):
    """Download best_model.pt from S3 if not already cached locally."""
    import boto3
    dest.parent.mkdir(parents=True, exist_ok=True)
    bucket = settings.aws_s3_bucket
    key = settings.aws_s3_model_key
    print(f"[model] downloading s3://{bucket}/{key} -> {dest}")
    s3 = boto3.client("s3", region_name=settings.aws_s3_region)
    s3.download_file(bucket, key, str(dest))
    print(f"[model] download complete ({dest.stat().st_size / 1e6:.1f} MB)")


def _load_label_map(path: Path) -> dict[int, str]:
    """
    Load label map JSON. Supports two formats:
      1. Teammate's format: {"gloss_to_index": {...}, "index_to_gloss": [...], ...}
      2. Simple format: {"gloss": index, ...}
    Returns: {int_index: gloss_string}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "gloss_to_index" in data:
        g2i = data["gloss_to_index"]
        return {int(v): k for k, v in g2i.items()}

    if "index_to_gloss" in data and isinstance(data["index_to_gloss"], list):
        return {i: g for i, g in enumerate(data["index_to_gloss"])}

    return {int(v): k for k, v in data.items()}


def load_model(settings):
    """
    Load the trained InceptionI3d model and label map.

    Returns:
        (model, index_to_gloss: dict[int, str])
    """
    import torch
    from ml.i3d_msft.pytorch_i3d import InceptionI3d

    model_path = Path(settings.model_path).resolve()
    label_map_path = Path(settings.label_map_path).resolve()

    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found: {label_map_path}")

    index_to_gloss = _load_label_map(label_map_path)
    num_classes = len(index_to_gloss)
    print(f"[model] label map: {num_classes} classes from {label_map_path.name}")

    if not model_path.exists():
        _download_model_from_s3(settings, model_path)

    model = InceptionI3d(num_classes=400, in_channels=3)
    model.replace_logits(num_classes)

    device = settings.model_device
    state = torch.load(str(model_path), map_location=device, weights_only=False)

    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append(k)
    model.load_state_dict(compatible, strict=False)
    if skipped:
        print(f"[model] loaded {len(compatible)} keys, skipped {len(skipped)}: {skipped[:5]}")
    else:
        print(f"[model] loaded all {len(compatible)} keys")

    model.to(device)
    model.eval()
    return model, index_to_gloss


def predict(model, index_to_gloss: dict, video_tensor, top_k: int = 5, device: str = "cpu"):
    """
    Run inference on a single video clip tensor.

    Args:
        model: Loaded InceptionI3d
        index_to_gloss: Map from class index to gloss string
        video_tensor: (1, C, T, H, W) float32
        top_k: Number of top predictions
        device: Device the model is on

    Returns:
        List of {"sign": str, "confidence": float}
    """
    import torch

    if video_tensor.dim() == 4:
        video_tensor = video_tensor.unsqueeze(0)
    video_tensor = video_tensor.to(device)

    with torch.no_grad():
        logits_t = model(video_tensor, pretrained=False)
        # I3D output: (B, num_classes, T') — max-pool over temporal dim
        if logits_t.dim() == 3:
            clip_logits = torch.max(logits_t, dim=2)[0]
        else:
            clip_logits = logits_t
        probs = torch.softmax(clip_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=min(top_k, clip_logits.size(-1)), dim=-1)

    results = []
    for prob, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
        sign = index_to_gloss.get(int(idx), f"class_{idx}")
        results.append({"sign": sign, "confidence": round(prob, 4)})
    return results
