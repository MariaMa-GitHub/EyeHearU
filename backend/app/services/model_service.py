"""
Model loading and inference service.

Loads the trained ASL video classifier and label map, runs inference on
preprocessed video clips (C, T, H, W), returns top-k predictions with gloss labels.
"""

import json
from pathlib import Path

# torch imported only inside load_model/predict so backend can start without PyTorch

# Repo root so we can import ml.models
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_REPO_ROOT))

# Lazy import to avoid loading torch/ml when not needed
def _get_classifier():
    from ml.models.classifier import ASLVideoClassifier
    return ASLVideoClassifier


def _find_label_map(model_path: str) -> Path:
    """Label map is next to the model or in the same directory."""
    p = Path(model_path).resolve()
    candidates = [p.parent / "label_map.json", p.with_suffix(".label_map.json")]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"label_map.json not found next to {model_path}")


def load_model(model_path: str, device: str = "cpu"):
    """
    Load the trained ASLVideoClassifier and label map.

    Args:
        model_path: Path to best_model.pt (or other .pt checkpoint)
        device: "cpu", "cuda", or "mps"

    Returns:
        (model, index_to_gloss: dict[int, str])
    """
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    label_map_path = _find_label_map(str(model_path))
    with open(label_map_path, encoding="utf-8") as f:
        label_map = json.load(f)
    # gloss -> int; build int -> gloss for inference
    index_to_gloss = {int(v): k for k, v in label_map.items()}
    num_classes = len(label_map)

    import torch
    ASLVideoClassifier = _get_classifier()
    model = ASLVideoClassifier(
        num_classes=num_classes,
        backbone="r3d_18",
        pretrained=False,
        head_dropout=0.0,
    )
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    return model, index_to_gloss


def predict(model, index_to_gloss: dict, video_tensor, top_k: int = 5, device: str = "cpu"):
    """
    Run inference on a single video clip tensor.

    Args:
        model: Loaded ASLVideoClassifier
        index_to_gloss: Map from class index to gloss string
        video_tensor: (1, C, T, H, W) float32, normalized (e.g. from preprocess_video)
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
        logits = model(video_tensor)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=min(top_k, logits.size(-1)), dim=-1)

    results = []
    for prob, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
        sign = index_to_gloss.get(int(idx), f"class_{idx}")
        results.append({"sign": sign, "confidence": round(prob, 4)})
    return results
