"""
Profile 5 key functions in the EyeHearU codebase using cProfile.

Profiled functions:
  1. preprocess_video         — video preprocessing pipeline (backend)
  2. predict                  — model inference (backend)
  3. i3d_msft.evaluate        — full I3D evaluation loop (ML)
  4. i3d_msft.train_one_epoch — single I3D training epoch (ML)
  5. build_gloss_dict_from_csv — label map construction (ML)

Usage:
    cd ml/
    python -m profiling.profile_functions

Outputs per-function cProfile stats (sorted by cumulative time) and a
summary table to stdout.  Optionally writes .prof files for pstats
exploration.
"""

import cProfile
import csv
import io
import json
import os
import pstats
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Ensure project roots are importable
# ---------------------------------------------------------------------------
_ML_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_ROOT = _ML_ROOT.parent / "backend"
for p in (_ML_ROOT, _BACKEND_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.services.preprocessing import preprocess_video
from app.services.model_service import predict
from i3d_msft.evaluate import evaluate as i3d_evaluate
from i3d_msft.export_label_map import build_gloss_dict_from_csv
from i3d_msft.pytorch_i3d import InceptionI3d
from i3d_msft.train import train_one_epoch

PROFILE_DIR = _ML_ROOT / "profiling" / "results"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers: synthetic data generators
# ---------------------------------------------------------------------------

def _make_synthetic_video_bytes(n_frames: int = 90, h: int = 480, w: int = 640) -> bytes:
    """Create a minimal MP4 file with solid-colour frames using OpenCV."""
    import cv2

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, 30.0, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), fill_value=(i * 3) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    with open(tmp.name, "rb") as f:
        data = f.read()
    os.unlink(tmp.name)
    return data


def _make_synthetic_csv(n_rows: int = 1000, n_classes: int = 50) -> Path:
    """Write a temporary CSV with (user, filename, gloss) rows."""
    glosses = [f"sign_{i:03d}" for i in range(n_classes)]
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    )
    writer = csv.DictWriter(tmp, fieldnames=["user", "filename", "gloss"])
    writer.writeheader()
    for i in range(n_rows):
        writer.writerow({
            "user": f"user_{i % 10}",
            "filename": f"clip_{i}.mp4",
            "gloss": glosses[i % n_classes],
        })
    tmp.close()
    return Path(tmp.name)


def _make_fake_loader(
    num_batches: int = 5,
    batch_size: int = 4,
    num_classes: int = 20,
    num_frames: int = 64,
):
    """Create a DataLoader with random tensors (B, C, T, H, W)."""
    clips = torch.randn(num_batches * batch_size, 3, num_frames, 224, 224)
    labels = torch.randint(0, num_classes, (num_batches * batch_size,))
    return DataLoader(TensorDataset(clips, labels), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Profile wrappers
# ---------------------------------------------------------------------------

def profile_preprocess_video():
    """Profile 1: preprocess_video — full video preprocessing pipeline."""
    video_bytes = _make_synthetic_video_bytes(n_frames=90, h=480, w=640)

    def run():
        for _ in range(3):
            preprocess_video(video_bytes)

    return _run_profile("preprocess_video", run)


def profile_predict():
    """Profile 2: predict — model inference with I3D-like output."""
    from i3d_msft.pytorch_i3d import InceptionI3d

    num_classes = 50
    label_map = {i: f"sign_{i}" for i in range(num_classes)}

    model = InceptionI3d(num_classes=400, in_channels=3)
    model.replace_logits(num_classes)
    model.eval()

    # I3D expects (B, 3, 64, 224, 224)
    video_tensor = torch.randn(1, 3, 64, 224, 224)

    def run():
        for _ in range(3):
            predict(model, label_map, video_tensor, top_k=5, device="cpu")

    return _run_profile("predict", run)


def profile_evaluate_model():
    """Profile 3: i3d_msft.evaluate — evaluation over a synthetic dataset."""
    num_classes = 20
    model = InceptionI3d(num_classes=400, in_channels=3)
    model.replace_logits(num_classes)
    model.eval()
    device = torch.device("cpu")

    loader = _make_fake_loader(
        num_batches=5, batch_size=4, num_classes=num_classes, num_frames=64
    )

    def run():
        i3d_evaluate(model, loader, device, topk=[1, 5])

    return _run_profile("i3d_msft.evaluate", run)


def profile_train_one_epoch():
    """Profile 4: i3d_msft.train_one_epoch — one full training pass."""
    num_classes = 20
    model = InceptionI3d(num_classes=400, in_channels=3)
    model.replace_logits(num_classes)
    device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loader = _make_fake_loader(
        num_batches=5, batch_size=4, num_classes=num_classes, num_frames=64
    )

    def run():
        train_one_epoch(model, loader, criterion, optimizer, device)

    return _run_profile("i3d_msft.train_one_epoch", run)


def profile_build_gloss_dict():
    """Profile 5: build_gloss_dict_from_csv — label map construction."""
    csv_path = _make_synthetic_csv(n_rows=5000, n_classes=200)

    def run():
        for _ in range(50):
            build_gloss_dict_from_csv(csv_path)

    result = _run_profile("build_gloss_dict_from_csv", run)
    os.unlink(csv_path)
    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_profile(name: str, func) -> dict:
    """cProfile a callable, print stats, save .prof, return summary."""
    prof = cProfile.Profile()

    wall_start = time.perf_counter()
    prof.enable()
    func()
    prof.disable()
    wall_elapsed = time.perf_counter() - wall_start

    # Save binary profile for later pstats analysis
    prof_path = PROFILE_DIR / f"{name}.prof"
    prof.dump_stats(str(prof_path))

    # Print top-20 by cumulative time
    stream = io.StringIO()
    stats = pstats.Stats(prof, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    output = stream.getvalue()

    print(f"\n{'=' * 72}")
    print(f"  PROFILE: {name}")
    print(f"  Wall time: {wall_elapsed:.3f}s")
    print(f"  Profile saved to: {prof_path}")
    print(f"{'=' * 72}")
    print(output)

    # Extract total calls and total time from stats
    # pstats key = (file, line, name), value = (cc, nc, tt, ct, callers)
    total_calls = 0
    total_tt = 0.0
    for key, value in stats.stats.items():
        cc, nc, tt, ct, callers = value
        total_calls += nc
        total_tt += tt

    return {
        "name": name,
        "wall_time_s": round(wall_elapsed, 4),
        "total_calls": total_calls,
        "total_internal_time_s": round(total_tt, 4),
        "prof_file": str(prof_path),
    }


def main():
    print("=" * 72)
    print("  EyeHearU — cProfile profiling of 5 key functions")
    print("=" * 72)

    results = []
    for profiler in [
        profile_preprocess_video,
        profile_predict,
        profile_evaluate_model,
        profile_train_one_epoch,
        profile_build_gloss_dict,
    ]:
        results.append(profiler())

    # Summary table
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"{'Function':<30} {'Wall (s)':>10} {'Calls':>10} {'Internal (s)':>14}")
    print("-" * 72)
    for r in results:
        print(f"{r['name']:<30} {r['wall_time_s']:>10.4f} {r['total_calls']:>10,} {r['total_internal_time_s']:>14.4f}")

    # Save JSON summary
    json_path = PROFILE_DIR / "profile_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON summary saved to {json_path}")
    print(f"Binary .prof files in {PROFILE_DIR}/ — explore with:")
    print(f"  python -c \"import pstats; p=pstats.Stats('{results[0]['prof_file']}'); p.sort_stats('cumulative'); p.print_stats(30)\"")


if __name__ == "__main__":
    main()
