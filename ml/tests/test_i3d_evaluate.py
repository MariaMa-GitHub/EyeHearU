"""Tests for i3d_msft/evaluate.py — I3D evaluation pipeline."""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from i3d_msft.evaluate import (
    get_device,
    _read_split_rows,
    _build_gloss_dict_from_csv,
    _is_readable_video,
    _write_filtered_split,
    _topk_hits,
    _compute_mrr_and_dcg,
    evaluate,
    build_parser,
)


# ── get_device ─────────────────────────────────────────────────────────

def test_get_device_cpu():
    assert get_device("cpu") == torch.device("cpu")


def test_get_device_auto_cpu():
    with patch("i3d_msft.evaluate.torch") as mt:
        mt.cuda.is_available.return_value = False
        mt.backends = MagicMock()
        mt.backends.mps.is_available.return_value = False
        mt.device = torch.device
        assert get_device("auto") == torch.device("cpu")


def test_get_device_auto_cuda():
    with patch("i3d_msft.evaluate.torch") as mt:
        mt.cuda.is_available.return_value = True
        mt.device = torch.device
        assert get_device("auto") == torch.device("cuda")


def test_get_device_auto_mps():
    with patch("i3d_msft.evaluate.torch") as mt:
        mt.cuda.is_available.return_value = False
        mt.backends = MagicMock()
        mt.backends.mps.is_available.return_value = True
        mt.device = torch.device
        assert get_device("auto") == torch.device("mps")


# ── _read_split_rows ──────────────────────────────────────────────────

def test_read_split_rows(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("user,filename,gloss\na,f.mp4,hi\n")
    rows = _read_split_rows(p)
    assert len(rows) == 1


# ── _build_gloss_dict_from_csv ─────────────────────────────────────────

def test_build_gloss_dict(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("user,filename,gloss\na,f1.mp4,Hello\nb,f2.mp4,bye\nc,f3.mp4,Hello\n")
    d = _build_gloss_dict_from_csv(p)
    assert "bye" in d and "hello" in d
    assert d["bye"] == 0  # alphabetically first


# ── _is_readable_video ─────────────────────────────────────────────────

def test_is_readable_missing(tmp_path):
    assert _is_readable_video(tmp_path / "nope") is False


def test_is_readable_empty(tmp_path):
    p = tmp_path / "e.mp4"
    p.write_bytes(b"")
    assert _is_readable_video(p) is False


def test_is_readable_valid(tmp_path):
    import cv2
    path = tmp_path / "good.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))
    w.write(np.zeros((240, 320, 3), dtype=np.uint8))
    w.release()
    assert _is_readable_video(path) is True


# ── _write_filtered_split ──────────────────────────────────────────────

def _make_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["user", "filename", "gloss"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_write_filtered_split_basic(tmp_path):
    src = tmp_path / "src.csv"
    _make_csv(src, [
        {"user": "a", "filename": "a.mp4", "gloss": "hi"},
        {"user": "b", "filename": "b.mp4", "gloss": "bye"},
    ])
    clips_root = tmp_path / "clips"
    clips_root.mkdir()
    (clips_root / "a.mp4").write_text("data")
    dst = tmp_path / "out.csv"
    kept, dropped = _write_filtered_split(src, dst, clips_root, limit=None, verify_readable=False)
    assert kept == 1  # only a.mp4 exists
    assert dropped == 1


def test_write_filtered_split_with_limit(tmp_path):
    src = tmp_path / "src.csv"
    _make_csv(src, [
        {"user": "a", "filename": "a.mp4", "gloss": "hi"},
        {"user": "b", "filename": "b.mp4", "gloss": "bye"},
    ])
    clips_root = tmp_path / "clips"
    clips_root.mkdir()
    for name in ["a.mp4", "b.mp4"]:
        (clips_root / name).write_text("data")
    dst = tmp_path / "out.csv"
    kept, dropped = _write_filtered_split(src, dst, clips_root, limit=1, verify_readable=False)
    assert kept == 1


def test_write_filtered_split_empty_gloss(tmp_path):
    src = tmp_path / "src.csv"
    _make_csv(src, [{"user": "a", "filename": "a.mp4", "gloss": ""}])
    clips_root = tmp_path / "clips"
    clips_root.mkdir()
    (clips_root / "a.mp4").write_text("data")
    dst = tmp_path / "out.csv"
    kept, dropped = _write_filtered_split(src, dst, clips_root, limit=None, verify_readable=False)
    assert kept == 0
    assert dropped == 1


def test_write_filtered_split_verify_readable(tmp_path):
    src = tmp_path / "src.csv"
    _make_csv(src, [{"user": "a", "filename": "a.mp4", "gloss": "hi"}])
    clips_root = tmp_path / "clips"
    clips_root.mkdir()
    (clips_root / "a.mp4").write_text("not a real video")
    dst = tmp_path / "out.csv"
    kept, dropped = _write_filtered_split(src, dst, clips_root, limit=None, verify_readable=True)
    assert kept == 0  # not a real video
    assert dropped == 1


# ── _topk_hits ─────────────────────────────────────────────────────────

def test_topk_hits_perfect():
    logits = torch.tensor([[10.0, 1.0, 0.0], [0.0, 10.0, 0.0]])
    labels = torch.tensor([0, 1])
    assert _topk_hits(logits, labels, k=1) == 2


def test_topk_hits_top3():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    labels = torch.tensor([1])  # 2nd value
    assert _topk_hits(logits, labels, k=3) == 1


def test_topk_hits_miss():
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    labels = torch.tensor([2])
    assert _topk_hits(logits, labels, k=1) == 0


# ── _compute_mrr_and_dcg ──────────────────────────────────────────────

def test_mrr_and_dcg_perfect():
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    labels = torch.tensor([0, 1])
    mrr, dcg = _compute_mrr_and_dcg(logits, labels)
    assert mrr == 1.0
    assert dcg > 0


def test_mrr_and_dcg_worst():
    logits = torch.tensor([[0.0, 10.0]])  # pred=1
    labels = torch.tensor([0])  # correct=0
    mrr, dcg = _compute_mrr_and_dcg(logits, labels)
    assert mrr == 0.5  # rank=2, 1/2
    assert dcg > 0


# ── evaluate ───────────────────────────────────────────────────────────

class _MockI3D(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.fc = nn.Linear(3, nc)
        self.nc = nc

    def forward(self, x, pretrained=False):
        b = x.shape[0]
        out = self.fc(torch.randn(b, 3))
        return out.unsqueeze(2).expand(b, self.nc, 4)


def _make_loader(n=8, nc=10):
    clips = torch.randn(n, 3, 16, 56, 56)
    labels = torch.randint(0, nc, (n,))
    return DataLoader(TensorDataset(clips, labels), batch_size=4)


def test_evaluate_metrics():
    model = _MockI3D(10)
    loader = _make_loader(8, 10)
    metrics = evaluate(model, loader, torch.device("cpu"), topk=[1, 5])
    assert "num_samples" in metrics
    assert metrics["num_samples"] == 8
    assert "top1_acc" in metrics
    assert "top5_acc" in metrics
    assert "loss" in metrics
    assert "mrr" in metrics
    assert "dcg" in metrics
    assert "top_confusions" in metrics


def test_evaluate_empty_loader():
    model = _MockI3D(10)
    loader = DataLoader(
        TensorDataset(torch.empty(0, 3, 16, 56, 56), torch.empty(0, dtype=torch.long)),
        batch_size=4,
    )
    metrics = evaluate(model, loader, torch.device("cpu"), topk=[1])
    assert metrics["num_samples"] == 0


# ── build_parser ───────────────────────────────────────────────────────

def test_build_parser_defaults():
    p = build_parser()
    args = p.parse_args(["--bucket", "b", "--checkpoint-local", "c.pt"])
    assert args.bucket == "b"
    assert args.split == "test"
    assert args.topk == "1,5,15,20"
    assert args.batch_size == 6


def test_build_parser_all_options():
    p = build_parser()
    args = p.parse_args([
        "--bucket", "b", "--region", "us-east-1", "--plan-id", "p1",
        "--split", "val", "--checkpoint-s3-key", "models/m.pt",
        "--topk", "1,3,5", "--batch-size", "8", "--device", "cpu",
        "--clip-limit", "50", "--no-verify-readable",
        "--output-json", "out.json", "--gloss-dict-csv", "train.csv",
    ])
    assert args.split == "val"
    assert args.checkpoint_s3_key == "models/m.pt"


# ── main (integration) ────────────────────────────────────────────────

def test_main_missing_checkpoint():
    with patch("sys.argv", ["evaluate", "--bucket", "b"]):
        with pytest.raises(ValueError, match="checkpoint"):
            from i3d_msft.evaluate import main
            main()


def test_main_invalid_topk():
    from i3d_msft.evaluate import main
    with patch("sys.argv", ["evaluate", "--bucket", "b", "--checkpoint-local", "c.pt",
                            "--topk", ""]):
        with pytest.raises(ValueError, match="topk"):
            main()


def test_main_integration(tmp_path):
    """Full integration test for evaluate.main() with local checkpoint."""
    import cv2
    from i3d_msft.evaluate import main as eval_main
    from i3d_msft.pytorch_i3d import InceptionI3d

    # Create a model and save checkpoint
    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(2)
    ckpt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Create split CSVs and clips
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True)
    clips_root = tmp_path / "clips"
    clips_root.mkdir()

    for name in ["a.mp4", "b.mp4"]:
        p = clips_root / name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(str(p), fourcc, 30.0, (256, 256))
        for _ in range(10):
            w.write(np.zeros((256, 256, 3), dtype=np.uint8))
        w.release()

    test_csv = splits_dir / "test.csv"
    _make_csv(test_csv, [
        {"user": "u1", "filename": "a.mp4", "gloss": "hi"},
        {"user": "u2", "filename": "b.mp4", "gloss": "bye"},
    ])
    train_csv = splits_dir / "train.csv"
    _make_csv(train_csv, [
        {"user": "u1", "filename": "a.mp4", "gloss": "hi"},
        {"user": "u2", "filename": "b.mp4", "gloss": "bye"},
    ])

    mock_s3 = MagicMock()

    def fake_download_splits(s3, bucket, plan_id, local_dir, mvp=True):
        import shutil
        local_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for s in ("train", "val", "test"):
            src = splits_dir / "test.csv"
            dst = local_dir / f"{s}.csv"
            shutil.copy(src, dst)
            out[s] = dst
        return out

    def fake_download_clips(s3, bucket, filenames, root, mvp=True):
        import shutil
        root.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            src = clips_root / name
            dst = root / name
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)
        return len(filenames), 0

    workdir = tmp_path / "workdir"
    with patch("i3d_msft.evaluate.get_s3_client", return_value=mock_s3), \
         patch("i3d_msft.evaluate.get_active_plan_id", return_value="p1"), \
         patch("i3d_msft.evaluate.download_splits", side_effect=fake_download_splits), \
         patch("i3d_msft.evaluate.download_clip_subset", side_effect=fake_download_clips), \
         patch("sys.argv", ["evaluate", "--bucket", "b",
                            "--checkpoint-local", str(ckpt_path),
                            "--plan-id", "p1",
                            "--split", "test",
                            "--topk", "1,2",
                            "--batch-size", "2",
                            "--device", "cpu",
                            "--no-verify-readable",
                            "--gloss-dict-csv", str(train_csv),
                            "--local-root", str(workdir)]):
        eval_main()

    metrics_path = workdir / "eval" / "p1" / "test_metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert "top1_acc" in metrics
    assert "mrr" in metrics
    assert metrics["num_samples"] == 2


def test_main_with_s3_checkpoint(tmp_path):
    """Test main() with --checkpoint-s3-key path."""
    import cv2
    from i3d_msft.evaluate import main as eval_main
    from i3d_msft.pytorch_i3d import InceptionI3d

    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(2)
    ckpt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt_path)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True)
    clips_root = tmp_path / "clips"
    clips_root.mkdir()

    for name in ["a.mp4"]:
        p = clips_root / name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(str(p), fourcc, 30.0, (256, 256))
        for _ in range(10):
            w.write(np.zeros((256, 256, 3), dtype=np.uint8))
        w.release()

    test_csv = splits_dir / "test.csv"
    _make_csv(test_csv, [{"user": "u1", "filename": "a.mp4", "gloss": "hi"}])

    mock_s3 = MagicMock()

    def fake_download_splits(s3, bucket, plan_id, local_dir, mvp=True):
        import shutil
        local_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for s in ("train", "val", "test"):
            dst = local_dir / f"{s}.csv"
            shutil.copy(test_csv, dst)
            out[s] = dst
        return out

    def fake_download_clips(s3, bucket, filenames, root, mvp=True):
        import shutil
        root.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            src = clips_root / name
            dst = root / name
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)
        return len(filenames), 0

    def fake_s3_download_file(bucket, key, dest):
        import shutil
        shutil.copy(ckpt_path, dest)

    mock_s3.download_file = MagicMock(side_effect=fake_s3_download_file)

    workdir = tmp_path / "workdir"
    with patch("i3d_msft.evaluate.get_s3_client", return_value=mock_s3), \
         patch("i3d_msft.evaluate.get_active_plan_id", return_value="p1"), \
         patch("i3d_msft.evaluate.download_splits", side_effect=fake_download_splits), \
         patch("i3d_msft.evaluate.download_clip_subset", side_effect=fake_download_clips), \
         patch("sys.argv", ["evaluate", "--bucket", "b",
                            "--checkpoint-s3-key", "models/best.pt",
                            "--plan-id", "p1",
                            "--split", "test",
                            "--device", "cpu",
                            "--no-verify-readable",
                            "--local-root", str(workdir)]):
        eval_main()

    metrics_path = workdir / "eval" / "p1" / "test_metrics.json"
    assert metrics_path.exists()


def test_main_with_clip_limit(tmp_path):
    """Test --clip-limit in main()."""
    import cv2
    from i3d_msft.evaluate import main as eval_main
    from i3d_msft.pytorch_i3d import InceptionI3d

    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(2)
    ckpt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt_path)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True)
    clips_root = tmp_path / "clips"
    clips_root.mkdir()

    for name in ["a.mp4", "b.mp4", "c.mp4"]:
        p = clips_root / name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(str(p), fourcc, 30.0, (256, 256))
        for _ in range(10):
            w.write(np.zeros((256, 256, 3), dtype=np.uint8))
        w.release()

    test_csv = splits_dir / "test.csv"
    _make_csv(test_csv, [
        {"user": "u1", "filename": "a.mp4", "gloss": "hi"},
        {"user": "u2", "filename": "b.mp4", "gloss": "bye"},
        {"user": "u3", "filename": "c.mp4", "gloss": "hi"},
    ])

    mock_s3 = MagicMock()

    def fake_download_splits(s3, bucket, plan_id, local_dir, mvp=True):
        import shutil
        local_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for s in ("train", "val", "test"):
            dst = local_dir / f"{s}.csv"
            shutil.copy(test_csv, dst)
            out[s] = dst
        return out

    def fake_download_clips(s3, bucket, filenames, root, mvp=True):
        import shutil
        root.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            src = clips_root / name
            dst = root / name
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)
        return len(filenames), 0

    workdir = tmp_path / "workdir"
    with patch("i3d_msft.evaluate.get_s3_client", return_value=mock_s3), \
         patch("i3d_msft.evaluate.get_active_plan_id", return_value="p1"), \
         patch("i3d_msft.evaluate.download_splits", side_effect=fake_download_splits), \
         patch("i3d_msft.evaluate.download_clip_subset", side_effect=fake_download_clips), \
         patch("sys.argv", ["evaluate", "--bucket", "b",
                            "--checkpoint-local", str(ckpt_path),
                            "--plan-id", "p1",
                            "--clip-limit", "2",
                            "--device", "cpu",
                            "--no-verify-readable",
                            "--local-root", str(workdir)]):
        eval_main()

    metrics_path = workdir / "eval" / "p1" / "test_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    assert metrics["num_samples"] <= 2


def test_main_custom_output_json(tmp_path):
    """Test --output-json flag."""
    import cv2
    from i3d_msft.evaluate import main as eval_main
    from i3d_msft.pytorch_i3d import InceptionI3d

    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(1)
    ckpt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt_path)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True)
    clips_root = tmp_path / "clips"
    clips_root.mkdir()

    p = clips_root / "a.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(p), fourcc, 30.0, (256, 256))
    for _ in range(10):
        w.write(np.zeros((256, 256, 3), dtype=np.uint8))
    w.release()

    test_csv = splits_dir / "test.csv"
    _make_csv(test_csv, [{"user": "u1", "filename": "a.mp4", "gloss": "hi"}])

    mock_s3 = MagicMock()

    def fake_download_splits(s3, bucket, plan_id, local_dir, mvp=True):
        import shutil
        local_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for s in ("train", "val", "test"):
            dst = local_dir / f"{s}.csv"
            shutil.copy(test_csv, dst)
            out[s] = dst
        return out

    def fake_download_clips(s3, bucket, filenames, root, mvp=True):
        import shutil
        root.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            src = clips_root / name
            dst = root / name
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)
        return len(filenames), 0

    custom_output = tmp_path / "custom_output" / "metrics.json"
    workdir = tmp_path / "workdir"
    with patch("i3d_msft.evaluate.get_s3_client", return_value=mock_s3), \
         patch("i3d_msft.evaluate.get_active_plan_id", return_value="p1"), \
         patch("i3d_msft.evaluate.download_splits", side_effect=fake_download_splits), \
         patch("i3d_msft.evaluate.download_clip_subset", side_effect=fake_download_clips), \
         patch("sys.argv", ["evaluate", "--bucket", "b",
                            "--checkpoint-local", str(ckpt_path),
                            "--plan-id", "p1",
                            "--device", "cpu",
                            "--no-verify-readable",
                            "--output-json", str(custom_output),
                            "--local-root", str(workdir)]):
        eval_main()

    assert custom_output.exists()
