"""Tests for i3d_msft/train.py — I3D training pipeline."""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from i3d_msft.train import (
    set_seed,
    get_device,
    train_one_epoch,
    evaluate,
    build_arg_parser,
    _read_split_rows,
    _select_filenames_with_val_coverage,
    _is_readable_video,
    _write_filtered_split,
    _load_compatible_checkpoint,
    _upload_checkpoint_to_s3,
    _set_backbone_trainable,
    _build_optimizer,
)


# ── set_seed ───────────────────────────────────────────────────────────

def test_set_seed_deterministic():
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


# ── get_device ─────────────────────────────────────────────────────────

def test_get_device_cpu():
    assert get_device("cpu") == torch.device("cpu")


def test_get_device_auto_fallback():
    with patch("i3d_msft.train.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.device = torch.device
        result = get_device("auto")
        assert result == torch.device("cpu")


def test_get_device_auto_cuda():
    with patch("i3d_msft.train.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device = torch.device
        result = get_device("auto")
        assert result == torch.device("cuda")


def test_get_device_auto_mps():
    with patch("i3d_msft.train.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.device = torch.device
        result = get_device("auto")
        assert result == torch.device("mps")


# ── train_one_epoch ────────────────────────────────────────────────────

class _FakeI3DModel(nn.Module):
    """Mimics I3D output shape (B, C, T) for max-over-time pooling."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(3, num_classes)
        self.nc = num_classes

    def forward(self, x, pretrained=False):
        b = x.shape[0]
        # Produce (B, num_classes, 4) to simulate temporal output
        out = self.linear(torch.randn(b, 3))
        return out.unsqueeze(2).expand(b, self.nc, 4)


def _make_i3d_loader(n=8, nc=10, T=16):
    clips = torch.randn(n, 3, T, 56, 56)
    labels = torch.randint(0, nc, (n,))
    return DataLoader(TensorDataset(clips, labels), batch_size=4)


def test_train_one_epoch_basic():
    model = _FakeI3DModel(10)
    loader = _make_i3d_loader(8, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loss, acc = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
    assert isinstance(loss, float) and loss >= 0
    assert 0 <= acc <= 1


def test_train_one_epoch_empty_loader():
    model = _FakeI3DModel(10)
    loader = DataLoader(TensorDataset(torch.empty(0, 3, 16, 56, 56), torch.empty(0, dtype=torch.long)), batch_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loss, acc = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
    assert loss == 0.0
    assert acc == 0.0


# ── evaluate ───────────────────────────────────────────────────────────

def test_evaluate_basic():
    model = _FakeI3DModel(10)
    loader = _make_i3d_loader(8, 10)
    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, loader, criterion, torch.device("cpu"))
    assert isinstance(loss, float) and loss >= 0
    assert 0 <= acc <= 1


def test_evaluate_empty_loader():
    model = _FakeI3DModel(10)
    loader = DataLoader(TensorDataset(torch.empty(0, 3, 16, 56, 56), torch.empty(0, dtype=torch.long)), batch_size=4)
    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, loader, criterion, torch.device("cpu"))
    assert loss == 0.0 and acc == 0.0


# ── build_arg_parser ───────────────────────────────────────────────────

def test_build_arg_parser_defaults():
    parser = build_arg_parser()
    args = parser.parse_args(["--bucket", "test-bucket"])
    assert args.bucket == "test-bucket"
    assert args.epochs == 20
    assert args.batch_size == 6
    assert args.device == "auto"
    assert args.plan_id is None
    assert args.clip_limit is None


def test_build_arg_parser_all_options():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--bucket", "b", "--region", "us-east-1", "--plan-id", "p1",
        "--epochs", "5", "--batch-size", "8", "--lr", "0.01",
        "--backbone-lr", "0.001", "--head-lr", "0.1",
        "--head-only-epochs", "3", "--weight-decay", "1e-5",
        "--seed", "123", "--device", "cpu", "--clip-limit", "100",
        "--init-checkpoint-s3-key", "models/best.pt", "--init-strict",
        "--s3-checkpoint-prefix", "prefix", "--no-verify-readable",
    ])
    assert args.epochs == 5
    assert args.head_only_epochs == 3
    assert args.init_strict is True


# ── _read_split_rows ──────────────────────────────────────────────────

def test_read_split_rows(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("user,filename,gloss\na,f1.mp4,hi\n")
    rows = _read_split_rows(p)
    assert len(rows) == 1 and rows[0]["gloss"] == "hi"


# ── _select_filenames_with_val_coverage ────────────────────────────────

def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["user", "filename", "gloss"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_select_filenames_no_limit(tmp_path):
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    _write_csv(train, [
        {"user": "a", "filename": "t1.mp4", "gloss": "hi"},
        {"user": "b", "filename": "t2.mp4", "gloss": "bye"},
    ])
    _write_csv(val, [
        {"user": "c", "filename": "v1.mp4", "gloss": "hi"},
    ])
    result = _select_filenames_with_val_coverage(train, val, limit=None)
    assert set(result) == {"t1.mp4", "t2.mp4", "v1.mp4"}


def test_select_filenames_with_limit(tmp_path):
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    _write_csv(train, [
        {"user": "a", "filename": f"t{i}.mp4", "gloss": f"g{i}"} for i in range(20)
    ])
    _write_csv(val, [
        {"user": "b", "filename": f"v{i}.mp4", "gloss": f"g{i}"} for i in range(10)
    ])
    result = _select_filenames_with_val_coverage(train, val, limit=10)
    assert len(result) <= 10


def test_select_filenames_limit_zero(tmp_path):
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    _write_csv(train, [{"user": "a", "filename": "t1.mp4", "gloss": "hi"}])
    _write_csv(val, [{"user": "b", "filename": "v1.mp4", "gloss": "hi"}])
    result = _select_filenames_with_val_coverage(train, val, limit=0)
    assert result == []


def test_select_filenames_no_overlap_fallback(tmp_path):
    """When train budget misses val glosses, fallback seeds from val[0] gloss."""
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    # Train has many glosses; val has a gloss that exists in train but
    # train_budget clips won't include it. The fallback replaces the last
    # selected_train entry with one that matches val's gloss.
    _write_csv(train, [
        {"user": "a", "filename": "t1.mp4", "gloss": "alpha"},
        {"user": "b", "filename": "t2.mp4", "gloss": "beta"},
        {"user": "c", "filename": "t3.mp4", "gloss": "gamma"},  # matches val
    ])
    _write_csv(val, [
        {"user": "d", "filename": "v1.mp4", "gloss": "gamma"},
    ])
    # limit=2: val_budget=1, train_budget=1 → selected_train = [t1 (alpha)]
    # No overlap with val (gamma). Fallback: seed_gloss="gamma",
    # extra_train=[t3], selected_train[-1] = t3 → now overlap exists.
    result = _select_filenames_with_val_coverage(train, val, limit=2)
    assert "v1.mp4" in result  # val clip included


def test_select_filenames_duplicate_val_filenames(tmp_path):
    """Duplicate filenames in val are deduplicated (hits line 224)."""
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    _write_csv(train, [
        {"user": "a", "filename": "t1.mp4", "gloss": "hi"},
    ])
    _write_csv(val, [
        {"user": "b", "filename": "v1.mp4", "gloss": "hi"},
        {"user": "c", "filename": "v1.mp4", "gloss": "hi"},  # duplicate
    ])
    result = _select_filenames_with_val_coverage(train, val, limit=10)
    assert result.count("v1.mp4") == 1  # deduped


def test_select_filenames_train_budget_break(tmp_path):
    """Train budget limit triggers break (hits line 229)."""
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    # Need more train rows than train_budget so the for-loop hits the break
    # limit=4: val_budget=1, train_budget=3. selected_train has 3 entries.
    # val_candidates has 1 match (v1). After val: out=[v1], len=1.
    # Train loop: add t0 (len=2), t1 (len=3), check t2: len(3) >= 1+3=4? No.
    # Add t2 (len=4). Check next: 4 >= 4? Yes → break. But need 4+ entries.
    # So we need 5 train rows, limit=4: val_budget=1, train_budget=3.
    # selected_train = first 3. val_candidates=[v1 if hi in selected gloss].
    # After val: [v1]. After train: [v1, t0, t1, t2]. Then len=4 >= 1+3=4 → break on next.
    # But we only have 3 selected_train entries! Need selected_train > budget.
    # Actually the condition is: len(out) >= min(val_budget, len(val_candidates)) + len(selected_train)
    # So it's val_count + train_count, not val_budget + train_budget.
    # With val_candidates=1 entry, selected_train=3: threshold = 1 + 3 = 4.
    # After adding v1 + t0 + t1 + t2 = 4 entries, len=4 >= 4 → break on NEXT iter.
    # But there is no next iter since we only had 3 selected_train.
    # To hit break, we need threshold < len(selected_train) entries in train loop.
    # This happens when val_candidates is empty (0 val matched), so threshold=0+3=3,
    # and selected_train has 5 entries → breaks after 3.
    _write_csv(train, [
        {"user": "a", "filename": f"t{i}.mp4", "gloss": "hi"} for i in range(10)
    ])
    _write_csv(val, [])  # empty val
    result = _select_filenames_with_val_coverage(train, val, limit=5)
    # val_budget=min(1, 0)=0(?), train_budget=max(0, 5-0)=5
    # selected_train=10 rows[:5]=5. val_candidates=[].
    # threshold = 0 + 5 = 5. After 5 train entries: break.
    assert len(result) <= 5


def test_select_filenames_duplicate_across_val_train(tmp_path):
    """Filenames appearing in both val and train are deduplicated (hits line 232)."""
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    _write_csv(train, [
        {"user": "a", "filename": "shared.mp4", "gloss": "hi"},
        {"user": "b", "filename": "t2.mp4", "gloss": "bye"},
    ])
    _write_csv(val, [
        {"user": "c", "filename": "shared.mp4", "gloss": "hi"},  # same as train
    ])
    result = _select_filenames_with_val_coverage(train, val, limit=10)
    assert result.count("shared.mp4") == 1  # deduped


# ── _is_readable_video ─────────────────────────────────────────────────

def test_is_readable_video_missing(tmp_path):
    assert _is_readable_video(tmp_path / "nope.mp4") is False


def test_is_readable_video_empty(tmp_path):
    p = tmp_path / "empty.mp4"
    p.write_bytes(b"")
    assert _is_readable_video(p) is False


def test_is_readable_video_valid(tmp_path):
    import cv2
    path = tmp_path / "good.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))
    w.write(np.zeros((240, 320, 3), dtype=np.uint8))
    w.release()
    assert _is_readable_video(path) is True


# ── _write_filtered_split ──────────────────────────────────────────────

def test_write_filtered_split_basic(tmp_path):
    src = tmp_path / "src.csv"
    _write_csv(src, [
        {"user": "a", "filename": "clip1.mp4", "gloss": "hi"},
        {"user": "b", "filename": "clip2.mp4", "gloss": "bye"},
    ])
    clips_root = tmp_path / "clips"
    clips_root.mkdir()
    # Only clip1 exists
    import cv2
    path = clips_root / "clip1.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))
    w.write(np.zeros((240, 320, 3), dtype=np.uint8))
    w.release()

    dst = tmp_path / "dst.csv"
    kept, dropped = _write_filtered_split(src, dst, clips_root, allowed_filenames=None, verify_readable=True)
    assert kept == 1
    assert dropped == 1
    assert dst.exists()


def test_write_filtered_split_allowed_filenames(tmp_path):
    src = tmp_path / "src.csv"
    _write_csv(src, [
        {"user": "a", "filename": "clip1.mp4", "gloss": "hi"},
        {"user": "b", "filename": "clip2.mp4", "gloss": "bye"},
    ])
    clips_root = tmp_path / "clips"
    clips_root.mkdir()
    for name in ["clip1.mp4", "clip2.mp4"]:
        (clips_root / name).write_text("data")
    dst = tmp_path / "dst.csv"
    kept, dropped = _write_filtered_split(
        src, dst, clips_root, allowed_filenames={"clip1.mp4"}, verify_readable=False,
    )
    assert kept == 1
    assert dropped == 1


def test_write_filtered_split_no_verify(tmp_path):
    src = tmp_path / "src.csv"
    _write_csv(src, [{"user": "a", "filename": "c.mp4", "gloss": "hi"}])
    clips_root = tmp_path / "clips"
    clips_root.mkdir()
    (clips_root / "c.mp4").write_text("notavideo")
    dst = tmp_path / "dst.csv"
    kept, dropped = _write_filtered_split(src, dst, clips_root, allowed_filenames=None, verify_readable=False)
    assert kept == 1


def test_write_filtered_split_verify_drops_unreadable(tmp_path):
    """verify_readable=True drops files that exist but aren't valid video."""
    src = tmp_path / "src.csv"
    _write_csv(src, [{"user": "a", "filename": "bad.mp4", "gloss": "hi"}])
    clips_root = tmp_path / "clips"
    clips_root.mkdir()
    (clips_root / "bad.mp4").write_text("this is not a video file")
    dst = tmp_path / "dst.csv"
    kept, dropped = _write_filtered_split(src, dst, clips_root, allowed_filenames=None, verify_readable=True)
    assert kept == 0
    assert dropped == 1


def test_write_filtered_split_empty_gloss(tmp_path):
    src = tmp_path / "src.csv"
    _write_csv(src, [{"user": "a", "filename": "c.mp4", "gloss": ""}])
    clips_root = tmp_path / "clips"
    clips_root.mkdir()
    (clips_root / "c.mp4").write_text("data")
    dst = tmp_path / "dst.csv"
    kept, dropped = _write_filtered_split(src, dst, clips_root, allowed_filenames=None, verify_readable=False)
    assert kept == 0
    assert dropped == 1


# ── _load_compatible_checkpoint ────────────────────────────────────────

def test_load_compatible_checkpoint_strict(tmp_path):
    model = nn.Linear(10, 5)
    ckpt = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), ckpt)
    new_model = nn.Linear(10, 5)
    stats = _load_compatible_checkpoint(new_model, ckpt, strict=True)
    assert stats["loaded"] == 2  # weight + bias
    assert stats["skipped"] == 0


def test_load_compatible_checkpoint_partial(tmp_path):
    # Save a model with different output size (shape mismatch on weight+bias)
    model = nn.Linear(10, 5)
    ckpt = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), ckpt)
    new_model = nn.Linear(10, 3)  # different output
    stats = _load_compatible_checkpoint(new_model, ckpt, strict=False)
    assert stats["skipped"] == 2  # both keys have shape mismatch


def test_load_compatible_checkpoint_mixed(tmp_path):
    """strict=False with some matching and some mismatched keys."""
    # Build a checkpoint with weight (10,5) and bias (5,)
    model = nn.Linear(10, 5)
    ckpt = tmp_path / "ckpt.pt"
    state = model.state_dict()
    # Add an extra key with a shape that won't match anything
    state["extra.weight"] = torch.randn(7, 7)
    torch.save(state, ckpt)
    new_model = nn.Linear(10, 5)  # same shape — weight+bias match
    stats = _load_compatible_checkpoint(new_model, ckpt, strict=False)
    assert stats["loaded"] == 2   # weight + bias match
    assert stats["skipped"] == 1  # extra.weight not in model


# ── _upload_checkpoint_to_s3 ──────────────────────────────────────────

def test_upload_checkpoint_success(tmp_path):
    s3 = MagicMock()
    ckpt = tmp_path / "model.pt"
    ckpt.write_text("data")
    _upload_checkpoint_to_s3(s3, "bucket", ckpt, "models/model.pt")
    s3.upload_file.assert_called_once_with(str(ckpt), "bucket", "models/model.pt")


def test_upload_checkpoint_failure(tmp_path, capsys):
    s3 = MagicMock()
    s3.upload_file.side_effect = Exception("network error")
    ckpt = tmp_path / "model.pt"
    ckpt.write_text("data")
    _upload_checkpoint_to_s3(s3, "bucket", ckpt, "key")
    assert "WARNING" in capsys.readouterr().out


# ── _set_backbone_trainable ───────────────────────────────────────────

def test_set_backbone_trainable_freeze():
    from i3d_msft.pytorch_i3d import InceptionI3d
    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(10)
    _set_backbone_trainable(model, trainable=False)
    for name, param in model.named_parameters():
        if name.startswith("logits."):
            assert param.requires_grad
        else:
            assert not param.requires_grad


def test_set_backbone_trainable_unfreeze():
    from i3d_msft.pytorch_i3d import InceptionI3d
    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(10)
    _set_backbone_trainable(model, trainable=False)
    _set_backbone_trainable(model, trainable=True)
    for param in model.parameters():
        assert param.requires_grad


# ── _build_optimizer ──────────────────────────────────────────────────

def test_build_optimizer_head_only():
    from i3d_msft.pytorch_i3d import InceptionI3d
    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(10)
    _set_backbone_trainable(model, trainable=False)
    opt = _build_optimizer(model, head_lr=0.01, backbone_lr=0.001, weight_decay=1e-6, backbone_trainable=False)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == 0.01


def test_build_optimizer_full_finetune():
    from i3d_msft.pytorch_i3d import InceptionI3d
    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(10)
    _set_backbone_trainable(model, trainable=True)
    opt = _build_optimizer(model, head_lr=0.01, backbone_lr=0.001, weight_decay=1e-6, backbone_trainable=True)
    assert len(opt.param_groups) == 2
    lrs = {g["lr"] for g in opt.param_groups}
    assert 0.01 in lrs
    assert 0.001 in lrs


# ── main (integration with mocks) ─────────────────────────────────────

def test_main_smoke(tmp_path):
    """Smoke test main() with fully mocked S3 and minimal data."""
    # Prepare local split CSVs
    import cv2
    clips_root = tmp_path / "clips"
    clips_root.mkdir()
    for name in ["a.mp4", "b.mp4", "c.mp4"]:
        p = clips_root / name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(p), fourcc, 30.0, (256, 256))
        for _ in range(10):
            writer.write(np.zeros((256, 256, 3), dtype=np.uint8))
        writer.release()

    splits_dir = tmp_path / "splits" / "plan1"
    splits_dir.mkdir(parents=True)
    for split, rows in [
        ("train", [{"user": "u1", "filename": "a.mp4", "gloss": "hi"},
                    {"user": "u2", "filename": "b.mp4", "gloss": "bye"}]),
        ("val", [{"user": "u3", "filename": "c.mp4", "gloss": "hi"}]),
        ("test", [{"user": "u4", "filename": "c.mp4", "gloss": "hi"}]),
    ]:
        _write_csv(splits_dir / f"{split}.csv", rows)

    mock_s3 = MagicMock()

    def fake_download_splits(s3, bucket, plan_id, local_dir, mvp=True):
        # Copy our pre-made CSVs
        import shutil
        local_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for s in ("train", "val", "test"):
            src = splits_dir / f"{s}.csv"
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

    with patch("i3d_msft.train.get_s3_client", return_value=mock_s3), \
         patch("i3d_msft.train.get_active_plan_id", return_value="plan1"), \
         patch("i3d_msft.train.download_splits", side_effect=fake_download_splits), \
         patch("i3d_msft.train.download_clip_subset", side_effect=fake_download_clips), \
         patch("sys.argv", ["train", "--bucket", "b", "--plan-id", "plan1",
                            "--epochs", "1", "--batch-size", "2", "--device", "cpu",
                            "--no-verify-readable",
                            "--local-root", str(tmp_path / "workdir")]):
        from i3d_msft.train import main
        main()

    # Verify training ran (checkpoint saved only if val_acc > 0)
    ckpt_dir = tmp_path / "workdir" / "checkpoints" / "plan1"
    assert ckpt_dir.exists()


def test_main_empty_dataset_raises(tmp_path):
    """Empty dataset after sync raises RuntimeError."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True)
    for s in ("train", "val", "test"):
        (splits_dir / f"{s}.csv").write_text("user,filename,gloss\na,missing.mp4,hi\n")

    mock_s3 = MagicMock()

    def fake_download_splits(s3, bucket, plan_id, local_dir, mvp=True):
        import shutil
        local_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for s in ("train", "val", "test"):
            src = splits_dir / f"{s}.csv"
            dst = local_dir / f"{s}.csv"
            shutil.copy(src, dst)
            out[s] = dst
        return out

    def fake_download_clips(s3, bucket, filenames, root, mvp=True):
        root.mkdir(parents=True, exist_ok=True)
        return 0, 0

    with patch("i3d_msft.train.get_s3_client", return_value=mock_s3), \
         patch("i3d_msft.train.get_active_plan_id", return_value="p1"), \
         patch("i3d_msft.train.download_splits", side_effect=fake_download_splits), \
         patch("i3d_msft.train.download_clip_subset", side_effect=fake_download_clips), \
         patch("sys.argv", ["train", "--bucket", "b", "--plan-id", "p1",
                            "--epochs", "1", "--batch-size", "2", "--device", "cpu",
                            "--no-verify-readable",
                            "--local-root", str(tmp_path / "workdir")]):
        with pytest.raises(RuntimeError, match="empty"):
            from i3d_msft.train import main
            main()
