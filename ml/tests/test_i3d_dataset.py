"""Tests for i3d_msft/dataset.py — I3D dataset loader and frame utilities."""

import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from i3d_msft.dataset import (
    load_rgb_frames_from_video,
    video_to_tensor,
    ASLCitizenI3DDataset,
)


# ── load_rgb_frames_from_video ──────────────────────────────────────────

def _make_video(tmp_path, name="clip.mp4", n_frames=30, h=240, w=320):
    """Create a real video file using OpenCV."""
    import cv2

    path = tmp_path / name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), fill_value=(i * 5) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_load_rgb_frames_basic(tmp_path):
    path = _make_video(tmp_path, n_frames=30, h=240, w=320)
    frames = load_rgb_frames_from_video(str(path), max_frames=64)
    assert frames.dtype == np.float32
    assert frames.ndim == 4  # (T, H, W, C)
    assert frames.shape[0] <= 64
    assert frames.shape[3] == 3
    # Normalized to [-1, 1]
    assert frames.min() >= -1.0
    assert frames.max() <= 1.0


def test_load_rgb_frames_zero_total_frames(tmp_path):
    """Empty/corrupt video returns empty array."""
    path = tmp_path / "empty.mp4"
    path.write_bytes(b"")
    frames = load_rgb_frames_from_video(str(path), max_frames=64)
    assert frames.shape[0] == 0


def test_load_rgb_frames_truncated_video(tmp_path):
    """Video that fails mid-read still returns partial frames."""
    import cv2
    path = tmp_path / "trunc.mp4"
    # Write a very short video then corrupt the tail
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))
    for _ in range(3):
        w.write(np.zeros((240, 320, 3), dtype=np.uint8))
    w.release()
    # Truncate file to simulate corruption
    data = path.read_bytes()
    path.write_bytes(data[:len(data) // 2])
    frames = load_rgb_frames_from_video(str(path), max_frames=64)
    # Either empty or partial is acceptable
    assert frames.ndim == 4


def test_load_rgb_frames_frameskip_2(tmp_path):
    """Video with >= 96 frames triggers frameskip=2."""
    path = _make_video(tmp_path, n_frames=100, h=240, w=320)
    frames = load_rgb_frames_from_video(str(path), max_frames=64)
    assert frames.shape[0] <= 64


def test_load_rgb_frames_frameskip_3(tmp_path):
    """Video with >= 160 frames triggers frameskip=3."""
    path = _make_video(tmp_path, n_frames=170, h=240, w=320)
    frames = load_rgb_frames_from_video(str(path), max_frames=64)
    assert frames.shape[0] <= 64


def test_load_rgb_frames_small_video_upscale(tmp_path):
    """Video with min(h,w) < 226 triggers upscale."""
    path = _make_video(tmp_path, n_frames=10, h=200, w=200)
    frames = load_rgb_frames_from_video(str(path), max_frames=64)
    assert frames.shape[0] > 0
    assert min(frames.shape[1], frames.shape[2]) >= 226


def test_load_rgb_frames_large_video_downscale(tmp_path):
    """Video with h > 256 or w > 256 triggers downscale."""
    path = _make_video(tmp_path, n_frames=10, h=480, w=640)
    frames = load_rgb_frames_from_video(str(path), max_frames=64)
    assert frames.shape[0] > 0
    assert max(frames.shape[1], frames.shape[2]) <= 256


# ── video_to_tensor ────────────────────────────────────────────────────

def test_video_to_tensor():
    arr = np.random.rand(16, 224, 224, 3).astype(np.float32)
    t = video_to_tensor(arr)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (3, 16, 224, 224)


def test_video_to_tensor_single_frame():
    arr = np.random.rand(1, 100, 100, 3).astype(np.float32)
    t = video_to_tensor(arr)
    assert t.shape == (3, 1, 100, 100)


# ── ASLCitizenI3DDataset ───────────────────────────────────────────────

def _make_csv(tmp_path, rows, name="split.csv"):
    path = tmp_path / name
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user", "filename", "gloss"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def test_dataset_init_builds_gloss_dict(tmp_path):
    csv_path = _make_csv(tmp_path, [
        {"user": "u1", "filename": "a.mp4", "gloss": "hello"},
        {"user": "u2", "filename": "b.mp4", "gloss": "bye"},
        {"user": "u3", "filename": "c.mp4", "gloss": "hello"},
    ])
    # Create video files
    for name in ["a.mp4", "b.mp4", "c.mp4"]:
        _make_video(tmp_path, name=name, n_frames=10, h=256, w=256)

    ds = ASLCitizenI3DDataset(
        video_root=tmp_path,
        split_csv=csv_path,
        transforms=lambda x: x,
        total_frames=16,
    )
    assert "bye" in ds.gloss_dict
    assert "hello" in ds.gloss_dict
    assert len(ds) == 3


def test_dataset_custom_gloss_dict(tmp_path):
    csv_path = _make_csv(tmp_path, [
        {"user": "u1", "filename": "a.mp4", "gloss": "hello"},
    ])
    _make_video(tmp_path, name="a.mp4", n_frames=10, h=256, w=256)
    custom = {"hello": 5}
    ds = ASLCitizenI3DDataset(
        video_root=tmp_path,
        split_csv=csv_path,
        transforms=lambda x: x,
        gloss_dict=custom,
        total_frames=16,
    )
    assert ds.gloss_dict == custom
    assert ds.labels[0] == 5


def test_dataset_filters_missing_files(tmp_path):
    csv_path = _make_csv(tmp_path, [
        {"user": "u1", "filename": "exists.mp4", "gloss": "hi"},
        {"user": "u2", "filename": "missing.mp4", "gloss": "bye"},
    ])
    _make_video(tmp_path, name="exists.mp4", n_frames=10, h=256, w=256)
    ds = ASLCitizenI3DDataset(
        video_root=tmp_path,
        split_csv=csv_path,
        transforms=lambda x: x,
        total_frames=16,
        require_existing=True,
    )
    assert len(ds) == 1


def test_dataset_filters_empty_files(tmp_path):
    csv_path = _make_csv(tmp_path, [
        {"user": "u1", "filename": "empty.mp4", "gloss": "hi"},
    ])
    (tmp_path / "empty.mp4").write_bytes(b"")
    ds = ASLCitizenI3DDataset(
        video_root=tmp_path,
        split_csv=csv_path,
        transforms=lambda x: x,
        total_frames=16,
        require_existing=True,
    )
    assert len(ds) == 0


def test_dataset_skips_empty_gloss(tmp_path):
    csv_path = _make_csv(tmp_path, [
        {"user": "u1", "filename": "a.mp4", "gloss": ""},
    ])
    _make_video(tmp_path, name="a.mp4", n_frames=5, h=256, w=256)
    ds = ASLCitizenI3DDataset(
        video_root=tmp_path,
        split_csv=csv_path,
        transforms=lambda x: x,
        total_frames=16,
    )
    assert len(ds) == 0


def test_dataset_skips_unknown_gloss_with_dict(tmp_path):
    csv_path = _make_csv(tmp_path, [
        {"user": "u1", "filename": "a.mp4", "gloss": "unknown"},
    ])
    _make_video(tmp_path, name="a.mp4", n_frames=5, h=256, w=256)
    ds = ASLCitizenI3DDataset(
        video_root=tmp_path,
        split_csv=csv_path,
        transforms=lambda x: x,
        gloss_dict={"hello": 0},
        total_frames=16,
    )
    assert len(ds) == 0


def test_dataset_no_require_existing(tmp_path):
    csv_path = _make_csv(tmp_path, [
        {"user": "u1", "filename": "missing.mp4", "gloss": "hi"},
    ])
    ds = ASLCitizenI3DDataset(
        video_root=tmp_path,
        split_csv=csv_path,
        transforms=lambda x: x,
        total_frames=16,
        require_existing=False,
    )
    assert len(ds) == 1


# ── pad ────────────────────────────────────────────────────────────────

def test_pad_short_clip(tmp_path):
    csv_path = _make_csv(tmp_path, [{"user": "u", "filename": "a.mp4", "gloss": "hi"}])
    _make_video(tmp_path, name="a.mp4", n_frames=5, h=256, w=256)
    ds = ASLCitizenI3DDataset(
        video_root=tmp_path, split_csv=csv_path,
        transforms=lambda x: x, total_frames=16,
    )
    short = np.random.rand(5, 256, 256, 3).astype(np.float32)
    padded = ds.pad(short, 16)
    assert padded.shape[0] == 16


def test_pad_long_clip_no_change(tmp_path):
    csv_path = _make_csv(tmp_path, [{"user": "u", "filename": "a.mp4", "gloss": "hi"}])
    _make_video(tmp_path, name="a.mp4", n_frames=5, h=256, w=256)
    ds = ASLCitizenI3DDataset(
        video_root=tmp_path, split_csv=csv_path,
        transforms=lambda x: x, total_frames=16,
    )
    long = np.random.rand(20, 256, 256, 3).astype(np.float32)
    result = ds.pad(long, 16)
    assert result.shape[0] == 20  # pad doesn't truncate, just returns as-is


def test_pad_empty_clip(tmp_path):
    csv_path = _make_csv(tmp_path, [{"user": "u", "filename": "a.mp4", "gloss": "hi"}])
    _make_video(tmp_path, name="a.mp4", n_frames=5, h=256, w=256)
    ds = ASLCitizenI3DDataset(
        video_root=tmp_path, split_csv=csv_path,
        transforms=lambda x: x, total_frames=16,
    )
    empty = np.zeros((0, 256, 256, 3), dtype=np.float32)
    result = ds.pad(empty, 16)
    assert result.shape == (16, 256, 256, 3)


# ── __getitem__ ────────────────────────────────────────────────────────

def test_getitem_returns_tensor_and_label(tmp_path):
    csv_path = _make_csv(tmp_path, [
        {"user": "u1", "filename": "clip.mp4", "gloss": "hello"},
    ])
    _make_video(tmp_path, name="clip.mp4", n_frames=20, h=256, w=256)
    ds = ASLCitizenI3DDataset(
        video_root=tmp_path,
        split_csv=csv_path,
        transforms=lambda x: x,
        total_frames=16,
    )
    tensor, label = ds[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[0] == 3  # C dimension
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long
