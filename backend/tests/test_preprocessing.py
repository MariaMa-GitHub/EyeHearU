"""Tests for the video preprocessing pipeline."""

import numpy as np
import pytest

from app.services.preprocessing import (
    _pad_frames,
    _center_crop,
    _ensure_both_sides_at_least,
    TOTAL_FRAMES,
    CROP_SIZE,
    MAX_SIDE,
)


def test_pad_frames_short_video():
    imgs = np.random.rand(10, MAX_SIDE, MAX_SIDE, 3).astype(np.float32)
    padded = _pad_frames(imgs, TOTAL_FRAMES)
    assert padded.shape == (TOTAL_FRAMES, MAX_SIDE, MAX_SIDE, 3)
    np.testing.assert_array_equal(padded[:10], imgs)
    np.testing.assert_array_equal(padded[10], imgs[-1])


def test_pad_frames_exact():
    imgs = np.random.rand(TOTAL_FRAMES, MAX_SIDE, MAX_SIDE, 3).astype(np.float32)
    padded = _pad_frames(imgs, TOTAL_FRAMES)
    assert padded.shape == (TOTAL_FRAMES, MAX_SIDE, MAX_SIDE, 3)
    np.testing.assert_array_equal(padded, imgs)


def test_pad_frames_long_video():
    imgs = np.random.rand(100, MAX_SIDE, MAX_SIDE, 3).astype(np.float32)
    padded = _pad_frames(imgs, TOTAL_FRAMES)
    assert padded.shape == (TOTAL_FRAMES, MAX_SIDE, MAX_SIDE, 3)


def test_pad_frames_empty():
    imgs = np.zeros((0, MAX_SIDE, MAX_SIDE, 3), dtype=np.float32)
    padded = _pad_frames(imgs, TOTAL_FRAMES)
    assert padded.shape == (TOTAL_FRAMES, MAX_SIDE, MAX_SIDE, 3)


def test_center_crop():
    imgs = np.random.rand(TOTAL_FRAMES, MAX_SIDE, MAX_SIDE, 3).astype(np.float32)
    cropped = _center_crop(imgs, CROP_SIZE)
    assert cropped.shape == (TOTAL_FRAMES, CROP_SIZE, CROP_SIZE, 3)


def test_center_crop_preserves_content():
    imgs = np.zeros((1, 256, 256, 3), dtype=np.float32)
    imgs[0, 128, 128, :] = 1.0
    cropped = _center_crop(imgs, 224)
    offset = round((256 - 224) / 2.0)
    assert cropped[0, 128 - offset, 128 - offset, 0] == 1.0


def test_ensure_both_sides_noop_when_large_enough():
    imgs = np.random.rand(4, 256, 256, 3).astype(np.float32)
    out = _ensure_both_sides_at_least(imgs, CROP_SIZE)
    assert out is imgs


def test_ensure_both_sides_upscales_narrow_frame():
    """Mimics max-side cap leaving one dimension < 224 (e.g. 256×50)."""
    imgs = np.random.rand(4, 256, 50, 3).astype(np.float32)
    out = _ensure_both_sides_at_least(imgs, CROP_SIZE)
    assert out.shape[0] == 4
    assert out.shape[1] >= CROP_SIZE and out.shape[2] >= CROP_SIZE


def test_center_crop_raises_if_too_small():
    imgs = np.zeros((2, 100, 100, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="smaller than crop"):
        _center_crop(imgs, CROP_SIZE)
