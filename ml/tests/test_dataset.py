"""Unit tests for training/dataset.py.

These tests exercise ASLVideoDataset without requiring real video files
or cv2 — they mock filesystem and video reading where needed.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from training.dataset import MEAN, STD, ASLVideoDataset


@pytest.fixture
def data_dir(tmp_path):
    """Create a minimal data directory with label_map and clip stubs."""
    label_map = {"hello": 0, "thanks": 1, "yes": 2}
    (tmp_path / "label_map.json").write_text(json.dumps(label_map))

    # Create clip directories with empty .mp4 files
    for split in ("train", "test"):
        for gloss in ("hello", "thanks"):
            clip_dir = tmp_path / "clips" / split / gloss
            clip_dir.mkdir(parents=True)
            (clip_dir / "clip_001.mp4").touch()
            (clip_dir / "clip_002.mp4").touch()

    return tmp_path


@pytest.fixture
def mock_cv2_capture():
    """Patch cv2.VideoCapture to return synthetic frames."""

    def _make_capture(path):
        cap = MagicMock()
        # Return 8 frames of 224x224x3, then stop
        frames = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
        call_count = [0]

        def read_side_effect():
            if call_count[0] < len(frames):
                frame = frames[call_count[0]]
                call_count[0] += 1
                return True, frame
            return False, None

        cap.read.side_effect = read_side_effect
        cap.release = MagicMock()
        return cap

    with patch("training.dataset.cv2") as mock_cv2:
        mock_cv2.VideoCapture = _make_capture
        mock_cv2.cvtColor = lambda frame, code: frame  # pass through
        mock_cv2.COLOR_BGR2RGB = 4
        mock_cv2.resize = lambda f, size, interpolation=None: np.zeros(
            (size[1], size[0], 3), dtype=np.uint8
        )
        mock_cv2.INTER_LINEAR = 1
        yield mock_cv2


class TestASLVideoDatasetInit:
    def test_loads_label_map(self, data_dir, mock_cv2_capture):
        ds = ASLVideoDataset(data_dir, split="train", num_frames=16)
        assert ds.label_map == {"hello": 0, "thanks": 1, "yes": 2}

    def test_collects_samples(self, data_dir, mock_cv2_capture):
        ds = ASLVideoDataset(data_dir, split="train", num_frames=16)
        # 2 glosses x 2 clips each = 4 samples
        assert len(ds) == 4

    def test_missing_clips_dir(self, data_dir, mock_cv2_capture):
        ds = ASLVideoDataset(data_dir, split="val", num_frames=16)
        assert len(ds) == 0

    def test_ignores_unknown_gloss(self, data_dir, mock_cv2_capture):
        unknown_dir = data_dir / "clips" / "train" / "unknown_gloss"
        unknown_dir.mkdir(parents=True)
        (unknown_dir / "clip.mp4").touch()
        ds = ASLVideoDataset(data_dir, split="train", num_frames=16)
        # "unknown_gloss" is not in label_map, so still 4
        assert len(ds) == 4

    def test_ignores_non_directory(self, data_dir, mock_cv2_capture):
        """Files in the split dir that aren't directories should be skipped."""
        stray_file = data_dir / "clips" / "train" / "stray.txt"
        stray_file.touch()
        ds = ASLVideoDataset(data_dir, split="train", num_frames=16)
        assert len(ds) == 4


class TestASLVideoDatasetGetItem:
    def test_returns_tensor_and_label(self, data_dir, mock_cv2_capture):
        ds = ASLVideoDataset(data_dir, split="train", num_frames=16)
        tensor, label = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(label, int)

    def test_tensor_shape(self, data_dir, mock_cv2_capture):
        ds = ASLVideoDataset(data_dir, split="train", num_frames=16)
        tensor, _ = ds[0]
        # (C, T, H, W) = (3, 16, 224, 224)
        assert tensor.shape == (3, 16, 224, 224)

    def test_tensor_dtype(self, data_dir, mock_cv2_capture):
        ds = ASLVideoDataset(data_dir, split="train", num_frames=16)
        tensor, _ = ds[0]
        assert tensor.dtype == torch.float32


class TestNormalize:
    def test_normalize_output_range(self, data_dir, mock_cv2_capture):
        ds = ASLVideoDataset(data_dir, split="train", num_frames=16)
        tensor, _ = ds[0]
        # After ImageNet normalisation, values should be roughly in [-3, 3]
        assert tensor.min() >= -5.0
        assert tensor.max() <= 5.0

    def test_normalize_shape(self):
        """Test _normalize directly: (T,H,W,C) -> (C,T,H,W)."""
        ds_cls = ASLVideoDataset.__new__(ASLVideoDataset)
        clip = np.random.randint(0, 256, (4, 10, 10, 3), dtype=np.uint8)
        tensor = ds_cls._normalize(clip)
        assert tensor.shape == (3, 4, 10, 10)


class TestReadClip:
    def test_pad_short_clip(self, data_dir):
        """When video has fewer frames than num_frames, pad by repeating last frame."""
        ds_cls = ASLVideoDataset.__new__(ASLVideoDataset)
        ds_cls.num_frames = 16

        frames = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(4)]
        call_count = [0]

        mock_cap = MagicMock()
        def read_side_effect():
            if call_count[0] < len(frames):
                f = frames[call_count[0]]
                call_count[0] += 1
                return True, f
            return False, None
        mock_cap.read.side_effect = read_side_effect

        with patch("training.dataset.cv2") as mock_cv2:
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.cvtColor = lambda frame, code: frame
            mock_cv2.COLOR_BGR2RGB = 4
            result = ds_cls._read_clip(Path("dummy.mp4"))

        assert result.shape[0] == 16

    def test_truncate_long_clip(self, data_dir):
        """When video has more frames than num_frames, subsample."""
        ds_cls = ASLVideoDataset.__new__(ASLVideoDataset)
        ds_cls.num_frames = 4

        frames = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(20)]
        call_count = [0]

        mock_cap = MagicMock()
        def read_side_effect():
            if call_count[0] < len(frames):
                f = frames[call_count[0]]
                call_count[0] += 1
                return True, f
            return False, None
        mock_cap.read.side_effect = read_side_effect

        with patch("training.dataset.cv2") as mock_cv2:
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.cvtColor = lambda frame, code: frame
            mock_cv2.COLOR_BGR2RGB = 4
            result = ds_cls._read_clip(Path("dummy.mp4"))

        assert result.shape[0] == 4

    def test_empty_video_returns_zeros(self, data_dir):
        """When no frames can be read, return zeros."""
        ds_cls = ASLVideoDataset.__new__(ASLVideoDataset)
        ds_cls.num_frames = 16

        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)

        with patch("training.dataset.cv2") as mock_cv2:
            mock_cv2.VideoCapture.return_value = mock_cap
            result = ds_cls._read_clip(Path("empty.mp4"))

        assert result.shape == (16, 224, 224, 3)
        assert np.all(result == 0)


class TestAugmentations:
    def test_augment_preserves_shape(self, data_dir, mock_cv2_capture):
        ds = ASLVideoDataset(data_dir, split="train", num_frames=16, augment=True)
        tensor, _ = ds[0]
        assert tensor.shape == (3, 16, 224, 224)

    def test_augment_flag_false(self, data_dir, mock_cv2_capture):
        ds = ASLVideoDataset(data_dir, split="train", num_frames=16, augment=False)
        assert ds.augment is False

    def test_augment_flag_true(self, data_dir, mock_cv2_capture):
        ds = ASLVideoDataset(data_dir, split="train", num_frames=16, augment=True)
        assert ds.augment is True

    def test_temporal_shift_applied(self):
        """When RNG triggers temporal shift (lines 109-111), clip is np.rolled."""
        ds_cls = ASLVideoDataset.__new__(ASLVideoDataset)
        clip = np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3)

        # Force all augmentation branches to fire: rand() < 0.5 for temporal,
        # rand() < 0.5 for brightness, rand() < 0.5 for spatial crop
        with patch("training.dataset.np.random.rand", return_value=0.1), \
             patch("training.dataset.np.random.randint", return_value=1), \
             patch("training.dataset.np.random.uniform", return_value=1.0), \
             patch("training.dataset.cv2") as mock_cv2:
            mock_cv2.resize = lambda f, size, interpolation=None: np.zeros(
                (size[1], size[0], 3), dtype=np.uint8
            )
            mock_cv2.INTER_LINEAR = 1
            result = ds_cls._apply_augmentations(clip)
        assert result.shape[0] == 4  # temporal dimension preserved

    def test_spatial_crop_branch(self):
        """When RNG triggers spatial crop (lines 120-129), clip is cropped and resized."""
        ds_cls = ASLVideoDataset.__new__(ASLVideoDataset)
        clip = np.random.randint(0, 256, (4, 100, 100, 3), dtype=np.uint8)

        call_count = [0]
        rand_values = [0.9, 0.9, 0.1]  # skip temporal, skip brightness, DO spatial

        def mock_rand(*args, **kwargs):
            idx = min(call_count[0], len(rand_values) - 1)
            call_count[0] += 1
            return rand_values[idx]

        with patch("training.dataset.np.random.rand", side_effect=mock_rand), \
             patch("training.dataset.np.random.randint", return_value=0), \
             patch("training.dataset.cv2") as mock_cv2:
            mock_cv2.resize = lambda f, size, interpolation=None: np.zeros(
                (size[1], size[0], 3), dtype=np.uint8
            )
            mock_cv2.INTER_LINEAR = 1
            result = ds_cls._apply_augmentations(clip)
        assert result.shape[0] == 4


class TestConstants:
    def test_mean_shape(self):
        assert MEAN.shape == (3,)

    def test_std_shape(self):
        assert STD.shape == (3,)

    def test_mean_values(self):
        np.testing.assert_allclose(MEAN, [0.485, 0.456, 0.406], atol=1e-6)

    def test_std_values(self):
        np.testing.assert_allclose(STD, [0.229, 0.224, 0.225], atol=1e-6)
