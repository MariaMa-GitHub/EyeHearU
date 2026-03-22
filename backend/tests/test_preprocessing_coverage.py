"""Branch coverage for preprocessing (cv2 paths, errors, preprocess_video)."""

import builtins
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from app.services import preprocessing as pre


def _bgr_frame(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_load_rgb_frames_total_zero():
    import cv2

    cap = MagicMock()
    cap.get.return_value = 0
    with patch("cv2.VideoCapture", return_value=cap):
        out = pre._load_rgb_frames("/fake/path.mp4")
    cap.release.assert_called_once()
    assert out.shape[0] == 0


def test_resize_bgr_uint8_down_and_up():
    import numpy as np

    img = np.zeros((400, 400, 3), dtype=np.uint8)
    down = pre._resize_bgr_uint8(img, 100, 100, shrinking=True)
    assert down.shape == (100, 100, 3)
    up = pre._resize_bgr_uint8(img, 800, 800, shrinking=False)
    assert up.shape == (800, 800, 3)


def test_load_rgb_frames_mobile_4k_downscale():
    """Very large iPhone-style frame triggers MOBILE_MAX_LONG_SIDE path."""
    import cv2
    import numpy as np

    cap = MagicMock()
    huge = np.zeros((2160, 3840, 3), dtype=np.uint8)

    def _get(prop):
        return 30 if prop == cv2.CAP_PROP_FRAME_COUNT else 0

    cap.get.side_effect = _get
    cap.read.return_value = (True, huge)
    with patch("cv2.VideoCapture", return_value=cap):
        out = pre._load_rgb_frames("/fake.mp4", max_frames=4)
    assert out.shape[0] >= 1
    assert max(out.shape[1], out.shape[2]) <= pre.MAX_SIDE


def test_load_rgb_frames_frameskip_1():
    import cv2

    cap = MagicMock()

    def _get(prop):
        return 50 if prop == cv2.CAP_PROP_FRAME_COUNT else 0

    cap.get.side_effect = _get
    img = _bgr_frame(240, 320)
    cap.read.return_value = (True, img)
    with patch("cv2.VideoCapture", return_value=cap):
        out = pre._load_rgb_frames("/fake.mp4", max_frames=64)
    assert out.shape[0] > 0
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_load_rgb_frames_frameskip_2():
    import cv2

    cap = MagicMock()

    def _get(prop):
        return 100 if prop == cv2.CAP_PROP_FRAME_COUNT else 0

    cap.get.side_effect = _get
    img = _bgr_frame(240, 320)
    cap.read.return_value = (True, img)
    with patch("cv2.VideoCapture", return_value=cap):
        out = pre._load_rgb_frames("/fake.mp4", max_frames=64)
    assert out.shape[0] > 0


def test_load_rgb_frames_frameskip_3():
    import cv2

    cap = MagicMock()

    def _get(prop):
        return 200 if prop == cv2.CAP_PROP_FRAME_COUNT else 0

    cap.get.side_effect = _get
    img = _bgr_frame(240, 320)
    cap.read.return_value = (True, img)
    with patch("cv2.VideoCapture", return_value=cap):
        out = pre._load_rgb_frames("/fake.mp4", max_frames=64)
    assert out.shape[0] > 0


def test_load_rgb_frames_small_side_resize():
    import cv2

    cap = MagicMock()

    def _get(prop):
        return 80 if prop == cv2.CAP_PROP_FRAME_COUNT else 0

    cap.get.side_effect = _get
    img = _bgr_frame(100, 100)
    cap.read.return_value = (True, img)
    with patch("cv2.VideoCapture", return_value=cap):
        out = pre._load_rgb_frames("/fake.mp4", max_frames=64)
    assert out.shape[0] > 0


def test_load_rgb_frames_large_side_resize():
    import cv2

    cap = MagicMock()

    def _get(prop):
        return 80 if prop == cv2.CAP_PROP_FRAME_COUNT else 0

    cap.get.side_effect = _get
    img = _bgr_frame(400, 400)
    cap.read.return_value = (True, img)
    with patch("cv2.VideoCapture", return_value=cap):
        out = pre._load_rgb_frames("/fake.mp4", max_frames=64)
    assert out.shape[0] > 0


def test_load_rgb_frames_read_fails():
    import cv2

    cap = MagicMock()

    def _get(prop):
        return 10 if prop == cv2.CAP_PROP_FRAME_COUNT else 0

    cap.get.side_effect = _get
    cap.read.return_value = (False, None)
    with patch("cv2.VideoCapture", return_value=cap):
        out = pre._load_rgb_frames("/fake.mp4", max_frames=64)
    assert out.shape[0] == 0


def test_load_rgb_frames_early_break():
    import cv2

    cap = MagicMock()

    def _get(prop):
        return 80 if prop == cv2.CAP_PROP_FRAME_COUNT else 0

    cap.get.side_effect = _get
    img = _bgr_frame(240, 320)
    cap.read.side_effect = [(True, img), (False, None)]
    with patch("cv2.VideoCapture", return_value=cap):
        out = pre._load_rgb_frames("/fake.mp4", max_frames=64)
    assert out.shape[0] >= 1


def test_load_rgb_frames_import_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cv2":
            raise ImportError("no cv2")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="opencv"):
        pre._load_rgb_frames("/x.mp4")


def test_preprocess_video_success_and_unlink_oserror():
    """End-to-end preprocess_video with mocked decode (avoids codec quirks in CI)."""
    frames = np.random.rand(64, 256, 256, 3).astype(np.float32) * 2 - 1
    with (
        patch.object(pre, "_load_rgb_frames", return_value=frames),
        patch("app.services.preprocessing.os.unlink", side_effect=OSError("ignored")),
    ):
        tensor = pre.preprocess_video(b"fake-mp4-bytes")
    assert tensor.shape == (1, 3, 64, 224, 224)
    assert tensor.dtype == torch.float32


def test_preprocess_video_no_frames_raises():
    import cv2

    with patch("cv2.VideoCapture") as VC:
        cap = MagicMock()
        cap.get.return_value = 0
        VC.return_value = cap
        with pytest.raises(ValueError, match="no decodable"):
            pre.preprocess_video(b"fakebytes")


def test_preprocess_video_decode_error_propagates():
    with patch.object(pre, "_load_rgb_frames", side_effect=ValueError("bad")):
        with pytest.raises(ValueError, match="bad"):
            pre.preprocess_video(b"x")
