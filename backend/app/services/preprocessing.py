"""
Video preprocessing for I3D inference — **optimized for iPhone / mobile recordings**
while staying aligned with training (ASL Citizen / I3D dataloader).

## Training data vs. app users

- **ASL Citizen (training)** clips are preprocessed dataset videos: mostly **fixed aspect**
  and resolution bands from the official pipeline; frames are already “hand-sized” in the
  frame after ingest.
- **App users (iOS / Expo)** record with **rear or front camera**, often **1080p or 4K**,
  **portrait (9:16)** or **landscape**, **HEVC/H.264**, and **~3 s** clips at **30–60 fps**.
  That means **much wider range of resolutions and aspect ratios** than typical training
  clips.

## Pipeline (order matters)

1. Decode with OpenCV; **optional coarse downscale** if either side exceeds
   ``MOBILE_MAX_LONG_SIDE`` so 4K/8K frames don’t waste memory or amplify codec quirks.
2. Same **temporal** logic as ``ml/i3d_msft/dataset.py``: adaptive frame skip, centered
   window (matches short phone clips).
3. Per frame, **training-style spatial** rules: min side **226**, max side **256**
   (then RGB, normalize to **[-1, 1]**).
4. Pad / trim to **64** frames; **ensure both H and W ≥ 224** (mobile can leave one side
   too small after the max-256 cap on panoramic aspect ratios — breaks center-crop + I3D).
5. **Center-crop 224×224** (eval-style).
6. Tensor **(1, 3, 64, 224, 224)**.

Interpolation: **INTER_AREA** when shrinking (typical for phone → model resolution),
**INTER_LINEAR** when enlarging — better detail for small previews / old devices.
"""

import math
import tempfile
import os

import numpy as np
import torch

# --- Temporal (match training dataloader) ---
TOTAL_FRAMES = 64

# --- Spatial (match training clip loader) ---
CROP_SIZE = 224
MIN_SIDE = 226
MAX_SIDE = 256

# --- Mobile / iPhone: cap extreme sensor resolution before training-style scaling ---
# iPhone 4K vertical is 2160 px tall; we don’t need full-res for a 224×224 model input.
MOBILE_MAX_LONG_SIDE = 1280


def _resize_bgr_uint8(img, new_w: int, new_h: int, *, shrinking: bool) -> np.ndarray:
    """Resize BGR uint8 frame; use AREA when downscaling (sharper for photos/video)."""
    import cv2

    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _load_rgb_frames(video_path: str, max_frames: int = TOTAL_FRAMES) -> np.ndarray:
    """
    Load and normalize RGB frames from an mp4 clip.
    Mirrors ``ml/i3d_msft/dataset.py:load_rgb_frames_from_video`` with **mobile-first**
    guards for very large iPhone frames.
    Returns (T, H, W, C) float32 in [-1, 1].
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python-headless required: pip install opencv-python-headless")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return np.zeros((0, MAX_SIDE, MAX_SIDE, 3), dtype=np.float32)

    frameskip = 1
    if total >= 96:
        frameskip = 2
    if total >= 160:
        frameskip = 3

    if frameskip == 3:
        start = np.clip(int((total - 192) // 2), 0, max(total - 1, 0))
    elif frameskip == 2:
        start = np.clip(int((total - 128) // 2), 0, max(total - 1, 0))
    else:
        start = np.clip(int((total - 64) // 2), 0, max(total - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))

    limit = min(max_frames * frameskip, max(total - int(start), 0))
    frames = []
    for offset in range(limit):
        ok, img = cap.read()
        if not ok or img is None:
            break
        if offset % frameskip != 0:
            continue

        h, w = img.shape[:2]

        # --- Mobile: tame 4K / very large captures before training-style min/max rules ---
        long_side = max(h, w)
        if long_side > MOBILE_MAX_LONG_SIDE:
            scale = MOBILE_MAX_LONG_SIDE / float(long_side)
            nw = max(1, int(math.ceil(w * scale)))
            nh = max(1, int(math.ceil(h * scale)))
            img = _resize_bgr_uint8(img, nw, nh, shrinking=True)
            h, w = img.shape[:2]

        if min(h, w) < MIN_SIDE:
            sc = 1.0 + (MIN_SIDE - float(min(h, w))) / float(min(h, w))
            nw = int(math.ceil(w * sc))
            nh = int(math.ceil(h * sc))
            img = _resize_bgr_uint8(img, nw, nh, shrinking=False)
            h, w = img.shape[:2]

        if h > MAX_SIDE or w > MAX_SIDE:
            scale = min(MAX_SIDE / float(h), MAX_SIDE / float(w))
            nw = max(1, int(math.ceil(w * scale)))
            nh = max(1, int(math.ceil(h * scale)))
            img = _resize_bgr_uint8(img, nw, nh, shrinking=True)
            h, w = img.shape[:2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img / 255.0) * 2.0 - 1.0
        frames.append(img.astype(np.float32))

    cap.release()
    return np.asarray(frames, dtype=np.float32) if frames else np.zeros((0, MAX_SIDE, MAX_SIDE, 3), dtype=np.float32)


def _pad_frames(imgs: np.ndarray, total_frames: int = TOTAL_FRAMES) -> np.ndarray:
    """Pad video to total_frames by repeating the last frame."""
    if imgs.shape[0] >= total_frames:
        return imgs[:total_frames]
    if imgs.shape[0] == 0:
        return np.zeros((total_frames, MAX_SIDE, MAX_SIDE, 3), dtype=np.float32)
    pad = np.tile(imgs[-1:], (total_frames - imgs.shape[0], 1, 1, 1))
    return np.concatenate([imgs, pad], axis=0)


def _ensure_both_sides_at_least(imgs: np.ndarray, min_side: int = CROP_SIZE) -> np.ndarray:
    """
    Scale frames so height and width are both >= min_side before center-crop.

    **Why (phones):** After the training-style max-side **256** cap, **portrait** or
    **ultra-wide** frames can become **256×51**-style strips. Center-cropping **224×224**
    is then invalid and I3D fails. Training batches often come from **cropped square-ish**
    ASL clips; this step restores a **valid crop region** for arbitrary **iOS aspect
    ratios** without changing the **[-1, 1]** normalization.
    """
    import cv2

    _, h, w, _ = imgs.shape
    if h >= min_side and w >= min_side:
        return imgs
    scale = float(min_side) / float(min(h, w))
    nh = max(int(math.ceil(h * scale)), min_side)
    nw = max(int(math.ceil(w * scale)), min_side)
    out = np.empty((imgs.shape[0], nh, nw, 3), dtype=np.float32)
    # Always upscaling here (min side was < min_side); LINEAR is appropriate for float RGB.
    for t in range(imgs.shape[0]):
        out[t] = cv2.resize(imgs[t], (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out


def _center_crop(imgs: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    """Center-crop (T, H, W, C) array to (T, size, size, C)."""
    _, h, w, _ = imgs.shape
    if h < size or w < size:
        raise ValueError(
            f"Frame spatial size ({h}x{w}) is smaller than crop {size}; "
            "upstream resize should have prevented this."
        )
    i = int(np.round((h - size) / 2.0))
    j = int(np.round((w - size) / 2.0))
    return imgs[:, i:i + size, j:j + size, :]


def preprocess_video(video_bytes: bytes) -> torch.Tensor:
    """
    Full preprocessing for one uploaded video (e.g. **iPhone / Expo** recording).

    Returns:
        Tensor of shape (1, 3, 64, 224, 224), float32 in [-1, 1].
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        imgs = _load_rgb_frames(tmp_path, TOTAL_FRAMES)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if imgs.shape[0] == 0:
        raise ValueError("Video has no decodable frames")

    imgs = _pad_frames(imgs, TOTAL_FRAMES)
    imgs = _ensure_both_sides_at_least(imgs, CROP_SIZE)
    imgs = _center_crop(imgs, CROP_SIZE)

    # (T, H, W, C) -> (C, T, H, W) -> (1, C, T, H, W)
    tensor = torch.from_numpy(imgs.transpose(3, 0, 1, 2)).unsqueeze(0)
    return tensor
