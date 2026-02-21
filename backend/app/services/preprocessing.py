"""
Image and video preprocessing for inference.

Image: single frame for compatibility.
Video: decode mp4, sample 16 frames, resize 224x224, normalize → (1, C, T, H, W) tensor
for the ASL video classifier.
"""

import io
import numpy as np
import torch
from PIL import Image

# Model input (must match training / preprocess_clips.py)
NUM_FRAMES = 16
INPUT_SIZE = (224, 224)  # H, W

# ImageNet normalization (standard for pretrained CNN backbones)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_video(video_bytes: bytes) -> torch.Tensor:
    """
    Decode video bytes (mp4), sample NUM_FRAMES frames, resize, normalize.
    Returns tensor (1, C, T, H, W) float32 for the video classifier.
    """
    import tempfile
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python required for video inference: pip install opencv-python")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not decode video")
        frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        cap.release()
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if len(frames) == 0:
        raise ValueError("Video has no frames")
    arr = np.stack(frames)  # (T, H, W, 3)
    t = arr.shape[0]
    if t < NUM_FRAMES:
        pad = np.repeat(arr[-1:], NUM_FRAMES - t, axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    elif t > NUM_FRAMES:
        indices = np.linspace(0, t - 1, NUM_FRAMES, dtype=int)
        arr = arr[indices]
    # Resize each frame to INPUT_SIZE
    out = np.stack([
        np.array(
            Image.fromarray(f).resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        for f in arr
    ])  # (T, H, W, 3)
    out = (out - MEAN) / STD
    out = np.transpose(out, (3, 0, 1, 2))  # (C, T, H, W)
    out = np.expand_dims(out, axis=0)
    return torch.from_numpy(out.astype(np.float32))


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes into a normalized numpy array
    ready for model inference.

    Args:
        image_bytes: Raw bytes from the uploaded image file.

    Returns:
        Numpy array of shape (1, 3, H, W), float32, normalized.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(INPUT_SIZE, Image.BILINEAR)

    # Convert to float32 array in [0, 1]
    arr = np.array(image, dtype=np.float32) / 255.0

    # Normalize with ImageNet stats
    arr = (arr - MEAN) / STD

    # HWC → CHW, add batch dimension
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    return arr
