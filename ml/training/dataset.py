"""
PyTorch Dataset for ASL sign video clips.

Each sample is a (video_tensor, label_index) tuple where
video_tensor has shape (C, T, H, W) — the standard input
format for 3D CNNs like I3D and SlowFast.

Expected data layout (produced by the data pipeline):
    data/processed/
      clips/{split}/{gloss}/*.mp4
      label_map.json
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ImageNet normalisation (standard for Kinetics-pretrained backbones)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ASLVideoDataset(Dataset):
    """
    Dataset of preprocessed ASL sign video clips.

    Each clip is a short .mp4 with a fixed number of frames
    at a fixed resolution, produced by preprocess_clips.py.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        num_frames: int = 16,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_frames = num_frames
        self.augment = augment

        # Load label map
        label_map_path = self.data_dir / "label_map.json"
        with open(label_map_path) as f:
            self.label_map: dict[str, int] = json.load(f)

        self.samples = self._collect_samples()

    def _collect_samples(self) -> list[tuple[Path, int]]:
        clips_dir = self.data_dir / "clips" / self.split
        samples = []

        if not clips_dir.exists():
            print(f"[WARNING] Clip directory not found: {clips_dir}")
            return samples

        for class_dir in sorted(clips_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            gloss = class_dir.name
            if gloss not in self.label_map:
                continue
            label_idx = self.label_map[gloss]
            for clip_path in sorted(class_dir.glob("*.mp4")):
                samples.append((clip_path, label_idx))

        print(f"[{self.split}] {len(samples)} clips, "
              f"{len(self.label_map)} classes")
        return samples

    def _read_clip(self, path: Path) -> np.ndarray:
        """Read all frames from a preprocessed clip and return (T, H, W, C)."""
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            h, w = 224, 224
            return np.zeros((self.num_frames, h, w, 3), dtype=np.uint8)

        arr = np.stack(frames)  # (T, H, W, 3)

        # Pad or truncate to exactly self.num_frames
        t = arr.shape[0]
        if t < self.num_frames:
            pad = np.repeat(arr[-1:], self.num_frames - t, axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        elif t > self.num_frames:
            indices = np.linspace(0, t - 1, self.num_frames, dtype=int)
            arr = arr[indices]

        return arr

    def _apply_augmentations(self, clip: np.ndarray) -> np.ndarray:
        """Simple temporal and spatial augmentations for training."""
        # Random temporal shift: offset the start by ±1 frame
        if clip.shape[0] > 2 and np.random.rand() < 0.5:
            shift = np.random.randint(-1, 2)
            clip = np.roll(clip, shift, axis=0)

        # Random brightness/contrast jitter
        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.8, 1.2)   # contrast
            beta = np.random.uniform(-20, 20)      # brightness
            clip = np.clip(clip.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Random spatial crop (instead of center crop)
        if np.random.rand() < 0.5:
            t, h, w, c = clip.shape
            crop_h, crop_w = int(h * 0.9), int(w * 0.9)
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)
            clip = clip[:, top:top+crop_h, left:left+crop_w, :]
            clip = np.stack([
                cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
                for f in clip
            ])

        return clip

    def _normalize(self, clip: np.ndarray) -> torch.Tensor:
        """
        Normalise pixel values and convert to tensor.

        Input:  (T, H, W, C) uint8
        Output: (C, T, H, W) float32, ImageNet-normalised
        """
        arr = clip.astype(np.float32) / 255.0
        arr = (arr - MEAN) / STD                # broadcast over T, H, W
        arr = arr.transpose(3, 0, 1, 2)         # (T,H,W,C) → (C,T,H,W)
        return torch.from_numpy(arr)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        clip_path, label_idx = self.samples[idx]
        clip = self._read_clip(clip_path)        # (T, H, W, C) uint8

        if self.augment:
            clip = self._apply_augmentations(clip)

        tensor = self._normalize(clip)           # (C, T, H, W) float32
        return tensor, label_idx
