"""
ASL Citizen style dataset loader for Microsoft I3D, adapted for EyeHearU splits.

Expected split CSV format:
  user,filename,gloss

`filename` can be either:
  - clip.mp4
  - train/gloss/clip.mp4

and will be resolved relative to `video_root`.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl


def load_rgb_frames_from_video(video_path: str, max_frames: int = 64) -> np.ndarray:
    """
    Load and normalize RGB frames from an mp4 clip.
    Output shape: (T, H, W, C), float32 in [-1, 1].
    """
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        vidcap.release()
        return np.zeros((0, 256, 256, 3), dtype=np.float32)

    frameskip = 1
    if total_frames >= 96:
        frameskip = 2
    if total_frames >= 160:
        frameskip = 3

    if frameskip == 3:
        start = np.clip(int((total_frames - 192) // 2), 0, max(total_frames - 1, 0))
    elif frameskip == 2:
        start = np.clip(int((total_frames - 128) // 2), 0, max(total_frames - 1, 0))
    else:
        start = np.clip(int((total_frames - 64) // 2), 0, max(total_frames - 1, 0))
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, int(start))

    limit = min(max_frames * frameskip, max(total_frames - int(start), 0))
    for offset in range(limit):
        success, img = vidcap.read()
        if not success or img is None:
            break
        if offset % frameskip != 0:
            continue

        h, w, c = img.shape
        if min(h, w) < 226:
            d = 226.0 - float(min(h, w))
            sc = 1.0 + d / float(min(h, w))
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
            h, w = img.shape[:2]

        if h > 256 or w > 256:
            scale_h = 256.0 / float(h)
            scale_w = 256.0 / float(w)
            scale = min(scale_h, scale_w)
            img = cv2.resize(
                img,
                (int(math.ceil(w * scale)), int(math.ceil(h * scale))),
                interpolation=cv2.INTER_LINEAR,
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img / 255.0) * 2.0 - 1.0
        frames.append(img.astype(np.float32))

    vidcap.release()
    return np.asarray(frames, dtype=np.float32)


def video_to_tensor(pic: np.ndarray) -> torch.Tensor:
    """Convert ndarray (T,H,W,C) to tensor (C,T,H,W)."""
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


class ASLCitizenI3DDataset(data_utl.Dataset):
    def __init__(
        self,
        video_root: str | Path,
        split_csv: str | Path,
        transforms,
        gloss_dict: dict[str, int] | None = None,
        total_frames: int = 64,
        require_existing: bool = True,
    ):
        self.transforms = transforms
        self.video_root = Path(video_root)
        self.split_csv = Path(split_csv)
        self.total_frames = total_frames
        self.require_existing = require_existing
        self.video_paths: list[Path] = []
        self.labels: list[int] = []

        rows = []
        with open(self.split_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                user = (row.get("user") or "").strip()
                filename = (row.get("filename") or "").strip()
                gloss = (row.get("gloss") or "").strip().lower()
                if not filename or not gloss:
                    continue
                rows.append({"user": user, "filename": filename, "gloss": gloss})

        if gloss_dict is None:
            gloss_list = sorted({r["gloss"] for r in rows})
            self.gloss_dict = {g: i for i, g in enumerate(gloss_list)}
        else:
            self.gloss_dict = gloss_dict

        for row in rows:
            gloss = row["gloss"]
            if gloss not in self.gloss_dict:
                continue
            path = self.video_root / row["filename"]
            if self.require_existing and not path.exists():
                continue
            if self.require_existing and path.exists() and path.stat().st_size == 0:
                continue
            self.video_paths.append(path)
            self.labels.append(self.gloss_dict[gloss])

    def __len__(self) -> int:
        return len(self.video_paths)

    def pad(self, imgs: np.ndarray, total_frames: int) -> np.ndarray:
        if imgs.shape[0] >= total_frames:
            return imgs
        if imgs.shape[0] == 0:
            return np.zeros((total_frames, 256, 256, 3), dtype=np.float32)
        num_padding = total_frames - imgs.shape[0]
        pad_img = imgs[0] if np.random.random_sample() > 0.5 else imgs[-1]
        pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
        return np.concatenate([imgs, pad], axis=0)

    def __getitem__(self, index: int):
        video_path = self.video_paths[index]
        label_index = self.labels[index]

        imgs = load_rgb_frames_from_video(str(video_path), self.total_frames)
        imgs = self.pad(imgs, self.total_frames)
        imgs = self.transforms(imgs)
        tensor = video_to_tensor(imgs)

        return tensor, torch.tensor(label_index, dtype=torch.long)

