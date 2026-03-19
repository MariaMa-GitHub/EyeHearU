"""
Video transforms sourced from microsoft/ASL-citizen-code (I3D), adapted as a
local module for reproducible training in this repository.
"""

import numbers
import random

import numpy as np


class RandomCrop:
    """Crop a video tensor (T, H, W, C) at a random spatial location."""

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(imgs, output_size):
        t, h, w, c = imgs.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th) if h != th else 0
        j = random.randint(0, w - tw) if w != tw else 0
        return i, j, th, tw

    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs, self.size)
        return imgs[:, i : i + h, j : j + w, :]

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class CenterCrop:
    """Center-crop a video tensor (T, H, W, C)."""

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.0))
        j = int(np.round((w - tw) / 2.0))
        return imgs[:, i : i + th, j : j + tw, :]

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class RandomHorizontalFlip:
    """Horizontally flip a video tensor with probability p."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        if random.random() < self.p:
            return np.flip(imgs, axis=2).copy()
        return imgs

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

