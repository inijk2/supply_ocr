from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    upscale: int = 3
    denoise_strength: int = 10
    adaptive_block_size: int = 15
    adaptive_c: int = 5
    use_morph: bool = False
    morph_kernel: Tuple[int, int] = (2, 2)


def upscale3x(img: np.ndarray, factor: int = 3) -> np.ndarray:
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)


def denoise(img: np.ndarray, strength: int = 10) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.fastNlMeansDenoising(img, h=strength)
    return cv2.fastNlMeansDenoisingColored(img, h=strength, hColor=strength)


def adaptive_threshold(img: np.ndarray, block_size: int = 15, c: int = 5) -> np.ndarray:
    if block_size % 2 == 0:
        block_size += 1
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )


def morph_cleanup(img: np.ndarray, kernel: Tuple[int, int] = (2, 2)) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k, iterations=1)
    return opened


def sharpness_score(img: np.ndarray) -> float:
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def preprocess(img: np.ndarray, cfg: PreprocessConfig | None = None) -> np.ndarray:
    cfg = cfg or PreprocessConfig()
    out = upscale3x(img, cfg.upscale)
    out = denoise(out, cfg.denoise_strength)
    out = adaptive_threshold(out, cfg.adaptive_block_size, cfg.adaptive_c)
    if cfg.use_morph:
        out = morph_cleanup(out, cfg.morph_kernel)
    return out
