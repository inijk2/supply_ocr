from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class DiffConfig:
    threshold: float = 0.03


def diff_score(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    gray_a = a if len(a.shape) == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = b if len(b.shape) == 2 else cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_a, gray_b)
    return float(np.mean(diff) / 255.0)


def changed(a: np.ndarray, b: np.ndarray, cfg: DiffConfig | None = None) -> bool:
    cfg = cfg or DiffConfig()
    return diff_score(a, b) >= cfg.threshold
