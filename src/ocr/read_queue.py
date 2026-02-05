from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .engine import OCREngine, OCRResult
from .preprocess import preprocess, preprocess_text


@dataclass
class QueueOCRResult:
    queue_text: OCRResult


def read_queue(
    roi_img: np.ndarray,
    ocr: OCREngine,
    text_line: Tuple[int, int, int, int] | None = None,
) -> QueueOCRResult:
    h, w = roi_img.shape[:2]
    if text_line is None:
        text_line = (0, 0, w, max(1, int(h * 0.4)))
    x, y, cw, ch = text_line
    crop = roi_img[y : y + ch, x : x + cw]
    pre = preprocess_text(crop)
    res = ocr.read_text(pre, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ")
    return QueueOCRResult(res)
