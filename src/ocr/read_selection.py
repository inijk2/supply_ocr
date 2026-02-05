from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .engine import OCREngine, OCRResult
from .preprocess import preprocess, preprocess_text


@dataclass
class SelectionOCRResult:
    selected_name: OCRResult
    hp_text: OCRResult


def read_selection(
    roi_img: np.ndarray,
    ocr: OCREngine,
    name_line: Tuple[int, int, int, int] | None = None,
    hp_line: Tuple[int, int, int, int] | None = None,
) -> SelectionOCRResult:
    h, w = roi_img.shape[:2]
    if name_line is None:
        name_line = (0, 0, w, max(1, int(h * 0.35)))
    if hp_line is None:
        hp_line = (0, int(h * 0.35), w, max(1, int(h * 0.25)))

    x, y, cw, ch = name_line
    name_crop = roi_img[y : y + ch, x : x + cw]
    name_pre = preprocess_text(name_crop)
    name_res = ocr.read_text(name_pre, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ")

    x, y, cw, ch = hp_line
    hp_crop = roi_img[y : y + ch, x : x + cw]
    hp_pre = preprocess_text(hp_crop)
    hp_res = ocr.read_text(hp_pre, whitelist="0123456789/ ")

    return SelectionOCRResult(name_res, hp_res)
