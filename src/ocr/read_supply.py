from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from .engine import OCREngine, OCRResult
from .preprocess import preprocess


@dataclass
class SupplyReadResult:
    used: int | None
    total: int | None
    raw_text: str
    conf: float


def _load_templates(templates_dir: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    digits = {}
    for d in range(10):
        for ext in ("png", "jpg", "jpeg"):
            p = templates_dir / f"{d}.{ext}"
            if p.exists():
                digits[str(d)] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                break
    slash = None
    for ext in ("png", "jpg", "jpeg"):
        p = templates_dir / f"slash.{ext}"
        if p.exists():
            slash = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            break
    if slash is None:
        raise FileNotFoundError("slash template not found")
    return digits, slash


def _match_template(image: np.ndarray, template: np.ndarray) -> Tuple[Tuple[int, int], float]:
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    inv = 255 - image
    res_inv = cv2.matchTemplate(inv, template, cv2.TM_CCOEFF_NORMED)
    _, max_val_inv, _, max_loc_inv = cv2.minMaxLoc(res_inv)
    if max_val_inv > max_val:
        return max_loc_inv, float(max_val_inv)
    return max_loc, float(max_val)


def _template_digit_read(img: np.ndarray, digit_templates: Dict[str, np.ndarray]) -> OCRResult:
    if not digit_templates:
        return OCRResult("", 0.0)
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    best_digit = ""
    best_conf = 0.0
    for digit, tmpl in digit_templates.items():
        if tmpl is None:
            continue
        if gray.shape[0] < tmpl.shape[0] or gray.shape[1] < tmpl.shape[1]:
            continue
        _, conf = _match_template(gray, tmpl)
        _, conf_inv = _match_template(inv, tmpl)
        conf = max(conf, conf_inv)
        if conf > best_conf:
            best_conf = conf
            best_digit = digit
    return OCRResult(best_digit, best_conf)


def read_supply(
    roi_img: np.ndarray,
    templates_dir: Path,
    ocr: OCREngine,
) -> SupplyReadResult:
    digit_templates, slash_template = _load_templates(templates_dir)
    gray = roi_img if len(roi_img.shape) == 2 else cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    if gray.shape[0] < slash_template.shape[0] or gray.shape[1] < slash_template.shape[1]:
        return SupplyReadResult(None, None, "", 0.0)

    slash_loc, slash_conf = _match_template(gray, slash_template)
    sx, sy = slash_loc
    sh, sw = slash_template.shape[:2]

    left_raw = gray[:, : max(1, sx - 1)]
    right_raw = gray[:, sx + sw + 1 :]

    left_raw = left_raw[:, : max(1, left_raw.shape[1])]
    right_raw = right_raw[:, : max(1, right_raw.shape[1])]

    left_pre = preprocess(left_raw)
    right_pre = preprocess(right_raw)
    left_ocr = ocr.read_text(left_pre, whitelist="0123456789")
    right_ocr = ocr.read_text(right_pre, whitelist="0123456789")

    if not left_ocr.text:
        left_ocr = _template_digit_read(left_raw, digit_templates)
    if not right_ocr.text:
        right_ocr = _template_digit_read(right_raw, digit_templates)

    raw = ""
    used = None
    total = None
    conf = min(left_ocr.conf, right_ocr.conf)

    if left_ocr.text and right_ocr.text:
        raw = f"{left_ocr.text}/{right_ocr.text}"
        try:
            used = int(left_ocr.text)
            total = int(right_ocr.text)
            if used > total or total <= 0:
                return SupplyReadResult(None, None, raw, conf)
        except Exception:
            used = None
            total = None

    conf = min(conf, slash_conf)
    return SupplyReadResult(used, total, raw, conf)
