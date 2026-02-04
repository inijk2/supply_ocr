from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

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


def _extract_components(binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 10:
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    return boxes


def _split_by_projection(binary: np.ndarray) -> int | None:
    h, w = binary.shape[:2]
    col_sum = (binary > 0).sum(axis=0)
    left = int(w * 0.3)
    right = int(w * 0.7)
    if right <= left:
        return None
    window = col_sum[left:right]
    if window.size == 0:
        return None
    idx = int(window.argmin()) + left
    if col_sum[idx] > max(1, int(h * 0.1)):
        return None
    return idx


def _template_digits_from_contours(img: np.ndarray, digit_templates: Dict[str, np.ndarray]) -> OCRResult:
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = []

    for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, binary = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)
        boxes = _extract_components(binary)
        split_result = OCRResult("", 0.0)
        split_idx = _split_by_projection(binary)
        if split_idx is not None:
            left = gray[:, :split_idx]
            right = gray[:, split_idx:]
            left_res = _template_digit_read(left, digit_templates)
            right_res = _template_digit_read(right, digit_templates)
            if left_res.text and right_res.text:
                conf = float((left_res.conf + right_res.conf) / 2.0)
                split_result = OCRResult(left_res.text + right_res.text, conf)

        if not boxes:
            results.append(split_result)
            continue

        digits = []
        confs = []
        for x, y, w, h in boxes[:2]:
            crop = gray[y : y + h, x : x + w]
            res = _template_digit_read(crop, digit_templates)
            if res.text:
                digits.append(res.text)
                confs.append(res.conf)
        contour_result = OCRResult("", 0.0)
        if digits:
            conf = float(sum(confs) / len(confs)) if confs else 0.0
            contour_result = OCRResult("".join(digits), conf)
        results.append(max([split_result, contour_result], key=lambda r: (len(r.text), r.conf)))

    return max(results, key=lambda r: (len(r.text), r.conf))


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

    left_raw = None
    right_raw = None

    if slash_conf >= 0.6:
        left_raw = gray[:, : max(1, sx - 1)]
        right_raw = gray[:, sx + sw + 1 :]
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        split_idx = _split_by_projection(binary)
        if split_idx is not None:
            left_raw = gray[:, :split_idx]
            right_raw = gray[:, split_idx:]

    if left_raw is None or right_raw is None:
        return SupplyReadResult(None, None, "", 0.0)

    left_raw = left_raw[:, : max(1, left_raw.shape[1])]
    right_raw = right_raw[:, : max(1, right_raw.shape[1])]

    if left_raw.size == 0 or right_raw.size == 0:
        return SupplyReadResult(None, None, "", 0.0)

    left_pre = preprocess(left_raw)
    right_pre = preprocess(right_raw)
    left_ocr = ocr.read_text(left_pre, whitelist="0123456789")
    right_ocr = ocr.read_text(right_pre, whitelist="0123456789")

    if not left_ocr.text:
        left_ocr = _template_digits_from_contours(left_raw, digit_templates)
    if not right_ocr.text:
        right_ocr = _template_digits_from_contours(right_raw, digit_templates)

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
