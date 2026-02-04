from pathlib import Path

import cv2
import numpy as np

from ocr.engine import OCREngine
from ocr.read_supply import read_supply


def _load_template(name: str) -> np.ndarray:
    base = Path(__file__).resolve().parents[1] / "a"
    for ext in ("png", "jpg", "jpeg"):
        p = base / f"{name}.{ext}"
        if p.exists():
            return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    raise FileNotFoundError(name)


def test_read_supply_with_templates():
    digit_left = _load_template("4")
    digit_right = _load_template("9")
    slash = _load_template("slash")

    h = max(digit_left.shape[0], digit_right.shape[0], slash.shape[0]) + 4
    w = digit_left.shape[1] + slash.shape[1] + digit_right.shape[1] + 8
    canvas = np.zeros((h, w), dtype=np.uint8)

    x = 2
    y = 2
    canvas[y : y + digit_left.shape[0], x : x + digit_left.shape[1]] = digit_left
    x += digit_left.shape[1] + 2
    canvas[y : y + slash.shape[0], x : x + slash.shape[1]] = slash
    x += slash.shape[1] + 2
    canvas[y : y + digit_right.shape[0], x : x + digit_right.shape[1]] = digit_right

    roi = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    templates_dir = Path(__file__).resolve().parents[1] / "a"
    ocr = OCREngine("none")
    result = read_supply(roi, templates_dir, ocr)

    assert result.used == 4
    assert result.total == 9


def test_read_supply_two_digits():
    digit_left1 = _load_template("1")
    digit_left2 = _load_template("2")
    digit_right1 = _load_template("3")
    digit_right2 = _load_template("4")
    slash = _load_template("slash")

    h = max(
        digit_left1.shape[0],
        digit_left2.shape[0],
        digit_right1.shape[0],
        digit_right2.shape[0],
        slash.shape[0],
    ) + 4
    w = (
        digit_left1.shape[1]
        + digit_left2.shape[1]
        + slash.shape[1]
        + digit_right1.shape[1]
        + digit_right2.shape[1]
        + 12
    )
    canvas = np.zeros((h, w), dtype=np.uint8)

    x = 2
    y = 2
    canvas[y : y + digit_left1.shape[0], x : x + digit_left1.shape[1]] = digit_left1
    x += digit_left1.shape[1] + 1
    canvas[y : y + digit_left2.shape[0], x : x + digit_left2.shape[1]] = digit_left2
    x += digit_left2.shape[1] + 2
    canvas[y : y + slash.shape[0], x : x + slash.shape[1]] = slash
    x += slash.shape[1] + 2
    canvas[y : y + digit_right1.shape[0], x : x + digit_right1.shape[1]] = digit_right1
    x += digit_right1.shape[1] + 1
    canvas[y : y + digit_right2.shape[0], x : x + digit_right2.shape[1]] = digit_right2

    roi = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    templates_dir = Path(__file__).resolve().parents[1] / "a"
    ocr = OCREngine("none")
    result = read_supply(roi, templates_dir, ocr)

    assert result.used == 12
    assert result.total == 34
