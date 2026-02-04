from pathlib import Path

import cv2
import numpy as np

from roi.crop import crop_roi


def test_crop_roi_template_match():
    base = Path(__file__).resolve().parents[1]
    tpl_path = base / "a" / "supply_frame.png"
    template = cv2.imread(str(tpl_path), cv2.IMREAD_GRAYSCALE)

    frame = np.zeros((120, 200), dtype=np.uint8)
    y = 5
    x = frame.shape[1] - template.shape[1] - 5
    frame[y : y + template.shape[0], x : x + template.shape[1]] = template
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    profile = base / "src" / "roi" / "profile_480p.json"
    roi = crop_roi(frame, profile, "supply")
    assert roi is not None
    assert roi.shape[0] == template.shape[0]
    assert roi.shape[1] == template.shape[1]
