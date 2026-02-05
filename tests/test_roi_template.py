from pathlib import Path
import json

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

    profile = {
        "resolution": [200, 120],
        "rois": {
            "supply": {
                "mode": "template",
                "template": str(tpl_path),
                "template_min_conf": 0.5,
                "padding": [0, 0, 0, 0],
                "enabled": True,
            }
        },
    }
    profile_path = base / "tests" / "_tmp_profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    roi = crop_roi(frame, profile_path, "supply")
    assert roi is not None
    assert roi.shape[0] == template.shape[0]
    assert roi.shape[1] >= template.shape[1]
