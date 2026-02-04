import json
from pathlib import Path

import cv2
import numpy as np

from roi.crop import crop_roi


def test_template_min_conf_rejects_low_match(tmp_path: Path):
    template = np.zeros((10, 10), dtype=np.uint8)
    template[2:8, 2:8] = 255
    tpl_path = tmp_path / "tpl.png"
    cv2.imwrite(str(tpl_path), template)

    profile = {
        "resolution": [100, 100],
        "rois": {
            "supply": {
                "mode": "template",
                "template": str(tpl_path),
                "template_min_conf": 0.99,
                "padding": [0, 0, 0, 0],
                "enabled": True,
            }
        },
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    roi = crop_roi(frame, profile_path, "supply")
    assert roi is None
