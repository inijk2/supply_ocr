import cv2
import numpy as np

from ocr.preprocess import sharpness_score


def test_sharpness_score_increases_with_detail():
    img = np.zeros((50, 50), dtype=np.uint8)
    img[10:40, 10:40] = 255
    sharp = sharpness_score(img)

    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    blurred_score = sharpness_score(blurred)

    assert sharp > blurred_score
