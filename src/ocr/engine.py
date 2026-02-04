from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class OCRResult:
    text: str
    conf: float


class OCREngine:
    def __init__(self, engine: Optional[str] = None) -> None:
        self.engine = (engine or os.environ.get("OCR_ENGINE") or "auto").lower()
        self._impl = None
        self._impl_name = None
        self._init_impl()

    @property
    def name(self) -> str:
        return self._impl_name or "none"

    def _init_impl(self) -> None:
        if self.engine == "auto":
            for name in ("paddleocr", "easyocr", "tesseract"):
                if self._try_init(name):
                    return
        else:
            self._try_init(self.engine)

    def _try_init(self, name: str) -> bool:
        if name == "paddleocr":
            try:
                from paddleocr import PaddleOCR  # type: ignore

                self._impl = PaddleOCR(lang="en", show_log=False)
                self._impl_name = "paddleocr"
                return True
            except Exception:
                return False
        if name == "easyocr":
            try:
                import easyocr  # type: ignore

                self._impl = easyocr.Reader(["en"], gpu=False)
                self._impl_name = "easyocr"
                return True
            except Exception:
                return False
        if name == "tesseract":
            try:
                import pytesseract  # type: ignore

                self._impl = pytesseract
                self._impl_name = "tesseract"
                return True
            except Exception:
                return False
        return False

    def read_text(self, img: np.ndarray, whitelist: str | None = None) -> OCRResult:
        if self._impl_name is None:
            return OCRResult("", 0.0)
        if self._impl_name == "paddleocr":
            return self._read_paddle(img, whitelist)
        if self._impl_name == "easyocr":
            return self._read_easy(img, whitelist)
        if self._impl_name == "tesseract":
            return self._read_tesseract(img, whitelist)
        return OCRResult("", 0.0)

    def _read_paddle(self, img: np.ndarray, whitelist: str | None) -> OCRResult:
        config = {}
        if whitelist:
            config["rec_char_dict_path"] = None
        result = self._impl.ocr(img, cls=False)
        if not result:
            return OCRResult("", 0.0)
        best = max(result, key=lambda r: r[1][1])
        text, conf = best[1][0], float(best[1][1])
        if whitelist:
            text = "".join([c for c in text if c in whitelist])
        return OCRResult(text.strip(), conf)

    def _read_easy(self, img: np.ndarray, whitelist: str | None) -> OCRResult:
        result = self._impl.readtext(img)
        if not result:
            return OCRResult("", 0.0)
        best = max(result, key=lambda r: r[2])
        text, conf = best[1], float(best[2])
        if whitelist:
            text = "".join([c for c in text if c in whitelist])
        return OCRResult(text.strip(), conf)

    def _read_tesseract(self, img: np.ndarray, whitelist: str | None) -> OCRResult:
        config = "--psm 7"
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"
        data = self._impl.image_to_data(img, output_type=self._impl.Output.DICT, config=config)
        if not data or not data.get("text"):
            return OCRResult("", 0.0)
        texts = []
        confs = []
        for txt, conf in zip(data["text"], data["conf"]):
            if txt.strip():
                texts.append(txt)
                try:
                    confs.append(float(conf) / 100.0)
                except Exception:
                    pass
        if not texts:
            return OCRResult("", 0.0)
        text = " ".join(texts).strip()
        conf = float(sum(confs) / max(1, len(confs))) if confs else 0.0
        if whitelist:
            text = "".join([c for c in text if c in whitelist])
        return OCRResult(text, conf)
