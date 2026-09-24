from typing import NamedTuple

import numpy as np

from .schema import BoundingBox


class Detection(NamedTuple):
    poly: np.ndarray
    box: BoundingBox
    text: str
    score: float


class PaddleDetector:
    """Wraps PaddleX's fused OCR pipeline.

    PaddleOCR 3.x/PaddleX has no standalone detection-only call at this API
    level: one predict() runs detection and first-pass recognition together.
    recognizer_fast.py reads the recognition this call already produced
    instead of invoking PaddleOCR's recognizer a second time.
    """

    def __init__(self, lang: str = "en", device: str = "cpu", use_textline_orientation: bool = True):
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            lang=lang,
            device=device,
            use_textline_orientation=use_textline_orientation,
            enable_mkldnn=False,  # works around a PaddlePaddle 3.3.x CPU oneDNN bug
        )

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self._ocr.predict(image)[0]
        detections = []
        for poly, box, text, score in zip(
            result["dt_polys"], result["rec_boxes"], result["rec_texts"], result["rec_scores"]
        ):
            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append(
                Detection(
                    poly=np.asarray(poly, dtype=np.float32),
                    box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    text=text,
                    score=float(score),
                )
            )
        return detections
