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

    def __init__(
        self,
        lang: str = "en",
        device: str = "cpu",
        use_textline_orientation: bool = True,
        use_doc_unwarping: bool = False,
        det_limit_side_len: int | None = None,
    ):
        from paddleocr import PaddleOCR

        # Detection runs on the image shrunk so its longer side is at most this;
        # boxes come back in original coordinates, so crops stay full resolution.
        limit = {} if det_limit_side_len is None else {
            "text_det_limit_type": "max",
            "text_det_limit_side_len": det_limit_side_len,
        }
        self._ocr = PaddleOCR(
            lang=lang,
            device=device,
            use_textline_orientation=use_textline_orientation,
            use_doc_unwarping=use_doc_unwarping,
            **limit,
            enable_mkldnn=False,  # works around a PaddlePaddle 3.3.x CPU oneDNN bug, still present in 3.3.1
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
