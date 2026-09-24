from typing import NamedTuple

import numpy as np

from .detector import Detection
from .recognizer_specialist import GotOcrRecognizer
from .schema import BoundingBox
from .utils import crop_polygon


class RoutedBlock(NamedTuple):
    box: BoundingBox
    text: str
    confidence: float
    source: str


def route(
    image: np.ndarray,
    detections: list[Detection],
    specialist: GotOcrRecognizer,
    confidence_threshold: float = 0.85,
) -> list[RoutedBlock]:
    routed = []
    for det in detections:
        if det.score >= confidence_threshold:
            routed.append(RoutedBlock(box=det.box, text=det.text, confidence=det.score, source="paddleocr"))
        else:
            crop = crop_polygon(image, det.poly)
            text = specialist.recognize(crop)
            routed.append(RoutedBlock(box=det.box, text=text, confidence=det.score, source="got_ocr2"))
    return routed
