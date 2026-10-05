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
    escalated = [det for det in detections if det.score < confidence_threshold]
    specialist_texts = iter(specialist.recognize_batch([crop_polygon(image, det.poly) for det in escalated]))
    routed = []
    for det in detections:
        if det.score >= confidence_threshold:
            routed.append(RoutedBlock(box=det.box, text=det.text, confidence=det.score, source="paddleocr"))
        else:
            routed.append(RoutedBlock(box=det.box, text=next(specialist_texts), confidence=det.score, source="got_ocr2"))
    return routed
