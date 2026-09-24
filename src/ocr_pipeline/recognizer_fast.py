from .detector import Detection


class PaddleRecognizer:
    """Reads the recognition PaddleDetector already computed for a box.

    PaddleOCR 3.x fuses detection and recognition into one call (see
    detector.py), so there is no separate recognizer invocation here — this
    just exposes that result under the "fast path" name the architecture uses.
    """

    def recognize(self, detection: Detection) -> tuple[str, float]:
        return detection.text, detection.score
