"""KServe custom predictor for the PaddleOCR detection + fast-path recognition stage.

Wraps ocr_pipeline.detector.PaddleDetector (same class used by the local
main.py/api.py) behind KServe's v1 predict protocol, so the orchestrator can
call it as a remote service instead of loading PaddleOCR in-process.

Request:  {"instances": [{"image_b64": "<base64-encoded image bytes>"}]}
Response: {"predictions": [{"detections": [
              {"poly": [[x,y], ...], "box": {"x1":.., "y1":.., "x2":.., "y2":..},
               "text": "...", "score": 0.97},
              ...
          ]}]}
"""

import base64
import os

import cv2
import numpy as np
from kserve import Model, ModelServer

from ocr_pipeline.detector import PaddleDetector


class PaddleOCRModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.detector: PaddleDetector | None = None
        self.ready = False

    def load(self):
        self.detector = PaddleDetector(
            lang=os.environ.get("PADDLEOCR_LANG", "en"),
            device=os.environ.get("PADDLEOCR_DEVICE", "cpu"),
        )
        self.ready = True

    def predict(self, payload: dict, headers: dict | None = None) -> dict:
        predictions = []
        for instance in payload["instances"]:
            image_bytes = base64.b64decode(instance["image_b64"])
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                predictions.append({"detections": []})
                continue

            detections = [
                {
                    "poly": det.poly.tolist(),
                    "box": {"x1": det.box.x1, "y1": det.box.y1, "x2": det.box.x2, "y2": det.box.y2},
                    "text": det.text,
                    "score": det.score,
                }
                for det in self.detector.detect(image)
            ]
            predictions.append({"detections": detections})

        return {"predictions": predictions}


if __name__ == "__main__":
    model = PaddleOCRModel(os.environ.get("MODEL_NAME", "paddleocr"))
    model.load()
    ModelServer().start([model])
