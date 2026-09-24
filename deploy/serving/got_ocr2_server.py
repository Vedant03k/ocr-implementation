"""KServe custom predictor for the GOT-OCR2.0 specialist (handwriting) recognizer.

Wraps ocr_pipeline.recognizer_specialist.GotOcrRecognizer behind KServe's v1
predict protocol. Each instance is one already-cropped text-line image (the
orchestrator does the cropping itself, using the box/poly the PaddleOCR
service returned, since only it has the original full-resolution image).

Request:  {"instances": [{"image_b64": "<base64-encoded crop bytes>"}]}
Response: {"predictions": ["recognized text", ...]}
"""

import base64
import os

import cv2
import numpy as np
from kserve import Model, ModelServer

from ocr_pipeline.recognizer_specialist import GotOcrRecognizer


class GotOcr2Model(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.recognizer: GotOcrRecognizer | None = None
        self.ready = False

    def load(self):
        self.recognizer = GotOcrRecognizer(
            model_dir=os.environ.get("GOT_OCR2_MODEL_DIR", "/model"),
            device=os.environ.get("GOT_OCR2_DEVICE", "cuda"),
        )
        self.ready = True

    def predict(self, payload: dict, headers: dict | None = None) -> dict:
        predictions = []
        for instance in payload["instances"]:
            image_bytes = base64.b64decode(instance["image_b64"])
            crop = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            predictions.append(self.recognizer.recognize(crop) if crop is not None else "")

        return {"predictions": predictions}


if __name__ == "__main__":
    model = GotOcr2Model(os.environ.get("MODEL_NAME", "got-ocr2"))
    model.load()
    ModelServer().start([model])
