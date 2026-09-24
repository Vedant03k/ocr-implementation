import os
import tempfile

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .cleanup import TextCleaner
from .detector import PaddleDetector
from .frame_sampler import sample_frames
from .main import VIDEO_EXTENSIONS, load_config
from .recognizer_specialist import GotOcrRecognizer
from .utils import crop_polygon

config = load_config(os.environ.get("OCR_CONFIG", "config/config.yaml"))
detector = PaddleDetector(lang=config["paddleocr"]["lang"], device=config["paddleocr"].get("device", "cpu"))
specialist = GotOcrRecognizer(model_dir=config["got_ocr2"]["model_dir"], device=config["got_ocr2"].get("device"))
_cleanup_cfg = config.get("llm_cleanup", {})
cleaner = (
    TextCleaner(_cleanup_cfg["model_dir"], _cleanup_cfg.get("device")) if _cleanup_cfg.get("enabled") else None
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("OCR_CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class RawDetectionOut(BaseModel):
    id: str
    x: float
    y: float
    w: float
    h: float
    candidateText: str
    specialistText: str | None = None
    cleanedCandidateText: str | None = None
    cleanedSpecialistText: str | None = None
    rawConfidence: float
    relativeTimestamp: float | None = None


class OcrResponse(BaseModel):
    detections: list[RawDetectionOut]


def _detections_for_frame(
    image: np.ndarray, prefix: str, relative_timestamp: float | None = None
) -> list[RawDetectionOut]:
    # Unlike router.py (which only escalates crops below the confidence threshold,
    # as the real pipeline does), this always runs GOT-OCR2.0 too, so the GUI's
    # threshold slider can re-route results live without re-running the backend.
    height, width = image.shape[:2]
    detections = detector.detect(image)
    candidate_texts = [det.text for det in detections]
    specialist_texts = [specialist.recognize(crop_polygon(image, det.poly)) for det in detections]

    # Cleaned as two whole-frame passes (not per box) so the model has each
    # line's neighbors as context — see cleanup.py's docstring for why that
    # matters for disambiguating a line cut mid-phrase by the line detector.
    cleaned_candidates = cleaner.clean_lines(candidate_texts) if cleaner and candidate_texts else None
    cleaned_specialists = cleaner.clean_lines(specialist_texts) if cleaner and specialist_texts else None

    out = []
    for i, det in enumerate(detections):
        out.append(
            RawDetectionOut(
                id=f"{prefix}-{i}",
                x=det.box.x1 / width,
                y=det.box.y1 / height,
                w=(det.box.x2 - det.box.x1) / width,
                h=(det.box.y2 - det.box.y1) / height,
                candidateText=candidate_texts[i],
                specialistText=specialist_texts[i],
                cleanedCandidateText=cleaned_candidates[i] if cleaned_candidates else None,
                cleanedSpecialistText=cleaned_specialists[i] if cleaned_specialists else None,
                rawConfidence=det.score,
                relativeTimestamp=relative_timestamp,
            )
        )
    return out


def _run_video(path: str) -> list[RawDetectionOut]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = (frame_count / fps) if fps else 0.0
    cap.release()

    sampler_cfg = config["frame_sampler"]
    detections: list[RawDetectionOut] = []
    for frame_index, (frame, timestamp) in enumerate(
        sample_frames(path, fps=sampler_cfg["fps"], dedupe_similarity_threshold=sampler_cfg["dedupe_similarity_threshold"])
    ):
        relative_timestamp = (timestamp / duration) if duration else 0.0
        detections.extend(_detections_for_frame(frame, f"f{frame_index}", relative_timestamp))
    return detections


@app.post("/api/ocr", response_model=OcrResponse)
async def run_ocr(file: UploadFile = File(...)) -> OcrResponse:
    suffix = os.path.splitext(file.filename or "")[1].lower()
    data = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        if suffix in VIDEO_EXTENSIONS:
            detections = _run_video(tmp_path)
        else:
            image = cv2.imread(tmp_path)
            detections = _detections_for_frame(image, "d") if image is not None else []
    finally:
        os.remove(tmp_path)

    return OcrResponse(detections=detections)
