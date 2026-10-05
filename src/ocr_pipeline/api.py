import os
import tempfile

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .cleanup import TextCleaner
from .detector import PaddleDetector
from .frame_sampler import sample_frames
from .main import VIDEO_EXTENSIONS, load_config
from .recognizer_specialist import GotOcrRecognizer
from .utils import crop_polygon
from .word_corrector import WordCorrector

config = load_config(os.environ.get("OCR_CONFIG", "config/config.yaml"))
detector = PaddleDetector(
    lang=config["paddleocr"]["lang"],
    device=config["paddleocr"].get("device", "cpu"),
    use_doc_unwarping=config["paddleocr"].get("use_doc_unwarping", False),
)
specialist = GotOcrRecognizer(
    model_dir=config["got_ocr2"]["model_dir"],
    device=config["got_ocr2"].get("device"),
    max_new_tokens=config["got_ocr2"].get("max_new_tokens", 128),
    batch_size=config["got_ocr2"].get("batch_size", 4),
)
_cleanup_cfg = config.get("llm_cleanup", {})
cleaner = (
    TextCleaner(_cleanup_cfg["model_dir"], _cleanup_cfg.get("device")) if _cleanup_cfg.get("enabled") else None
)
_correction_cfg = config.get("word_correction", {})
corrector = (
    WordCorrector(
        max_edit_distance=_correction_cfg.get("max_edit_distance", 2),
        real_word_max_confidence=_correction_cfg.get("real_word_max_confidence", 0.9),
        min_candidate_frequency=_correction_cfg.get("min_candidate_frequency", 200_000),
    )
    if _correction_cfg.get("enabled")
    else None
)
default_threshold = config["router"]["confidence_threshold"]

# Pays CUDA init and kernel selection now instead of on the first real upload.
specialist.recognize_batch([np.full((48, 320, 3), 255, dtype=np.uint8)])
if cleaner:
    cleaner.clean_lines(["warm up"])

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
    image: np.ndarray, prefix: str, relative_timestamp: float | None = None, threshold: float | None = None
) -> list[RawDetectionOut]:
    # Only lines below the caller's threshold go to GOT-OCR2.0, as router.py does.
    # The GUI sends its slider value with the upload, so what it shows by default
    # is exactly what was computed; lines above the threshold have no
    # specialistText, and the GUI can only re-route lines that have one.
    threshold = default_threshold if threshold is None else threshold
    height, width = image.shape[:2]
    detections = detector.detect(image)
    escalated = [i for i, det in enumerate(detections) if det.score < threshold]
    specialist_texts = dict(
        zip(escalated, specialist.recognize_batch([crop_polygon(image, detections[i].poly) for i in escalated]))
    )
    routed = [specialist_texts.get(i, det.text) for i, det in enumerate(detections)]

    # One whole-frame pass over the routed text (not per box), so each line's
    # neighbors are available as context.
    cleaned = None
    if routed and corrector:
        cleaned = corrector.correct_lines(routed, [det.score for det in detections])
    elif routed and cleaner:
        cleaned = cleaner.clean_lines(routed)

    out = []
    for i, det in enumerate(detections):
        is_escalated = i in specialist_texts
        out.append(
            RawDetectionOut(
                id=f"{prefix}-{i}",
                x=det.box.x1 / width,
                y=det.box.y1 / height,
                w=(det.box.x2 - det.box.x1) / width,
                h=(det.box.y2 - det.box.y1) / height,
                candidateText=det.text,
                specialistText=specialist_texts.get(i),
                cleanedCandidateText=cleaned[i] if cleaned and not is_escalated else None,
                cleanedSpecialistText=cleaned[i] if cleaned and is_escalated else None,
                rawConfidence=det.score,
                relativeTimestamp=relative_timestamp,
            )
        )
    return out


def _run_video(path: str, threshold: float | None = None) -> list[RawDetectionOut]:
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
        detections.extend(_detections_for_frame(frame, f"f{frame_index}", relative_timestamp, threshold))
    return detections


@app.post("/api/ocr", response_model=OcrResponse)
async def run_ocr(file: UploadFile = File(...), threshold: float | None = Form(None)) -> OcrResponse:
    if threshold is not None:
        threshold = min(max(threshold, 0.0), 1.0)
    suffix = os.path.splitext(file.filename or "")[1].lower()
    data = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        if suffix in VIDEO_EXTENSIONS:
            detections = _run_video(tmp_path, threshold)
        else:
            image = cv2.imread(tmp_path)
            detections = _detections_for_frame(image, "d", threshold=threshold) if image is not None else []
    finally:
        os.remove(tmp_path)

    return OcrResponse(detections=detections)
