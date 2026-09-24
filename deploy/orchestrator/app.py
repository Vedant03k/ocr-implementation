"""Production API: calls the three KServe InferenceServices over HTTP instead
of loading PaddleOCR/GOT-OCR2.0/the cleanup LLM in-process, the way the local
dev backend (src/ocr_pipeline/api.py) does. Deliberately returns the exact
same RawDetectionOut/OcrResponse shape as api.py, so web/ only needs
NEXT_PUBLIC_OCR_API_URL repointed at this service — no frontend changes.

Same GUI-only shortcut as api.py: cleanup and GOT-OCR2.0 run on every
detected box regardless of confidence, so the threshold slider can re-route
already-fetched results live. See src/ocr_pipeline/api.py's docstring note
for why, and STEPS_TO_DEVELOP.md section 10.
"""

import base64
import os
import tempfile

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ocr_pipeline.cleanup import CLEANUP_SYSTEM_PROMPT, build_numbered_input, parse_numbered_lines
from ocr_pipeline.frame_sampler import sample_frames
from ocr_pipeline.utils import crop_polygon

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

PADDLEOCR_URL = os.environ["PADDLEOCR_URL"]
GOT_OCR2_URL = os.environ["GOT_OCR2_URL"]
LLM_CLEANUP_URL = os.environ["LLM_CLEANUP_URL"]

FRAME_SAMPLER_FPS = float(os.environ.get("FRAME_SAMPLER_FPS", "1"))
FRAME_SAMPLER_DEDUPE_THRESHOLD = float(os.environ.get("FRAME_SAMPLER_DEDUPE_THRESHOLD", "0.97"))

client = httpx.Client(timeout=60.0)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("OCR_CORS_ORIGINS", "*").split(","),
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


def _encode_png(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _call_paddleocr(image: np.ndarray) -> list[dict]:
    response = client.post(PADDLEOCR_URL, json={"instances": [{"image_b64": _encode_png(image)}]})
    response.raise_for_status()
    return response.json()["predictions"][0]["detections"]


def _call_got_ocr2(crops: list[np.ndarray]) -> list[str]:
    if not crops:
        return []
    instances = [{"image_b64": _encode_png(crop)} for crop in crops]
    response = client.post(GOT_OCR2_URL, json={"instances": instances})
    response.raise_for_status()
    return response.json()["predictions"]


def _call_cleanup(lines: list[str]) -> list[str]:
    if not lines or not any(line.strip() for line in lines):
        return list(lines)
    payload = {
        "model": "llm-cleanup",
        "messages": [
            {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
            {"role": "user", "content": build_numbered_input(lines)},
        ],
        "temperature": 0,
    }
    response = client.post(LLM_CLEANUP_URL, json=payload)
    response.raise_for_status()
    output = response.json()["choices"][0]["message"]["content"]
    parsed = parse_numbered_lines(output, expected=len(lines))
    return parsed if parsed is not None else list(lines)


def _detections_for_frame(
    image: np.ndarray, prefix: str, relative_timestamp: float | None = None
) -> list[RawDetectionOut]:
    height, width = image.shape[:2]
    raw_detections = _call_paddleocr(image)

    candidate_texts = [d["text"] for d in raw_detections]
    crops = [crop_polygon(image, np.asarray(d["poly"], dtype=np.float32)) for d in raw_detections]
    specialist_texts = _call_got_ocr2(crops)

    cleaned_candidates = _call_cleanup(candidate_texts)
    cleaned_specialists = _call_cleanup(specialist_texts)

    out = []
    for i, det in enumerate(raw_detections):
        box = det["box"]
        out.append(
            RawDetectionOut(
                id=f"{prefix}-{i}",
                x=box["x1"] / width,
                y=box["y1"] / height,
                w=(box["x2"] - box["x1"]) / width,
                h=(box["y2"] - box["y1"]) / height,
                candidateText=candidate_texts[i],
                specialistText=specialist_texts[i] if i < len(specialist_texts) else None,
                cleanedCandidateText=cleaned_candidates[i] if i < len(cleaned_candidates) else None,
                cleanedSpecialistText=cleaned_specialists[i] if i < len(cleaned_specialists) else None,
                rawConfidence=det["score"],
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

    detections: list[RawDetectionOut] = []
    for frame_index, (frame, timestamp) in enumerate(
        sample_frames(path, fps=FRAME_SAMPLER_FPS, dedupe_similarity_threshold=FRAME_SAMPLER_DEDUPE_THRESHOLD)
    ):
        relative_timestamp = (timestamp / duration) if duration else 0.0
        detections.extend(_detections_for_frame(frame, f"f{frame_index}", relative_timestamp))
    return detections


@app.post("/api/ocr", response_model=OcrResponse)
async def run_ocr(file: UploadFile = File(...)) -> OcrResponse:
    import os as _os
    import tempfile

    suffix = _os.path.splitext(file.filename or "")[1].lower()
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
        _os.remove(tmp_path)

    return OcrResponse(detections=detections)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
