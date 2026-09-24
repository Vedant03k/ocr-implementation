from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class TextBlock(BaseModel):
    text: str
    box: BoundingBox
    confidence: float
    source: str  # "paddleocr" | "got_ocr2"
    frame_timestamp: float | None = None
    cleaned_text: str | None = None  # LLM spelling/OCR-error correction, when enabled


class OCRResult(BaseModel):
    blocks: list[TextBlock]
