# OCR implementation

OCR pipeline for images and video that combines [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for fast text detection/recognition with [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0) as a specialist recognizer for messy/cursive handwriting.

## Why two models

PaddleOCR's detector (PP-OCRv4, DBNet) is fast and mature, and its recognizer handles printed text well — but it's noticeably weaker on messy cursive handwriting. GOT-OCR2.0 is a newer unified OCR model that performs much better on handwriting, at higher compute cost. Rather than running the heavy model on every crop, a confidence router escalates only the crops PaddleOCR is unsure about.

## Architecture

![OCR pipeline architecture](docs/architecture.svg)

1. **Input** — an image or a video file.
2. **Frame sampler** — for video, samples frames (fixed FPS or scene-change detection) and dedupes near-identical ones; images pass through unchanged.
3. **PaddleOCR detector** — PP-OCRv4/v5 (DBNet) locates text-line/word bounding boxes.
4. **Confidence router** — runs the PaddleOCR recognizer on each crop; boxes below a confidence threshold (or flagged as handwriting) are escalated.
5. **PaddleOCR recognizer** (fast path) — handles confidently printed text.
6. **GOT-OCR2.0** (specialist path) — handles messy handwriting the fast path is unsure about.
7. **Merge & reorder** — combines results from both paths, restores reading order, optional LLM cleanup pass for obvious OCR typos.
8. **Output** — text + bounding boxes (+ timestamps for video frames) as JSON.

## Status

Architecture design stage — implementation in progress.

## Roadmap

- [ ] Frame sampler for video input
- [ ] PaddleOCR detection + recognition integration
- [ ] Confidence-based routing logic
- [ ] GOT-OCR2.0 integration for handwriting
- [ ] Merge/post-process step
- [ ] JSON output schema
- [ ] UI for uploading images/video and viewing results
