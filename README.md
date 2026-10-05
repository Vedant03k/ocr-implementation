# OCR implementation

OCR pipeline for images and video that combines [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for fast text detection/recognition with [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) as a specialist recognizer for messy/cursive handwriting, plus a word-level corrector that fixes OCR misreads.

## Why two OCR models

PaddleOCR's detector (PP-OCRv6, DBNet) is fast and mature, and its recognizer handles printed text well — but it's noticeably weaker on messy cursive handwriting. PaddleOCR-VL-1.5, a 0.9B vision-language model, reads handwriting far better, at higher compute cost. Rather than running it on every crop, a confidence router escalates only the crops PaddleOCR is unsure about.

The specialist is chosen by `specialist.engine` in `config/config.yaml`. GOT-OCR2.0 (`got_ocr2`) is still supported, but it lost the comparison below.

## Why a word-level correction stage

Even a correctly-routed recognition can still misread individual characters ("olease", "oudget", "fir our"). `word_corrector.py` fixes these one word at a time and leaves every other word untouched. It works on the whole document, not line by line, because neighbouring lines give context.
- **Non-dictionary words:** gets candidate spellings from SymSpell and ranks them by how often each appears next to its neighbours.
- **Real-word misreads:** changes a real word ("fir") only on lines PaddleOCR was unsure about, and only when the replacement strongly fits its neighbours ("for our").
- **Dropped spaces:** splits run-together lines into words.
- **Left alone:** names, acronyms, numbers and abbreviations ("recvd").

It's additive, not authoritative. The raw OCR text is always kept alongside the correction, and the GUI has a toggle to show the raw read.

It replaced a whole-document LLM rewrite (Qwen2.5-1.5B-Instruct, still available as `llm_cleanup` in `config/config.yaml`, off by default). On the benchmark below, the LLM made accuracy worse: it capitalised line starts, swapped in synonyms ("rush" → "expedite") and invented words ("oudget" → "Outgotten"). It also pushed GPU memory past 4 GB. Text-only correction has a limit: a badly misread word ("infrasture", 4 letters off) can't be recovered from text, so those need an image re-read.

## Benchmark

`scripts/benchmark.py` measures per-stage latency, GPU memory and character/word error rate on any folder of images with same-named `.txt` transcriptions. `scripts/make_synthetic_benchmark.py` generates a starter set from handwriting-style fonts. That set is much cleaner than real cursive, so decisions still need confirming on real images in `benchmark/real/`. `benchmark/` is git-ignored.

Measured on 10 synthetic 3–6 line images, on a laptop with an RTX 3050 (4 GB):

| | Before | After |
|---|---|---|
| Mean time per image | 56.6s | 7.5s |
| Character error rate of the text shown in the GUI | 3.19% | 0.06% |
| Word error rate | 13.65% | 0.34% |
| Peak GPU memory | 7.2 GB (spilling into system RAM) | 4.07 GB |

What changed:
- GOT-OCR2.0 only runs on lines below the threshold the GUI sends with each upload (it used to run on every line).
- One correction pass instead of two.
- The word corrector replaces the LLM.
- PaddleOCR's document unwarping is off: on flat images it distorted letters and cost about 1.3s.

Remaining time is PaddleOCR on CPU. GOT-OCR2.0 takes about 2s per escalated line, can't batch on a 4 GB GPU, and was less accurate than PaddleOCR on the synthetic set.

### Real handwriting (GNHK)

The synthetic set is too clean to choose models with, so the numbers below use the test split of [GNHK](https://github.com/GoodNotes/GNHK-dataset): phone photos of real English handwriting (Lee et al., ICDAR 2021, CC BY 4.0). `scripts/prepare_gnhk.py` turns it into line crops and page images with transcriptions.

**Specialist models**, on 300 line crops (`scripts/eval_recognizers.py`), RTX 3050 4 GB:

| Model | Char error | Word error | Time per line | GPU memory |
|---|---|---|---|---|
| PP-OCRv6 recognizer (fast path, CPU) | 33.5% | 60.0% | 0.18s | — |
| GOT-OCR2.0 (previous specialist) | 21.0% | 46.3% | 1.88s | 3.4 GB |
| TrOCR-large-handwritten | 18.7% | 55.3% | 0.16s | 1.1 GB |
| **PaddleOCR-VL-1.5 (current specialist)** | **10.2%** | 28.6% | 1.11s | 1.8 GB |
| Qwen3-VL-2B-Instruct, 4-bit | 12.1% | 28.3% | 0.66s | 1.7 GB |

Qwen3-VL was close, but it sometimes added words that weren't in the image. TrOCR drops most punctuation. Batching PaddleOCR-VL gave no speedup on this GPU.

**Whole pages**, on 58 photos (`scripts/benchmark.py --threshold 1 --sweep ...`). Word F1 counts matched words regardless of order: page-level CER also penalises reading order, so it moves much less.

| Lines re-read by PaddleOCR-VL | Threshold | Word F1 | Char error | Time per page |
|---|---|---|---|---|
| none (PaddleOCR only) | 0 | 0.737 | 29.7% | 17.5s |
| 11% | 0.85 | 0.773 | 27.5% | 19.3s |
| **55% (default)** | **0.97** | **0.812** | **27.1%** | **26.8s** |
| 96% | 1.0 | 0.818 | 27.2% | 33.6s |

Changes these measurements led to:
- **The detector sees photos at 1280px on the long side** (`paddleocr.det_limit_side_len`). At full 4000×3000 it took 63s per page and split lines into fragments (char error 50.7% vs 37.3% on 8 pages).
- **Output guards on PaddleOCR-VL:**
  - Runaway repeats ("Ave Ave Ave …") are collapsed, and output length is capped by the crop's shape.
  - Prose answers on blank crops ("The image is too blurry…") and formula-style misreads ("∆V-11-38" for "LV-11-38", LaTeX) are rejected. The line keeps PaddleOCR's text; this happened on 25 of 908 lines.
- **The default threshold is now 0.97.**

The word corrector changes real-handwriting accuracy by less than 0.3 points either way. Misreads on real handwriting are too far from the right word for text-only correction to fix.

What limits accuracy now is detection: on some pages PaddleOCR misses or merges whole lines, and no recognizer can fix that.

## Architecture

![OCR pipeline architecture](docs/architecture.svg)

1. **Input** — an image or a video file.
2. **Frame sampler** — for video, samples frames (fixed FPS) and dedupes near-identical ones; images pass through unchanged. (Scene-change sampling mode is not yet implemented.)
3. **PaddleOCR detector** — PP-OCRv6 (DBNet) locates text-line bounding boxes.
4. **Confidence router** — runs the PaddleOCR recognizer on each crop; boxes below a confidence threshold are escalated.
5. **PaddleOCR recognizer** (fast path) — handles confidently printed text.
6. **PaddleOCR-VL-1.5** (specialist path, GPU) — re-reads lines the fast path is unsure about, mainly messy handwriting.
7. **Merge & reorder** — combines results from both paths, restores reading order.
8. **Word correction** — fixes misread words across the merged text, with the original OCR text kept alongside the correction.
9. **Output** — text + bounding boxes (+ timestamps for video frames) as JSON, plus a web GUI for uploading files and inspecting results.

## Web app architecture

![Web app architecture](docs/web-architecture.svg)

The diagram above is the OCR pipeline itself (a single request's processing steps). Around it: a Next.js browser GUI (`web/`) uploads a file to a FastAPI backend (`src/ocr_pipeline/api.py`), which loads the models once at startup (not per-request — reloading them per call would add 10s+ to every upload) and runs the pipeline, returning JSON that the browser renders as a bounding-box overlay and a recognized-text panel.

## Status

Implemented and browser-tested end-to-end, including against a real photo of handwritten notes. See [STEPS_TO_DEVELOP.md](STEPS_TO_DEVELOP.md) for the full setup log, environment gotchas found along the way, and known limitations.

## Roadmap

- [x] Frame sampler for video input (fixed-FPS mode)
- [x] PaddleOCR detection + recognition integration
- [x] Confidence-based routing logic
- [x] GOT-OCR2.0 integration for handwriting
- [x] Specialist chosen on real handwriting (GNHK): PaddleOCR-VL-1.5 replaces GOT-OCR2.0
- [ ] Better line detection on real photos (current accuracy ceiling)
- [x] Merge/post-process step
- [x] JSON output schema
- [x] LLM cleanup pass for OCR typos
- [x] UI for uploading images/video and viewing results
- [ ] Scene-change frame sampling mode
- [x] AWS EKS + KServe deployment applied to a real cluster — all four components (PaddleOCR, GOT-OCR2.0, LLM cleanup, orchestrator) live and verified end-to-end; GOT-OCR2.0/LLM cleanup run on CPU pending an AWS GPU-quota approval (see [deploy/README.md](deploy/README.md#current-deployment-status-2026-09-24) for live status, latency notes, and how to restore GPU)
