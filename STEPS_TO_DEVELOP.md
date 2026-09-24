# Steps to be developed

Setup log and commands for standing up the OCR pipeline environment, models and folder structure. Run from the repo root in PowerShell.

## 1. Branching

All work happens on `dev-ved`. Nothing gets pushed to `main` until explicitly said so.

```powershell
git checkout -b dev-ved
```

## 2. Python environment

PaddlePaddle does not yet reliably support Python 3.13, so the venv uses 3.11.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
```

If that combined install is troublesome, install in two passes instead (what was actually run to build this environment):

```powershell
pip install paddlepaddle==3.3.1 paddleocr==3.7.0 pydantic opencv-python pyyaml
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate huggingface_hub tiktoken verovio
```

Adjust the `cu124` index URL to match your CUDA driver if different (`nvidia-smi` shows the driver's max supported CUDA version). `paddlepaddle` (not `-gpu`) is a CPU-only build — that's intentional, PaddleOCR is the light model; GOT-OCR2.0 gets the GPU torch build since it's the heavy one.

**PaddleOCR 2.x vs 3.x:** we initially tried downgrading to `paddleocr==2.10.0` (last pre-3.0 release) because its `det_model_dir`/`rec_model_dir` download straight into a folder you choose, which looked simpler for an S3/Kubeflow deploy. That turned out to be a dead end: PaddleOCR 2.x's exported inference models are incompatible with PaddlePaddle 3.3.1's backend (`OneDnnContext ... Filter not found`), and downgrading PaddlePaddle too would have cascaded into more version-matching risk. We reverted to 3.7.0 and solved the portability concern a different way — see step 4.

## 4. Download model weights

```powershell
python scripts/download_models.py
```

This pulls:
- PaddleOCR det/rec + orientation/unwarp weights → `models/paddleocr/official_models/`. PaddleOCR 3.x's PaddleX backend normally caches to `~/.paddlex/official_models`, not project-local; `src/ocr_pipeline/__init__.py` sets the `PADDLE_PDX_CACHE_HOME` env var (before paddlex is imported anywhere) to redirect that cache into the repo, so it's a self-contained, S3/container-portable folder instead of something living in the user's home directory.
- GOT-OCR2.0 weights (`stepfun-ai/GOT-OCR-2.0-hf`, the `transformers`-native port) → `models/got_ocr2/`

`models/` is gitignored — weights are never committed.

## 5. Two environment quirks that had to be worked around

Both are handled automatically in code, documented here so they aren't "mysteriously" reintroduced:

- **PaddlePaddle 3.3.1's CPU oneDNN backend crashes on PP-OCRv6 detection** (`NotImplementedError: ConvertPirAttribute2RuntimeAttribute ... pir::ArrayAttribute<pir::DoubleAttribute>`) unless MKLDNN is disabled. `detector.py` always constructs `PaddleOCR(..., enable_mkldnn=False)`.
- **Importing `paddleocr` before `torch` crashes with a native DLL conflict on Windows** (`OSError: [WinError 127] ... shm.dll`) — importing `torch` first avoids it. `src/ocr_pipeline/__init__.py` imports `torch` as its very first import so any submodule importing `paddleocr` afterward is safe, regardless of import order elsewhere.

## 6. Folder hierarchy

```
ocr-implementation/
├── README.md
├── STEPS_TO_DEVELOP.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── architecture.svg
├── config/
│   └── config.yaml
├── models/                    # gitignored, populated by scripts/download_models.py
│   ├── paddleocr/official_models/
│   └── got_ocr2/
├── src/ocr_pipeline/
│   ├── __init__.py            # PADDLE_PDX_CACHE_HOME + torch-first import order
│   ├── main.py                # CLI entrypoint, wires the full pipeline together
│   ├── frame_sampler.py       # video -> sampled frames (fixed FPS + dedupe)
│   ├── detector.py            # PaddleOCR detection + first-pass recognition
│   ├── recognizer_fast.py     # reads PaddleOCR's own recognition result
│   ├── recognizer_specialist.py  # GOT-OCR2.0 (transformers)
│   ├── router.py              # confidence-based escalation to the specialist
│   ├── merge.py                # reading-order sort + final TextBlock list
│   ├── schema.py              # pydantic JSON output schema
│   └── utils.py               # quad-box perspective crop
├── scripts/
│   └── download_models.py
└── tests/
```

## 7. Status

- [x] Repo scaffold (config, src package, scripts, tests dir)
- [x] `dev-ved` branch created
- [x] Dependencies installed
- [x] Model weights downloaded
- [x] Frame sampler implementation (fixed-FPS mode; scene-change mode not implemented — raises `NotImplementedError`)
- [x] PaddleOCR detection + recognition integration
- [x] Confidence-based routing logic
- [x] GOT-OCR2.0 integration for handwriting
- [x] Merge/post-process step
- [x] JSON output schema wired end-to-end
- [x] Smoke-tested end-to-end on a synthetic image, both the PaddleOCR fast path and a forced GOT-OCR2.0 escalation
- [x] UI for uploading images/video and viewing results — `web/` (merged from `dev-amrita`) wired to the real backend (see §10); browser-tested end-to-end, including live threshold re-routing
- [ ] Validated against real handwriting samples (only tested on a synthetic printed-text image so far)

## 10. Web GUI integration

`web/` (merged from `dev-amrita`) originally ran entirely on mock data — a static fixture array plus a client-side simulated progress animation, with no backend call at all. Wired it to the real pipeline:

- **`src/ocr_pipeline/api.py`** — a FastAPI server. Loads `PaddleDetector` and `GotOcrRecognizer` once at startup (not per-request — reloading both models per call would make every upload take 10s+ just for weight loading) and exposes `POST /api/ocr` (multipart file upload, image or video).
- **Deliberate deviation from `router.py`'s real routing logic**: the API always runs GOT-OCR2.0 on every detected box, regardless of confidence, and returns both PaddleOCR's and GOT-OCR2.0's answer for each one. This matches the frontend's existing `RawDetection` contract (`candidateText` + `specialistText` + `rawConfidence`), which was designed so the GUI's threshold slider can re-route already-fetched results live without re-hitting the backend. The real pipeline (`router.py`, used by `main.py`/CLI) only escalates crops below the threshold, as the architecture intends — this GUI-only shortcut trades some extra GOT-OCR2.0 calls for a responsive demo slider. Don't copy this pattern into the Kubeflow inference path.
- **`web/src/lib/ocr-api.ts`** — `fetchDetections()`, POSTs the uploaded file and returns `RawDetection[]`. Reads `NEXT_PUBLIC_OCR_API_URL` (defaults to `http://localhost:8000`).
- **`web/src/hooks/use-ocr-pipeline.ts`** — the "processing" phase now runs the real fetch and the cosmetic stage-progress animation in parallel (`Promise.all`), completing only once the real backend response arrives. On failure, reverts to the upload screen with an error banner instead of leaving the UI stuck.
- **`web/src/components/processing-view.tsx`** — no longer shows a routing-stats preview during processing (it used to read counts off the *unrelated* static mock invoice data, which would have been actively misleading once real uploads were wired in). Real routing stats still appear once results are in (`results-view.tsx`/`ocr-routing.ts`, unchanged).

**Bug found and fixed during browser testing:** `box-detail-panel.tsx` checks `box.timestamp !== undefined` before calling `.toFixed()`, but the real backend serializes an absent timestamp as JSON `null` (Pydantic `Optional[float] = None`), not `undefined` — `null !== undefined` is `true`, so it took the branch and crashed on `null.toFixed()`. The mock fixtures never hit this because they either had a real number or omitted the field entirely. Fixed at the API boundary in `ocr-api.ts`, normalizing `null → undefined` right after parsing the response, so every downstream file's `number | undefined` typing is actually true at runtime.

### Running it

```powershell
# Terminal 1 — backend
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m uvicorn ocr_pipeline.api:app --port 8000

# Terminal 2 — frontend
cd web
npm run dev
```

Open `http://localhost:3000`, upload an image or video, and results come from the real pipeline.

### Known limitations of this integration pass

- Running GOT-OCR2.0 on every detected box (not just escalated ones) is fine for a handful of boxes in a demo image, but doesn't scale to a busy frame or a long video — this is explicitly a GUI-only convenience, not how the Kubeflow-bound inference path should work.
- The cosmetic stage-progress animation (frame-sampling/detection/recognition/merging bars) runs on a fixed ~3–4s timer unrelated to real backend latency; for a slow request the bars will sit at 100% while still waiting on the actual response. No real progress streaming (SSE/WebSocket) is wired up yet.
- Only browser-tested against a single synthetic printed-text image (both the fast path and forced escalation) — not yet tried against a real photo, a real video, or real handwriting.

## 8. Running the pipeline

Run from the repo root, with `src` on the Python path (since `ocr_pipeline` lives under `src/`):

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m ocr_pipeline.main path\to\input.jpg --output result.json
```

`--config` defaults to `config/config.yaml`; pass a different path with `--config path\to\other.yaml`.

## 9. Known design trade-off: detector/recognizer split

The README's architecture diagram shows detection and fast-path recognition as separate stages. In practice, PaddleOCR 3.x/PaddleX has no standalone detection-only call at the level this pipeline uses — one `predict()` call runs detection and first-pass recognition together. `detector.py` runs that fused call; `recognizer_fast.py` reads the recognition it already produced instead of invoking PaddleOCR's recognizer a second time (which would duplicate work for no benefit). The module boundary from the architecture is kept, but both are backed by the same underlying call — documented in each file's docstring.
