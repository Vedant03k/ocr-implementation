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

Fast path (PaddleOCR, CPU is fine — it's the light model):

```powershell
pip install paddlepaddle paddleocr pydantic opencv-python pyyaml
```

Specialist path (GOT-OCR2.0 — GPU build of torch, since this is the heavy model):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate huggingface_hub tiktoken verovio
```

Adjust the `cu124` index URL to match your CUDA driver if different (`nvidia-smi` shows the driver's max supported CUDA version).

## 4. Download model weights

```powershell
python scripts/download_models.py
```

This pulls:
- PaddleOCR det/rec weights → PaddleOCR 3.x uses a PaddleX backend that manages its own cache at `~/.paddlex/official_models`, not an in-repo directory (its `det_model_dir`/`rec_model_dir` params load an *existing* local model rather than choosing a download target)
- GOT-OCR2.0 weights (`stepfun-ai/GOT-OCR-2.0-hf`, the `transformers`-native port) → `models/got_ocr2`

`models/` is gitignored — weights are never committed.

## 5. Folder hierarchy

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
│   ├── paddleocr/{det,rec}/
│   └── got_ocr2/
├── src/ocr_pipeline/
│   ├── main.py
│   ├── frame_sampler.py
│   ├── detector.py
│   ├── router.py
│   ├── recognizer_fast.py
│   ├── recognizer_specialist.py
│   ├── merge.py
│   └── schema.py
├── scripts/
│   └── download_models.py
└── tests/
```

## 6. Status

- [x] Repo scaffold (config, src package stubs, scripts, tests dir)
- [x] `dev-ved` branch created
- [x] Dependencies installed
- [x] Model weights downloaded
- [ ] Frame sampler implementation
- [ ] PaddleOCR detection + recognition integration
- [ ] Confidence-based routing logic
- [ ] GOT-OCR2.0 integration for handwriting
- [ ] Merge/post-process step
- [ ] JSON output schema wired end-to-end
- [ ] UI for uploading images/video and viewing results

## 7. Running the pipeline (once implemented)

```powershell
python -m ocr_pipeline.main path\to\input.jpg --output result.json
```
