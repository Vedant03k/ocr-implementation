# AWS EKS + KServe deployment

Everything needed to run the OCR pipeline as three independently-scaled KServe `InferenceService`s (PaddleOCR, GOT-OCR2.0, the LLM cleanup stage) plus a lightweight orchestrator, instead of the local dev setup's single process holding all three models in memory (`src/ocr_pipeline/api.py`).

**Status: written, not yet applied to a real cluster.** These manifests follow KServe's documented patterns as of writing, but haven't been validated against a live EKS cluster — no cluster exists to test against from this repo. Validate with `kubectl apply --dry-run=server` against your actual KServe version before applying for real, especially `kserve/llm-cleanup-isvc.yaml` (the HuggingFace+vLLM runtime API has evolved across KServe releases).

## Why three separate services instead of one process

The local dev backend (`api.py`) loads all three models into one Python process — simple, but it means every replica needs GPU memory for GOT-OCR2.0 *and* the cleanup LLM *and* CPU for PaddleOCR, and you scale all three together even though they have completely different resource needs and request volumes. Splitting them lets PaddleOCR (cheap, CPU) scale to many replicas independently of GOT-OCR2.0 and the cleanup LLM (expensive, GPU, usually idle), and lets the GPU services scale to zero when nothing's using them.

## What's here

```
deploy/
├── serving/
│   ├── paddleocr_server.py   # KServe custom predictor, wraps ocr_pipeline.detector.PaddleDetector
│   └── got_ocr2_server.py    # KServe custom predictor, wraps ocr_pipeline.recognizer_specialist.GotOcrRecognizer
├── orchestrator/
│   └── app.py                # FastAPI app, same /api/ocr contract as src/ocr_pipeline/api.py, calls the 3 services over HTTP
├── docker/
│   ├── paddleocr.Dockerfile
│   ├── got-ocr2.Dockerfile
│   └── orchestrator.Dockerfile
├── requirements/              # per-service deps — deliberately not one shared requirements.txt,
│   │                           # so e.g. the orchestrator image never pulls in torch/paddlepaddle
│   ├── paddleocr.txt
│   ├── got-ocr2.txt
│   └── orchestrator.txt
├── kserve/
│   ├── paddleocr-isvc.yaml    # custom predictor, CPU, weights baked into image
│   ├── got-ocr2-isvc.yaml     # custom predictor, GPU, weights pulled from S3 at pod startup
│   └── llm-cleanup-isvc.yaml  # KServe's built-in HuggingFace runtime, vLLM backend
├── k8s/
│   └── orchestrator.yaml      # plain Deployment + Service (not a KServe model)
└── eks/
    └── cluster.yaml            # eksctl cluster config: CPU + GPU (scale-to-zero) node groups
```

The LLM cleanup service is *not* a custom container the way the other two are — it uses KServe's built-in `huggingface` model format with `--backend vllm`, which is the actual point of moving off raw `transformers.generate()` (the local dev setup's approach): continuous batching and paged attention instead of one request at a time.

## Order of operations

1. **Create the cluster**
   ```powershell
   eksctl create cluster -f deploy/eks/cluster.yaml
   ```
   Then install the NVIDIA device plugin, Cluster Autoscaler, and KServe itself — see the comments at the bottom of `deploy/eks/cluster.yaml` for links. None of that is covered by the cluster config alone.

2. **Push model weights to S3**
   ```powershell
   aws s3 sync models/got_ocr2/    s3://<BUCKET>/models/got_ocr2/
   aws s3 sync models/llm_cleanup/ s3://<BUCKET>/models/llm_cleanup/
   ```
   PaddleOCR's weights are baked into its image instead (small, and it's a CPU service — no storage-initializer needed). Version the S3 path (e.g. `models/got_ocr2/v1/`) rather than overwriting in place, so a redeploy doesn't silently pick up different weights than what was tested.

3. **Build and push the three images to ECR**
   ```powershell
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <ECR_REPO>

   docker build -f deploy/docker/paddleocr.Dockerfile    -t <ECR_REPO>/paddleocr:latest    .
   docker build -f deploy/docker/got-ocr2.Dockerfile      -t <ECR_REPO>/got-ocr2:latest      .
   docker build -f deploy/docker/orchestrator.Dockerfile  -t <ECR_REPO>/orchestrator:latest  .

   docker push <ECR_REPO>/paddleocr:latest
   docker push <ECR_REPO>/got-ocr2:latest
   docker push <ECR_REPO>/orchestrator:latest
   ```
   All three `docker build` commands run from the **repo root** (not `deploy/`), since the images need to `COPY src/ocr_pipeline/`.

4. **Fill in the placeholders** in `deploy/kserve/*.yaml` and `deploy/k8s/orchestrator.yaml`: `<ECR_REPO>`, `<BUCKET>`, and the Vercel frontend URL for CORS.

5. **Apply the manifests**
   ```powershell
   kubectl create namespace ocr-pipeline
   kubectl apply -f deploy/kserve/paddleocr-isvc.yaml
   kubectl apply -f deploy/kserve/got-ocr2-isvc.yaml
   kubectl apply -f deploy/kserve/llm-cleanup-isvc.yaml
   kubectl apply -f deploy/k8s/orchestrator.yaml
   ```

6. **Point the frontend at it** — set `NEXT_PUBLIC_OCR_API_URL` in Vercel's project settings to the orchestrator Service's external address (`kubectl get svc orchestrator -n ocr-pipeline` once it's up), and make sure `OCR_CORS_ORIGINS` in `deploy/k8s/orchestrator.yaml` includes the actual Vercel URL.

## Local dev is unaffected

`src/ocr_pipeline/api.py` (single-process, all three models in memory) is still what `npm run dev` + `uvicorn ocr_pipeline.api:app` uses locally — nothing here changes that. The two share the same underlying `ocr_pipeline` modules (`detector.py`, `recognizer_specialist.py`, `cleanup.py`'s prompt/parsing helpers) so behavior stays consistent between local dev and the deployed version.
