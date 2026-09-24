# AWS EKS + KServe deployment

Everything needed to run the OCR pipeline as three independently-scaled KServe `InferenceService`s (PaddleOCR, GOT-OCR2.0, the LLM cleanup stage) plus a lightweight orchestrator, instead of the local dev setup's single process holding all three models in memory (`src/ocr_pipeline/api.py`).

## Current deployment status (2026-09-24)

**Fully live, end-to-end, on CPU.** A real image was posted to the orchestrator's public URL and came back with correct PaddleOCR + GOT-OCR2.0 + LLM-cleanup output. Quick reference:

| Resource | Value |
| --- | --- |
| EKS cluster | `ocr-pipeline`, region `us-east-1` |
| AWS account | `711446557912` |
| S3 model bucket | `s3://ocr-pipeline-models-711446557912/models/{got_ocr2,llm_cleanup}/v1/` |
| ECR repos | `711446557912.dkr.ecr.us-east-1.amazonaws.com/ocr-pipeline/{paddleocr,got-ocr2,llm-cleanup,orchestrator}` |
| Orchestrator URL | `http://a39c1f170492e470b8d22a14810ebd5d-1634070530.us-east-1.elb.amazonaws.com` |

**KServe was installed in RawDeployment mode, not Serverless.** The manifests here were originally written assuming Knative-style autoscaling (`scaleMetric: concurrency`), but installing Istio + Knative just for that turned out to be more moving parts / resource use than this cluster's CPU nodegroup comfortably supports. RawDeployment mode (cert-manager + KServe controller only, plain HPA autoscaling) is what's actually running.

**GPU quota was never approved — `got-ocr2` and `llm-cleanup` were switched to run on CPU instead of waiting.** This account's quota for "Running On-Demand G and VT instances" (`g4dn.xlarge`'s family) is **0 vCPUs across every region and for spot too** (checked us-east-1, us-east-2, us-west-2, both on-demand `L-DB2E81BA` and spot `L-3819A6DF`) — a new-account default with an increase request stuck in manual AWS review (`CASE_OPENED`, filed via `aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-DB2E81BA --desired-value 8 --region us-east-1`). Rather than block the whole deployment on that, both GPU services were re-pointed at the existing CPU nodegroup:
- `got-ocr2`: same custom container, just `nodeSelector: {workload: cpu}` instead of the GPU node group, no `nvidia.com/gpu` resources. `GotOcrRecognizer` already fell back to CPU automatically when CUDA isn't available, so no application code changed.
- `llm-cleanup`: this one *did* need a real change — KServe's built-in HuggingFace+vLLM ServingRuntime (the original plan) has no practical CPU inference path, so it was replaced with a custom container (`deploy/serving/llm_cleanup_server.py`) running plain `transformers.generate()` on CPU via `ocr_pipeline.cleanup.TextCleaner` — the exact same class the local dev pipeline uses, so cleanup behavior is identical. The orchestrator's `_call_cleanup` was updated to call this new predict-protocol endpoint (`POST {"instances":[{"lines":[...]}]}`) instead of an OpenAI-style chat-completions endpoint.

**Cost of the CPU fallback: latency.** A single-line smoke test took **~148s round-trip** (PaddleOCR + GOT-OCR2.0 + 2x LLM cleanup calls, all CPU, run sequentially by the orchestrator). Multi-line documents will take proportionally longer. Fine for the demo/low-volume use case this was built for; revisit (see "To restore GPU" below) before any real usage.

Component status — all four Ready/Running:

| Component | Status |
| --- | --- |
| EKS cluster + Cluster Autoscaler + KServe control plane | ✅ Up |
| `paddleocr` InferenceService | ✅ Ready (CPU, as always) |
| `got-ocr2` InferenceService | ✅ Ready (CPU fallback) |
| `llm-cleanup` InferenceService | ✅ Ready (CPU fallback, custom container) |
| `orchestrator` Deployment | ✅ Running (2/2), LoadBalancer Service live, verified end-to-end |

**Bugs found and fixed while bringing this up for real** (all already applied in the files below, kept here as a record since they weren't visible from manifests alone):
- `docker/paddleocr.Dockerfile` was missing `libgl1`, `libglib2.0-0`, `libgomp1` — opencv/paddlepaddle's compiled deps that `python:3.11-slim` doesn't ship. Caused an immediate `ImportError: libGL.so.1` crash loop.
- All custom-container images needed `imagePullPolicy: Always` added explicitly. Without it, `IfNotPresent` (which the KServe/K8s pod spec used) meant a node that had already pulled a broken `:latest` image kept reusing it after rebuilds/pushes — `kubectl rollout restart` alone did not fix this.
- `docker login`/`aws ecr get-login-password | docker login` returned a `400 Bad Request` on this machine regardless of encoding fixes — root cause not fully identified (possibly a Docker Desktop / AWS CLI version interaction). Worked around by using the `docker-credential-ecr-login` credential helper (`credHelpers` in `~/.docker/config.json`) instead of `docker login` token auth.
- Service DNS names in `k8s/orchestrator.yaml` needed to be `<name>-predictor.ocr-pipeline.svc.cluster.local`, not `<name>.ocr-pipeline.svc.cluster.local` — KServe's RawDeployment mode suffixes the generated Service with `-predictor`.
- The `cpu-workers` node IAM role had **no S3 permissions at all** — the storage-initializer init containers for `got-ocr2`/`llm-cleanup` got `403 Forbidden` pulling model weights. Fixed with an inline policy (`ocr-model-bucket-read`) on the node role granting `s3:GetObject`/`s3:ListBucket` on the model bucket. (This gap existed from the start; it just was never hit before because those pods never got scheduled anywhere until the CPU fallback.)
- `deploy/requirements/got-ocr2.txt` was missing `pillow` — `recognizer_specialist.py` imports `PIL.Image` directly, not just transitively through `transformers`/`accelerate`. Caused a `ModuleNotFoundError` crash loop in the main container (separate from the init-container S3 issue above).
- **The orchestrator blocked its own event loop.** `deploy/orchestrator/app.py` called the downstream KServe services with a *synchronous* `httpx.Client` from inside `async def` endpoints — fine with GPU-speed inference, but with CPU-speed inference the blocking call held the event loop long enough that the `/healthz` readiness probe missed its window, Kubernetes pulled the pod out of the Service, and in-flight requests died with "empty reply from server". Fixed by switching to `httpx.AsyncClient` and `await`-ing every downstream call end to end (`_call_paddleocr`/`_call_got_ocr2`/`_call_cleanup`/`_detections_for_frame`/`_run_video` are all `async def` now).
- **The Classic ELB's default 60s idle timeout** was killing any request that took longer than a minute — which, on CPU, is basically all of them. `k8s/orchestrator.yaml`'s Service now carries `service.beta.kubernetes.io/aws-load-balancer-connection-idle-timeout: "300"`.

**To restore GPU (once quota clears):** `got-ocr2-isvc.yaml` and `llm-cleanup-isvc.yaml` both carry comments marking exactly what to revert (nodeSelector/tolerations/resources back to GPU, `llm-cleanup` back to the vLLM ServingRuntime if desired for real batching throughput) — see git history for the original GPU versions of both files and `docker/got-ocr2.Dockerfile`. Check quota status with:
```powershell
aws service-quotas list-requested-service-quota-change-history --service-code ec2 --region us-east-1 --query "RequestedQuotas[?QuotaCode=='L-DB2E81BA']"
```

**To pause and save cost:** nothing GPU is running right now (the fallback avoided the GPU nodegroup entirely), so there's no GPU cost to pause. The CPU nodegroup (`m5.large` x2-4, autoscaled) is the only compute running.

---

**Original status note (superseded by the table above, kept for context):** these manifests followed KServe's documented patterns as of writing but hadn't been validated against a live EKS cluster. Validate with `kubectl apply --dry-run=server` against your actual KServe version before applying for real, especially `kserve/llm-cleanup-isvc.yaml` (the HuggingFace+vLLM runtime API has evolved across KServe releases).

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
