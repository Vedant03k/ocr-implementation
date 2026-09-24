# AWS EKS + KServe deployment

Everything needed to run the OCR pipeline as three independently-scaled KServe `InferenceService`s (PaddleOCR, GOT-OCR2.0, the LLM cleanup stage) plus a lightweight orchestrator, instead of the local dev setup's single process holding all three models in memory (`src/ocr_pipeline/api.py`).

## Current deployment status (2026-09-24)

A real cluster exists and is partially live. Quick reference:

| Resource | Value |
| --- | --- |
| EKS cluster | `ocr-pipeline`, region `us-east-1` |
| AWS account | `711446557912` |
| S3 model bucket | `s3://ocr-pipeline-models-711446557912/models/{got_ocr2,llm_cleanup}/v1/` |
| ECR repos | `711446557912.dkr.ecr.us-east-1.amazonaws.com/ocr-pipeline/{paddleocr,got-ocr2,orchestrator}` |
| Orchestrator URL | `http://a39c1f170492e470b8d22a14810ebd5d-1634070530.us-east-1.elb.amazonaws.com` |

**KServe was installed in RawDeployment mode, not Serverless.** The manifests here were originally written assuming Knative-style autoscaling (`scaleMetric: concurrency`), but installing Istio + Knative just for that turned out to be more moving parts / resource use than this cluster's 2x `m5.large` CPU nodegroup comfortably supports. RawDeployment mode (cert-manager + KServe controller only, plain HPA autoscaling) is what's actually running. Consequence: **the GPU services do not scale to zero** — RawDeployment's scale-from-zero (KEDA-based) is unverified/undocumented as of KServe v0.20, so `got-ocr2` and `llm-cleanup` both run `minReplicas: 1`. Since this is low-volume test/demo traffic where cold starts would hurt more than they'd save, that trade was made deliberately — see the comments in `kserve/got-ocr2-isvc.yaml`. Scale the GPU nodegroup to 0 manually between sessions to control cost (see below).

Component status:

| Component | Status |
| --- | --- |
| EKS cluster + Cluster Autoscaler + KServe control plane | ✅ Up |
| `paddleocr` InferenceService | ✅ Ready (CPU) |
| `orchestrator` Deployment | ✅ Running (2/2), LoadBalancer Service live |
| `got-ocr2` InferenceService | ⛔ Blocked — see below |
| `llm-cleanup` InferenceService | ⛔ Blocked — see below |

**Blocker: GPU instance quota.** This AWS account's default quota for "Running On-Demand G and VT instances" (the family `g4dn.xlarge` belongs to) was **0 vCPUs** — a new-account default, not a bug in the manifests. An increase request to 8 vCPUs was filed (`aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-DB2E81BA --desired-value 8 --region us-east-1`) and is sitting as an AWS support case (`CASE_OPENED`), not auto-approved. Check status with:
```powershell
aws service-quotas list-requested-service-quota-change-history --service-code ec2 --region us-east-1 --query "RequestedQuotas[?QuotaCode=='L-DB2E81BA']"
```
Once `Status` shows `CASE_CLOSED` and the quota value reflects, the two pending GPU pods (already applied, sitting `Pending`) should schedule on their own once Cluster Autoscaler brings up a `g4dn.xlarge` node — no re-apply needed.

**Bugs found and fixed while bringing this up for real** (all already applied in the files below, kept here as a record since they weren't visible from manifests alone):
- `docker/paddleocr.Dockerfile` was missing `libgl1`, `libglib2.0-0`, `libgomp1` — opencv/paddlepaddle's compiled deps that `python:3.11-slim` doesn't ship. Caused an immediate `ImportError: libGL.so.1` crash loop.
- All three custom-container images needed `imagePullPolicy: Always` added explicitly. Without it, `IfNotPresent` (which the KServe/K8s pod spec used) meant a node that had already pulled a broken `:latest` image kept reusing it after rebuilds/pushes — `kubectl rollout restart` alone did not fix this.
- `docker login`/`aws ecr get-login-password | docker login` returned a `400 Bad Request` on this machine regardless of encoding fixes — root cause not fully identified (possibly a Docker Desktop / AWS CLI version interaction). Worked around by using the `docker-credential-ecr-login` credential helper (`credHelpers` in `~/.docker/config.json`) instead of `docker login` token auth.
- Service DNS names in `k8s/orchestrator.yaml` needed to be `<name>-predictor.ocr-pipeline.svc.cluster.local`, not `<name>.ocr-pipeline.svc.cluster.local` — KServe's RawDeployment mode suffixes the generated Service with `-predictor`.

**To resume from here:** once the quota case clears, verify with `kubectl -n ocr-pipeline get inferenceservices`, then follow step 6 below (point the frontend at the orchestrator URL above, tighten `OCR_CORS_ORIGINS` in `k8s/orchestrator.yaml` off of `"*"`).

**To pause and save cost:** scale the GPU nodegroup to 0 when not actively testing —
```powershell
eksctl scale nodegroup --cluster=ocr-pipeline --region=us-east-1 --name=gpu-workers --nodes=0 --nodes-min=0 --nodes-max=3
```
Cluster Autoscaler will bring it back to 1 automatically the next time a GPU pod needs scheduling.

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
