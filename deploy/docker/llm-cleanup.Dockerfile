# Build from the repo root: docker build -f deploy/docker/llm-cleanup.Dockerfile .
#
# CPU build, custom container — not KServe's built-in HuggingFace+vLLM
# ServingRuntime (see deploy/kserve/llm-cleanup-isvc.yaml for why: vLLM has
# no practical CPU path, and this account has no approved GPU quota).
FROM python:3.11-slim

WORKDIR /app

# torch (CPU build) needs libgomp at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements/llm-cleanup.txt .
RUN pip install --no-cache-dir -r llm-cleanup.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# Weights are NOT baked into the image (~3GB) — KServe's storage-initializer
# pulls them from storageUri (S3) into /mnt/models at pod startup instead.
# See deploy/kserve/llm-cleanup-isvc.yaml.
COPY src/ocr_pipeline/ ./src/ocr_pipeline/
COPY deploy/serving/llm_cleanup_server.py .
ENV PYTHONPATH=/app/src
ENV LLM_CLEANUP_MODEL_DIR=/mnt/models
ENV LLM_CLEANUP_DEVICE=cpu

EXPOSE 8080
CMD ["python3", "llm_cleanup_server.py"]
