# Build from the repo root: docker build -f deploy/docker/got-ocr2.Dockerfile .
#
# CPU build. Originally targeted a GPU base image (nvidia/cuda + cu124 torch
# wheels) — switched to CPU because this AWS account's G/VT instance quota is
# 0 with no approved increase (see deploy/README.md "Current deployment
# status"). GotOcrRecognizer (src/ocr_pipeline/recognizer_specialist.py)
# already falls back to CPU automatically when CUDA isn't available, so no
# code change was needed there. Revert to the CUDA base image + cu124 wheels
# once the quota clears, for real GPU throughput.
FROM python:3.11-slim

WORKDIR /app

# torch (CPU build) still needs libgomp at import time; opencv needs libGL/glib.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements/got-ocr2.txt .
RUN pip install --no-cache-dir -r got-ocr2.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# Weights are NOT baked into the image (GOT-OCR2.0 is ~1.1GB) — KServe's
# storage-initializer pulls them from storageUri (S3) into /mnt/models at pod
# startup instead. See deploy/kserve/got-ocr2-isvc.yaml.
COPY src/ocr_pipeline/ ./src/ocr_pipeline/
COPY deploy/serving/got_ocr2_server.py .
ENV PYTHONPATH=/app/src
ENV GOT_OCR2_MODEL_DIR=/mnt/models
ENV GOT_OCR2_DEVICE=cpu

EXPOSE 8080
CMD ["python3", "got_ocr2_server.py"]
