# Build from the repo root: docker build -f deploy/docker/got-ocr2.Dockerfile .
# GPU base image — must match the CUDA driver on the EKS GPU node group.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deploy/requirements/got-ocr2.txt .
RUN pip install --no-cache-dir -r got-ocr2.txt --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple

# Weights are NOT baked into the image (GOT-OCR2.0 is ~1.1GB) — KServe's
# storage-initializer pulls them from storageUri (S3) into /mnt/models at pod
# startup instead. See deploy/kserve/got-ocr2-isvc.yaml.
COPY src/ocr_pipeline/ ./src/ocr_pipeline/
COPY deploy/serving/got_ocr2_server.py .
ENV PYTHONPATH=/app/src
ENV GOT_OCR2_MODEL_DIR=/mnt/models
ENV GOT_OCR2_DEVICE=cuda

EXPOSE 8080
CMD ["python3", "got_ocr2_server.py"]
