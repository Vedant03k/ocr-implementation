# Build from the repo root: docker build -f deploy/docker/paddleocr.Dockerfile .
FROM python:3.11-slim

WORKDIR /app

# opencv-python (a PaddleOCR dependency) needs libGL/glib at import time, which
# python:3.11-slim doesn't ship.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements/paddleocr.txt .
RUN pip install --no-cache-dir -r paddleocr.txt

# PaddleOCR's own weights: baked in at build time from a pre-populated
# models/paddleocr/ (synced from S3 beforehand), NOT downloaded at pod
# startup — see deploy/README.md for why (no live network dependency at
# inference-server boot).
COPY models/paddleocr/ ./models/paddleocr/
ENV PADDLE_PDX_CACHE_HOME=/app/models/paddleocr

COPY src/ocr_pipeline/ ./src/ocr_pipeline/
COPY deploy/serving/paddleocr_server.py .
ENV PYTHONPATH=/app/src
ENV PADDLEOCR_DEVICE=cpu

EXPOSE 8080
CMD ["python", "paddleocr_server.py"]
