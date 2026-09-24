# Build from the repo root: docker build -f deploy/docker/orchestrator.Dockerfile .
# Deliberately no torch/paddlepaddle/transformers here — this service only
# makes HTTP calls to the three KServe InferenceServices, so it stays small
# and scales independently of the (expensive) GPU services it talks to.
FROM python:3.11-slim

WORKDIR /app

COPY deploy/requirements/orchestrator.txt .
RUN pip install --no-cache-dir -r orchestrator.txt

COPY src/ocr_pipeline/ ./src/ocr_pipeline/
COPY deploy/orchestrator/app.py .
ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
