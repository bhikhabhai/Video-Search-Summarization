# Mini-VSS — single image containing both layers:
#   perception/  → video ingestion + vision inference (YOLO, VLM, OCR, pose, CLIP)
#   retrieval/   → vector store, hybrid retrieval, summariser, FastAPI app
FROM python:3.11-slim

# System deps: ffmpeg (decode/probe), libgl1 + libglib2.0-0 (OpenCV runtime).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so code changes don't bust this (slow) layer.
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

# Copy both application layers into the image.
COPY perception/ perception/
COPY retrieval/ retrieval/

# Model weights load from the mounted model-cache volume (see docker-compose.yml),
# so nothing re-downloads across rebuilds. HuggingFace/Transformers, torch, and
# Ultralytics all point into it.
#
# NOTE: data/ (uploads, thumbs, chroma) is deliberately NOT copied — it is a
# runtime bind-mount volume, never baked into the image.
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model-cache/huggingface \
    TRANSFORMERS_CACHE=/app/model-cache/huggingface \
    TORCH_HOME=/app/model-cache/torch \
    YOLO_CONFIG_DIR=/app/model-cache/ultralytics

EXPOSE 8000

# FastAPI app lives at retrieval/api/main.py after the restructure.
CMD ["uvicorn", "retrieval.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
