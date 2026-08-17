# Mini-VSS — GPU-enabled image (NVIDIA CUDA 12.1 + cuDNN 8)
#   perception/  → video ingestion + vision inference (YOLO, VLM, OCR, pose, CLIP)
#   retrieval/   → vector store, hybrid retrieval, summariser, FastAPI app
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# System deps: Python 3.11, ffmpeg (decode/probe), libgl1 + libglib2.0-0 (OpenCV runtime).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.11 python3.11-dev python3-pip \
       ffmpeg libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && pip install --no-cache-dir --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install PyTorch with CUDA 12.1 support first ──────────────
# Must come BEFORE -r requirements.txt so pip doesn't replace it
# with a CPU-only build from PyPI.
RUN pip install --no-cache-dir \
    torch==2.4.1+cu121 \
    torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining Python deps (torch/torchvision already satisfied above).
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
    YOLO_CONFIG_DIR=/app/model-cache/ultralytics \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000

# FastAPI app lives at retrieval/api/main.py after the restructure.
CMD ["uvicorn", "retrieval.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
