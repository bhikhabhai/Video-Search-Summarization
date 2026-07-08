FROM python:3.11-slim

# ffmpeg: frame extraction / probing
# libgl1 + libglib2.0-0: OpenCV runtime deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so code changes don't bust the layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

ARG CACHEBUST=0
COPY src/ src/
COPY tests/ tests/

# Model weights (HuggingFace, torch hub, ultralytics, open_clip) all cache
# under /root/.cache — mounted as a named volume in docker-compose so they
# survive container rebuilds.
ENV HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch \
    YOLO_CONFIG_DIR=/root/.cache/ultralytics \
    PYTHONUNBUFFERED=1

EXPOSE 8000 7860

# Default command: the API. The UI service overrides this in docker-compose.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
