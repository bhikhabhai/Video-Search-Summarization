# 🎬 Mini-VSS — Video Search & Summarization

A deployable, end-to-end **video-understanding stack** that mirrors the shape of
NVIDIA's VSS Blueprint at a fraction of the complexity. Upload a short video, then
chat with it: ask natural-language questions ("when does a person appear in a red
jacket?"), retrieve semantic clips (timestamp + thumbnail), or stream a full
summary — all from a browser.

Runs on a **single GPU or CPU-only** (GPU auto-detected), and works fully offline
with a local/network Ollama LLM.

## Directory structure

The codebase is split into two purpose-named packages that mirror the pipeline —
**perception** (turn a video into structured observations) and **retrieval**
(index, search, and answer over them):

```
VSS/
├── perception/                 # Layer A: turn video → structured observations
│   ├── config.py               #   central settings + GPU/VRAM detection (shared)
│   ├── models.py               #   shared Pydantic schemas (shared)
│   ├── ingestion/              #   video probe + scene-change keyframe extraction
│   │   ├── video_loader.py     #     FFmpeg probe, limits, RTSP support
│   │   └── frame_extractor.py  #     dual-threshold scene detection + time fallback
│   └── inference/              #   the vision models (the GPU-heavy work)
│       ├── detector.py         #     YOLOv8 object detection
│       ├── captioner.py        #     Qwen2.5-VL / BLIP / Ollama structured captions
│       ├── embedder.py         #     CLIP image/text embeddings
│       ├── ocr.py              #     PaddleOCR / Ollama text + plate reading
│       └── pose.py             #     YOLO-pose + fall/fight/running rule engine
│
├── retrieval/                  # Layer B: index, search, answer
│   ├── rag/                    #   retrieval-augmented generation (no LangChain)
│   │   ├── vector_store.py     #     ChromaDB wrapper + video registry
│   │   ├── retriever.py        #     hybrid CLIP + BM25 + cross-similarity re-rank
│   │   └── summariser.py       #     windowed map-reduce streaming summariser + LLM
│   ├── api/                    #   the FastAPI application
│   │   └── main.py             #     endpoints + background ingestion pipeline (SSE)
│   └── ui/                     #   legacy single-file Gradio UI (kept for reference)
│
├── frontend/                   # Next.js UI (App Router + Tailwind), talks to the API
├── tests/                      # unit tests (scene detection, parsing, pose rules, …)
├── data/                       # RUNTIME volume — uploads, thumbs, chroma (not in image)
├── model-cache/                # RUNTIME volume — HuggingFace / torch / YOLO weights
├── Dockerfile                  # single image: perception + retrieval
├── docker-compose.yml          # api (+ web) services, bind-mounted data & weights
├── build.sh                    # prerequisite checks + build/run helper
└── requirements.txt
```

Dependencies flow one way — `retrieval` imports from `perception`, never the
reverse — so `config.py`/`models.py` live in `perception` as the shared foundation.

## Prerequisites

- **Docker** — https://docs.docker.com/get-docker/
- **Docker Compose** — v2 (`docker compose`) or v1 (`docker-compose`)
- **For GPU (optional):** NVIDIA drivers **and** the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Without a GPU the stack runs CPU-only automatically — no config change needed.

## How to run

```bash
./build.sh                 # or: docker-compose up --build
```

`build.sh` checks your prerequisites, builds the image, and starts the stack.
Add `--no-cache` for a clean rebuild:

```bash
./build.sh --no-cache
```

**Enabling a GPU:** uncomment the `deploy:` block in
[docker-compose.yml](docker-compose.yml) (it's commented so CPU-only hosts don't
fail on startup), then run the build again.

## Accessing the API & where data lives

| Service | URL |
|---|---|
| **API** (FastAPI) | http://localhost:8000 — interactive docs at `/docs` |
| **UI** (Next.js) | http://localhost:3000 |

Upload a video via the UI, or directly against the API:

```bash
curl -F "file=@clip.mp4" http://localhost:8000/ingest      # → {job_id, video_id}
curl http://localhost:8000/status/<job_id>                 # ingestion progress
```

All runtime data is **bind-mounted to the host** (survives rebuilds):

| Host path | Container path | Contents |
|---|---|---|
| `./data/uploads` | `/app/data/uploads` | uploaded source videos |
| `./data/thumbs` | `/app/data/thumbs` | keyframe thumbnails |
| `./data/chroma` | `/app/data/chroma` | ChromaDB vector store + video registry |

### API reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Multipart video upload → `{job_id, video_id}` |
| `GET` | `/status/{job_id}` | Ingestion job state + progress % |
| `POST` | `/query` | `{video_id, question}` → SSE-streamed grounded answer |
| `GET` | `/summary/{video_id}` | SSE-streamed chunked summary |
| `GET` | `/clip/{video_id}?q=` | Best-matching moment: timestamp + thumbnail |
| `GET` | `/frames/{video_id}` | All keyframes with timestamps + thumbnails |
| `GET` | `/videos` | Registry of processed videos |
| `DELETE` | `/videos/{video_id}` | Remove a video from the store |
| `GET` | `/health` | Liveness probe |

## Model weights & caching

Weights load from the bind-mounted `./model-cache` directory, so they persist on
the host and are **never re-downloaded** on rebuild. The image points the caches
there via env vars:

- `HF_HOME=/app/model-cache/huggingface`
- `TRANSFORMERS_CACHE=/app/model-cache/huggingface`
- `TORCH_HOME=/app/model-cache/torch`
- `YOLO_CONFIG_DIR=/app/model-cache/ultralytics`

**First run** downloads the models you enable (YOLO, the caption VLM, CLIP, …)
into `./model-cache` — this can take a while. Every subsequent start reuses them.

**Pre-populating the cache** (skip the first-run download): drop an existing
HuggingFace cache into place before building, e.g.

```bash
mkdir -p model-cache/huggingface
# copy a warmed cache from another machine / your host:
cp -r ~/.cache/huggingface/hub  model-cache/huggingface/hub
```

Any weights already present under `model-cache/huggingface` are used as-is; only
missing ones are fetched.

## Environment variables

All are optional — defaults run out of the box. Set them in `.env` (see the
committed `.env` for the full list). The Docker image also sets the cache/path
vars inline (above).

| Name | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Python logging level |
| `ANTHROPIC_API_KEY` | *(empty)* | If set, use Anthropic for Q&A/summaries; else Ollama |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Anthropic model id |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | LLM (Q&A/summary) endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Ollama text model |
| `MAX_VIDEO_SECONDS` / `MAX_VIDEO_MB` | `500` / `300` | Upload limits |
| `DIFF_THRESHOLD` / `HIST_THRESHOLD` | `0.08` / `0.15` | Scene-cut thresholds (diff ∧ χ²) |
| `MIN_SCENE_GAP_SEC` | `1.0` | Cool-down between keyframes |
| `MAX_GAP_SEC` | `3.0` | Force a keyframe every N s even with no cut |
| `ANALYSIS_FPS` / `MAX_KEYFRAMES` | `6.0` / `120` | Decode rate / hard keyframe cap |
| `DETECTOR_MODEL` / `DETECTOR_CONFIDENCE` | `yolov8n.pt` / `0.35` | Object detector |
| `CAPTION_BACKEND` | `qwen` | `qwen` \| `blip` \| `auto` \| `ollama` |
| `QWEN_VL_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | Local VLM checkpoint |
| `CAPTION_MAX_NEW_TOKENS` | `256` | Max tokens per caption (JSON needs headroom) |
| `OLLAMA_CONCURRENCY` | `4` | Concurrent requests for the Ollama caption/OCR path |
| `OCR_BACKEND` | `none` | `none` \| `paddle` \| `ollama` |
| `POSE_ENABLED` | `false` | Enable YOLO-pose fall/fight/running events |
| `RETRIEVAL_TOP_K` | `10` | Frames returned per query |
| `SUMMARY_WINDOW_SEC` | `30` | Summarisation window size |
| `DATA_DIR` / `CHROMA_DIR` | `data` / `data/chroma` | Runtime storage (mounted) |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Where the browser UI finds the API |

## Tests

```bash
# inside the container (has all deps):
docker-compose exec api python -m pytest tests/ -q
```

## What this showcases

- **Dual-threshold scene-change detection** (diff ∧ χ² histogram) with a
  time-based fallback for static-camera activity.
- **Structured VLM captioning** — one JSON schema per frame (scene / people /
  interactions / objects / vehicles / OCR / events), robust to truncated output.
- **Dedicated OCR** and **pose-based action rules** (fall / fight / running),
  independent of the VLM.
- **Hybrid RAG from first principles** — CLIP + BM25 fusion with cross-similarity
  re-rank; no LangChain / LlamaIndex.
- **Streaming LLM** Q&A + map-reduce summarisation over SSE.
