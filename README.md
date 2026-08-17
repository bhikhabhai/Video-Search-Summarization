<div align="center">

# 🎬 Mini-VSS — Video Search & Summarization

**Upload a video, then chat with it.** Ask natural-language questions, retrieve the
exact moment something happens (timestamp + thumbnail), or stream a full summary —
all from your browser.

A compact, self-hostable video-understanding stack, built from first principles
to run on a **single GPU or CPU-only**.

![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

</div>

---

## Table of contents

- [What it does](#what-it-does)
- [Demo](#demo)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Running inference on a remote server](#running-inference-on-a-remote-server)
- [API reference](#api-reference)
- [Model weights & caching](#model-weights--caching)
- [Tests](#tests)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## What it does

- **💬 Ask** — "When does a person in a red jacket appear?" → grounded, streamed
  answers with timestamp citations and the supporting frames.
- **🔍 Retrieve** — semantic clip search returns the best-matching moment
  (timestamp + thumbnail), ranked by relevance.
- **📝 Summarize** — a streamed, chronological summary of the whole video.
- **🧠 Understands frames deeply** — object detection, a vision-language model
  producing **structured** observations (scene / people / interactions / objects /
  vehicles / on-screen text / events), dedicated **OCR** for number plates &
  signage, and **pose-based** action detection (fall / fight / running).
- **🏃 Runs anywhere** — CPU-only out of the box, GPU auto-detected. Fully offline
  with a local/network Ollama LLM, or bring an Anthropic API key.

## Demo

> 📸 _Add a screenshot or short GIF of the three tabs here — e.g._ `docs/demo.gif`.
> Suggested shots: the **Upload** tab with the keyframe grid, the **Ask** tab
> streaming an answer with retrieved thumbnails, and the frame **lightbox** showing
> caption / detected objects / OCR / events.

## Architecture

Two purpose-named packages mirror the pipeline: **`perception/`** turns a video
into structured observations; **`retrieval/`** indexes them and answers questions.

```
                        ┌─────────────────────────────────────────────┐
                        │               Next.js UI  (:3000)            │
                        │        Upload · Ask (chat) · Summary         │
                        └───────────────────────┬─────────────────────┘
                                                 │  HTTP / SSE  (via /api proxy)
                        ┌───────────────────────▼─────────────────────┐
                        │            FastAPI   ·  retrieval/api        │
                        │  /ingest  /query  /summary  /clip  /frames   │
                        │      async background jobs · SSE streaming   │
                        └──────────┬────────────────────────┬─────────┘
              ingest pipeline      │                        │   query pipeline
     ┌───────────────────────────▼─────────┐     ┌──────────▼──────────────────────┐
     │  perception/   understand the video │     │  retrieval/    search & answer   │
     │                                      │     │                                  │
     │  ingestion/                          │     │  rag/                            │
     │    • video_loader   FFmpeg probe     │     │    • vector_store  ChromaDB      │
     │    • frame_extractor  scene cuts     │     │    • retriever     CLIP ⊕ BM25   │
     │  inference/  (the vision models)     │     │    • summariser    map-reduce    │
     │    • detector    YOLOv8              │     │            │                     │
     │    • captioner   Qwen2.5-VL / BLIP   │     │            ▼                     │
     │    • ocr         PaddleOCR / Ollama  │     │   LLM   Anthropic / Ollama       │
     │    • pose        YOLO-pose + rules   │     └──────────────────────────────────┘
     │    • embedder    CLIP                │
     └──────────────┬───────────────────────┘
                    │  FrameRecord{ caption, objects, ocr_text, events, embedding }
                    ▼
               ChromaDB  ◄──────────────────  retriever reads at query time
```

Dependencies flow one way — `retrieval` imports from `perception`, never the
reverse — so the shared `config.py` / `models.py` live in `perception` as the
foundation.

## Tech stack

| Area | Choice |
|---|---|
| API | FastAPI + Uvicorn (async, SSE streaming) |
| UI | Next.js 14 (App Router) + Tailwind |
| Object detection | YOLOv8 (Ultralytics) |
| Captioning (VLM) | Qwen2.5-VL-3B · BLIP fallback · Ollama backend |
| OCR | PaddleOCR or an Ollama vision model |
| Pose / actions | YOLOv8-pose + a rule engine |
| Embeddings | CLIP (open_clip) |
| Vector store | ChromaDB (local, persistent) |
| Retrieval | Hybrid CLIP + BM25 + cross-similarity re-rank (no LangChain) |
| LLM | Anthropic Claude _or_ local/network Ollama |
| Packaging | Docker + Docker Compose |

## Quickstart

**Prerequisites:** [Docker](https://docs.docker.com/get-docker/) and Docker Compose
(v2 `docker compose` or v1 `docker-compose`). For GPU: NVIDIA drivers + the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
(the app falls back to CPU automatically when absent).

```bash
git clone <your-repo-url> && cd VSS
./build.sh                 # checks prerequisites, builds, and starts the stack
#   or:  docker-compose up --build
#   clean rebuild:  ./build.sh --no-cache
```

Then open:

| Service | URL |
|---|---|
| **UI** (Next.js) | http://localhost:3000 |
| **API** (FastAPI) | http://localhost:8000 — interactive docs at `/docs` |

Upload a short clip in the UI and switch to **Ask** / **Summary**. Or drive the API
directly:

```bash
curl -F "file=@clip.mp4" http://localhost:8000/ingest      # → {job_id, video_id}
curl http://localhost:8000/status/<job_id>                 # ingestion progress 0–100%
```

> **First run** downloads model weights into `./model-cache` (cached for every
> later start). **CPU-only note:** the local VLM is accurate but slow
> (~15–30 s/frame) — see [Configuration](#configuration) to trim keyframes, use a
> GPU, or offload to a [remote server](#running-inference-on-a-remote-server).
> **GPU:** uncomment the `deploy:` block in `docker-compose.yml`.

## How it works

**Ingestion** (background job, 0–100% progress):

1. **Probe & sample** — FFmpeg reads the video; keyframes are chosen by
   **dual-threshold scene-change detection** (a cut fires only when the pixel
   diff *and* the χ² colour-histogram distance both exceed thresholds), with a
   time-based fallback so static-camera activity is still sampled.
2. **Perceive each keyframe** — YOLO detection → a VLM emits a **structured JSON**
   observation (scene, people, interactions, objects, vehicles, on-screen text,
   events) → dedicated OCR → pose-based action rules → CLIP embedding.
3. **Index** — everything is written to ChromaDB with the frame's metadata.

**Query:**

- **Ask / Retrieve** — the question is embedded and fused with a BM25 keyword
  score over captions, OCR text, and events; the top candidates are re-ranked by
  query↔caption cross-similarity, then a grounded answer is streamed by the LLM.
- **Summarize** — frames are grouped into time windows, each window is summarized,
  and the summaries are recursively merged and streamed.

## Project structure

```
VSS/
├── perception/                 # turn video → structured observations
│   ├── config.py               #   central settings + GPU/VRAM detection (shared)
│   ├── models.py               #   shared Pydantic schemas (shared)
│   ├── ingestion/
│   │   ├── video_loader.py     #   FFmpeg probe, limits, RTSP support
│   │   └── frame_extractor.py  #   dual-threshold scene detection + time fallback
│   └── inference/              #   the vision models
│       ├── detector.py         #   YOLOv8 object detection
│       ├── captioner.py        #   Qwen2.5-VL / BLIP / Ollama structured captions
│       ├── embedder.py         #   CLIP image/text embeddings
│       ├── ocr.py              #   PaddleOCR / Ollama text + plate reading
│       └── pose.py             #   YOLO-pose + fall/fight/running rule engine
│
├── retrieval/                  # index, search, answer
│   ├── rag/
│   │   ├── vector_store.py     #   ChromaDB wrapper + video registry
│   │   ├── retriever.py        #   hybrid CLIP + BM25 + cross-similarity re-rank
│   │   └── summariser.py       #   windowed map-reduce streaming summariser + LLM
│   ├── api/main.py             #   FastAPI endpoints + background ingestion (SSE)
│   └── ui/app.py               #   legacy single-file Gradio UI (kept for reference)
│
├── frontend/                   # Next.js UI (App Router + Tailwind)
├── tests/                      # unit tests (scene detection, parsing, pose rules, …)
├── data/                       # RUNTIME volume — uploads, thumbs, chroma
├── model-cache/                # RUNTIME volume — HuggingFace / torch / YOLO weights
├── Dockerfile                  # single image: perception + retrieval
├── docker-compose.yml          # api + web services, bind-mounted data & weights
└── build.sh                    # prerequisite checks + build/run helper
```

## Configuration

All settings are environment variables (see `.env`); every one has a sensible
default. The most useful:

| Name | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(empty)* | If set, use Anthropic for Q&A/summaries; else Ollama |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | LLM (Q&A/summary) endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Ollama text model |
| `MAX_VIDEO_SECONDS` / `MAX_VIDEO_MB` | `500` / `300` | Upload limits |
| `DIFF_THRESHOLD` / `HIST_THRESHOLD` | `0.08` / `0.15` | Scene-cut thresholds (diff ∧ χ²) |
| `MIN_SCENE_GAP_SEC` / `MAX_GAP_SEC` | `1.0` / `3.0` | Keyframe cool-down / time-based fallback |
| `MAX_KEYFRAMES` | `120` | Hard cap on keyframes per video |
| `CAPTION_BACKEND` | `qwen` | `qwen` \| `blip` \| `auto` \| `ollama` |
| `QWEN_VL_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | Local VLM checkpoint |
| `CAPTION_BATCH_SIZE` | `1` | Frames per forward pass (>1 = faster, more RAM) |
| `CAPTION_MAX_NEW_TOKENS` | `256` | Max tokens per caption (JSON needs headroom) |
| `OLLAMA_CONCURRENCY` | `4` | Concurrent requests for the Ollama caption/OCR path |
| `OCR_BACKEND` | `none` | `none` \| `paddle` \| `ollama` |
| `POSE_ENABLED` | `false` | Enable YOLO-pose fall/fight/running events |
| `RETRIEVAL_TOP_K` | `10` | Frames returned per query |
| `SUMMARY_WINDOW_SEC` | `30` | Summarisation window size |

## Running inference on a remote server

The heavy vision work (captioning, OCR) and the LLM can run on **separate machines**
— useful when the app runs on a small box and a GPU server does the compute. No code
changes; just point the backends at your servers in `.env`:

```ini
# LLM (Q&A + summaries) — already remote-capable
OLLAMA_BASE_URL=http://<gpu-host>:11434

# Offload captioning to a network Ollama vision model
CAPTION_BACKEND=ollama
OLLAMA_CAPTION_BASE_URL=http://<gpu-host>:11434
OLLAMA_CAPTION_MODEL=qwen2.5vl
OLLAMA_CONCURRENCY=4              # parallelizes the network requests

# Offload OCR too (optional)
OCR_BACKEND=ollama
OCR_OLLAMA_BASE_URL=http://<gpu-host>:11434
```

The remote Ollama must be bound to `0.0.0.0`, have the model pulled
(`ollama pull qwen2.5vl`), and allow port `11434`. Detection / embedding / pose
stay local (they're light). A single dedicated GPU inference service that hosts
*everything* is on the [roadmap](#roadmap).

## API reference

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
the host and are **never re-downloaded** on rebuild. The image points every cache
there (`HF_HOME`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `YOLO_CONFIG_DIR` →
`/app/model-cache/...`).

**Pre-populate the cache** to skip the first-run download:

```bash
mkdir -p model-cache/huggingface
cp -r ~/.cache/huggingface/hub  model-cache/huggingface/hub   # warmed cache
```

## Tests

```bash
docker-compose exec api python -m pytest tests/ -q
```

Fast tests need no model weights — they cover scene-change detection, structured-
JSON parsing/repair, OCR merging, and the pose rule engine with synthetic inputs.

## Roadmap

- [x] Structured VLM captions (scene / people / objects / OCR / events)
- [x] Dedicated OCR (PaddleOCR / Ollama)
- [x] Pose-based action rules (fall / fight / running / crowding)
- [x] Batched & concurrent captioning
- [ ] Object **tracking** (ByteTrack) → temporal event chains across frames
- [ ] Video-VLM on **clips** for true temporal understanding
- [ ] A dedicated, containerized **GPU inference server** (all vision models behind one API)

## Acknowledgements

Built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
  [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL),
  [open_clip](https://github.com/mlfoundations/open_clip),
  [ChromaDB](https://github.com/chroma-core/chroma), and
  [Ollama](https://github.com/ollama/ollama).

## License

Released under the **MIT License** — add a `LICENSE` file to make it official.
