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

## Documentation

| Document | Read it when |
|---|---|
| **README.md** (this file) | You want to run it, configure it, or see what it does |
| **[PIPELINE.md](PIPELINE.md)** | You want to know exactly what happens to a video, file by file |
| **[AGENTS.md](AGENTS.md)** | You (or an AI assistant) are about to change the code — hardware constraints, dev loop, conventions, current state, gotchas |

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
     │    • tracker      YOLO + ByteTrack   │     │    • fts_index     SQLite FTS5   │
     │    • iva_loader   external analytics │     │    • retriever     CLIP ⊕ BM25   │
     │    • frame_extractor  scene cuts     │     │    • summariser    map-reduce    │
     │  inference/  (the vision models)     │     │            │                     │
     │    • detector    YOLOv8              │     │            ▼                     │
     │    • weapon      YOLOv11m fine-tune  │     │   LLM   Anthropic / Ollama       │
     │    • captioner   Qwen2.5-VL / BLIP   │     └──────────────────────────────────┘
     │    • ocr         PaddleOCR / Ollama  │
     │    • pose        YOLO-pose + rules   │
     │    • temporal_action  per-track pose │
     │    • embedder    CLIP                │
     └──────────────┬───────────────────────┘
                    │  FrameRecord{ caption, objects, ocr_text, events, embedding }
                    ▼
       ChromaDB + FTS5 + thumbs/  ◄────────  retriever reads at query time
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
| Tracking | ByteTrack / BoT-SORT (appearance ReID) |
| Weapons | fine-tuned YOLOv11m (handgun / long_gun / knife) |
| Captioning (VLM) | Qwen2.5-VL-3B (4-bit) · BLIP fallback · Ollama backend |
| OCR | PaddleOCR or an Ollama vision model |
| Pose / actions | YOLOv8-pose + single-frame rules and per-track temporal actions |
| Embeddings | CLIP ViT-L/14, 768-d (open_clip) |
| Vector store | ChromaDB (local, persistent) + SQLite FTS5 lexical index |
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
cp .env.example .env       # then edit; defaults run offline on CPU + Ollama
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

> Full detail, file by file, in **[PIPELINE.md](PIPELINE.md)**.

**Ingestion** (background job, 0–100% progress):

1. **Probe & track** — FFmpeg reads the video; **YOLO + ByteTrack** runs over
   every frame and picks keyframes when the set of active track IDs changes, on
   a periodic tick, or when an event fires (tripwire, loiter, pose action). An
   attached IVA `meta.jsonl` replaces tracking with replayed analytics; the
   legacy dual-threshold scene-cut extractor remains as a fallback mode.
2. **Perceive each keyframe** — weapon detection → a VLM emits a **structured**
   observation (scene, people, interactions, objects, vehicles, on-screen text,
   events), guided by the detections and track facts → dedicated OCR → CLIP
   embedding. Deterministic analytics events are folded into the caption text so
   they are always searchable regardless of what the VLM wrote.
3. **Index** — metadata into ChromaDB, lexical text into SQLite FTS5, thumbnails
   onto disk, per-video track/event summary into `data/tracks/`.

**Query:**

- **Ask / Retrieve** — the question is embedded (CLIP ANN) and fused with an FTS5
  BM25 lexical score over captions, labels, OCR text and events; both channels
  pull a *bounded* candidate pool, so cost scales with the pool rather than the
  video length. Top candidates are re-ranked by query↔caption cross-similarity,
  the deterministic event timeline is prepended, then a grounded answer streams
  from the LLM.
- **Summarize** — frames are grouped into time windows, each window is summarized,
  and the summaries are recursively merged and streamed.

## Project structure

```
VSS/
├── AGENTS.md                   # context brief for AI assistants (constraints, conventions)
├── PIPELINE.md                 # how a video flows from upload to answer
│
├── perception/                 # turn video → structured observations
│   ├── config.py               #   central settings + GPU/VRAM detection (shared)
│   ├── models.py               #   shared Pydantic schemas (shared)
│   ├── ingestion/
│   │   ├── video_loader.py     #   FFmpeg probe, limits, RTSP support
│   │   ├── tracker.py          #   YOLO + ByteTrack → keyframes, tracks, events
│   │   ├── iva_loader.py       #   replay external IVA meta/manifest jsonl
│   │   └── frame_extractor.py  #   dual-threshold scene detection + time fallback
│   └── inference/              #   the vision models
│       ├── detector.py         #   YOLOv8 object detection
│       ├── weapon.py           #   fine-tuned YOLOv11m handgun/long_gun/knife
│       ├── captioner.py        #   Qwen2.5-VL / BLIP / Ollama structured captions
│       ├── embedder.py         #   CLIP image/text embeddings
│       ├── ocr.py              #   PaddleOCR / Ollama text + plate reading
│       ├── pose.py             #   YOLO-pose + fall/fight/running rule engine
│       └── temporal_action.py  #   multi-frame pose actions buffered per track id
│
├── retrieval/                  # index, search, answer
│   ├── rag/
│   │   ├── vector_store.py     #   ChromaDB wrapper + video registry + thumbnails
│   │   ├── fts_index.py        #   persistent SQLite FTS5 lexical (BM25) index
│   │   ├── retriever.py        #   hybrid CLIP + BM25 + cross-similarity re-rank
│   │   ├── track_store.py      #   per-video tracks / events JSON
│   │   └── summariser.py       #   windowed map-reduce streaming summariser + LLM
│   ├── api/main.py             #   FastAPI endpoints + background ingestion (SSE)
│   └── ui/app.py               #   legacy single-file Gradio UI (kept for reference)
│
├── frontend/                   # Next.js UI (App Router + Tailwind)
├── tests/                      # unit tests (scene detection, parsing, pose rules, …)
├── benchmarks/
│   ├── real_bench.py           #   time + space of ONE real video, end to end
│   └── scale_bench.py          #   synthetic storage/latency benchmark at 90-day scale
├── data/                       # RUNTIME volume — uploads, thumbs, chroma, tracks
├── model-cache/                # RUNTIME volume — HuggingFace / torch / YOLO weights
├── Dockerfile                  # single image: perception + retrieval
├── docker-compose.yml          # api + web services, bind-mounted data & weights
└── build.sh                    # prerequisite checks + build/run helper
```

## Configuration

All settings are environment variables (template: `.env.example`); every one has a sensible
default. The most useful:

| Name | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(empty)* | If set, use Anthropic for Q&A/summaries; else Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | LLM (Q&A/summary) endpoint; in Docker use `http://host.docker.internal:11434` for a host-side Ollama (as in `.env.example`) |
| `OLLAMA_MODEL` | `llama3.2` | Ollama text model |
| `MAX_VIDEO_SECONDS` / `MAX_VIDEO_MB` | `500` / `300` | Upload limits |
| `DIFF_THRESHOLD` / `HIST_THRESHOLD` | `0.08` / `0.15` | Scene-cut thresholds (diff ∧ χ²) |
| `MIN_SCENE_GAP_SEC` / `MAX_GAP_SEC` | `1.0` / `3.0` | Keyframe cool-down / time-based fallback |
| `MAX_KEYFRAMES` | `120` | Hard cap on keyframes per video |
| `KEYFRAME_MODE` | `track` | `track` (YOLO+ByteTrack) \| `scene` (dual-threshold cuts) |
| `TRACKER_CONFIG` | `botsort.yaml` | `botsort.yaml` (appearance ReID) \| `bytetrack.yaml` (faster) |
| `TRACK_CLASSES` / `TRACK_IMGSZ` | `0,2,3,5,7` / `960` | COCO ids to track / tracking resolution |
| `CAPTION_BACKEND` | `qwen` | `qwen` \| `blip` \| `auto` \| `ollama` |
| `CAPTION_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` — force CPU on a tiny/shared GPU |
| `QWEN_VL_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | Local VLM checkpoint |
| `CAPTION_BATCH_SIZE` | `1` | Frames per forward pass — **must stay 1 on 4 GB VRAM** |
| `CAPTION_MAX_NEW_TOKENS` | `192` | Max tokens per caption |
| `CAPTION_FORMAT` | `flat` | `flat` (`KEY: value`, robust) \| `json` (needs a capable model) |
| `OLLAMA_CONCURRENCY` | `4` | Concurrent requests for the Ollama caption/OCR path |
| `OCR_BACKEND` | `none` | `none` \| `paddle` \| `ollama` |
| `POSE_ENABLED` | `false` | Single-frame YOLO-pose fall/fight/crowding events |
| `POSE_ACTION_ENABLED` | `false` | Per-track temporal actions (fall/run/handshake/fight) |
| `WEAPON_ENABLED` / `WEAPON_DEVICE` | `false` / `auto` | YOLOv11m weapon detection; `cpu` frees VRAM |
| `CLIP_DEVICE` | `cpu` | Keep CLIP off the GPU so the caption VLM gets it all |
| `RETRIEVAL_TOP_K` | `5` | Frames returned per query |
| `SUMMARY_WINDOW_SEC` | `30` | Summarisation window size |

> **4 GB GPU?** `CAPTION_BATCH_SIZE=1`, `CLIP_DEVICE=cpu` and `WEAPON_DEVICE=cpu`
> are not arbitrary — see [AGENTS.md](AGENTS.md#2-the-constraint-that-explains-everything).

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
| `POST` | `/ingest` | Multipart video (+ optional IVA `meta.jsonl` / `manifest.jsonl`) → `{job_id, video_id}` |
| `GET` | `/status/{job_id}` | Ingestion job state + progress % |
| `POST` | `/query` | `{video_id, question, t_start?, t_end?}` → SSE-streamed grounded answer |
| `GET` | `/summary/{video_id}` | SSE-streamed chunked summary |
| `GET` | `/clip/{video_id}?q=` | Best-matching moment: timestamp + thumbnail |
| `GET` | `/frames/{video_id}` | All keyframes with timestamps + thumbnails |
| `GET` | `/track_video/{video_id}` | Annotated tracking video (boxes + track IDs) |
| `GET` | `/videos` | Registry of processed videos |
| `DELETE` | `/videos/{video_id}` | Remove a video from the store |
| `GET` | `/health` | Liveness probe |

## Model weights & caching

Weights load from the bind-mounted `./model-cache` directory, so they persist on
the host and are **never re-downloaded** on rebuild. The image points every cache
there (`HF_HOME`, `TORCH_HOME`, `YOLO_CONFIG_DIR` →
`/app/model-cache/...`).

**Pre-populate the cache** to skip the first-run download:

```bash
mkdir -p model-cache/huggingface
cp -r ~/.cache/huggingface/hub  model-cache/huggingface/hub   # warmed cache
```

## Tests

Dependencies are installed **inside the image**, not on the host — run pytest in
the container:

```bash
docker compose exec api python -m pytest tests/ -q
docker compose exec api python -m pytest tests/ -q -m slow   # loads real weights
```

Fast tests need no model weights — they cover scene-change detection, structured
caption parsing/repair, OCR merging, the pose rule engine, the FTS5 index, and
hybrid retrieval, all on synthetic inputs. `pytest.ini` excludes `-m slow` by
default.

Several modules also carry a runnable self-check:

```bash
docker compose exec api python -m retrieval.rag.fts_index
docker compose exec api python -m retrieval.rag.retriever <video_id> "<query>"
```

## Benchmarking your own video

How long does *your* video take, and what does it cost on disk? Point
`real_bench.py` at it while the stack is running — it drives the real pipeline
through the API and needs no Python deps on the host:

```bash
docker compose up -d
python3 benchmarks/real_bench.py my_video.mp4 --cleanup
```

It reports per-stage ingest timing (captioning usually dominates), per-keyframe
storage split across Chroma / thumbnails / tracks, retrieval latency, and a
**realtime factor** — above `1.0x` the box ingests slower than a camera records,
which is the number that decides whether live capture is feasible.

For DB and query cost at scale *without* waiting on the models, use the
synthetic `benchmarks/scale_bench.py` instead (see [AGENTS.md](AGENTS.md#benchmarks)).

## Roadmap

- [x] Structured VLM captions (scene / people / objects / OCR / events)
- [x] Dedicated OCR (PaddleOCR / Ollama)
- [x] Pose-based action rules (fall / fight / running / crowding)
- [x] Batched & concurrent captioning
- [x] Object **tracking** (ByteTrack / BoT-SORT) → temporal event chains across frames
- [x] Weapon detection (fine-tuned YOLOv11m)
- [x] Temporal, per-track pose actions (fall / running / handshake / fight)
- [x] Persistent FTS5 lexical index + bounded candidate retrieval — **~68× smaller
      vector DB, sub-second queries at 50k frames/video** ([details](AGENTS.md#measured-impact))
- [ ] **Cross-video / multi-camera** search — retrieval is still per-`video_id`
- [ ] **Streaming ingest** (live RTSP) instead of upload-then-process
- [ ] Video-VLM on **clips** for true temporal understanding
- [ ] A dedicated, containerized **GPU inference server** (all vision models behind one API)

## Acknowledgements

Built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
  [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL),
  [open_clip](https://github.com/mlfoundations/open_clip),
  [ChromaDB](https://github.com/chroma-core/chroma), and
  [Ollama](https://github.com/ollama/ollama).

## License

Released under the [MIT License](LICENSE).
