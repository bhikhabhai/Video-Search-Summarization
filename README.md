# 🎬 Mini-VSS — Video Search & Summarization

A deployable, end-to-end **video understanding stack** that intentionally mirrors the
architecture of NVIDIA's VSS Blueprint at a fraction of the complexity. Upload a short
video, then chat with it: ask natural-language questions ("when does a person appear in
a red jacket?"), retrieve semantic clips (timestamp + thumbnail), or stream a full
summary — all from a browser.

Runs on a **single GPU or CPU-only** (GPU is auto-detected), fully offline if you want
(local Ollama fallback), and starts with one command.

## Quickstart

```bash
docker compose up
```

Then open **http://localhost:3000** (Next.js UI) — the API lives at
http://localhost:8000 (interactive docs at `/docs`).

Optional: `cp .env.example .env` and set `ANTHROPIC_API_KEY` for higher-quality Q&A and
summaries. Without a key, Mini-VSS talks to a local [Ollama](https://ollama.com) server
(`ollama pull llama3.2` + default endpoint) — no paid APIs required.

**Cold start:** the first run downloads model weights (YOLOv8n ~6 MB, CLIP ViT-B/32
~350 MB, BLIP base ~1 GB — or BLIP-2 ~4 GB in 4-bit if you have ≥8 GB VRAM). On a
16 GB-RAM CPU-only machine with a decent connection, expect **10–20 minutes** for the
first `docker compose up` (image build + weight downloads). Weights are cached in a
named Docker volume (`model-cache`), so every subsequent start takes seconds.

## Manual dev setup

```bash
# Prereqs: Python 3.11, ffmpeg on PATH
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1: API
uvicorn src.api.main:app --port 8000

# Terminal 2: UI (Next.js dev server on http://localhost:3000)
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev

# Tests (fast, no model downloads)
pytest tests/ -q
# Include the slow CLIP integration test:
pytest tests/ -q -m slow
```

Each CV/inference module is also runnable standalone for debugging:

```bash
python -m src.ingestion.frame_extractor demo.mp4 --out sheet.png   # tune thresholds visually
python -m src.inference.detector photo.jpg
python -m src.inference.captioner photo.jpg
python -m src.inference.embedder photo.jpg
```

## Architecture

```
                ┌────────────────────────────────────────────────────────────┐
                │                 Layer 5: UI  (Next.js)                     │
                │        Upload tab · Ask tab (chat) · Summary tab            │
                └───────────────┬────────────────────────────────────────────┘
                                │ HTTP / SSE
                ┌───────────────▼────────────────────────────────────────────┐
                │                Layer 4: API  (FastAPI)                      │
                │  POST /ingest   GET /status/{job}   POST /query (stream)    │
                │  GET /summary/{vid} (stream)   GET /clip   GET /frames      │
                │        async background jobs + thread-pool inference        │
                └──────┬──────────────────────────────┬──────────────────────┘
                       │ ingest pipeline              │ query path
        ┌──────────────▼──────────────┐  ┌────────────▼───────────────────────┐
        │ Layer 1: VideoIngestion     │  │ Layer 3: RAGPipeline               │
        │  video_loader (FFmpeg probe)│  │  vector_store (ChromaDB, local)    │
        │  frame_extractor            │  │  retriever (CLIP + BM25 + re-rank) │
        │   · dual-threshold scene    │  │  summariser (30s windows,          │
        │     detection (diff ∧ χ²)   │  │   recursive reduce, streaming)     │
        │   · min_scene_gap_sec       │  │        │                           │
        └──────────────┬──────────────┘  │        ▼                           │
                       │ keyframes       │  LLM: Anthropic claude-haiku-4-5   │
        ┌──────────────▼──────────────┐  │       └─fallback→ local Ollama     │
        │ Layer 2: InferencePipeline  │  └────────────────────────────────────┘
        │  detector  (YOLOv8n, batch=8)                 ▲
        │  captioner (Qwen2.5-VL-3B ─fallback→ BLIP)   │
        │  embedder  (CLIP ViT-L/16, unit-norm 768-d)───┘
        └─────────────────────────────┘
```

**Ingest flow:** upload → FFmpeg probe → scene-change keyframes → YOLO detection →
VLM captions (deduped) → CLIP embeddings → ChromaDB, with 0–100 % progress events.

**Query flow:** question → hybrid retrieval (CLIP text→image similarity ⊕ BM25 over
captions, top-10 re-ranked by query↔caption similarity) → grounded LLM answer streamed
token-by-token with timestamp citations and supporting thumbnails.

## Environment variables

All variables are optional — defaults run out of the box. See `.env.example`.

| Name | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Python logging level |
| `ANTHROPIC_API_KEY` | *(empty)* | If set, use Anthropic for Q&A/summaries; else Ollama |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Anthropic model id |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local LLM fallback endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `MAX_VIDEO_SECONDS` | `500` | Upload duration limit |
| `MAX_VIDEO_MB` | `300` | Upload size limit |
| `DIFF_THRESHOLD` | `0.12` | Scene cut: min mean abs pixel diff (0–1) |
| `HIST_THRESHOLD` | `0.35` | Scene cut: min χ² colour-histogram distance |
| `MIN_SCENE_GAP_SEC` | `1.5` | Cool-down between keyframes (anti burst-sampling) |
| `ANALYSIS_FPS` | `6.0` | Decode rate used for cut detection |
| `MAX_KEYFRAMES` | `120` | Hard cap on keyframes per video |
| `DETECTOR_MODEL` | `yolov8n.pt` | Ultralytics checkpoint |
| `DETECTOR_CONFIDENCE` | `0.35` | Detection confidence threshold |
| `DETECTOR_BATCH_SIZE` | `8` | Frames per YOLO batch |
| `DETECTOR_CLASSES` | *(empty = all 80)* | Comma-separated COCO class filter |
| `VRAM_THRESHOLD_GB` | `8` | Below this VRAM, BLIP-2 → BLIP base fallback |
| `CAPTION_DEDUP_THRESHOLD` | `0.92` | Cosine sim above which consecutive captions merge |
| `RETRIEVAL_TOP_K` | `5` | Frames returned per query |
| `SUMMARY_WINDOW_SEC` | `30` | Summarisation window size |
| `LLM_MAX_TOKENS` | `1024` | Max tokens per LLM call |
| `DATA_DIR` | `data` | Uploads / thumbnails / registry |
| `CHROMA_DIR` | `data/chroma` | ChromaDB persistence directory |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Where the browser UI finds the API (build-time) |
| `API_PORT` | `8000` | API service port (UI serves on `3000`) |

## What this showcases

- **Dual-threshold scene-change detection** (image processing): a cut is declared only
  when *both* the absolute frame diff and the χ² colour-histogram distance exceed their
  thresholds — the diff catches structural change, the histogram vetoes motion-only
  false positives, and a `min_scene_gap_sec` cool-down prevents burst-sampling fast
  cuts. Includes a contact-sheet visualiser for threshold tuning
  (`python -m src.ingestion.frame_extractor video.mp4`). See the rationale docstring in
  [`src/ingestion/frame_extractor.py`](src/ingestion/frame_extractor.py).
- **Inference pipeline design**: batched YOLO inference (groups of 8),
  Qwen2.5-VL-3B captioning (captions + actions + OCR in one prompt, 4-bit on GPU,
  float32 on CPU) with a BLIP fallback if it can't load, CLIP embeddings
  normalised to the unit sphere, consecutive-caption deduplication (cosine > 0.92),
  lazy model loading, and per-batch error isolation (a failing frame never kills a job).
- **Hybrid RAG, implemented directly** (no LangChain/LlamaIndex): CLIP text→image
  vector similarity fused with BM25 over captions + detection labels, then top-10
  candidates re-ranked by query↔caption cross-similarity.
- **Streaming LLM integration**: token-by-token SSE from Anthropic
  (`claude-haiku-4-5-20251001`) with a zero-config offline Ollama fallback; chunked
  map-reduce summarisation (30 s windows, recursive reduce past 10 windows).
- **Async background processing with real-time progress**: FastAPI background jobs run
  CPU-bound inference in a thread pool, publishing 0–100 % progress that the UI polls
  every second.

## API reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Multipart video upload → `{job_id, video_id}` |
| `GET` | `/status/{job_id}` | Job state + progress % |
| `POST` | `/query` | `{video_id, question}` → SSE-streamed grounded answer |
| `GET` | `/summary/{video_id}` | SSE-streamed chunked summary |
| `GET` | `/clip/{video_id}?q=...` | Best-matching moment: timestamp + thumbnail |
| `GET` | `/frames/{video_id}` | All keyframes with timestamps + thumbnails |
| `GET` | `/videos` | Registry of processed videos |

## Repo layout

```
├── docker-compose.yml        # api + web services, model-cache volume
├── Dockerfile                # API image (Python)
├── .env.example
├── requirements.txt
├── src/
│   ├── config.py             # pydantic-settings + GPU/VRAM detection
│   ├── models.py             # shared Pydantic schemas
│   ├── ingestion/            # Layer 1: probe + scene-change keyframes
│   ├── inference/            # Layer 2: YOLO, BLIP(-2), CLIP
│   ├── rag/                  # Layer 3: ChromaDB, hybrid retrieval, summariser
│   ├── api/                  # Layer 4: FastAPI
│   └── ui/                   # Layer 5: legacy Gradio UI (kept for reference)
├── frontend/                 # Layer 5: Next.js UI (App Router + Tailwind)
│   ├── app/                  # layout, page, global styles
│   ├── components/           # Upload / Ask / Summary tabs, Gallery
│   ├── lib/                  # API client + SSE stream parsing
│   └── Dockerfile            # multi-stage Node build (standalone output)
└── tests/                    # scene detection, normalisation, hybrid retrieval
```

## Constraints honoured

- No Kubernetes, no cloud dependencies; `ANTHROPIC_API_KEY` optional (Ollama works offline).
- Cold-starts on 16 GB RAM, CPU-only (BLIP-base path); GPU used automatically when present.
- RAG implemented from first principles — no LangChain, no LlamaIndex.
- Python backend; Next.js frontend (App Router + Tailwind) talking to the API over REST + SSE.
