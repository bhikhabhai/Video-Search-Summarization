# AGENTS.md — context brief for AI assistants

Read this first, then [PIPELINE.md](PIPELINE.md) for the data flow. `README.md` is
the user-facing pitch; this file is the working brief. Standard `AGENTS.md`
location, so Cursor / Codex / Copilot / Gemini CLI / Claude Code all pick it up.

---

## 1. What this is

**Mini-VSS** — upload a video, then search and question it in natural language.
Two Python packages mirror the pipeline:

- **`perception/`** — video → structured observations (track, detect, caption, pose, OCR, embed).
- **`retrieval/`** — index those observations, search them, answer with an LLM.

Plus a **Next.js** frontend (`frontend/`). Dependencies flow **one way**:
`retrieval` imports `perception`, never the reverse. Shared `config.py` and
`models.py` therefore live in `perception/`.

Everything runs in Docker. ~7.7k lines of Python, ~1.1k of TypeScript.

---

## 2. The constraint that explains everything

**Target hardware is a GTX 1650 — 4 GB card, ~3.62 GB usable VRAM.**

Most of the odd-looking design decisions are this constraint, not preference.
Before you "optimise" any of these, know that reverting them causes CUDA OOM:

| Decision | Where | Why |
|---|---|---|
| `CAPTION_BATCH_SIZE=1` | `.env` | Qwen2.5-VL 3B in 4-bit is ~2.5 GB alone; batching OOMs |
| CLIP runs on **CPU** (`CLIP_DEVICE=cpu`) | `.env`, `config.py` | Frees the GPU for the caption VLM; costs ~5–10 s/batch |
| Weapon YOLO on **CPU** (`WEAPON_DEVICE=cpu`) | `.env` | Same reason; it's a small model |
| Pose runs **before** captioning, then `p.pose.unload()` | `main.py:_run_ingest` | Pose and the VLM must never co-reside in VRAM |
| Qwen 4-bit NF4 via bitsandbytes | `captioner.py` | fp16 does not fit |
| `QWEN_MAX_PIXELS=200704` | `.env` | Caps visual tokens; higher blows the KV cache |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | `docker-compose.yml` | Allocator fragmentation caused "unknown error" OOMs |

If you are running on a bigger GPU, change these in `.env` — **not in code**.

---

## 3. Dev loop

The stack is running in Docker right now (`vss-api-1`, `vss-web-1`).

```bash
docker compose ps                        # is it up?
docker compose up --build                # first time / after requirements.txt
./build.sh                               # same, with prerequisite checks
docker compose logs -f api               # watch ingestion
```

**`perception/` and `retrieval/` are bind-mounted into the container.** Editing
Python on the host and running:

```bash
docker compose restart api               # ~5 s — picks up code changes
```

is the normal loop. **Only rebuild when `requirements.txt` changes.**

### Tests

Deps are **not installed on the host** — `python3 -m pytest` on the host fails at
`import pydantic_settings`. Tests run inside the container:

```bash
docker compose exec api python -m pytest tests/ -q
docker compose exec api python -m pytest tests/ -q -m slow   # loads real weights
```

`pytest.ini` excludes `-m slow` by default. The fast suite uses synthetic inputs
and needs no model weights.

Several modules also carry a runnable `__main__` self-check — that is the house
style for new non-trivial logic:

```bash
docker compose exec api python -m retrieval.rag.fts_index      # asserts + prints OK
docker compose exec api python -m retrieval.rag.retriever <video_id> "<query>"
```

### Benchmarks

Two, answering different questions. Pick the right one:

**`real_bench.py` — one real video, end to end.** Where the wall-clock time
actually goes, what one video costs on disk, and whether the box can keep up
with a live camera. Runs on the **host** (stdlib only, no venv) against the
running API:

```bash
docker compose up -d
python3 benchmarks/real_bench.py my_video.mp4            # add --cleanup to not keep it
```

Reports per-stage ingest timing, per-keyframe storage split by component,
retrieval latency, a **realtime factor** (>1 = slower than the camera records),
and a 90-day extrapolation. Each upload gets a fresh `video_id`, so runs never
collide.

**`scale_bench.py` — synthetic records at fabricated scale.** DB size and query
latency only; it never loads a model, so it says nothing about ingest time. Runs
in the container:

```bash
docker compose run --rm -v "$(pwd)/benchmarks:/app/benchmarks" api \
  python /app/benchmarks/scale_bench.py --video-sizes 120 1000 10000
```

It writes to a throwaway temp Chroma dir and asserts it is not the production one.

---

## 4. Repo map

```
perception/
  config.py            ALL settings, env-overridable (pydantic-settings). Single source of truth.
  models.py            Pydantic schemas crossing layer boundaries (FrameRecord, RetrievedFrame, …)
  ingestion/
    video_loader.py    FFmpeg probe, size/duration limits
    tracker.py         YOLO + ByteTrack/BoT-SORT over every frame → keyframes, tracks, events
    iva_loader.py      replays external IVA meta.jsonl / manifest.jsonl instead of tracking
    frame_extractor.py legacy dual-threshold scene-cut extractor (fallback mode)
    botsort.yaml / bytetrack.yaml   tracker configs
  inference/
    captioner.py       Qwen2.5-VL 3B (4-bit) — the big one, 1.1k lines
    detector.py        YOLOv8n object detection
    weapon.py          fine-tuned YOLOv11m: handgun / long_gun / knife
    pose.py            YOLOv8-pose single-frame rules (fall / fight / crowding)
    temporal_action.py multi-frame pose actions per track id (fall/run/handshake/fight)
    embedder.py        CLIP ViT-L/14, 768-d, L2-normalised
    ocr.py             PaddleOCR or an Ollama vision model
retrieval/
  api/main.py          FastAPI: all endpoints + the ingest pipeline (836 lines). Start here.
  rag/
    vector_store.py    ChromaDB wrapper, thumbnails-on-disk, video registry
    fts_index.py       SQLite FTS5 lexical index (BM25)
    retriever.py       hybrid CLIP ⊕ BM25 fusion + cross-similarity rerank
    summariser.py      windowed map-reduce streaming summariser + LLM client
    track_store.py     per-video tracks/events JSON
  ui/app.py            legacy Gradio UI — superseded by frontend/, kept for reference
frontend/              Next.js 14 App Router + Tailwind (Upload / Ask / Summary tabs)
tests/                 pytest, synthetic inputs, no weights needed
benchmarks/            real_bench.py (one real video, end-to-end — host, stdlib only)
                       scale_bench.py (synthetic, DB + query cost only)
data/                  RUNTIME — uploads, thumbs, chroma, tracks. NOT source. Do not edit/delete.
model-cache/           RUNTIME — HF/torch/YOLO weights (GBs). Do not edit/delete.
```

**`retrieval/api/main.py` is the spine.** `Pipeline` (line ~71) constructs every
model as a lazy singleton; `_run_ingest()` (line ~321) is the whole ingest
pipeline top to bottom. Read those two before changing anything.

---

## 5. Conventions

- **Config is env vars only.** Add a field to `Settings` in `perception/config.py`
  with a default and a comment explaining the *why*, then override in `.env`.
  Never hardcode a tunable; never read `os.environ` outside `config.py`.
- **`get_settings()`, not `Settings()`** — it is `lru_cache`d.
- **Pydantic v2 models** for anything crossing a layer or process boundary;
  a plain dataclass is fine for in-process data carrying a live `PIL.Image`
  (see `ExtractedFrame`).
- **Comments explain why, not what.** This codebase is unusually well-commented
  about trade-offs and VRAM budgets — match that. If you delete such a comment,
  you are deleting the reason someone chose it.
- **No new dependencies** unless a few lines cannot do it. `requirements.txt` is
  already heavy (torch, transformers, chromadb, gradio, optional paddle).
- **New non-trivial logic leaves one runnable check** — a `__main__` assert
  self-check or one `test_*.py`. No fixtures, no frameworks.
- Everything must keep working **CPU-only** and **fully offline** (Ollama), as
  well as with an Anthropic key.

---

## 6. Current state (2026-08-21)

Last commit is `157a2dd dev(vlm): Add weapon detection and update tracker instruction`.
There is **substantial uncommitted work** in the tree — a retrieval scaling
rework. Do not `git checkout .` or `git stash` without asking.

### Committed and working

Track-based keyframing (ByteTrack/BoT-SORT), IVA meta.jsonl replay, weapon
detection, structured flat-format captioning, pose rules, hybrid retrieval,
streaming Q&A and summaries, the Next.js UI.

### Uncommitted — the scaling rework

Goal: retrieval that survives **90 days of continuous single-camera footage**
(~3.9M keyframes). The old query path loaded *every* frame of a video and
rebuilt a `BM25Okapi` index per request — O(corpus) per query, and thumbnails
were base64 strings inside the vector DB. Three changes fix that:

1. **`retrieval/rag/fts_index.py`** *(new)* — persistent SQLite **FTS5** index.
   One indexed query replaces the per-request BM25 rebuild. `bm25()` returns
   lower-is-better, so it is negated to match the CLIP channel's polarity.
2. **Thumbnails moved to disk** — `data/thumbs/<video_id>/<frame_idx>.jpg`,
   loaded lazily only for frames actually rendered. Shrinks Chroma ~60×.
   `FrameRecord.thumbnail_b64` is now written as `""` at ingest.
3. **Bounded candidate pools** — `retriever.search()` pulls `CANDIDATE_CAP=300`
   per channel (ANN for CLIP, FTS5 for lexical) and unions them, instead of
   scoring the whole video. Cost is now O(candidates).

#### Measured impact

From `benchmarks/scale_bench.py` — synthetic data, real code paths. These are the
owner's numbers from the last run; **not re-verified in this tree.** Re-run before
trusting them after any retrieval change.

Storage, Chroma DB per keyframe:

| | before | after |
|---|---|---|
| per keyframe (DB) | 515 KB | **7.5 KB** |
| 90 days @ 3.9M frames | ~1.9 TB | **~28 GB** (~68× smaller) |

Query latency:

| frames/video | ANN | FTS | **total (new)** | legacy full-load alone (old) |
|---|---|---|---|---|
| 500 | 30 ms | 50 ms | **80 ms** | 19 ms |
| 5,000 | 65 ms | 53 ms | **118 ms** | 226 ms |
| 50,000 | 553 ms | 84 ms | **636 ms** | 2,720 ms |

Three things to read off this before you touch retrieval:

- **FTS5 is near-flat** — 50 → 84 ms across 100× more data. The lexical channel
  is solved; it is not where to spend effort.
- **ANN is now the dominant cost** (553 of 636 ms at 50k, and growing
  super-linearly). Chroma's filtered HNSW is the next bottleneck, not BM25.
- **The new path is *slower* below ~1–2k frames** (80 ms vs 19 ms at 500). That
  crossover is expected — bounded candidate work has fixed overhead that only
  pays off at scale. Do not "fix" it; both are imperceptible, and above 5k the
  new path wins 4×.

The old regime for context: the legacy path did a 2.7 s full load at 50k *before*
rebuilding BM25 over all 50k frames, and a merged 1M-frame collection took minutes
or OOM'd. The new path stays sub-second per video.

Also uncommitted: `perception/inference/temporal_action.py` (multi-frame pose
actions, `POSE_ACTION_ENABLED=true`), an optional `t_start`/`t_end` time window
on `/query`, and `benchmarks/scale_bench.py`.

`rank-bm25` is still in `requirements.txt` but `retriever.py` no longer imports
it — droppable once the FTS path is confirmed.

**Not yet verified:** the full test suite has not been run against this working
tree in-container. Run it before committing.

### Known ceilings

- Retrieval is **per-video**. There is no cross-video or cross-camera search; a
  90-day deployment is still one long `video_id`.
- **ANN is the remaining scaling wall** — 553 ms at 50k frames and rising
  faster than linear. Chroma's HNSW is in-process, single-node, and the
  `where={"video_id": …}` filter is applied around the index rather than inside
  it. This is the next thing to hit, not BM25.
- Ingestion is **batch, not streaming** — a file is uploaded, then processed.
  There is no live RTSP consumer loop.
- Ingest runs in a **single-worker thread pool**; one video at a time.
- `MAX_KEYFRAMES=120` per video is a VRAM guard, not a scale answer — the storage
  numbers above assume that cap is lifted for long recordings.

---

## 7. Gotchas

- **Do not commit `.env`.** It holds an Anthropic API key and internal LAN IPs
  for the Ollama servers. It is gitignored — keep it that way.
- **`data/` and `model-cache/` are bind-mounted runtime volumes**, not source.
  `model-cache/` is gigabytes of weights; deleting it forces a re-download.
  `data/` holds every ingested video and the vector DB.
- Deleting a video must clear **five** places: Chroma rows, the FTS index, the
  on-disk thumbnail dir, `videos.json`, and `data/tracks/`. See
  `VectorStore.delete_video` + `DELETE /videos/{id}`.
- The Ollama servers in `.env` are **remote LAN hosts**. If they are unreachable,
  Q&A and summaries fail while ingestion still works. Set `ANTHROPIC_API_KEY` to
  route the LLM through Anthropic instead.
- CORS is wide open (`allow_origins=["*"]`) — deliberate, single-user local
  deployment. There is **no auth anywhere**. Do not deploy this publicly as-is.
- The frontend calls the API from the browser on `:8000`, not through a proxy.
- `retrieval/ui/app.py` (Gradio) is dead code kept for reference. Do not extend it.

---

## 8. API surface

| Method | Path | Notes |
|---|---|---|
| `POST` | `/ingest` | multipart: video + optional `meta.jsonl` + `manifest.jsonl` → `{job_id, video_id}` |
| `GET` | `/status/{job_id}` | job state + 0–100 progress |
| `POST` | `/query` | `{video_id, question, t_start?, t_end?}` → SSE. First chunk is `\x00FRAMES:<json>` |
| `GET` | `/summary/{video_id}` | SSE chunked summary |
| `GET` | `/clip/{video_id}?q=` | best-matching moment |
| `GET` | `/frames/{video_id}` | all keyframes **with** thumbnails (the gallery) |
| `GET` | `/videos` · `DELETE /videos/{id}` | registry |
| `GET` | `/track_video/{video_id}` | annotated tracking video |
| `GET` | `/health` | liveness |

Interactive docs at `http://localhost:8000/docs`.
