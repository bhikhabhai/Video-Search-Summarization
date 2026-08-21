# Mini-VSS — How a Video Is Processed & Stored

The end-to-end path from an uploaded video to a searchable, question-answerable
index. Every step below maps to a file you can open.

```
upload → keyframes (IVA/track) + pose/events → weapon + caption + OCR
       → CLIP 768-d embedding → ChromaDB (cosine) + tracks.json
       → query: CLIP + BM25 + event timeline → grounded LLM answer
```

---

## 1. Upload

`POST /ingest` (in `retrieval/api/main.py`) accepts up to **three** files:

| File | Required | Purpose |
|---|---|---|
| **video** | yes | `.mp4` / `.avi` |
| **`meta.jsonl`** | no | IVA per-frame detections / tracking / license plates |
| **`manifest.jsonl`** | no | IVA ROI geometry (tripwire lines, zones) |

Files are saved to `data/uploads/{video_id}.*`. Ingestion runs as a **background
job** in a single-worker thread pool; the UI polls `GET /status/{job_id}` for
0–100 % progress. The whole pipeline lives in `_run_ingest()`.

---

## 2. Keyframe selection

The system does **not** caption every frame — it selects meaningful **keyframes**.
There are three modes:

| Mode | Chosen when | How it works |
|---|---|---|
| **IVA** | `meta.jsonl` attached | `IVALoader` replays IVA's per-frame objects/events. Keyframe on: active track-ID set changes · periodic (2 s) · any event. |
| **track** (default) | no meta | `VideoTracker` runs **YOLO + ByteTrack** over every frame. Same keyframe triggers. |
| **scene** | fallback | Dual-threshold scene-change extractor (pixel diff ∧ colour histogram). |

Files: `perception/ingestion/{iva_loader,tracker,frame_extractor}.py`.

Every mode outputs, per keyframe: **detections** (label + normalised bbox +
track_id), **track facts** (who's present, since when), **events**, plus a
per-video **summary** (unique counts, track list, event timeline, ROI regions).

**Two analyses run per frame *during* the track/IVA pass:**

- **Temporal pose actions** (`PoseActionRunner`, `perception/inference/temporal_action.py`)
  — YOLO-pose runs on every person-frame, keypoints are buffered per track id, and
  motion over the window yields **fall / running / handshake / fight / sudden-movement**.
  Pose is loaded during this pass and unloaded before the caption VLM (so they never
  share VRAM).
- **Event-forced keyframes** — a tripwire crossing, loiter, entered/left, or pose
  action **forces a keyframe** at that moment, so events between periodic samples
  aren't missed.

---

## 3. Per-keyframe enrichment

For each selected keyframe:

1. **Weapon detection** (optional) — fine-tuned **YOLOv11m** →
   `handgun / long_gun / knife`, merged into the frame's detections + events.
   `perception/inference/weapon.py`
2. **Captioning** — **Qwen2.5-VL-3B** (4-bit) produces a flat `KEY: value`
   description (scene, people, activity, clothing, PPE, objects, interactions,
   vehicles, plates, text, events), *guided by* the detections, track facts, and
   weapon hits. Flat format is used because it's robust for a small model.
   `perception/inference/captioner.py`
3. **OCR** (optional) — dedicated text / number-plate reading.
4. **Deterministic event injection** — analytics events (tripwire, pose actions,
   weapons) are folded straight into the caption text so they're always searchable,
   regardless of what the VLM wrote.

---

## 4. Embedding (the vector part)

- **Model:** CLIP **ViT-L/14** (`open_clip`, OpenAI weights), running on **CPU**
  (`CLIP_DEVICE=cpu`, to keep the GPU free for the caption VLM).
- Each keyframe image → a **768-dimensional, L2-normalised `float32`** vector.
  Unit-norm means cosine similarity = dot product everywhere.
- `perception/inference/embedder.py` (`EMBED_DIM = 768`).
- Text queries are embedded with the **same** CLIP model at query time — that's
  what makes cross-modal *text → image* search work.

---

## 5. Storage

Embeddings + all per-frame metadata live in **ChromaDB** (persistent, on disk at
`data/chroma/`), collection **`"frames"`**, distance metric **cosine**
(`hnsw:space="cosine"`). `retrieval/rag/vector_store.py`.

Per keyframe, one `upsert`:

| Field | Value |
|---|---|
| **id** | `"{video_id}:{frame_idx}"` |
| **embedding** | the 768-d CLIP image vector |
| **document** | the caption |
| **metadata** | `caption, scene, detection_labels, detection_counts, actions, ocr_text, events, attributes, detections (label+bbox+track_id, JSON), frame_events (JSON), video_id, frame_idx, timestamp_sec` |

Two things deliberately live **outside** the vector DB:

- **Thumbnails** — JPEGs at `data/thumbs/{video_id}/{frame_idx}.jpg`, loaded
  lazily only for the frames actually rendered. Keeping base64 out of Chroma
  shrinks the DB ~60×.
- **The lexical index** — a persistent SQLite **FTS5** table at
  `data/chroma/fts.sqlite3` (`retrieval/rag/fts_index.py`) holding
  caption + labels + OCR + events per frame, so BM25 is one indexed query
  instead of a per-request rebuild over the whole video.

Alongside Chroma:

- `data/chroma/videos.json` — video registry (powers the UI dropdown).
- `data/tracks/{video_id}.json` — per-video summary: `counts`, `tracks`,
  `events` timeline, `regions` (ROI geometry).
- `data/tracks_video/{video_id}.mp4` — annotated tracking video (boxes + IDs).

Deleting a video (`DELETE /videos/{id}`) removes all of the above.

---

## 6. Query / Q&A

`POST /query` (optional `t_start` / `t_end` narrows both channels to a time window):

1. Embed the question with CLIP → **ANN cosine search** in Chroma
   (`score = 1.0 − distance`), capped at `CANDIDATE_CAP = 300` candidates.
2. **Fuse** with an FTS5 BM25 lexical search (same cap), unioning the two
   candidate sets — cost scales with the pool, not with frames-in-video.
3. Select the frames actually relevant to the question (event frames for event
   queries; word matches otherwise), not just the semantic top-K.
4. Prepend the **deterministic event timeline** and **unique object counts**
   (from `tracks.json`) to the prompt.
5. The LLM answers, grounded **only** in the retrieved observations.
6. Retrieved thumbnails are returned with boxes drawn on the queried object or
   event (green = object, amber = event, red = ROI).

---

## Where things live

```
perception/
  ingestion/   iva_loader.py · tracker.py · frame_extractor.py · video_loader.py
  inference/   captioner.py · embedder.py · detector.py · weapon.py
               pose.py · temporal_action.py · ocr.py
  config.py    all settings (env-overridable)
retrieval/
  api/main.py  ingest pipeline + endpoints (/ingest /query /summary /videos …)
  rag/         vector_store.py (ChromaDB) · fts_index.py (FTS5) · retriever.py
               summariser.py · track_store.py
data/
  uploads/     source video + IVA jsonl
  chroma/      the vector DB + videos.json + fts.sqlite3
  thumbs/      per-frame JPEG thumbnails
  tracks/      per-video summary JSON
  tracks_video/ annotated tracking videos
```
