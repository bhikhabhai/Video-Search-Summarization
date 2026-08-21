"""ChromaDB-backed persistent store for frame records + a tiny video registry.

Frame text/metadata is stored as Chroma metadata (caption, detection_labels,
events, …). Two things deliberately live OUTSIDE the vector DB:

  * **Thumbnails** — JPEGs on disk at ``<data>/thumbs/<video_id>/<frame_idx>.jpg``
    (see :meth:`save_thumbnail`/:meth:`load_thumbnail`). Keeping base64 JPEGs out
    of Chroma shrinks the DB ~60x and keeps them off the query read path.
  * **A lexical index** — a persistent SQLite FTS5 store (:class:`FTSIndex`) so
    BM25 lexical search is a single indexed query, not a per-request rebuild.

Embeddings are supplied by the caller (CLIP image vectors at ingest time,
CLIP text vectors at query time) — Chroma's built-in embedding functions are
deliberately not used so no surprise model downloads happen here.

The video registry is a small JSON file next to the Chroma directory; it
powers the UI's video dropdown without scanning the whole collection.
"""

from __future__ import annotations

import base64
import json
import logging
import shutil
import threading
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from perception.config import get_settings
from perception.models import FrameRecord, RetrievedFrame, VideoSummaryInfo
from retrieval.rag.fts_index import FTSIndex

logger = logging.getLogger(__name__)

_COLLECTION = "frames"


def _time_where(video_id: str, t_start: float | None, t_end: float | None) -> dict:
    """Chroma ``where`` filter: this video, optionally within a timestamp range."""
    clauses: list[dict] = [{"video_id": video_id}]
    if t_start is not None:
        clauses.append({"timestamp_sec": {"$gte": t_start}})
    if t_end is not None:
        clauses.append({"timestamp_sec": {"$lte": t_end}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class VectorStore:
    """Persistent ChromaDB collection of captioned, embedded keyframes."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        # Disable anonymized telemetry: on some chromadb/posthog version combos
        # it raises "capture() takes 1 positional argument but 3 were given" on
        # every collection op — harmless but very noisy. This silences it.
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        # Thumbnails live on disk (sibling to the Chroma dir), NOT in metadata —
        # keeps the vector DB ~60x smaller and query reads off the JPEG hot path.
        self.thumbs_dir = self.persist_dir.parent / "thumbs"
        # Persistent lexical index (replaces per-query BM25Okapi rebuilds).
        self._fts = FTSIndex(self.persist_dir / "fts.sqlite3")
        self._registry_path = self.persist_dir / "videos.json"
        self._registry_lock = threading.Lock()

    # ── Thumbnails (on disk) ───────────────────────────────────

    def _thumb_path(self, video_id: str, frame_idx: int) -> Path:
        return self.thumbs_dir / video_id / f"{frame_idx}.jpg"

    def save_thumbnail(self, video_id: str, frame_idx: int, jpeg_bytes: bytes) -> None:
        path = self._thumb_path(video_id, frame_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg_bytes)

    def load_thumbnail(self, video_id: str, frame_idx: int) -> str:
        """Base64 JPEG for one keyframe, or "" if the file is missing."""
        path = self._thumb_path(video_id, frame_idx)
        if path.is_file():
            return base64.b64encode(path.read_bytes()).decode()
        return ""

    @staticmethod
    def _fts_body(r: FrameRecord) -> str:
        """Lexical blob: caption + labels (expanded by count) + ocr + events + …

        Expanding labels by their count ("person person person") lets count-style
        lexical queries rank multi-object frames higher, matching the old BM25
        corpus construction.
        """
        if r.detection_counts:
            labels = [lbl.replace(" ", "_")
                      for lbl, n in r.detection_counts.items() for _ in range(n)]
        else:
            labels = [lbl.replace(" ", "_") for lbl in r.detection_labels]
        return " ".join([
            r.caption, " ".join(labels), r.ocr_text,
            " ".join(r.events), " ".join(r.actions), " ".join(r.attributes),
        ])

    # ── Frames ─────────────────────────────────────────────────

    def add_frames(self, records: list[FrameRecord], embeddings: list[np.ndarray]) -> None:
        """Insert (or overwrite) frame records with their image embeddings."""
        if len(records) != len(embeddings):
            raise ValueError("records and embeddings must be the same length")
        if not records:
            return

        self._collection.upsert(
            ids=[f"{r.video_id}:{r.frame_idx}" for r in records],
            embeddings=[e.astype(np.float32).tolist() for e in embeddings],
            documents=[r.caption for r in records],
            metadatas=[
                {
                    "video_id": r.video_id,
                    "frame_idx": r.frame_idx,
                    "timestamp_sec": r.timestamp_sec,
                    "caption": r.caption,
                    "detection_labels": ",".join(r.detection_labels),
                    "detection_counts": json.dumps(r.detection_counts),
                    "actions": json.dumps(r.actions),
                    "scene": r.scene,
                    "ocr_text": r.ocr_text,
                    "events": json.dumps(r.events),
                    "attributes": json.dumps(r.attributes),
                    "detections": json.dumps([d.model_dump() for d in r.detections]),
                    "frame_events": json.dumps(r.frame_events),
                    # thumbnail_b64 intentionally NOT stored here — it lives on
                    # disk (see save_thumbnail); keeping JPEGs out of the vector
                    # DB is the single biggest storage + query-latency win.
                }
                for r in records
            ],
        )
        self._fts.add([
            (r.video_id, r.frame_idx, r.timestamp_sec, self._fts_body(r)) for r in records
        ])
        logger.info("Stored %d frames for video %s", len(records), records[0].video_id)

    @staticmethod
    def _json_or(raw, default):
        try:
            return json.loads(raw) if raw else default
        except (json.JSONDecodeError, TypeError):
            return default

    @staticmethod
    def _meta_to_frame(meta: dict, score: float = 0.0) -> RetrievedFrame:
        labels = [l for l in str(meta.get("detection_labels", "")).split(",") if l]
        counts = VectorStore._json_or(meta.get("detection_counts", "{}"), {})
        actions = VectorStore._json_or(meta.get("actions", "[]"), [])
        events = VectorStore._json_or(meta.get("events", "[]"), [])
        attributes = VectorStore._json_or(meta.get("attributes", "[]"), [])
        detections = VectorStore._json_or(meta.get("detections", "[]"), [])
        frame_events = VectorStore._json_or(meta.get("frame_events", "[]"), [])
        return RetrievedFrame(
            video_id=str(meta["video_id"]),
            frame_idx=int(meta["frame_idx"]),
            timestamp_sec=float(meta["timestamp_sec"]),
            caption=str(meta.get("caption", "")),
            detection_labels=labels,
            detection_counts=counts,
            actions=actions,
            scene=str(meta.get("scene", "")),
            ocr_text=str(meta.get("ocr_text", "")),
            attributes=attributes,
            events=events,
            detections=detections,
            frame_events=frame_events,
            thumbnail_b64=str(meta.get("thumbnail_b64", "")),
            score=round(score, 4),
        )

    def search_by_embedding(
        self, query_embedding: np.ndarray, video_id: str, top_k: int = 5,
        t_start: float | None = None, t_end: float | None = None,
    ) -> list[RetrievedFrame]:
        """Approximate-NN cosine search over one video's frames (HNSW).

        ``top_k`` bounds the candidate pool returned by HNSW — the caller no
        longer passes ``len(frames)``, so this is a real ANN lookup, not a full
        scan. An optional ``[t_start, t_end]`` timestamp window pre-filters.
        """
        result = self._collection.query(
            query_embeddings=[query_embedding.astype(np.float32).tolist()],
            where=_time_where(video_id, t_start, t_end),
            n_results=top_k,
        )
        frames: list[RetrievedFrame] = []
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        for meta, dist in zip(metadatas[0], distances[0]):
            # Chroma cosine distance = 1 - cosine similarity.
            frames.append(self._meta_to_frame(meta, score=1.0 - float(dist)))
        return frames

    def search_by_text(
        self, query: str, embedder, video_id: str, top_k: int = 5,
        t_start: float | None = None, t_end: float | None = None,
    ) -> list[RetrievedFrame]:
        """Embed ``query`` with the supplied CLIP embedder and ANN-search."""
        return self.search_by_embedding(
            embedder.embed_text(query), video_id, top_k, t_start, t_end
        )

    def search_lexical(
        self, query: str, video_id: str, top_k: int = 5,
        t_start: float | None = None, t_end: float | None = None,
    ) -> list[RetrievedFrame]:
        """Top lexical (FTS5 BM25) matches for ``query`` in one video."""
        hits = self._fts.search(video_id, query, top_k, t_start, t_end)
        if not hits:
            return []
        by_idx = {f.frame_idx: f for f in self.get_by_ids(video_id, [i for i, _ in hits])}
        out: list[RetrievedFrame] = []
        for idx, score in hits:
            frame = by_idx.get(idx)
            if frame is not None:
                frame.score = round(score, 4)
                out.append(frame)
        return out

    def get_by_ids(self, video_id: str, frame_idxs: list[int]) -> list[RetrievedFrame]:
        """Fetch specific frames by id (no thumbnails loaded)."""
        if not frame_idxs:
            return []
        result = self._collection.get(ids=[f"{video_id}:{i}" for i in frame_idxs])
        return [self._meta_to_frame(meta) for meta in (result.get("metadatas") or [])]

    def get_by_video_id(
        self, video_id: str, with_thumbnails: bool = False
    ) -> list[RetrievedFrame]:
        """Return all frames of a video, ordered by timestamp.

        Thumbnails are loaded from disk only when ``with_thumbnails`` is set
        (e.g. the /frames gallery); query/summary paths skip them and load the
        handful they actually render lazily.
        """
        result = self._collection.get(where={"video_id": video_id})
        frames = [self._meta_to_frame(meta) for meta in (result.get("metadatas") or [])]
        if with_thumbnails:
            for f in frames:
                f.thumbnail_b64 = self.load_thumbnail(f.video_id, f.frame_idx)
        frames.sort(key=lambda f: f.timestamp_sec)
        return frames

    def delete_video(self, video_id: str) -> int:
        """Remove all frames of a video from Chroma and drop its registry entry.

        Returns the number of frames deleted (0 if the video was unknown).
        """
        existing = self._collection.get(where={"video_id": video_id})
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        self._fts.delete(video_id)
        thumb_dir = self.thumbs_dir / video_id
        if thumb_dir.is_dir():
            shutil.rmtree(thumb_dir, ignore_errors=True)

        with self._registry_lock:
            registry = self._read_registry()
            if registry.pop(video_id, None) is not None:
                self._registry_path.write_text(json.dumps(registry, indent=2))

        logger.info("Deleted video %s (%d frames)", video_id, len(ids))
        return len(ids)

    # ── Video registry ─────────────────────────────────────────

    def register_video(self, info: VideoSummaryInfo) -> None:
        with self._registry_lock:
            registry = self._read_registry()
            registry[info.video_id] = info.model_dump()
            self._registry_path.write_text(json.dumps(registry, indent=2))

    def list_videos(self) -> list[VideoSummaryInfo]:
        with self._registry_lock:
            registry = self._read_registry()
        return [VideoSummaryInfo(**entry) for entry in registry.values()]

    def _read_registry(self) -> dict[str, dict]:
        if self._registry_path.is_file():
            try:
                return json.loads(self._registry_path.read_text())
            except json.JSONDecodeError:
                logger.warning("Corrupt video registry; starting fresh")
        return {}


if __name__ == "__main__":
    from perception.config import configure_logging

    configure_logging()
    store = VectorStore()
    videos = store.list_videos()
    print(f"{len(videos)} video(s) in store:")
    for video in videos:
        print(f"  {video.video_id}  {video.filename}  "
              f"{video.duration_sec:.1f}s  {video.keyframe_count} keyframes")
