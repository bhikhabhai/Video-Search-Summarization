"""ChromaDB-backed persistent store for frame records + a tiny video registry.

Document schema per frame (stored as Chroma metadata):
    {video_id, frame_idx, timestamp_sec, caption, detection_labels,
     thumbnail_b64}

Embeddings are supplied by the caller (CLIP image vectors at ingest time,
CLIP text vectors at query time) — Chroma's built-in embedding functions are
deliberately not used so no surprise model downloads happen here.

The video registry is a small JSON file next to the Chroma directory; it
powers the UI's video dropdown without scanning the whole collection.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import chromadb
import numpy as np

from src.config import get_settings
from src.models import FrameRecord, RetrievedFrame, VideoSummaryInfo

logger = logging.getLogger(__name__)

_COLLECTION = "frames"


class VectorStore:
    """Persistent ChromaDB collection of captioned, embedded keyframes."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self._registry_path = self.persist_dir / "videos.json"
        self._registry_lock = threading.Lock()

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
                    "thumbnail_b64": r.thumbnail_b64,
                }
                for r in records
            ],
        )
        logger.info("Stored %d frames for video %s", len(records), records[0].video_id)

    @staticmethod
    def _meta_to_frame(meta: dict, score: float = 0.0) -> RetrievedFrame:
        labels = [l for l in str(meta.get("detection_labels", "")).split(",") if l]
        counts_raw = meta.get("detection_counts", "{}")
        try:
            counts: dict[str, int] = json.loads(counts_raw) if counts_raw else {}
        except (json.JSONDecodeError, TypeError):
            counts = {}
        actions_raw = meta.get("actions", "[]")
        try:
            actions: list[str] = json.loads(actions_raw) if actions_raw else []
        except (json.JSONDecodeError, TypeError):
            actions = []
        events_raw = meta.get("events", "[]")
        try:
            events: list[str] = json.loads(events_raw) if events_raw else []
        except (json.JSONDecodeError, TypeError):
            events = []
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
            events=events,
            thumbnail_b64=str(meta.get("thumbnail_b64", "")),
            score=round(score, 4),
        )

    def search_by_embedding(
        self, query_embedding: np.ndarray, video_id: str, top_k: int = 5
    ) -> list[RetrievedFrame]:
        """Cosine-similarity search over one video's frames."""
        result = self._collection.query(
            query_embeddings=[query_embedding.astype(np.float32).tolist()],
            where={"video_id": video_id},
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
        self, query: str, embedder, video_id: str, top_k: int = 5
    ) -> list[RetrievedFrame]:
        """Embed ``query`` with the supplied CLIP embedder and search."""
        return self.search_by_embedding(embedder.embed_text(query), video_id, top_k)

    def get_by_video_id(self, video_id: str) -> list[RetrievedFrame]:
        """Return all frames of a video, ordered by timestamp."""
        result = self._collection.get(where={"video_id": video_id})
        frames = [self._meta_to_frame(meta) for meta in (result.get("metadatas") or [])]
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
    from src.config import configure_logging

    configure_logging()
    store = VectorStore()
    videos = store.list_videos()
    print(f"{len(videos)} video(s) in store:")
    for video in videos:
        print(f"  {video.video_id}  {video.filename}  "
              f"{video.duration_sec:.1f}s  {video.keyframe_count} keyframes")
