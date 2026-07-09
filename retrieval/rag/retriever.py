"""Hybrid retrieval: CLIP vector similarity + BM25 keyword search + re-ranking.

Pipeline for a query against one video:

1. **CLIP channel** — embed the query text with CLIP and score it against the
   stored *image* embeddings (cross-modal: text query vs frame pixels).
2. **BM25 channel** — classic lexical ranking of the query against the frame
   *captions* (catches exact words CLIP may smooth over, e.g. "red jacket").
3. **Fusion** — both channels are min-max normalised to [0, 1] and blended
   (60 % CLIP / 40 % BM25) to produce candidate scores.
4. **Re-rank** — the top-10 candidates are re-scored with a cross-modal
   text-text similarity between the query and each caption in CLIP space
   (a lightweight stand-in for a full BLIP cross-attention scorer that stays
   fast on CPU), then blended 50/50 with the fusion score.

The RAG logic is implemented directly (no LangChain / LlamaIndex) on purpose.

Dependencies (store, embedder) are injected, so the ranking logic is testable
with fakes and no model downloads.
"""

from __future__ import annotations

import logging
import re

import numpy as np
from rank_bm25 import BM25Okapi

from perception.config import get_settings
from perception.inference.embedder import Embedder
from perception.models import RetrievedFrame
from retrieval.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

CLIP_WEIGHT = 0.6
BM25_WEIGHT = 0.4
RERANK_POOL = 10
RERANK_BLEND = 0.5  # final = 0.5 * fused + 0.5 * rerank


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Normalise scores to [0, 1].

    When all values are equal, tied items get 1.0 if their shared value is
    positive (they all deserve top credit), or 0.0 if it is zero/negative
    (no useful signal to amplify).  This matters most with a single frame:
    zeroing it out would discard real CLIP similarity.
    """
    low, high = float(scores.min()), float(scores.max())
    if high - low < 1e-9:
        return np.ones_like(scores) if high > 1e-9 else np.zeros_like(scores)
    return (scores - low) / (high - low)


class HybridRetriever:
    """CLIP + BM25 hybrid retrieval with cross-modal re-ranking."""

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def search(self, video_id: str, query: str, top_k: int | None = None) -> list[RetrievedFrame]:
        """Return the ``top_k`` most relevant frames of ``video_id`` for ``query``."""
        top_k = top_k or get_settings().retrieval_top_k
        frames = self.store.get_by_video_id(video_id)
        if not frames:
            logger.warning("No frames stored for video %s", video_id)
            return []

        # ── Channel 1: CLIP text→image similarity ──────────────
        # Query the vector store for similarity against every frame so the
        # two channels score an identical candidate set.
        clip_hits = self.store.search_by_text(
            query, self.embedder, video_id, top_k=len(frames)
        )
        clip_by_key = {(h.video_id, h.frame_idx): h.score for h in clip_hits}
        clip_scores = np.array(
            [clip_by_key.get((f.video_id, f.frame_idx), 0.0) for f in frames],
            dtype=np.float32,
        )

        # ── Channel 2: BM25 over captions + detection labels ───
        # Expand each detected label by its count so "person person person"
        # ranks higher than "person" for count-based queries (e.g. "how many
        # persons?"). Fall back to unique labels when counts are absent.
        def _label_tokens(frame: RetrievedFrame) -> list[str]:
            if frame.detection_counts:
                tokens: list[str] = []
                for label, count in frame.detection_counts.items():
                    tokens.extend([label.replace(" ", "_")] * count)
                return tokens
            return [lbl.replace(" ", "_") for lbl in frame.detection_labels]

        # Index caption + expanded labels + OCR text + events + actions so
        # lexical queries (a plate number, "handover", "fall") can hit directly.
        corpus = [
            _tokenize(
                " ".join([
                    f.caption,
                    " ".join(_label_tokens(f)),
                    f.ocr_text,
                    " ".join(f.events),
                    " ".join(f.actions),
                ])
            )
            for f in frames
        ]
        query_tokens = _tokenize(query)
        if any(corpus) and query_tokens:
            bm25 = BM25Okapi([tokens or ["<empty>"] for tokens in corpus])
            bm25_scores = np.array(bm25.get_scores(query_tokens), dtype=np.float32)
        else:
            bm25_scores = np.zeros(len(frames), dtype=np.float32)

        # ── Fusion ─────────────────────────────────────────────
        fused = CLIP_WEIGHT * _minmax(clip_scores) + BM25_WEIGHT * _minmax(bm25_scores)

        # ── Re-rank the top candidates ─────────────────────────
        pool_size = min(RERANK_POOL, len(frames))
        pool_indices = np.argsort(fused)[::-1][:pool_size]
        reranked = self._rerank(query, frames, fused, pool_indices)

        reranked.sort(key=lambda f: f.score, reverse=True)
        results = reranked[:top_k]
        logger.info(
            "Query '%s' on %s: %d candidates -> top score %.3f",
            query, video_id, len(frames), results[0].score if results else 0.0,
        )
        return results

    def _rerank(
        self,
        query: str,
        frames: list[RetrievedFrame],
        fused: np.ndarray,
        pool_indices: np.ndarray,
    ) -> list[RetrievedFrame]:
        """Re-score the candidate pool with query↔caption similarity in CLIP space."""
        pool = [frames[i] for i in pool_indices]
        try:
            query_vec = self.embedder.embed_text(query)
            captions = [f.caption or "<no caption>" for f in pool]
            caption_vecs = self.embedder.embed_texts(captions)
            cross = np.array([float(query_vec @ v) for v in caption_vecs], dtype=np.float32)
            cross = _minmax(cross)
        except Exception:
            logger.exception("Re-ranking failed; falling back to fused scores")
            cross = np.zeros(len(pool), dtype=np.float32)

        results: list[RetrievedFrame] = []
        for rank, idx in enumerate(pool_indices):
            frame = pool[rank].model_copy()
            frame.score = round(
                float((1 - RERANK_BLEND) * fused[idx] + RERANK_BLEND * cross[rank]), 4
            )
            results.append(frame)
        return results


if __name__ == "__main__":
    import argparse

    from perception.config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description="Hybrid search over a processed video")
    parser.add_argument("video_id")
    parser.add_argument("query")
    args = parser.parse_args()

    retriever = HybridRetriever(VectorStore(), Embedder())
    for hit in retriever.search(args.video_id, args.query):
        print(f"  t={hit.timestamp_sec:7.2f}s  score={hit.score:.3f}  {hit.caption}")
