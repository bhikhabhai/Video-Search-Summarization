"""Persistent SQLite FTS5 full-text index over keyframe text.

Replaces rebuilding a `rank_bm25.BM25Okapi` index from every frame on every
query (which re-tokenised the whole video per request). FTS5 keeps a persistent
inverted index on disk and returns the top lexical matches for one video in a
single query, scoped by an optional timestamp range.

`bm25()` in SQLite returns *lower = better* (typically negative); we negate it so
callers get *higher = better*, matching the CLIP cosine channel.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _match_expr(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expr: OR of quoted alnum tokens."""
    tokens = _TOKEN_RE.findall(query.lower())
    return " OR ".join(f'"{t}"' for t in tokens)


class FTSIndex:
    """Thread-safe FTS5 store of ``(video_id, frame_idx) -> body`` keyed rows."""

    def __init__(self, db_path: Path) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS frames_fts USING fts5("
            "video_id UNINDEXED, frame_idx UNINDEXED, ts UNINDEXED, body, "
            "tokenize='unicode61')"
        )
        self._conn.commit()

    def add(self, rows: list[tuple[str, int, float, str]]) -> None:
        """Upsert ``(video_id, frame_idx, timestamp_sec, body)`` rows.

        FTS5 has no UPSERT, so we delete any existing rows for these frames first
        (re-ingesting a video replaces its entries cleanly).
        """
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "DELETE FROM frames_fts WHERE video_id = ? AND frame_idx = ?",
                [(vid, idx) for vid, idx, _, _ in rows],
            )
            self._conn.executemany(
                "INSERT INTO frames_fts (video_id, frame_idx, ts, body) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def search(
        self,
        video_id: str,
        query: str,
        top_k: int,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> list[tuple[int, float]]:
        """Return ``[(frame_idx, score)]`` best lexical matches, higher = better."""
        expr = _match_expr(query)
        if not expr:
            return []
        sql = ("SELECT frame_idx, -bm25(frames_fts) FROM frames_fts "
               "WHERE video_id = ? AND frames_fts MATCH ?")
        params: list = [video_id, expr]
        if t_start is not None:
            sql += " AND ts >= ?"; params.append(t_start)
        if t_end is not None:
            sql += " AND ts <= ?"; params.append(t_end)
        sql += " ORDER BY bm25(frames_fts) LIMIT ?"
        params.append(top_k)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [(int(idx), float(score)) for idx, score in rows]

    def delete(self, video_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM frames_fts WHERE video_id = ?", (video_id,))
            self._conn.commit()


if __name__ == "__main__":
    import tempfile

    idx = FTSIndex(Path(tempfile.mktemp(suffix=".sqlite3")))
    idx.add([
        ("v1", 0, 1.0, "a person walking past a car"),
        ("v1", 1, 5.0, "a dog in the park"),
        ("v2", 0, 2.0, "a person with a backpack"),
    ])
    assert [i for i, _ in idx.search("v1", "person", 10)] == [0], "scope + match"
    assert idx.search("v1", "person", 10, t_start=3.0) == [], "time filter excludes t=1"
    assert [i for i, _ in idx.search("v1", "dog park", 10)] == [1]
    assert idx.search("v1", "???", 10) == [], "no alnum tokens -> empty"
    print("fts_index self-check passed")
