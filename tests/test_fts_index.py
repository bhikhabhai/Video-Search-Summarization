"""Unit tests for the persistent SQLite FTS5 lexical index."""

from __future__ import annotations

from pathlib import Path

from retrieval.rag.fts_index import FTSIndex


def _index(tmp_path: Path) -> FTSIndex:
    idx = FTSIndex(tmp_path / "fts.sqlite3")
    idx.add([
        ("v1", 0, 1.0, "a person walking past a car"),
        ("v1", 1, 5.0, "a dog in the park"),
        ("v1", 2, 9.0, "a person with a backpack"),
        ("v2", 0, 2.0, "a truck on the highway"),
    ])
    return idx


def test_scopes_to_video(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    assert {i for i, _ in idx.search("v1", "person", 10)} == {0, 2}
    assert idx.search("v2", "person", 10) == []  # v2 has no "person"


def test_ranks_and_limits(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    hits = idx.search("v1", "person backpack", 1)
    assert len(hits) == 1
    assert hits[0][0] == 2  # frame 2 matches both terms, ranks first


def test_time_window(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    assert {i for i, _ in idx.search("v1", "person", 10, t_start=3.0)} == {2}
    assert idx.search("v1", "person", 10, t_end=0.5) == []


def test_non_alnum_query_is_safe(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    assert idx.search("v1", "??? !!!", 10) == []  # no tokens -> empty, no SQL error


def test_reingest_replaces(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    idx.add([("v1", 0, 1.0, "now shows a bicycle only")])  # overwrite frame 0
    assert idx.search("v1", "car", 10) == []          # old body gone
    assert {i for i, _ in idx.search("v1", "bicycle", 10)} == {0}


def test_delete_video(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    idx.delete("v1")
    assert idx.search("v1", "person", 10) == []
    assert {i for i, _ in idx.search("v2", "truck", 10)} == {0}  # v2 untouched
