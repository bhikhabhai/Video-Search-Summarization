"""Startup sweep removes files of videos that never reached the registry."""

from __future__ import annotations

from types import SimpleNamespace

from retrieval.api import main
from retrieval.rag import track_store

KEPT, ORPHAN = "aaaaaaaaaaaa", "bbbbbbbbbbbb"


class FakeStore:
    def __init__(self, thumbs_dir):
        self.thumbs_dir = thumbs_dir
        self.deleted: list[str] = []

    def list_videos(self):
        return [SimpleNamespace(video_id=KEPT)]

    def delete_video(self, video_id):
        self.deleted.append(video_id)
        return 0


def test_purge_orphans(tmp_path, monkeypatch):
    settings = SimpleNamespace(data_dir=tmp_path, uploads_dir=tmp_path / "uploads")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(track_store, "get_settings", lambda: settings)
    for d in ("uploads", "tracks", "tracks_video", "thumbs"):
        (tmp_path / d).mkdir()
    files = [
        f"uploads/{KEPT}.mp4", f"uploads/{ORPHAN}.avi", f"uploads/{ORPHAN}.meta.jsonl",
        f"tracks/{ORPHAN}.json", f"tracks_video/{ORPHAN}.mp4", "uploads/.gitkeep",
    ]
    for f in files:
        (tmp_path / f).write_text("x")
    store = FakeStore(tmp_path / "thumbs")

    main._purge_orphans(store)

    left = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert left == ["uploads/.gitkeep", f"uploads/{KEPT}.mp4"]
    assert store.deleted == [ORPHAN]
