"""Standalone scale benchmark for the Mini-VSS retrieval pipeline.

Measures REAL per-keyframe on-disk size and per-query latency at 1-day / 30-day /
90-day scale, using synthetic data so no 90 days of video are required. It reuses
the actual production code paths (`VectorStore`, `HybridRetriever`) so the numbers
reflect the real implementation, not a re-model.

It NEVER touches production data: it writes to an ephemeral throwaway Chroma dir
(a fresh temp dir by default) and deletes it on exit unless --keep is given.

Each `--video-sizes` entry is ingested as its own `video_id`, so a big single
size (e.g. 100000) doubles as the "merged single collection" collapse test:
`HybridRetriever.search()` full-loads and rebuilds BM25 over the whole video, so
its latency is what one giant video/collection would cost per query.

Run (Docker, one-off mount so benchmarks/ is visible inside the container):
    docker compose run --rm -v "$(pwd)/benchmarks:/app/benchmarks" api \
        python /app/benchmarks/scale_bench.py --video-sizes 120 1000 10000
(fake embedder is the default; pass --real-embedder for true CLIP text-embed latency.)

See the module bottom / the chat for a full run recipe.
"""

from __future__ import annotations

import argparse
import base64
import io
import random
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from perception.models import Detection, FrameRecord
from retrieval.rag.retriever import HybridRetriever
from retrieval.rag.vector_store import VectorStore

EMBED_DIM = 768

QUERIES = [
    "When did a person enter the restricted area?",
    "Show me when someone was carrying a handgun.",
    "When did a person leave an object behind?",
    "How many people entered the restricted area yesterday?",
]

_WORDS = (
    "person people man woman walking standing carrying bag backpack car vehicle "
    "gate zone entrance corridor camera night day parked handbag box object left "
    "restricted area loitering entered crossed tripwire handgun weapon running fall"
).split()


class FakeEmbedder:
    """Deterministic-ish random unit vectors — lets us benchmark the DB / BM25 /
    load paths without downloading the 1.7 GB CLIP model. Use --real-embedder to
    measure true CLIP text-embed latency instead."""

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(EMBED_DIM).astype(np.float32)
            out.append(v / (np.linalg.norm(v) or 1.0))
        return out


def make_thumbnail(width: int = 480) -> str:
    """A realistic ~compressible JPEG base64 string (noise → worst-case size)."""
    arr = (np.random.rand(int(width * 9 / 16), width, 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def make_record(video_id: str, idx: int, thumb: str, with_thumb: bool) -> FrameRecord:
    n_people = random.randint(0, 4)
    caption = " ".join(random.choices(_WORDS, k=28))
    dets = [
        Detection(
            label="person",
            confidence=round(random.uniform(0.5, 0.95), 2),
            bbox=(random.random(), random.random(), random.random(), random.random()),
            track_id=f"person_P{random.randint(0, 9999):04d}",
        )
        for _ in range(n_people)
    ]
    return FrameRecord(
        video_id=video_id,
        frame_idx=idx,
        timestamp_sec=idx * 2.0,
        caption=caption,
        detection_labels=["person"] * n_people,
        detection_counts={"person": n_people} if n_people else {},
        actions=random.choice([[], ["walking"], ["standing", "carrying"]]),
        scene="outdoor, night, gate",
        ocr_text=random.choice(["", "GATE A", "PLATE MH12AB1234"]),
        events=random.choice([[], ["person_1 entered"], ["person_1 loitering (10s)"]]),
        attributes=random.choice([[], ["orange backpack"], ["blue jacket"]]),
        detections=dets,
        frame_events=[],
        thumbnail_b64=thumb if with_thumb else "",
    )


def ingest(store: VectorStore, video_id: str, n: int, with_thumb: bool, batch: int = 2000) -> None:
    thumb = make_thumbnail() if with_thumb else ""
    done = 0
    while done < n:
        k = min(batch, n - done)
        records = [make_record(video_id, done + i, thumb, with_thumb) for i in range(k)]
        emb = np.random.standard_normal((k, EMBED_DIM)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        store.add_frames(records, list(emb))
        done += k
        print(f"    ingested {done}/{n} for {video_id}", end="\r", flush=True)
    print(f"    ingested {n}/{n} for {video_id}        ")


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def median_ms(fn, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def est_prompt_tokens(frames, cap: int = 40) -> int:
    """Char/4 heuristic over what actually goes to the LLM (text only)."""
    chars = 0
    for f in frames[:cap]:
        chars += len(f.caption) + len(f.ocr_text)
        chars += sum(len(x) for x in f.detection_labels + f.actions + f.events + f.attributes)
        chars += 20  # timestamp + separators
    return chars // 4


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-sizes", type=int, nargs="+", default=[120, 1000, 10000],
                    help="keyframes per synthetic video; each is its own video_id")
    ap.add_argument("--top-k", type=int, nargs="+", default=[10, 25, 50, 100])
    ap.add_argument("--no-thumbnail", action="store_true",
                    help="omit thumbnails to isolate their storage cost")
    ap.add_argument("--real-embedder", action="store_true",
                    help="use real CLIP (needs torch+open_clip+model) for true embed latency")
    ap.add_argument("--persist-dir", type=Path, default=None,
                    help="throwaway Chroma dir (default: fresh temp dir, deleted on exit)")
    ap.add_argument("--keep", action="store_true", help="do not delete the throwaway dir")
    args = ap.parse_args()

    persist = args.persist_dir or Path(tempfile.mkdtemp(prefix="vss_bench_"))
    persist.mkdir(parents=True, exist_ok=True)
    assert "data/chroma" not in str(persist).replace("\\", "/"), "refusing to write to prod chroma dir"
    print(f"Throwaway Chroma dir: {persist}")
    with_thumb = not args.no_thumbnail

    if args.real_embedder:
        from perception.inference.embedder import Embedder
        embedder = Embedder()
    else:
        embedder = FakeEmbedder()

    store = VectorStore(persist_dir=persist)
    retriever = HybridRetriever(store, embedder)

    try:
        # ── Phase 1: ingest ────────────────────────────────────
        print("\n== Phase 1: ingest ==")
        videos = {}
        for n in args.video_sizes:
            vid = f"bench_{n}"
            print(f"  video {vid}: {n} keyframes (thumbnails={'on' if with_thumb else 'off'})")
            ingest(store, vid, n, with_thumb)
            videos[vid] = n

        # ── Phase 2: storage ───────────────────────────────────
        total_kf = sum(videos.values())
        total_bytes = dir_size_bytes(persist)
        print("\n== Phase 2: storage ==")
        print(f"  total keyframes : {total_kf:,}")
        print(f"  on-disk size    : {human(total_bytes)}")
        print(f"  per keyframe    : {human(total_bytes / total_kf)}")
        print(f"  projected 90d @ ~3.9M keyframes: {human(total_bytes / total_kf * 3_900_000)}")

        # ── Phase 3: latency ───────────────────────────────────
        # Times the NEW retrieval path: bounded ANN candidates + FTS5 lexical +
        # fusion/rerank. `legacy_scan` is the OLD per-query cost (full video load
        # + BM25-over-everything), shown for contrast — the new path bypasses it.
        from retrieval.rag.retriever import RERANK_POOL, CANDIDATE_CAP
        print("\n== Phase 3: query latency (median over 4 queries) ==")
        print(f"  {'frames/vid':>11} {'top_k':>6} {'ann(ms)':>9} {'fts(ms)':>9} "
              f"{'total(ms)':>10} {'legacy_scan(ms)':>16}")
        for vid, n in videos.items():
            for k in args.top_k:
                cand = min(max(k, RERANK_POOL) * 5, CANDIDATE_CAP)
                anns, ftss, totals, legacy = [], [], [], []
                for q in QUERIES:
                    anns.append(median_ms(lambda: store.search_by_text(q, embedder, vid, cand)))
                    ftss.append(median_ms(lambda: store.search_lexical(q, vid, cand)))
                    totals.append(median_ms(lambda: retriever.search(vid, q, top_k=k)))
                    legacy.append(median_ms(lambda: store.get_by_video_id(vid)))
                print(f"  {n:>11,} {k:>6} {statistics.median(anns):>9.1f} "
                      f"{statistics.median(ftss):>9.1f} {statistics.median(totals):>10.1f} "
                      f"{statistics.median(legacy):>16.1f}")

        # ── Phase 4: LLM context size ──────────────────────────
        print("\n== Phase 4: LLM context (text-only, cap 40 frames) ==")
        sample_vid = next(iter(videos))
        res = retriever.search(sample_vid, QUERIES[0], top_k=max(args.top_k))
        toks = est_prompt_tokens(store.get_by_video_id(sample_vid))
        print(f"  retrieved frames returned : {len(res)}")
        print(f"  est. prompt tokens (40 fr): ~{toks:,} (LLM never receives images)")

        assert total_bytes > 0 and res, "benchmark self-check failed"
        print("\nDone. (self-check passed)")
    finally:
        if not args.keep:
            shutil.rmtree(persist, ignore_errors=True)
            print(f"Removed throwaway dir {persist}")
        else:
            print(f"Kept throwaway dir {persist}")


if __name__ == "__main__":
    main()
