#!/usr/bin/env python3
"""End-to-end benchmark of ONE real video through the running Mini-VSS API.

Answers the questions `scale_bench.py` cannot: how long does *my* video actually
take, which stage eats that time, how much disk does it leave behind, and can
this hardware keep up with a live camera?

`scale_bench.py` fabricates records to stress the DB and query path. It skips
tracking, weapon detection, the caption VLM and CLIP entirely — which is where
essentially all the wall-clock time goes. This script drives the real pipeline
by posting to a running API, so the numbers include the models.

Runs on the **host** (stdlib only — no venv needed) because that is where
``./data`` is bind-mounted and where the API port is published.

    docker compose up -d
    python3 benchmarks/real_bench.py my_video.mp4

Each upload gets a fresh ``video_id``, so re-running never overwrites an earlier
result — pass --cleanup to delete the benchmark video when it finishes.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

DEFAULT_QUERIES = [
    "a person walking",
    "when did someone enter",
    "any vehicle in the scene",
]

# Per-video artefacts, relative to data_dir. `chroma` is shared across all
# videos so it is measured as a before/after delta instead.
PER_VIDEO_DIRS = {
    "thumbs": "thumbs/{vid}",
    "tracks json": "tracks/{vid}.json",
    "tracks video": "tracks_video/{vid}.mp4",
}


def dir_size(path: Path) -> int:
    """Total bytes under `path` (a file or a directory); 0 if missing."""
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def upload(api: str, video: Path) -> dict:
    """POST the video to /ingest as streamed multipart; return {job_id, video_id}.

    Uses http.client directly so the file is streamed from disk rather than read
    into memory — uploads can legitimately be hundreds of MB.
    """
    parsed = urllib.parse.urlparse(api)
    boundary = "----vssbench" + uuid.uuid4().hex
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{video.name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()

    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=600)
    conn.putrequest("POST", "/ingest")
    conn.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
    conn.putheader("Content-Length", str(len(prefix) + video.stat().st_size + len(suffix)))
    conn.endheaders()
    conn.send(prefix)
    with video.open("rb") as fh:
        conn.send(fh)
    conn.send(suffix)

    resp = conn.getresponse()
    body = resp.read()
    if resp.status != 200:
        raise SystemExit(f"/ingest failed ({resp.status}): {body[:400].decode(errors='replace')}")
    return json.loads(body)


_COUNTER_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*$")


def _eta(msg: str, stage_sec: float) -> str:
    """"  ~27.3 s/item, ETA 9.1 min" when the message carries an N/M counter."""
    m = _COUNTER_RE.search(msg)
    if not m:
        return ""
    done, total = int(m.group(1)), int(m.group(2))
    if done < 1 or stage_sec <= 0:
        return ""
    rate = stage_sec / done
    return f"  ~{rate:.1f} s/item, ETA {(total - done) * rate / 60:.1f} min"


def poll(api: str, job_id: str, interval: float,
         heartbeat: float = 15.0) -> tuple[float, list[tuple[float, str]]]:
    """Block until the job ends. Return (total_sec, [(start_sec, stage), ...]).

    Progress messages like "captioning keyframes 7/120" tick once per frame, so
    stages are keyed on the message with digits stripped — all caption ticks
    collapse into a single row in the breakdown.

    That collapsing must NOT silence the live output: captioning a full video can
    run for the better part of an hour, and no output for that long reads as a
    hang. So a throttled heartbeat (with an ETA, when the message carries an N/M
    counter) still prints inside a long-running stage.
    """
    t0 = time.perf_counter()
    stages: list[tuple[float, str]] = []
    last_key: str | None = None
    stage_start = t0
    last_beat = 0.0
    tty = sys.stdout.isatty()

    while True:
        st = get_json(f"{api}/status/{job_id}")
        msg = st.get("message", "")
        now = time.perf_counter()
        elapsed = now - t0
        key = re.sub(r"\d+", "", msg).strip()

        if key and key != last_key:
            if tty and last_key is not None:
                print()  # close off the heartbeat line
            stages.append((elapsed, msg))
            last_key, stage_start, last_beat = key, now, elapsed
            print(f"    {elapsed:7.1f}s  {st.get('progress', 0):3d}%  {msg}", flush=True)
        elif elapsed - last_beat >= heartbeat:
            last_beat = elapsed
            line = (f"    {elapsed:7.1f}s  {st.get('progress', 0):3d}%  "
                    f"{msg}{_eta(msg, now - stage_start)}")
            # Overwrite in place on a terminal; plain lines when piped to a file.
            print(line.ljust(78), end="\r" if tty else "\n", flush=True)

        if st.get("state") == "error":
            raise SystemExit(f"ingest failed: {st.get('error')}")
        if st.get("state") == "done":
            if tty:
                print()
            return elapsed, stages
        time.sleep(interval)


def time_queries(api: str, video_id: str, queries: list[str]) -> list[float]:
    """Median-of-3 wall time per query against /clip (retrieval only, no LLM)."""
    out = []
    for q in queries:
        url = f"{api}/clip/{video_id}?q={urllib.parse.quote(q)}"
        times = []
        for _ in range(3):
            t = time.perf_counter()
            get_json(url)
            times.append((time.perf_counter() - t) * 1000)
        times.sort()
        out.append(times[1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path, help="video file to benchmark")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--data-dir", type=Path, default=Path("data"),
                    help="host path of the bind-mounted data dir (default: data)")
    ap.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    ap.add_argument("--poll-interval", type=float, default=0.5,
                    help="status poll period; also the stage-timing resolution")
    ap.add_argument("--cleanup", action="store_true",
                    help="DELETE the benchmark video from the store when done")
    args = ap.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"no such video: {args.video}")
    chroma = args.data_dir / "chroma"
    if not chroma.is_dir():
        raise SystemExit(f"{chroma} not found — run from the repo root or pass --data-dir")
    get_json(f"{args.api}/health")  # fail fast if the API is down

    source_bytes = args.video.stat().st_size
    chroma_before = dir_size(chroma)

    print(f"\n== Ingesting {args.video.name} ({human(source_bytes)}) ==")
    job = upload(args.api, args.video)
    vid = job["video_id"]
    print(f"  video_id={vid}  job_id={job['job_id']}")
    total_sec, stages = poll(args.api, job["job_id"], args.poll_interval)

    info = next((v for v in get_json(f"{args.api}/videos") if v["video_id"] == vid), {})
    duration = float(info.get("duration_sec") or 0.0)
    keyframes = int(info.get("keyframe_count") or 0)

    # ── Time ───────────────────────────────────────────────────
    print(f"\n== Time ==")
    print(f"  video duration   : {duration:.1f} s")
    print(f"  keyframes        : {keyframes}")
    print(f"  total ingest     : {total_sec:.1f} s")
    if keyframes:
        print(f"  per keyframe     : {total_sec / keyframes:.2f} s")
    if duration:
        rt = total_sec / duration
        print(f"  realtime factor  : {rt:.2f}x  "
              f"({'SLOWER' if rt > 1 else 'faster'} than the camera records)")

    print("\n  stage breakdown:")
    for i, (start, msg) in enumerate(stages):
        end = stages[i + 1][0] if i + 1 < len(stages) else total_sec
        dur = end - start
        print(f"    {dur:8.1f} s  {dur / total_sec * 100:5.1f}%  {msg}")

    # ── Space ──────────────────────────────────────────────────
    # Chroma is shared, so attribute only its growth to this video. Compaction
    # elsewhere can make this a slight under-count; the per-video dirs are exact.
    chroma_delta = dir_size(chroma) - chroma_before
    parts = {"chroma (delta)": chroma_delta}
    for label, tmpl in PER_VIDEO_DIRS.items():
        parts[label] = dir_size(args.data_dir / tmpl.format(vid=vid))
    index_total = sum(parts.values())

    print(f"\n== Space ==")
    for label, n in parts.items():
        per_kf = f"{human(n / keyframes):>10} /keyframe" if keyframes else ""
        print(f"  {label:<16} : {human(n):>10}   {per_kf}")
    print(f"  {'-' * 46}")
    print(f"  {'index total':<16} : {human(index_total):>10}   "
          f"{human(index_total / keyframes) + ' /keyframe' if keyframes else ''}")
    print(f"  {'source video':<16} : {human(source_bytes):>10}   (kept in data/uploads)")

    # ── Retrieval ──────────────────────────────────────────────
    times = time_queries(args.api, vid, args.queries)
    times.sort()
    print(f"\n== Retrieval (/clip — retrieval only, no LLM) ==")
    print(f"  median over {len(times)} queries: {times[len(times) // 2]:.1f} ms")

    # ── Extrapolation ──────────────────────────────────────────
    # Straight-line from this clip's rates. Honest for storage; optimistic for
    # time, which assumes no thermal throttling and a constantly-busy scene.
    if duration and keyframes:
        ninety_days = 90 * 24 * 3600
        scale = ninety_days / duration
        print(f"\n== 90-day extrapolation (this clip's rates, 1 camera) ==")
        print(f"  keyframes        : ~{keyframes * scale / 1e6:.1f} M")
        print(f"  index size       : ~{human(index_total * scale)}")
        print(f"  source video     : ~{human(source_bytes * scale)}")
        ingest_days = total_sec * scale / 86400
        flag = "  <-- EXCEEDS 90 days of footage" if ingest_days > 90 else ""
        print(f"  ingest wall time : ~{ingest_days:.0f} days{flag}")

    if args.cleanup:
        req = urllib.request.Request(f"{args.api}/videos/{vid}", method="DELETE")
        urllib.request.urlopen(req, timeout=60).read()
        print(f"\nDeleted benchmark video {vid}.")
    else:
        print(f"\nKept as {vid}. Remove with:"
              f"\n  curl -X DELETE {args.api}/videos/{vid}")


if __name__ == "__main__":
    main()
