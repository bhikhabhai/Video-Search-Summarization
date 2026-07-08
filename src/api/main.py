"""Mini-VSS FastAPI application.

Endpoints:
    POST /ingest              multipart video upload -> {job_id, video_id}
    GET  /status/{job_id}     IngestStatus with progress 0-100
    POST /query               {video_id, question} -> text/event-stream answer
    GET  /summary/{video_id}  streamed chunked summary (text/event-stream)
    GET  /clip/{video_id}?q=  best matching frame (timestamp + thumbnail)
    GET  /frames/{video_id}   all keyframes with timestamps + thumbnails
    GET  /videos              registry of processed videos

The ingestion pipeline (extract -> detect -> caption -> embed -> store) runs
as an asyncio background task; CPU-bound inference steps execute inside a
single-worker thread pool so the event loop stays responsive. Progress is
tracked in an in-memory dict keyed by job_id.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator, Iterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from src.config import configure_logging, get_settings
from src.inference.captioner import Captioner
from src.inference.detector import Detector
from src.inference.embedder import Embedder
from src.ingestion.frame_extractor import FrameExtractor
from src.ingestion.video_loader import VideoLoadError, load_video
from src.models import (
    ExtractedFrame,
    FrameRecord,
    IngestStatus,
    JobState,
    QueryRequest,
    RetrievedFrame,
    VideoSummaryInfo,
)
from src.rag.retriever import HybridRetriever
from src.rag.summariser import LLMClient, Summariser, _format_window
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# In-memory job registry (single-process deployment by design).
JOBS: dict[str, IngestStatus] = {}

# One worker: the models are not thread-safe and a single GPU/CPU pipeline
# gains nothing from parallel videos; queued jobs run in FIFO order.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")


class Pipeline:
    """Lazily-constructed singletons shared by all requests."""

    def __init__(self) -> None:
        self.embedder = Embedder()
        self.detector = Detector()
        self.captioner = Captioner(embedder=self.embedder)
        self.extractor = FrameExtractor()
        self.store = VectorStore()
        self.retriever = HybridRetriever(self.store, self.embedder)
        self.llm = LLMClient()
        self.summariser = Summariser(self.llm)


PIPELINE: Pipeline | None = None


def pipeline() -> Pipeline:
    assert PIPELINE is not None, "app not started"
    return PIPELINE


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global PIPELINE
    configure_logging()
    PIPELINE = Pipeline()
    logger.info("Mini-VSS API ready")
    yield
    _EXECUTOR.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Mini-VSS", version="0.1.0", lifespan=lifespan)

# The Next.js frontend runs on a separate origin (e.g. http://localhost:3000)
# and calls this API directly from the browser, so permissive CORS is needed.
# This is a local, single-user portfolio deployment — no auth surface to protect.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Background ingestion ───────────────────────────────────────


def _thumbnail_b64(frame: ExtractedFrame, width: int = 480) -> str:
    img = frame.pil_image.copy()
    ratio = width / img.width
    img = img.resize((width, max(1, int(img.height * ratio))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def _run_ingest(job_id: str, video_path: str, filename: str) -> None:
    """Full pipeline for one video. Runs inside the ingest worker thread."""
    status = JOBS[job_id]
    p = pipeline()

    def update(progress: int, message: str) -> None:
        status.progress = progress
        status.message = message
        logger.info("[job %s] %d%% %s", job_id, progress, message)

    try:
        status.state = JobState.PROCESSING
        update(2, "probing video")
        meta = load_video(video_path, video_id=status.video_id)
        meta.filename = filename

        update(5, "extracting keyframes (scene-change detection)")
        frames = p.extractor.extract(meta)
        if not frames:
            raise RuntimeError("No keyframes could be extracted from the video")
        status.keyframe_count = len(frames)

        update(30, f"detecting objects in {len(frames)} keyframes")
        images = [f.pil_image for f in frames]
        detections = p.detector.detect_batch(images)

        update(55, "captioning keyframes (guided by detections)")
        captions = p.captioner.caption_frames(images, detections=detections)

        update(70, "detecting actions/interactions between people")
        actions_per_frame = p.captioner.caption_actions(images, detections=detections)

        update(85, "embedding keyframes")
        embeddings = p.embedder.embed_images(images)

        update(95, "writing to vector store")
        records = []
        for frame, caption, dets, actions in zip(frames, captions, detections, actions_per_frame):
            counts: dict[str, int] = {}
            for d in dets:
                counts[d.label] = counts.get(d.label, 0) + 1
            records.append(
                FrameRecord(
                    video_id=meta.video_id,
                    frame_idx=frame.frame_idx,
                    timestamp_sec=frame.timestamp_sec,
                    caption=caption,
                    detection_labels=sorted(counts.keys()),
                    detection_counts=counts,
                    actions=actions,
                    thumbnail_b64=_thumbnail_b64(frame),
                )
            )
        p.store.add_frames(records, embeddings)
        p.store.register_video(
            VideoSummaryInfo(
                video_id=meta.video_id,
                filename=meta.filename,
                duration_sec=meta.duration_sec,
                keyframe_count=len(records),
            )
        )

        status.state = JobState.DONE
        update(100, f"done: {len(records)} keyframes indexed")
    except Exception as exc:
        logger.exception("[job %s] ingestion failed", job_id)
        status.state = JobState.ERROR
        status.error = str(exc)
        status.message = "failed"


# A friendly assistant scoped to THIS video. Grounded in the observations, but
# conversational — it engages with greetings and follow-ups instead of robotically
# refusing, and only declines tasks that are clearly unrelated to the video
# (writing code, general trivia, system administration, etc.).
_QUERY_SYSTEM = (
    "You are a helpful assistant for a single video. Answer the user's questions "
    "about this video using the visual observations provided in the user message "
    "(timestamps, captions, detected objects with counts like '3× person', and "
    "actions). Guidelines:\n"
    "- Be conversational and friendly. Handle greetings and follow-up questions "
    "naturally (the user may say things like 'what about this' referring to the "
    "video — interpret them in the video's context).\n"
    "- Ground every factual claim in the observations and cite timestamps "
    "(e.g. 'at 12.5s'). If the observations don't cover something, say so plainly "
    "rather than guessing.\n"
    "- If a question is clearly unrelated to this video (e.g. writing code, general "
    "trivia, maths, shell commands), don't answer it — briefly say you're here to "
    "help with this video and invite a video-related question. Do not write code.\n"
    "- Keep answers concise."
)


# ── Endpoints ──────────────────────────────────────────────────


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict[str, str]:
    settings = get_settings()
    contents = await file.read()
    if len(contents) > settings.max_video_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_video_mb:.0f} MB limit")

    video_id = uuid.uuid4().hex[:12]
    job_id = uuid.uuid4().hex[:12]
    suffix = (file.filename or "video.mp4").rsplit(".", 1)[-1]
    dest = settings.uploads_dir / f"{video_id}.{suffix}"
    dest.write_bytes(contents)

    JOBS[job_id] = IngestStatus(job_id=job_id, video_id=video_id, message="queued")
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _EXECUTOR, _run_ingest, job_id, str(dest), file.filename or dest.name
    )
    logger.info("Queued job %s for video %s (%s)", job_id, video_id, file.filename)
    return {"job_id": job_id, "video_id": video_id}


@app.get("/status/{job_id}")
async def status(job_id: str) -> IngestStatus:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job {job_id}")
    return job


def _sse(generator: Iterator[str]) -> StreamingResponse:
    """Wrap a token generator as a Server-Sent-Events response."""

    async def event_stream() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        iterator = iter(generator)
        while True:
            token = await loop.run_in_executor(None, next, iterator, None)
            if token is None:
                break
            yield f"data: {json.dumps(token)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/query")
async def query(request: QueryRequest) -> StreamingResponse:
    p = pipeline()
    frames = await asyncio.get_running_loop().run_in_executor(
        None, p.retriever.search, request.video_id, request.question
    )
    if not frames:
        raise HTTPException(404, f"No indexed frames for video {request.video_id}")

    # Reuse the summariser's formatter so counts (e.g. "3× person") and actions
    # surface identically here — otherwise "how many people?" only ever sees the
    # deduplicated label list and undercounts.
    context = _format_window(frames)
    prompt = (
        "Visual observations from the video (most relevant first). Each line is one "
        "keyframe: its timestamp, a caption, the objects detected in it (with "
        "per-object counts like '3× person'), and any observed actions.\n"
        "When counting, trust the detected object counts over the caption wording — "
        "captions often say 'a person' even when several were detected.\n\n"
        f"{context}\n\n"
        f"Question: {request.question}\n\n"
        "Answer using ONLY the observations above. Cite timestamps (e.g. 'at 12.5s') "
        "for every claim."
    )

    def generate() -> Iterator[str]:
        # Ship the supporting frames first so the UI can render thumbnails.
        # These are the top-K retrieved matches (RETRIEVAL_TOP_K), not all
        # keyframes — retrieval deliberately returns only the most relevant.
        yield "\x00FRAMES:" + json.dumps(
            [f.model_dump(exclude={"thumbnail_b64"}) | {"thumbnail_b64": f.thumbnail_b64}
             for f in frames]
        )
        yield from p.llm.stream(prompt, system=_QUERY_SYSTEM)

    return _sse(generate())


@app.get("/summary/{video_id}")
async def summary(video_id: str) -> StreamingResponse:
    p = pipeline()
    frames = p.store.get_by_video_id(video_id)
    if not frames:
        raise HTTPException(404, f"No indexed frames for video {video_id}")
    return _sse(p.summariser.summarise(frames))


@app.get("/clip/{video_id}")
async def clip(video_id: str, q: str) -> dict:
    p = pipeline()
    hits = await asyncio.get_running_loop().run_in_executor(
        None, p.retriever.search, video_id, q, 1
    )
    if not hits:
        raise HTTPException(404, f"No match for '{q}' in video {video_id}")
    best = hits[0]
    return {
        "video_id": best.video_id,
        "timestamp_sec": best.timestamp_sec,
        "caption": best.caption,
        "thumbnail_b64": best.thumbnail_b64,
        "score": best.score,
    }


@app.get("/frames/{video_id}")
async def frames(video_id: str) -> list[RetrievedFrame]:
    result = pipeline().store.get_by_video_id(video_id)
    if not result:
        raise HTTPException(404, f"No indexed frames for video {video_id}")
    return result


@app.get("/videos")
async def videos() -> list[VideoSummaryInfo]:
    return pipeline().store.list_videos()


@app.delete("/videos/{video_id}")
async def delete_video(video_id: str) -> dict:
    p = pipeline()
    deleted = p.store.delete_video(video_id)
    if deleted == 0 and not any(
        v.video_id == video_id for v in p.store.list_videos()
    ):
        # Nothing in Chroma and nothing in the registry -> genuinely unknown.
        raise HTTPException(404, f"Unknown video {video_id}")

    # Best-effort removal of the uploaded source file(s).
    settings = get_settings()
    for path in settings.uploads_dir.glob(f"{video_id}.*"):
        try:
            path.unlink()
        except OSError:
            logger.warning("Could not delete upload %s", path)

    return {"video_id": video_id, "deleted_frames": deleted}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
