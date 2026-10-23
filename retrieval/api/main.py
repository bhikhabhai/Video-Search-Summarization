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
import difflib
import io
import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator, Iterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, ImageDraw

from perception.config import configure_logging, get_settings
from perception.inference.captioner import Captioner
from perception.inference.detector import Detector
from perception.inference.embedder import Embedder
from perception.inference.ocr import OCRReader
from perception.inference.pose import PoseEstimator
from perception.inference.weapon import WeaponDetector
from perception.ingestion.frame_extractor import FrameExtractor
from perception.ingestion.iva_loader import IVALoader
from perception.ingestion.tracker import VideoTracker
from perception.ingestion.video_loader import VideoLoadError, load_video
from perception.models import (
    ExtractedFrame,
    FrameRecord,
    IngestStatus,
    JobState,
    QueryRequest,
    RetrievedFrame,
    VideoSummaryInfo,
)
from retrieval.rag import track_store
from retrieval.rag.retriever import HybridRetriever, _tokenize
from retrieval.rag.summariser import LLMClient, Summariser, _format_window
from retrieval.rag.vector_store import VectorStore

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
        self.ocr = OCRReader()
        self.pose = PoseEstimator()
        self.weapon = WeaponDetector()
        self.extractor = FrameExtractor()
        self.tracker = VideoTracker()
        self.iva = IVALoader()
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
    _purge_orphans(PIPELINE.store)
    logger.info("Mini-VSS API ready")
    yield
    _EXECUTOR.shutdown(wait=False, cancel_futures=True)


def _delete_video_files(video_id: str) -> None:
    """Best-effort removal of a video's on-disk files (upload + IVA sidecars,
    track summary, annotated tracking video)."""
    settings = get_settings()
    for path in settings.uploads_dir.glob(f"{video_id}.*"):
        try:
            path.unlink()
        except OSError:
            logger.warning("Could not delete upload %s", path)
    track_store.delete_tracks(video_id)
    (settings.data_dir / "tracks_video" / f"{video_id}.mp4").unlink(missing_ok=True)


def _purge_orphans(store) -> None:
    """Delete data left by ingests that never registered (crash / restart
    mid-job). Runs at startup, before any job can be in flight."""
    settings = get_settings()
    known = {v.video_id for v in store.list_videos()}
    on_disk = (
        list(settings.uploads_dir.glob("*"))
        + list((settings.data_dir / "tracks_video").glob("*.mp4"))
        + list((settings.data_dir / "tracks").glob("*.json"))
        + [d for d in store.thumbs_dir.glob("*") if d.is_dir()]
    )
    ids = {p.name.split(".", 1)[0] for p in on_disk}
    orphans = {i for i in ids if re.fullmatch(r"[0-9a-f]{12}", i)} - known
    for video_id in orphans:
        store.delete_video(video_id)
        _delete_video_files(video_id)
    if orphans:
        logger.info("Purged %d orphaned video(s): %s", len(orphans), sorted(orphans))


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


def _thumbnail_bytes(frame: ExtractedFrame, width: int = 480) -> bytes:
    img = frame.pil_image.copy()
    ratio = width / img.width
    img = img.resize((width, max(1, int(img.height * ratio))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# Synonyms/plurals so "people"/"men" match a "person" box, "cars" a "vehicle", etc.
_LABEL_ALIASES: dict[str, set[str]] = {
    "person": {"person", "persons", "people", "human", "humans", "man", "men",
               "woman", "women", "pedestrian", "pedestrians", "guy", "guys",
               "someone", "somebody", "individual", "individuals"},
    "vehicle": {"vehicle", "vehicles", "car", "cars", "truck", "trucks", "bus",
                "buses", "van", "vans", "automobile", "automobiles"},
    "car": {"car", "cars", "vehicle", "vehicles", "sedan", "automobile", "auto"},
    "truck": {"truck", "trucks", "lorry", "lorries", "vehicle", "vehicles"},
    "bus": {"bus", "buses", "vehicle", "vehicles"},
    "motorcycle": {"motorcycle", "motorcycles", "motorbike", "motorbikes",
                   "bike", "bikes", "scooter", "vehicle", "vehicles"},
    "bicycle": {"bicycle", "bicycles", "bike", "bikes", "cycle", "cycles"},
    "backpack": {"backpack", "backpacks", "bag", "bags", "luggage"},
    "handbag": {"handbag", "handbags", "bag", "bags", "purse"},
    "suitcase": {"suitcase", "suitcases", "bag", "bags", "luggage"},
    # Weapon classes (handgun / long_gun / knife) from the weapon detector.
    "handgun": {"handgun", "handguns", "gun", "guns", "pistol", "pistols",
                "firearm", "firearms", "weapon", "weapons"},
    "long_gun": {"long_gun", "longgun", "rifle", "rifles", "shotgun", "shotguns",
                 "gun", "guns", "firearm", "firearms", "weapon", "weapons"},
    "knife": {"knife", "knives", "blade", "blades", "weapon", "weapons"},
}


def _label_aliases(label: str) -> set[str]:
    l = label.lower()
    return _LABEL_ALIASES.get(l, {l, l + "s"})  # default: word + naive plural


# Words that signal the question is about an analytics event, not an object.
_EVENT_QUERY_WORDS = {
    "event", "events", "alert", "alerts", "tripwire", "trip", "wire", "line",
    "cross", "crossed", "crossing", "loiter", "loitering", "intrusion", "intrude",
    "entered",
    "enter", "leaving", "left", "fall", "fell", "fallen", "fight", "fighting",
    "running", "run", "fire", "smoke", "abandoned", "unattended", "crowd",
    "crowding", "wrong", "happened", "happening", "parking", "parked", "park",
    "zone", "zones", "roi", "region",
    # Weapon terms — so "any weapons?"/"gun" highlights the weapon detections.
    "weapon", "weapons", "gun", "guns", "firearm", "firearms", "pistol",
    "handgun", "rifle", "rifles", "shotgun", "knife", "knives", "armed",
    # Temporal pose actions.
    "handshake", "shaking", "hands", "handover", "sudden", "movement",
    "anomaly", "waving", "punch",
}
_GENERIC_EVENT_WORDS = {"event", "events", "alert", "alerts", "happened", "happening"}


def _boxed_thumbnail(
    frame: RetrievedFrame, question: str, regions: list[dict] | None = None
) -> str:
    """Draw boxes on the objects the question refers to — either by object label
    ("people", "cars") or by analytics event ("tripwire", "loitering") — plus the
    analytics ROIs (tripwire lines / zones) on event-related questions.

    Object matches are green, event matches amber (labelled with the event
    phrase, box found via the event's triggering track id), ROIs red. If nothing
    matches, nothing is drawn. All coords are normalised [0,1].
    """
    q_tokens = set(_tokenize(question))
    event_query = bool(q_tokens & _EVENT_QUERY_WORDS)
    draw_regions = event_query and bool(regions)
    if not frame.detections and not draw_regions:
        return frame.thumbnail_b64

    # Object matches (green).
    obj_matches = [d for d in frame.detections if _label_aliases(d.label) & q_tokens]

    # Event matches (amber): question uses event words AND the frame has events.
    event_matches: list[tuple] = []  # (detection, label_text)
    if event_query and frame.frame_events:
        by_id = {d.track_id: d for d in frame.detections if d.track_id}
        generic = bool(q_tokens & _GENERIC_EVENT_WORDS)
        for ev in frame.frame_events:
            phrase = str(ev.get("phrase", ""))
            specific = set(_tokenize(phrase)) & _EVENT_QUERY_WORDS
            if generic or (q_tokens & specific):
                det = by_id.get(ev.get("id"))
                if det is not None:
                    # Drop the leading "person_P0002 " id from the box label.
                    text = phrase[len(det.track_id):].strip() if phrase.startswith(det.track_id) else phrase
                    event_matches.append((det, text or "event"))

    if not obj_matches and not event_matches and not draw_regions:
        return frame.thumbnail_b64
    try:
        img = Image.open(io.BytesIO(base64.b64decode(frame.thumbnail_b64))).convert("RGB")
    except Exception:
        return frame.thumbnail_b64
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # Analytics ROIs (red): tripwire lines with a direction arrow, zone polygons.
    if draw_regions:
        for reg in regions:
            pts = [(round(p[0] * w), round(p[1] * h)) for p in reg.get("points", [])]
            if len(pts) < 2:
                continue
            if reg.get("kind") == "line":
                draw.line(pts, fill=(235, 40, 40), width=3)
            else:  # polygon zone
                draw.polygon(pts, outline=(235, 40, 40))
            draw.text((pts[0][0] + 3, max(0, pts[0][1] - 12)),
                      str(reg.get("name", "ROI")), fill=(235, 40, 40))

    event_ids = {id(d) for d, _ in event_matches}
    for d in obj_matches:
        if id(d) in event_ids:
            continue  # drawn below with its event label instead
        x1, y1, x2, y2 = d.bbox
        box = (x1 * w, y1 * h, x2 * w, y2 * h)
        draw.rectangle(box, outline=(16, 220, 120), width=3)
        draw.text((box[0] + 3, max(0, box[1] - 12)), d.label, fill=(16, 220, 120))
    for d, text in event_matches:
        x1, y1, x2, y2 = d.bbox
        box = (x1 * w, y1 * h, x2 * w, y2 * h)
        draw.rectangle(box, outline=(245, 170, 30), width=4)
        draw.text((box[0] + 3, max(0, box[1] - 12)), text[:48], fill=(245, 170, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _merge_ocr(ocr_lines: list[str], vlm_ocr: str) -> str:
    """Combine dedicated-OCR text with the VLM's OCR, deduped, OCR-model first."""
    seen: set[str] = set()
    parts: list[str] = []
    vlm_items = [s.strip() for s in vlm_ocr.split(";")] if vlm_ocr else []
    for item in [*ocr_lines, *vlm_items]:
        item = item.strip()
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            parts.append(item)
    return "; ".join(parts)


def _merge_events(pose_events: list[str], vlm_events: list[str]) -> list[str]:
    """Union of rule-based pose events and VLM events, deduped (pose first)."""
    seen: set[str] = set()
    merged: list[str] = []
    for event in [*pose_events, *vlm_events]:
        event = str(event).strip()
        key = event.lower()
        if event and key not in seen:
            seen.add(key)
            merged.append(event)
    return merged


def _run_ingest(
    job_id: str, video_path: str, filename: str,
    meta_path: str | None = None, manifest_path: str | None = None,
) -> None:
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

        if meta_path:
            update(5, "loading IVA output (detections / tracking / plates)")
            frames, detections, track_facts, kf_events, track_summary = p.iva.process(
                meta, meta_path, manifest_path
            )
            track_store.write_tracks(meta.video_id, track_summary)
        elif get_settings().keyframe_mode == "track":
            update(5, "tracking objects across every frame (ByteTrack)")
            frames, detections, track_facts, kf_events, track_summary = p.tracker.process(meta)
            track_store.write_tracks(meta.video_id, track_summary)
        else:
            update(5, "extracting keyframes (scene-change detection)")
            frames = p.extractor.extract(meta)
            detections = p.detector.detect_batch([f.pil_image for f in frames]) if frames else []
            track_facts = ["" for _ in frames]
            kf_events = [[] for _ in frames]
        if not frames:
            raise RuntimeError("No keyframes could be extracted from the video")
        status.keyframe_count = len(frames)
        images = [f.pil_image for f in frames]

        # Pose events (optional) — run up front (at "tracker time"), then free
        # the pose model from VRAM so the caption VLM gets that memory back.
        if p.pose.enabled:
            update(48, "detecting pose-based events (fall/fight/running)")
            pose_events_per_frame = p.pose.events_batch(images)
            p.pose.unload()
        else:
            pose_events_per_frame = [[] for _ in images]

        # Weapon detection (optional) — before captioning so the VLM is told a
        # weapon is present; results are merged into detections/events below.
        if p.weapon.enabled:
            update(50, "detecting weapons")
        weapon_dets_per_frame = p.weapon.detect_batch(images)
        for i, wdets in enumerate(weapon_dets_per_frame):
            if wdets:
                labels = ", ".join(sorted({w.label for w in wdets}))
                track_facts[i] = (f"{track_facts[i]}; " if track_facts[i] else "") + \
                    f"WEAPON DETECTED: {labels}"

        update(55, "describing keyframes (scene/actions/OCR/events, guided by detections)")

        # Keep /status responsive during long captioning runs (especially CPU Qwen)
        # by mapping per-frame completion to a 55->74 progress span.
        def on_caption_progress(done: int, total: int) -> None:
            if total <= 0:
                return
            span = 74 - 55
            pct = 55 + int((done / total) * span)
            update(max(status.progress, pct), f"captioning keyframes {done}/{total}")

        descriptions = p.captioner.describe_frames(
            images,
            detections=detections,
            context=track_facts,
            progress_cb=on_caption_progress,
        )

        if p.ocr.enabled:
            update(75, f"reading text/plates ({p.ocr.backend} OCR)")
            ocr_per_frame = p.ocr.read_batch(images)
        else:
            ocr_per_frame = [[] for _ in images]

        update(85, "embedding keyframes")
        embeddings = p.embedder.embed_images(images)

        update(95, "writing to vector store")
        records = []
        for frame, desc, dets, ocr_lines, pose_events, fevents, wdets in zip(
            frames, descriptions, detections, ocr_per_frame, pose_events_per_frame,
            kf_events, weapon_dets_per_frame,
        ):
            # Merge weapon hits into this frame's detections + events, so they
            # draw, count, appear in the event timeline, and enter the caption.
            if wdets:
                dets = list(dets) + wdets
                fevents = list(fevents) + [
                    {"id": w.track_id, "phrase": f"weapon detected: {w.label}"}
                    for w in wdets
                ]
            counts: dict[str, int] = {}
            for d in dets:
                counts[d.label] = counts.get(d.label, 0) + 1
            # Fold analytics events into the caption + events deterministically,
            # so event frames are properly captioned and searchable regardless of
            # whether the VLM echoed the event.
            event_phrases = [str(ev.get("phrase", "")).strip()
                             for ev in fevents if ev.get("phrase")]
            caption = desc.get("caption", "")
            if event_phrases:
                ev_text = "; ".join(event_phrases)
                caption = (f"{caption} " if caption else "") + f"Analytics event: {ev_text}."
            records.append(
                FrameRecord(
                    video_id=meta.video_id,
                    frame_idx=frame.frame_idx,
                    timestamp_sec=frame.timestamp_sec,
                    caption=caption,
                    detection_labels=sorted(counts.keys()),
                    detection_counts=counts,
                    actions=desc.get("actions", []),
                    scene=desc.get("scene", ""),
                    # Dedicated OCR takes precedence; fall back to the VLM's OCR.
                    ocr_text=_merge_ocr(ocr_lines, desc.get("ocr_text", "")),
                    # Pose events + VLM events + the analytics event phrases.
                    events=_merge_events(pose_events, desc.get("events", []) + event_phrases),
                    # Colour phrases straight from the VLM ("orange backpack").
                    attributes=desc.get("items", []),
                    detections=dets,
                    # Analytics events at this keyframe (with triggering track id).
                    frame_events=fevents,
                    # Thumbnail lives on disk (below), not in the vector DB.
                    thumbnail_b64="",
                )
            )
            p.store.save_thumbnail(
                meta.video_id, frame.frame_idx, _thumbnail_bytes(frame)
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
        # Never registered -> invisible in the UI, so drop everything it wrote.
        p.store.delete_video(status.video_id)
        _delete_video_files(status.video_id)
        status.message = "failed"


# A friendly assistant scoped to THIS video. Grounded in the observations, but
# conversational — it engages with greetings and follow-ups instead of robotically
# refusing, and only declines tasks that are clearly unrelated to the video
# (writing code, general trivia, system administration, etc.).
_QUERY_SYSTEM = (
    "You are a friendly assistant that helps an everyday, NON-TECHNICAL person "
    "understand a video. You are given machine-generated observations for keyframes "
    "(timestamps, captions, detected objects with counts, on-screen text, actions) — "
    "but the user must NEVER see that raw format.\n"
    "\n"
    "How to answer:\n"
    "- Reply in plain, natural sentences, the way a person would describe the video "
    "out loud. Do NOT paste or quote the observations, and NEVER use the internal "
    "notation such as 't=11.5s', '[objects: ...]', '1× backpack', 'People:', or "
    "'Vehicles:'. Translate all of it into everyday language.\n"
    "- Refer to time naturally, e.g. 'around 11 seconds in' or 'at about 47 seconds'.\n"
    "- When a caption gives an attribute (colour, size) and the detected objects "
    "name the item, merge them into one thing — a 'green bag' plus a detected "
    "'backpack' becomes 'a green backpack'.\n"
    "- When asked to find or list when something happens, mention EVERY time it "
    "appears, in order — don't stop at the first.\n"
    "- Ground everything in the observations; if they don't show something, say so "
    "simply. Never invent details. Handle greetings and follow-ups naturally.\n"
    "- Be warm, clear, and concise.\n"
    "- If the question is clearly unrelated to this video (coding, trivia, maths, "
    "shell commands), politely say you can only help with this video. Don't write code.\n"
    "\n"
    "GOOD answer: \"Yes — someone's carrying a backpack twice: around 11 seconds and "
    "again near 47 seconds. Both times they're walking outdoors past several parked "
    "cars.\"\n"
    "BAD answer (never do this): quoting 't=11.5s: ... [objects: 1× backpack, 5× car]'."
)


# Query words too common to drive an exhaustive-search filter.
_ENUM_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "have", "has",
    "had", "do", "does", "did", "when", "where", "what", "which", "who", "show",
    "find", "list", "time", "times", "video", "frame", "frames", "see", "there",
    "that", "this", "with", "and", "or", "of", "in", "on", "at", "to", "for",
    "all", "any", "every", "someone", "anyone", "people", "person",
}

# Cap on frames fed to the LLM / UI for one query (enumeration can be broad).
_QUERY_FRAME_CAP = 40


def _frame_tokens(frame: RetrievedFrame) -> set[str]:
    blob = " ".join([
        frame.caption,
        " ".join(frame.detection_labels),
        frame.ocr_text,
        " ".join(frame.events),
        " ".join(frame.actions),
        " ".join(frame.attributes),
    ])
    return set(_tokenize(blob))


def _expand_query_terms(query: str, vocab_df: dict[str, int], n_frames: int) -> set[str]:
    """Map query words to the video's actual vocabulary, tolerating typos.

    Handles misspellings ("bagpack"), split words ("back pack" → "backpack"),
    and variants by fuzzy-matching (edit-distance via difflib) against the terms
    that actually occur in the video. Ubiquitous terms (in >60% of frames, e.g.
    "person") are dropped so they don't select the whole video.
    """
    raw = [t for t in _tokenize(query) if len(t) >= 3 and t not in _ENUM_STOPWORDS]
    # Adjacent joins catch split words: "back pack" / "bag pak" -> "backpack".
    joins = [raw[i] + raw[i + 1] for i in range(len(raw) - 1)]
    vocab = list(vocab_df)

    terms: set[str] = set()
    for candidate in raw + joins:
        if candidate in vocab_df:
            terms.add(candidate)
            continue
        # Fuzzy: nearest vocabulary words (0.7 ≈ one or two typos).
        for match in difflib.get_close_matches(candidate, vocab, n=2, cutoff=0.7):
            terms.add(match)

    # Drop only near-ubiquitous words (in >85% of frames, e.g. "person"); an
    # object in a moderate fraction of frames is still worth enumerating.
    limit = max(1, int(0.85 * n_frames))
    return {t for t in terms if 1 <= vocab_df.get(t, 0) <= limit}


def _enumeration_candidates(
    all_frames: list[RetrievedFrame], query: str, cap: int = _QUERY_FRAME_CAP
) -> list[RetrievedFrame]:
    """Every frame whose observations contain a *discriminative* query term.

    Powers "find all times X appears" queries, which pure top-K ranking misses,
    and is robust to typos/variants via :func:`_expand_query_terms`.
    """
    if not all_frames:
        return []
    frame_tokens = [_frame_tokens(f) for f in all_frames]
    vocab_df: dict[str, int] = {}
    for ft in frame_tokens:
        for token in ft:
            vocab_df[token] = vocab_df.get(token, 0) + 1

    terms = _expand_query_terms(query, vocab_df, len(all_frames))
    if not terms:
        return []
    matches = [f for f, ft in zip(all_frames, frame_tokens) if terms & ft]
    matches.sort(key=lambda f: f.timestamp_sec)
    return matches[:cap]


def _merge_frames(
    primary: list[RetrievedFrame], extra: list[RetrievedFrame], cap: int = _QUERY_FRAME_CAP
) -> list[RetrievedFrame]:
    """Union two frame lists (dedup by id), chronological, capped."""
    seen: set[tuple[str, int]] = set()
    out: list[RetrievedFrame] = []
    for f in [*primary, *extra]:
        key = (f.video_id, f.frame_idx)
        if key not in seen:
            seen.add(key)
            out.append(f)
    out.sort(key=lambda f: f.timestamp_sec)
    return out[:cap]


def _frame_matches_event(frame: RetrievedFrame, q_tokens: set[str], generic: bool) -> bool:
    for ev in frame.frame_events or []:
        spec = set(_tokenize(str(ev.get("phrase", "")))) & _EVENT_QUERY_WORDS
        if generic or (q_tokens & spec):
            return True
    return False


def _frame_matches_words(frame: RetrievedFrame, q_tokens: set[str]) -> bool:
    blob = set(_tokenize(" ".join([
        frame.caption, " ".join(frame.detection_labels), " ".join(frame.attributes),
        frame.ocr_text, " ".join(frame.actions),
    ])))
    if q_tokens & blob:
        return True
    return any(_label_aliases(l) & q_tokens for l in frame.detection_labels)


def _select_display_frames(
    all_frames: list[RetrievedFrame],
    fallback: list[RetrievedFrame],
    question: str,
) -> list[RetrievedFrame]:
    """Frames to SHOW for a query — relevant, not random.

    Event question -> the actual keyframes where that event fired.
    Otherwise      -> keyframes whose observations contain the query words.
    If neither matches, fall back to the semantic top-K (better than empty).
    """
    q = set(_tokenize(question))
    if q & _EVENT_QUERY_WORDS:
        generic = bool(q & _GENERIC_EVENT_WORDS)
        evs = [f for f in all_frames if _frame_matches_event(f, q, generic)]
        if evs:
            return sorted(evs, key=lambda f: f.timestamp_sec)[:_QUERY_FRAME_CAP]
    words = [f for f in all_frames if _frame_matches_words(f, q)]
    if words:
        return sorted(words, key=lambda f: f.timestamp_sec)[:_QUERY_FRAME_CAP]
    return fallback


# ── Endpoints ──────────────────────────────────────────────────


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    meta: UploadFile | None = File(None),
    manifest: UploadFile | None = File(None),
) -> dict[str, str]:
    """Ingest a video. Optionally attach an IVA per-frame ``meta.jsonl`` (drives
    detection/tracking/plates instead of YOLO) and a ``manifest.jsonl`` (ROI
    geometry — tripwire lines / zones drawn on retrieved frames)."""
    settings = get_settings()
    contents = await file.read()
    if len(contents) > settings.max_video_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_video_mb:.0f} MB limit")

    video_id = uuid.uuid4().hex[:12]
    job_id = uuid.uuid4().hex[:12]
    suffix = (file.filename or "video.mp4").rsplit(".", 1)[-1]
    dest = settings.uploads_dir / f"{video_id}.{suffix}"
    dest.write_bytes(contents)

    meta_path = None
    if meta is not None:
        meta_dest = settings.uploads_dir / f"{video_id}.meta.jsonl"
        meta_dest.write_bytes(await meta.read())
        meta_path = str(meta_dest)

    manifest_path = None
    if manifest is not None:
        man_dest = settings.uploads_dir / f"{video_id}.manifest.jsonl"
        man_dest.write_bytes(await manifest.read())
        manifest_path = str(man_dest)

    JOBS[job_id] = IngestStatus(job_id=job_id, video_id=video_id, message="queued")
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _EXECUTOR, _run_ingest, job_id, str(dest), file.filename or dest.name,
        meta_path, manifest_path,
    )
    logger.info("Queued job %s for video %s (%s, iva=%s, manifest=%s)",
                job_id, video_id, file.filename, bool(meta_path), bool(manifest_path))
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
    loop = asyncio.get_running_loop()
    frames = await loop.run_in_executor(
        None,
        lambda: p.retriever.search(
            request.video_id, request.question,
            t_start=request.t_start, t_end=request.t_end,
        ),
    )
    if not frames:
        raise HTTPException(404, f"No indexed frames for video {request.video_id}")

    # Exhaustive recall: add every frame whose observations contain a
    # discriminative query term, so "find all times X" doesn't stop at top-K.
    all_frames = await loop.run_in_executor(
        None, p.store.get_by_video_id, request.video_id
    )
    frames = _merge_frames(frames, _enumeration_candidates(all_frames, request.question))

    # Reuse the summariser's formatter so counts (e.g. "3× person") and actions
    # surface identically here.
    context = _format_window(frames)
    # Frames to SHOW: relevant to the query (event frames / word matches), not the
    # loosely-related semantic top-K. The LLM context keeps the fuller set above.
    display_frames = _select_display_frames(all_frames, frames, request.question)
    # Whole-video unique counts from tracking (precise; deduped across frames).
    tracks = track_store.read_tracks(request.video_id)
    regions = (tracks or {}).get("regions") or []  # IVA ROI geometry, for drawing
    counts_line = ""
    if tracks and tracks.get("counts"):
        pretty = ", ".join(f"{n} unique {cls}(s)" for cls, n in tracks["counts"].items())
        counts_line = (
            "Whole-video unique object counts (from tracking, already deduplicated "
            f"across frames — use these for 'how many unique X' questions): {pretty}.\n\n"
        )
    if tracks and tracks.get("events"):
        # Deterministic event timeline (entered/left/loitering/tripwire/…) — the
        # trustworthy source for "did anyone cross/loiter/enter" questions.
        ev = "; ".join(f"at {e['t']:.0f}s {e['event']}" for e in tracks["events"][:60])
        counts_line += f"Event timeline (from analytics, authoritative): {ev}.\n\n"
    prompt = (
        "Visual observations from the video, in chronological order. Each line is one "
        "keyframe: its timestamp, a caption, the objects detected in it (with "
        "per-object counts like '3× person'), and any observed actions.\n"
        "When counting, trust the detected object counts over the caption wording — "
        "captions often say 'a person' even when several were detected.\n\n"
        f"{counts_line}"
        f"{context}\n\n"
        f"Question: {request.question}\n\n"
        "Answer in plain, natural language for a non-technical viewer, using only "
        "these observations. Mention when things happen in a natural way (e.g. "
        "'around 12 seconds in'), and never quote the raw observation format above. "
        "If the question asks to find or list occurrences, mention every matching "
        "moment, not just the first."
    )

    def generate() -> Iterator[str]:
        # Ship the supporting frames first so the UI can render thumbnails:
        # the semantic top-K plus any frames matching a discriminative query
        # term (for enumeration), chronological and capped. Thumbnails are loaded
        # from disk here — only for the handful of frames we actually render.
        for f in display_frames:
            f.thumbnail_b64 = p.store.load_thumbnail(f.video_id, f.frame_idx)
        yield "\x00FRAMES:" + json.dumps(
            [f.model_dump(exclude={"thumbnail_b64"})
             | {"thumbnail_b64": _boxed_thumbnail(f, request.question, regions)}
             for f in display_frames]
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
        "thumbnail_b64": p.store.load_thumbnail(best.video_id, best.frame_idx),
        "score": best.score,
    }


@app.get("/frames/{video_id}")
async def frames(video_id: str) -> list[RetrievedFrame]:
    result = pipeline().store.get_by_video_id(video_id, with_thumbnails=True)
    if not result:
        raise HTTPException(404, f"No indexed frames for video {video_id}")
    return result


@app.get("/videos")
async def videos() -> list[VideoSummaryInfo]:
    return pipeline().store.list_videos()


@app.get("/track_video/{video_id}")
async def track_video(video_id: str) -> FileResponse:
    path = get_settings().data_dir / "tracks_video" / f"{video_id}.mp4"
    if not path.is_file():
        raise HTTPException(404, f"No tracking video for {video_id}")
    return FileResponse(path, media_type="video/mp4")


@app.delete("/videos/{video_id}")
async def delete_video(video_id: str) -> dict:
    p = pipeline()
    deleted = p.store.delete_video(video_id)
    if deleted == 0 and not any(
        v.video_id == video_id for v in p.store.list_videos()
    ):
        # Nothing in Chroma and nothing in the registry -> genuinely unknown.
        raise HTTPException(404, f"Unknown video {video_id}")

    _delete_video_files(video_id)
    return {"video_id": video_id, "deleted_frames": deleted}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
