"""Shared data models used across all layers of Mini-VSS.

Pure-data payloads that cross layer or process boundaries are Pydantic v2
models. :class:`ExtractedFrame` is a plain dataclass because it carries a live
``PIL.Image`` and only ever travels inside the ingestion process.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PIL import Image
from pydantic import BaseModel, Field


# ── Ingestion ──────────────────────────────────────────────────


class VideoMeta(BaseModel):
    """Probe result for an ingested video."""

    video_id: str
    source: str  # file path or RTSP URL
    duration_sec: float
    fps: float
    width: int
    height: int
    size_bytes: int = 0
    filename: str = ""


@dataclass
class ExtractedFrame:
    """A keyframe selected by the scene-change detector."""

    frame_idx: int
    timestamp_sec: float
    pil_image: Image.Image
    diff_score: float


# ── Inference ──────────────────────────────────────────────────


class Detection(BaseModel):
    """A single object detection on one frame."""

    label: str
    confidence: float
    # Normalised xyxy in [0, 1] so boxes are resolution-independent.
    bbox: tuple[float, float, float, float]


class FrameRecord(BaseModel):
    """The fully-processed representation of a keyframe, ready for storage."""

    video_id: str
    frame_idx: int
    timestamp_sec: float
    caption: str
    detection_labels: list[str] = Field(default_factory=list)
    # Per-class object counts, e.g. {"person": 3, "car": 1}.
    detection_counts: dict[str, int] = Field(default_factory=dict)
    # VQA-derived action/interaction descriptions, e.g. ["shaking hands", "walking"]
    actions: list[str] = Field(default_factory=list)
    thumbnail_b64: str = ""


# ── Jobs / API ─────────────────────────────────────────────────


class JobState(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class IngestStatus(BaseModel):
    """Progress of a background ingestion job."""

    job_id: str
    video_id: str
    state: JobState = JobState.QUEUED
    progress: int = 0  # 0-100
    message: str = ""
    error: str | None = None
    keyframe_count: int = 0


class QueryRequest(BaseModel):
    video_id: str
    question: str


class RetrievedFrame(BaseModel):
    """A frame returned by hybrid retrieval, with its relevance score."""

    video_id: str
    frame_idx: int
    timestamp_sec: float
    caption: str
    detection_labels: list[str] = Field(default_factory=list)
    # Per-class object counts, e.g. {"person": 3, "car": 1}.
    detection_counts: dict[str, int] = Field(default_factory=dict)
    # VQA-derived action/interaction descriptions.
    actions: list[str] = Field(default_factory=list)
    thumbnail_b64: str = ""
    score: float = 0.0


class ClipResult(BaseModel):
    """Best-matching moment for a semantic clip query."""

    video_id: str
    timestamp_sec: float
    caption: str
    thumbnail_b64: str
    score: float


class VideoSummaryInfo(BaseModel):
    """Registry entry for a processed video (populates the UI dropdown)."""

    video_id: str
    filename: str
    duration_sec: float
    keyframe_count: int
