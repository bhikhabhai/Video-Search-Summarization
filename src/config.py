"""Central configuration for Mini-VSS.

All settings are read from environment variables (or an ``.env`` file) via
pydantic-settings. Import :func:`get_settings` everywhere instead of
instantiating :class:`Settings` directly so the parsed settings are cached.

Device / VRAM detection helpers live here too so every module makes the same
CPU-vs-GPU decision.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Runtime settings, overridable via environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── General ────────────────────────────────────────────────
    log_level: str = "INFO"
    data_dir: Path = Path("data")

    # ── Ingestion limits ───────────────────────────────────────
    max_video_seconds: float = 500.0  # 8 minutes 20 seconds
    max_video_mb: float = 300.0

    # ── Scene-change detection ─────────────────────────────────
    diff_threshold: float = 0.08       # mean abs pixel diff, normalised to [0, 1]
    hist_threshold: float = 0.15       # chi-squared distance between colour histograms
    min_scene_gap_sec: float = 1.0     # suppress burst-sampling on fast cuts
    max_gap_sec: float = 3.0           # force a keyframe if this many seconds pass without one
    analysis_fps: float = 6.0          # frames per second analysed for cuts
    analysis_width: int = 640          # decode width for keyframes (aspect preserved)
    max_keyframes: int = 120           # hard cap per video

    # ── Detection (YOLOv8n) ────────────────────────────────────
    detector_model: str = "yolov8n.pt"
    detector_confidence: float = 0.35
    detector_batch_size: int = 8
    # Comma-separated COCO class names; empty string = all 80 classes.
    detector_classes: str = ""

    # ── Captioning (VLM) ───────────────────────────────────────
    # Backend: "qwen" (Qwen2.5-VL, richest — captions + actions + OCR in one
    # pass), "blip" (BLIP-2 on GPU / BLIP-base on CPU), or "auto" (Qwen when a
    # capable GPU is present, else BLIP-base).
    caption_backend: str = "qwen"
    qwen_vl_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    # Concurrent requests for the network (Ollama) caption/OCR path only. Local
    # CPU/GPU models ignore this — one inference already uses all cores, so
    # parallel threads there just contend. Raise if your Ollama server (or a
    # pool of them) can serve requests in parallel.
    ollama_concurrency: int = 4
    # Structured JSON needs headroom or it truncates mid-object; 256 is a safe
    # floor. Raise for very detailed frames; lower to speed up CPU inference.
    caption_max_new_tokens: int = 256
    # Bound Qwen's vision-token budget to keep CPU inference tractable.
    qwen_min_pixels: int = 256 * 28 * 28
    qwen_max_pixels: int = 768 * 28 * 28
    blip2_model: str = "Salesforce/blip2-opt-2.7b"
    blip_base_model: str = "Salesforce/blip-image-captioning-base"
    vram_threshold_gb: float = 8.0     # below this, fall back to BLIP base
    caption_dedup_threshold: float = 0.92

    # ── Pose / action detection (fall / fight / running) ───────
    # Rule-based events from YOLO-pose keypoints, independent of the VLM.
    pose_enabled: bool = False
    pose_model: str = "yolov8n-pose.pt"
    pose_confidence: float = 0.3        # person detection confidence
    pose_kp_confidence: float = 0.3     # per-keypoint confidence floor
    crowd_person_threshold: int = 5     # >= this many people => "crowding"

    # ── OCR (dedicated text / number-plate reading) ────────────
    # Backend: "none" (rely on the VLM caption's OCR), "paddle" (local
    # PaddleOCR), or "ollama" (send the frame to an Ollama vision model with an
    # OCR-only prompt). Runs independently of the captioner.
    ocr_backend: str = "none"
    ocr_languages: str = "en"          # PaddleOCR language code(s)
    ocr_min_confidence: float = 0.5    # PaddleOCR: drop lines below this score
    # Ollama OCR endpoint/model; empty URL falls back to the caption/main server.
    ocr_ollama_base_url: str = ""
    ocr_ollama_model: str = "qwen2.5vl"

    # ── Embeddings (CLIP) ──────────────────────────────────────
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"

    # ── Vector store ───────────────────────────────────────────
    chroma_dir: Path = Path("data/chroma")
    retrieval_top_k: int = 5

    # ── LLM ────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    # Separate Ollama server for caption generation (CAPTION_BACKEND=ollama).
    # Defaults to the same server as ollama_base_url if not set.
    ollama_caption_base_url: str = ""
    # Vision model on the caption Ollama server — must support image input.
    ollama_caption_model: str = "qwen2.5vl"
    llm_max_tokens: int = 1024
    summary_window_sec: float = 30.0

    # ── Servers ────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_url: str = "http://localhost:8000"  # used by the Gradio UI
    ui_port: int = 7860

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.thumbs_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return settings


def configure_logging() -> None:
    """Configure root logging from the LOG_LEVEL env var (idempotent)."""
    level = os.environ.get("LOG_LEVEL", get_settings().log_level).upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


@functools.lru_cache(maxsize=1)
def get_device() -> str:
    """Return ``"cuda"`` when a usable GPU is present, else ``"cpu"``.

    Torch is imported lazily so that merely importing config does not pull in
    the full ML stack (useful for lightweight tooling and tests).
    """
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("CUDA available: using GPU (%s)", name)
            return "cuda"
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("GPU probe failed (%s); falling back to CPU", exc)
    logger.info("No GPU detected: running on CPU")
    return "cpu"


@functools.lru_cache(maxsize=1)
def get_vram_gb() -> float:
    """Total VRAM of GPU 0 in GiB, or 0.0 when running on CPU."""
    if get_device() != "cuda":
        return 0.0
    import torch

    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024**3)
