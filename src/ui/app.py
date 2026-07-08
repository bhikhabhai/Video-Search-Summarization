"""Mini-VSS Gradio UI — single file, no build step.

Three tabs backed entirely by the FastAPI service (``API_URL`` env var):

* **Upload**  — file upload, live progress bar (polls /status every 1 s),
  keyframe grid on completion.
* **Ask**     — video selector, chat history, streamed answers, retrieved
  frame thumbnails alongside each answer.
* **Summary** — one-click streamed summary + keyframe timeline strip.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from collections.abc import Iterator

import gradio as gr
import httpx
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")


# ── API helpers ────────────────────────────────────────────────


def _b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _list_videos() -> list[tuple[str, str]]:
    """Return (label, video_id) choices for the dropdowns."""
    try:
        response = httpx.get(f"{API_URL}/videos", timeout=10)
        response.raise_for_status()
    except httpx.HTTPError:
        return []
    return [
        (f"{v['filename']} ({v['duration_sec']:.0f}s, {v['keyframe_count']} frames)", v["video_id"])
        for v in response.json()
    ]


def _fetch_frames(video_id: str) -> list[tuple[Image.Image, str]]:
    """Keyframe gallery items: (image, timestamp caption)."""
    response = httpx.get(f"{API_URL}/frames/{video_id}", timeout=30)
    response.raise_for_status()
    gallery = []
    for frame in response.json():
        if frame.get("thumbnail_b64"):
            gallery.append(
                (_b64_to_image(frame["thumbnail_b64"]), f"t={frame['timestamp_sec']:.1f}s")
            )
    return gallery


def _iter_sse(response: httpx.Response) -> Iterator[str]:
    """Yield decoded data payloads from a text/event-stream response."""
    for line in response.iter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            break
        yield json.loads(payload)


# ── Tab 1: Upload ──────────────────────────────────────────────


def upload_video(file_path: str | None):
    """Upload, then poll /status every second, streaming UI updates."""
    if not file_path:
        yield "No file selected.", gr.update(), []
        return

    with open(file_path, "rb") as fh:
        try:
            response = httpx.post(
                f"{API_URL}/ingest",
                files={"file": (os.path.basename(file_path), fh, "video/mp4")},
                timeout=120,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            yield f"Upload rejected: {exc.response.text}", gr.update(), []
            return
        except httpx.HTTPError as exc:
            yield f"Upload failed: {exc}", gr.update(), []
            return

    job = response.json()
    job_id, video_id = job["job_id"], job["video_id"]
    yield f"Queued (job {job_id})…", gr.update(), []

    while True:
        time.sleep(1)
        try:
            status = httpx.get(f"{API_URL}/status/{job_id}", timeout=10).json()
        except httpx.HTTPError:
            continue
        state, progress = status["state"], status["progress"]
        message = f"[{progress:3d}%] {status['message']}"

        if state == "error":
            yield f"❌ {status.get('error', 'unknown error')}", gr.update(), []
            return
        if state == "done":
            gallery = _fetch_frames(video_id)
            choices = _list_videos()
            yield (
                f"✅ {message} — video id: {video_id}",
                gr.update(choices=choices, value=video_id),
                gallery,
            )
            return
        yield message, gr.update(), []


# ── Tab 2: Ask ─────────────────────────────────────────────────


def ask_question(video_id: str | None, question: str, history: list[dict]):
    """Stream an answer into the chatbot; show retrieved frames alongside."""
    if not video_id:
        history = history + [
            {"role": "assistant", "content": "Pick a processed video first."}
        ]
        yield history, [], ""
        return
    if not question.strip():
        yield history, [], ""
        return

    history = history + [{"role": "user", "content": question}]
    history.append({"role": "assistant", "content": ""})
    thumbs: list[tuple[Image.Image, str]] = []
    yield history, thumbs, ""

    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/query",
            json={"video_id": video_id, "question": question},
            timeout=300,
        ) as response:
            response.raise_for_status()
            for chunk in _iter_sse(response):
                if chunk.startswith("\x00FRAMES:"):
                    frames = json.loads(chunk[len("\x00FRAMES:"):])
                    thumbs = [
                        (_b64_to_image(f["thumbnail_b64"]),
                         f"t={f['timestamp_sec']:.1f}s  {f['caption'][:60]}")
                        for f in frames
                        if f.get("thumbnail_b64")
                    ]
                    yield history, thumbs, ""
                    continue
                history[-1]["content"] += chunk
                yield history, thumbs, ""
    except httpx.HTTPError as exc:
        history[-1]["content"] += f"\n\n[error contacting API: {exc}]"
        yield history, thumbs, ""


# ── Tab 3: Summary ─────────────────────────────────────────────


def summarise_video(video_id: str | None):
    """Stream the summary text; also load the keyframe timeline strip."""
    if not video_id:
        yield "Pick a processed video first.", []
        return

    try:
        timeline = _fetch_frames(video_id)
    except httpx.HTTPError:
        timeline = []

    text = ""
    yield "⏳ summarising…", timeline
    try:
        with httpx.stream("GET", f"{API_URL}/summary/{video_id}", timeout=600) as response:
            response.raise_for_status()
            for chunk in _iter_sse(response):
                text += chunk
                yield text, timeline
    except httpx.HTTPError as exc:
        yield text + f"\n\n[error contacting API: {exc}]", timeline


# ── Layout ─────────────────────────────────────────────────────


_CSS = """
/* Gradio 4.44 Gallery scroll container is .grid-wrap — already has
   overflow-y:scroll but the bar is invisible on most OS/browsers.
   Force a visible thin scrollbar. */
.grid-wrap {
    scrollbar-width: thin;
    scrollbar-color: rgba(128,128,128,0.55) transparent;
}
.grid-wrap::-webkit-scrollbar { width: 7px; }
.grid-wrap::-webkit-scrollbar-track { background: transparent; }
.grid-wrap::-webkit-scrollbar-thumb {
    background: rgba(128,128,128,0.55);
    border-radius: 4px;
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Mini-VSS", theme=gr.themes.Soft(), css=_CSS) as demo:
        gr.Markdown("# 🎬 Mini-VSS — Video Search & Summarization")

        with gr.Tab("📤 Upload"):
            video_file = gr.File(label="Video (≤8 min 20 sec / 300 MB)", file_types=["video"], type="filepath")
            upload_button = gr.Button("Ingest video", variant="primary")
            upload_status = gr.Markdown("")
            keyframe_grid = gr.Gallery(label="Extracted keyframes", columns=6, height=400, object_fit="contain")

        with gr.Tab("💬 Ask"):
            with gr.Row():
                ask_video = gr.Dropdown(label="Video", choices=_list_videos(), interactive=True)
                refresh_ask = gr.Button("🔄", scale=0)
            chatbot = gr.Chatbot(label="Q&A", height=380, type="messages")
            retrieved = gr.Gallery(label="Retrieved frames", columns=3, height=280, object_fit="contain")
            question_box = gr.Textbox(
                label="Question",
                placeholder="e.g. when does a person appear in a red jacket?",
            )

        with gr.Tab("📝 Summary"):
            with gr.Row():
                summary_video = gr.Dropdown(label="Video", choices=_list_videos(), interactive=True)
                refresh_summary = gr.Button("🔄", scale=0)
            summarise_button = gr.Button("Generate summary", variant="primary")
            summary_output = gr.Markdown("")
            timeline_strip = gr.Gallery(label="Timeline", columns=10, height=180, object_fit="contain")

        # Wiring
        upload_button.click(
            upload_video,
            inputs=[video_file],
            outputs=[upload_status, ask_video, keyframe_grid],
            api_name=False,
        )
        question_box.submit(
            ask_question,
            inputs=[ask_video, question_box, chatbot],
            outputs=[chatbot, retrieved, question_box],
            api_name=False,
        )
        summarise_button.click(
            summarise_video,
            inputs=[summary_video],
            outputs=[summary_output, timeline_strip],
            api_name=False,
        )
        refresh_ask.click(
            lambda: gr.update(choices=_list_videos()), outputs=[ask_video], api_name=False,
        )
        refresh_summary.click(
            lambda: gr.update(choices=_list_videos()), outputs=[summary_video], api_name=False,
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("UI_PORT", "7860"))
    build_app().launch(server_name="0.0.0.0", server_port=port, show_api=False)
