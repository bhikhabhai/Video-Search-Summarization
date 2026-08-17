"""Chunked, recursive video summarisation with a streaming LLM backend.

Strategy:
1. Group frame records into ``SUMMARY_WINDOW_SEC`` (default 30 s) windows.
2. Summarise each window with the LLM, using captions + detection labels as
   grounding context.
3. Recursive reduce: if more than 10 window summaries exist, summarise the
   summaries (repeat until <= 10 remain).
4. Stream the final synthesis token by token via a generator.

LLM backend selection (:class:`LLMClient`):
* ``ANTHROPIC_API_KEY`` set  → Anthropic SDK (claude-haiku-4-5-20251001).
* otherwise                  → local Ollama HTTP endpoint (offline-capable).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import httpx

from perception.config import get_settings
from perception.models import RetrievedFrame

logger = logging.getLogger(__name__)

MAX_WINDOWS_BEFORE_REDUCE = 10


class LLMClient:
    """Streaming text generation via Anthropic, with an Ollama fallback."""

    def __init__(self) -> None:
        settings = get_settings()
        self.max_tokens = settings.llm_max_tokens
        self.anthropic_key = settings.anthropic_api_key
        self.anthropic_model = settings.anthropic_model
        self.ollama_base_url = settings.ollama_base_url.rstrip("/")
        self.ollama_model = settings.ollama_model
        self._anthropic_client = None
        if self.anthropic_key:
            logger.info("LLM backend: Anthropic (%s)", self.anthropic_model)
        else:
            logger.info("LLM backend: Ollama (%s at %s)", self.ollama_model, self.ollama_base_url)

    @property
    def backend(self) -> str:
        return "anthropic" if self.anthropic_key else "ollama"

    def _anthropic(self):
        if self._anthropic_client is None:
            import anthropic

            self._anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
        return self._anthropic_client

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        """Yield generated text chunk by chunk."""
        if self.anthropic_key:
            yield from self._stream_anthropic(prompt, system)
        else:
            yield from self._stream_ollama(prompt, system)

    def complete(self, prompt: str, system: str = "") -> str:
        """Blocking helper: full generated text as one string."""
        return "".join(self.stream(prompt, system))

    def _stream_anthropic(self, prompt: str, system: str) -> Iterator[str]:
        client = self._anthropic()
        kwargs: dict = {
            "model": self.anthropic_model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    def _stream_ollama(self, prompt: str, system: str) -> Iterator[str]:
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "options": {"num_predict": self.max_tokens},
        }
        try:
            with httpx.stream(
                "POST", f"{self.ollama_base_url}/api/generate", json=payload, timeout=300.0
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if chunk := data.get("response"):
                        yield chunk
                    if data.get("done"):
                        break
        except httpx.HTTPError as exc:
            logger.error("Ollama request failed: %s", exc)
            yield (
                "[LLM unavailable] Set ANTHROPIC_API_KEY or run an Ollama server at "
                f"{self.ollama_base_url}. Error: {exc}"
            )


def _format_window(frames: list[RetrievedFrame]) -> str:
    lines = []
    for frame in frames:
        if frame.detection_counts:
            obj_str = ", ".join(
                f"{count}× {label}" for label, count in sorted(frame.detection_counts.items())
            )
            labels = f" [objects: {obj_str}]"
        elif frame.detection_labels:
            labels = f" [objects: {', '.join(frame.detection_labels)}]"
        else:
            labels = ""
        actions = f" [actions: {'; '.join(frame.actions)}]" if frame.actions else ""
        events = f" [events: {', '.join(frame.events)}]" if frame.events else ""
        ocr = f" [text: {frame.ocr_text}]" if frame.ocr_text else ""
        attrs = f" [items: {', '.join(frame.attributes)}]" if frame.attributes else ""
        lines.append(
            f"- t={frame.timestamp_sec:.1f}s: {frame.caption}{labels}{actions}{events}{ocr}{attrs}"
        )
    return "\n".join(lines)


class Summariser:
    """Windowed map-reduce summarisation over a video's frame records."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()
        self.window_sec = get_settings().summary_window_sec

    def _windows(self, frames: list[RetrievedFrame]) -> list[list[RetrievedFrame]]:
        windows: dict[int, list[RetrievedFrame]] = {}
        for frame in frames:
            windows.setdefault(int(frame.timestamp_sec // self.window_sec), []).append(frame)
        return [windows[key] for key in sorted(windows)]

    def _summarise_window(self, index: int, frames: list[RetrievedFrame]) -> str:
        start = index * self.window_sec
        prompt = (
            f"These are visual observations from a video segment "
            f"({start:.0f}s to {start + self.window_sec:.0f}s):\n\n"
            f"{_format_window(frames)}\n\n"
            "Write 1-2 sentences describing what happens in this segment. "
            "Be concrete and only state what the observations support."
        )
        return self.llm.complete(prompt).strip()

    def _reduce(self, summaries: list[str]) -> list[str]:
        """Merge summaries pairwise-in-batches until at most 10 remain."""
        while len(summaries) > MAX_WINDOWS_BEFORE_REDUCE:
            batch_size = 4
            merged: list[str] = []
            for start in range(0, len(summaries), batch_size):
                batch = summaries[start : start + batch_size]
                prompt = (
                    "Combine these consecutive video segment summaries into one "
                    "concise summary, preserving chronology:\n\n"
                    + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(batch))
                )
                merged.append(self.llm.complete(prompt).strip())
            summaries = merged
        return summaries

    def summarise(self, frames: list[RetrievedFrame]) -> Iterator[str]:
        """Stream the final video summary token by token.

        Window/reduce steps run eagerly (they are short LLM calls); only the
        final synthesis streams — that is where the user-visible latency is.
        """
        if not frames:
            yield "No keyframes are stored for this video."
            return

        windows = self._windows(frames)
        logger.info("Summarising %d frames in %d windows", len(frames), len(windows))

        window_summaries: list[str] = []
        for i, window in enumerate(windows):
            try:
                window_summaries.append(self._summarise_window(i, window))
            except Exception:
                logger.exception("Window %d summarisation failed; skipping", i)

        if not window_summaries:
            yield "Summarisation failed for every segment; check the LLM backend logs."
            return

        window_summaries = self._reduce(window_summaries)

        final_prompt = (
            "You are describing a video based on chronological segment summaries:\n\n"
            + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(window_summaries))
            + "\n\nWrite a coherent overall summary of the video (3-6 sentences). "
            "Mention the main subjects, actions, and how the scene evolves over time."
        )
        yield from self.llm.stream(
            final_prompt,
            system="You summarise videos from machine-generated visual observations. "
            "Never invent details that are not supported by the observations.",
        )


if __name__ == "__main__":
    import argparse

    from perception.config import configure_logging
    from retrieval.rag.vector_store import VectorStore

    configure_logging()
    parser = argparse.ArgumentParser(description="Stream a summary for a processed video")
    parser.add_argument("video_id")
    args = parser.parse_args()

    store = VectorStore()
    summariser = Summariser()
    for token in summariser.summarise(store.get_by_video_id(args.video_id)):
        print(token, end="", flush=True)
    print()
