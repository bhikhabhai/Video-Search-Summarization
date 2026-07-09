"""Unit tests for batched Qwen describe logic (model stubbed — no weights)."""

from __future__ import annotations

import numpy as np

from perception.inference.captioner import Captioner


class _DummyEmbedder:
    def embed_text(self, text: str) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)


def _make_captioner() -> Captioner:
    # Passing an embedder avoids constructing the real CLIP model.
    c = Captioner(embedder=_DummyEmbedder())
    c._backend = "qwen"
    return c


class TestBatchedDescribe:
    def test_chunks_and_parses_each_frame(self):
        c = _make_captioner()
        chunk_sizes: list[int] = []

        def fake_batch(images, prompts):
            chunk_sizes.append(len(images))
            return [f'{{"scene": "s{i}"}}' for i in range(len(images))]

        c._qwen_generate_batch = fake_batch  # type: ignore[assignment]

        images = [object()] * 5
        hints = [""] * 5
        out = c._describe_qwen_batched(images, hints, batch_size=2)

        assert len(out) == 5
        assert chunk_sizes == [2, 2, 1]        # 5 frames in batches of 2
        assert out[0]["scene"] == "s0"
        assert out[4]["scene"] == "s0"         # last (size-1) batch re-indexes from 0

    def test_failed_batch_falls_back_per_frame(self):
        c = _make_captioner()

        def boom(images, prompts):
            raise RuntimeError("simulated OOM")

        c._qwen_generate_batch = boom  # type: ignore[assignment]
        # Stub the per-frame path the fallback uses.
        c._describe_one = lambda img, hint: {  # type: ignore[assignment]
            "caption": "fallback", "scene": "", "actions": [], "ocr_text": "", "events": [],
        }

        out = c._describe_qwen_batched([object(), object()], ["", ""], batch_size=2)
        assert len(out) == 2
        assert all(o["caption"] == "fallback" for o in out)

    def test_hint_included_in_prompt(self):
        c = _make_captioner()
        seen_prompts: list[str] = []

        def capture(images, prompts):
            seen_prompts.extend(prompts)
            return ["{}"] * len(images)

        c._qwen_generate_batch = capture  # type: ignore[assignment]
        c._describe_qwen_batched([object()], ["2 persons, 1 car"], batch_size=4)
        assert "2 persons, 1 car" in seen_prompts[0]
