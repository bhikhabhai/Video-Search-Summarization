"""Unit tests for embedding normalisation.

These cover the pure :func:`l2_normalise` contract (unit sphere, float32,
zero-vector safety) without loading CLIP. A slow opt-in integration test at
the bottom exercises the real model when it is installed and cached.
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.inference.embedder import EMBED_DIM, l2_normalise


class TestL2Normalise:
    def test_unit_norm(self) -> None:
        vec = np.arange(1, 513, dtype=np.float64)
        out = l2_normalise(vec)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)

    def test_returns_float32(self) -> None:
        for dtype in (np.float16, np.float32, np.float64, np.int64):
            out = l2_normalise(np.ones(8, dtype=dtype))
            assert out.dtype == np.float32

    def test_preserves_direction(self) -> None:
        vec = np.array([3.0, 4.0], dtype=np.float32)
        out = l2_normalise(vec)
        np.testing.assert_allclose(out, [0.6, 0.8], atol=1e-6)

    def test_zero_vector_is_safe(self) -> None:
        out = l2_normalise(np.zeros(16))
        assert out.dtype == np.float32
        assert not np.any(np.isnan(out))
        np.testing.assert_array_equal(out, np.zeros(16, dtype=np.float32))

    def test_idempotent(self) -> None:
        vec = np.random.default_rng(1).normal(size=EMBED_DIM)
        once = l2_normalise(vec)
        twice = l2_normalise(once)
        np.testing.assert_allclose(once, twice, atol=1e-6)

    def test_scale_invariant(self) -> None:
        vec = np.random.default_rng(2).normal(size=64)
        np.testing.assert_allclose(
            l2_normalise(vec), l2_normalise(vec * 1000.0), atol=1e-5
        )


@pytest.mark.slow
class TestEmbedderIntegration:
    """Loads real CLIP weights — run with ``pytest -m slow`` when available."""

    def test_image_and_text_embeddings_are_unit_norm_512(self) -> None:
        pytest.importorskip("open_clip")
        pytest.importorskip("torch")
        from PIL import Image

        from perception.inference.embedder import Embedder

        embedder = Embedder()
        image = Image.new("RGB", (224, 224), (200, 30, 30))

        img_vec = embedder.embed_image(image)
        txt_vec = embedder.embed_text("a solid red square")

        for vec in (img_vec, txt_vec):
            assert vec.shape == (EMBED_DIM,)
            assert vec.dtype == np.float32
            assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-3)

        # A red square should be closer to "red" than to "a photo of the ocean".
        other = embedder.embed_text("a photo of the blue ocean")
        assert float(img_vec @ txt_vec) > float(img_vec @ other)
