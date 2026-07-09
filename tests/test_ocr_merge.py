"""Unit tests for OCR text merging + dedupe (no models required)."""

from __future__ import annotations

from retrieval.api.main import _merge_ocr
from perception.inference.ocr import _dedupe


class TestDedupe:
    def test_case_insensitive_order_preserved(self):
        assert _dedupe(["EXIT", "exit", "Gate 4"]) == ["EXIT", "Gate 4"]

    def test_empty(self):
        assert _dedupe([]) == []


class TestMergeOcr:
    def test_ocr_model_takes_precedence_order(self):
        # Dedicated OCR lines come first, then any new VLM items.
        out = _merge_ocr(["GJ01AB1234"], "EXIT; welcome")
        assert out == "GJ01AB1234; EXIT; welcome"

    def test_dedupes_across_sources(self):
        out = _merge_ocr(["GJ01AB1234", "EXIT"], "exit; GJ01AB1234")
        assert out == "GJ01AB1234; EXIT"

    def test_only_vlm(self):
        assert _merge_ocr([], "Plate ABC123") == "Plate ABC123"

    def test_only_ocr_model(self):
        assert _merge_ocr(["ABC123"], "") == "ABC123"

    def test_both_empty(self):
        assert _merge_ocr([], "") == ""
