"""
Tests for APHELION Phase 23 — HEPHAESTUS Progress Streaming.

Covers ForgeProgressStream thread-safe emit/get_latest/reset.
"""

from __future__ import annotations

import threading
import time

import pytest

from aphelion.hephaestus.progress import ForgeProgressStream, ForgeUpdate


class TestForgeUpdate:
    def test_creation(self):
        u = ForgeUpdate(stage="parse", message="Parsing...", percent=0.25)
        assert u.stage == "parse"
        assert u.percent == 0.25
        assert u.timestamp > 0

    def test_details(self):
        u = ForgeUpdate(stage="done", message="Ok", percent=1.0, details={"key": "val"})
        assert u.details["key"] == "val"


class TestForgeProgressStream:
    def test_initial_state(self):
        stream = ForgeProgressStream()
        assert stream.is_complete is False
        assert stream.was_successful is False
        assert stream.get_latest() == []

    def test_emit_and_get(self):
        stream = ForgeProgressStream()
        stream.emit("detect", "Detecting...", 0.10)
        stream.emit("parse", "Parsing...", 0.25)
        updates = stream.get_latest()
        assert len(updates) == 2
        assert updates[0].stage == "detect"
        assert updates[1].stage == "parse"
        # Second call returns empty (consumed)
        assert stream.get_latest() == []

    def test_mark_complete(self):
        stream = ForgeProgressStream()
        stream.emit("work", "Working", 0.5)
        stream.mark_complete(True, "Deployed!")
        assert stream.is_complete is True
        assert stream.was_successful is True
        updates = stream.get_latest()
        assert any(u.stage == "complete" for u in updates)

    def test_mark_complete_failure(self):
        stream = ForgeProgressStream()
        stream.mark_complete(False, "Rejected")
        assert stream.is_complete is True
        assert stream.was_successful is False

    def test_reset(self):
        stream = ForgeProgressStream()
        stream.emit("work", "msg", 0.5)
        stream.mark_complete(True)
        stream.reset()
        assert stream.is_complete is False
        assert stream.was_successful is False
        assert stream.get_latest() == []

    def test_thread_safety(self):
        """Test concurrent emit/get_latest from different threads."""
        stream = ForgeProgressStream()
        errors = []
        collected = []

        def producer():
            try:
                for i in range(100):
                    stream.emit(f"stage_{i}", f"msg_{i}", i / 100)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def consumer():
            try:
                for _ in range(200):
                    updates = stream.get_latest()
                    collected.extend(updates)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert errors == [], f"Thread errors: {errors}"
        # Should have collected at least some updates
        # The exact count varies due to timing
        remaining = stream.get_latest()
        total = len(collected) + len(remaining)
        assert total == 100
