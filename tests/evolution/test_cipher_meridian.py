"""Tests for CIPHER config manager and MERIDIAN state bus."""

import os
import tempfile
import pytest

from aphelion.evolution.cipher.config_manager import CipherConfigManager, ConfigChange
from aphelion.evolution.meridian.state_bus import StateBus, StateSnapshot, StateRecovery


# ── CipherConfigManager ──────────────────────────────────────────


class TestCipherConfigManager:
    def test_set_get(self):
        cm = CipherConfigManager()
        cm.set("key1", "value1", reason="test")
        assert cm.get("key1") == "value1"

    def test_default_value(self):
        cm = CipherConfigManager()
        assert cm.get("missing", "default") == "default"

    def test_audit_log(self):
        cm = CipherConfigManager()
        cm.set("k1", "v1", reason="initial")
        cm.set("k1", "v2", reason="updated")
        log = cm.audit_log
        assert len(log) == 2
        assert log[0].key == "k1"
        assert log[1].new_value == "v2"

    def test_secret_store_retrieve(self):
        cm = CipherConfigManager()
        cm.set_secret("api_key", "my_secret_123")
        assert cm.get_secret("api_key") == "my_secret_123"

    def test_secret_missing(self):
        cm = CipherConfigManager()
        assert cm.get_secret("nonexistent") is None

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            cm = CipherConfigManager(config_path=path)
            cm.set("x", 42)
            cm.save()
            # Load into new manager
            cm2 = CipherConfigManager(config_path=path)
            cm2.load()
            assert cm2.get("x") == 42

    def test_to_dict(self):
        cm = CipherConfigManager()
        cm.set("a", 1)
        cm.set("b", "hello")
        d = cm.to_dict()
        assert d == {"a": 1, "b": "hello"}


# ── StateBus ──────────────────────────────────────────────────────


class TestStateBus:
    def test_publish_get(self):
        bus = StateBus()
        bus.publish("regime", "TRENDING_BULL")
        assert bus.get("regime") == "TRENDING_BULL"

    def test_subscribe_callback(self):
        bus = StateBus()
        received = []
        bus.subscribe("regime", lambda k, v: received.append((k, v)))
        bus.publish("regime", "VOLATILE")
        assert len(received) == 1
        assert received[0] == ("regime", "VOLATILE")

    def test_get_default(self):
        bus = StateBus()
        assert bus.get("missing", "default") == "default"

    def test_snapshot_create(self):
        bus = StateBus()
        bus.publish("a", 1)
        bus.publish("b", 2)
        snap = bus.create_snapshot("test_snap")
        assert snap.label == "test_snap"
        assert snap.state == {"a": 1, "b": 2}

    def test_snapshot_restore(self):
        bus = StateBus()
        bus.publish("x", 10)
        snap = bus.create_snapshot()
        bus.publish("x", 99)
        assert bus.get("x") == 99
        bus.restore_snapshot(snap)
        assert bus.get("x") == 10

    def test_snapshot_count(self):
        bus = StateBus()
        bus.create_snapshot()
        bus.create_snapshot()
        assert bus.snapshot_count == 2

    def test_clear(self):
        bus = StateBus()
        bus.publish("a", 1)
        bus.clear()
        assert bus.get("a") is None


# ── StateRecovery ─────────────────────────────────────────────────


class TestStateRecovery:
    def test_persist_and_recover(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = StateBus()
            bus.publish("mode", "ACTIVE")
            bus.publish("equity", 10000)

            recovery = StateRecovery(bus, recovery_dir=tmpdir)
            recovery.persist_state()

            # Create new bus and recover
            bus2 = StateBus()
            recovery2 = StateRecovery(bus2, recovery_dir=tmpdir)
            result = recovery2.recover_latest()
            assert result is True
            assert bus2.get("mode") == "ACTIVE"
            assert bus2.get("equity") == 10000

    def test_recover_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = StateBus()
            recovery = StateRecovery(bus, recovery_dir=tmpdir)
            result = recovery.recover_latest()
            assert result is False
