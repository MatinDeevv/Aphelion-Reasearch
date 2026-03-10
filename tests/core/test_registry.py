"""Tests for APHELION Component Registry."""

import pytest
from aphelion.core.registry import Registry
from aphelion.core.config import ComponentStatus, Tier


class TestRegistry:
    def setup_method(self):
        self.registry = Registry()

    def test_register_valid_module(self):
        self.registry.register("SENTINEL")
        state = self.registry.get_status("SENTINEL")
        assert state.status == ComponentStatus.INITIALIZING

    def test_register_invalid_module(self):
        with pytest.raises(ValueError):
            self.registry.register("NONEXISTENT")

    def test_set_status(self):
        self.registry.register("HYDRA")
        self.registry.set_status("HYDRA", ComponentStatus.ACTIVE)
        assert self.registry.get_status("HYDRA").status == ComponentStatus.ACTIVE

    def test_heartbeat(self):
        self.registry.register("DATA")
        self.registry.heartbeat("DATA")
        state = self.registry.get_status("DATA")
        assert state.last_heartbeat > 0

    def test_report_error(self):
        self.registry.register("ARES")
        self.registry.report_error("ARES", "Connection failed")
        state = self.registry.get_status("ARES")
        assert state.error_count == 1
        assert state.last_error == "Connection failed"

    def test_error_threshold(self):
        self.registry.register("ARES")
        for i in range(10):
            self.registry.report_error("ARES", f"Error {i}")
        assert self.registry.get_status("ARES").status == ComponentStatus.ERROR

    def test_health_score_bounds(self):
        self.registry.register("HYDRA")
        self.registry.set_health("HYDRA", 150.0)
        assert self.registry.get_status("HYDRA").health_score == 100.0

        self.registry.set_health("HYDRA", -50.0)
        assert self.registry.get_status("HYDRA").health_score == 0.0

    def test_get_active_components(self):
        self.registry.register("HYDRA")
        self.registry.register("SENTINEL")
        self.registry.set_status("HYDRA", ComponentStatus.ACTIVE)
        active = self.registry.get_active_components()
        assert "HYDRA" in active
        assert "SENTINEL" not in active

    def test_get_by_tier(self):
        self.registry.register("OLYMPUS")
        self.registry.register("SENTINEL")
        self.registry.register("HYDRA")
        general = self.registry.get_components_by_tier(Tier.GENERAL)
        assert "OLYMPUS" in general
        assert "SENTINEL" in general
        assert "HYDRA" not in general

    def test_system_health_empty(self):
        health = self.registry.system_health()
        assert health["overall"] == 0.0

    def test_unregistered_access_fails(self):
        with pytest.raises(KeyError):
            self.registry.get_status("HYDRA")
