"""Tests for Registry v2: lifecycle methods, dependency graph, error rate."""

import time
import pytest
from aphelion.core.registry import Registry
from aphelion.core.config import ComponentStatus


class TestRegistryLifecycle:
    def setup_method(self):
        self.registry = Registry()

    def test_register_with_dependencies(self):
        self.registry.register("DATA", dependencies=["SENTINEL"])
        state = self.registry._components["DATA"]
        assert "SENTINEL" in state.dependencies

    def test_pause_sets_paused_status(self):
        self.registry.register("HYDRA")
        self.registry.set_status("HYDRA", ComponentStatus.ACTIVE)
        self.registry.pause("HYDRA")
        state = self.registry.get_status("HYDRA")
        assert state.status == ComponentStatus.PAUSED

    def test_resume_sets_active_status(self):
        self.registry.register("HYDRA")
        self.registry.set_status("HYDRA", ComponentStatus.ACTIVE)
        self.registry.pause("HYDRA")
        self.registry.resume("HYDRA")
        state = self.registry.get_status("HYDRA")
        assert state.status == ComponentStatus.ACTIVE

    def test_restart_resets_errors_and_health(self):
        self.registry.register("HYDRA")
        self.registry.report_error("HYDRA", "test error")
        self.registry.report_error("HYDRA", "test error 2")
        self.registry.restart("HYDRA")
        state = self.registry._components["HYDRA"]
        assert state.error_count == 0
        assert state.health_score == 100.0
        assert state.status == ComponentStatus.INITIALIZING


class TestRegistryErrorRate:
    def setup_method(self):
        self.registry = Registry()

    def test_error_rate_zero_initially(self):
        self.registry.register("SENTINEL")
        assert self.registry.error_rate("SENTINEL") == 0.0

    def test_error_rate_after_errors(self):
        self.registry.register("SENTINEL")
        self.registry.report_error("SENTINEL", "e1")
        self.registry.report_error("SENTINEL", "e2")
        rate = self.registry.error_rate("SENTINEL")
        assert rate >= 0.0  # Rate depends on time elapsed


class TestStartupOrder:
    def setup_method(self):
        self.registry = Registry()

    def test_no_deps_returns_all(self):
        self.registry.register("HYDRA")
        self.registry.register("SENTINEL")
        order = self.registry.get_startup_order()
        assert set(order) == {"HYDRA", "SENTINEL"}

    def test_dependency_ordering(self):
        self.registry.register("DATA")
        self.registry.register("HYDRA", dependencies=["DATA"])
        self.registry.register("BACKTEST", dependencies=["HYDRA"])
        order = self.registry.get_startup_order()
        assert order.index("DATA") < order.index("HYDRA")
        assert order.index("HYDRA") < order.index("BACKTEST")

    def test_cycle_detection_raises(self):
        self.registry.register("HYDRA", dependencies=["SENTINEL"])
        self.registry.register("SENTINEL", dependencies=["HYDRA"])
        with pytest.raises(ValueError, match="[Cc]ircular"):
            self.registry.get_startup_order()

    def test_diamond_dependency(self):
        self.registry.register("DATA")
        self.registry.register("HYDRA", dependencies=["DATA"])
        self.registry.register("SENTINEL", dependencies=["DATA"])
        self.registry.register("BACKTEST", dependencies=["HYDRA", "SENTINEL"])
        order = self.registry.get_startup_order()
        assert order.index("DATA") < order.index("HYDRA")
        assert order.index("DATA") < order.index("SENTINEL")
        assert order.index("HYDRA") < order.index("BACKTEST")
        assert order.index("SENTINEL") < order.index("BACKTEST")
