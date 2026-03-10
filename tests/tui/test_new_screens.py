"""Tests for TUI v2 screens — build_* panel functions."""

import pytest

# We need TUIState available
from aphelion.tui.state import TUIState


class TestSystemHealthPanel:
    def test_builds_without_error(self):
        from aphelion.tui.screens.system_health import build_system_health_panel
        state = TUIState()
        state.feed_connected = True
        state.cpu_usage = 45.0
        state.memory_mb = 256
        state.latency_ms = 30
        panel = build_system_health_panel(state)
        assert panel is not None
        assert panel.title is not None

    def test_disconnected_feed(self):
        from aphelion.tui.screens.system_health import build_system_health_panel
        state = TUIState()
        state.feed_connected = False
        panel = build_system_health_panel(state)
        assert panel is not None


class TestSOLAPanel:
    def test_builds_without_error(self):
        from aphelion.tui.screens.sola_panel import build_sola_panel
        state = TUIState()
        panel = build_sola_panel(state)
        assert panel is not None

    def test_with_module_rankings(self):
        from aphelion.tui.screens.sola_panel import build_sola_panel
        state = TUIState()
        state.sola_mode = "DEFENSIVE"
        state.sola_edge_confidence = 0.6
        state.sola_module_rankings = [("HYDRA", 0.8), ("ECHO", 0.3), ("FORGE", -0.1)]
        panel = build_sola_panel(state)
        assert panel is not None


class TestOmegaPanel:
    def test_builds_without_error(self):
        from aphelion.tui.screens.omega_panel import build_omega_panel
        state = TUIState()
        panel = build_omega_panel(state)
        assert panel is not None

    def test_with_trend_data(self):
        from aphelion.tui.screens.omega_panel import build_omega_panel
        state = TUIState()
        state.omega_trend = "BULL"
        state.omega_trend_strength = 0.7
        state.omega_win_rate = 0.55
        panel = build_omega_panel(state)
        assert panel is not None


class TestEvolutionPanel:
    def test_builds_without_error(self):
        from aphelion.tui.screens.evolution_panel import build_evolution_panel
        state = TUIState()
        panel = build_evolution_panel(state)
        assert panel is not None

    def test_with_evolution_data(self):
        from aphelion.tui.screens.evolution_panel import build_evolution_panel
        state = TUIState()
        state.evo_generation = 42
        state.evo_best_fitness = 0.85
        state.cipher_top_features = [("atr", 0.15), ("rsi", 0.12)]
        panel = build_evolution_panel(state)
        assert panel is not None


class TestReplayPanel:
    def test_builds_without_error(self):
        from aphelion.tui.screens.replay_panel import build_replay_panel
        state = TUIState()
        panel = build_replay_panel(state)
        assert panel is not None

    def test_replay_inactive(self):
        from aphelion.tui.screens.replay_panel import build_replay_panel
        state = TUIState()
        state.replay_mode = False
        panel = build_replay_panel(state)
        assert panel is not None
