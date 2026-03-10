"""
Tests for APHELION Phase 23 — New TUI Screens.

Covers Launcher (F6), Setup (F7), HEPHAESTUS (F8), Training (F9),
Backtest (F10), and dashboard routing for new views.
"""

from __future__ import annotations

import pytest

rich = pytest.importorskip("rich")

from rich.panel import Panel
from rich.layout import Layout

from aphelion.tui.config import AphelionConfig


# ═══════════════════════════════════════════════════════════════════════════
# Launcher (F6)
# ═══════════════════════════════════════════════════════════════════════════

from aphelion.tui.screens.launcher import build_launcher_panel, BANNER


class TestLauncherPanel:
    def test_builds_without_error(self):
        config = AphelionConfig()
        panel = build_launcher_panel(config)
        assert isinstance(panel, Panel)

    def test_with_last_session(self):
        config = AphelionConfig()
        session = {
            "date": "2025-01-15",
            "capital": 10_200,
            "pnl": 200,
            "trades": 42,
            "wr": 0.62,
        }
        panel = build_launcher_panel(config, last_session=session)
        assert isinstance(panel, Panel)

    def test_banner_exists(self):
        assert "AUTONOMOUS" in BANNER

    def test_with_configured_mt5(self):
        config = AphelionConfig()
        config.mt5.login = 12345
        config.mt5.server = "Demo-Server"
        panel = build_launcher_panel(config)
        assert isinstance(panel, Panel)

    def test_with_hydra_checkpoint(self, tmp_path):
        ckpt = tmp_path / "hydra.pt"
        ckpt.write_text("dummy")
        config = AphelionConfig()
        config.hydra.checkpoint = str(ckpt)
        panel = build_launcher_panel(config)
        assert isinstance(panel, Panel)


# ═══════════════════════════════════════════════════════════════════════════
# Setup (F7)
# ═══════════════════════════════════════════════════════════════════════════

from aphelion.tui.screens.setup import (
    build_setup_form,
    build_setup_panel,
    apply_form_to_config,
)


class TestSetupForm:
    def test_build_form_from_config(self):
        config = AphelionConfig()
        form = build_setup_form(config)
        assert len(form.fields) >= 10  # At least 10 fields across sections
        assert form.active_index == 0

    def test_form_sections(self):
        config = AphelionConfig()
        form = build_setup_form(config)
        assert "Trading" in form.sections
        assert "MT5 Connection" in form.sections
        assert "Risk" in form.sections

    def test_form_preserves_values(self):
        config = AphelionConfig()
        config.trading.symbol = "GBPUSD"
        config.trading.capital = 50_000
        form = build_setup_form(config)
        assert form.get_value("trading.symbol") == "GBPUSD"


class TestSetupPanel:
    def test_builds_without_error(self):
        config = AphelionConfig()
        form = build_setup_form(config)
        panel = build_setup_panel(form, config)
        assert isinstance(panel, Panel)

    def test_with_save_message(self):
        config = AphelionConfig()
        form = build_setup_form(config)
        panel = build_setup_panel(form, config, save_message="✓ Saved successfully")
        assert isinstance(panel, Panel)

    def test_with_error_message(self):
        config = AphelionConfig()
        form = build_setup_form(config)
        panel = build_setup_panel(form, config, save_message="Error: Invalid capital")
        assert isinstance(panel, Panel)


class TestApplyFormToConfig:
    def test_apply_valid_form(self):
        config = AphelionConfig()
        form = build_setup_form(config)
        form.set_value("trading.symbol", "EURUSD")
        errors = apply_form_to_config(form, config)
        assert errors == []
        assert config.trading.symbol == "EURUSD"

    def test_apply_invalid_capital(self):
        config = AphelionConfig()
        form = build_setup_form(config)
        form.set_value("trading.capital", "not_a_number")
        errors = apply_form_to_config(form, config)
        # Should have a validation error from the number field
        assert len(errors) >= 1

    def test_apply_percentage_fields(self):
        config = AphelionConfig()
        form = build_setup_form(config)
        form.set_value("risk.max_daily_dd", "5.0%")
        errors = apply_form_to_config(form, config)
        assert errors == []
        assert config.risk.max_daily_dd == pytest.approx(0.05)


# ═══════════════════════════════════════════════════════════════════════════
# HEPHAESTUS Forge (F8)
# ═══════════════════════════════════════════════════════════════════════════

from aphelion.tui.screens.hephaestus_panel import build_hephaestus_panel
from aphelion.tui.controller import ForgeProgress
from aphelion.tui.widgets.text_area import TextAreaState


class TestHephaestusPanel:
    def test_builds_idle(self):
        ta = TextAreaState()
        panel = build_hephaestus_panel(ta)
        assert isinstance(panel, Panel)

    def test_with_code_in_text_area(self):
        ta = TextAreaState()
        ta.content = "rsi = ta.rsi(close, 14)\nif rsi < 30:\n    buy()"
        panel = build_hephaestus_panel(ta)
        assert isinstance(panel, Panel)

    def test_with_forge_progress_active(self):
        ta = TextAreaState()
        fp = ForgeProgress(stage="backtest", message="Backtesting...", percent=0.7)
        panel = build_hephaestus_panel(ta, forge_progress=fp)
        assert isinstance(panel, Panel)

    def test_with_forge_complete_success(self):
        ta = TextAreaState()
        fp = ForgeProgress(
            stage="complete", message="DEPLOYED: RSI_Cross",
            percent=1.0, complete=True, success=True,
        )
        panel = build_hephaestus_panel(ta, forge_progress=fp)
        assert isinstance(panel, Panel)

    def test_with_forge_complete_failure(self):
        ta = TextAreaState()
        fp = ForgeProgress(
            stage="complete", message="REJECTED: Bad_Strat",
            percent=1.0, complete=True, success=False,
        )
        panel = build_hephaestus_panel(ta, forge_progress=fp)
        assert isinstance(panel, Panel)

    def test_with_deployed_strategies(self):
        ta = TextAreaState()
        deployed = [
            {"name": "RSI_Cross", "sharpe": 1.5, "wr": 0.58,
             "mode": "paper", "trades": 150, "status": "DEPLOYED"},
        ]
        panel = build_hephaestus_panel(ta, deployed=deployed)
        assert isinstance(panel, Panel)

    def test_with_rejections(self):
        ta = TextAreaState()
        rejections = [
            {"name": "Bad_Strat", "reason": "Sharpe < 0, Win rate < 40%"},
        ]
        panel = build_hephaestus_panel(ta, rejections=rejections)
        assert isinstance(panel, Panel)


# ═══════════════════════════════════════════════════════════════════════════
# Training (F9)
# ═══════════════════════════════════════════════════════════════════════════

from aphelion.tui.screens.training_panel import build_training_panel, TRAINING_PRESETS
from aphelion.tui.controller import TrainingProgress, TrainingState


class TestTrainingPanel:
    def test_builds_idle(self):
        panel = build_training_panel()
        assert isinstance(panel, Panel)

    def test_with_active_training(self):
        tp = TrainingProgress(
            state=TrainingState.RUNNING,
            current_epoch=5,
            total_epochs=20,
            train_loss=0.045,
            val_loss=0.052,
            elapsed_seconds=120,
            eta_seconds=360,
        )
        panel = build_training_panel(tp)
        assert isinstance(panel, Panel)

    def test_with_loss_history(self):
        tp = TrainingProgress(
            state=TrainingState.RUNNING,
            current_epoch=10,
            total_epochs=20,
            loss_history=[0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.16, 0.15, 0.14, 0.13],
        )
        panel = build_training_panel(tp)
        assert isinstance(panel, Panel)

    def test_complete_training(self):
        tp = TrainingProgress(
            state=TrainingState.COMPLETE,
            current_epoch=20,
            total_epochs=20,
            message="Training complete",
        )
        panel = build_training_panel(tp)
        assert isinstance(panel, Panel)

    def test_with_presets(self):
        assert len(TRAINING_PRESETS) >= 2
        for preset in TRAINING_PRESETS:
            assert "name" in preset
            assert "epochs" in preset

    def test_selected_preset(self):
        panel = build_training_panel(selected_preset=0)
        assert isinstance(panel, Panel)

    def test_with_gpu_info(self):
        panel = build_training_panel(gpu_info="NVIDIA RTX 4090 24GB")
        assert isinstance(panel, Panel)


# ═══════════════════════════════════════════════════════════════════════════
# Backtest (F10)
# ═══════════════════════════════════════════════════════════════════════════

from aphelion.tui.screens.backtest_panel import build_backtest_panel, _render_equity_curve
from aphelion.tui.controller import BacktestProgress


class TestBacktestPanel:
    def test_builds_idle(self):
        panel = build_backtest_panel()
        assert isinstance(panel, Panel)

    def test_with_running_progress(self):
        bp = BacktestProgress(
            running=True,
            bars_processed=5000,
            total_bars=10000,
            elapsed_seconds=45,
            message="Running...",
        )
        panel = build_backtest_panel(bp)
        assert isinstance(panel, Panel)

    def test_with_results(self):
        bp = BacktestProgress(
            running=False,
            results={
                "total_return_pct": 12.5,
                "sharpe": 1.8,
                "max_drawdown_pct": 3.2,
                "total_trades": 150,
                "win_rate": 0.62,
                "profit_factor": 1.75,
            },
        )
        panel = build_backtest_panel(bp)
        assert isinstance(panel, Panel)

    def test_with_config(self):
        config = {"symbol": "XAUUSD", "start_date": "2024-01-01", "end_date": "2025-01-01"}
        panel = build_backtest_panel(config=config)
        assert isinstance(panel, Panel)

    def test_with_equity_curve(self):
        curve = [10000, 10100, 10050, 10200, 10300, 10250, 10400, 10500]
        panel = build_backtest_panel(equity_curve=curve)
        assert isinstance(panel, Panel)


class TestEquityCurveRenderer:
    def test_renders_curve(self):
        data = [10000 + i * 10 for i in range(100)]
        txt = _render_equity_curve(data)
        from rich.text import Text
        assert isinstance(txt, Text)

    def test_empty_data(self):
        txt = _render_equity_curve([])
        assert txt is not None

    def test_single_point(self):
        txt = _render_equity_curve([10000])
        assert txt is not None

    def test_flat_curve(self):
        txt = _render_equity_curve([10000] * 50)
        assert txt is not None


# ═══════════════════════════════════════════════════════════════════════════
# Dashboard routing for new views
# ═══════════════════════════════════════════════════════════════════════════

from aphelion.tui.screens.dashboard import build_dashboard_layout
from aphelion.tui.state import TUIState


class TestDashboardNewViews:
    def test_launcher_view(self):
        state = TUIState()
        layout = build_dashboard_layout(state, view="launcher")
        assert isinstance(layout, Layout)

    def test_setup_view(self):
        state = TUIState()
        layout = build_dashboard_layout(state, view="setup")
        assert isinstance(layout, Layout)

    def test_hephaestus_view(self):
        state = TUIState()
        layout = build_dashboard_layout(state, view="hephaestus")
        assert isinstance(layout, Layout)

    def test_training_view(self):
        state = TUIState()
        layout = build_dashboard_layout(state, view="training")
        assert isinstance(layout, Layout)

    def test_backtest_view(self):
        state = TUIState()
        layout = build_dashboard_layout(state, view="backtest")
        assert isinstance(layout, Layout)

    def test_evolution_view(self):
        state = TUIState()
        layout = build_dashboard_layout(state, view="evolution")
        assert isinstance(layout, Layout)

    def test_sola_view(self):
        state = TUIState()
        layout = build_dashboard_layout(state, view="sola")
        assert isinstance(layout, Layout)

    def test_overview_still_works(self):
        state = TUIState()
        layout = build_dashboard_layout(state, view="overview")
        assert isinstance(layout, Layout)

    def test_launcher_with_config_attached(self):
        state = TUIState()
        state._aphelion_config = AphelionConfig()
        layout = build_dashboard_layout(state, view="launcher")
        assert isinstance(layout, Layout)
