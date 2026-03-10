"""
Tests for APHELION Phase 23 — Config Layer.

Covers AphelionConfig, section dataclasses, serialization, validation,
load_config, and save/reset functionality.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from aphelion.tui.config import (
    AphelionConfig,
    BacktestConfig,
    HydraConfig,
    MT5Config,
    RiskConfig,
    TradingConfig,
    TrainingPreset,
    config_to_runner_config,
    load_config,
)


# ═══════════════════════════════════════════════════════════════════════════
# Section dataclasses
# ═══════════════════════════════════════════════════════════════════════════


class TestTradingConfig:
    def test_defaults(self):
        cfg = TradingConfig()
        assert cfg.symbol == "XAUUSD"
        assert cfg.capital == 10_000.0
        assert cfg.warmup_bars == 200
        assert cfg.mode == "paper"

    def test_custom_values(self):
        cfg = TradingConfig(symbol="EURUSD", capital=50_000, mode="simulated")
        assert cfg.symbol == "EURUSD"
        assert cfg.capital == 50_000
        assert cfg.mode == "simulated"


class TestMT5Config:
    def test_defaults(self):
        cfg = MT5Config()
        assert cfg.login == 0
        assert cfg.password == ""
        assert cfg.server == ""
        assert cfg.terminal_path == ""


class TestRiskConfig:
    def test_defaults(self):
        cfg = RiskConfig()
        assert cfg.max_daily_dd == 0.03
        assert cfg.max_positions == 3
        assert cfg.risk_per_trade == 0.01


class TestHydraConfig:
    def test_defaults(self):
        cfg = HydraConfig()
        assert cfg.min_confidence == 0.65
        assert "hydra_checkpoint" in cfg.checkpoint


class TestTrainingPreset:
    def test_defaults(self):
        preset = TrainingPreset()
        assert preset.name == "custom"
        assert preset.epochs == 20
        assert preset.data_source == "synthetic"
        assert preset.train_split + preset.val_split + preset.test_split == pytest.approx(1.0)


class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.symbol == "XAUUSD"
        assert cfg.capital == 10_000.0
        assert cfg.commission_pips == 0.35


# ═══════════════════════════════════════════════════════════════════════════
# AphelionConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestAphelionConfig:
    def test_default_creation(self):
        cfg = AphelionConfig()
        assert isinstance(cfg.trading, TradingConfig)
        assert isinstance(cfg.mt5, MT5Config)
        assert isinstance(cfg.risk, RiskConfig)
        assert isinstance(cfg.hydra, HydraConfig)
        assert isinstance(cfg.training, TrainingPreset)
        assert isinstance(cfg.backtest, BacktestConfig)
        assert cfg.first_run is True

    def test_to_dict_excludes_private(self):
        cfg = AphelionConfig()
        d = cfg.to_dict()
        assert "_path" not in d
        assert "trading" in d
        assert "mt5" in d
        assert "risk" in d

    def test_roundtrip_serialization(self):
        cfg = AphelionConfig()
        cfg.trading.symbol = "GBPUSD"
        cfg.risk.max_positions = 5
        cfg.hydra.min_confidence = 0.80
        cfg.first_run = False

        d = cfg.to_dict()
        restored = AphelionConfig.from_dict(d)

        assert restored.trading.symbol == "GBPUSD"
        assert restored.risk.max_positions == 5
        assert restored.hydra.min_confidence == 0.80
        assert restored.first_run is False

    def test_from_dict_partial_data(self):
        """from_dict should handle missing sub-sections gracefully."""
        cfg = AphelionConfig.from_dict({"trading": {"symbol": "NZDUSD"}})
        assert cfg.trading.symbol == "NZDUSD"
        # Other sections should have defaults
        assert cfg.risk.max_daily_dd == 0.03
        assert cfg.mt5.login == 0

    def test_from_dict_extra_keys_ignored(self):
        data = {"trading": {"symbol": "EURUSD", "unknown_key": 42}}
        cfg = AphelionConfig.from_dict(data)
        assert cfg.trading.symbol == "EURUSD"


class TestConfigValidation:
    def test_valid_config(self):
        cfg = AphelionConfig()
        errors = cfg.validate()
        assert errors == []

    def test_zero_capital(self):
        cfg = AphelionConfig()
        cfg.trading.capital = 0
        errors = cfg.validate()
        assert any("capital" in e.lower() for e in errors)

    def test_negative_capital(self):
        cfg = AphelionConfig()
        cfg.trading.capital = -100
        errors = cfg.validate()
        assert any("capital" in e.lower() for e in errors)

    def test_low_warmup_bars(self):
        cfg = AphelionConfig()
        cfg.trading.warmup_bars = 5
        errors = cfg.validate()
        assert any("warmup" in e.lower() for e in errors)

    def test_bad_max_daily_dd(self):
        cfg = AphelionConfig()
        cfg.risk.max_daily_dd = 1.5
        errors = cfg.validate()
        assert any("dd" in e.lower() for e in errors)

    def test_bad_max_exposure(self):
        cfg = AphelionConfig()
        cfg.risk.max_exposure = -0.1
        errors = cfg.validate()
        assert any("exposure" in e.lower() for e in errors)

    def test_zero_max_positions(self):
        cfg = AphelionConfig()
        cfg.risk.max_positions = 0
        errors = cfg.validate()
        assert any("positions" in e.lower() for e in errors)

    def test_bad_risk_per_trade(self):
        cfg = AphelionConfig()
        cfg.risk.risk_per_trade = 0.15
        errors = cfg.validate()
        assert any("risk per trade" in e.lower() for e in errors)

    def test_bad_confidence(self):
        cfg = AphelionConfig()
        cfg.hydra.min_confidence = -0.5
        errors = cfg.validate()
        assert any("confidence" in e.lower() for e in errors)

    def test_multiple_errors(self):
        cfg = AphelionConfig()
        cfg.trading.capital = -1
        cfg.risk.max_daily_dd = 5.0
        cfg.risk.risk_per_trade = 0.50
        errors = cfg.validate()
        assert len(errors) >= 3


class TestConfigStatusChecks:
    def test_has_mt5_credentials_false(self):
        cfg = AphelionConfig()
        assert cfg.has_mt5_credentials() is False

    def test_has_mt5_credentials_true(self):
        cfg = AphelionConfig()
        cfg.mt5.login = 12345
        cfg.mt5.server = "MetaQuotes-Demo"
        assert cfg.has_mt5_credentials() is True

    def test_has_hydra_checkpoint_default(self):
        cfg = AphelionConfig()
        # Default path likely doesn't exist
        assert cfg.has_hydra_checkpoint() is False

    def test_has_hydra_checkpoint_with_real_file(self, tmp_path):
        p = tmp_path / "checkpoint.pt"
        p.write_text("dummy")
        cfg = AphelionConfig()
        cfg.hydra.checkpoint = str(p)
        assert cfg.has_hydra_checkpoint() is True


class TestConfigReset:
    def test_reset_to_defaults(self):
        cfg = AphelionConfig()
        cfg.trading.symbol = "SOMETHING"
        cfg.risk.max_positions = 99
        cfg.reset_to_defaults()
        assert cfg.trading.symbol == "XAUUSD"
        assert cfg.risk.max_positions == 3


# ═══════════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigPersistence:
    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "test_config.json")
        cfg = AphelionConfig(_path=path)
        cfg.trading.symbol = "AUDCHF"
        cfg.trading.capital = 25_000
        cfg.first_run = False
        cfg.save(path)

        assert os.path.isfile(path)
        loaded = load_config(path)
        assert loaded.trading.symbol == "AUDCHF"
        assert loaded.trading.capital == 25_000

    def test_load_creates_default_on_missing(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        cfg = load_config(path)
        assert cfg.first_run is True
        assert os.path.isfile(path)

    def test_load_handles_corrupt_json(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("NOT VALID JSON {{{")
        cfg = load_config(path)
        assert isinstance(cfg, AphelionConfig)
        assert cfg.trading.symbol == "XAUUSD"  # defaults

    def test_save_creates_directory(self, tmp_path):
        path = str(tmp_path / "nested" / "deep" / "config.json")
        cfg = AphelionConfig(_path=path)
        cfg.save(path)
        assert os.path.isfile(path)


# ═══════════════════════════════════════════════════════════════════════════
# config_to_runner_config
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigToRunnerConfig:
    def test_returns_something(self):
        """Should always return a config object (dict or PaperRunnerConfig)."""
        cfg = AphelionConfig()
        result = config_to_runner_config(cfg, "simulated")
        assert result is not None

    def test_simulated_mode(self):
        cfg = AphelionConfig()
        result = config_to_runner_config(cfg, "simulated")
        # Either a PaperRunnerConfig or dict fallback
        if isinstance(result, dict):
            assert result["feed_mode"] == "simulated"
            assert result["enable_tui"] is True
        else:
            assert result.enable_tui is True

    def test_paper_mode(self):
        cfg = AphelionConfig()
        result = config_to_runner_config(cfg, "paper")
        assert result is not None

    def test_preserves_symbol(self):
        cfg = AphelionConfig()
        cfg.trading.symbol = "EURUSD"
        result = config_to_runner_config(cfg, "simulated")
        if isinstance(result, dict):
            assert result["symbol"] == "EURUSD"

    def test_preserves_capital(self):
        cfg = AphelionConfig()
        cfg.trading.capital = 50_000
        result = config_to_runner_config(cfg, "paper")
        if isinstance(result, dict):
            assert result["capital"] == 50_000
