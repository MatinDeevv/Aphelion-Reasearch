"""
APHELION TUI — Persistent Configuration (Phase 23)

Loads, saves, and validates user configuration from a JSON file.
All fields that used to be CLI args are now config fields.
On first run, creates ``config/aphelion.json`` with sensible defaults.

The config object converts cleanly into PaperRunnerConfig so the
session controller never parses raw JSON.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_DIR = "config"
DEFAULT_CONFIG_FILE = "aphelion.json"
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_FILE)


# ─── Section dataclasses ────────────────────────────────────────────────────


@dataclass
class TradingConfig:
    """Trading session parameters."""
    symbol: str = "XAUUSD"
    capital: float = 10_000.0
    warmup_bars: int = 200
    mode: str = "paper"  # paper | simulated | backtest


@dataclass
class MT5Config:
    """MetaTrader 5 connection details."""
    login: int = 0
    password: str = ""
    server: str = ""
    terminal_path: str = ""


@dataclass
class RiskConfig:
    """Risk parameters mapped to SENTINEL limits."""
    max_daily_dd: float = 0.03
    max_exposure: float = 0.06
    max_positions: int = 3
    risk_per_trade: float = 0.01


@dataclass
class HydraConfig:
    """HYDRA model configuration."""
    checkpoint: str = "models/hydra_checkpoint.pt"
    min_confidence: float = 0.65


@dataclass
class TrainingPreset:
    """Training configuration preset."""
    name: str = "custom"
    data_source: str = "synthetic"   # synthetic | real
    data_path: str = ""
    epochs: int = 20
    batch_size: int = 512
    train_split: float = 0.80
    val_split: float = 0.10
    test_split: float = 0.10
    save_path: str = "models/hydra_checkpoint.pt"


@dataclass
class BacktestConfig:
    """Backtest parameters."""
    symbol: str = "XAUUSD"
    start_date: str = "2023-01-01"
    end_date: str = "2025-01-01"
    capital: float = 10_000.0
    commission_pips: float = 0.35
    slippage: str = "adaptive"
    data_path: str = ""


@dataclass
class AphelionConfig:
    """Complete APHELION configuration — persisted as JSON.

    Loaded at startup from ``config/aphelion.json``.
    Saved whenever the user presses [S] in the SETUP screen.
    """
    trading: TradingConfig = field(default_factory=TradingConfig)
    mt5: MT5Config = field(default_factory=MT5Config)
    risk: RiskConfig = field(default_factory=RiskConfig)
    hydra: HydraConfig = field(default_factory=HydraConfig)
    training: TrainingPreset = field(default_factory=TrainingPreset)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    _path: str = field(default=DEFAULT_CONFIG_PATH, repr=False)
    first_run: bool = True

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict (excludes private fields)."""
        d = asdict(self)
        d.pop("_path", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str = DEFAULT_CONFIG_PATH) -> "AphelionConfig":
        """Construct from a parsed JSON dict."""
        cfg = cls(_path=path)
        if "trading" in data:
            cfg.trading = TradingConfig(**{k: v for k, v in data["trading"].items()
                                           if k in TradingConfig.__dataclass_fields__})
        if "mt5" in data:
            cfg.mt5 = MT5Config(**{k: v for k, v in data["mt5"].items()
                                   if k in MT5Config.__dataclass_fields__})
        if "risk" in data:
            cfg.risk = RiskConfig(**{k: v for k, v in data["risk"].items()
                                     if k in RiskConfig.__dataclass_fields__})
        if "hydra" in data:
            cfg.hydra = HydraConfig(**{k: v for k, v in data["hydra"].items()
                                       if k in HydraConfig.__dataclass_fields__})
        if "training" in data:
            cfg.training = TrainingPreset(**{k: v for k, v in data["training"].items()
                                             if k in TrainingPreset.__dataclass_fields__})
        if "backtest" in data:
            cfg.backtest = BacktestConfig(**{k: v for k, v in data["backtest"].items()
                                             if k in BacktestConfig.__dataclass_fields__})
        cfg.first_run = data.get("first_run", False)
        return cfg

    def save(self, path: str | None = None) -> None:
        """Persist config to JSON file."""
        target = path or self._path
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Config saved → %s", target)

    # ── Validation ───────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: list[str] = []
        if self.trading.capital <= 0:
            errors.append("Starting capital must be positive")
        if self.trading.warmup_bars < 10:
            errors.append("Warmup bars must be ≥ 10")
        if self.risk.max_daily_dd <= 0 or self.risk.max_daily_dd > 1:
            errors.append("Max daily DD must be in (0, 1]")
        if self.risk.max_exposure <= 0 or self.risk.max_exposure > 1:
            errors.append("Max exposure must be in (0, 1]")
        if self.risk.max_positions < 1:
            errors.append("Max positions must be ≥ 1")
        if self.risk.risk_per_trade <= 0 or self.risk.risk_per_trade > 0.10:
            errors.append("Risk per trade must be in (0, 0.10]")
        if self.hydra.min_confidence < 0 or self.hydra.min_confidence > 1:
            errors.append("HYDRA min confidence must be in [0, 1]")
        return errors

    # ── Status checks ────────────────────────────────────────────────

    def has_mt5_credentials(self) -> bool:
        """Whether MT5 connection details are configured."""
        return bool(self.mt5.login and self.mt5.server)

    def has_hydra_checkpoint(self) -> bool:
        """Whether a HYDRA checkpoint file exists."""
        return os.path.isfile(self.hydra.checkpoint)

    # ── Reset ────────────────────────────────────────────────────────

    def reset_to_defaults(self) -> None:
        """Reset all fields to defaults (except path)."""
        path = self._path
        default = AphelionConfig()
        self.__dict__.update(default.__dict__)
        self._path = path


def load_config(path: str = DEFAULT_CONFIG_PATH) -> AphelionConfig:
    """Load config from file, or create defaults if missing."""
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg = AphelionConfig.from_dict(data, path=path)
            logger.info("Config loaded from %s", path)
            return cfg
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse config %s: %s — using defaults", path, exc)

    # First run — create defaults
    cfg = AphelionConfig(_path=path, first_run=True)
    cfg.save(path)
    logger.info("Created default config at %s", path)
    return cfg


# ─── Config → PaperRunnerConfig converter ────────────────────────────────


def config_to_runner_config(config: AphelionConfig, mode: str = "paper"):
    """Convert ``AphelionConfig`` + mode string → ``PaperRunnerConfig``.

    This is the single place that bridges the TUI config world and the
    paper-runner world.  Returns a PaperRunnerConfig suitable for
    ``PaperRunner(runner_config)``.

    Falls back to a minimal config if the paper module is unavailable.
    """
    try:
        from aphelion.paper.runner import PaperRunnerConfig, FeedMode

        if mode == "simulated":
            feed_mode = FeedMode.SIMULATED
        elif mode == "backtest":
            feed_mode = FeedMode.SIMULATED  # backtest uses simulated feed
        else:
            feed_mode = FeedMode.SIMULATED  # default to simulated

        runner_config = PaperRunnerConfig(
            feed_mode=feed_mode,
            enable_tui=True,
        )

        # Map MT5 fields if present
        if config.has_mt5_credentials():
            try:
                runner_config.mt5_config = {
                    "login": config.mt5.login,
                    "password": config.mt5.password,
                    "server": config.mt5.server,
                    "terminal_path": config.mt5.terminal_path,
                }
            except AttributeError:
                pass

        # Map session / risk fields where the runner supports them
        for attr, val in [
            ("symbol", config.trading.symbol),
            ("initial_capital", config.trading.capital),
            ("warmup_bars", config.trading.warmup_bars),
        ]:
            if hasattr(runner_config, attr):
                setattr(runner_config, attr, val)

        return runner_config

    except ImportError:
        # paper module not importable — return a dict placeholder
        return {
            "feed_mode": mode,
            "symbol": config.trading.symbol,
            "capital": config.trading.capital,
            "enable_tui": True,
        }
