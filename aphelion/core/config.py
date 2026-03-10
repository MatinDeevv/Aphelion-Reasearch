"""
APHELION System Configuration
All system constants, SENTINEL hard limits, session times, and resource allocations.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Final


# ─── Market Sessions (UTC) ────────────────────────────────────────────────────

class Session(Enum):
    ASIAN = auto()
    LONDON = auto()
    NEW_YORK = auto()
    OVERLAP_LDN_NY = auto()
    DEAD_ZONE = auto()


@dataclass(frozen=True)
class SessionWindow:
    name: Session
    open_hour: int   # UTC hour
    open_minute: int
    close_hour: int
    close_minute: int

    def adjusted(self, offset_minutes: int = 0) -> "SessionWindow":
        """Return a new SessionWindow shifted by *offset_minutes* (e.g. -60 for summer DST)."""
        if offset_minutes == 0:
            return self
        o = self.open_hour * 60 + self.open_minute + offset_minutes
        c = self.close_hour * 60 + self.close_minute + offset_minutes
        return SessionWindow(
            name=self.name,
            open_hour=(o // 60) % 24,
            open_minute=o % 60,
            close_hour=(c // 60) % 24,
            close_minute=c % 60,
        )


SESSION_WINDOWS: Final = [
    SessionWindow(Session.ASIAN,          0,  0,  8, 0),
    SessionWindow(Session.LONDON,         8,  0, 12, 0),
    SessionWindow(Session.OVERLAP_LDN_NY, 12, 0, 16, 30),
    SessionWindow(Session.NEW_YORK,       16, 30, 21, 0),
    SessionWindow(Session.DEAD_ZONE,      21, 0, 24, 0),
]


# ─── SENTINEL Hard Limits (Section 6.4 — NON-NEGOTIABLE) ─────────────────────

@dataclass(frozen=True)
class SentinelLimits:
    max_position_pct: float = 0.02          # 2% of account per trade
    max_simultaneous_positions: int = 3     # Max 3 open at once
    stop_loss_mandatory: bool = True        # Every trade MUST have SL
    min_risk_reward: float = 1.5            # Minimum 1.5:1 R:R
    pre_news_lockout_minutes: int = 5       # Block 5 min before high-impact
    post_news_lockout_minutes: int = 2      # Block 2 min after high-impact
    friday_close_lockout_minutes: int = 30  # Close all 30 min before market close
    daily_equity_drawdown_l1: float = 0.03  # 3% drawdown → L1 warning (reduce size)
    daily_equity_drawdown_l2: float = 0.06  # 6% drawdown → L2 halt (no new trades)
    daily_equity_drawdown_l3: float = 0.10  # 10% drawdown → L3 disconnect
    lot_size_oz: float = 100.0              # XAU/USD lot size: 1 lot = 100 oz

    def __post_init__(self) -> None:
        """Validate invariants — breaker tiers must be strictly ascending."""
        if not (0 < self.daily_equity_drawdown_l1
                < self.daily_equity_drawdown_l2
                < self.daily_equity_drawdown_l3 <= 1.0):
            raise ValueError(
                f"Breaker tiers must satisfy 0 < L1 < L2 < L3 <= 1.0, "
                f"got L1={self.daily_equity_drawdown_l1}, "
                f"L2={self.daily_equity_drawdown_l2}, "
                f"L3={self.daily_equity_drawdown_l3}"
            )
        if self.max_position_pct <= 0:
            raise ValueError(f"max_position_pct must be > 0, got {self.max_position_pct}")
        if self.min_risk_reward <= 0:
            raise ValueError(f"min_risk_reward must be > 0, got {self.min_risk_reward}")
        if self.max_simultaneous_positions < 1:
            raise ValueError(f"max_simultaneous_positions must be >= 1, got {self.max_simultaneous_positions}")


SENTINEL: Final = SentinelLimits()


# ─── Leverage Tiers (Section 6.1) ────────────────────────────────────────────

@dataclass(frozen=True)
class LeverageTier:
    conditions_met: int
    min_leverage: float
    max_leverage: float


LEVERAGE_TIERS: Final = [
    LeverageTier(7, 50.0, 100.0),   # All 7 conditions
    LeverageTier(5, 20.0, 50.0),    # 5-6 conditions
    LeverageTier(3, 5.0, 20.0),     # 3-4 conditions
    LeverageTier(0, 1.0, 2.0),      # Under 3 conditions
]


# ─── Kelly Criterion (Section 6.3) ───────────────────────────────────────────

KELLY_FRACTION: Final[float] = 0.25  # Quarter-Kelly for safety
KELLY_MAX_F: Final[float] = 0.02    # Hard cap at 2% regardless of Kelly output


# ─── Timeframes ──────────────────────────────────────────────────────────────

class Timeframe(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    D1 = "1d"
    W1 = "1w"


TIMEFRAMES: Final = [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1]

# Canonical seconds per timeframe — single source of truth for all modules
TIMEFRAME_SECONDS: Final[dict[Timeframe, int]] = {
    Timeframe.M1: 60,
    Timeframe.M5: 300,
    Timeframe.M15: 900,
    Timeframe.H1: 3600,
    Timeframe.D1: 86400,
    Timeframe.W1: 604800,
}


# ─── Event Bus Topics ────────────────────────────────────────────────────────

class EventTopic(Enum):
    TICK = "tick"
    BAR = "bar"
    SIGNAL = "signal"
    RISK = "risk"
    VOTE = "vote"
    SYSTEM = "system"
    HEALTH = "health"
    EVOLUTION = "evolution"


# ─── Component Registry ─────────────────────────────────────────────────────

class ComponentStatus(Enum):
    ACTIVE = auto()
    PAUSED = auto()
    ERROR = auto()
    DISABLED = auto()
    INITIALIZING = auto()


class Tier(Enum):
    SOVEREIGN = 1       # SOLA — infinite votes
    GENERAL = 2         # OLYMPUS — 20 votes
    ORACLE = 3          # HYDRA Ensemble — 15 votes
    COMMANDER = 4       # PROMETHEUS, FLOW, MACRO, ATLAS LIVE — 10 votes
    LIEUTENANT = 5      # NEMESIS, FORGE, SHADOW — 5 votes
    SERGEANT = 6        # KRONOS, ECHO, MERIDIAN — 2 votes
    PRIVATE = 7         # Individual sub-models within HYDRA — 1 vote


TIER_VOTE_WEIGHTS: Final = {
    Tier.SOVEREIGN: float('inf'),
    Tier.GENERAL: 20,
    Tier.ORACLE: 15,
    Tier.COMMANDER: 10,
    Tier.LIEUTENANT: 5,
    Tier.SERGEANT: 2,
    Tier.PRIVATE: 1,
}


# ─── Module Definitions ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModuleInfo:
    name: str
    tier: Tier
    description: str


MODULES: Final = {
    # Tier 2 — General
    "OLYMPUS":    ModuleInfo("OLYMPUS", Tier.GENERAL, "System Governor & Auto-Tuner"),
    "SENTINEL":   ModuleInfo("SENTINEL", Tier.GENERAL, "Supreme Risk Authority"),
    "ARES":       ModuleInfo("ARES", Tier.GENERAL, "LLM Brain"),

    # Tier 3 — Oracle
    "HYDRA":      ModuleInfo("HYDRA", Tier.ORACLE, "Neural Intelligence Core"),

    # Tier 4 — Commander
    "PROMETHEUS": ModuleInfo("PROMETHEUS", Tier.COMMANDER, "Evolution Engine"),
    "PHANTOM":    ModuleInfo("PHANTOM", Tier.COMMANDER, "Institutional Flow Detection"),
    "ATLAS":      ModuleInfo("ATLAS", Tier.COMMANDER, "Macro Intelligence"),
    "DATA":       ModuleInfo("DATA", Tier.COMMANDER, "Data Layer & Features"),
    "SIGNAL_TOWER": ModuleInfo("SIGNAL_TOWER", Tier.COMMANDER, "Technical Analysis Voters"),
    "FLOW":       ModuleInfo("FLOW", Tier.COMMANDER, "Liquidity & Microstructure"),
    "MACRO":      ModuleInfo("MACRO", Tier.COMMANDER, "Market Regime Intelligence"),

    # Tier 5 — Lieutenant
    "NEMESIS":    ModuleInfo("NEMESIS", Tier.LIEUTENANT, "War Simulator"),
    "FORGE":      ModuleInfo("FORGE", Tier.LIEUTENANT, "Online Learning"),
    "SHADOW":     ModuleInfo("SHADOW", Tier.LIEUTENANT, "Personal Trading DNA"),

    # Tier 6 — Sergeant
    "KRONOS":     ModuleInfo("KRONOS", Tier.SERGEANT, "Trade Journal & Analytics"),
    "ECHO":       ModuleInfo("ECHO", Tier.SERGEANT, "Historical Analogs"),
    "MERIDIAN":   ModuleInfo("MERIDIAN", Tier.SERGEANT, "Multi-Timeframe Weights"),

    # Tier 7 — Private
    "BACKTEST":   ModuleInfo("BACKTEST", Tier.PRIVATE, "Backtesting Engine & Monte Carlo"),
    "VENOM":      ModuleInfo("VENOM", Tier.PRIVATE, "Statistical Arbitrage"),
    "REAPER":     ModuleInfo("REAPER", Tier.PRIVATE, "Liquidity Vacuum"),
    "APEX":       ModuleInfo("APEX", Tier.PRIVATE, "Volatility Breakout"),
    "WRAITH":     ModuleInfo("WRAITH", Tier.PRIVATE, "News Reversion"),
    "CASSANDRA":  ModuleInfo("CASSANDRA", Tier.PRIVATE, "24H Direction Predictor"),
    "ORACLE":     ModuleInfo("ORACLE", Tier.PRIVATE, "Macro Regime Decoder"),
    "TITAN":      ModuleInfo("TITAN", Tier.PRIVATE, "Quality Gate"),
    "GHOST":      ModuleInfo("GHOST", Tier.PRIVATE, "Stealth Execution"),
    "FUND":       ModuleInfo("FUND", Tier.PRIVATE, "Performance Reporting"),
}


# ─── CPU Core Allocation (Section 2.2) ──────────────────────────────────────

CPU_ALLOCATION_LIVE: Final = {
    0:          "DATA",        # C00: Data ingestion
    1:          "SENTINEL",    # C01: Risk monitoring (isolated)
    2:          "TUI",         # C02: Terminal UI
    3:          "CORE",        # C03: Event bus
    # C04-C08: Live module inference
    # C09-C12: Money maker signals
    # C13-C20: Background evolution (reduced)
    # C21-C31: Burst compute pool
}


# ─── GPU VRAM Allocation (Section 2.3) ──────────────────────────────────────

@dataclass(frozen=True)
class VRAMAllocation:
    module: str
    size_gb: float
    purpose: str


VRAM_ALLOCATION_LIVE: Final = [
    VRAMAllocation("HYDRA", 8.0, "Neural inference"),
    VRAMAllocation("ARES", 12.0, "LLM reasoning"),
    VRAMAllocation("RAPIDS", 2.0, "Feature computation"),
    VRAMAllocation("BURST", 2.0, "Emergency compute"),
]


# ─── External API Configuration ─────────────────────────────────────────────

@dataclass(frozen=True)
class APIConfig:
    name: str
    update_interval_seconds: int
    latency_target_ms: int


EXTERNAL_APIS: Final = {
    "MT5":       APIConfig("MetaTrader 5", 0, 10),        # Real-time
    "FRED":      APIConfig("FRED API", 86400, 50),         # Daily
    "GDELT":     APIConfig("GDELT Project", 900, 100),     # 15 min
    "REDDIT":    APIConfig("Reddit API", 60, 500),         # Every 60s
    "TWITTER":   APIConfig("Twitter/X API", 60, 500),      # Every 60s
    "YAHOO":     APIConfig("Yahoo Finance", 300, 500),     # Every 5min
    "KITCO":     APIConfig("Kitco RSS", 900, 500),         # Every 15min
    "CFTC_COT":  APIConfig("CFTC COT", 604800, 2000),     # Weekly
    "CME_FW":    APIConfig("CME FedWatch", 0, 100),        # Real-time
}


# ─── NEMESIS Gauntlet Levels (Section 8.1) ───────────────────────────────────

class LeviathanMode(Enum):
    PASSIVE = auto()
    MIRROR = auto()
    HUNTER = auto()
    FOGGER = auto()
    AVALANCHE = auto()
    ALL_MODES = auto()
    MAXIMUM = auto()


@dataclass(frozen=True)
class GauntletLevel:
    level: int
    name: str
    era: str
    min_seed_leverage: float
    max_seed_leverage: float
    leviathan_mode: LeviathanMode


GAUNTLET_LEVELS: Final = [
    GauntletLevel(1, "INITIATION", "1971-1986", 3, 5, LeviathanMode.PASSIVE),
    GauntletLevel(2, "APPRENTICE", "1987-1999", 5, 10, LeviathanMode.MIRROR),
    GauntletLevel(3, "JOURNEYMAN", "2000-2007", 10, 20, LeviathanMode.HUNTER),
    GauntletLevel(4, "VETERAN", "2008", 15, 30, LeviathanMode.FOGGER),
    GauntletLevel(5, "ELITE", "2020", 20, 40, LeviathanMode.AVALANCHE),
    GauntletLevel(6, "MASTER", "All eras", 30, 50, LeviathanMode.ALL_MODES),
    GauntletLevel(7, "NEMESIS MODE", "Random", 3, 50, LeviathanMode.MAXIMUM),
]

NEMESIS_PASS_THRESHOLD: Final[int] = 80
NEMESIS_DISTINCTION_THRESHOLD: Final[int] = 90


# ─── Feature Engine Constants ────────────────────────────────────────────────

VPIN_BUCKET_SIZE: Final[int] = 50          # Volume bucket size for VPIN
OFI_FILTER_MS: Final[int] = 50            # 50ms OFI filter
HAWKES_DECAY: Final[float] = 0.1          # Hawkes process decay parameter
ENTROPY_WINDOW: Final[int] = 100          # Tick window for Shannon entropy
SWING_CONFIRMATION_BARS: Final[int] = 5   # N-bar fractal confirmation
FVG_MIN_GAP_PIPS: Final[float] = 1.0      # Minimum FVG gap in pips
LIQUIDITY_POOL_TOLERANCE_PIPS: Final[float] = 5.0  # Equal high/low tolerance
VOLUME_IMBALANCE_MULTIPLIER: Final[float] = 2.0    # >2x avg volume

# VWAP
VWAP_STD_BANDS: Final[list] = [1, 2]      # ±1σ and ±2σ bands

# Cointegration
COINTEGRATION_WINDOW: Final[int] = 50     # Rolling window (bars)
COINTEGRATION_PVALUE: Final[float] = 0.05 # Significance threshold

# VENOM thresholds
VENOM_SPREAD_ENTRY: Final[float] = 2.0    # 2 std dev entry
VENOM_WIN_RATE_TARGET: Final[float] = 0.72  # 72-78% historical

# APEX thresholds
APEX_BB_PERCENTILE: Final[int] = 15       # Below 15th percentile
APEX_ATR_CONTRACTING_BARS: Final[int] = 8 # 8+ bars contracting
APEX_VPIN_THRESHOLD: Final[float] = 0.4   # VPIN below 0.4
APEX_MIN_RR: Final[float] = 3.0           # Minimum 1:3 R:R

# WRAITH thresholds
WRAITH_ATR_MULTIPLIER: Final[float] = 1.5 # Spike > 1.5x ATR
WRAITH_ENTRY_DELAY_MIN_SEC: Final[int] = 45   # Min wait 45s
WRAITH_ENTRY_DELAY_MAX_SEC: Final[int] = 90   # Max wait 90s
WRAITH_RETRACEMENT_MIN: Final[float] = 0.38   # 38% retracement target
WRAITH_RETRACEMENT_MAX: Final[float] = 0.50   # 50% retracement target


# ─── Cross-Asset Symbols (NEXUS) ────────────────────────────────────────────

NEXUS_ASSETS: Final[list] = [
    "DXY", "US10Y", "US2Y", "TLT", "GLD",
    "SLV", "XLE", "VIX", "SP500", "EURUSD",
]


# ─── System Paths ────────────────────────────────────────────────────────────

import os

PROJECT_ROOT: Final = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR: Final = os.path.join(PROJECT_ROOT, "data")
TICK_DIR: Final = os.path.join(DATA_DIR, "ticks")
BAR_DIR: Final = os.path.join(DATA_DIR, "bars")
FEATURE_CACHE_DIR: Final = os.path.join(DATA_DIR, "features")
EXTERNAL_DATA_DIR: Final = os.path.join(DATA_DIR, "external")

MODEL_DIR: Final = os.path.join(PROJECT_ROOT, "models")
GENOME_DIR: Final = os.path.join(PROJECT_ROOT, "genomes")
LOG_DIR: Final = os.path.join(PROJECT_ROOT, "logs")

TRADES_DB: Final = os.path.join(LOG_DIR, "trades.db")
VOTES_DB: Final = os.path.join(LOG_DIR, "votes.db")
ARES_DB: Final = os.path.join(LOG_DIR, "ares.db")
NEMESIS_DB: Final = os.path.join(LOG_DIR, "nemesis.db")
SYSTEM_LOG: Final = os.path.join(LOG_DIR, "system.log")
AUDIT_LOG: Final = os.path.join(LOG_DIR, "audit.log")


# ─── Trading Symbol ──────────────────────────────────────────────────────────

SYMBOL: Final[str] = "XAUUSD"
PIP_SIZE: Final[float] = 0.01  # Gold pip = $0.01
