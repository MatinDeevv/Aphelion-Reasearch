# APHELION

Fully autonomous XAU/USD (Gold) trading system built on a 20-module architecture with tiered governance, hardcoded risk limits, and 60+ engineered features per bar.

---

## Architecture

APHELION uses a 5-tier governance hierarchy where modules vote on trade decisions with weighted authority. Every order must survive SENTINEL validation before it reaches the broker.

| Tier | Role | Votes | Modules |
|------|------|-------|---------|
| **Sovereign** | System Core | ∞ | Event Bus, Clock, Registry |
| **Council** | Strategic Decision | 100 | OLYMPUS, SENTINEL, ARES |
| **Minister** | Intelligence | 40 | HYDRA, PROMETHEUS, PHANTOM, NEMESIS, FORGE, ATLAS, DATA |
| **Commander** | Execution | 10 | BACKTEST, VENOM, REAPER, APEX, WRAITH, SHADOW, KRONOS, ECHO, CASSANDRA, ORACLE, TITAN, GHOST |
| **Operator** | Support | 1 | FUND |

### Module Overview

```
OLYMPUS      System Governor & Auto-Tuner
SENTINEL     Supreme Risk Authority (hardcoded, immutable limits)
ARES         LLM Brain (Mixtral 8x7B → GPT-OSS-120B)
HYDRA        Neural Intelligence Core (TFT+LSTM+CNN+MoE+Gate, ~314M params)
PROMETHEUS   Evolution Engine (NEAT + PBT + Bayesian)
PHANTOM      Institutional Flow Detection
NEMESIS      War Simulator (7 Gauntlet levels)
FORGE        Online Learning (MAML)
ATLAS        Macro Intelligence
DATA         Data Layer & 60+ Features
BACKTEST     Backtesting Engine & Monte Carlo
VENOM        Statistical Arbitrage (Cointegration)
REAPER       Momentum/Trend Following
APEX         Breakout Detection
WRAITH       Mean Reversion
SHADOW       Personal Trading DNA
KRONOS       TimesFM Foundation Model
ECHO         Historical Analog Matching
CASSANDRA    24H Direction Predictor
ORACLE       Macro Regime Decoder
TITAN        Portfolio Allocation
GHOST        Stealth Execution
FUND         Performance Reporting
```

---

## Project Structure

```
aphelion/
  core/             Event bus, config, data layer, clock, registry
  features/         60+ feature engine (microstructure, market structure,
                    volume, VWAP, sessions, MTF, cointegration)
  risk/
    sentinel/       SENTINEL runtime enforcement
      core.py             SentinelCore — position tracking, L3 disconnect
      validator.py        TradeValidator — 7-rule pre-trade filter
      position_sizer.py   Quarter-Kelly sizing with hard caps
      circuit_breaker.py  CircuitBreaker — 3-level drawdown response
      monitor.py          SentinelMonitor — 100ms stop-loss polling loop
      execution/
        enforcer.py       ExecutionEnforcer — final order gate
    titan/          TITAN portfolio allocation (Phase 9)
  intelligence/     HYDRA, KRONOS, ECHO, FORGE, SHADOW
  evolution/        PROMETHEUS, CIPHER, MERIDIAN, ZEUS
  nemesis/          PANDORA, LEVIATHAN, CHRONOS, VERDICT
  macro/            ORACLE, ATLAS, NEXUS, ARGUS, HERALD
  flow/             PHANTOM, SPECTER
  money/            VENOM, REAPER, APEX, WRAITH, GHOST
  ares/             LLM brain, sentiment, reasoning, strategist
  governance/       OLYMPUS, COUNCIL
  backtest/         Engine, metrics, Monte Carlo
  aphelion_model/   APHELION-120B fine-tune pipeline
  tui/              Terminal UI (12 screens)
tests/
  core/             Config, clock, event bus, data layer, registry
  features/         Microstructure, market structure, volume profile,
                    VWAP, MTF, cointegration
  risk/             Circuit breaker, position sizer, SENTINEL core
  integration/      End-to-end pipeline and SENTINEL integration tests
```

---

## Current Status

### Phase 1: Data Foundation — Complete

Core infrastructure and feature engine. 187 tests passing.

**Core components:**
- Async priority event bus (CRITICAL SENTINEL events always dispatched first, even when queue is full)
- `MarketClock` — session detection, news lockout, Friday close, calendar markers, `is_trading_session`
- `ComponentRegistry` — health scoring, tier-based module lifecycle management
- `DataLayer` — MT5 tick ingestion, OHLCV aggregation across M1/M5/M15/H1, data quality validation, Parquet/CSV file loaders for backtesting

**60+ engineered features across 7 sub-engines:**

| Sub-engine | Features |
|-----------|---------|
| **Microstructure** | VPIN, OFI, tick entropy, Hawkes intensity, micro-price divergence, spread dynamics, quote depth |
| **Market Structure** | Order blocks, fair value gaps, swing highs/lows, liquidity pools, breaker blocks, volume imbalances, CHoCH |
| **Volume Profile** | Volume delta, cumulative delta (CVD), point of control (POC), value area high/low (VAH/VAL), absorption |
| **VWAP** | Session VWAP, anchored VWAP, rolling VWAP with ±1σ and ±2σ bands |
| **Sessions** | Session flags, time-to-open/close, news proximity, `is_trading_session`, calendar markers |
| **Multi-Timeframe** | Alignment scoring across M1/M5/M15/H1 with weighted directional consensus |
| **Cointegration** | Engle-Granger + `statsmodels` ADF for XAU vs DXY, real yields, silver; spread z-scores and half-life |

**Phase 1 fixes applied (pre-Phase 2 hardening):**
- `cointegration.py` — replaced hand-rolled ADF (hardcoded MacKinnon buckets) with `statsmodels.tsa.stattools.adfuller`; real p-values via `result[1]`
- `clock.py` — `session_features()` now includes `"is_trading_session"` key
- `config.py` — `BACKTEST` module registered (Tier.COMMANDER) for Phase 3
- `data_layer.py` — `load_from_parquet()` and `load_from_csv()` added; validate required columns, return DataFrames without touching `self._bars`

---

### Phase 2: Risk Layer (SENTINEL) — Complete

Real-time enforcement layer that wraps every trade from proposal to execution. Every order passes through a 4-stage pipeline before it can reach the broker.

**Order enforcement pipeline:**

```
TradeProposal
    │
    ▼
[SentinelCore.is_trading_allowed()]   ← L3, news lockout, Friday lockout, market hours
    │
    ▼
[TradeValidator.validate()]           ← 7 immutable rules checked in sequence
    │
    ▼
[CircuitBreaker.apply_multiplier()]   ← scales size down based on drawdown level
    │
    ▼
[ExecutionEnforcer.approve_order()]   ← re-validates adjusted proposal, logs rejections
    │
    ▼
  Broker
```

**`SentinelCore` (`core.py`)**
- Tracks all open positions, account equity, session peak equity, and daily P&L
- Computes real-time drawdown; triggers L3 disconnect at 10% and publishes CRITICAL event
- `is_trading_allowed()` gates on L3 state, news lockout, Friday lockout, and market hours
- `get_open_positions()` exposes position list to `SentinelMonitor`

**`TradeValidator` (`validator.py`)**
- Validates every `TradeProposal` against 7 hardcoded rules (all must pass):
  1. Trading allowed (no L3 / lockout / closed market)
  2. Stop-loss present and positive
  3. Stop-loss correctly placed for direction (LONG below entry, SHORT above)
  4. Risk:reward ≥ 1.5:1
  5. Simultaneous open positions < 3
  6. Position size ≤ 2% of account
  7. Total exposure ≤ 6% (3 × 2%)
  8. Symbol is XAUUSD

**`PositionSizer` (`position_sizer.py`)**
- Quarter-Kelly sizing: `full_kelly × 0.25`, hard-capped at 2%
- `compute_size_pct(win_rate, avg_win, avg_loss, confidence)` — scales by signal confidence
- `pct_to_lots()` converts account percentage to broker lot size
- `validate_size()` checks SENTINEL limits before sizing is committed

**`CircuitBreaker` (`circuit_breaker.py`)**
- Monitors equity drawdown in real time with 3 escalating responses:

  | Level | Drawdown Threshold | Response | Size Multiplier |
  |-------|--------------------|----------|----------------|
  | NORMAL | < 5% | No action | 1.00× |
  | L1 | ≥ 5% | Reduce size, HIGH alert | 0.50× |
  | L2 | ≥ 7.5% | Reduce size further, HIGH alert | 0.25× |
  | L3 | ≥ 10% | Full halt, close all, CRITICAL kill signal | 0.00× |

- L1 auto-resets to NORMAL when drawdown recovers below 5%
- `apply_multiplier(proposed_size_pct)` clamps output to `[0.0, SENTINEL.max_position_pct]`
- Full trigger history kept; last 10 events returned in `get_summary()`

**`SentinelMonitor` (`monitor.py`)**
- Async background task polling every 100ms via `asyncio.create_task`
- Checks every open position against current price per direction:
  - LONG: breach if `price ≤ stop_loss`
  - SHORT: breach if `price ≥ stop_loss`
- Publishes CRITICAL `SL_BREACH` event with position ID, entry, SL, and current price on each hit
- `_check_friday_close()` stubbed for Phase 5 (force-close 30 min before Friday 21:00 UTC)

**`ExecutionEnforcer` (`execution/enforcer.py`)**
- Final gate before the broker: applies CB multiplier → re-runs full `TradeValidator`
- Returns `(approved: bool, reason: str, final_size_pct: float)`
- Maintains rolling rejection log (`deque(maxlen=1000)`) with UTC timestamp and proposing module name
- `get_rejection_summary()` — approved count, rejected count, rejection rate, last 10 rejections

---

### Build Phases

| # | Phase | Status |
|---|-------|--------|
| 1 | Data Foundation (DATA module) | **Done** |
| 2 | Risk Layer (SENTINEL runtime enforcement) | **Done** |
| 3 | Backtesting Engine | Planned |
| 4 | HYDRA v1 (TFT only) | Planned |
| 5 | Paper Trading | Planned |
| 6 | TUI v1 | Planned |
| 7 | HYDRA Full Ensemble | Planned |
| 8 | PROMETHEUS v1 (NEAT only) | Planned |
| 9 | Money Makers (VENOM, REAPER, APEX, WRAITH) | Planned |
| 10 | ARES Integration | Planned |
| 11 | Full PROMETHEUS | Planned |
| 12 | Flow Intelligence (PHANTOM, SPECTER, NEXUS) | Planned |
| 13 | Macro Intelligence (ATLAS, ARGUS, HERALD, ORACLE) | Planned |
| 14 | Advanced ML (KRONOS, ECHO, FORGE, SHADOW) | Planned |
| 15 | NEMESIS War Simulator | Planned |
| 16 | Full System Optimization | Planned |

---

## Risk Controls (SENTINEL)

SENTINEL enforces hardcoded, immutable limits via a frozen dataclass — any attempted modification raises `AttributeError` at runtime. No module, including the evolution engine, can override these.

| Rule | Limit |
|------|-------|
| Max position size | 2% of account per trade |
| Max simultaneous positions | 3 |
| Max total exposure | 6% of account (3 × 2%) |
| Stop-loss | Mandatory on every trade |
| Min risk:reward | 1.5:1 |
| News lockout (pre) | 5 min before high-impact events |
| News lockout (post) | 2 min after high-impact events |
| Friday close lockout | 30 min before 21:00 UTC |
| L3 circuit breaker | 10% daily equity drawdown → full halt |
| Kelly fraction | Quarter-Kelly (0.25), hard cap 2% |

---

## Leverage

Dynamic leverage from 1× to 100× based on the number of confluence conditions met:

| Conditions Met | Leverage Range |
|----------------|----------------|
| 7 (all) | 50× – 100× |
| 5–6 | 20× – 50× |
| 3–4 | 5× – 20× |
| 0–2 | 1× – 2× |

---

## Setup

### Requirements

- Python 3.11+
- MetaTrader5 terminal (for live/paper trading — optional for development)

### Installation

```bash
# Clone the repository
git clone https://github.com/MatinDeevv/Aphelion-Reasearch.git
cd Aphelion-Reasearch

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install core dependencies
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"

# Or install specific extras
pip install -e ".[ml]"          # PyTorch, XGBoost, LightGBM, Optuna
pip install -e ".[broker]"      # MetaTrader5
pip install -e ".[tui]"         # Textual terminal UI
pip install -e ".[dev]"         # pytest, pytest-asyncio
```

### Running Tests

```bash
pytest tests/ -v
```

Expected: **187 tests passing** (144 unit + 42 SENTINEL integration xpassed + 1 expected xfail).

---

## Technical Details

### Event Bus

Async pub/sub messaging with priority dispatch. SENTINEL CRITICAL events are never dropped — if the queue is full they are force-inserted.

| Priority | Value | Used by |
|----------|-------|---------|
| CRITICAL | 0 | SENTINEL kill switches, SL breach alerts, L3 disconnect |
| HIGH | 1 | Circuit breaker L1/L2 alerts, trade signals, position updates |
| NORMAL | 2 | Bar completions, feature updates |
| LOW | 3 | Health checks, diagnostics, logging |

### Feature Pipeline

Data processes at two frequencies:

- **Per-tick (< 1ms target):** Microstructure — VPIN, OFI, tick entropy, Hawkes intensity, spread dynamics, quote depth
- **Per-bar:** Full feature set — market structure, volume profile, VWAP, technical indicators (ATR, Bollinger Bands, RSI, EMA), session features, MTF alignment, cointegration (H1 only for performance)

### Data Layer

- Connects to MetaTrader5 via the Python API; graceful fallback when MT5 is unavailable
- Aggregates raw ticks into OHLCV bars across 4 timeframes (M1, M5, M15, H1)
- Validates every tick and bar: rejects negative prices, crossed spreads, spreads > 50 pips, single-tick moves > 5%
- `load_from_parquet(filepath, timeframe)` and `load_from_csv(filepath, timeframe)` — offline data loading for the Phase 3 backtester; validate required columns, return DataFrames without touching live bar state

### SENTINEL Runtime

All trading activity passes through the SENTINEL enforcement pipeline in sequence:

1. `SentinelCore.is_trading_allowed()` — fast pre-check (L3 / lockout / market hours)
2. `TradeValidator.validate()` — full 7-rule immutable evaluation
3. `CircuitBreaker.apply_multiplier()` — scales position size to current drawdown level
4. `ExecutionEnforcer.approve_order()` — final validation of size-adjusted proposal
5. `SentinelMonitor` — continuous 100ms background loop watching stop-loss levels for all open positions

---

## License

Proprietary. All rights reserved.
