# APHELION

Fully autonomous XAU/USD (Gold) trading system built on a 20-module architecture with tiered governance, hardcoded risk limits, and 60+ engineered features.

## Architecture

APHELION uses a 5-tier governance hierarchy where modules vote on trade decisions with weighted authority:

| Tier | Role | Votes | Modules |
|------|------|-------|---------|
| **Sovereign** | System Core | - | Event Bus, Clock, Registry |
| **Council** | Strategic Decision | 100 | OLYMPUS, SENTINEL, ARES |
| **Minister** | Intelligence | 40 | HYDRA, PROMETHEUS, PHANTOM, NEMESIS, FORGE, ATLAS, DATA |
| **Commander** | Execution | 10 | BACKTEST, VENOM, REAPER, APEX, WRAITH, SHADOW, KRONOS, ECHO, CASSANDRA, ORACLE, TITAN, GHOST |
| **Operator** | Support | 1 | FUND |

### Module Overview

```
OLYMPUS      System Governor & Auto-Tuner
SENTINEL     Supreme Risk Authority (hardcoded, immutable)
ARES         LLM Brain (Mixtral 8x7B -> GPT-OSS-120B)
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
```

## Project Structure

```
aphelion/
  core/             Event bus, config, data layer, clock, registry
  features/         60+ feature engine (microstructure, market structure, volume, VWAP, sessions, MTF, cointegration)
  intelligence/     HYDRA, KRONOS, ECHO, FORGE, SHADOW
  evolution/        PROMETHEUS, CIPHER, MERIDIAN, ZEUS
  nemesis/          PANDORA, LEVIATHAN, CHRONOS, VERDICT
  macro/            ORACLE, ATLAS, NEXUS, ARGUS, HERALD
  flow/             PHANTOM, SPECTER
  risk/             SENTINEL, TITAN
  money/            VENOM, REAPER, APEX, WRAITH, GHOST
  ares/             LLM brain, sentiment, reasoning, strategist
  governance/       OLYMPUS, COUNCIL
  backtest/         Engine, metrics, Monte Carlo
  aphelion_model/   APHELION-120B fine-tune pipeline
  tui/              Terminal UI (12 screens)
tests/
  core/             Config, clock, event bus, data layer, registry tests
  features/         Microstructure, market structure, volume profile, VWAP, MTF, cointegration tests
  integration/      End-to-end pipeline tests
```

## Current Status

**Phase 1: Data Foundation** -- Complete

- Core infrastructure: async event bus, market clock, component registry, MT5 data layer
- 60+ engineered features across 7 sub-engines:
  - **Microstructure** -- VPIN, OFI, tick entropy, Hawkes intensity, micro-price divergence, spread dynamics, quote depth
  - **Market Structure** -- Order blocks, fair value gaps, swing detection, liquidity pools, breaker blocks, volume imbalances, change of character (CHoCH)
  - **Volume Profile** -- Volume delta, cumulative delta (CVD), point of control (POC), value area high/low (VAH/VAL), absorption detection
  - **VWAP** -- Session VWAP, anchored VWAP, rolling VWAP with +/-1 and +/-2 sigma bands
  - **Sessions** -- Session flags, time-to-open/close, news proximity, calendar markers
  - **Multi-Timeframe** -- Alignment scoring across M1, M5, M15, H1 with weighted consensus
  - **Cointegration** -- Engle-Granger tests for XAU vs DXY, real yields, silver; spread z-scores
- 127 unit tests passing

**Phase 1 fixes (pre-Phase 2 hardening):**

- **Cointegration ADF test** (`aphelion/features/cointegration.py`) — replaced hand-rolled `_adf_test_simple()` (hardcoded MacKinnon critical values, bucketed p-values) with `statsmodels.tsa.stattools.adfuller`. Real MacKinnon p-values are now returned via `result[1]`.
- **Clock session features** (`aphelion/core/clock.py`) — added `"is_trading_session"` key to the dict returned by `session_features()`. The method already existed; it was simply missing from the output.
- **BACKTEST module registration** (`aphelion/core/config.py`) — added `BACKTEST` (Tier.COMMANDER, "Backtesting Engine & Monte Carlo") to the `MODULES` dict, placed alphabetically among Commanders. Required by Phase 3.
- **DataLayer file loaders** (`aphelion/core/data_layer.py`) — added `load_from_parquet(filepath, timeframe)` and `load_from_csv(filepath, timeframe)`. Both validate required columns (`timestamp`, `open`, `high`, `low`, `close`, `volume`, `tick_volume`, `spread`), raise `ValueError` on missing columns, and return a `DataFrame` without touching `self._bars`. The CSV loader parses `timestamp` as UTC-aware datetime. These methods are the entry points for the Phase 3 backtester.

### Build Phases

| # | Phase | Status |
|---|-------|--------|
| 1 | Data Foundation (DATA module) | Done |
| 2 | Risk Layer (SENTINEL core rules) | Planned |
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

## Risk Controls (SENTINEL)

SENTINEL enforces hardcoded, immutable limits that cannot be overridden by any module, including the evolution engine:

- **Max position size:** 2% of account per trade
- **Max simultaneous positions:** 3
- **Stop-loss:** Mandatory on every trade
- **Min risk-reward:** 1.5:1
- **News lockout:** 5 min pre / 2 min post high-impact events
- **Friday close:** 30 min lockout before market close
- **Drawdown circuit breaker:** 10% daily equity drawdown triggers L3 disconnect
- **Kelly fraction:** Quarter-Kelly (0.25) with 2% hard cap

## Leverage

Dynamic leverage from 1x to 100x based on confluence conditions:

| Conditions Met | Leverage Range |
|----------------|----------------|
| 7 (all) | 50x - 100x |
| 5-6 | 20x - 50x |
| 3-4 | 5x - 20x |
| 0-2 | 1x - 2x |

## Setup

### Requirements

- Python 3.11+
- MetaTrader5 terminal (for live/paper trading, optional for development)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/aphelion.git
cd aphelion

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
pip install -e ".[broker]"     # MetaTrader5
pip install -e ".[tui]"        # Textual terminal UI
pip install -e ".[dev]"        # pytest, pytest-asyncio
```

### Running Tests

```bash
pytest tests/ -v
```

## Technical Details

### Feature Pipeline

The feature engine processes data at two frequencies:

- **Per-tick:** Microstructure features (VPIN, OFI, entropy, Hawkes, spread dynamics, quote depth)
- **Per-bar:** Full feature computation including market structure, volume profile, VWAP, technical indicators (ATR, Bollinger Bands, RSI, EMA), session features, MTF alignment, and cointegration (H1 only)

### Event Bus

Async pub/sub messaging with priority dispatch:

- **CRITICAL (0):** SENTINEL risk events -- always dispatched, even when queue is full
- **HIGH (1):** Trade signals, position updates
- **NORMAL (2):** Bar completions, feature updates
- **LOW (3):** Logging, diagnostics

### Data Layer

- Connects to MetaTrader5 via the Python API
- Aggregates raw ticks into OHLCV bars across 4 timeframes (M1, M5, M15, H1)
- Validates data quality: rejects negative prices, crossed spreads, >50 pip spreads, >5% price jumps
- Graceful fallback when MT5 is unavailable (development mode)
- `load_from_parquet(filepath, timeframe)` and `load_from_csv(filepath, timeframe)` for offline/backtest data loading (Phase 3)

## License

Proprietary. All rights reserved.
