# APHELION Paper Trading Setup Guide

## Overview

The APHELION paper trading system simulates live trading of XAU/USD (Gold) using real-time MT5 data or synthetic feeds, with full SENTINEL risk management, HYDRA signal generation, and Bloomberg-grade TUI dashboard.

### Architecture

```
MT5TickFeed / SimulatedFeed / ReplayFeed
        │
        ▼
   EventBus (tick / bar events)
        │
   ┌────┴────┐
   ▼         ▼
DataLayer  TUIBridge ──► TUI Dashboard
   │
   ▼
FeatureEngine (60+ features)
   │
   ▼
HydraInference (Ensemble TFT)
   │
   ▼
HydraStrategy ──► Order generation
   │
   ▼
SENTINEL Stack (L1/L2/L3 circuit breakers)
   │
   ▼
PaperExecutor ──► PaperLedger (JSON-Lines audit trail)
```

## Prerequisites

### Required
- Python 3.11+ (3.14 tested)
- APHELION installed in editable mode: `pip install -e .`
- PyTorch (for HYDRA inference)

### Optional (for live MT5 data)
- Windows OS
- MetaTrader 5 terminal installed and running  
- MT5 Python package: `pip install MetaTrader5`
- Broker account (e.g. Eightcap demo account)

## Quick Start

### 1. Simulated Mode (No MT5 Required)

```bash
python run_paper.py
```

This runs 500 synthetic random-walk bars through the full pipeline. No external dependencies needed.

### 2. Live MT5 Tick Feed

```bash
python run_paper.py \
    --mode mt5_tick \
    --mt5-login 12345678 \
    --mt5-password "your_password" \
    --mt5-server "Eightcap-Demo" \
    --capital 10000
```

### 3. With HYDRA Model

```bash
python run_paper.py \
    --mode mt5_tick \
    --hydra-checkpoint models/hydra/hydra_ensemble_best_sharpe.pt \
    --capital 25000
```

### 4. Headless (No TUI)

```bash
python run_paper.py --no-tui --verbose
```

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `simulated` | Feed mode: `simulated`, `mt5_tick`, `live`, `replay` |
| `--capital` | `10000` | Starting capital (USD) |
| `--symbol` | `XAUUSD` | Trading symbol |
| `--hydra-checkpoint` | (none) | Path to HYDRA `.pt` checkpoint |
| `--mt5-login` | `0` | MT5 account number |
| `--mt5-password` | (none) | MT5 account password |
| `--mt5-server` | (none) | MT5 broker server name |
| `--mt5-terminal` | (auto) | Path to `terminal64.exe` |
| `--poll-ms` | `100` | Tick poll interval (milliseconds) |
| `--warmup` | `200` | Historical bars to pre-load |
| `--sim-bars` | `500` | Simulated bars (0 = infinite) |
| `--no-tui` | `false` | Disable TUI dashboard |
| `--verbose` / `-v` | `false` | Enable DEBUG-level logging |

## Data Feed Modes

### `simulated` (Default)
Random-walk price generator. Useful for:
- Testing the pipeline without MT5
- CI/CD automated tests
- Development iteration

### `mt5_tick`
Production tick-level streaming from MetaTrader 5:
- Polls `MT5Connection.get_last_tick()` every 100ms
- Aggregates ticks into M1/M5/M15/H1 bars via `BarAggregator`
- Automatic reconnection (up to 10 attempts)
- Warmup bar pre-loading
- Duplicate tick filtering
- Stale data alerting (30s threshold)

### `live`
Bar-level polling from MT5 (simpler, lower throughput):
- Polls `MT5Connection.get_bars()` every 5s
- Yields only new bars (deduplication by timestamp)

### `replay`
Replays a pre-loaded list of `Bar` objects:
- Optional real-time pacing with speed multiplier
- Useful for backtesting with paper executor

## SENTINEL Risk Management

All paper trades pass through the full SENTINEL stack:

| Layer | Trigger | Action |
|-------|---------|--------|
| **L1** | 3% daily drawdown | Warning — reduce position sizing 50% |
| **L2** | 6% daily drawdown | Halt — no new trades, close-only mode |
| **L3** | 10% daily drawdown | Emergency disconnect — close all positions |

Additional protections:
- Max 2% of account per trade
- Max 3 simultaneous positions
- Mandatory stop-loss on every trade
- Min 1.5:1 risk-reward ratio
- Friday close lockout (30 min before market close)
- Pre/post high-impact news lockout

## TUI Dashboard

The Bloomberg-grade TUI shows:
- **Price ticker**: Live bid/ask with sparkline
- **HYDRA signals**: Direction, confidence, horizon agreement
- **SENTINEL status**: Circuit breaker levels, exposure, drawdown
- **Equity curve**: P&L tracking with sparkline
- **Positions table**: Open positions with SL/TP/PnL
- **Event log**: Fills, rejections, system events
- **System stats**: CPU, memory, latency, uptime

Keyboard shortcuts: F1-F5 for view switching.

## Output Files

| Path | Description |
|------|-------------|
| `logs/paper/paper_{session_id}.jsonl` | Audit ledger (JSON-Lines) |
| `logs/paper_run.log` | Runtime log |

## Programmatic Usage

```python
import asyncio
from aphelion.paper.runner import PaperRunner, PaperRunnerConfig
from aphelion.paper.feed import FeedMode, SimulatedFeedConfig
from aphelion.paper.session import PaperSessionConfig

config = PaperRunnerConfig(
    feed_mode=FeedMode.SIMULATED,
    session_config=PaperSessionConfig(
        initial_capital=10_000.0,
        warmup_bars=64,
    ),
    sim_config=SimulatedFeedConfig(max_bars=1000),
    enable_tui=False,
)

async def main():
    runner = PaperRunner(config)
    result = await runner.run()
    print(result.summary())

asyncio.run(main())
```

## Testing

```bash
# Run paper trading tests only
pytest tests/paper/ -v

# Run full suite
pytest tests/ -v
```

Test files:
- `tests/paper/test_paper.py` — PaperExecutor, feeds, ledger, session (39 tests)
- `tests/paper/test_feed.py` — MT5TickFeed, FeedConfig, FeedStats (16 tests)
- `tests/paper/test_runner.py` — PaperRunner end-to-end (10 tests)

## Troubleshooting

### "MetaTrader5 package not installed"
```bash
pip install MetaTrader5
```
Only available on Windows.

### "MT5 connection failed"
1. Ensure MT5 terminal is running
2. Check login credentials
3. Verify the server name matches your broker
4. Try increasing `--poll-ms` to reduce load

### "No HYDRA checkpoint — signals disabled"
Paper session runs without generating trades. To enable:
```bash
python run_paper.py --hydra-checkpoint models/hydra/hydra_ensemble_best_sharpe.pt
```

### Stale tick warnings
If no tick arrives for 30s, the feed emits a stale data alert. This usually means:
- Market is closed (weekends, holidays)
- MT5 terminal lost connection
- Symbol is not subscribed in MT5
