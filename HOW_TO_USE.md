# APHELION — How to Use

Quick-start guide for setting up, running, and extending the APHELION autonomous XAU/USD trading system.

---

## 1. Installation

```bash
# Clone
git clone https://github.com/MatinDeevv/Aphelion-Reasearch.git
cd Aphelion-Reasearch

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Optional: GPU Support (for HYDRA neural training)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Optional: MetaTrader 5 (for live/paper trading)

```bash
pip install MetaTrader5
```

---

## 2. Project Structure at a Glance

```
aphelion/
├── core/          # Event bus, clock, config, data layer, registry
├── features/      # Feature engineering pipeline (microstructure, VWAP, MTF...)
├── risk/sentinel/ # Risk enforcement (hard limits, circuit breakers, position sizing)
├── backtest/      # Backtesting engine, broker sim, portfolio, metrics
├── intelligence/
│   └── hydra/     # Neural ensemble (TFT + LSTM + CNN + MoE)
├── evolution/
│   └── prometheus/ # NEAT evolutionary strategy optimisation
├── money/         # Position sizing, capital allocation, risk budgeting
├── ares/          # LLM brain & strategy coordinator
├── paper/         # Paper trading (MT5 tick feed, runner, ledger)
└── tui/           # Bloomberg-style terminal dashboard
```

---

## 3. Running Tests

```bash
# Full suite
pytest tests/ -v

# Quick run (no verbose)
pytest tests/ -q

# Specific module
pytest tests/backtest/ -v
pytest tests/intelligence/ -v

# Single test file
pytest tests/backtest/test_metrics.py -v
```

---

## 4. Training HYDRA (Neural Ensemble)

HYDRA is the neural intelligence core with 4 sub-models: TFT, LSTM, CNN, MoE.

```bash
# Train with default settings
python scripts/train_hydra.py

# Custom training
python -c "
from aphelion.intelligence.hydra import *
import torch

# Configure
cfg = EnsembleConfig(
    tft_config=TFTConfig(hidden_dim=64, lstm_layers=2, attention_heads=4),
    lstm_config=LSTMConfig(hidden_size=64, num_layers=2),
    cnn_config=CNNConfig(hidden_size=64),
    moe_config=MoEConfig(hidden_size=64),
)

# Build dataset from your feature dicts
ds_cfg = DatasetConfig(lookback_bars=64, batch_size=32)
train_ds, val_ds, test_ds, means, stds = build_dataset_from_feature_dicts(
    feature_dicts, close_prices, ds_cfg
)

# Train
trainer_cfg = TrainerConfig(max_epochs=50, ensemble_config=cfg)
trainer = HydraTrainer(trainer_cfg)
result = trainer.train(train_loader, val_loader)
print(f'Best Sharpe: {result[\"best_val_sharpe\"]:.3f}')
"
```

### Using HYDRA for Inference

```python
from aphelion.intelligence.hydra import HydraInference

inf = HydraInference()
inf.load_checkpoint("models/hydra/hydra_ensemble_best_sharpe.pt")

# Process bars one at a time
signal = inf.process_bar(feature_dict)
if signal and signal.is_actionable:
    print(f"Direction: {signal.direction}, Confidence: {signal.confidence:.2f}")
    print(f"Gate weights: {signal.gate_weights}")
```

---

## 5. Running a Backtest

```python
from aphelion.backtest.engine import BacktestEngine, BacktestConfig
from aphelion.backtest.metrics import compute_metrics
from aphelion.core.data_layer import DataLayer

# Setup
config = BacktestConfig(
    initial_capital=10_000,
    risk_per_trade=0.02,
    warmup_bars=50,
)

data_layer = DataLayer()
sentinel_stack = data_layer.build_sentinel_stack(config.initial_capital)
engine = BacktestEngine(config, sentinel_stack, data_layer)

# Set your strategy (any callable: bar, features, portfolio -> list[Order])
engine.set_strategy(my_strategy)

# Run
results = engine.run(bars)
metrics = compute_metrics(results.trades, results.daily_returns)
print(f"Sharpe: {metrics.sharpe:.2f}, Trades: {metrics.total_trades}")
```

---

## 6. Evolutionary Strategy Optimisation (PROMETHEUS)

PROMETHEUS uses NEAT to evolve trading strategy parameters:

```python
from aphelion.evolution.prometheus import (
    PrometheusEngine, EvolutionConfig, NEATConfig,
    evaluate_genome,
)

# Configure evolution
evo_config = EvolutionConfig(
    neat=NEATConfig(population_size=50, elite_count=5),
    max_generations=100,
    target_sharpe=1.5,
)

# Create engine
engine = PrometheusEngine(evo_config)

# Set evaluator (runs backtest for each genome)
engine.set_evaluator(lambda genome: evaluate_genome(
    genome, historical_bars, engine_factory
))

# Run evolution
result = engine.run()

best = result["best_genome"]
print(f"Best genome: {best.genome_id}")
print(f"Sharpe: {best.fitness.sharpe:.2f}")
print(f"Win rate: {best.fitness.win_rate:.1%}")
print(f"Strategy config: {best.to_strategy_config()}")
```

### Genome Files

Evolved genomes are saved to `genomes/`:
- `genomes/live/` — Currently active genomes
- `genomes/hall_of_fame/` — All-time best performers
- `genomes/archive/` — Historical genomes
- `genomes/paper/` — Paper-trading candidates

---

## 7. Money Management (Capital Allocation)

```python
from aphelion.money import (
    PositionManager, PositionManagerConfig, SizingMethod,
    CapitalAllocator, CapitalAllocatorConfig, AllocationMethod, StrategySlot,
    RiskBudget, RiskBudgetConfig,
)

# Position sizing
pm = PositionManager(PositionManagerConfig(method=SizingMethod.KELLY))
size = pm.compute_size(
    equity=10_000,
    signal_confidence=0.75,
    atr=5.0,
    entry_price=2850,
    sl_distance=10.0,
)
print(f"Size: {size.size_lots} lots ({size.size_pct:.1%} risk)")

# Capital allocation across strategies
allocator = CapitalAllocator(
    total_equity=50_000,
    config=CapitalAllocatorConfig(method=AllocationMethod.PERFORMANCE_WEIGHTED),
)
allocator.register_strategy(StrategySlot(strategy_id="HYDRA", name="Neural"))
allocator.register_strategy(StrategySlot(strategy_id="EVO", name="Evolved"))
weights = allocator.rebalance()

# Risk budgeting
rb = RiskBudget(initial_equity=50_000)
rb.register_strategy("HYDRA")
allowed, reason = rb.can_trade("HYDRA", risk_amount=500)
```

---

## 8. ARES Strategy Coordination

ARES aggregates signals from all strategies and makes final trade decisions:

```python
from aphelion.ares import (
    AresCoordinator, AresConfig, StrategyVote, SignalSource,
    AresReasoner, ReasonerConfig, LLMProvider,
    DecisionJournal,
)
from aphelion.core.config import Tier

# Create coordinator with LLM reasoner
reasoner = AresReasoner(ReasonerConfig(provider=LLMProvider.MOCK))
coordinator = AresCoordinator(AresConfig(), reasoner=reasoner)

# Collect votes from strategies
votes = [
    StrategyVote(source=SignalSource.HYDRA, direction=1, confidence=0.80, tier=Tier.MINISTER),
    StrategyVote(source=SignalSource.PROMETHEUS, direction=1, confidence=0.65, tier=Tier.MINISTER),
    StrategyVote(source=SignalSource.APEX, direction=-1, confidence=0.40, tier=Tier.COMMANDER),
]

# Aggregate
signal = coordinator.aggregate(votes)
print(f"Direction: {signal.direction}, Confidence: {signal.confidence:.2f}")
print(f"Reasoning: {signal.reasoning}")

# Decision journal for audit
journal = DecisionJournal("logs/ares_decisions.jsonl")
journal.record(
    direction=signal.direction,
    consensus_score=signal.consensus_score,
    confidence=signal.confidence,
    agreement_ratio=signal.agreement_ratio,
    reasoning=signal.reasoning,
)
```

---

## 9. Paper Trading

Connect to MetaTrader 5 for live paper trading:

```bash
# Quick start
python run_paper.py

# Or run the all-in-one demo
python scripts/runall.py
```

```python
from aphelion.paper.feed import MT5TickFeed
from aphelion.paper.runner import PaperRunner

# The runner manages the full paper trading loop:
# MT5 feed → features → HYDRA → SENTINEL → execution → portfolio → TUI
runner = PaperRunner()
await runner.run()
```

---

## 10. TUI Dashboard

The Bloomberg-style terminal dashboard shows real-time trading state:

```bash
python -m aphelion.tui.app
```

Displays: equity curve, open positions, signal heatmap, SENTINEL status,
feature panels, and trade log.

---

## 11. Configuration

All system constants live in `aphelion/core/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `SENTINEL.max_position_pct` | 2% | Max risk per trade |
| `SENTINEL.max_simultaneous_positions` | 3 | Max open positions |
| `SENTINEL.min_risk_reward` | 1.5:1 | Minimum R:R ratio |
| `SENTINEL.daily_equity_drawdown_l1` | 3% | L1 warning (reduce size) |
| `SENTINEL.daily_equity_drawdown_l2` | 6% | L2 halt (no new trades) |
| `SENTINEL.daily_equity_drawdown_l3` | 10% | L3 disconnect (close all) |
| `KELLY_FRACTION` | 0.25 | Quarter-Kelly safety |

> **SENTINEL limits are immutable** — never bypass them in any execution path.

---

## 12. Environment Variables

Create a `.env` file in the project root:

```env
# MetaTrader 5
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server

# ARES LLM (optional)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Training
HYDRA_DEVICE=cuda   # or 'cpu'
HYDRA_EPOCHS=50
```

---

## 13. Development Workflow

```bash
# Run tests before committing
pytest tests/ -v

# Commit format
git commit -m "feat: add new strategy parameter"
git commit -m "fix: correct Kelly sizing edge case"
git commit -m "test: add PROMETHEUS evolution tests"

# Branch naming
git checkout -b feature/prometheus-neat
git checkout -b fix/kelly-sizing
```

---

## Quick Reference: Module Tiers

| Tier | Weight | Modules |
|------|--------|---------|
| Sovereign | ∞ | User override |
| Council | 100 | OLYMPUS, SENTINEL, ARES |
| Minister | 40 | HYDRA, PROMETHEUS, PHANTOM, NEMESIS, FORGE, ATLAS, DATA |
| Commander | 10 | BACKTEST, VENOM, REAPER, APEX, WRAITH, SHADOW, KRONOS, ECHO... |
| Operator | 1 | Sub-modules, utilities |
