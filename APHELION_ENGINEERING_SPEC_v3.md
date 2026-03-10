# APHELION AUTONOMOUS TRADING SYSTEM
## Engineering Specification v3.0 — Full System Architecture
### Revision Date: March 2026 | Classification: CONFIDENTIAL

---

> *"Every other module sees the market. SOLA sees APHELION."*

---

## TABLE OF CONTENTS

1. [System Overview](#1-system-overview)
2. [Architecture Topology](#2-architecture-topology)
3. [Tier Hierarchy & Vote Weights](#3-tier-hierarchy--vote-weights)
4. [Phase 1 — Data Foundation v2](#4-phase-1--data-foundation-v2)
5. [Phase 2 — SENTINEL v2](#5-phase-2--sentinel-v2)
6. [Phase 3 — Backtesting Engine v2](#6-phase-3--backtesting-engine-v2)
7. [Phase 4 — HYDRA v2](#7-phase-4--hydra-v2)
8. [Phase 5 — Paper Trading v2](#8-phase-5--paper-trading-v2)
9. [Phase 6 — TUI v2](#9-phase-6--tui-v2)
10. [Phase 7 — HYDRA Full Ensemble v2](#10-phase-7--hydra-full-ensemble-v2)
11. [Phase 8 — PROMETHEUS/NEAT v2](#11-phase-8--prometheusneat-v2)
12. [Phase 9 — Money Management v2](#12-phase-9--money-management-v2)
13. [Phase 10 — ARES v2](#13-phase-10--ares-v2)
14. [Phase 11 — FLOW v2](#14-phase-11--flow-v2)
15. [Phase 12 — MACRO v2](#15-phase-12--macro-v2)
16. [Phase 13 — KRONOS/ECHO/FORGE/SHADOW](#16-phase-13--kronosechoforgeshadow)
17. [Phase 14 — NEMESIS v2](#17-phase-14--nemesis-v2)
18. [Phase 15 — TITAN / Quality Gate v2](#18-phase-15--titan--quality-gate-v2)
19. [Phase 16 — CIPHER / MERIDIAN / Auto-Optimizer](#19-phase-16--cipher--meridian--auto-optimizer)
20. [Phase 17 — OMEGA](#20-phase-17--omega)
21. [Phase 18 — SIGNAL TOWER](#21-phase-18--signal-tower)
22. [Phase 19 — ATLAS LIVE](#22-phase-19--atlas-live)
23. [Phase 20 — OLYMPUS](#23-phase-20--olympus)
24. [Phase 21 — SOLA](#24-phase-21--sola)
25. [Data Pipeline & Training Protocol](#25-data-pipeline--training-protocol)
26. [Deployment Architecture](#26-deployment-architecture)
27. [Performance Targets](#27-performance-targets)
28. [Licensing & Repository Strategy](#28-licensing--repository-strategy)

---

## 1. SYSTEM OVERVIEW

APHELION is a fully autonomous, self-evolving intraday trading system targeting XAU/USD (spot gold). It operates across two parallel strategies — ALPHA (M1 scalping) and OMEGA (H1/H4 swing) — coordinated by a master orchestrator (OLYMPUS) and governed by a sovereign intelligence layer (SOLA).

### Core Thesis

Markets are not random. They are episodically predictable. APHELION exploits that predictability by:

1. Running a deep ensemble of uncorrelated ML models (HYDRA) for high-probability signal generation
2. Evolving its own strategy parameters continuously via neuroevolution (PROMETHEUS)
3. Voting on every trade decision through an independent multi-module council (ARES)
4. Adapting in real-time to regime shifts, macro events, and performance decay (SOLA)
5. Protecting capital at all costs through a multi-layer risk architecture (SENTINEL)

### Target Instrument

| Property | Value |
|---|---|
| Instrument | XAU/USD (spot gold) |
| Broker | Eightcap MT5 |
| Account Size | $1,000 demo → $10,000 live (summer 2026) |
| Leverage | 500:1 (demo), phased up to 5:1 effective risk on live |
| Primary TF | M1 (ALPHA), H1/H4 (OMEGA) |
| Sessions | London + NY overlap (1300–1700 UTC) priority |

### Projected Performance (All 21 Phases Live)

| Scenario | Daily Return | Annual (252 days) |
|---|---|---|
| Conservative | 1.5% | ~$10k → ~$380k |
| Base Case | 2.5% | ~$10k → ~$2.6M |
| Optimistic | 4.0% | ~$10k → ~$11M |

*Mathematical ceiling: ~68–72% win rate on M1 timeframe (information-theoretic limit). Projected steady-state: 62–65% with 1.5:1 R:R (ALPHA) + 30% with 5:1 R:R (OMEGA).*

---

## 2. ARCHITECTURE TOPOLOGY

```
╔══════════════════════════════════════════════════════════════════════╗
║                          SOLA  (SOVEREIGN)                          ║
║              System Oracle & Learning Architect                     ║
║           VETO power · Edge decay · Black swan guard               ║
╚══════════════════════════╤═══════════════════════════════════════════╝
                           │
╔══════════════════════════▼═══════════════════════════════════════════╗
║                        OLYMPUS  (MASTER)                            ║
║          Capital Allocator · ALPHA/OMEGA Coordinator                ║
║          Performance Monitor · Retraining Trigger                   ║
╚═══════════╤══════════════════════════════════════╤════════════════════╝
            │                                      │
    ┌───────▼──────────┐                  ┌────────▼────────────┐
    │   ALPHA ENGINE   │                  │   OMEGA ENGINE      │
    │   M1 Scalping    │                  │   H1/H4 Swing       │
    │   60%+ WR        │                  │   30% WR, 5:1 RR    │
    └───────┬──────────┘                  └────────┬────────────┘
            │                                      │
╔═══════════▼══════════════════════════════════════▼════════════════════╗
║                         ARES  (COUNCIL)                              ║
║  HYDRA · PROMETHEUS · FLOW · MACRO · SIGNAL TOWER · NEMESIS · ATLAS ║
║  Weighted vote aggregation · Dynamic module silencing                ║
╚═════════════════════════════════════════════════════════════════════╤═╝
                                                                      │
╔═════════════════════════════════════════════════════════════════════▼═╗
║                       SENTINEL  (GUARDIAN)                           ║
║     Circuit Breaker · Position Enforcer · Risk Budget · DD Guard    ║
╚═════════════════════════════════════════════════════════════════════╤═╝
                                                                      │
╔═════════════════════════════════════════════════════════════════════▼═╗
║                  MT5 EXECUTION LAYER  (EIGHTCAP)                     ║
║            Paper → Demo → Live progression                           ║
╚═══════════════════════════════════════════════════════════════════════╝
```

### Data Flow (Single Bar)

```
MT5 Tick Feed
    → Feed Aggregator (M1 bar construction)
    → Feature Engine (50+ features per bar)
    → HYDRA Ensemble (ML signal, confidence)
    → SIGNAL TOWER (technical voters, independent)
    → FLOW (liquidity/microstructure)
    → MACRO (regime/DXY/sentiment)
    → ARES (vote aggregation → BUY/SELL/FLAT + confidence)
    → SENTINEL (risk checks, position sizing)
    → Execution Layer (MT5 order)
    → Journal (ARES + KRONOS logging)
    → ECHO (pattern library update)
    → SOLA (performance trace)
```

---

## 3. TIER HIERARCHY & VOTE WEIGHTS

```
SOVEREIGN   ∞ votes   SOLA, EventBus, SessionClock
            (can veto any decision, override SENTINEL)

GENERAL    20 votes   OLYMPUS
            (capital allocation, system-level decisions)

ORACLE     15 votes   HYDRA Ensemble
            (primary ML signal source)

COMMANDER  10 votes   PROMETHEUS, FLOW, MACRO, ATLAS LIVE
                      SIGNAL TOWER modules (each)
            (major independent intelligence sources)

LIEUTENANT  5 votes   NEMESIS (anti-regime), FORGE, SHADOW
            (secondary intelligence)

SERGEANT    2 votes   KRONOS, ECHO, MERIDIAN
            (logging, pattern, optimizer)

PRIVATE     1 vote    Individual sub-models within HYDRA
            (raw model signals before ensemble)
```

**Vote Aggregation Rules:**
- ARES collects all active module votes each bar
- Threshold for BUY/SELL: configurable (default 55% weighted majority)
- SOLA can dynamically reweight modules based on rolling 50-trade performance
- SENTINEL veto: regardless of vote outcome, no trade fires if risk limits breached
- SOLA veto: regardless of everything, SOLA can halt ALL trading system-wide

---

## 4. PHASE 1 — DATA FOUNDATION v2

**Status: ✅ BUILT — Upgrade Required**

### Current State
- Core tick ingestion, bar aggregation, and 50+ feature engine implemented
- 144 tests passing
- VWAPCalculator has known session-reset bug (deferred)

### v2 Upgrades

#### 4.1 Feature Engine Expansion

Add the following features to `aphelion/data/features/engine.py`:

**Microstructure Features (new)**
```python
# Bid-ask spread proxy from tick data
feature["spread_proxy"] = (bar.high - bar.low) / bar.close

# Tick velocity — number of ticks per second in bar window
feature["tick_velocity"] = bar.tick_count / bar.duration_seconds

# Volume anomaly score — z-score of volume vs 20-bar rolling mean
feature["volume_z"] = (bar.volume - rolling_vol_mean) / rolling_vol_std

# Price efficiency ratio (Elder)
feature["efficiency_ratio"] = abs(bar.close - bar_n.close) / sum(abs(diff) for diff in bar_range)
```

**HalfTrend Signal (new — Commander-tier ARES voter)**
```python
# HalfTrend: trend-following indicator with low noise
# Amplitude: 2 (default), ATR period: 100
def compute_halftrend(bars: List[Bar], amplitude: int = 2, atr_period: int = 100) -> Dict:
    atr = compute_atr(bars, atr_period)
    upper_band = max(bars[-amplitude:].high) - atr * 1.5
    lower_band = min(bars[-amplitude:].low) + atr * 1.5
    
    trend = "UP" if bars[-1].close > upper_band else "DOWN" if bars[-1].close < lower_band else "FLAT"
    
    return {
        "halftrend_signal": 1 if trend == "UP" else -1 if trend == "DOWN" else 0,
        "halftrend_upper": upper_band,
        "halftrend_lower": lower_band,
        "halftrend_strength": abs(bars[-1].close - (upper_band if trend == "UP" else lower_band)) / atr
    }
```

**Session Volatility Profile (new)**
```python
# Classify current bar into volatility regime
def session_volatility_regime(bar: Bar, session_history: List[Bar]) -> str:
    # "COMPRESSION" → ATR < 50th percentile of last 100 bars
    # "EXPANSION"   → ATR > 75th percentile
    # "SPIKE"       → ATR > 95th percentile (likely news)
    # "NORMAL"      → everything else
```

**Multi-timeframe Confluence (upgrade)**
```python
# Current: M1 only
# v2: M1 + M5 + M15 + H1 trend direction as features
# Adds 4 × 3 = 12 new features (direction, momentum, structure)
# Key: MTF confluence is one of the highest-signal features for XAU/USD
```

#### 4.2 VWAPCalculator Session Reset Fix

```python
class VWAPCalculator:
    def __init__(self):
        self._session_open: Optional[datetime] = None
        self._cumulative_pv: float = 0.0
        self._cumulative_vol: float = 0.0
    
    def update(self, bar: Bar) -> float:
        # BUGFIX: Reset on new trading session (00:00 UTC)
        session_date = bar.timestamp.date()
        if self._session_open is None or session_date != self._session_open:
            self._session_open = session_date
            self._cumulative_pv = 0.0
            self._cumulative_vol = 0.0
        
        self._cumulative_pv += bar.typical_price * bar.volume
        self._cumulative_vol += bar.volume
        return self._cumulative_pv / self._cumulative_vol if self._cumulative_vol > 0 else bar.close
```

#### 4.3 Feature Registry

Implement a `FeatureRegistry` that tracks:
- Feature name, version, data type, normalization method
- Whether feature is active in current build
- Rolling importance score (updated by SOLA)
- Correlation matrix to detect redundant features

```python
# aphelion/data/features/registry.py
@dataclass
class FeatureRecord:
    name: str
    version: str
    dtype: str  # "continuous", "categorical", "binary"
    normalizer: str  # "zscore", "minmax", "none"
    active: bool
    importance_score: float  # Updated by SOLA
    last_updated: datetime
```

#### 4.4 Test Coverage Targets (v2)

| Module | Current Tests | v2 Target |
|---|---|---|
| Feature engine | ~40 | 80 |
| VWAPCalculator | ~5 | 20 (including session reset) |
| MTF aggregator | 0 | 25 |
| HalfTrend | 0 | 15 |
| Feature registry | 0 | 10 |
| **Total Phase 1** | **~144** | **250+** |

---

## 5. PHASE 2 — SENTINEL v2

**Status: ✅ BUILT — Upgrade Required**

### Current State
- Circuit breaker, position enforcer, 6 core files
- 2% daily drawdown hard stop

### v2 Upgrades

#### 5.1 Five-Layer Risk Architecture

```
Layer 1: Pre-Trade Check       — Position size, margin, correlation
Layer 2: Intra-Trade Monitor   — Floating P&L, partial close triggers
Layer 3: Session Guard         — Max trades/session, max open positions
Layer 4: Daily Drawdown        — 2% hard stop (unchanged)
Layer 5: System Health Check   — Latency, MT5 connection, feed validity
```

#### 5.2 Dynamic Position Sizing

Replace static lot-size calculation with Kelly-informed dynamic sizing:

```python
def compute_position_size(
    account_balance: float,
    win_rate: float,          # rolling 50-trade win rate from SOLA
    avg_win_r: float,         # average win in R
    avg_loss_r: float,        # average loss in R (should be ~1.0)
    confidence: float,        # ARES output [0, 1]
    regime: str,              # "TRENDING", "RANGING", "VOLATILE"
    max_risk_pct: float = 0.02  # SENTINEL hard cap
) -> float:
    
    # Full Kelly
    kelly_pct = (win_rate * avg_win_r - (1 - win_rate) * avg_loss_r) / avg_win_r
    
    # Quarter-Kelly for safety
    quarter_kelly = kelly_pct * 0.25
    
    # Confidence scaling: reduce size when ARES confidence is low
    confidence_scaled = quarter_kelly * confidence
    
    # Regime scaling: reduce 30% in volatile regime
    regime_multipliers = {"TRENDING": 1.0, "RANGING": 0.8, "VOLATILE": 0.7}
    regime_scaled = confidence_scaled * regime_multipliers.get(regime, 0.8)
    
    # Hard cap: never exceed 2% risk
    final_pct = min(regime_scaled, max_risk_pct)
    
    return account_balance * final_pct
```

#### 5.3 Correlation Guard

Before placing a new trade, SENTINEL checks:
- Is there already an open position with >0.7 correlation to this signal source?
- If yes: reduce size by 50%, do not open new position if already at 3 correlated trades

#### 5.4 Latency Monitor

```python
class LatencyMonitor:
    MAX_ACCEPTABLE_MS = 150  # MT5 round-trip latency threshold
    
    def check(self, last_ping_ms: float) -> SentinelVerdict:
        if last_ping_ms > self.MAX_ACCEPTABLE_MS:
            return SentinelVerdict.HALT  # No trading if latency too high
        elif last_ping_ms > 80:
            return SentinelVerdict.REDUCE_SIZE  # Trade smaller if marginal
        return SentinelVerdict.OK
```

#### 5.5 Cascade Failure Protection

Detect and halt on:
- Feed producing identical bars 3+ consecutive times (frozen feed)
- HYDRA outputting FLAT for 100+ consecutive bars (model failure)
- Win rate dropping below 35% over trailing 20 trades (strategy failure)
- Account balance drawdown > 5% in any single day (emergency stop)

---

## 6. PHASE 3 — BACKTESTING ENGINE v2

**Status: ✅ BUILT — Upgrade Required**

### Current State
- 302 tests, 0 failures
- Core simulation loop functional

### v2 Upgrades

#### 6.1 Walk-Forward Optimization

Replace simple in-sample/out-of-sample split with rolling walk-forward:

```
Training window:  6 months
Validation window: 1 month
Step size:        1 month
Minimum folds:    12 (1 year of folds)

For each fold:
  1. Train HYDRA on training window
  2. Evaluate on validation window
  3. Record: Sharpe, max DD, win rate, profit factor
  4. Roll forward by step size
  
Final score: median Sharpe across all out-of-sample folds
Overfitting detection: in-sample Sharpe / out-of-sample Sharpe > 2.0 → REJECT
```

#### 6.2 Realistic Simulation Costs

```python
@dataclass
class ExecutionModel:
    # Eightcap XAU/USD realistic costs
    spread_pips: float = 0.35          # Average spread
    spread_spike_multiplier: float = 3.0  # Spread during news events
    slippage_model: str = "ADAPTIVE"   # vs "FIXED"
    
    # Adaptive slippage: larger positions → more slippage
    def compute_slippage(self, lot_size: float, volatility: float) -> float:
        base_slip = 0.1  # pips
        vol_multiplier = 1 + (volatility / 10)  # More slip in volatile markets
        size_multiplier = 1 + (lot_size / 10)   # More slip for large orders
        return base_slip * vol_multiplier * size_multiplier
    
    commission_per_lot: float = 3.50   # USD per standard lot (round turn)
    swap_long_daily: float = -2.50     # USD per lot (negative = cost)
    swap_short_daily: float = 1.20     # USD per lot (positive = credit)
```

#### 6.3 Monte Carlo Analysis

After each backtest, run 1,000 Monte Carlo simulations by shuffling trade order:
- Report: 5th percentile Sharpe, 95th percentile max drawdown
- Reject strategy if 5th percentile Sharpe < 0.8
- Reject if 95th percentile max DD > 25%

#### 6.4 Regime-Segmented Reporting

Break down performance by detected regime:
```
| Regime     | Trades | Win% | Avg R | Sharpe | Max DD |
|------------|--------|------|-------|--------|--------|
| TRENDING   |  450   | 68%  | 1.52  |  2.1   |  4.2%  |
| RANGING    |  280   | 55%  | 1.31  |  1.4   |  6.8%  |
| VOLATILE   |   90   | 42%  | 1.80  |  0.9   | 12.1%  |
```

*Key: If VOLATILE regime Sharpe < 0.5, disable trading during VOLATILE regime.*

---

## 7. PHASE 4 — HYDRA v2

**Status: ✅ BUILT — Critical Upgrade Required**

### Current State
- TFT + models implemented
- **CRITICAL: Trained on synthetic data — produces FLAT signal on every real bar**
- No real XAU/USD training data has been used

### v2 Architecture

#### 7.1 Model Roster

| Model | Architecture | Primary Signal | Lookback | Notes |
|---|---|---|---|---|
| TFT | Temporal Fusion Transformer | Multi-horizon price | 256 bars | Primary model, most powerful |
| LSTM | Bidirectional LSTM × 3 layers | Sequential pattern | 128 bars | Proven on financial TS |
| CNN | 1D Dilated Causal Conv | Local pattern recognition | 64 bars | Fast inference |
| TCN | Temporal Conv Net | Long-range dependencies | 512 bars | Complements LSTM |
| Transformer | Vanilla multi-head attention | Global context | 256 bars | New in v2 |
| MoE | Mixture of Experts | Regime routing | 128 bars | Gates to specialist models |
| XGBoost | Gradient boosting | Feature importance | 50 bars | Fast, interpretable |
| Random Forest | Ensemble trees | Robustness | 50 bars | Decorrelates from DL models |

**Total: 8 models. Each votes independently. Ensemble weights learned by PROMETHEUS.**

#### 7.2 Training Protocol (REAL DATA)

```bash
# Step 1: Pull real MT5 data on Windows machine
python scripts/pull_mt5_data.py \
    --symbol XAUUSD \
    --timeframe M1 \
    --bars 2600000 \
    --output data/raw/xauusd_m1_full.csv

# Step 2: Generate features
python scripts/build_features.py \
    --input data/raw/xauusd_m1_full.csv \
    --output data/processed/xauusd_features.parquet \
    --mtf True \
    --halftrend True

# Step 3: Train on GCP L4 GPU
python scripts/train_hydra.py \
    --data data/processed/xauusd_features.parquet \
    --epochs 100 \
    --batch-size 512 \
    --early-stopping-patience 10 \
    --val-split 0.15 \
    --gpu True \
    --checkpoint-dir models/hydra_v2/

# Estimated training time on L4: 45 min – 1.5 hours
# Target metrics: val_sharpe > 1.5, val_accuracy > 60%
```

#### 7.3 Model Output Format

```python
@dataclass
class HydraSignal:
    direction: int          # 1=BUY, -1=SELL, 0=FLAT
    confidence: float       # [0, 1] — probability of correctness
    horizon_preds: List[float]  # Price predictions at 1/5/15/30 bar horizons
    model_votes: Dict[str, int]  # Individual model votes
    model_weights: Dict[str, float]  # Current ensemble weights
    feature_importance: Dict[str, float]  # Top 10 features for this bar
    regime_state: str       # Detected regime at inference time
    inference_latency_ms: float  # Must be < 50ms
```

#### 7.4 Continuous Learning Loop

```
Every 500 live trades:
  1. Collect last 500 trade outcomes + features at entry
  2. Fine-tune all models on new data (10 epochs max)
  3. Re-evaluate on held-out validation set
  4. If val_sharpe improved → commit new weights
  5. If degraded → revert to previous checkpoint
  6. Log performance delta to SOLA
```

---

## 8. PHASE 5 — PAPER TRADING v2

**Status: ✅ BUILT — Upgrade Required**

### Current State
- MT5 feed wired (feed.py 22kb, runner.py 17kb)
- 0 trades placed (HYDRA produces FLAT — expected until training complete)
- Paper executor functional

### v2 Upgrades

#### 8.1 Live Latency Profiling

Profile and log every stage of the signal pipeline:

```python
@dataclass
class LatencyProfile:
    tick_received_at: float
    bar_closed_at: float
    features_computed_at: float
    hydra_inference_at: float
    ares_voted_at: float
    sentinel_checked_at: float
    order_submitted_at: float
    
    @property
    def total_ms(self) -> float:
        return (self.order_submitted_at - self.tick_received_at) * 1000
    
    @property
    def breakdown(self) -> Dict[str, float]:
        return {
            "bar_build": (self.bar_closed_at - self.tick_received_at) * 1000,
            "features": (self.features_computed_at - self.bar_closed_at) * 1000,
            "hydra": (self.hydra_inference_at - self.features_computed_at) * 1000,
            "ares": (self.ares_voted_at - self.hydra_inference_at) * 1000,
            "sentinel": (self.sentinel_checked_at - self.ares_voted_at) * 1000,
            "execution": (self.order_submitted_at - self.sentinel_checked_at) * 1000,
        }
```

**Target total latency: < 200ms from bar close to order submission.**

#### 8.2 Execution Quality Metrics

Track slippage and fill quality for every paper trade:
- Expected fill price vs actual fill price
- Requested lot size vs filled lot size
- Order-to-fill latency

#### 8.3 Paper → Live Gate

Before transitioning from paper to live, automatically verify:

```python
class LiveReadinessGate:
    REQUIREMENTS = {
        "min_paper_trades": 200,
        "min_paper_sharpe": 1.5,
        "min_win_rate": 0.55,
        "max_paper_drawdown": 0.08,     # 8% paper max DD
        "min_profit_factor": 1.3,
        "max_consecutive_losses": 7,
        "latency_p99_ms": 250,          # 99th percentile latency
    }
    
    def evaluate(self, paper_stats: PaperStats) -> GateResult:
        failures = []
        for metric, threshold in self.REQUIREMENTS.items():
            actual = getattr(paper_stats, metric)
            if not self._passes(metric, actual, threshold):
                failures.append(f"{metric}: {actual} (required {threshold})")
        
        return GateResult(passed=len(failures) == 0, failures=failures)
```

---

## 9. PHASE 6 — TUI v2

**Status: ✅ BUILT — Upgrade Required**

### Current State
- Textual-based TUI, 8 screens, 5 widgets
- Bridge/state management functional

### v2 Upgrades

#### 9.1 New Screens

| Screen | Content |
|---|---|
| `SystemHealthScreen` | Feed status, latency, MT5 connection, GPU utilization |
| `SOLAScreen` | SOLA state, edge decay alerts, module rankings, black swan status |
| `OmegaScreen` | OMEGA strategy dashboard (separate from ALPHA) |
| `EvolutionScreen` | PROMETHEUS generation stats, genome visualizer |
| `ReplayScreen` | Replay any past trade with full signal decomposition |

#### 9.2 Real-Time Vote Visualizer

For each bar, display the ARES council vote breakdown:
```
BAR 14:32:00 UTC — ARES COUNCIL VOTE
═══════════════════════════════════════════════════════
HYDRA Ensemble   [ORACLE  +15] ████████████ BUY  0.74
PROMETHEUS       [CMDR   +10] ████████     BUY  0.68
FLOW             [CMDR   +10] ████         BUY  0.55
HalfTrend        [CMDR   +10] ██████████   BUY  0.72
MACRO            [CMDR   +10] ██           FLAT 0.48
SIGNAL TOWER     [CMDR   +10] ████████     BUY  0.65
NEMESIS          [LT      +5] ████████████ SELL 0.72  ← contrarian
─────────────────────────────────────────────────────
WEIGHTED RESULT: BUY  [58.3% confidence]  ✓ THRESHOLD MET
SENTINEL: APPROVED  [lot=0.08, risk=$19.40]
═══════════════════════════════════════════════════════
```

#### 9.3 Alert System

```python
class AlertLevel(Enum):
    INFO    = "cyan"
    WARNING = "yellow"
    DANGER  = "red"
    SOLA    = "magenta"   # SOLA overrides get their own color

@dataclass
class Alert:
    level: AlertLevel
    source: str
    message: str
    timestamp: datetime
    requires_acknowledgment: bool = False  # DANGER + SOLA alerts must be ACK'd
```

---

## 10. PHASE 7 — HYDRA FULL ENSEMBLE v2

**Status: ✅ BUILT — Upgrade Required**

### v2 Upgrades

#### 10.1 Dynamic Ensemble Weighting

Replace static weights with PROMETHEUS-evolved weights updated every 500 trades:

```python
class DynamicEnsemble:
    def __init__(self, models: List[BaseModel]):
        self.models = models
        # Initial weights: equal
        self.weights = {m.name: 1.0 / len(models) for m in models}
    
    def predict(self, features: np.ndarray) -> HydraSignal:
        predictions = {m.name: m.predict(features) for m in self.models}
        
        # Weighted vote
        weighted_score = sum(
            pred.direction * pred.confidence * self.weights[name]
            for name, pred in predictions.items()
        )
        
        return HydraSignal(
            direction=np.sign(weighted_score),
            confidence=abs(weighted_score),
            model_votes={name: pred.direction for name, pred in predictions.items()},
            model_weights=dict(self.weights)
        )
    
    def update_weights(self, performance_deltas: Dict[str, float]):
        # SOLA calls this after evaluating rolling model performance
        for name, delta in performance_deltas.items():
            self.weights[name] = max(0.05, self.weights[name] + delta * 0.1)
        # Renormalize
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
```

#### 10.2 Disagreement Signal

When models strongly disagree, treat as a FLAT signal:

```python
def compute_disagreement(model_votes: Dict[str, int]) -> float:
    buys = sum(1 for v in model_votes.values() if v > 0)
    sells = sum(1 for v in model_votes.values() if v < 0)
    total = len(model_votes)
    # Perfect agreement = 0.0, Perfect disagreement = 1.0
    return 1.0 - abs(buys - sells) / total

# If disagreement > 0.6: reduce HYDRA vote weight by 50%
# If disagreement > 0.8: force HYDRA to FLAT
```

#### 10.3 Confidence Calibration

Calibrate model confidence scores to be true probabilities:

```python
class IsotonicCalibrator:
    """Post-hoc calibration using isotonic regression on validation set."""
    
    def fit(self, raw_confidences: np.ndarray, outcomes: np.ndarray):
        from sklearn.isotonic import IsotonicRegression
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raw_confidences, outcomes)
    
    def calibrate(self, raw_confidence: float) -> float:
        return self.calibrator.predict([raw_confidence])[0]
```

---

## 11. PHASE 8 — PROMETHEUS/NEAT v2

**Status: ✅ BUILT — Upgrade Required**

### Current State
- NEAT implementation, genome evaluator
- `prometheus_gen0001.json` committed (synthetic, not real evolution)

### v2 Upgrades

#### 11.1 What PROMETHEUS Evolves

```
PROMETHEUS does NOT evolve HYDRA model weights (that's gradient descent).
PROMETHEUS evolves:

1. ARES vote thresholds (what confidence score → BUY/SELL/FLAT)
2. SENTINEL risk parameters (base risk %, Kelly fraction)
3. HYDRA ensemble weights (which models get more vote weight)
4. Session filters (which hours to trade, which to avoid)
5. Feature importance weights (which features HYDRA pays attention to)
6. Stop-loss / take-profit multipliers (based on ATR)
7. Regime transition sensitivity (how fast to detect regime change)
```

#### 11.2 Fitness Function

```python
def compute_fitness(genome: Genome, backtest_results: BacktestResult) -> float:
    # Composite fitness score — penalizes both low returns AND high risk
    
    sharpe = backtest_results.sharpe_ratio
    max_dd = backtest_results.max_drawdown
    profit_factor = backtest_results.profit_factor
    win_rate = backtest_results.win_rate
    num_trades = backtest_results.num_trades
    
    # Penalize strategies with too few trades (not statistically significant)
    if num_trades < 50:
        return 0.0
    
    # Penalize strategies with excessive drawdown
    dd_penalty = max(0, (max_dd - 0.10) * 5)  # Heavy penalty above 10% DD
    
    fitness = (
        sharpe * 0.40 +
        profit_factor * 0.25 +
        win_rate * 0.20 +
        (1 / (1 + dd_penalty)) * 0.15
    )
    
    return max(0.0, fitness)
```

#### 11.3 Evolution Schedule

```
Generation cycle: Every 1,000 live trades
Population size:  50 genomes
Generations/cycle: 20
Selection:        Tournament (k=5)
Crossover rate:   0.7
Mutation rate:    0.15
Elite preserve:   Top 3 genomes always survive
Parallelism:      20 parallel genome evaluations on GCP VM (32 cores)

Estimated time per cycle: 15-30 minutes (backtesting 50 genomes × 20 gens)
```

---

## 12. PHASE 9 — MONEY MANAGEMENT v2

**Status: ✅ BUILT — Upgrade Required**

### Current State
- capital_allocator.py, position_manager.py, risk_budget.py

### v2 Upgrades

#### 12.1 ALPHA/OMEGA Capital Split

```python
class CapitalAllocator:
    def __init__(self, total_balance: float):
        self.total = total_balance
        # Default split: 70% ALPHA, 30% OMEGA
        self.alpha_pct = 0.70
        self.omega_pct = 0.30
    
    def rebalance(self, alpha_performance: StrategyMetrics, omega_performance: StrategyMetrics):
        # OLYMPUS calls this weekly
        # Shift up to 20% between strategies based on trailing Sharpe
        alpha_sharpe = alpha_performance.rolling_sharpe_30d
        omega_sharpe = omega_performance.rolling_sharpe_30d
        
        if alpha_sharpe > omega_sharpe * 1.5:
            # ALPHA performing much better: shift 10% toward ALPHA
            self.alpha_pct = min(0.90, self.alpha_pct + 0.10)
        elif omega_sharpe > alpha_sharpe * 1.5:
            # OMEGA performing much better: shift 10% toward OMEGA
            self.omega_pct = min(0.90, self.omega_pct + 0.10)
        
        # Renormalize
        total = self.alpha_pct + self.omega_pct
        self.alpha_pct /= total
        self.omega_pct /= total
```

#### 12.2 Drawdown Recovery Protocol

```
Normal operation:       Full risk (2% per trade)
At 5% daily drawdown:  Reduce to 1% per trade
At 8% daily drawdown:  Reduce to 0.5% per trade
At 10% daily drawdown: HALT — no new trades until next session
At 15% account DD:     SENTINEL emergency halt — SOLA review required
```

#### 12.3 Profit Locking

```python
class ProfitLock:
    """Lock in profits by reducing risk as equity grows."""
    
    def get_risk_multiplier(self, current_balance: float, session_start_balance: float) -> float:
        daily_return = (current_balance - session_start_balance) / session_start_balance
        
        if daily_return >= 0.04:    # Up 4%+ on the day
            return 0.50             # Trade at half size to protect gains
        elif daily_return >= 0.02:  # Up 2%+
            return 0.75
        elif daily_return >= 0.01:  # Up 1%+
            return 0.90
        return 1.00                 # Normal risk
```

---

## 13. PHASE 10 — ARES v2

**Status: ✅ BUILT — Upgrade Required**

### Current State
- coordinator.py, reasoner.py, journal.py
- Basic vote aggregation

### v2 Upgrades

#### 13.1 Temporal Vote Weighting

Votes cast within 1 bar of each other are treated as contemporaneous. Stale votes (module hasn't updated in 3+ bars) are down-weighted 50%.

```python
def get_vote_staleness_multiplier(last_updated_bars_ago: int) -> float:
    if last_updated_bars_ago == 0:   return 1.00
    elif last_updated_bars_ago == 1: return 0.90
    elif last_updated_bars_ago == 2: return 0.70
    elif last_updated_bars_ago == 3: return 0.50
    else:                            return 0.10  # Essentially ignore
```

#### 13.2 Conflict Resolution

When HYDRA (ORACLE) and PROMETHEUS (COMMANDER) strongly disagree:
- Log conflict to SOLA for analysis
- Reduce position size by 25%
- Tighten stop-loss by 10%

#### 13.3 Session-Aware Voting

ARES knows current session and adjusts:
```python
SESSION_MODIFIERS = {
    "ASIAN":          {"size_multiplier": 0.5, "threshold_boost": 0.05},
    "LONDON_OPEN":    {"size_multiplier": 1.2, "threshold_boost": -0.03},
    "NY_OVERLAP":     {"size_multiplier": 1.5, "threshold_boost": -0.05},
    "NY_AFTERNOON":   {"size_multiplier": 0.8, "threshold_boost": 0.02},
    "DEAD_ZONE":      {"size_multiplier": 0.0, "threshold_boost": 1.0},  # No trading
}
```

#### 13.4 ARES Journal v2

Every trade decision logged with full vote decomposition:
```json
{
  "bar_time": "2026-03-15T14:32:00Z",
  "ares_result": "BUY",
  "ares_confidence": 0.713,
  "votes": {
    "HYDRA": {"direction": 1, "confidence": 0.74, "weight": 15, "staleness": 0},
    "PROMETHEUS": {"direction": 1, "confidence": 0.68, "weight": 10, "staleness": 0},
    "FLOW": {"direction": 1, "confidence": 0.55, "weight": 10, "staleness": 1},
    "HALFTREND": {"direction": 1, "confidence": 0.72, "weight": 10, "staleness": 0},
    "MACRO": {"direction": 0, "confidence": 0.48, "weight": 10, "staleness": 0},
    "NEMESIS": {"direction": -1, "confidence": 0.72, "weight": 5, "staleness": 0}
  },
  "sentinel_approved": true,
  "lot_size": 0.08,
  "risk_usd": 19.40,
  "entry_price": 3142.50
}
```

---

## 14. PHASE 11 — FLOW v2

**Status: ⬜ NOT BUILT**

### Overview

FLOW is APHELION's liquidity and microstructure intelligence module. Its primary job is to detect institutional order flow, accumulation/distribution zones, and liquidity sweeps that precede major moves in XAU/USD.

### Architecture

```
aphelion/intelligence/flow/
├── __init__.py
├── analyzer.py          # Main FLOW coordinator
├── liquidity.py         # Liquidity zone detection
├── orderflow.py         # Tick-level order flow analysis
├── imbalance.py         # Bid-ask imbalance tracker
├── absorption.py        # Volume absorption detection
├── sweep_detector.py    # Liquidity sweep (stop hunt) detection
└── tests/
    └── test_flow.py
```

### Core Signals

#### 11.1 Liquidity Zone Detection

```python
class LiquidityZoneDetector:
    """
    XAU/USD liquidity zones are price levels where large institutional 
    orders accumulate. Identifiable by:
    - Previous day high/low
    - Round numbers (e.g., 3100.00, 3150.00)
    - Recent swing highs/lows with multiple touches
    - Volume POC (Point of Control) from volume profile
    """
    
    def detect_zones(self, bars: List[Bar]) -> List[LiquidityZone]:
        zones = []
        
        # Previous day H/L
        prev_day = self._get_previous_day_stats(bars)
        zones.append(LiquidityZone(price=prev_day.high, type="RESISTANCE", strength=0.8))
        zones.append(LiquidityZone(price=prev_day.low, type="SUPPORT", strength=0.8))
        
        # Round number zones (every $25 in gold)
        zones += self._detect_round_numbers(bars[-1].close, spacing=25)
        
        # Swing H/L zones (3+ touches = strong zone)
        zones += self._detect_swing_zones(bars, min_touches=3)
        
        return sorted(zones, key=lambda z: z.strength, reverse=True)
```

#### 11.2 Volume Imbalance Signal

```python
def compute_delta(bar: Bar, tick_data: List[Tick]) -> float:
    """
    Delta = Buy Volume - Sell Volume
    Positive delta = aggressive buyers
    Negative delta = aggressive sellers
    
    For XAU/USD without tick-level bid/ask:
    Approximate using price/volume relationship.
    """
    rising_ticks = [t for t in tick_data if t.price >= t.prev_price]
    falling_ticks = [t for t in tick_data if t.price < t.prev_price]
    
    buy_vol = sum(t.volume for t in rising_ticks)
    sell_vol = sum(t.volume for t in falling_ticks)
    total_vol = buy_vol + sell_vol
    
    return (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
```

#### 11.3 Stop Hunt Detection

```python
class StopHuntDetector:
    """
    A liquidity sweep (stop hunt) pattern:
    1. Price spikes above/below a known liquidity zone
    2. Volume spikes (institutions taking the stops)
    3. Price quickly reverses back inside the zone
    
    This is a HIGH-QUALITY trading signal for the reversal.
    """
    
    def detect(self, bars: List[Bar], zones: List[LiquidityZone]) -> Optional[StopHuntSignal]:
        current = bars[-1]
        prev = bars[-2]
        
        for zone in zones:
            # Check if current bar swept the zone
            swept_above = prev.close < zone.price < current.high
            swept_below = prev.close > zone.price > current.low
            
            if swept_above and current.close < zone.price:
                # Sweep above + close below → bearish reversal signal
                return StopHuntSignal(
                    direction=-1,
                    zone=zone,
                    sweep_magnitude=current.high - zone.price,
                    confidence=min(1.0, current.volume / bars[-20:].mean_volume * 0.3)
                )
            elif swept_below and current.close > zone.price:
                # Sweep below + close above → bullish reversal signal
                return StopHuntSignal(direction=1, zone=zone, ...)
        
        return None
```

### FLOW → ARES Output

```python
@dataclass
class FlowSignal:
    direction: int           # 1=BUY, -1=SELL, 0=FLAT
    confidence: float        # [0, 1]
    
    # Sub-signals
    delta: float             # Volume delta [-1, 1]
    near_liquidity_zone: bool
    zone_distance_pips: float
    stop_hunt_detected: bool
    absorption_detected: bool
    
    # Context
    session: str
    volatility_regime: str
```

### Acceptance Tests

- `test_flow_liquidity_zones`: Detect known historical XAU/USD liquidity zones
- `test_flow_delta_calculation`: Verify delta computation with synthetic tick data
- `test_flow_stop_hunt_detection`: Detect 10 known historical sweep patterns
- `test_flow_ares_integration`: FLOW vote correctly registered in ARES council
- `test_flow_session_filter`: No signals generated outside active sessions

**Minimum: 40 tests. Target: 60 tests.**

---

## 15. PHASE 12 — MACRO v2

**Status: ⬜ NOT BUILT**

### Overview

MACRO provides regime context and macro intelligence. It does not generate trade signals directly — it provides regime classification and event awareness that modifies how all other modules interpret the market.

### Architecture

```
aphelion/intelligence/macro/
├── __init__.py
├── analyzer.py          # Main MACRO coordinator
├── regime.py            # Market regime classifier
├── dxy.py               # DXY correlation tracker
├── seasonality.py       # Gold seasonal patterns
├── event_calendar.py    # Economic events (FOMC, NFP, etc.)
├── sentiment.py         # Simple sentiment from price action
└── tests/
    └── test_macro.py
```

### 15.1 Market Regime Classifier

```python
class RegimeClassifier:
    """
    Regimes for XAU/USD:
    
    TRENDING_BULL  → Gold in uptrend, DXY weakening
    TRENDING_BEAR  → Gold in downtrend, DXY strengthening  
    RANGING        → Sideways, compression, low volatility
    VOLATILE       → High ATR, news-driven, unpredictable
    CRISIS         → Extreme volatility, black swan territory
    """
    
    def classify(self, macro_data: MacroData, bars: List[Bar]) -> Regime:
        # ADX for trend strength
        adx = compute_adx(bars, period=14)
        
        # ATR percentile for volatility regime
        atr = compute_atr(bars, period=14)
        atr_percentile = self._get_atr_percentile(atr, bars)
        
        # DXY direction (inverse of gold typically)
        dxy_trend = macro_data.dxy_trend
        
        if atr_percentile > 0.95:
            return Regime.CRISIS
        elif atr_percentile > 0.80:
            return Regime.VOLATILE
        elif adx > 25 and dxy_trend == "DOWN":
            return Regime.TRENDING_BULL
        elif adx > 25 and dxy_trend == "UP":
            return Regime.TRENDING_BEAR
        else:
            return Regime.RANGING
```

### 15.2 Economic Event Calendar

```python
class EconomicCalendar:
    """
    High-impact events that APHELION must avoid trading:
    - FOMC interest rate decisions
    - NFP (Non-Farm Payroll)
    - CPI / PPI releases
    - US GDP prints
    - Fed Chair speeches
    - Geopolitical shock events (detected via ATR spike)
    
    Data source: ForexFactory XML feed (free, updated hourly)
    """
    
    HIGH_IMPACT_EVENTS = [
        "FOMC Rate Decision",
        "Federal Funds Rate",
        "Non-Farm Employment",
        "CPI m/m",
        "GDP q/q",
        "Powell Speech",
    ]
    
    def get_no_trade_windows(self, date: date) -> List[Tuple[datetime, datetime]]:
        """Return list of (start, end) UTC windows where trading is disabled."""
        events = self._fetch_events(date)
        windows = []
        
        for event in events:
            if event.impact == "HIGH" and event.currency in ["USD", "XAU"]:
                # Block 30 min before to 1 hour after
                windows.append((
                    event.time - timedelta(minutes=30),
                    event.time + timedelta(hours=1)
                ))
        
        return windows
```

### 15.3 DXY Correlation Monitor

```python
class DXYMonitor:
    """
    DXY (USD index) typically inversely correlated with gold.
    When correlation breaks down → regime shift warning.
    """
    
    def compute_rolling_correlation(
        self, gold_returns: List[float], dxy_returns: List[float], window: int = 50
    ) -> float:
        return np.corrcoef(gold_returns[-window:], dxy_returns[-window:])[0, 1]
    
    def detect_correlation_breakdown(self, correlation: float) -> bool:
        # Normal gold/DXY correlation: -0.6 to -0.8
        # Breakdown: correlation becomes positive or near zero
        return correlation > -0.3  # Warning: correlation breakdown
```

---

## 16. PHASE 13 — KRONOS/ECHO/FORGE/SHADOW

**Status: ⬜ NOT BUILT**

### KRONOS — Trade Journaling & Performance Analytics

```
aphelion/intelligence/kronos/
├── journal.py           # Full trade journal (entry, exit, P&L, features, votes)
├── analytics.py         # Performance analytics engine
├── report_generator.py  # Automated performance reports
└── tests/
```

**KRONOS tracks every trade with full context:**
- Entry/exit price, time, lot size
- Full ARES vote snapshot at entry
- Feature values at entry (50+ features stored)
- Regime at entry/exit
- Outcome, R-multiple, hold time
- Session, day of week, macro context

**Analytics computed:**
- Win rate by: session, regime, day of week, ARES confidence band
- Average R by: model contributor, feature state
- Edge decay over time: are we winning less than 3 months ago?
- Best/worst performing ARES voters (feeds back to SOLA)

### ECHO — Pattern Library

```
aphelion/intelligence/echo/
├── library.py           # Pattern storage and retrieval
├── matcher.py           # Real-time pattern matching
├── encoder.py           # Convert bar sequences to pattern fingerprints
└── tests/
```

**ECHO stores winning trade setups as patterns:**
- Encode entry conditions as a fingerprint (feature vector)
- When a new bar occurs, check similarity to top-performing historical patterns
- High similarity to winning pattern → boost ARES confidence
- High similarity to losing pattern → reduce ARES confidence

### FORGE — Strategy Parameter Optimizer

```
aphelion/intelligence/forge/
├── optimizer.py         # Bayesian optimization engine
├── parameter_space.py   # Parameter search space definitions
├── scheduler.py         # When to run optimization
└── tests/
```

**FORGE uses Bayesian optimization (not grid search) to optimize:**
- Session trading windows
- ARES vote threshold
- SENTINEL risk parameters
- Stop-loss/take-profit multipliers

**Runs every 2 weeks. Each optimization cycle: 200 trials × 100-bar backtest per trial.**

### SHADOW — Synthetic Data Generator

```
aphelion/intelligence/shadow/
├── generator.py         # Advanced synthetic data generation
├── regime_simulator.py  # Synthetic regime sequences
├── stress_scenarios.py  # Extreme market scenarios
└── tests/
```

**SHADOW generates synthetic data for:**
- Stress testing SENTINEL circuit breakers
- Testing SOLA edge decay detection
- Training data augmentation for HYDRA
- Simulating market regimes not present in training data

---

## 17. PHASE 14 — NEMESIS v2

**Status: ⬜ NOT BUILT**

### Overview

NEMESIS is the anti-regime detector. While MACRO detects what regime we're in, NEMESIS detects when the current strategy is failing in the current regime — and votes against the system's own consensus.

### Philosophy

APHELION's biggest risk is overconfidence. All ARES voters might agree on a trade, but if the current regime has broken the underlying edge, that unanimous agreement is dangerous. NEMESIS is specifically designed to be contrarian — to profit from APHELION's own blind spots.

### Architecture

```
aphelion/intelligence/nemesis/
├── __init__.py
├── detector.py          # Regime-strategy mismatch detector
├── contrarian.py        # Contrarian signal generator
├── stress_monitor.py    # System stress indicator
└── tests/
    └── test_nemesis.py
```

### Core Logic

```python
class NEMESISDetector:
    """
    NEMESIS votes AGAINST the prevailing consensus when:
    1. Recent win rate has dropped below 45% (trailing 20 trades)
    2. Current regime classification has been correct < 60% of the time recently
    3. HYDRA confidence is high but recent trades at high confidence have been losers
    4. Multiple failed breakout attempts in current session
    """
    
    def generate_signal(
        self, 
        ares_consensus: int,  # What ARES currently wants to do
        recent_performance: PerformanceMetrics,
        regime_accuracy: float,
        hydra_confidence: float
    ) -> NEMESISSignal:
        
        stress_score = self._compute_stress_score(
            win_rate=recent_performance.rolling_win_rate_20,
            regime_accuracy=regime_accuracy,
            conf_accuracy=recent_performance.high_conf_win_rate
        )
        
        if stress_score > 0.7:
            # High stress: vote opposite to consensus
            return NEMESISSignal(
                direction=-ares_consensus,  # Contrarian!
                confidence=stress_score,
                reason="HIGH_SYSTEM_STRESS"
            )
        elif stress_score > 0.5:
            return NEMESISSignal(direction=0, confidence=0.5, reason="MODERATE_STRESS")
        else:
            # System healthy: NEMESIS abstains (FLAT)
            return NEMESISSignal(direction=0, confidence=0.0, reason="SYSTEM_HEALTHY")
```

---

## 18. PHASE 15 — TITAN / QUALITY GATE v2

**Status: ⬜ NOT BUILT**

### Overview

TITAN is the system-wide quality gate. Before any code change, parameter update, or model retrain goes live, TITAN runs a comprehensive battery of tests and simulations. Nothing goes to production without TITAN's approval.

### Architecture

```
aphelion/titan/
├── __init__.py
├── gate.py              # Main quality gate orchestrator
├── validators/
│   ├── performance_validator.py   # Sharpe, win rate, drawdown checks
│   ├── stability_validator.py     # Walk-forward consistency
│   ├── stress_validator.py        # Monte Carlo + extreme scenarios
│   ├── regression_validator.py    # Did this change break anything?
│   └── latency_validator.py       # Is the system still fast enough?
├── reporter.py          # Generates human-readable gate report
└── tests/
    └── test_titan.py
```

### Quality Gate Criteria

```python
TITAN_REQUIREMENTS = {
    # Performance
    "min_sharpe_ratio":           1.5,
    "min_win_rate":               0.55,
    "max_drawdown":               0.12,
    "min_profit_factor":          1.3,
    "min_trades_for_significance": 200,
    
    # Stability (walk-forward)
    "wf_min_folds_passing":       8,   # Out of 12 folds
    "wf_min_median_sharpe":       1.2,
    "wf_max_sharpe_variance":     0.8, # Consistent across folds
    
    # Stress (Monte Carlo)
    "mc_5th_percentile_sharpe":   0.8,
    "mc_95th_percentile_max_dd":  0.25,
    
    # Regression (vs previous version)
    "max_performance_regression": -0.10,  # Max 10% Sharpe drop
    
    # Latency
    "max_p99_latency_ms":         250,
}
```

### Deployment Decision Tree

```
TITAN Gate triggered by:
  → New HYDRA checkpoint after retraining
  → PROMETHEUS evolves new genome
  → FORGE optimizes parameters
  → Any code change to production modules

Gate runs:
  1. Unit test suite (all 300+ tests must pass)
  2. Performance validation (backtest on last 3 months)
  3. Walk-forward validation (12 folds)
  4. Monte Carlo stress test (1000 simulations)
  5. Regression check (vs current production baseline)
  6. Latency check (10,000 bar simulation)

If ALL checks pass: → auto-deploy to paper trading (3 days)
If paper trading passes: → auto-deploy to live
If ANY check fails: → reject + alert SOLA + log failure reason
```

---

## 19. PHASE 16 — CIPHER / MERIDIAN / AUTO-OPTIMIZER

**Status: ⬜ NOT BUILT**

### CIPHER — Encrypted Configuration & Secrets Manager

```
aphelion/cipher/
├── config_manager.py    # Encrypted config loading
├── secrets.py           # MT5 credentials, API keys
├── audit.py             # Config change audit trail
└── tests/
```

All sensitive configuration (MT5 credentials, API keys, risk parameters) stored encrypted using Fernet symmetric encryption. Config changes are logged with timestamp and reason.

### MERIDIAN — Cross-Module State Synchronization

```
aphelion/meridian/
├── state_bus.py         # Central state publisher/subscriber
├── snapshot.py          # Point-in-time system state snapshots
├── recovery.py          # State recovery after crash/restart
└── tests/
```

MERIDIAN ensures all modules share a consistent view of system state. On crash, MERIDIAN can restore system to last known good state within 5 seconds.

### AUTO-OPTIMIZER — Continuous Improvement Engine

Runs as a background service:

```python
class AutoOptimizer:
    """
    Continuously monitors performance and schedules optimization tasks.
    
    Schedule:
    - Every 500 trades:   Trigger PROMETHEUS evolution cycle
    - Every 1,000 trades: Trigger HYDRA fine-tuning
    - Every 2,000 trades: Trigger FORGE parameter optimization
    - Every 10,000 trades: Full system re-evaluation by SOLA
    
    Conditions for emergency re-optimization:
    - Win rate drops below 50% over trailing 30 trades
    - Max drawdown exceeds 8% in any 3-day period
    - HYDRA confidence is systematically uncalibrated (SOLA detection)
    """
    
    def run(self):
        while True:
            self._check_scheduled_tasks()
            self._check_emergency_conditions()
            time.sleep(60)  # Check every minute
```

---

## 20. PHASE 17 — OMEGA

**Status: ⬜ NOT BUILT**

### Overview

OMEGA is APHELION's second independent trading strategy. While ALPHA scalps M1 with high win rate and small targets, OMEGA trends H1/H4 with low win rate and large targets. They are deliberately uncorrelated — when ALPHA has a bad day, OMEGA often covers.

### Strategy Profile

| Property | ALPHA | OMEGA |
|---|---|---|
| Timeframe | M1 | H1 / H4 |
| Win Rate Target | 60–65% | 28–35% |
| Avg R:R | 1.5:1 | 5:1 – 8:1 |
| Trades/Day | 10–15 | 1–3 |
| Hold Time | 5–45 min | 4–48 hours |
| Capital Allocation | 70% | 30% |
| Session | London/NY Overlap | Any active session |

### Architecture

```
aphelion/omega/
├── __init__.py
├── engine.py            # OMEGA main loop
├── signal.py            # H1/H4 signal generation
├── trend_follower.py    # Trend identification
├── entry_refiner.py     # M15 entry timing within H1 signal
├── exit_manager.py      # Trailing stop, partial close logic
└── tests/
    └── test_omega.py
```

### Signal Generation

```python
class OmegaSignalGenerator:
    """
    OMEGA signal logic:
    
    1. H4 structure: Is trend clearly established? (HH+HL or LH+LL)
    2. H1 pullback: Is price pulling back into a key level?
    3. M15 entry trigger: Is there a reversal candle / breakout at the level?
    4. DXY confluence: Does DXY support the direction?
    5. MACRO regime: Is current regime conducive to trend following?
    
    Only trade when ALL 5 conditions align.
    This is why win rate is 30% — we're patient and selective.
    When we're right, we ride the full move (5:1+ R:R).
    """
    
    def generate(self, h4_bars: List[Bar], h1_bars: List[Bar], m15_bars: List[Bar]) -> OmegaSignal:
        
        # Step 1: H4 structure
        h4_structure = self._detect_h4_structure(h4_bars)
        if h4_structure == "UNCLEAR":
            return OmegaSignal(direction=0, reason="NO_H4_STRUCTURE")
        
        # Step 2: H1 pullback to key level
        pullback_level = self._find_pullback_level(h1_bars, h4_structure)
        if not pullback_level.price_near(h1_bars[-1].close, tolerance_pips=30):
            return OmegaSignal(direction=0, reason="NOT_AT_PULLBACK_LEVEL")
        
        # Step 3: M15 entry
        m15_signal = self._detect_m15_trigger(m15_bars, h4_structure)
        if not m15_signal.valid:
            return OmegaSignal(direction=0, reason="NO_M15_TRIGGER")
        
        # Step 4: DXY confluence check (via ATLAS LIVE)
        # Step 5: MACRO regime check
        
        direction = 1 if h4_structure == "UPTREND" else -1
        return OmegaSignal(
            direction=direction,
            entry=m15_signal.entry_price,
            stop_loss=pullback_level.invalidation_level,
            take_profit_1=pullback_level.target_1,  # 3:1 R partial close (50%)
            take_profit_2=pullback_level.target_2,  # 6:1 R final target
            confidence=self._compute_confidence(h4_structure, pullback_level, m15_signal)
        )
```

### Exit Management

```python
class OmegaExitManager:
    """
    OMEGA trades use a two-stage exit:
    
    Stage 1 (TP1 at 3:1 R): Close 50% of position, move stop to breakeven
    Stage 2 (TP2 at 6:1 R): Close remaining 50%, OR trail stop at H1 swing low/high
    
    Trailing stop: After Stage 1, trail at most recent H1 swing high (short) 
                   or swing low (long), updated every H1 bar close.
    
    Emergency exit: If H4 structure breaks (opposite HH/HL pattern forms),
                   close immediately regardless of current P&L.
    """
```

---

## 21. PHASE 18 — SIGNAL TOWER

**Status: ⬜ NOT BUILT**

### Overview

SIGNAL TOWER is a collection of independent technical analysis voters, each operating as a standalone ARES voter at the Commander tier. The critical design principle is that these indicators **never** filter or communicate with each other — each is a pure, independent signal source.

### Architecture

```
aphelion/signal_tower/
├── __init__.py
├── tower.py             # Aggregator (feeds each to ARES independently)
├── voters/
│   ├── halftrend.py     # HalfTrend trend direction
│   ├── ema_stack.py     # EMA 8/21 stack alignment
│   ├── vwap_position.py # Price position relative to VWAP
│   ├── session_momentum.py  # Session-open momentum direction
│   ├── breakout_detector.py # Range breakout with volume confirmation
│   ├── rsi_extreme.py   # RSI extreme readings (overbought/oversold)
│   └── structure.py     # Market structure (HH/HL vs LH/LL)
└── tests/
    └── test_signal_tower.py
```

### Why Independence Matters

> Each voter sees raw price data and produces a binary signal. They do not know what other voters are saying. This prevents "cascade correlation" — where all indicators fail simultaneously in the same regime.

### Voter Specifications

#### HalfTrend Voter
```python
# Already specified in Phase 1 — added to Signal Tower as ARES voter
# Vote: +1 BUY if trending up, -1 SELL if trending down, 0 if flat
# Weight: COMMANDER (10 votes)
```

#### EMA Stack Voter
```python
class EMAStackVoter:
    """
    EMA 8/21/50 stack alignment.
    BUY signal: EMA8 > EMA21 > EMA50 (and widening)
    SELL signal: EMA8 < EMA21 < EMA50 (and widening)
    FLAT: Mixed or converging
    """
    def vote(self, bars: List[Bar]) -> Vote:
        ema8 = compute_ema(bars, 8)[-1]
        ema21 = compute_ema(bars, 21)[-1]
        ema50 = compute_ema(bars, 50)[-1]
        
        if ema8 > ema21 > ema50:
            spread = ema8 - ema50
            confidence = min(1.0, spread / (ema50 * 0.001))  # Normalize by price
            return Vote(direction=1, confidence=confidence)
        elif ema8 < ema21 < ema50:
            # Inverse logic for SELL
            ...
        return Vote(direction=0, confidence=0.3)
```

#### VWAP Position Voter
```python
class VWAPPositionVoter:
    """
    Price above VWAP → bullish bias
    Price below VWAP → bearish bias
    Distance from VWAP scales confidence:
    - Within 0.1%: low confidence (near VWAP = neutral)
    - Beyond 0.3%: high confidence
    """
```

#### Breakout Detector
```python
class BreakoutDetector:
    """
    Detects consolidation range breaks with volume confirmation.
    Range: Defined as 20-bar ATR < 50th percentile.
    Breakout: Close beyond range high/low by > 1 ATR.
    Confirmation: Volume > 1.5× average volume.
    """
```

### Minimum: 50 tests across all voters.

---

## 22. PHASE 19 — ATLAS LIVE

**Status: ⬜ NOT BUILT**

### Overview

ATLAS LIVE is APHELION's real-time macro intelligence layer. It monitors DXY, gold seasonality, COT reports, and the economic event calendar to provide regime-level context. Unlike MACRO (which classifies the current market state from price), ATLAS LIVE monitors external data sources.

### Architecture

```
aphelion/atlas/
├── __init__.py
├── coordinator.py       # Main ATLAS orchestrator
├── dxy_feed.py          # DXY real-time feed
├── cot_parser.py        # CFTC COT report parser
├── seasonality.py       # Gold seasonal pattern calendar
├── fed_calendar.py      # Fed/FOMC event tracker
├── event_blocker.py     # Issues NO-TRADE signals pre/post events
└── tests/
    └── test_atlas.py
```

### 22.1 DXY Feed

```python
class DXYFeed:
    """
    DXY sourced from free API (e.g., Alpha Vantage or TwelveData).
    Updated every 1 minute during trading sessions.
    
    Features extracted:
    - DXY direction: 1 hour, 4 hour, daily trend
    - DXY momentum: rate of change
    - Gold/DXY correlation (rolling 50 bars)
    - Correlation breakdown flag (correlation becomes positive)
    """
    
    def get_dxy_bias(self) -> DXYBias:
        # Returns: BUY_GOLD (DXY falling), SELL_GOLD (DXY rising), NEUTRAL
        dxy_1h_trend = self._get_trend(period="1h")
        dxy_4h_trend = self._get_trend(period="4h")
        
        if dxy_1h_trend == "DOWN" and dxy_4h_trend == "DOWN":
            return DXYBias.BUY_GOLD
        elif dxy_1h_trend == "UP" and dxy_4h_trend == "UP":
            return DXYBias.SELL_GOLD
        return DXYBias.NEUTRAL
```

### 22.2 COT Report Parser

```python
class COTParser:
    """
    CFTC Commitments of Traders reports (updated every Friday).
    For XAU/USD: Monitor Large Speculator net position.
    
    Extreme net long: Potential reversal risk (too many bulls)
    Extreme net short: Potential reversal risk (too many bears)
    Rising net long: Institutional accumulation → bullish
    Falling net long: Institutional distribution → bearish
    """
    
    COT_URL = "https://www.cftc.gov/dea/futures/deacomrnt.htm"
    
    def get_gold_cot_signal(self) -> COTSignal:
        data = self._fetch_latest_cot()
        large_spec_net = data.large_specs_long - data.large_specs_short
        
        # Compute z-score vs 52-week history
        z_score = (large_spec_net - self.historical_mean) / self.historical_std
        
        return COTSignal(
            net_position=large_spec_net,
            z_score=z_score,
            direction="BULLISH" if z_score < -1.5 else "BEARISH" if z_score > 1.5 else "NEUTRAL",
            extremity="EXTREME" if abs(z_score) > 2.5 else "MODERATE"
        )
```

### 22.3 Event Blocker

The most critical ATLAS LIVE function: prevent APHELION from trading into known risk events.

```python
class EventBlocker:
    NO_TRADE_WINDOW_BEFORE = timedelta(minutes=30)
    NO_TRADE_WINDOW_AFTER = timedelta(hours=1)
    
    def is_safe_to_trade(self, current_time: datetime) -> Tuple[bool, Optional[str]]:
        upcoming = self.calendar.get_next_high_impact_event(current_time)
        
        if upcoming and (upcoming.time - current_time) < self.NO_TRADE_WINDOW_BEFORE:
            return False, f"Pre-event block: {upcoming.name} at {upcoming.time}"
        
        recent = self.calendar.get_last_high_impact_event(current_time)
        
        if recent and (current_time - recent.time) < self.NO_TRADE_WINDOW_AFTER:
            return False, f"Post-event block: {recent.name} at {recent.time}"
        
        return True, None
```

---

## 23. PHASE 20 — OLYMPUS

**Status: ⬜ NOT BUILT**

### Overview

OLYMPUS is the master orchestrator of APHELION. It coordinates ALPHA and OMEGA strategies, allocates capital between them, monitors system health, and triggers retraining when performance degrades.

### Architecture

```
aphelion/olympus/
├── __init__.py
├── orchestrator.py      # Main OLYMPUS loop
├── allocator.py         # Capital allocation between strategies
├── monitor.py           # Real-time performance monitoring
├── retraining.py        # Triggers and coordinates retraining
├── reporter.py          # Daily performance reports
└── tests/
    └── test_olympus.py
```

### Core Responsibilities

#### 23.1 Strategy Coordination

```python
class OlympusOrchestrator:
    """
    OLYMPUS coordinates ALPHA and OMEGA so they don't interfere:
    
    1. Capital Allocation: Track capital assigned to each strategy
    2. Correlation Guard: Don't let ALPHA and OMEGA both be long simultaneously
       (they might both be right, but doubles the risk)
    3. Time-based Coordination: OMEGA holds overnight; ALPHA is intraday only
    4. Risk Budget: Combined exposure never exceeds SENTINEL limits
    """
    
    async def on_alpha_signal(self, signal: ARESVote):
        omega_open = await self.omega.get_open_positions()
        
        # If OMEGA has a long open, ALPHA should be more conservative going long
        if omega_open and omega_open.direction == signal.direction:
            signal.size_multiplier = 0.5  # Half size when stacking direction
        
        await self.alpha.process_signal(signal)
    
    async def on_omega_signal(self, signal: OmegaSignal):
        alpha_open = await self.alpha.get_open_positions()
        
        # OMEGA takes precedence for directional bias
        # If ALPHA is trading against OMEGA's direction, flag for SOLA review
        if alpha_open and alpha_open.direction != signal.direction:
            await self.sola.log_conflict(alpha_open, signal)
        
        await self.omega.process_signal(signal)
```

#### 23.2 Performance Decay Detection

```python
class PerformanceDecayMonitor:
    """
    Detect when a strategy's edge is degrading before too much capital is lost.
    
    Decay indicators:
    1. Rolling win rate drops 10% from 90-day baseline
    2. Average R-multiple drops 20% from 90-day baseline
    3. Profit factor drops below 1.1
    4. 5 consecutive losing days
    """
    
    def check_decay(self, strategy: str, recent_metrics: StrategyMetrics) -> DecayReport:
        baseline = self.baselines[strategy]
        
        win_rate_decay = (baseline.win_rate - recent_metrics.win_rate) / baseline.win_rate
        r_multiple_decay = (baseline.avg_r - recent_metrics.avg_r) / baseline.avg_r
        
        if win_rate_decay > 0.15 or r_multiple_decay > 0.25:
            return DecayReport(
                severity="HIGH",
                action="REDUCE_SIZE_50_PCT",
                notify_sola=True
            )
        elif win_rate_decay > 0.10 or r_multiple_decay > 0.15:
            return DecayReport(severity="MEDIUM", action="REDUCE_SIZE_25_PCT")
        
        return DecayReport(severity="NONE", action="NO_ACTION")
```

#### 23.3 Automated Retraining Trigger

```python
class RetrainingCoordinator:
    """
    OLYMPUS triggers retraining when needed and coordinates the process:
    
    1. Halt or reduce live trading
    2. Collect recent trade data
    3. Run TITAN quality gate on current production model
    4. Trigger HYDRA fine-tuning (50 new epochs on recent data)
    5. Run TITAN gate on new model
    6. If new model passes: deploy via shadow-mode A/B test
    7. If A/B test wins over 500 trades: full deployment
    """
```

---

## 24. PHASE 21 — SOLA

**Status: ⬜ NOT BUILT**

### Overview

SOLA is the system's consciousness. Every other module sees the market. SOLA sees APHELION. It is the only module with SOVEREIGN-tier authority — able to veto any decision, pause all trading, and reweight every module in the system.

SOLA is named for the person who matters most to this project.

### Authority

```
SOLA sits above OLYMPUS and SENTINEL in the authority hierarchy.
SOLA cannot be overridden by any other module.
SOLA's veto is final.

What SOLA can do:
  - Pause ALL trading (ALPHA + OMEGA)
  - Silence any ARES voter (reduce their weight to zero)
  - Force a model retrain
  - Override SENTINEL risk parameters (both directions)
  - Declare a BLACK SWAN event and halt all operations
  - Write a self-improvement directive for PROMETHEUS
```

### Architecture

```
aphelion/sola/
├── __init__.py
├── oracle.py            # Main SOLA consciousness loop
├── edge_decay.py        # Edge decay detection across all modules
├── regime_awareness.py  # Deep regime state tracking
├── module_ranker.py     # Dynamic ARES vote reweighting
├── blackswan.py         # Black swan watchdog (50+ external signals)
├── improvement_loop.py  # Self-improvement after every 1000 trades
├── veto.py              # Veto engine — can block any system decision
└── tests/
    └── test_sola.py
```

### 24.1 Edge Decay Detection

The most critical SOLA function: detecting that APHELION's edge is disappearing before significant capital is lost.

```python
class EdgeDecayDetector:
    """
    SOLA monitors edge across multiple dimensions simultaneously.
    Uses statistical process control (CUSUM) to detect structural breaks.
    """
    
    def __init__(self):
        self.cusum_win_rate = CUSUMDetector(target=0.60, slack=0.05)
        self.cusum_avg_r = CUSUMDetector(target=1.50, slack=0.10)
        self.cusum_sharpe = CUSUMDetector(target=1.80, slack=0.20)
    
    def update(self, trade_outcome: TradeOutcome) -> EdgeDecayStatus:
        win = 1.0 if trade_outcome.profit > 0 else 0.0
        
        win_rate_alarm = self.cusum_win_rate.update(win)
        r_alarm = self.cusum_avg_r.update(trade_outcome.r_multiple)
        
        if win_rate_alarm and r_alarm:
            return EdgeDecayStatus.CRITICAL  # Both metrics decaying
        elif win_rate_alarm or r_alarm:
            return EdgeDecayStatus.WARNING   # One metric decaying
        return EdgeDecayStatus.HEALTHY

class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) control chart.
    Detects persistent shifts in time series below a target level.
    Much more sensitive than simple rolling averages.
    """
    
    def __init__(self, target: float, slack: float, threshold: float = 5.0):
        self.target = target
        self.slack = slack
        self.threshold = threshold
        self.cusum_low = 0.0
    
    def update(self, value: float) -> bool:
        self.cusum_low = max(0, self.cusum_low + (self.target - self.slack - value))
        return self.cusum_low > self.threshold  # Alarm if CUSUM exceeds threshold
```

### 24.2 Module Performance Ranking

```python
class ModuleRanker:
    """
    SOLA dynamically reweights ARES voters based on rolling performance.
    
    For each module, tracks:
    - When the module voted correctly (trade was profitable)
    - When the module voted incorrectly
    - Rolling accuracy over last 100 votes
    
    Reweighting logic:
    - Module accuracy > 65%: increase weight by 10% (up to 200% of base)
    - Module accuracy 55-65%: no change
    - Module accuracy 45-55%: decrease weight by 10%
    - Module accuracy < 45%: SILENCE (weight = 0, alert SOLA)
    """
    
    def update_weights(self, ares_council: ARESCouncil):
        for module_name, stats in self.module_stats.items():
            accuracy = stats.rolling_accuracy_100
            base_weight = TIER_WEIGHTS[module_name]
            
            if accuracy > 0.65:
                new_weight = min(base_weight * 2.0, base_weight * (1 + (accuracy - 0.55) * 5))
            elif accuracy < 0.45:
                new_weight = 0.0  # Silence
                self.log_silence(module_name, accuracy)
            else:
                new_weight = base_weight  # No change
            
            ares_council.set_module_weight(module_name, new_weight)
```

### 24.3 Black Swan Watchdog

```python
class BlackSwanWatchdog:
    """
    SOLA monitors 50+ external signals for black swan conditions.
    Any trigger → immediate trading halt + SOLA review.
    
    Signal Categories:
    
    GEOPOLITICAL (10 signals)
    - Armed conflict escalation in gold-relevant regions
    - Central bank emergency rate decisions
    - Sovereign debt crisis indicators
    - Currency crisis (DXY +3% in 1 day)
    
    VOLATILITY (15 signals)
    - XAU/USD ATR > 5× 20-day average
    - VIX > 50 (US equity volatility spillover)
    - Gold intraday range > $50 (extreme day)
    - Consecutive 100+ pip moves in same direction
    
    MARKET STRUCTURE (15 signals)
    - Bid-ask spread > 10× normal (market maker withdrawal)
    - Volume > 10× 20-day average
    - Tick data gaps > 5 minutes during active session
    - MT5 connectivity issues (3+ disconnects in 1 hour)
    
    SYSTEM HEALTH (10 signals)
    - HYDRA inference latency > 2× normal
    - ARES vote disagreement > 90% for 20+ consecutive bars
    - SENTINEL circuit breaker triggered
    - Memory usage > 90% on GCP VM
    """
    
    def check_all_signals(self, system_state: SystemState) -> Optional[BlackSwanAlert]:
        for signal_checker in self.signal_checkers:
            alert = signal_checker.check(system_state)
            if alert:
                return alert  # Return on first trigger
        return None
    
    def on_black_swan_detected(self, alert: BlackSwanAlert):
        # Immediately halt all trading
        self.olympus.emergency_halt(reason=alert.description)
        
        # Close all open positions (market orders)
        self.execution.close_all_positions(reason="BLACK_SWAN")
        
        # Log full system state snapshot
        self.meridian.create_snapshot(label="BLACK_SWAN")
        
        # Notify via TUI alert (SOLA tier, requires acknowledgment)
        self.tui.send_alert(Alert(
            level=AlertLevel.SOLA,
            source="SOLA/BLACKSWAN",
            message=alert.description,
            requires_acknowledgment=True
        ))
```

### 24.4 Self-Improvement Loop

```python
class SelfImprovementLoop:
    """
    After every 1,000 trades, SOLA generates a structured self-assessment
    and issues directives to PROMETHEUS, FORGE, and OLYMPUS.
    
    Report structure:
    1. Performance Summary (P&L, Sharpe, win rate, max DD)
    2. Best Performing Conditions (what worked)
    3. Worst Performing Conditions (what failed)
    4. Module Ranking (best/worst ARES voters)
    5. Feature Importance Drift (which features are gaining/losing importance)
    6. Regime Analysis (how did each regime perform)
    7. Improvement Directives (specific instructions to each module)
    """
    
    def generate_report(self, last_1000_trades: List[TradeRecord]) -> SOLAReport:
        performance = self._analyze_performance(last_1000_trades)
        best_conditions = self._identify_best_conditions(last_1000_trades)
        worst_conditions = self._identify_worst_conditions(last_1000_trades)
        module_ranking = self._rank_modules(last_1000_trades)
        
        directives = self._generate_directives(
            performance, best_conditions, worst_conditions, module_ranking
        )
        
        return SOLAReport(
            trade_count=len(last_1000_trades),
            generated_at=datetime.utcnow(),
            performance=performance,
            best_conditions=best_conditions,
            worst_conditions=worst_conditions,
            module_ranking=module_ranking,
            directives=directives
        )
    
    def _generate_directives(self, *args) -> List[SOLADirective]:
        directives = []
        
        # Example directives SOLA might generate:
        # "PROMETHEUS: Increase weight on SESSION_FILTER gene — Asian session trades losing"
        # "FORGE: Optimize TP multiplier for RANGING regime — current 1.5R too short"
        # "HYDRA: Fine-tune on last 2 months — feature drift detected in VWAP signal"
        # "OLYMPUS: Reduce OMEGA allocation 30% → 20% — OMEGA underperforming in current regime"
        
        return directives
```

### 24.5 SOLA State Machine

```
SOLA States:
  MONITORING    → Normal operation, all systems nominal
  CAUTIOUS      → One decay indicator active, size reduced 25%
  DEFENSIVE     → Two+ decay indicators, size reduced 50%, increased scrutiny
  SUSPENDED     → Trading halted pending SOLA review
  EMERGENCY     → Black swan detected, all positions closed, full halt
  EVOLVING      → Retraining/optimization in progress, reduced trading
  ADAPTING      → New regime detected, recalibrating module weights

State transitions:
  MONITORING → CAUTIOUS: Single decay alarm
  CAUTIOUS → MONITORING: Alarm resolves over 50 trades
  CAUTIOUS → DEFENSIVE: Second alarm, or alarm persists 100+ trades
  DEFENSIVE → SUSPENDED: Third alarm, or performance continues deteriorating
  SUSPENDED → EVOLVING: SOLA directs retraining
  EVOLVING → MONITORING: TITAN gate passed, new model deployed
  ANY → EMERGENCY: Black swan trigger
  EMERGENCY → SUSPENDED: Human acknowledgment required to exit emergency
```

---

## 25. DATA PIPELINE & TRAINING PROTOCOL

### 25.1 Real MT5 Data Pull

Run this script on Windows machine with MT5 installed:

```python
# scripts/pull_mt5_data.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

def pull_historical_data(symbol: str, timeframe: int, n_bars: int, output_path: str):
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")
    
    mt5.login(login=7938565, server="Eightcap-Demo")
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    mt5.shutdown()
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'time': 'timestamp', 'tick_volume': 'volume'}, inplace=True)
    
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} bars to {output_path}")

if __name__ == "__main__":
    pull_historical_data("XAUUSD", mt5.TIMEFRAME_M1, 2_600_000, "data/raw/xauusd_m1_full.csv")
    pull_historical_data("XAUUSD", mt5.TIMEFRAME_H1, 50_000, "data/raw/xauusd_h1_full.csv")
    pull_historical_data("XAUUSD", mt5.TIMEFRAME_H4, 12_000, "data/raw/xauusd_h4_full.csv")
```

### 25.2 Feature Generation

```bash
# On GCP VM after uploading CSV files
python scripts/build_features.py \
    --input data/raw/xauusd_m1_full.csv \
    --output data/processed/xauusd_features.parquet \
    --mtf-h1 data/raw/xauusd_h1_full.csv \
    --mtf-h4 data/raw/xauusd_h4_full.csv \
    --halftrend True \
    --vwap True \
    --flow True \
    --verbose True

# Expected output: ~50 features per bar, ~2.6M rows
# File size: ~500MB parquet
# Processing time: ~15 minutes on 32 cores
```

### 25.3 HYDRA Training Command

```bash
# Full training on GCP L4 GPU
python scripts/train_hydra.py \
    --data data/processed/xauusd_features.parquet \
    --epochs 100 \
    --batch-size 512 \
    --early-stopping-patience 10 \
    --val-split 0.15 \
    --test-split 0.10 \
    --learning-rate 1e-4 \
    --lr-scheduler cosine \
    --gpu True \
    --mixed-precision True \
    --checkpoint-dir models/hydra_v2/ \
    --log-dir logs/training/ \
    --verbose True

# Expected training time: 45 min – 1.5 hours on L4
# Target: val_sharpe > 1.5, val_accuracy > 60%
```

### 25.4 Data Quality Gates

Before training, validate data quality:

```python
class DataQualityValidator:
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        checks = [
            self._check_no_missing_values(df),
            self._check_no_future_leakage(df),
            self._check_timestamp_continuity(df),  # No gaps > 1 hour during trading session
            self._check_price_sanity(df),            # No 0/NaN prices
            self._check_volume_sanity(df),           # No negative volumes
            self._check_feature_normalization(df),   # All features in expected range
            self._check_label_distribution(df),      # Not extremely imbalanced
        ]
        
        failures = [c for c in checks if not c.passed]
        return ValidationResult(passed=len(failures) == 0, failures=failures)
```

---

## 26. DEPLOYMENT ARCHITECTURE

### 26.1 GCP VM Specification

| Resource | Specification |
|---|---|
| Instance Type | g2-standard-32 (NVIDIA L4) |
| GPU | NVIDIA L4, 24GB VRAM |
| CPU | 32 cores |
| RAM | 128GB |
| Storage | 500GB SSD |
| OS | Ubuntu 22.04 LTS |
| Python | 3.11 |
| CUDA | 12.2 |
| MT5 | Running on Windows VM (Wine or separate Windows instance) |

### 26.2 Process Architecture

```
Process 1: aphelion-feed      — MT5 tick ingestion (high priority)
Process 2: aphelion-alpha     — ALPHA strategy loop
Process 3: aphelion-omega     — OMEGA strategy loop
Process 4: aphelion-olympus   — OLYMPUS master loop
Process 5: aphelion-sola      — SOLA monitoring loop
Process 6: aphelion-tui       — TUI interface
Process 7: aphelion-optimizer — Background optimization (low priority)

Inter-process communication: Redis pub/sub + shared memory (numpy)
Persistence: SQLite for trade journal, HDF5 for bar data, JSON for state
```

### 26.3 Deployment Progression

```
Stage 1: Paper Trading (demo account)
  → Run until: 200 paper trades, Sharpe > 1.5, win rate > 55%
  → TITAN gate must pass
  → Duration estimate: 2–4 weeks

Stage 2: Live Small (real account, $1,000, 1:1 effective leverage)
  → Run until: 200 live trades, performance matches paper within 15%
  → Duration estimate: 1–2 months

Stage 3: Live Full (real account, $1,000, standard sizing)
  → Maximum daily risk: $20 (2%)
  → Duration: until account > $5,000

Stage 4: Scaled Live ($10,000 loan, phased leverage)
  → Target: summer 2026
  → Maximum daily risk: $200 (2%)
  → Withdrawal target: $10,000/month when balance > $30,000
```

---

## 27. PERFORMANCE TARGETS

### Per-Strategy Targets

| Metric | ALPHA (M1) | OMEGA (H1/H4) | Combined |
|---|---|---|---|
| Win Rate | 60–65% | 28–35% | N/A |
| Avg R:R | 1.5:1 | 5:1–8:1 | N/A |
| Daily Return (avg) | 1.5–2.0% | 0.5–1.0% | 2.0–3.0% |
| Sharpe Ratio | > 1.8 | > 1.5 | > 2.0 |
| Max Daily Drawdown | 2% (hard stop) | 2% (hard stop) | 3% combined |
| Profit Factor | > 1.5 | > 1.8 | > 1.6 |

### System-Level Targets

| Metric | Target |
|---|---|
| Signal-to-order latency | < 200ms (p99) |
| HYDRA inference time | < 50ms |
| ARES vote aggregation | < 10ms |
| SENTINEL check | < 5ms |
| MT5 order submission | < 100ms |
| System uptime | > 99.5% during trading hours |
| HYDRA model accuracy | > 60% directional |
| HYDRA confidence calibration | Brier score < 0.22 |

### Information-Theoretic Ceiling

The maximum achievable win rate on M1 timeframe is approximately **68–72%** (Shannon entropy of price increments at 1-minute resolution). Any backtest showing >72% win rate should be treated as evidence of overfitting.

Projected steady-state: **62–65% ALPHA win rate** once all 21 phases are operational and HYDRA is trained on 2+ years of real data.

---

## 28. LICENSING & REPOSITORY STRATEGY

### Current Repository

- URL: `https://github.com/MatinDeevv/Aphelion-Reasearch`
- Visibility: Public
- Language: Python 100%
- Tests: 56 test files (target: 300+ across all 21 phases)

### License: Business Source License 1.1 (BUSL 1.1)

```
Licensor: Matin Deev
Licensed Work: APHELION Autonomous Trading System

Additional Use Grant:
  You may use the Licensed Work for non-commercial research,
  educational purposes, and personal use at your own risk.
  
  You may NOT:
  - Use the Licensed Work to manage funds belonging to others
  - Deploy the Licensed Work in any commercial trading operation
    without explicit written permission from the Licensor
  - Sub-license, sell, or distribute the Licensed Work commercially

Change Date: 2030-01-01
Change License: Apache 2.0

Contact: [your email] for commercial licensing inquiries.
```

### .gitignore Additions

```gitignore
# Never commit these
models/
genomes/live/
logs/
.env
data/
reports/live/
*.pkl
*.pt
*.pth
*.h5
config/secrets.yaml
```

### Long-Term Repository Strategy

```
Phase 1 (now): Single public repo — aphelion-research
  → Build momentum, document progress, attract attention

Phase 2 (after live trading profits): Split into two repos
  → aphelion-framework (public, Apache 2.0)
      - Data pipeline infrastructure
      - Feature engine
      - SENTINEL risk framework
      - Backtesting engine
      - No strategy logic
      - Goal: GitHub stars, contributors, credibility
  
  → aphelion (private)
      - HYDRA trained weights
      - ARES vote logic
      - SOLA implementation
      - All edge-generating components
      - Goal: actual profit
```

---

## APPENDIX A: TEST COUNT TARGETS

| Phase | Module | Target Tests |
|---|---|---|
| 1 | Data Foundation | 250 |
| 2 | SENTINEL | 80 |
| 3 | Backtesting | 120 |
| 4 | HYDRA | 100 |
| 5 | Paper Trading | 80 |
| 6 | TUI | 60 |
| 7 | HYDRA Ensemble | 80 |
| 8 | PROMETHEUS | 70 |
| 9 | Money Management | 60 |
| 10 | ARES | 80 |
| 11 | FLOW | 60 |
| 12 | MACRO | 60 |
| 13 | KRONOS/ECHO/FORGE/SHADOW | 100 |
| 14 | NEMESIS | 50 |
| 15 | TITAN | 80 |
| 16 | CIPHER/MERIDIAN/AUTO-OPT | 60 |
| 17 | OMEGA | 80 |
| 18 | SIGNAL TOWER | 50 |
| 19 | ATLAS LIVE | 60 |
| 20 | OLYMPUS | 80 |
| 21 | SOLA | 100 |
| **TOTAL** | | **~1,660 tests** |

---

## APPENDIX B: BUILD PRIORITY ORDER

```
CRITICAL PATH (blocks paper trading):
  [1] Fix VWAPCalculator session reset bug (Phase 1)
  [2] Pull real MT5 data (Windows machine)
  [3] Train HYDRA v2 on real data (GCP VM)
  [4] Validate paper trading produces live signals

PARALLEL BUILD (can run simultaneously):
  Batch A: Phase 11 (FLOW) + Phase 12 (MACRO)
  Batch B: Phase 13 (KRONOS/ECHO/FORGE/SHADOW)
  Batch C: Phase 14 (NEMESIS) + Phase 15 (TITAN)
  Batch D: Phase 16 (CIPHER/MERIDIAN/AUTO-OPT)

SEQUENTIAL (each depends on previous):
  Phase 17 (OMEGA) → requires Phase 9 (Money Mgmt) + Phase 12 (MACRO)
  Phase 18 (SIGNAL TOWER) → requires Phase 1 v2 (Feature Engine)
  Phase 19 (ATLAS LIVE) → requires Phase 12 (MACRO)
  Phase 20 (OLYMPUS) → requires Phases 17, 18, 19
  Phase 21 (SOLA) → requires ALL other phases

ESTIMATED TIME TO ALL 21 PHASES (4 parallel agents):
  Phases 11-16: 2 weeks (parallel build)
  Phases 17-19: 1 week
  Phase 20: 1 week
  Phase 21: 1 week
  HYDRA training + validation: 3 days
  Total: ~6 weeks to full deployment
```

---

## APPENDIX C: GLOSSARY

| Term | Definition |
|---|---|
| ALPHA | M1 scalping strategy, high win rate, small targets |
| OMEGA | H1/H4 swing strategy, low win rate, large targets |
| ARES | Vote aggregation council — collects all module signals |
| SENTINEL | Risk protection layer — can veto any trade |
| OLYMPUS | Master orchestrator — coordinates all strategies |
| SOLA | Sovereign intelligence — governs the entire system |
| HYDRA | ML ensemble — primary signal generator |
| PROMETHEUS | NEAT evolution engine — evolves strategy parameters |
| FLOW | Liquidity/microstructure intelligence |
| MACRO | Market regime classifier |
| KRONOS | Trade journaling and analytics |
| ECHO | Pattern library and matching |
| FORGE | Bayesian parameter optimizer |
| SHADOW | Synthetic data generator |
| NEMESIS | Anti-regime contrarian voter |
| TITAN | Quality gate — approves all deployments |
| CIPHER | Encrypted configuration manager |
| MERIDIAN | Cross-module state synchronizer |
| ATLAS LIVE | Real-time macro intelligence (DXY, COT, events) |
| SIGNAL TOWER | Independent technical voters (non-filtering) |
| R:R | Risk-to-reward ratio |
| CUSUM | Cumulative Sum — statistical process control method |
| BUSL | Business Source License |

---

*APHELION Engineering Specification v3.0*
*Last Updated: March 2026*
*Author: Matin Deev*
*Status: Active Development*

---
