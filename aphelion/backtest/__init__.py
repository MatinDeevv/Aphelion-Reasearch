"""
APHELION Backtest Module
Event-driven backtesting engine with full SENTINEL risk integration,
institutional metrics, Monte Carlo simulation, and walk-forward validation.
"""

from aphelion.backtest.engine import BacktestEngine, BacktestConfig, BacktestResults
from aphelion.backtest.broker_sim import BrokerSimulator, BrokerConfig
from aphelion.backtest.portfolio import Portfolio
from aphelion.backtest.order import (
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    Fill,
    BacktestTrade,
)
from aphelion.backtest.metrics import (
    BacktestMetrics,
    compute_metrics,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    omega_ratio,
    profit_factor,
    max_drawdown,
)
from aphelion.backtest.monte_carlo import (
    MonteCarloEngine,
    MonteCarloConfig,
    MonteCarloResults,
)
from aphelion.backtest.walk_forward import (
    WalkForwardEngine,
    WalkForwardConfig,
    WalkForwardResults,
    WalkForwardWindow,
)

__all__ = [
    # Engine
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResults",
    # Broker
    "BrokerSimulator",
    "BrokerConfig",
    # Portfolio
    "Portfolio",
    # Orders
    "Order",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "Fill",
    "BacktestTrade",
    # Metrics
    "BacktestMetrics",
    "compute_metrics",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "omega_ratio",
    "profit_factor",
    "max_drawdown",
    # Monte Carlo
    "MonteCarloEngine",
    "MonteCarloConfig",
    "MonteCarloResults",
    # Walk-Forward
    "WalkForwardEngine",
    "WalkForwardConfig",
    "WalkForwardResults",
    "WalkForwardWindow",
]
