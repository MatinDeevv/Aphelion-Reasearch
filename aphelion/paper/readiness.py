"""
APHELION Paper Trading v2 — LatencyProfile & LiveReadinessGate
Phase 5 v2 — Engineering Spec v3.0

LatencyProfile: Measures and validates execution latency characteristics.
LiveReadinessGate: Checklist that must be satisfied before going live.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LatencyBucket:
    """Latency statistics for a specific operation type."""
    operation: str
    samples: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0


@dataclass
class LatencyProfile:
    """Comprehensive latency profiling for paper trading sessions.

    Tracks round-trip latency for:
    - Tick ingestion (feed→engine)
    - Feature computation (engine→features)
    - Model inference (features→HYDRA)
    - Signal aggregation (HYDRA→ARES)
    - Order execution (ARES→broker sim)
    """

    def __init__(self):
        self._records: Dict[str, List[float]] = {}
        self._session_start: float = time.time()

    def record(self, operation: str, latency_ms: float) -> None:
        """Record a latency measurement."""
        if operation not in self._records:
            self._records[operation] = []
        self._records[operation].append(latency_ms)
        # Keep bounded
        if len(self._records[operation]) > 10_000:
            self._records[operation] = self._records[operation][-5_000:]

    def get_bucket(self, operation: str) -> LatencyBucket:
        """Get statistics for a specific operation."""
        samples = self._records.get(operation, [])
        if not samples:
            return LatencyBucket(operation=operation)
        arr = np.array(samples)
        return LatencyBucket(
            operation=operation,
            samples=len(arr),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            max_ms=float(np.max(arr)),
            mean_ms=float(np.mean(arr)),
        )

    def get_all_buckets(self) -> Dict[str, LatencyBucket]:
        """Get statistics for all tracked operations."""
        return {op: self.get_bucket(op) for op in self._records}

    def get_total_pipeline_p99(self) -> float:
        """Sum of p99 across all pipeline stages."""
        total = 0.0
        for op in self._records:
            bucket = self.get_bucket(op)
            total += bucket.p99_ms
        return total

    @property
    def operations(self) -> List[str]:
        return list(self._records.keys())

    @property
    def total_samples(self) -> int:
        return sum(len(v) for v in self._records.values())

    def reset(self) -> None:
        self._records.clear()
        self._session_start = time.time()


@dataclass
class ReadinessCheck:
    """A single readiness gate check item."""
    name: str
    passed: bool
    message: str = ""
    required: bool = True


class LiveReadinessGate:
    """
    Pre-go-live checklist. All required checks must pass before transitioning
    from paper trading to live trading.

    Categories:
    - Performance: Min win rate, Sharpe, profit factor
    - Risk: Drawdown within limits, SENTINEL healthy, no L2/L3 triggers
    - Infrastructure: Latency acceptable, feeds connected, models loaded
    - Duration: Minimum paper trading duration (default 10 days)
    """

    def __init__(
        self,
        min_trades: int = 100,
        min_win_rate: float = 0.52,
        min_sharpe: float = 0.5,
        min_profit_factor: float = 1.2,
        max_drawdown_pct: float = 0.08,
        max_pipeline_p99_ms: float = 500.0,
        min_paper_days: int = 10,
    ):
        self._min_trades = min_trades
        self._min_win_rate = min_win_rate
        self._min_sharpe = min_sharpe
        self._min_profit_factor = min_profit_factor
        self._max_drawdown_pct = max_drawdown_pct
        self._max_pipeline_p99_ms = max_pipeline_p99_ms
        self._min_paper_days = min_paper_days

    def evaluate(
        self,
        total_trades: int,
        win_rate: float,
        sharpe: float,
        profit_factor: float,
        max_drawdown_pct: float,
        sentinel_l2: bool,
        sentinel_l3: bool,
        pipeline_p99_ms: float,
        paper_trading_days: int,
        models_loaded: bool = True,
        feed_connected: bool = True,
    ) -> List[ReadinessCheck]:
        """Run all readiness checks and return results."""
        checks = []

        # Performance gates
        checks.append(ReadinessCheck(
            name="min_trades",
            passed=total_trades >= self._min_trades,
            message=f"{total_trades}/{self._min_trades} trades",
        ))
        checks.append(ReadinessCheck(
            name="win_rate",
            passed=win_rate >= self._min_win_rate,
            message=f"{win_rate:.1%} >= {self._min_win_rate:.1%}",
        ))
        checks.append(ReadinessCheck(
            name="sharpe_ratio",
            passed=sharpe >= self._min_sharpe,
            message=f"{sharpe:.2f} >= {self._min_sharpe:.2f}",
        ))
        checks.append(ReadinessCheck(
            name="profit_factor",
            passed=profit_factor >= self._min_profit_factor,
            message=f"{profit_factor:.2f} >= {self._min_profit_factor:.2f}",
        ))

        # Risk gates
        checks.append(ReadinessCheck(
            name="max_drawdown",
            passed=max_drawdown_pct <= self._max_drawdown_pct,
            message=f"{max_drawdown_pct:.1%} <= {self._max_drawdown_pct:.1%}",
        ))
        checks.append(ReadinessCheck(
            name="sentinel_healthy",
            passed=not sentinel_l2 and not sentinel_l3,
            message="L2={}, L3={}".format(sentinel_l2, sentinel_l3),
        ))

        # Infrastructure gates
        checks.append(ReadinessCheck(
            name="pipeline_latency",
            passed=pipeline_p99_ms <= self._max_pipeline_p99_ms,
            message=f"p99={pipeline_p99_ms:.0f}ms <= {self._max_pipeline_p99_ms:.0f}ms",
        ))
        checks.append(ReadinessCheck(
            name="models_loaded",
            passed=models_loaded,
            message="Models loaded" if models_loaded else "Models NOT loaded",
        ))
        checks.append(ReadinessCheck(
            name="feed_connected",
            passed=feed_connected,
            message="Feed OK" if feed_connected else "Feed DISCONNECTED",
        ))

        # Duration gate
        checks.append(ReadinessCheck(
            name="paper_duration",
            passed=paper_trading_days >= self._min_paper_days,
            message=f"{paper_trading_days}/{self._min_paper_days} days",
        ))

        return checks

    def is_ready(self, checks: List[ReadinessCheck]) -> bool:
        """Return True if all required checks pass."""
        return all(c.passed for c in checks if c.required)

    def summary(self, checks: List[ReadinessCheck]) -> str:
        """Human-readable summary of readiness checks."""
        lines = ["=== Live Readiness Gate ==="]
        for c in checks:
            icon = "✓" if c.passed else "✗"
            req = " [REQUIRED]" if c.required else ""
            lines.append(f"  {icon} {c.name}: {c.message}{req}")
        ready = self.is_ready(checks)
        lines.append(f"\n  Overall: {'READY' if ready else 'NOT READY'}")
        return "\n".join(lines)
