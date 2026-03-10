"""
SENTINEL v2 — Advanced Risk Controls
Phase 2 — Engineering Spec v3.0

Additional protection layers beyond the core L1/L2/L3 drawdown system:
- Correlation Guard: Prevents opening correlated positions
- Latency Monitor: Halts trading if execution latency degrades
- Cascade Failure Protection: Detects rapid multi-system failures
- Dynamic Position Scaling: Adjusts size based on regime & volatility
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
class LatencyRecord:
    timestamp: float
    latency_ms: float
    operation: str = ""


class CorrelationGuard:
    """
    Prevents opening positions that are too correlated with existing ones.
    For the current single-instrument system (XAU/USD), this guards against
    same-direction stacking. When OMEGA is added, it will guard cross-strategy
    correlation.
    """

    MAX_SAME_DIRECTION = 2          # Max positions in same direction
    MAX_TOTAL_POSITIONS = 3         # Max total open positions
    CORRELATION_THRESHOLD = 0.85    # Block if asset correlation > this

    def __init__(self):
        self._positions: List[dict] = []

    def register_position(self, position_id: str, direction: str, strategy: str = "") -> None:
        self._positions.append({
            "id": position_id, "direction": direction, "strategy": strategy
        })

    def remove_position(self, position_id: str) -> None:
        self._positions = [p for p in self._positions if p["id"] != position_id]

    def can_open(self, direction: str, strategy: str = "") -> tuple[bool, str]:
        # Total positions check
        if len(self._positions) >= self.MAX_TOTAL_POSITIONS:
            return False, f"MAX_POSITIONS ({self.MAX_TOTAL_POSITIONS}) reached"

        # Same-direction stacking check
        same_dir = sum(1 for p in self._positions if p["direction"] == direction)
        if same_dir >= self.MAX_SAME_DIRECTION:
            return False, f"MAX_SAME_DIRECTION ({self.MAX_SAME_DIRECTION}) reached for {direction}"

        # Cross-strategy check (ALPHA + OMEGA shouldn't both be in same direction heavily)
        if strategy:
            other_strategy_same_dir = sum(
                1 for p in self._positions
                if p["direction"] == direction and p["strategy"] != strategy
            )
            if other_strategy_same_dir >= 1:
                # Allow but warn — both strategies shouldn't stack same direction
                logger.warning(
                    "CorrelationGuard: Both strategies have %s positions", direction
                )

        return True, "OK"

    def clear(self) -> None:
        self._positions.clear()

    @property
    def open_count(self) -> int:
        return len(self._positions)


class LatencyMonitor:
    """
    Monitors execution and data feed latency.
    If p99 latency exceeds threshold, halts trading to prevent slippage.
    """

    P99_THRESHOLD_MS = 250.0      # Max acceptable p99 latency
    P50_THRESHOLD_MS = 100.0      # Max acceptable median latency
    WINDOW_SIZE = 100              # Rolling window

    def __init__(self):
        self._records: List[LatencyRecord] = []
        self._halted: bool = False

    def record(self, latency_ms: float, operation: str = "") -> None:
        self._records.append(LatencyRecord(
            timestamp=time.time(),
            latency_ms=latency_ms,
            operation=operation,
        ))
        # Keep only recent records
        if len(self._records) > self.WINDOW_SIZE * 2:
            self._records = self._records[-self.WINDOW_SIZE:]

        self._check_thresholds()

    def _check_thresholds(self) -> None:
        if len(self._records) < 10:
            return

        recent = [r.latency_ms for r in self._records[-self.WINDOW_SIZE:]]
        p99 = float(np.percentile(recent, 99))
        p50 = float(np.percentile(recent, 50))

        if p99 > self.P99_THRESHOLD_MS:
            logger.warning("LatencyMonitor: p99=%.1fms > %.1fms — HALTING", p99, self.P99_THRESHOLD_MS)
            self._halted = True
        elif p50 > self.P50_THRESHOLD_MS:
            logger.warning("LatencyMonitor: p50=%.1fms > %.1fms — DEGRADED", p50, self.P50_THRESHOLD_MS)
        else:
            self._halted = False

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def p99(self) -> float:
        if len(self._records) < 2:
            return 0.0
        recent = [r.latency_ms for r in self._records[-self.WINDOW_SIZE:]]
        return float(np.percentile(recent, 99))

    @property
    def p50(self) -> float:
        if len(self._records) < 2:
            return 0.0
        recent = [r.latency_ms for r in self._records[-self.WINDOW_SIZE:]]
        return float(np.percentile(recent, 50))

    def reset(self) -> None:
        self._records.clear()
        self._halted = False


class CascadeProtection:
    """
    Detects cascade failures — when multiple modules fail in rapid succession.
    If ≥ 3 modules fail within 60 seconds, triggers emergency halt.
    """

    FAILURE_THRESHOLD = 3          # Min failures to trigger
    WINDOW_SECONDS = 60.0          # Time window for cascade detection

    def __init__(self):
        self._failures: List[tuple] = []  # (timestamp, module_name, error_msg)
        self._cascade_active: bool = False

    def report_failure(self, module_name: str, error_msg: str = "") -> bool:
        """Report a module failure. Returns True if cascade detected."""
        now = time.time()
        self._failures.append((now, module_name, error_msg))

        # Prune old failures
        cutoff = now - self.WINDOW_SECONDS
        self._failures = [(t, m, e) for t, m, e in self._failures if t >= cutoff]

        # Check for cascade
        unique_modules = set(m for _, m, _ in self._failures)
        if len(unique_modules) >= self.FAILURE_THRESHOLD:
            self._cascade_active = True
            logger.critical(
                "CascadeProtection: %d modules failed in %.0fs — EMERGENCY",
                len(unique_modules), self.WINDOW_SECONDS,
            )
            return True

        return False

    @property
    def cascade_active(self) -> bool:
        return self._cascade_active

    @property
    def recent_failures(self) -> List[tuple]:
        return list(self._failures)

    def reset(self) -> None:
        self._failures.clear()
        self._cascade_active = False


@dataclass
class DynamicSizeConfig:
    """Configuration for dynamic position sizing based on regime."""
    normal_multiplier: float = 1.0
    trending_multiplier: float = 1.2     # Larger in trends
    ranging_multiplier: float = 0.8      # Smaller in ranges
    volatile_multiplier: float = 0.5     # Much smaller in volatility
    crisis_multiplier: float = 0.0       # No trading in crisis


class DynamicSizer:
    """Adjusts position size based on market regime and volatility."""

    def __init__(self, config: Optional[DynamicSizeConfig] = None):
        self._config = config or DynamicSizeConfig()

    def regime_multiplier(self, regime: str) -> float:
        mapping = {
            "TRENDING_BULL": self._config.trending_multiplier,
            "TRENDING_BEAR": self._config.trending_multiplier,
            "RANGING": self._config.ranging_multiplier,
            "VOLATILE": self._config.volatile_multiplier,
            "CRISIS": self._config.crisis_multiplier,
        }
        return mapping.get(regime, self._config.normal_multiplier)

    def volatility_scalar(self, current_atr: float, avg_atr: float) -> float:
        """Scale size inversely with volatility."""
        if current_atr <= 0 or avg_atr <= 0:
            return 1.0
        ratio = avg_atr / current_atr  # Inverse: high vol = smaller size
        return max(0.25, min(2.0, ratio))

    def compute_adjusted_size(
        self, base_size_pct: float, regime: str,
        current_atr: float, avg_atr: float,
    ) -> float:
        regime_mult = self.regime_multiplier(regime)
        vol_scalar = self.volatility_scalar(current_atr, avg_atr)
        adjusted = base_size_pct * regime_mult * vol_scalar
        return max(0.0, min(0.05, adjusted))  # Cap at 5%


class SentinelV2:
    """
    Aggregated SENTINEL v2 risk layer.
    Combines all sub-guards and provides a single is_trade_allowed() API.
    """

    def __init__(self):
        self.correlation_guard = CorrelationGuard()
        self.latency_monitor = LatencyMonitor()
        self.cascade_protection = CascadeProtection()
        self.dynamic_sizer = DynamicSizer()

    def is_trade_allowed(
        self,
        direction: str,
        strategy: str = "",
    ) -> tuple[bool, str]:
        """Master gate: checks all v2 conditions."""

        # Cascade failure
        if self.cascade_protection.cascade_active:
            return False, "CASCADE_FAILURE_ACTIVE"

        # Latency
        if self.latency_monitor.is_halted:
            return False, f"LATENCY_HALTED (p99={self.latency_monitor.p99:.0f}ms)"

        # Correlation
        can_open, reason = self.correlation_guard.can_open(direction, strategy)
        if not can_open:
            return False, f"CORRELATION_GUARD: {reason}"

        return True, "APPROVED"

    def compute_size(
        self, base_size_pct: float, regime: str = "RANGING",
        current_atr: float = 0.0, avg_atr: float = 0.0,
    ) -> float:
        """Compute regime/vol-adjusted position size."""
        return self.dynamic_sizer.compute_adjusted_size(
            base_size_pct, regime, current_atr, avg_atr
        )
