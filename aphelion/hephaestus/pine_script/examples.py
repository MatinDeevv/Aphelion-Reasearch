"""
HEPHAESTUS — Few-shot Pine Script examples for LLM code-generation.

Each example is a (pine_source, strategy_spec_summary, generated_class) triple
that gets injected into the codegen prompt so the LLM understands the expected
output format.
"""

from __future__ import annotations

EXAMPLES: list[dict[str, str]] = [
    # ── Example 1: Simple EMA Crossover ──────────────────────────────────
    {
        "pine_source": """\
//@version=5
strategy("EMA Cross", overlay=true)
fast = ta.ema(close, 8)
slow = ta.ema(close, 21)
if ta.crossover(fast, slow)
    strategy.entry("Long", strategy.long)
if ta.crossunder(fast, slow)
    strategy.entry("Short", strategy.short)
""",
        "spec_summary": (
            "Buy when EMA(8) crosses above EMA(21); "
            "sell when EMA(8) crosses below EMA(21)."
        ),
        "generated_class": """\
import numpy as np
from abc import ABC, abstractmethod
from aphelion.hephaestus.models import Vote

class BaseARESVoter(ABC):
    tier: str = "COMMANDER"
    weight: int = 10
    @abstractmethod
    def vote(self, bars, context): ...
    @property
    @abstractmethod
    def lookback(self) -> int: ...
    @property
    @abstractmethod
    def name(self) -> str: ...

class EMACrossoverVoter(BaseARESVoter):
    def __init__(self, fast: int = 8, slow: int = 21):
        self._fast = fast
        self._slow = slow

    @property
    def lookback(self) -> int:
        return self._slow + 5

    @property
    def name(self) -> str:
        return f"EMA_CROSSOVER_{self._fast}_{self._slow}"

    def vote(self, bars: np.ndarray, context: dict) -> Vote:
        if len(bars) < self.lookback:
            return Vote(0, 0.0, "INSUFFICIENT_DATA", {})
        try:
            close = bars[:, 4].astype(float)
            if np.any(np.isnan(close)):
                return Vote(0, 0.0, "NAN_IN_DATA", {})
            fast_ema = self._compute_ema(close, self._fast)
            slow_ema = self._compute_ema(close, self._slow)
            crossover  = fast_ema[-1] > slow_ema[-1] and fast_ema[-2] <= slow_ema[-2]
            crossunder = fast_ema[-1] < slow_ema[-1] and fast_ema[-2] >= slow_ema[-2]
            separation = abs(fast_ema[-1] - slow_ema[-1]) / slow_ema[-1]
            confidence = min(0.9, 0.5 + separation * 100)
            if crossover:
                return Vote(1, confidence, "EMA_CROSSOVER_BULL", {"separation": separation})
            elif crossunder:
                return Vote(-1, confidence, "EMA_CROSSOVER_BEAR", {"separation": separation})
            return Vote(0, 0.3, "NO_CROSS", {})
        except Exception as e:
            return Vote(0, 0.0, "ERROR", {"error": str(e)})

    @staticmethod
    def _compute_ema(data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2.0 / (period + 1)
        ema = np.empty_like(data, dtype=np.float64)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema
""",
    },
    # ── Example 2: RSI Reversal ──────────────────────────────────────────
    {
        "pine_source": """\
//@version=5
strategy("RSI Reversal")
rsi = ta.rsi(close, 14)
if ta.crossover(rsi, 30)
    strategy.entry("Long", strategy.long)
if ta.crossunder(rsi, 70)
    strategy.entry("Short", strategy.short)
""",
        "spec_summary": (
            "Buy when RSI(14) crosses above 30 (oversold bounce); "
            "sell when RSI(14) crosses below 70 (overbought reversal)."
        ),
        "generated_class": """\
import numpy as np
from abc import ABC, abstractmethod
from aphelion.hephaestus.models import Vote

class BaseARESVoter(ABC):
    tier: str = "COMMANDER"
    weight: int = 10
    @abstractmethod
    def vote(self, bars, context): ...
    @property
    @abstractmethod
    def lookback(self) -> int: ...
    @property
    @abstractmethod
    def name(self) -> str: ...

class RSIReversalVoter(BaseARESVoter):
    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        self._period = period
        self._oversold = oversold
        self._overbought = overbought

    @property
    def lookback(self) -> int:
        return self._period + 5

    @property
    def name(self) -> str:
        return f"RSI_REVERSAL_{self._period}"

    def vote(self, bars: np.ndarray, context: dict) -> Vote:
        if len(bars) < self.lookback:
            return Vote(0, 0.0, "INSUFFICIENT_DATA", {})
        try:
            close = bars[:, 4].astype(float)
            if np.any(np.isnan(close)):
                return Vote(0, 0.0, "NAN_IN_DATA", {})
            rsi = self._compute_rsi(close, self._period)
            cross_up   = rsi[-1] > self._oversold and rsi[-2] <= self._oversold
            cross_down = rsi[-1] < self._overbought and rsi[-2] >= self._overbought
            if cross_up:
                conf = min(0.85, 0.5 + (self._oversold - rsi[-2]) * 0.01)
                return Vote(1, conf, "RSI_OVERSOLD_CROSS", {"rsi": float(rsi[-1])})
            elif cross_down:
                conf = min(0.85, 0.5 + (rsi[-2] - self._overbought) * 0.01)
                return Vote(-1, conf, "RSI_OVERBOUGHT_CROSS", {"rsi": float(rsi[-1])})
            return Vote(0, 0.2, "NO_RSI_SIGNAL", {})
        except Exception as e:
            return Vote(0, 0.0, "ERROR", {"error": str(e)})

    @staticmethod
    def _compute_rsi(data: np.ndarray, period: int) -> np.ndarray:
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        rsi = np.full(len(data), 50.0)
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss < 1e-10:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
        return rsi
""",
    },
]
