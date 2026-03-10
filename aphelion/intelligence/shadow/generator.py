"""
SHADOW — Synthetic Data Generator
Phase 13 — Engineering Spec v3.0

Generates synthetic market data for stress testing, data augmentation,
and testing under regimes not present in historical data.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SyntheticBar:
    timestamp_idx: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    regime: str = ""


class RegimeSimulator:
    """Generate synthetic price series for specific market regimes."""

    def generate_trending(
        self, n_bars: int, start_price: float = 3000.0,
        direction: int = 1, volatility: float = 0.001, drift: float = 0.0005,
    ) -> List[SyntheticBar]:
        """Generate a trending market sequence."""
        bars = []
        price = start_price
        for i in range(n_bars):
            ret = direction * drift + np.random.normal(0, volatility)
            new_price = price * (1 + ret)
            high = max(price, new_price) * (1 + abs(np.random.normal(0, volatility * 0.3)))
            low = min(price, new_price) * (1 - abs(np.random.normal(0, volatility * 0.3)))
            volume = max(100, np.random.normal(5000, 1000))

            bars.append(SyntheticBar(
                timestamp_idx=i, open=price, high=high, low=low,
                close=new_price, volume=volume,
                regime="TRENDING_BULL" if direction > 0 else "TRENDING_BEAR",
            ))
            price = new_price
        return bars

    def generate_ranging(
        self, n_bars: int, center_price: float = 3000.0,
        amplitude: float = 10.0, volatility: float = 0.0005,
    ) -> List[SyntheticBar]:
        """Generate a mean-reverting ranging market."""
        bars = []
        price = center_price
        for i in range(n_bars):
            mean_reversion = -0.1 * (price - center_price) / center_price
            ret = mean_reversion + np.random.normal(0, volatility)
            new_price = price * (1 + ret)
            new_price = max(center_price - amplitude, min(center_price + amplitude, new_price))
            high = max(price, new_price) + abs(np.random.normal(0, 0.5))
            low = min(price, new_price) - abs(np.random.normal(0, 0.5))
            volume = max(100, np.random.normal(4000, 800))

            bars.append(SyntheticBar(
                timestamp_idx=i, open=price, high=high, low=low,
                close=new_price, volume=volume, regime="RANGING",
            ))
            price = new_price
        return bars

    def generate_volatile(
        self, n_bars: int, start_price: float = 3000.0,
        base_volatility: float = 0.003,
    ) -> List[SyntheticBar]:
        """Generate a high-volatility (news-driven) sequence."""
        bars = []
        price = start_price
        for i in range(n_bars):
            # Occasional spikes
            spike = np.random.choice([1, 3, 5], p=[0.8, 0.15, 0.05])
            vol = base_volatility * spike
            ret = np.random.normal(0, vol)
            new_price = price * (1 + ret)
            high = max(price, new_price) * (1 + abs(np.random.normal(0, vol * 0.5)))
            low = min(price, new_price) * (1 - abs(np.random.normal(0, vol * 0.5)))
            volume = max(200, np.random.normal(8000, 3000))

            bars.append(SyntheticBar(
                timestamp_idx=i, open=price, high=high, low=low,
                close=new_price, volume=volume, regime="VOLATILE",
            ))
            price = new_price
        return bars

    def generate_crisis(
        self, n_bars: int, start_price: float = 3000.0,
        crash_magnitude: float = 0.05,
    ) -> List[SyntheticBar]:
        """Generate a crisis/flash crash scenario."""
        bars = []
        price = start_price

        # Phase 1: crash (first 20% of bars)
        crash_bars = max(1, n_bars // 5)
        for i in range(crash_bars):
            ret = -crash_magnitude / crash_bars + np.random.normal(0, 0.005)
            new_price = price * (1 + ret)
            high = price * 1.002
            low = new_price * 0.998
            volume = max(500, np.random.normal(15000, 5000))
            bars.append(SyntheticBar(
                timestamp_idx=i, open=price, high=high, low=low,
                close=new_price, volume=volume, regime="CRISIS",
            ))
            price = new_price

        # Phase 2: volatile recovery
        for i in range(crash_bars, n_bars):
            ret = np.random.normal(0.001, 0.004)
            new_price = price * (1 + ret)
            high = max(price, new_price) * 1.003
            low = min(price, new_price) * 0.997
            volume = max(300, np.random.normal(10000, 3000))
            bars.append(SyntheticBar(
                timestamp_idx=i, open=price, high=high, low=low,
                close=new_price, volume=volume, regime="CRISIS",
            ))
            price = new_price

        return bars


class StressScenarioGenerator:
    """Generate extreme market scenarios for SENTINEL stress testing."""

    def __init__(self):
        self._simulator = RegimeSimulator()

    def generate_flash_crash(self, n_bars: int = 100) -> List[SyntheticBar]:
        """Sudden 5% drop followed by recovery."""
        return self._simulator.generate_crisis(n_bars, crash_magnitude=0.05)

    def generate_gap_up(self, start_price: float = 3000.0, gap_pct: float = 0.02) -> SyntheticBar:
        """Single bar gap up (e.g., overnight gap)."""
        new_price = start_price * (1 + gap_pct)
        return SyntheticBar(
            timestamp_idx=0, open=new_price, high=new_price * 1.001,
            low=start_price * 0.999, close=new_price * 0.998,
            volume=20000, regime="VOLATILE",
        )

    def generate_frozen_feed(self, n_bars: int = 5, price: float = 3000.0) -> List[SyntheticBar]:
        """Identical bars (frozen feed detection test)."""
        return [
            SyntheticBar(
                timestamp_idx=i, open=price, high=price + 0.01,
                low=price - 0.01, close=price, volume=100, regime="FROZEN",
            )
            for i in range(n_bars)
        ]

    def generate_spread_blowout(
        self, n_bars: int = 20, start_price: float = 3000.0
    ) -> List[Tuple[SyntheticBar, float]]:
        """Bars with abnormally wide spreads."""
        result = []
        price = start_price
        for i in range(n_bars):
            spread = np.random.uniform(2.0, 10.0)  # Normal is 0.35
            ret = np.random.normal(0, 0.002)
            new_price = price * (1 + ret)
            bar = SyntheticBar(
                timestamp_idx=i, open=price, high=max(price, new_price) + spread / 2,
                low=min(price, new_price) - spread / 2, close=new_price,
                volume=max(100, np.random.normal(3000, 1000)), regime="VOLATILE",
            )
            result.append((bar, spread))
            price = new_price
        return result


class ShadowGenerator:
    """
    Main SHADOW entry point.
    Lieutenant-tier (5 votes — generates data, doesn't trade).
    """

    def __init__(self):
        self.regime_sim = RegimeSimulator()
        self.stress = StressScenarioGenerator()

    def generate_mixed_regime(self, n_bars: int = 1000) -> List[SyntheticBar]:
        """Generate a sequence mixing multiple regimes."""
        bars = []
        remaining = n_bars
        price = 3000.0

        while remaining > 0:
            regime = np.random.choice(["trend_bull", "trend_bear", "ranging", "volatile"], 
                                       p=[0.3, 0.2, 0.35, 0.15])
            chunk_size = min(remaining, np.random.randint(50, 200))

            if regime == "trend_bull":
                chunk = self.regime_sim.generate_trending(chunk_size, price, direction=1)
            elif regime == "trend_bear":
                chunk = self.regime_sim.generate_trending(chunk_size, price, direction=-1)
            elif regime == "ranging":
                chunk = self.regime_sim.generate_ranging(chunk_size, price)
            else:
                chunk = self.regime_sim.generate_volatile(chunk_size, price)

            if chunk:
                price = chunk[-1].close
            bars.extend(chunk)
            remaining -= chunk_size

        return bars[:n_bars]
