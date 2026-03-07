"""
APHELION Market Structure Features
Order Blocks, Fair Value Gaps, Swing Detection, Liquidity Pools,
Breaker Blocks, Volume Imbalances, Change of Character.
Section 5.2 of the Engineering Spec.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class OrderBlock:
    index: int
    price_high: float
    price_low: float
    direction: str  # "BULLISH" or "BEARISH"
    strength: float
    still_valid: bool = True


@dataclass
class FairValueGap:
    index: int
    gap_high: float
    gap_low: float
    direction: str  # "BULLISH" or "BEARISH"
    filled: bool = False


@dataclass
class SwingPoint:
    index: int
    price: float
    type: str  # "HIGH" or "LOW"


@dataclass
class BreakerBlock:
    """Order block invalidated by price — now flipped as reversal zone."""
    index: int
    price_high: float
    price_low: float
    direction: str  # "BULLISH" or "BEARISH" (flipped direction)
    broken_at_index: int


@dataclass
class LiquidityPool:
    price_level: float
    count: int
    type: str  # "EQUAL_HIGHS" or "EQUAL_LOWS"


class MarketStructureEngine:
    """Computes smart money concept features from OHLCV data."""

    def __init__(self, swing_confirmation: int = 5,
                 fvg_min_gap_pips: float = 1.0,
                 liquidity_tolerance_pips: float = 5.0,
                 volume_imbalance_mult: float = 2.0,
                 pip_size: float = 0.01):
        self._swing_bars = swing_confirmation
        self._fvg_min_gap = fvg_min_gap_pips * pip_size
        self._liq_tolerance = liquidity_tolerance_pips * pip_size
        self._vol_imbalance_mult = volume_imbalance_mult
        self._pip_size = pip_size

    def detect_swing_highs(self, highs: np.ndarray, n: int = None) -> list[SwingPoint]:
        """Detect swing highs using N-bar fractal confirmation."""
        n = n or self._swing_bars
        swings = []
        for i in range(n, len(highs) - n):
            if all(highs[i] > highs[i - j] for j in range(1, n + 1)) and \
               all(highs[i] > highs[i + j] for j in range(1, n + 1)):
                swings.append(SwingPoint(index=i, price=highs[i], type="HIGH"))
        return swings

    def detect_swing_lows(self, lows: np.ndarray, n: int = None) -> list[SwingPoint]:
        """Detect swing lows using N-bar fractal confirmation."""
        n = n or self._swing_bars
        swings = []
        for i in range(n, len(lows) - n):
            if all(lows[i] < lows[i - j] for j in range(1, n + 1)) and \
               all(lows[i] < lows[i + j] for j in range(1, n + 1)):
                swings.append(SwingPoint(index=i, price=lows[i], type="LOW"))
        return swings

    def detect_order_blocks(self, opens: np.ndarray, highs: np.ndarray,
                            lows: np.ndarray, closes: np.ndarray) -> list[OrderBlock]:
        """
        Detect order blocks: last opposite candle before an impulse move.
        Bullish OB = last bearish candle before bullish impulse.
        """
        obs = []
        for i in range(1, len(closes) - 1):
            # Bullish OB: bearish candle (close < open) followed by bullish impulse
            if closes[i] < opens[i]:  # Bearish candle
                if i + 1 < len(closes) and closes[i + 1] > highs[i]:  # Bullish impulse
                    impulse_size = closes[i + 1] - opens[i + 1]
                    obs.append(OrderBlock(
                        index=i,
                        price_high=highs[i],
                        price_low=lows[i],
                        direction="BULLISH",
                        strength=abs(impulse_size),
                    ))

            # Bearish OB: bullish candle followed by bearish impulse
            if closes[i] > opens[i]:  # Bullish candle
                if i + 1 < len(closes) and closes[i + 1] < lows[i]:  # Bearish impulse
                    impulse_size = opens[i + 1] - closes[i + 1]
                    obs.append(OrderBlock(
                        index=i,
                        price_high=highs[i],
                        price_low=lows[i],
                        direction="BEARISH",
                        strength=abs(impulse_size),
                    ))

        return obs

    def detect_fair_value_gaps(self, highs: np.ndarray,
                                lows: np.ndarray) -> list[FairValueGap]:
        """
        Detect Fair Value Gaps (FVG): 3-bar pattern where there's a gap
        between candle 1 high and candle 3 low (or vice versa).
        """
        fvgs = []
        for i in range(2, len(highs)):
            # Bullish FVG: gap between candle 1 high and candle 3 low
            gap = lows[i] - highs[i - 2]
            if gap > self._fvg_min_gap:
                fvgs.append(FairValueGap(
                    index=i - 1,
                    gap_high=lows[i],
                    gap_low=highs[i - 2],
                    direction="BULLISH",
                ))

            # Bearish FVG: gap between candle 3 high and candle 1 low
            gap = lows[i - 2] - highs[i]
            if gap > self._fvg_min_gap:
                fvgs.append(FairValueGap(
                    index=i - 1,
                    gap_high=lows[i - 2],
                    gap_low=highs[i],
                    direction="BEARISH",
                ))

        return fvgs

    def detect_liquidity_pools(self, highs: np.ndarray,
                                lows: np.ndarray) -> list[LiquidityPool]:
        """Detect equal highs/lows within tolerance (stop hunt targets)."""
        pools = []

        # Equal highs
        for i in range(len(highs)):
            count = 1
            for j in range(i + 1, len(highs)):
                if abs(highs[i] - highs[j]) <= self._liq_tolerance:
                    count += 1
            if count >= 2:
                pools.append(LiquidityPool(
                    price_level=highs[i],
                    count=count,
                    type="EQUAL_HIGHS",
                ))

        # Equal lows
        for i in range(len(lows)):
            count = 1
            for j in range(i + 1, len(lows)):
                if abs(lows[i] - lows[j]) <= self._liq_tolerance:
                    count += 1
            if count >= 2:
                pools.append(LiquidityPool(
                    price_level=lows[i],
                    count=count,
                    type="EQUAL_LOWS",
                ))

        # Deduplicate nearby levels
        return self._deduplicate_pools(pools)

    def _deduplicate_pools(self, pools: list[LiquidityPool]) -> list[LiquidityPool]:
        if not pools:
            return []
        pools.sort(key=lambda p: p.price_level)
        deduped = [pools[0]]
        for pool in pools[1:]:
            if abs(pool.price_level - deduped[-1].price_level) > self._liq_tolerance:
                deduped.append(pool)
            elif pool.count > deduped[-1].count:
                deduped[-1] = pool
        return deduped

    def detect_volume_imbalance(self, closes: np.ndarray, opens: np.ndarray,
                                 volumes: np.ndarray) -> np.ndarray:
        """
        Detect volume imbalances: candle with >2x average volume
        on a directional move. Returns boolean array.
        """
        if len(volumes) < 20:
            return np.zeros(len(volumes), dtype=bool)

        avg_vol = pd.Series(volumes).rolling(20).mean().values
        directional = np.abs(closes - opens) > 0

        imbalance = (volumes > self._vol_imbalance_mult * avg_vol) & directional
        imbalance[:20] = False
        return imbalance

    def detect_breaker_blocks(self, opens: np.ndarray, highs: np.ndarray,
                              lows: np.ndarray, closes: np.ndarray) -> list[BreakerBlock]:
        """
        Detect breaker blocks: order blocks that price has broken through,
        flipping them into high-probability reversal zones.
        A bullish OB broken to the downside becomes a bearish breaker.
        A bearish OB broken to the upside becomes a bullish breaker.
        """
        order_blocks = self.detect_order_blocks(opens, highs, lows, closes)
        breakers = []

        for ob in order_blocks:
            # Check if price subsequently broke through the OB zone
            for i in range(ob.index + 2, len(closes)):
                if ob.direction == "BULLISH":
                    # Bullish OB invalidated when price closes below its low
                    if closes[i] < ob.price_low:
                        breakers.append(BreakerBlock(
                            index=ob.index,
                            price_high=ob.price_high,
                            price_low=ob.price_low,
                            direction="BEARISH",  # Flipped
                            broken_at_index=i,
                        ))
                        break
                elif ob.direction == "BEARISH":
                    # Bearish OB invalidated when price closes above its high
                    if closes[i] > ob.price_high:
                        breakers.append(BreakerBlock(
                            index=ob.index,
                            price_high=ob.price_high,
                            price_low=ob.price_low,
                            direction="BULLISH",  # Flipped
                            broken_at_index=i,
                        ))
                        break

        return breakers

    def detect_change_of_character(self, highs: np.ndarray, lows: np.ndarray,
                                    closes: np.ndarray) -> list[dict]:
        """
        Change of Character (CHoCH): first candle to close above/below
        a prior swing high/low.
        """
        swing_highs = self.detect_swing_highs(highs)
        swing_lows = self.detect_swing_lows(lows)
        choch_signals = []

        for sh in swing_highs:
            for i in range(sh.index + 1, len(closes)):
                if closes[i] > sh.price:
                    choch_signals.append({
                        "index": i,
                        "type": "BULLISH_CHOCH",
                        "broken_level": sh.price,
                        "swing_index": sh.index,
                    })
                    break

        for sl in swing_lows:
            for i in range(sl.index + 1, len(closes)):
                if closes[i] < sl.price:
                    choch_signals.append({
                        "index": i,
                        "type": "BEARISH_CHOCH",
                        "broken_level": sl.price,
                        "swing_index": sl.index,
                    })
                    break

        return sorted(choch_signals, key=lambda x: x["index"])

    def compute_all(self, df: pd.DataFrame) -> dict:
        """Compute all market structure features from OHLCV DataFrame."""
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        volumes = df["volume"].values if "volume" in df.columns else np.ones(len(df))

        swing_highs = self.detect_swing_highs(highs)
        swing_lows = self.detect_swing_lows(lows)
        order_blocks = self.detect_order_blocks(opens, highs, lows, closes)
        fvgs = self.detect_fair_value_gaps(highs, lows)
        liq_pools = self.detect_liquidity_pools(highs, lows)
        vol_imbalance = self.detect_volume_imbalance(closes, opens, volumes)
        choch = self.detect_change_of_character(highs, lows, closes)
        breaker_blocks = self.detect_breaker_blocks(opens, highs, lows, closes)

        # Compute nearest features for current bar
        last_idx = len(closes) - 1
        last_price = closes[-1] if len(closes) > 0 else 0

        nearest_ob_dist = float('inf')
        nearest_ob_dir = "NONE"
        for ob in reversed(order_blocks):
            if ob.still_valid:
                dist = min(abs(last_price - ob.price_high), abs(last_price - ob.price_low))
                if dist < nearest_ob_dist:
                    nearest_ob_dist = dist
                    nearest_ob_dir = ob.direction

        nearest_fvg_dist = float('inf')
        for fvg in reversed(fvgs):
            if not fvg.filled:
                dist = min(abs(last_price - fvg.gap_high), abs(last_price - fvg.gap_low))
                if dist < nearest_fvg_dist:
                    nearest_fvg_dist = dist

        # Nearest breaker block
        nearest_breaker_dist = float('inf')
        nearest_breaker_dir = "NONE"
        for bb in reversed(breaker_blocks):
            dist = min(abs(last_price - bb.price_high), abs(last_price - bb.price_low))
            if dist < nearest_breaker_dist:
                nearest_breaker_dist = dist
                nearest_breaker_dir = bb.direction

        return {
            "swing_high_count": len(swing_highs),
            "swing_low_count": len(swing_lows),
            "nearest_swing_high": swing_highs[-1].price if swing_highs else 0,
            "nearest_swing_low": swing_lows[-1].price if swing_lows else 0,
            "order_block_count": len(order_blocks),
            "nearest_ob_distance": nearest_ob_dist if nearest_ob_dist != float('inf') else 0,
            "nearest_ob_direction": nearest_ob_dir,
            "fvg_count": len(fvgs),
            "nearest_fvg_distance": nearest_fvg_dist if nearest_fvg_dist != float('inf') else 0,
            "liquidity_pool_count": len(liq_pools),
            "volume_imbalance": bool(vol_imbalance[-1]) if len(vol_imbalance) > 0 else False,
            "choch_count": len(choch),
            "last_choch_type": choch[-1]["type"] if choch else "NONE",
            "breaker_block_count": len(breaker_blocks),
            "nearest_breaker_distance": nearest_breaker_dist if nearest_breaker_dist != float('inf') else 0,
            "nearest_breaker_direction": nearest_breaker_dir,
        }
