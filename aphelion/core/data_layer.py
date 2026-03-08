"""
APHELION Data Layer
MT5 tick stream → clean OHLCV bars with data quality validation.
Includes gap detection, staleness tracking, configurable thresholds.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from aphelion.core.config import Timeframe, SYMBOL, EventTopic
from aphelion.core.event_bus import EventBus, Event, Priority


@dataclass
class Tick:
    timestamp: float
    bid: float
    ask: float
    last: float
    volume: float
    flags: int = 0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class Bar:
    timestamp: datetime
    timeframe: Timeframe
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_volume: int
    spread: float
    is_complete: bool = False

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "timeframe": self.timeframe.value,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "tick_volume": self.tick_volume,
            "spread": self.spread,
        }


class BarAggregator:
    """Aggregates ticks into OHLCV bars for a single timeframe."""

    TIMEFRAME_SECONDS = {
        Timeframe.M1: 60,
        Timeframe.M5: 300,
        Timeframe.M15: 900,
        Timeframe.H1: 3600,
    }

    def __init__(self, timeframe: Timeframe):
        self.timeframe = timeframe
        self._interval = self.TIMEFRAME_SECONDS[timeframe]
        self._current_bar: Optional[Bar] = None
        self._tick_count = 0

    def _bar_start_time(self, timestamp: float) -> datetime:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        bar_seconds = (total_seconds // self._interval) * self._interval
        bar_dt = dt.replace(
            hour=bar_seconds // 3600,
            minute=(bar_seconds % 3600) // 60,
            second=0,
            microsecond=0,
        )
        return bar_dt

    def process_tick(self, tick: Tick) -> Optional[Bar]:
        """Process a tick. Returns completed bar if bar boundary crossed."""
        bar_time = self._bar_start_time(tick.timestamp)
        completed = None

        if self._current_bar is None:
            self._current_bar = Bar(
                timestamp=bar_time,
                timeframe=self.timeframe,
                open=tick.last,
                high=tick.last,
                low=tick.last,
                close=tick.last,
                volume=tick.volume,
                tick_volume=1,
                spread=tick.spread,
            )
        elif bar_time > self._current_bar.timestamp:
            # New bar — complete the old one
            self._current_bar.is_complete = True
            completed = self._current_bar

            self._current_bar = Bar(
                timestamp=bar_time,
                timeframe=self.timeframe,
                open=tick.last,
                high=tick.last,
                low=tick.last,
                close=tick.last,
                volume=tick.volume,
                tick_volume=1,
                spread=tick.spread,
            )
        else:
            # Update current bar
            self._current_bar.high = max(self._current_bar.high, tick.last)
            self._current_bar.low = min(self._current_bar.low, tick.last)
            self._current_bar.close = tick.last
            self._current_bar.volume += tick.volume
            self._current_bar.tick_volume += 1
            self._current_bar.spread = tick.spread

        return completed


class DataQualityValidator:
    """Validates data integrity on every bar.

    Configurable thresholds for spread, price jump, and gap detection.
    """

    def __init__(
        self,
        max_spread: float = 50.0,
        max_price_jump_pct: float = 0.05,
        max_gap_seconds: float = 120.0,
    ):
        self._last_prices: deque = deque(maxlen=100)
        self._last_tick_time: float = 0.0
        self._gap_count = 0
        self._invalid_count = 0
        self._max_spread = max_spread
        self._max_price_jump_pct = max_price_jump_pct
        self._max_gap_seconds = max_gap_seconds

    def validate_tick(self, tick: Tick) -> tuple[bool, Optional[str]]:
        if tick.bid <= 0 or tick.ask <= 0:
            self._invalid_count += 1
            return False, "Non-positive price"

        if tick.ask < tick.bid:
            self._invalid_count += 1
            return False, "Ask < Bid (crossed spread)"

        if tick.spread > self._max_spread:
            self._invalid_count += 1
            return False, f"Spread too wide: {tick.spread}"

        if self._last_prices:
            last = self._last_prices[-1]
            pct_change = abs(tick.last - last) / last
            if pct_change > self._max_price_jump_pct:
                self._invalid_count += 1
                return False, f"Price jump: {pct_change:.2%}"

        # Gap detection: detect time gaps between ticks
        if self._last_tick_time > 0:
            gap = tick.timestamp - self._last_tick_time
            if gap > self._max_gap_seconds:
                self._gap_count += 1
                logger.warning("Data gap detected: %.1fs between ticks", gap)
        self._last_tick_time = tick.timestamp

        self._last_prices.append(tick.last)
        return True, None

    def validate_bar(self, bar: Bar) -> tuple[bool, Optional[str]]:
        if bar.high < bar.low:
            return False, "High < Low"
        if bar.open > bar.high or bar.open < bar.low:
            return False, "Open outside H/L range"
        if bar.close > bar.high or bar.close < bar.low:
            return False, "Close outside H/L range"
        if bar.volume < 0:
            return False, "Negative volume"
        return True, None

    @property
    def stats(self) -> dict:
        return {
            "gap_count": self._gap_count,
            "invalid_count": self._invalid_count,
        }


class DataLayer:
    """
    Central data layer. Manages MT5 connection, tick ingestion,
    bar aggregation, and quality validation.
    """

    def __init__(
        self,
        event_bus: EventBus,
        symbol: str = SYMBOL,
        max_spread: float = 50.0,
        max_price_jump_pct: float = 0.05,
        max_gap_seconds: float = 120.0,
    ):
        self.event_bus = event_bus
        self.symbol = symbol
        self._validator = DataQualityValidator(
            max_spread=max_spread,
            max_price_jump_pct=max_price_jump_pct,
            max_gap_seconds=max_gap_seconds,
        )
        self._aggregators = {tf: BarAggregator(tf) for tf in Timeframe}
        self._tick_buffer: deque[Tick] = deque(maxlen=50_000)
        self._bars: dict[Timeframe, deque[Bar]] = {
            tf: deque(maxlen=10_000) for tf in Timeframe
        }
        self._connected = False
        self._tick_count = 0
        self._last_tick_time: float = 0.0  # for staleness detection
        self._mt5 = None

    async def connect(self) -> bool:
        """Connect to MT5 terminal."""
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            if not mt5.initialize():
                return False
            self._connected = True
            return True
        except ImportError:
            # MT5 not available (dev/test mode)
            self._connected = False
            return False

    async def disconnect(self) -> None:
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False

    async def process_tick(self, tick: Tick) -> None:
        """Process incoming tick: validate → aggregate → publish events."""
        valid, error = self._validator.validate_tick(tick)
        if not valid:
            return

        self._tick_buffer.append(tick)
        self._tick_count += 1
        self._last_tick_time = time.time()

        # Publish raw tick event
        self.event_bus.publish_nowait(Event(
            topic=EventTopic.TICK,
            data=tick,
            source="DATA",
            priority=Priority.HIGH,
        ))

        # Aggregate into bars for each timeframe
        for tf, aggregator in self._aggregators.items():
            completed_bar = aggregator.process_tick(tick)
            if completed_bar is not None:
                bar_valid, bar_error = self._validator.validate_bar(completed_bar)
                if bar_valid:
                    self._bars[tf].append(completed_bar)
                    self.event_bus.publish_nowait(Event(
                        topic=EventTopic.BAR,
                        data=completed_bar,
                        source="DATA",
                        priority=Priority.NORMAL,
                    ))

    def get_bars(self, timeframe: Timeframe, count: int = 100) -> list[Bar]:
        """Get the last N bars for a timeframe."""
        bars = self._bars[timeframe]
        return list(bars)[-count:]

    def get_bars_df(self, timeframe: Timeframe, count: int = 100) -> pd.DataFrame:
        """Get bars as a pandas DataFrame."""
        bars = self.get_bars(timeframe, count)
        if not bars:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "tick_volume", "spread"])
        return pd.DataFrame([b.to_dict() for b in bars])

    def get_ticks(self, count: int = 1000) -> list[Tick]:
        """Get the last N raw ticks."""
        return list(self._tick_buffer)[-count:]

    async def load_historical(self, timeframe: Timeframe, count: int = 1000) -> Optional[pd.DataFrame]:
        """Load historical bars from MT5."""
        if not self._connected or self._mt5 is None:
            return None

        tf_map = {
            Timeframe.M1: self._mt5.TIMEFRAME_M1,
            Timeframe.M5: self._mt5.TIMEFRAME_M5,
            Timeframe.M15: self._mt5.TIMEFRAME_M15,
            Timeframe.H1: self._mt5.TIMEFRAME_H1,
        }

        rates = self._mt5.copy_rates_from_pos(self.symbol, tf_map[timeframe], 0, count)
        if rates is None:
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.rename(columns={"time": "timestamp", "real_volume": "volume"}, inplace=True)
        return df

    def load_from_parquet(self, filepath: str, timeframe: Timeframe, populate: bool = True) -> pd.DataFrame:
        """Load bar data from a Parquet file. Optionally populates internal bar buffer."""
        required = {"timestamp", "open", "high", "low", "close", "volume", "tick_volume", "spread"}
        df = pd.read_parquet(filepath)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if populate:
            self._populate_bars(df, timeframe)
        return df

    def load_from_csv(self, filepath: str, timeframe: Timeframe, populate: bool = True) -> pd.DataFrame:
        """Load bar data from a CSV file. Optionally populates internal bar buffer."""
        required = {"timestamp", "open", "high", "low", "close", "volume", "tick_volume", "spread"}
        df = pd.read_csv(filepath)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if populate:
            self._populate_bars(df, timeframe)
        return df

    def _populate_bars(self, df: pd.DataFrame, timeframe: Timeframe) -> None:
        """Convert DataFrame rows into Bar objects and populate internal buffer."""
        bars_deque = self._bars[timeframe]
        for _, row in df.iterrows():
            bar = Bar(
                timestamp=pd.Timestamp(row["timestamp"]),
                timeframe=timeframe,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                tick_volume=int(row.get("tick_volume", 0)),
                spread=float(row.get("spread", 0.0)),
                is_complete=True,
            )
            bars_deque.append(bar)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def staleness_seconds(self) -> float:
        """Seconds since last tick was received. 0 if no ticks yet."""
        if self._last_tick_time <= 0:
            return 0.0
        return time.time() - self._last_tick_time

    def is_stale(self, threshold_seconds: float = 60.0) -> bool:
        """True if no tick received for more than threshold_seconds."""
        if self._last_tick_time <= 0:
            return False  # No data yet, not stale
        return self.staleness_seconds() > threshold_seconds

    @property
    def stats(self) -> dict:
        return {
            "connected": self._connected,
            "tick_count": self._tick_count,
            "buffer_size": len(self._tick_buffer),
            "bars": {tf.value: len(bars) for tf, bars in self._bars.items()},
            "quality": self._validator.stats,
            "staleness_seconds": round(self.staleness_seconds(), 1),
        }
