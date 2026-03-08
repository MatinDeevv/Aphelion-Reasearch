"""
APHELION MT5 Broker Connection
Wraps the MetaTrader5 Python package for live tick/bar data streaming,
account info retrieval, and order execution.

In Phase 5 (paper trading) this provides the LIVE DATA FEED only.
Order execution through MT5 is stubbed for Phase 6 (live trading).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

from aphelion.core.data_layer import Bar, Tick

logger = logging.getLogger(__name__)

# Conditional import — MT5 only available on Windows with terminal installed
try:
    import MetaTrader5 as mt5  # type: ignore[import-untyped]
    HAS_MT5 = True
except ImportError:
    mt5 = None  # type: ignore[assignment]
    HAS_MT5 = False


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MT5Config:
    """MT5 connection configuration."""
    terminal_path: str = ""              # Path to terminal64.exe (auto-detect if empty)
    login: int = 0                       # Account number (0 = use default)
    password: str = ""                   # Account password
    server: str = ""                     # Broker server name
    symbol: str = "XAUUSD"
    timeout_ms: int = 10_000             # Connection timeout
    retry_attempts: int = 3              # Reconnection attempts
    retry_delay_seconds: float = 5.0     # Delay between retries
    tick_buffer_size: int = 1000         # Max ticks to buffer


# ── MT5 Connection ───────────────────────────────────────────────────────────

class MT5Connection:
    """
    Manages connection to MetaTrader5 terminal.
    Provides tick streaming, bar fetching, and account info.
    """

    def __init__(self, config: Optional[MT5Config] = None):
        self._config = config or MT5Config()
        self._connected: bool = False
        self._account_info: Optional[dict] = None
        self._symbol_info: Optional[dict] = None
        self._tick_count: int = 0

    # ── Connection lifecycle ──────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialize MT5 terminal connection."""
        if not HAS_MT5:
            logger.error("MetaTrader5 package not installed. pip install MetaTrader5")
            return False

        kwargs: dict = {}
        if self._config.terminal_path:
            kwargs["path"] = self._config.terminal_path
        if self._config.login:
            kwargs["login"] = self._config.login
        if self._config.password:
            kwargs["password"] = self._config.password
        if self._config.server:
            kwargs["server"] = self._config.server
        kwargs["timeout"] = self._config.timeout_ms

        for attempt in range(1, self._config.retry_attempts + 1):
            if mt5.initialize(**kwargs):
                self._connected = True
                self._load_account_info()
                self._load_symbol_info()
                logger.info(
                    "MT5 connected (attempt %d/%d) — account %s",
                    attempt, self._config.retry_attempts,
                    self._account_info.get("login", "?") if self._account_info else "?",
                )
                return True
            else:
                error = mt5.last_error() if hasattr(mt5, "last_error") else "unknown"
                logger.warning(
                    "MT5 connect attempt %d/%d failed: %s",
                    attempt, self._config.retry_attempts, error,
                )
                if attempt < self._config.retry_attempts:
                    import time
                    time.sleep(self._config.retry_delay_seconds)

        logger.error("MT5 connection failed after %d attempts", self._config.retry_attempts)
        return False

    def disconnect(self) -> None:
        """Shutdown MT5 connection."""
        if HAS_MT5 and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Account info ──────────────────────────────────────────────────────

    def _load_account_info(self) -> None:
        """Load and cache account details."""
        if not HAS_MT5 or not self._connected:
            return
        info = mt5.account_info()
        if info is not None:
            self._account_info = info._asdict()

    def _load_symbol_info(self) -> None:
        """Load and cache symbol details."""
        if not HAS_MT5 or not self._connected:
            return
        # Ensure symbol is visible
        mt5.symbol_select(self._config.symbol, True)
        info = mt5.symbol_info(self._config.symbol)
        if info is not None:
            self._symbol_info = info._asdict()

    def get_account_info(self) -> Optional[dict]:
        """Return cached account info dict."""
        return self._account_info

    def get_symbol_info(self) -> Optional[dict]:
        """Return cached symbol info dict."""
        return self._symbol_info

    # ── Tick data ─────────────────────────────────────────────────────────

    def get_last_tick(self) -> Optional[Tick]:
        """Get the latest tick for the configured symbol."""
        if not HAS_MT5 or not self._connected:
            return None

        tick = mt5.symbol_info_tick(self._config.symbol)
        if tick is None:
            return None

        self._tick_count += 1
        return Tick(
            timestamp=datetime.fromtimestamp(tick.time, tz=timezone.utc),
            bid=tick.bid,
            ask=tick.ask,
            last=tick.last if hasattr(tick, "last") else (tick.bid + tick.ask) / 2,
            volume=tick.volume if hasattr(tick, "volume") else 0.0,
        )

    def get_recent_ticks(self, count: int = 100) -> list[Tick]:
        """Get the last N ticks for the configured symbol."""
        if not HAS_MT5 or not self._connected:
            return []

        ticks = mt5.copy_ticks_from(
            self._config.symbol,
            datetime.now(timezone.utc) - timedelta(minutes=5),
            count,
            mt5.COPY_TICKS_ALL,
        )
        if ticks is None or len(ticks) == 0:
            return []

        result = []
        for t in ticks:
            result.append(Tick(
                timestamp=datetime.fromtimestamp(t[0], tz=timezone.utc),  # time
                bid=t[1],   # bid
                ask=t[2],   # ask
                last=(t[1] + t[2]) / 2,
                volume=t[5] if len(t) > 5 else 0.0,  # volume
            ))
        return result

    # ── Bar data ──────────────────────────────────────────────────────────

    def get_bars(self, timeframe_str: str = "1m", count: int = 100) -> list[Bar]:
        """
        Fetch the last N bars for the given timeframe.

        Args:
            timeframe_str: "1m", "5m", "15m", "1h"
            count: Number of bars to fetch
        """
        if not HAS_MT5 or not self._connected:
            return []

        tf_map = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "1h": mt5.TIMEFRAME_H1,
        }
        mt5_tf = tf_map.get(timeframe_str)
        if mt5_tf is None:
            logger.warning("Unknown timeframe: %s", timeframe_str)
            return []

        rates = mt5.copy_rates_from_pos(self._config.symbol, mt5_tf, 0, count)
        if rates is None or len(rates) == 0:
            return []

        bars = []
        for r in rates:
            bars.append(Bar(
                timestamp=datetime.fromtimestamp(r[0], tz=timezone.utc),
                open=r[1],
                high=r[2],
                low=r[3],
                close=r[4],
                volume=r[5],
                tick_volume=r[6] if len(r) > 6 else 0,
            ))
        return bars

    def get_current_price(self) -> Optional[float]:
        """Get current mid-price for the symbol."""
        tick = self.get_last_tick()
        if tick is None:
            return None
        return (tick.bid + tick.ask) / 2

    # ── Stats ─────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "connected": self._connected,
            "has_mt5": HAS_MT5,
            "tick_count": self._tick_count,
            "symbol": self._config.symbol,
            "account": self._account_info.get("login") if self._account_info else None,
        }
