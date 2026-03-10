"""
APHELION TUI — Live Simulation Engine (Phase 23)

A lightweight background thread that generates realistic simulated
trading data and pumps it into TUIState for live dashboard rendering.

This runs when the user presses [S] (Simulated) from the launcher.
No external dependencies (no MT5, no PaperRunner) — instant startup.
"""

from __future__ import annotations

import math
import random
import threading
import time
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class LiveSimulator:
    """Background thread that generates simulated market + trading data.

    Produces:
    - Realistic XAU/USD price ticks (GBM with mean reversion)
    - HYDRA signals at configurable intervals
    - Simulated trades with fills / stop-outs
    - SENTINEL risk updates
    - Equity curve movements
    """

    def __init__(self, tui_state, config=None):
        self._state = tui_state
        self._config = config
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Simulation params
        self._price = 2650.0  # Starting XAU/USD price
        self._spread = 0.30
        self._volatility = 0.0008  # Per-tick volatility
        self._tick_interval = 0.15  # Seconds between ticks
        self._bar_interval = 5.0    # Seconds per bar (compressed time)

        # Trading state
        self._capital = 10_000.0
        self._equity = 10_000.0
        self._position = None  # {"dir": "LONG"/"SHORT", "entry": float, "size": float, "sl": float, "tp": float}
        self._trade_count = 0
        self._win_count = 0
        self._loss_count = 0
        self._realized_pnl = 0.0
        self._bars_processed = 0
        self._session_peak = 10_000.0

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, capital: float = 10_000.0, symbol: str = "XAUUSD") -> None:
        """Start the simulation in a background thread."""
        if self._running:
            return

        self._capital = capital
        self._equity = capital
        self._session_peak = capital
        self._running = True
        self._stop_event.clear()

        # Init TUI state
        self._state.market_open = True
        self._state.feed_connected = True
        self._state.feed_mode = "SIMULATED"
        self._state.session_start = datetime.now(timezone.utc)
        self._state.price.symbol = symbol
        self._state.equity.account_equity = capital
        self._state.equity.session_peak = capital
        self._state.sentinel.trading_allowed = True
        self._state.push_log("INFO", f"Simulated session started — {symbol} — ${capital:,.0f}")

        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="tui-live-sim",
        )
        self._thread.start()
        logger.info("LiveSimulator started")

    def stop(self) -> None:
        """Stop the simulation."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._running = False
        self._state.market_open = False
        self._state.feed_connected = False
        self._state.push_log("INFO", "Simulated session stopped")
        logger.info("LiveSimulator stopped")

    def _run_loop(self) -> None:
        """Main simulation loop running in background thread."""
        tick_count = 0
        last_bar_time = time.monotonic()
        last_signal_time = time.monotonic()
        last_session_check = time.monotonic()

        try:
            while not self._stop_event.is_set():
                now = time.monotonic()

                # ── Price tick ──
                self._generate_tick()
                tick_count += 1

                # ── Bar aggregation (every bar_interval) ──
                if now - last_bar_time >= self._bar_interval:
                    self._bars_processed += 1
                    self._state.bars_processed = self._bars_processed
                    self._state.last_bar_time = datetime.now(timezone.utc)
                    self._state.current_time = datetime.now(timezone.utc)
                    last_bar_time = now

                    # Session cycling
                    if now - last_session_check >= 30:
                        sessions = ["TOKYO", "LONDON", "NEW_YORK", "OVERLAP"]
                        self._state.current_session = random.choice(sessions)
                        last_session_check = now

                # ── HYDRA signal (every ~10 bars) ──
                if now - last_signal_time >= self._bar_interval * 8:
                    self._generate_hydra_signal()
                    last_signal_time = now

                # ── Trade management every tick ──
                self._manage_position()

                # ── Equity + system stats ──
                self._update_equity()
                self._update_system_stats(tick_count)

                # Sleep
                self._stop_event.wait(self._tick_interval)

        except Exception as exc:
            logger.exception("LiveSimulator error: %s", exc)
            self._state.push_log("ERROR", f"Simulator error: {exc}")
        finally:
            self._running = False

    def _generate_tick(self) -> None:
        """Generate a realistic price tick using geometric Brownian motion."""
        # GBM with mean-reversion toward 2650
        drift = 0.00001 * (2650.0 - self._price)  # Mean reversion
        shock = random.gauss(0, self._volatility)
        self._price *= (1 + drift + shock)
        self._price = max(2500, min(2800, self._price))

        # Dynamic spread
        self._spread = 0.20 + random.uniform(0, 0.30)

        bid = round(self._price - self._spread / 2, 2)
        ask = round(self._price + self._spread / 2, 2)

        # Update state
        self._state.push_price_tick(bid, ask)

        # Calculate change
        if len(self._state.price.tick_history) >= 2:
            history = list(self._state.price.tick_history)
            first = history[0]
            self._state.price.change = self._price - first
            self._state.price.change_pct = (self._price - first) / first * 100 if first > 0 else 0

        # Track high/low
        if self._price > self._state.price.high or self._state.price.high == 0:
            self._state.price.high = self._price
        if self._price < self._state.price.low or self._state.price.low == 0:
            self._state.price.low = self._price

    def _generate_hydra_signal(self) -> None:
        """Generate a simulated HYDRA prediction signal."""
        # Random but somewhat persistent direction
        r = random.random()
        if r < 0.35:
            direction = "LONG"
            probs = [round(random.uniform(0.5, 0.85), 2), round(random.uniform(0.05, 0.20), 2), 0]
            probs[2] = round(1.0 - probs[0] - probs[1], 2)
        elif r < 0.70:
            direction = "SHORT"
            probs = [0, round(random.uniform(0.05, 0.20), 2), round(random.uniform(0.5, 0.85), 2)]
            probs[0] = round(1.0 - probs[1] - probs[2], 2)
        else:
            direction = "FLAT"
            probs = [round(random.uniform(0.2, 0.4), 2), round(random.uniform(0.3, 0.5), 2), 0]
            probs[2] = round(1.0 - probs[0] - probs[1], 2)

        confidence = round(random.uniform(0.45, 0.95), 2)
        uncertainty = round(1.0 - confidence + random.uniform(-0.1, 0.1), 2)
        uncertainty = max(0, min(1, uncertainty))

        hydra = self._state.hydra
        hydra.direction = direction
        hydra.confidence = confidence
        hydra.uncertainty = uncertainty
        hydra.probs_5m = probs
        hydra.probs_15m = [round(p + random.uniform(-0.05, 0.05), 2) for p in probs]
        hydra.probs_1h = [round(p + random.uniform(-0.08, 0.08), 2) for p in probs]
        hydra.horizon_agreement = round(random.uniform(0.4, 1.0), 2)
        hydra.gate_weights = [round(random.uniform(0.15, 0.35), 2) for _ in range(4)]
        hydra.moe_routing = [round(random.uniform(0.15, 0.35), 2) for _ in range(4)]
        hydra.top_features = [
            ("atr_14", round(random.uniform(0.05, 0.20), 3)),
            ("rsi_14", round(random.uniform(0.03, 0.15), 3)),
            ("macd_hist", round(random.uniform(0.02, 0.12), 3)),
            ("vwap_dev", round(random.uniform(0.01, 0.10), 3)),
        ]
        hydra.timestamp = datetime.now(timezone.utc)
        hydra.confidence_history.append(confidence)
        hydra.signal_count += 1

        self._state.push_log("HYDRA", f"Signal: {direction} conf={confidence:.0%} agree={hydra.horizon_agreement:.0%}")

        # Maybe open a trade
        if direction != "FLAT" and confidence > 0.60 and self._position is None:
            self._open_trade(direction, confidence)

    def _open_trade(self, direction: str, confidence: float) -> None:
        """Open a simulated trade."""
        if not self._state.sentinel.trading_allowed:
            self._state.push_log("REJECT", "SENTINEL blocked: trading not allowed")
            self._state.sentinel_rejections += 1
            return

        if len(self._state.positions) >= self._state.sentinel.max_positions:
            self._state.push_log("REJECT", f"SENTINEL blocked: max positions ({self._state.sentinel.max_positions})")
            self._state.sentinel_rejections += 1
            return

        # Calculate position
        risk_amount = self._equity * 0.01  # 1% risk
        entry = self._price
        atr = self._price * 0.003  # ~$8 ATR on gold

        if direction == "LONG":
            sl = round(entry - atr * 1.5, 2)
            tp = round(entry + atr * 2.5, 2)
        else:
            sl = round(entry + atr * 1.5, 2)
            tp = round(entry - atr * 2.5, 2)

        pip_risk = abs(entry - sl)
        lots = round(risk_amount / (pip_risk * 100), 2)
        lots = max(0.01, min(lots, 0.50))

        self._position = {
            "dir": direction,
            "entry": entry,
            "size": lots,
            "sl": sl,
            "tp": tp,
            "open_time": datetime.now(timezone.utc),
        }

        from aphelion.tui.state import PositionView
        pos_view = PositionView(
            position_id=f"SIM-{self._trade_count + 1:04d}",
            symbol=self._state.price.symbol,
            direction=direction,
            entry_price=entry,
            current_price=entry,
            stop_loss=sl,
            take_profit=tp,
            size_lots=lots,
            unrealized_pnl=0.0,
            open_time=datetime.now(timezone.utc),
        )
        self._state.positions.append(pos_view)
        self._state.sentinel.open_positions = len(self._state.positions)

        self._state.push_log("FILL", f"Opened {direction} {lots:.2f} lots @ {entry:.2f}  SL={sl:.2f}  TP={tp:.2f}")
        self._trade_count += 1

    def _manage_position(self) -> None:
        """Check SL/TP hits and manage open position."""
        if self._position is None:
            return

        pos = self._position
        entry = pos["entry"]
        sl = pos["sl"]
        tp = pos["tp"]
        lots = pos["size"]

        # P&L calculation
        if pos["dir"] == "LONG":
            pnl = (self._price - entry) * lots * 100
            hit_sl = self._price <= sl
            hit_tp = self._price >= tp
        else:
            pnl = (entry - self._price) * lots * 100
            hit_sl = self._price >= sl
            hit_tp = self._price <= tp

        # Update position view
        if self._state.positions:
            pv = self._state.positions[-1]
            pv.current_price = self._price
            pv.unrealized_pnl = round(pnl, 2)

        if hit_sl or hit_tp:
            self._close_trade(pnl, "TP" if hit_tp else "SL")

    def _close_trade(self, pnl: float, reason: str) -> None:
        """Close the current trade."""
        pos = self._position
        if pos is None:
            return

        self._realized_pnl += pnl
        self._equity += pnl

        if pnl > 0:
            self._win_count += 1
            self._state.equity.winning_trades += 1
            self._state.equity.consecutive_wins += 1
            self._state.equity.consecutive_losses = 0
            style = "bright_green"
        else:
            self._loss_count += 1
            self._state.equity.losing_trades += 1
            self._state.equity.consecutive_losses += 1
            self._state.equity.consecutive_wins = 0
            style = "bright_red"

        self._state.equity.total_trades += 1
        self._state.equity.realized_pnl = round(self._realized_pnl, 2)

        # Update best/worst
        if pnl > self._state.equity.best_trade:
            self._state.equity.best_trade = pnl
        if pnl < self._state.equity.worst_trade:
            self._state.equity.worst_trade = pnl

        self._state.push_log(
            "FILL",
            f"Closed {pos['dir']} {pos['size']:.2f} lots — {reason} — PnL: ${pnl:+,.2f}",
        )

        # Remove from positions list
        if self._state.positions:
            self._state.positions.pop()
        self._state.sentinel.open_positions = len(self._state.positions)
        self._position = None

    def _update_equity(self) -> None:
        """Update equity-related TUI state."""
        unrealized = 0.0
        if self._position and self._state.positions:
            unrealized = self._state.positions[-1].unrealized_pnl

        self._equity = self._capital + self._realized_pnl + unrealized
        if self._equity > self._session_peak:
            self._session_peak = self._equity

        self._state.equity.account_equity = round(self._equity, 2)
        self._state.equity.session_peak = round(self._session_peak, 2)
        self._state.equity.daily_pnl = round(self._realized_pnl + unrealized, 2)
        self._state.equity.unrealized_pnl = round(unrealized, 2)

        # Push to equity history every few ticks
        self._state.push_equity_tick(self._equity)

        # Sentinel updates
        if self._session_peak > 0:
            dd = (self._session_peak - self._equity) / self._session_peak
            self._state.sentinel.daily_drawdown_pct = round(dd, 4)
            self._state.sentinel.drawdown_history.append(dd)

            # Check circuit breaker
            if dd > self._state.sentinel.max_exposure_pct:
                if not self._state.sentinel.circuit_breaker_active:
                    self._state.sentinel.circuit_breaker_active = True
                    self._state.sentinel.trading_allowed = False
                    self._state.push_log("SENTINEL", "CIRCUIT BREAKER ENGAGED — max drawdown exceeded")
                    self._state.push_alert("CRITICAL", "CIRCUIT BREAKER", "Max drawdown exceeded — trading halted")

        # Exposure
        total_exposure = 0.0
        for p in self._state.positions:
            total_exposure += p.size_lots * p.current_price / self._equity if self._equity > 0 else 0
        self._state.sentinel.total_exposure_pct = round(total_exposure, 4)

        # Calculate Sharpe (simplified)
        hist = list(self._state.equity.equity_history)
        if len(hist) > 20:
            returns = [(hist[i] - hist[i-1]) / hist[i-1] for i in range(1, len(hist)) if hist[i-1] > 0]
            if returns:
                mean_r = sum(returns) / len(returns)
                std_r = (sum((r - mean_r)**2 for r in returns) / len(returns)) ** 0.5
                self._state.equity.sharpe_ratio = round(mean_r / std_r * (252 ** 0.5) if std_r > 0 else 0, 2)

        # Profit factor
        wins_total = self._state.equity.winning_trades * abs(self._state.equity.avg_win) if self._state.equity.avg_win else max(0, self._realized_pnl)
        losses_total = self._state.equity.losing_trades * abs(self._state.equity.avg_loss) if self._state.equity.avg_loss else max(0, -self._realized_pnl)
        if losses_total > 0:
            self._state.equity.profit_factor = round(wins_total / losses_total, 2)

    def _update_system_stats(self, tick_count: int) -> None:
        """Update system performance metrics."""
        # Calculate uptime from session start
        now = time.monotonic()
        if self._state.session_start:
            import datetime
            uptime = (datetime.datetime.now(datetime.timezone.utc) - self._state.session_start).total_seconds()
            self._state.uptime_seconds = uptime
        else:
            self._state.uptime_seconds = 0

        # Approximate system stats
        self._state.feed_ticks_per_min = tick_count / max(1, self._bars_processed * self._bar_interval / 60)
        self._state.latency_ms = round(random.uniform(1, 8), 1)
        self._state.cpu_usage = round(random.uniform(5, 25), 1)
        self._state.memory_mb = round(random.uniform(180, 350), 0)
