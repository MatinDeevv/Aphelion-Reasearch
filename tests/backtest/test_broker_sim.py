import numpy as np
import pytest
from datetime import datetime, timezone

from aphelion.backtest.broker_sim import BrokerConfig, BrokerSimulator
from aphelion.backtest.order import Order, OrderType, OrderSide, OrderStatus
from aphelion.core.config import Timeframe
from aphelion.core.data_layer import Bar
from aphelion.core.event_bus import EventBus
from aphelion.core.clock import MarketClock
from aphelion.risk.sentinel.core import SentinelCore, Position
from aphelion.risk.sentinel.validator import TradeValidator
from aphelion.risk.sentinel.circuit_breaker import CircuitBreaker


def _make_bar(close=2850.0, high=2855.0, low=2845.0, open_=2850.0, ts_offset=0):
    return Bar(
        timestamp=datetime.fromtimestamp(1704067200.0 + ts_offset, tz=timezone.utc),
        timeframe=Timeframe.M1, open=open_, high=high, low=low,
        close=close, volume=100.0, tick_volume=100, spread=0.20, is_complete=True,
    )


def _make_broker():
    bus = EventBus()
    clock = MarketClock()
    # FIXED: Set simulated time to a known market-open period (Monday 12:00 UTC)
    clock.set_simulated_time(datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc))
    core = SentinelCore(bus, clock)
    core.update_equity(10000.0)
    validator = TradeValidator(core, clock)
    rng = np.random.default_rng(42)
    config = BrokerConfig()
    broker = BrokerSimulator(config, validator, core, rng)
    return broker, core, validator


def _make_order(side=OrderSide.BUY, sl=2840.0, tp=2870.0, size_pct=0.02, order_id="test-001"):
    return Order(
        order_id=order_id, symbol="XAUUSD", order_type=OrderType.MARKET,
        side=side, size_lots=0.01, entry_price=0.0,
        stop_loss=sl, take_profit=tp, size_pct=size_pct, proposed_by="TEST",
    )


class TestBrokerSim:
    def test_buy_market_order_filled_above_close(self):
        broker, core, _ = _make_broker()
        bar = _make_bar(close=2850.0)
        order = _make_order(side=OrderSide.BUY)
        filled_order, fill = broker.submit_market_order(order, bar, 10000.0)
        assert fill is not None
        assert fill.filled_price > 2850.0
        assert fill.filled_price < 2851.0

    def test_sell_market_order_filled_below_close(self):
        broker, core, _ = _make_broker()
        bar = _make_bar(close=2850.0)
        order = _make_order(side=OrderSide.SELL, sl=2860.0, tp=2830.0)
        filled_order, fill = broker.submit_market_order(order, bar, 10000.0)
        assert fill is not None
        assert fill.filled_price < 2850.0

    def test_commission_computed_correctly(self):
        broker, core, _ = _make_broker()
        bar = _make_bar(close=2850.0)
        order = Order(
            order_id="c-001", symbol="XAUUSD", order_type=OrderType.MARKET,
            side=OrderSide.BUY, size_lots=0.10, entry_price=0.0,
            stop_loss=2840.0, take_profit=2870.0, size_pct=0.02, proposed_by="TEST",
        )
        _, fill = broker.submit_market_order(order, bar, 10000.0)
        assert fill is not None
        assert fill.commission == pytest.approx(0.70, abs=0.01)  # 7.0 * 0.10

    def test_sentinel_blocks_order_when_l3_triggered(self):
        broker, core, _ = _make_broker()
        core.update_equity(10000.0)
        core.update_equity(8900.0)  # >10% drawdown triggers L3
        assert core.l3_triggered
        bar = _make_bar(close=2850.0)
        order = _make_order()
        filled_order, fill = broker.submit_market_order(order, bar, 8900.0)
        assert fill is None
        assert filled_order.status == OrderStatus.REJECTED

    def test_sentinel_blocks_order_with_no_stop_loss(self):
        broker, core, _ = _make_broker()
        bar = _make_bar(close=2850.0)
        order = _make_order(sl=0.0)
        filled_order, fill = broker.submit_market_order(order, bar, 10000.0)
        assert fill is None
        assert filled_order.status == OrderStatus.REJECTED

    def test_sentinel_blocks_order_with_bad_rr(self):
        broker, core, _ = _make_broker()
        bar = _make_bar(close=2850.0)
        # SL = 2840 (risk=10), TP = 2855 (reward=5) -> RR=0.5 < 1.5
        order = _make_order(sl=2840.0, tp=2855.0)
        filled_order, fill = broker.submit_market_order(order, bar, 10000.0)
        assert fill is None
        assert filled_order.status == OrderStatus.REJECTED

    def test_sl_hit_long_returns_correct_exit(self):
        broker, core, _ = _make_broker()
        pos = Position(
            position_id="sl-001", symbol="XAUUSD", direction="LONG",
            entry_price=2850.0, stop_loss=2840.0, take_profit=2870.0,
            size_lots=0.01, size_pct=0.02, open_time=datetime.now(timezone.utc),
        )
        bar = _make_bar(close=2838.0, low=2835.0, open_=2842.0)
        exits = broker.check_sl_tp([pos], bar)
        assert len(exits) == 1
        pid, exit_price, reason = exits[0]
        assert reason == "SL_HIT"
        assert exit_price <= 2840.0  # SL or worse (slippage)

    def test_sl_hit_short_returns_correct_exit(self):
        broker, core, _ = _make_broker()
        pos = Position(
            position_id="sl-002", symbol="XAUUSD", direction="SHORT",
            entry_price=2850.0, stop_loss=2860.0, take_profit=2830.0,
            size_lots=0.01, size_pct=0.02, open_time=datetime.now(timezone.utc),
        )
        bar = _make_bar(close=2862.0, high=2865.0, open_=2858.0)
        exits = broker.check_sl_tp([pos], bar)
        assert len(exits) == 1
        pid, exit_price, reason = exits[0]
        assert reason == "SL_HIT"
        assert exit_price >= 2860.0  # SL or worse (slippage)

    def test_tp_hit_long_exact_price(self):
        broker, core, _ = _make_broker()
        pos = Position(
            position_id="tp-001", symbol="XAUUSD", direction="LONG",
            entry_price=2850.0, stop_loss=2840.0, take_profit=2870.0,
            size_lots=0.01, size_pct=0.02, open_time=datetime.now(timezone.utc),
        )
        bar = _make_bar(close=2872.0, high=2875.0, low=2865.0, open_=2868.0)
        exits = broker.check_sl_tp([pos], bar)
        assert len(exits) == 1
        pid, exit_price, reason = exits[0]
        assert reason == "TP_HIT"
        assert exit_price == 2870.0  # Exact TP, no slippage

    def test_gap_open_below_sl_long(self):
        broker, core, _ = _make_broker()
        pos = Position(
            position_id="gap-001", symbol="XAUUSD", direction="LONG",
            entry_price=2850.0, stop_loss=2840.0, take_profit=2870.0,
            size_lots=0.01, size_pct=0.02, open_time=datetime.now(timezone.utc),
        )
        # Gap open BELOW stop loss
        bar = _make_bar(close=2832.0, high=2835.0, low=2830.0, open_=2835.0)
        exits = broker.check_sl_tp([pos], bar)
        assert len(exits) == 1
        pid, exit_price, reason = exits[0]
        assert reason == "SL_HIT"
        assert exit_price == 2835.0  # Exits at open, not at SL

    def test_limit_buy_fills_when_low_touches_price(self):
        broker, core, _ = _make_broker()
        order = Order(
            order_id="lim-001", symbol="XAUUSD", order_type=OrderType.LIMIT,
            side=OrderSide.BUY, size_lots=0.01, entry_price=2845.0,
            stop_loss=2835.0, take_profit=2870.0, size_pct=0.02, proposed_by="TEST",
        )
        bar = _make_bar(close=2848.0, low=2844.0)
        results = broker.check_pending_orders([order], bar, 10000.0)
        assert len(results) == 1
        filled_order, fill = results[0]
        assert fill is not None
        assert filled_order.status == OrderStatus.FILLED

    def test_limit_order_cancelled_after_expiry(self):
        broker, core, _ = _make_broker()
        order = Order(
            order_id="exp-001", symbol="XAUUSD", order_type=OrderType.LIMIT,
            side=OrderSide.BUY, size_lots=0.01, entry_price=2800.0,
            stop_loss=2790.0, take_profit=2830.0, size_pct=0.02,
            proposed_by="TEST", expiry_bars=3,
        )
        # 4 bars without touching 2800
        for i in range(4):
            bar = _make_bar(close=2850.0, low=2845.0, ts_offset=i*60)
            results = broker.check_pending_orders([order], bar, 10000.0)
            if order.status != OrderStatus.PENDING:
                break
        assert order.status == OrderStatus.EXPIRED

    def test_broker_stats_track_fills_and_rejections(self):
        broker, core, _ = _make_broker()
        bar = _make_bar(close=2850.0)
        # 3 valid fills
        for i in range(3):
            order = _make_order(order_id=f"ok-{i}")
            broker.submit_market_order(order, bar, 10000.0)
        # Now at max positions (3) - next should be rejected
        order_bad = _make_order(order_id="rej-1")
        broker.submit_market_order(order_bad, bar, 10000.0)
        order_bad2 = _make_order(order_id="rej-2")
        broker.submit_market_order(order_bad2, bar, 10000.0)

        stats = broker.stats
        assert stats["fill_count"] == 3
        assert stats["rejection_count"] == 2
