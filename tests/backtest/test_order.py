"""Tests for aphelion.backtest.order — Order, Fill, BacktestTrade."""

import pytest
from datetime import datetime, timezone

from aphelion.backtest.order import (
    Order, OrderType, OrderSide, OrderStatus,
    Fill, BacktestTrade,
)


class TestOrder:
    def test_order_default_status_is_pending(self):
        o = Order(
            order_id="t-001", symbol="XAUUSD",
            order_type=OrderType.MARKET, side=OrderSide.BUY,
            size_lots=0.01, entry_price=0.0,
            stop_loss=2840.0, take_profit=2870.0,
        )
        assert o.status == OrderStatus.PENDING

    def test_order_side_and_type_enums_have_expected_values(self):
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP.value == "STOP"

    def test_fill_dataclass_fields_all_present(self):
        f = Fill(
            order_id="f-001", symbol="XAUUSD",
            side=OrderSide.BUY, filled_price=2850.5,
            size_lots=0.01, commission=0.07,
            slippage_cost=0.01, fill_time=datetime.now(timezone.utc),
            bar_index=5,
        )
        assert f.order_id == "f-001"
        assert f.bar_index == 5
        assert f.commission == 0.07


class TestBacktestTrade:
    def _make_trade(self, direction="LONG", entry=2850.0, exit_=2870.0,
                    sl=2840.0, tp=2870.0, lots=0.01, commission=0.14):
        if direction == "LONG":
            gross = (exit_ - entry) * lots * 100
        else:
            gross = (entry - exit_) * lots * 100
        return BacktestTrade(
            trade_id="bt-001", symbol="XAUUSD",
            direction=direction,
            entry_price=entry, exit_price=exit_,
            size_lots=lots, size_pct=0.02,
            stop_loss=sl, take_profit=tp,
            entry_time=datetime.now(timezone.utc),
            exit_time=datetime.now(timezone.utc),
            gross_pnl=gross, commission=commission,
            net_pnl=gross - commission,
            exit_reason="TP_HIT", bars_held=10,
            proposed_by="TEST",
            entry_bar_index=50, exit_bar_index=60,
        )

    def test_backtest_trade_r_multiple_long_2r(self):
        # LONG: entry=2850, sl=2840, exit=2870
        # risk per lot = |2850-2840| * 0.01 = 0.1
        # net_pnl = (2870-2850)*0.01*100 - 0.14 = 20 - 0.14 = 19.86
        # R = 19.86 / 0.1 = 198.6 ... wait that's too high
        # Actually r_multiple = net_pnl / (abs(entry-sl) * size_lots)
        # = 19.86 / (10 * 0.01) = 19.86 / 0.1 = 198.6
        # Hmm, that's not 2.0. Let me recalculate.
        # The r_multiple formula in order.py: risk = abs(entry - sl) * size_lots * 100
        # = |2850-2840| * 0.01 * 100 = 10.0
        # net_pnl / risk = 20.0 / 10.0 = 2.0R (CORRECT)
        t = self._make_trade(
            direction="LONG", entry=2850.0, exit_=2870.0,
            sl=2840.0, lots=0.01, commission=0.0,
        )
        # gross_pnl = (2870-2850)*0.01*100 = 20.0
        # risk = |2850-2840| * 0.01 * 100 = 10.0
        # r_multiple = 20.0 / 10.0 = 2.0R
        assert t.r_multiple == pytest.approx(2.0, rel=0.01)

    def test_backtest_trade_r_multiple_short_symmetric(self):
        t = self._make_trade(
            direction="SHORT", entry=2860.0, exit_=2840.0,
            sl=2870.0, lots=0.01, commission=0.0,
        )
        # gross_pnl = (2860-2840)*0.01*100 = 20.0
        # risk = |2860-2870| * 0.01 * 100 = 10.0
        # r_multiple = 20.0 / 10.0 = 2.0R
        assert t.r_multiple == pytest.approx(2.0, rel=0.01)

    def test_backtest_trade_r_multiple_zero_risk(self):
        t = self._make_trade(
            direction="LONG", entry=2850.0, exit_=2860.0,
            sl=2850.0, lots=0.01, commission=0.0,
        )
        # risk = |2850-2850| * 0.01 = 0.0
        assert t.r_multiple == 0.0  # No crash
