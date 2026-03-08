import pytest
from datetime import datetime, timezone, timedelta
from aphelion.backtest.order import Order, OrderType, OrderSide, OrderStatus, Fill, BacktestTrade
from aphelion.backtest.portfolio import Portfolio
from aphelion.core.config import Timeframe
from aphelion.core.data_layer import Bar
from aphelion.risk.sentinel.core import Position


class TestPortfolio:
    def _make_bar(self, close=2850.0, ts_offset=0):
        return Bar(
            timestamp=datetime.fromtimestamp(1704067200.0 + ts_offset, tz=timezone.utc),
            timeframe=Timeframe.M1, open=close-1, high=close+2, low=close-2,
            close=close, volume=100.0, tick_volume=100, spread=0.20, is_complete=True,
        )

    def _make_fill(self, price=2850.0, lots=0.01, commission=0.07, bar_index=0):
        return Fill(
            order_id="test-001", symbol="XAUUSD", side=OrderSide.BUY,
            filled_price=price, size_lots=lots, commission=commission,
            slippage_cost=0.01, fill_time=datetime.now(timezone.utc), bar_index=bar_index,
        )

    def _make_order(self, side=OrderSide.BUY, sl=2840.0, tp=2870.0, size_pct=0.02):
        return Order(
            order_id="test-001", symbol="XAUUSD", order_type=OrderType.MARKET,
            side=side, size_lots=0.01, entry_price=0.0,
            stop_loss=sl, take_profit=tp, size_pct=size_pct, proposed_by="TEST",
        )

    def test_initial_state(self):
        p = Portfolio(10000.0)
        assert p.equity == 10000.0
        assert p.cash == 10000.0
        assert p.total_return == 0.0

    def test_open_long_and_close_at_profit(self):
        p = Portfolio(10000.0)
        fill = self._make_fill(price=2850.0, lots=0.01, commission=0.07)
        order = self._make_order()
        p.open_position(fill, order)
        # Close at 2870: gross = (2870-2850)*0.01*100 = 20.0
        trade = p.close_position("test-001", 2870.0, datetime.now(timezone.utc), "TP_HIT", 10, 0.07)
        assert trade is not None
        assert trade.gross_pnl == pytest.approx(20.0, abs=0.01)
        assert trade.net_pnl == pytest.approx(20.0 - 0.14, abs=0.01)  # entry + exit commission
        assert p.cash > 10000.0

    def test_open_short_and_close_at_profit(self):
        p = Portfolio(10000.0)
        fill = Fill(
            order_id="s-001", symbol="XAUUSD", side=OrderSide.SELL,
            filled_price=2860.0, size_lots=0.01, commission=0.07,
            slippage_cost=0.01, fill_time=datetime.now(timezone.utc), bar_index=0,
        )
        order = Order(
            order_id="s-001", symbol="XAUUSD", order_type=OrderType.MARKET,
            side=OrderSide.SELL, size_lots=0.01, entry_price=0.0,
            stop_loss=2870.0, take_profit=2840.0, size_pct=0.02, proposed_by="TEST",
        )
        p.open_position(fill, order)
        trade = p.close_position("s-001", 2840.0, datetime.now(timezone.utc), "TP_HIT", 10, 0.07)
        assert trade is not None
        # SHORT: gross = (2860-2840)*0.01*100 = 20.0
        assert trade.gross_pnl == pytest.approx(20.0, abs=0.01)
        assert p.cash > 10000.0

    def test_unrealized_pnl_increases_equity(self):
        p = Portfolio(10000.0)
        fill = self._make_fill(price=2850.0)
        order = self._make_order()
        p.open_position(fill, order)
        bar = self._make_bar(close=2860.0, ts_offset=60)
        p.update_bar(bar, 1)
        assert p.equity > p.cash  # unrealized profit

    def test_close_losing_trade_reduces_cash(self):
        p = Portfolio(10000.0)
        fill = self._make_fill(price=2850.0)
        order = self._make_order()
        p.open_position(fill, order)
        p.close_position("test-001", 2840.0, datetime.now(timezone.utc), "SL_HIT", 5, 0.07)
        assert p.cash < 10000.0

    def test_equity_curve_grows_per_bar(self):
        p = Portfolio(10000.0)
        for i in range(5):
            bar = self._make_bar(ts_offset=i*60)
            p.update_bar(bar, i)
        # initial + 5 updates = 6 entries
        assert len(p._equity_curve) == 6

    def test_peak_equity_never_decreases(self):
        p = Portfolio(10000.0)
        fill = self._make_fill(price=2850.0)
        order = self._make_order()
        p.open_position(fill, order)
        # Price goes up
        bar_up = self._make_bar(close=2860.0, ts_offset=60)
        p.update_bar(bar_up, 1)
        peak_after_up = p.peak_equity
        # Price drops
        bar_down = self._make_bar(close=2840.0, ts_offset=120)
        p.update_bar(bar_down, 2)
        assert p.peak_equity >= peak_after_up

    def test_drawdown_correct(self):
        p = Portfolio(10000.0)
        p._equity = 11000.0
        p._peak_equity = 11000.0
        p._cash = 11000.0
        # Simulate drop
        p._equity = 9900.0
        p._cash = 9900.0
        assert p.current_drawdown == pytest.approx(0.10, abs=0.01)

    def test_total_return_positive(self):
        p = Portfolio(10000.0)
        fill = self._make_fill(price=2850.0, commission=0.0)
        order = self._make_order()
        p.open_position(fill, order)
        p.close_position("test-001", 2870.0, datetime.now(timezone.utc), "TP_HIT", 10, 0.0)
        bar = self._make_bar(ts_offset=60)
        p.update_bar(bar, 1)
        assert p.total_return > 0.0

    def test_daily_returns_list_populated(self):
        p = Portfolio(10000.0)
        # Day 1 bars
        for i in range(10):
            bar = self._make_bar(close=2850.0 + i, ts_offset=i * 60)
            p.update_bar(bar, i)
        # Day 2 bars (86400 seconds later)
        for i in range(10):
            bar = self._make_bar(close=2860.0 + i, ts_offset=86400 + i * 60)
            p.update_bar(bar, 10 + i)
        returns = p.get_daily_returns()
        assert len(returns) >= 1

    def test_multiple_positions_exposure_sums(self):
        p = Portfolio(10000.0)
        for j in range(3):
            fill = Fill(
                order_id=f"p-{j}", symbol="XAUUSD", side=OrderSide.BUY,
                filled_price=2850.0, size_lots=0.01, commission=0.07,
                slippage_cost=0.01, fill_time=datetime.now(timezone.utc), bar_index=0,
            )
            order = Order(
                order_id=f"p-{j}", symbol="XAUUSD", order_type=OrderType.MARKET,
                side=OrderSide.BUY, size_lots=0.01, entry_price=0.0,
                stop_loss=2840.0, take_profit=2870.0, size_pct=0.02, proposed_by="TEST",
            )
            p.open_position(fill, order)
        assert p.get_exposure_pct() == pytest.approx(0.06, abs=0.001)

    def test_equity_never_goes_negative(self):
        p = Portfolio(10000.0)
        # Catastrophic loss
        fill = self._make_fill(price=2850.0, lots=1.0, commission=0.0)
        order = Order(
            order_id="big-001", symbol="XAUUSD", order_type=OrderType.MARKET,
            side=OrderSide.BUY, size_lots=1.0, entry_price=0.0,
            stop_loss=2750.0, take_profit=2950.0, size_pct=0.02, proposed_by="TEST",
        )
        p.open_position(fill, order)
        # Close with massive loss: (2750-2850)*1.0*100 = -10000
        p.close_position("big-001", 2750.0, datetime.now(timezone.utc), "SL_HIT", 10, 0.0)
        # Cash = 10000 + (-10000) = 0
        assert p.cash == pytest.approx(0.0, abs=0.01)
