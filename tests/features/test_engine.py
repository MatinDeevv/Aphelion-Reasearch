"""End-to-end tests for APHELION FeatureEngine."""

import numpy as np
import pytest

from aphelion.core.config import Timeframe
from aphelion.core.data_layer import DataLayer, Tick
from aphelion.core.event_bus import EventBus
from aphelion.features.engine import FeatureEngine


async def make_data_layer(n_ticks: int = 200) -> tuple[DataLayer, list]:
    """Build a DataLayer and feed synthetic XAUUSD ticks."""
    bus = EventBus()
    data_layer = DataLayer(bus)
    rng = np.random.default_rng(42)

    price = 2850.0
    start_ts = 1704067200.0  # 2024-01-01 00:00:00 UTC (minute-aligned)

    for i in range(n_ticks):
        price += float(rng.normal(0.0, 0.03))
        bid = price - 0.10
        ask = price + 0.10
        volume = 1.0 + float(rng.uniform(0.0, 0.5))

        tick = Tick(
            timestamp=start_ts + i,
            bid=bid,
            ask=ask,
            last=price,
            volume=volume,
        )
        await data_layer.process_tick(tick)

    completed_bars = data_layer.get_bars(Timeframe.M1, count=10_000)
    return data_layer, completed_bars


@pytest.mark.asyncio
async def test_on_bar_returns_dict():
    data_layer, completed_bars = await make_data_layer()
    assert len(completed_bars) >= 2

    fe = FeatureEngine(data_layer)
    result = fe.on_bar(completed_bars[-1])

    assert isinstance(result, dict)
    assert result


@pytest.mark.asyncio
async def test_microstructure_features_present():
    data_layer, completed_bars = await make_data_layer()
    fe = FeatureEngine(data_layer)

    result = fe.on_bar(completed_bars[-1])
    for key in ("vpin", "ofi", "tick_entropy", "bid_ask_spread"):
        assert key in result


@pytest.mark.asyncio
async def test_session_features_present():
    data_layer, completed_bars = await make_data_layer()
    fe = FeatureEngine(data_layer)

    result = fe.on_bar(completed_bars[-1])
    for key in ("session", "day_of_week", "market_open"):
        assert key in result


@pytest.mark.asyncio
async def test_volume_profile_present():
    data_layer, completed_bars = await make_data_layer()
    fe = FeatureEngine(data_layer)

    result = fe.on_bar(completed_bars[-1])
    for key in ("volume_delta", "cumulative_delta", "poc"):
        assert key in result


@pytest.mark.asyncio
async def test_vwap_present():
    data_layer, completed_bars = await make_data_layer()
    fe = FeatureEngine(data_layer)

    result = fe.on_bar(completed_bars[-1])
    for key in ("session_vwap", "price_vs_vwap"):
        assert key in result


@pytest.mark.asyncio
async def test_technicals_after_20_bars():
    data_layer, completed_bars = await make_data_layer(n_ticks=1700)
    assert len(completed_bars) >= 25

    fe = FeatureEngine(data_layer)
    result = fe.on_bar(completed_bars[-1])

    assert "bb_width" in result
    assert "rsi" in result


@pytest.mark.asyncio
async def test_mtf_on_m5_bar():
    data_layer, _ = await make_data_layer(n_ticks=420)
    m5_bars = data_layer.get_bars(Timeframe.M5, count=100)
    assert m5_bars

    fe = FeatureEngine(data_layer)
    result = fe.on_bar(m5_bars[-1])

    assert "mtf_alignment_count" in result


@pytest.mark.asyncio
async def test_set_external_prices():
    data_layer, _ = await make_data_layer()
    fe = FeatureEngine(data_layer)

    fe.set_external_prices("DXY", np.array([104.0] * 60))


@pytest.mark.asyncio
async def test_session_reset():
    data_layer, completed_bars = await make_data_layer()
    fe = FeatureEngine(data_layer)

    fe.on_bar(completed_bars[-1])
    assert fe._vwap[Timeframe.M1].to_dict()["session_vwap"] > 0.0

    fe.reset_session()
    assert fe._vwap[Timeframe.M1].to_dict()["session_vwap"] == 0.0


# ── New technical indicator tests ───────────────────────────────


@pytest.mark.asyncio
async def test_macd_present_after_35_bars():
    data_layer, completed_bars = await make_data_layer(n_ticks=2500)
    assert len(completed_bars) >= 35

    fe = FeatureEngine(data_layer)
    result = fe.on_bar(completed_bars[-1])

    assert "macd_line" in result
    assert "macd_signal" in result
    assert "macd_histogram" in result
    assert isinstance(result["macd_line"], float)


@pytest.mark.asyncio
async def test_stochastic_present():
    data_layer, completed_bars = await make_data_layer(n_ticks=1200)
    assert len(completed_bars) >= 17

    fe = FeatureEngine(data_layer)
    result = fe.on_bar(completed_bars[-1])

    assert "stoch_k" in result
    assert "stoch_d" in result
    assert 0.0 <= result["stoch_k"] <= 100.0
    assert 0.0 <= result["stoch_d"] <= 100.0


@pytest.mark.asyncio
async def test_adx_present_after_28_bars():
    data_layer, completed_bars = await make_data_layer(n_ticks=2000)
    assert len(completed_bars) >= 28

    fe = FeatureEngine(data_layer)
    result = fe.on_bar(completed_bars[-1])

    assert "adx" in result
    assert result["adx"] >= 0.0


@pytest.mark.asyncio
async def test_obv_present():
    data_layer, completed_bars = await make_data_layer(n_ticks=300)
    assert len(completed_bars) >= 2

    fe = FeatureEngine(data_layer)
    result = fe.on_bar(completed_bars[-1])

    assert "obv" in result
    assert isinstance(result["obv"], float)


@pytest.mark.asyncio
async def test_roc_present_after_11_bars():
    data_layer, completed_bars = await make_data_layer(n_ticks=800)
    assert len(completed_bars) >= 11

    fe = FeatureEngine(data_layer)
    result = fe.on_bar(completed_bars[-1])

    assert "roc_10" in result
    assert isinstance(result["roc_10"], float)


class TestStaticIndicators:
    """Unit-level tests for static technical methods."""

    def test_macd_basic(self):
        # Uptrend → positive MACD
        closes = np.linspace(100, 130, 50)
        result = FeatureEngine._compute_macd(closes)
        assert result["macd_line"] > 0
        assert "macd_signal" in result
        assert "macd_histogram" in result

    def test_stochastic_range(self):
        highs = np.array([110.0] * 20)
        lows = np.array([90.0] * 20)
        closes = np.array([100.0] * 20)
        result = FeatureEngine._compute_stochastic(highs, lows, closes)
        assert 0 <= result["stoch_k"] <= 100
        assert 0 <= result["stoch_d"] <= 100

    def test_stochastic_at_high(self):
        highs = np.array([110.0] * 20)
        lows = np.array([90.0] * 20)
        closes = np.array([110.0] * 20)  # always at the high
        result = FeatureEngine._compute_stochastic(highs, lows, closes)
        assert result["stoch_k"] == pytest.approx(100.0)

    def test_adx_returns_float(self):
        rng = np.random.default_rng(42)
        n = 60
        closes = np.cumsum(rng.normal(0, 1, n)) + 100
        highs = closes + rng.uniform(0.5, 1.5, n)
        lows = closes - rng.uniform(0.5, 1.5, n)
        adx = FeatureEngine._compute_adx(highs, lows, closes, period=14)
        assert isinstance(adx, float)
        assert adx >= 0

    def test_adx_zero_on_short_data(self):
        adx = FeatureEngine._compute_adx(
            np.array([100.0] * 10),
            np.array([99.0] * 10),
            np.array([99.5] * 10),
            period=14,
        )
        assert adx == 0.0

    def test_obv_uptrend(self):
        closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        volumes = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        obv = FeatureEngine._compute_obv(closes, volumes)
        # All up closes → OBV = 200+300+400+500 = 1400
        assert obv == pytest.approx(1400.0)

    def test_obv_downtrend(self):
        closes = np.array([104.0, 103.0, 102.0, 101.0, 100.0])
        volumes = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        obv = FeatureEngine._compute_obv(closes, volumes)
        assert obv == pytest.approx(-1400.0)
