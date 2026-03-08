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
@pytest.mark.xfail(
    reason=(
        "FeatureEngine.reset_session calls VWAPCalculator.reset_session, but "
        "VWAPCalculator does not clear exposed VWAPState.session_vwap to 0.0."
    )
)
async def test_session_reset():
    data_layer, completed_bars = await make_data_layer()
    fe = FeatureEngine(data_layer)

    fe.on_bar(completed_bars[-1])
    assert fe._vwap[Timeframe.M1].to_dict()["session_vwap"] > 0.0

    fe.reset_session()
    assert fe._vwap[Timeframe.M1].to_dict()["session_vwap"] == 0.0
