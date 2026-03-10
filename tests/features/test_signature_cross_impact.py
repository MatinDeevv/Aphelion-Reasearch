import numpy as np
import pytest

from aphelion.features.signature import SignatureTransform
from aphelion.features.cross_impact import CrossImpactMatrix, CrossImpactConfig
from aphelion.features.engine import FeatureEngine
from aphelion.core.config import Timeframe
from aphelion.core.data_layer import DataLayer, Tick
from aphelion.core.event_bus import EventBus


def test_signature_transform_captures_path_structure():
    sig = SignatureTransform()

    n = 80
    close = 2000.0 + np.linspace(0.0, 12.0, n)
    volume = 100.0 + np.sin(np.linspace(0, 4 * np.pi, n)) * 20.0 + 50.0
    spread = 0.2 + 0.02 * np.cos(np.linspace(0, 6 * np.pi, n))

    out = sig.compute(close=close, volume=volume, spread=spread)

    assert abs(out["sig1_price"]) > 0.1
    assert out["sig_l2_frobenius"] > 0.0


def test_cross_impact_matrix_detects_external_driver():
    rng = np.random.default_rng(123)
    n = 320

    dxy_ret = rng.normal(0.0, 8e-4, n)
    tlt_ret = rng.normal(0.0, 8e-4, n)

    # XAU is mostly driven by lagged DXY returns in this synthetic process.
    xau_ret = 0.75 * np.roll(dxy_ret, 1) + 0.10 * np.roll(tlt_ret, 1) + rng.normal(0.0, 4e-4, n)
    xau_ret[0] = 0.0

    dxy = 100.0 * np.exp(np.cumsum(dxy_ret))
    tlt = 90.0 * np.exp(np.cumsum(tlt_ret))
    xau = 2000.0 * np.exp(np.cumsum(xau_ret))

    ci = CrossImpactMatrix(CrossImpactConfig(window=220, ridge=1e-4, target_symbol="XAUUSD"))
    feats = ci.fit({"XAUUSD": xau, "DXY": dxy, "TLT": tlt})

    assert feats["cross_impact_strength"] > 0.0
    assert abs(feats["cross_impact_beta_dxy"]) > abs(feats["cross_impact_beta_tlt"])


async def _make_data_layer(n_ticks: int = 2200):
    bus = EventBus()
    data_layer = DataLayer(bus)
    rng = np.random.default_rng(42)

    price = 2850.0
    start_ts = 1704067200.0

    for i in range(n_ticks):
        price += float(rng.normal(0.0, 0.03))
        tick = Tick(
            timestamp=start_ts + i,
            bid=price - 0.10,
            ask=price + 0.10,
            last=price,
            volume=1.0 + float(rng.uniform(0.0, 0.5)),
        )
        await data_layer.process_tick(tick)

    return data_layer


@pytest.mark.asyncio
async def test_feature_engine_emits_signature_and_cross_impact_features():
    data = await _make_data_layer()
    m1_bars = data.get_bars(Timeframe.M1, count=10_000)
    assert len(m1_bars) >= 30

    fe = FeatureEngine(data)

    # Provide external assets for cross-impact matrix.
    n = len(data.get_bars_df(Timeframe.M1, count=500)["close"].values)
    x = np.linspace(0, 1, n)
    dxy = 104.0 + 0.5 * np.sin(8 * np.pi * x)
    tlt = 95.0 + 0.4 * np.cos(6 * np.pi * x)
    fe.set_external_prices("DXY", dxy)
    fe.set_external_prices("TLT", tlt)

    out = fe.on_bar(m1_bars[-1])

    for key in (
        "sig1_price",
        "sig2_price_volume",
        "cross_impact_signal",
        "cross_impact_strength",
    ):
        assert key in out
