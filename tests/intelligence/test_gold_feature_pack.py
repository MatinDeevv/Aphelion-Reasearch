from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from aphelion.intelligence.hydra.ensemble import HydraGate
from aphelion_data import add_high_value_gold_features, feat_asian_range
from scripts.train_hydra_tiny import build_tiny_ensemble_config


def _sample_frame(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
    base = 2000 + np.cumsum(np.random.default_rng(1).normal(0, 0.4, n))
    return pd.DataFrame(
        {
            "open": base + np.random.default_rng(2).normal(0, 0.1, n),
            "high": base + 0.3,
            "low": base - 0.3,
            "close": base,
            "tick_vol": np.random.default_rng(7).integers(50, 200, n).astype(float),
            "xau_m5_close": base + 0.05,
            "xau_m15_close": base + 0.08,
            "xau_h1_close": base + 0.12,
            "xau_h4_close": base + 0.20,
            "xau_d1_close": base + 0.30,
            "xagusd_m5_close": 25 + np.cumsum(np.random.default_rng(3).normal(0, 0.01, n)),
            "usdx_m5_close": 103 + np.cumsum(np.random.default_rng(4).normal(0, 0.005, n)),
            "spx500_h1_close": 6000 + np.cumsum(np.random.default_rng(8).normal(0, 0.8, n)),
            "eurusd_m5_close": 1.08 + np.cumsum(np.random.default_rng(9).normal(0, 0.0002, n)),
            "spread": np.abs(np.random.default_rng(5).normal(0.2, 0.03, n)),
            "atr_14": np.abs(np.random.default_rng(6).normal(1.5, 0.1, n)),
            "future_ret_5": np.linspace(-0.2, 0.2, n),
            "future_ret_15": np.linspace(-0.3, 0.3, n),
            "future_ret_60": np.linspace(-0.4, 0.4, n),
        },
        index=idx,
    )


def test_gold_feature_pack_adds_expected_columns():
    df = add_high_value_gold_features(_sample_frame())
    expected = {
        "asian_high",
        "pdh",
        "bull_liq_sweep",
        "dxy_beta_20",
        "bull_ob",
        "displacement_any",
        "gold_silver_ratio",
        "spread_zscore_20",
        "mtf_score",
        "high_impact_news",
        "vpoc",
        "flow_imbalance_20",
        "ret_skew_60",
        "hourly_seasonal_ret",
        "fib_05_50",
        "session_vwap",
        "vr_4",
        "oi_balance_20",
        "mins_to_fix",
        "smc_net_score",
        "conviction_5",
    }
    missing = expected - set(df.columns)
    assert not missing


def test_asian_range_is_causal_during_session():
    idx = pd.to_datetime(
        ["2026-01-01 00:00:00+00:00", "2026-01-01 02:00:00+00:00", "2026-01-01 09:00:00+00:00"]
    )
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 105.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 104.0, 102.5],
        },
        index=idx,
    )

    out = feat_asian_range(df)

    assert out.iloc[0]["asian_high"] == pytest.approx(101.0)
    assert out.iloc[1]["asian_high"] == pytest.approx(105.0)
    assert out.iloc[2]["asian_high"] == pytest.approx(105.0)


def test_tiny_hydra_model_builds():
    model = HydraGate(build_tiny_ensemble_config(n_continuous=12, n_categorical=2, lookback=16))
    assert model.count_parameters() > 0
