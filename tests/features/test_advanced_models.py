import numpy as np

from aphelion.features.microstructure import (
    BivariateHawkesOrderFlow,
    HawkesMLEParams,
    TSRVEstimator,
    MicrostructureEngine,
)


def test_hawkes_order_flow_accelerates_on_clustered_buys():
    model = BivariateHawkesOrderFlow(
        HawkesMLEParams(
            mu_buy=0.2,
            mu_sell=0.2,
            alpha_bb=0.8,
            alpha_bs=0.2,
            alpha_sb=0.2,
            alpha_ss=0.6,
            beta_buy=1.5,
            beta_sell=1.5,
        )
    )

    times = [0.0, 0.1, 0.2, 0.25, 0.28]
    accels = []
    for t in times:
        out = model.update(t, "buy")
        accels.append(out["total_acceleration"])

    assert out["buy_intensity"] > out["sell_intensity"]
    assert max(accels[1:]) > 0.0


def test_hawkes_mle_path_returns_finite_likelihood():
    model = BivariateHawkesOrderFlow()
    events = []
    t = 0.0
    for i in range(80):
        t += 0.08 + (0.01 if i % 5 else 0.0)
        side = "buy" if i % 3 != 0 else "sell"
        events.append((t, side))

    nll0 = model.neg_log_likelihood(model.params.to_vector(), events)
    fit = model.fit_mle(events, maxiter=40)

    assert np.isfinite(nll0)
    assert np.isfinite(fit.neg_log_likelihood)
    assert fit.params.mu_buy > 0
    assert fit.params.mu_sell > 0


def test_tsrv_reduces_bid_ask_bounce_noise():
    rng = np.random.default_rng(7)
    n = 420

    # Latent efficient price random walk
    efficient = 2000.0 + np.cumsum(rng.normal(0.0, 0.03, n))

    # Add alternating microstructure bounce
    bounce = 0.08 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    observed = efficient + bounce

    est = TSRVEstimator(window=360, slow_scale=6)

    tsrv = 0.0
    noise = 0.0
    rv_fast = 0.0
    for p in observed:
        tsrv, noise, rv_fast = est.update(float(p))

    assert rv_fast > 0.0
    assert tsrv >= 0.0
    assert noise >= 0.0
    # In strong bounce data, fast RV should exceed TSRV due to noise contamination.
    assert rv_fast > tsrv


def test_microstructure_engine_emits_new_advanced_keys():
    engine = MicrostructureEngine()
    t = 1_700_000_000.0

    for i in range(250):
        price = 2850.0 + 0.02 * np.sin(i / 8)
        bid = price - 0.1
        ask = price + 0.1
        last = ask if i % 2 == 0 else bid
        engine.update(
            timestamp=t + i * 0.2,
            bid=bid,
            ask=ask,
            last_price=last,
            volume=1.0 + (i % 5) * 0.2,
            bid_size=100 + (i % 7),
            ask_size=100 + ((i + 3) % 7),
        )

    d = engine.to_dict()
    for key in (
        "hawkes_of_buy_intensity",
        "hawkes_of_sell_intensity",
        "hawkes_flow_acceleration",
        "tsrv_volatility",
        "tsrv_noise_ratio",
    ):
        assert key in d
        assert np.isfinite(d[key])
