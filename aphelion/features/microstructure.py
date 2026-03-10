"""
APHELION Microstructure Features (v4)

Includes legacy microstructure signals plus:
- Bivariate Hawkes order-flow model with self/cross-excitation
- Optional Hawkes MLE calibration for tick-event streams
- Two-Scale Realized Variance (TSRV) for noise-robust volatility
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


@dataclass
class MicrostructureState:
    """Rolling state for microstructure computation."""

    vpin: float = 0.0
    ofi: float = 0.0
    ofi_normalized: float = 0.0
    tick_entropy: float = 0.0

    hawkes_buy_intensity: float = 0.0
    hawkes_sell_intensity: float = 0.0
    hawkes_imbalance: float = 0.0

    # Advanced Hawkes order-flow features
    hawkes_of_buy_intensity: float = 0.0
    hawkes_of_sell_intensity: float = 0.0
    hawkes_flow_acceleration: float = 0.0
    hawkes_branching_ratio: float = 0.0

    micro_price_divergence: float = 0.0
    bid_ask_spread: float = 0.0
    spread_velocity: float = 0.0
    quote_depth: float = 0.0

    kyle_lambda: float = 0.0
    amihud_illiquidity: float = 0.0
    roll_spread: float = 0.0

    # Noise-robust volatility
    tsrv_variance: float = 0.0
    tsrv_volatility: float = 0.0
    tsrv_noise_variance: float = 0.0
    tsrv_noise_ratio: float = 0.0

    toxicity_index: float = 0.0


class VPINCalculator:
    """Volume-Synchronized Probability of Informed Trading."""

    def __init__(self, bucket_size: int = 50, n_buckets: int = 50):
        self._bucket_size = bucket_size
        self._n_buckets = n_buckets
        self._current_bucket_volume = 0.0
        self._current_buy_volume = 0.0
        self._buckets: deque[tuple[float, float]] = deque(maxlen=n_buckets)
        self._last_price: float = 0.0

    def update(self, price: float, volume: float) -> float:
        if self._last_price == 0:
            self._last_price = price
            return 0.0

        if price > self._last_price:
            buy_vol = volume
        elif price < self._last_price:
            buy_vol = 0.0
        else:
            buy_vol = volume * 0.5

        self._last_price = price
        self._current_bucket_volume += volume
        self._current_buy_volume += buy_vol

        if self._current_bucket_volume >= self._bucket_size:
            sell_vol = self._current_bucket_volume - self._current_buy_volume
            self._buckets.append((self._current_buy_volume, sell_vol))

            overflow = self._current_bucket_volume - self._bucket_size
            if overflow > 0:
                overflow_buy = buy_vol * (overflow / volume) if volume > 0 else 0.0
                self._current_bucket_volume = overflow
                self._current_buy_volume = overflow_buy
            else:
                self._current_bucket_volume = 0.0
                self._current_buy_volume = 0.0

        return self._compute()

    def _compute(self) -> float:
        if len(self._buckets) < 2:
            return 0.0
        total_imbalance = sum(abs(b - s) for b, s in self._buckets)
        total_volume = sum(b + s for b, s in self._buckets)
        if total_volume <= 0:
            return 0.0
        return total_imbalance / total_volume


class OFICalculator:
    """Order Flow Imbalance."""

    def __init__(self, window: int = 50):
        self._window = window
        self._prev_bid: float = 0.0
        self._prev_ask: float = 0.0
        self._prev_bid_size: float = 0.0
        self._prev_ask_size: float = 0.0
        self._ofi_values: deque[float] = deque(maxlen=window)

    def update(
        self,
        bid: float,
        ask: float,
        bid_size: float = 1.0,
        ask_size: float = 1.0,
    ) -> float:
        if self._prev_bid == 0:
            self._prev_bid = bid
            self._prev_ask = ask
            self._prev_bid_size = bid_size
            self._prev_ask_size = ask_size
            return 0.0

        if bid > self._prev_bid:
            bid_delta = bid_size
        elif bid == self._prev_bid:
            bid_delta = bid_size - self._prev_bid_size
        else:
            bid_delta = -self._prev_bid_size

        if ask < self._prev_ask:
            ask_delta = -ask_size
        elif ask == self._prev_ask:
            ask_delta = -(ask_size - self._prev_ask_size)
        else:
            ask_delta = self._prev_ask_size

        ofi = bid_delta + ask_delta
        self._ofi_values.append(ofi)

        self._prev_bid = bid
        self._prev_ask = ask
        self._prev_bid_size = bid_size
        self._prev_ask_size = ask_size

        return float(sum(self._ofi_values))

    @property
    def normalized(self) -> float:
        if len(self._ofi_values) < 2:
            return 0.0

        vals = np.asarray(self._ofi_values, dtype=float)
        raw = float(np.sum(vals))
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if std < 1e-12:
            return 0.0

        z = (raw - mean * len(vals)) / (std * len(vals))
        return float(np.clip(z, -1.0, 1.0))


class TickEntropyCalculator:
    """3-state Shannon entropy over tick direction sequence."""

    def __init__(self, window: int = 100):
        self._window = window
        self._directions: deque[int] = deque(maxlen=window)
        self._last_price: float = 0.0

    def update(self, price: float) -> float:
        if self._last_price == 0:
            self._last_price = price
            return 1.0

        if price > self._last_price:
            direction = 1
        elif price < self._last_price:
            direction = -1
        else:
            direction = 0

        self._directions.append(direction)
        self._last_price = price

        if len(self._directions) < 10:
            return 1.0

        return self._compute()

    def _compute(self) -> float:
        n = len(self._directions)
        if n == 0:
            return 1.0

        ups = sum(1 for d in self._directions if d == 1)
        downs = sum(1 for d in self._directions if d == -1)
        flats = n - ups - downs

        entropy = 0.0
        for count in (ups, downs, flats):
            if count > 0:
                p = count / n
                entropy -= p * np.log2(p)
        return float(entropy)


class HawkesIntensity:
    """Legacy single-stream O(1) Hawkes intensity."""

    def __init__(self, decay: float = 0.1, baseline: float = 1.0):
        self._decay = max(decay, 1e-6)
        self._baseline = max(baseline, 0.0)
        self._recursive_sum: float = 0.0
        self._last_event_time: float = 0.0
        self._event_count: int = 0
        self._intensity = self._baseline

    def update(self, timestamp: float) -> float:
        if self._event_count > 0:
            dt = max(float(timestamp) - self._last_event_time, 0.0)
            self._recursive_sum = self._recursive_sum * np.exp(-self._decay * dt) + 1.0
        else:
            self._recursive_sum = 1.0

        self._last_event_time = float(timestamp)
        self._event_count += 1
        self._intensity = self._baseline + self._recursive_sum
        return float(self._intensity)

    def current(self, timestamp: float) -> float:
        if self._event_count == 0:
            return float(self._baseline)
        dt = max(float(timestamp) - self._last_event_time, 0.0)
        decayed = self._recursive_sum * np.exp(-self._decay * dt)
        return float(self._baseline + decayed)

    @property
    def intensity(self) -> float:
        return float(self._intensity)


@dataclass
class HawkesMLEParams:
    """Bivariate Hawkes parameters."""

    mu_buy: float = 0.2
    mu_sell: float = 0.2
    alpha_bb: float = 0.6
    alpha_bs: float = 0.2
    alpha_sb: float = 0.2
    alpha_ss: float = 0.6
    beta_buy: float = 1.5
    beta_sell: float = 1.5

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.mu_buy,
                self.mu_sell,
                self.alpha_bb,
                self.alpha_bs,
                self.alpha_sb,
                self.alpha_ss,
                self.beta_buy,
                self.beta_sell,
            ],
            dtype=float,
        )

    @staticmethod
    def from_vector(v: np.ndarray) -> "HawkesMLEParams":
        return HawkesMLEParams(
            mu_buy=float(v[0]),
            mu_sell=float(v[1]),
            alpha_bb=float(v[2]),
            alpha_bs=float(v[3]),
            alpha_sb=float(v[4]),
            alpha_ss=float(v[5]),
            beta_buy=float(v[6]),
            beta_sell=float(v[7]),
        )


@dataclass
class HawkesMLEFitResult:
    """Result object for Hawkes MLE fitting."""

    params: HawkesMLEParams
    success: bool
    neg_log_likelihood: float
    message: str = ""


class BivariateHawkesOrderFlow:
    """
    Bivariate Hawkes process for buy/sell order arrivals.

    lambda_buy(t)  = mu_buy  + alpha_bb * R_buy(t) + alpha_bs * R_sell(t)
    lambda_sell(t) = mu_sell + alpha_sb * R_buy(t) + alpha_ss * R_sell(t)

    where R_side follows exponential-kernel recursion.
    """

    def __init__(self, params: Optional[HawkesMLEParams] = None):
        self.params = params or HawkesMLEParams()
        self._r_buy = 0.0
        self._r_sell = 0.0
        self._last_ts: Optional[float] = None
        self._prev_total_intensity: Optional[float] = None
        self._prev_intensity_ts: Optional[float] = None

    def reset(self) -> None:
        self._r_buy = 0.0
        self._r_sell = 0.0
        self._last_ts = None
        self._prev_total_intensity = None
        self._prev_intensity_ts = None

    def laplace_kernels(self, s: float) -> np.ndarray:
        """Laplace transform of exponential kernels Phi_ij(s)=alpha_ij/(s+beta_j)."""
        p = self.params
        den_buy = max(s + p.beta_buy, 1e-9)
        den_sell = max(s + p.beta_sell, 1e-9)
        return np.array(
            [
                [p.alpha_bb / den_buy, p.alpha_bs / den_sell],
                [p.alpha_sb / den_buy, p.alpha_ss / den_sell],
            ],
            dtype=float,
        )

    def branching_ratio(self) -> float:
        p = self.params
        kernel_mass = np.array(
            [
                [p.alpha_bb / max(p.beta_buy, 1e-9), p.alpha_bs / max(p.beta_sell, 1e-9)],
                [p.alpha_sb / max(p.beta_buy, 1e-9), p.alpha_ss / max(p.beta_sell, 1e-9)],
            ],
            dtype=float,
        )
        eigvals = np.linalg.eigvals(kernel_mass)
        return float(np.max(np.real(eigvals)))

    def _decay_to(self, timestamp: float) -> tuple[float, float, float]:
        ts = float(timestamp)
        if self._last_ts is None:
            self._last_ts = ts
            return 0.0, self._r_buy, self._r_sell

        dt = max(ts - self._last_ts, 0.0)
        p = self.params
        self._r_buy *= float(np.exp(-p.beta_buy * dt))
        self._r_sell *= float(np.exp(-p.beta_sell * dt))
        self._last_ts = ts
        return dt, self._r_buy, self._r_sell

    def current_intensities(self, timestamp: float) -> tuple[float, float]:
        _, r_buy, r_sell = self._decay_to(timestamp)
        p = self.params
        lam_buy = p.mu_buy + p.alpha_bb * r_buy + p.alpha_bs * r_sell
        lam_sell = p.mu_sell + p.alpha_sb * r_buy + p.alpha_ss * r_sell
        return float(max(lam_buy, 1e-8)), float(max(lam_sell, 1e-8))

    def update(self, timestamp: float, side: str) -> dict[str, float]:
        dt, _, _ = self._decay_to(timestamp)
        lam_buy_pre, lam_sell_pre = self.current_intensities(timestamp)

        if side.lower() == "buy":
            self._r_buy += 1.0
        else:
            self._r_sell += 1.0

        lam_buy_post, lam_sell_post = self.current_intensities(timestamp)
        total = lam_buy_post + lam_sell_post

        accel = 0.0
        if self._prev_total_intensity is not None and self._prev_intensity_ts is not None:
            dti = max(float(timestamp) - self._prev_intensity_ts, 1e-6)
            accel = (total - self._prev_total_intensity) / dti

        self._prev_total_intensity = total
        self._prev_intensity_ts = float(timestamp)

        return {
            "buy_intensity": float(lam_buy_post),
            "sell_intensity": float(lam_sell_post),
            "buy_intensity_pre": float(lam_buy_pre),
            "sell_intensity_pre": float(lam_sell_pre),
            "total_acceleration": float(accel),
            "branching_ratio": self.branching_ratio(),
        }

    @staticmethod
    def _events_to_arrays(events: list[tuple[float, str]]) -> tuple[np.ndarray, np.ndarray]:
        if not events:
            return np.array([], dtype=float), np.array([], dtype=int)

        ev_sorted = sorted(events, key=lambda x: x[0])
        ts = np.array([float(t) for t, _ in ev_sorted], dtype=float)
        side = np.array([0 if str(s).lower() == "buy" else 1 for _, s in ev_sorted], dtype=int)
        return ts, side

    @classmethod
    def neg_log_likelihood(
        cls,
        params_vec: np.ndarray,
        events: list[tuple[float, str]],
        end_time: Optional[float] = None,
    ) -> float:
        p = HawkesMLEParams.from_vector(params_vec)

        if min(params_vec) <= 0:
            return 1e12

        ts, side = cls._events_to_arrays(events)
        if len(ts) == 0:
            return 0.0

        if end_time is None:
            end_time = float(ts[-1])

        # Mild stationarity soft penalty (spectral radius < 1)
        kernel_mass = np.array(
            [
                [p.alpha_bb / p.beta_buy, p.alpha_bs / p.beta_sell],
                [p.alpha_sb / p.beta_buy, p.alpha_ss / p.beta_sell],
            ],
            dtype=float,
        )
        rho = float(np.max(np.real(np.linalg.eigvals(kernel_mass))))
        penalty = 0.0
        if rho >= 0.995:
            penalty = 1e5 * (rho - 0.995) ** 2

        r_buy = 0.0
        r_sell = 0.0
        t_prev = float(ts[0])
        log_term = 0.0
        integral = 0.0

        for k in range(len(ts)):
            t = float(ts[k])
            dt = max(t - t_prev, 0.0)

            # Integral over [t_prev, t]
            integral += (p.mu_buy + p.mu_sell) * dt
            if dt > 0:
                integral += (p.alpha_bb + p.alpha_sb) * r_buy * (1.0 - np.exp(-p.beta_buy * dt)) / p.beta_buy
                integral += (p.alpha_bs + p.alpha_ss) * r_sell * (1.0 - np.exp(-p.beta_sell * dt)) / p.beta_sell

            # Decay states to event time
            r_buy *= float(np.exp(-p.beta_buy * dt))
            r_sell *= float(np.exp(-p.beta_sell * dt))

            lam_buy = p.mu_buy + p.alpha_bb * r_buy + p.alpha_bs * r_sell
            lam_sell = p.mu_sell + p.alpha_sb * r_buy + p.alpha_ss * r_sell

            if side[k] == 0:
                log_term += np.log(max(lam_buy, 1e-12))
                r_buy += 1.0
            else:
                log_term += np.log(max(lam_sell, 1e-12))
                r_sell += 1.0

            t_prev = t

        # Tail integral [last_event, end_time]
        tail_dt = max(float(end_time) - t_prev, 0.0)
        integral += (p.mu_buy + p.mu_sell) * tail_dt
        if tail_dt > 0:
            integral += (p.alpha_bb + p.alpha_sb) * r_buy * (1.0 - np.exp(-p.beta_buy * tail_dt)) / p.beta_buy
            integral += (p.alpha_bs + p.alpha_ss) * r_sell * (1.0 - np.exp(-p.beta_sell * tail_dt)) / p.beta_sell

        nll = -(log_term - integral) + penalty
        if not np.isfinite(nll):
            return 1e12
        return float(nll)

    def fit_mle(
        self,
        events: list[tuple[float, str]],
        end_time: Optional[float] = None,
        maxiter: int = 300,
    ) -> HawkesMLEFitResult:
        if not events:
            return HawkesMLEFitResult(
                params=self.params,
                success=False,
                neg_log_likelihood=0.0,
                message="No events provided",
            )

        if not _HAS_SCIPY:
            nll = self.neg_log_likelihood(self.params.to_vector(), events, end_time=end_time)
            return HawkesMLEFitResult(
                params=self.params,
                success=False,
                neg_log_likelihood=nll,
                message="SciPy unavailable; returned current params",
            )

        x0 = self.params.to_vector()
        bounds = [(1e-6, 10.0)] * len(x0)

        result = minimize(
            self.neg_log_likelihood,
            x0=x0,
            args=(events, end_time),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter)},
        )

        fitted = HawkesMLEParams.from_vector(result.x)
        self.params = fitted

        return HawkesMLEFitResult(
            params=fitted,
            success=bool(result.success),
            neg_log_likelihood=float(result.fun),
            message=str(result.message),
        )


class TSRVEstimator:
    """Two-Scale Realized Variance estimator (Ait-Sahalia style)."""

    def __init__(self, window: int = 300, slow_scale: int = 5):
        self._window = max(int(window), 20)
        self._slow_scale = max(int(slow_scale), 2)
        self._prices: deque[float] = deque(maxlen=self._window)

    def update(self, price: float) -> tuple[float, float, float]:
        self._prices.append(float(price))
        return self.compute()

    def compute(self) -> tuple[float, float, float]:
        if len(self._prices) < max(20, self._slow_scale * 4):
            return 0.0, 0.0, 0.0

        prices = np.asarray(self._prices, dtype=float)
        prices = np.maximum(prices, 1e-9)
        logp = np.log(prices)

        fast_returns = np.diff(logp)
        n = len(fast_returns)
        if n < self._slow_scale + 2:
            return 0.0, 0.0, 0.0

        rv_fast = float(np.sum(fast_returns ** 2))

        rv_subsamples: list[float] = []
        n_subsamples: list[int] = []
        k = self._slow_scale

        for offset in range(k):
            sub = logp[offset::k]
            if len(sub) >= 2:
                r_sub = np.diff(sub)
                rv_subsamples.append(float(np.sum(r_sub ** 2)))
                n_subsamples.append(len(r_sub))

        if not rv_subsamples:
            return 0.0, 0.0, rv_fast

        rv_slow_avg = float(np.mean(rv_subsamples))
        nbar = float(np.mean(n_subsamples)) if n_subsamples else 0.0

        tsrv = rv_slow_avg - (nbar / max(n, 1)) * rv_fast
        tsrv = float(max(tsrv, 0.0))

        # Microstructure noise variance proxy
        noise_var = float(max((rv_fast - rv_slow_avg) / max(2 * n, 1), 0.0))

        return tsrv, noise_var, rv_fast


class KyleLambdaCalculator:
    """Kyle's lambda via rolling OLS slope of dP on signed sqrt(volume)."""

    def __init__(self, window: int = 100):
        self._window = window
        self._price_changes: deque[float] = deque(maxlen=window)
        self._signed_root_vol: deque[float] = deque(maxlen=window)
        self._last_price: float = 0.0

    def update(self, price: float, signed_volume: float) -> float:
        if self._last_price > 0:
            dp = price - self._last_price
            srv = np.sign(signed_volume) * np.sqrt(abs(signed_volume))
            self._price_changes.append(float(dp))
            self._signed_root_vol.append(float(srv))
        self._last_price = float(price)

        if len(self._price_changes) < 20:
            return 0.0

        y = np.asarray(self._price_changes, dtype=float)
        x = np.asarray(self._signed_root_vol, dtype=float)
        x_std = float(np.std(x))
        if x_std < 1e-12:
            return 0.0

        cov = float(np.mean((y - np.mean(y)) * (x - np.mean(x))))
        lam = cov / (x_std ** 2)
        return float(max(lam, 0.0))


class AmihudIlliquidity:
    """Amihud illiquidity: |return| / dollar_volume."""

    def __init__(self, window: int = 50):
        self._window = window
        self._ratios: deque[float] = deque(maxlen=window)
        self._last_price: float = 0.0

    def update(self, price: float, volume: float) -> float:
        p = float(price)
        v = float(volume)
        if self._last_price > 0 and v > 0:
            ret = abs((p - self._last_price) / self._last_price)
            dollar_vol = p * v
            self._ratios.append(ret / dollar_vol if dollar_vol > 0 else 0.0)
        self._last_price = p

        if len(self._ratios) < 5:
            return 0.0
        return float(np.mean(self._ratios))


class RollSpreadEstimator:
    """Roll implied spread: 2*sqrt(-cov(dp_t, dp_{t-1})) when covariance is negative."""

    def __init__(self, window: int = 100):
        self._window = window
        self._prices: deque[float] = deque(maxlen=window + 1)

    def update(self, price: float) -> float:
        self._prices.append(float(price))
        if len(self._prices) < 20:
            return 0.0

        prices = np.asarray(self._prices, dtype=float)
        changes = np.diff(prices)
        if len(changes) < 2:
            return 0.0

        cov = float(np.mean(changes[:-1] * changes[1:]) - np.mean(changes[:-1]) * np.mean(changes[1:]))
        if cov >= 0:
            return 0.0
        return float(2.0 * np.sqrt(-cov))


class MicrostructureEngine:
    """Computes all microstructure features from raw tick data."""

    def __init__(
        self,
        vpin_bucket_size: int = 50,
        ofi_window: int = 50,
        entropy_window: int = 100,
        hawkes_decay: float = 0.1,
    ):
        self._vpin = VPINCalculator(bucket_size=vpin_bucket_size)
        self._ofi = OFICalculator(window=ofi_window)
        self._entropy = TickEntropyCalculator(window=entropy_window)

        # Legacy univariate intensities kept for backward compatibility
        self._hawkes_buy = HawkesIntensity(decay=hawkes_decay)
        self._hawkes_sell = HawkesIntensity(decay=hawkes_decay)

        # Advanced bivariate model
        self._hawkes_orderflow = BivariateHawkesOrderFlow()

        self._kyle = KyleLambdaCalculator()
        self._amihud = AmihudIlliquidity()
        self._roll = RollSpreadEstimator()
        self._tsrv = TSRVEstimator(window=300, slow_scale=5)

        self._spread_history: deque[float] = deque(maxlen=50)
        self._state = MicrostructureState()

    @staticmethod
    def _to_timestamp(value: float) -> float:
        if hasattr(value, "timestamp"):
            return float(value.timestamp())
        return float(value)

    def update(
        self,
        timestamp: float,
        bid: float,
        ask: float,
        last_price: float,
        volume: float,
        bid_size: float = 1.0,
        ask_size: float = 1.0,
    ) -> MicrostructureState:
        ts = self._to_timestamp(timestamp)

        self._state.vpin = self._vpin.update(last_price, volume)

        raw_ofi = self._ofi.update(bid, ask, bid_size, ask_size)
        self._state.ofi = raw_ofi
        self._state.ofi_normalized = self._ofi.normalized

        self._state.tick_entropy = self._entropy.update(last_price)

        mid = (bid + ask) / 2.0
        is_buy = last_price >= mid

        # Legacy Hawkes update
        if is_buy:
            self._state.hawkes_buy_intensity = self._hawkes_buy.update(ts)
            self._state.hawkes_sell_intensity = self._hawkes_sell.current(ts)
        else:
            self._state.hawkes_sell_intensity = self._hawkes_sell.update(ts)
            self._state.hawkes_buy_intensity = self._hawkes_buy.current(ts)

        total_hawkes = self._state.hawkes_buy_intensity + self._state.hawkes_sell_intensity
        if total_hawkes > 0:
            self._state.hawkes_imbalance = (
                (self._state.hawkes_buy_intensity - self._state.hawkes_sell_intensity)
                / total_hawkes
            )
        else:
            self._state.hawkes_imbalance = 0.0

        # Advanced Hawkes order-flow process
        of_side = "buy" if is_buy else "sell"
        of_stats = self._hawkes_orderflow.update(ts, of_side)
        self._state.hawkes_of_buy_intensity = of_stats["buy_intensity"]
        self._state.hawkes_of_sell_intensity = of_stats["sell_intensity"]
        self._state.hawkes_flow_acceleration = of_stats["total_acceleration"]
        self._state.hawkes_branching_ratio = of_stats["branching_ratio"]

        if bid_size + ask_size > 0:
            weighted_mid = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
            self._state.micro_price_divergence = mid - weighted_mid
        else:
            self._state.micro_price_divergence = 0.0

        spread = ask - bid
        self._state.bid_ask_spread = spread
        self._spread_history.append(spread)
        if len(self._spread_history) >= 2:
            self._state.spread_velocity = spread - self._spread_history[-2]
        else:
            self._state.spread_velocity = 0.0

        self._state.quote_depth = bid_size + ask_size

        signed_vol = volume if is_buy else -volume
        self._state.kyle_lambda = self._kyle.update(last_price, signed_vol)
        self._state.amihud_illiquidity = self._amihud.update(last_price, volume)
        self._state.roll_spread = self._roll.update(last_price)

        tsrv_var, tsrv_noise, rv_fast = self._tsrv.update(last_price)
        self._state.tsrv_variance = tsrv_var
        self._state.tsrv_volatility = float(np.sqrt(tsrv_var)) if tsrv_var > 0 else 0.0
        self._state.tsrv_noise_variance = tsrv_noise
        self._state.tsrv_noise_ratio = float(tsrv_noise / (rv_fast + 1e-12)) if rv_fast > 0 else 0.0

        vpin_norm = min(self._state.vpin, 1.0)
        ofi_norm = min(abs(self._state.ofi_normalized), 1.0)
        hawkes_norm = min(abs(self._state.hawkes_imbalance), 1.0)
        accel_norm = min(abs(self._state.hawkes_flow_acceleration) / 10.0, 1.0)

        self._state.toxicity_index = (
            0.35 * vpin_norm
            + 0.25 * ofi_norm
            + 0.25 * hawkes_norm
            + 0.15 * accel_norm
        )

        return self._state

    def fit_hawkes_mle(
        self,
        events: list[tuple[float, str]],
        end_time: Optional[float] = None,
        maxiter: int = 300,
    ) -> HawkesMLEFitResult:
        return self._hawkes_orderflow.fit_mle(events=events, end_time=end_time, maxiter=maxiter)

    def to_dict(self) -> dict:
        return {
            "vpin": self._state.vpin,
            "ofi": self._state.ofi,
            "ofi_normalized": self._state.ofi_normalized,
            "tick_entropy": self._state.tick_entropy,

            "hawkes_buy_intensity": self._state.hawkes_buy_intensity,
            "hawkes_sell_intensity": self._state.hawkes_sell_intensity,
            "hawkes_buy": self._state.hawkes_buy_intensity,
            "hawkes_sell": self._state.hawkes_sell_intensity,
            "hawkes_imbalance": self._state.hawkes_imbalance,

            "hawkes_of_buy_intensity": self._state.hawkes_of_buy_intensity,
            "hawkes_of_sell_intensity": self._state.hawkes_of_sell_intensity,
            "hawkes_flow_acceleration": self._state.hawkes_flow_acceleration,
            "hawkes_branching_ratio": self._state.hawkes_branching_ratio,

            "micro_price_divergence": self._state.micro_price_divergence,
            "bid_ask_spread": self._state.bid_ask_spread,
            "spread_velocity": self._state.spread_velocity,
            "quote_depth": self._state.quote_depth,

            "kyle_lambda": self._state.kyle_lambda,
            "amihud_illiquidity": self._state.amihud_illiquidity,
            "roll_spread": self._state.roll_spread,

            "tsrv_variance": self._state.tsrv_variance,
            "tsrv_volatility": self._state.tsrv_volatility,
            "tsrv_noise_variance": self._state.tsrv_noise_variance,
            "tsrv_noise_ratio": self._state.tsrv_noise_ratio,

            "toxicity_index": self._state.toxicity_index,
        }
