"""
APHELION Feature Engine
Master pipeline orchestrating all 80+ feature computations.
Includes advanced algorithms: Hurst exponent, wavelet denoising,
Fisher Transform, Chaikin Money Flow, Williams %R, Keltner Channels,
fractal dimension, and information-theoretic measures.
"""

import numpy as np
import pandas as pd
from typing import Optional
from scipy import signal as sp_signal
from scipy.stats import entropy as sp_entropy

from aphelion.core.config import Timeframe, TIMEFRAMES
from aphelion.core.clock import MarketClock
from aphelion.core.data_layer import DataLayer, Tick, Bar
from aphelion.features.microstructure import MicrostructureEngine
from aphelion.features.market_structure import MarketStructureEngine
from aphelion.features.volume_profile import VolumeProfileEngine
from aphelion.features.vwap import VWAPCalculator
from aphelion.features.sessions import SessionFeatures
from aphelion.features.mtf import MTFAlignmentEngine
from aphelion.features.cointegration import CointegrationEngine
from aphelion.features.halftrend import HalfTrendCalculator
from aphelion.features.registry import get_registry, FeatureRecord


class FeatureEngine:
    """
    Master feature pipeline. Orchestrates all sub-engines and produces
    a unified feature dict on every bar update.
    """

    def __init__(self, data_layer: DataLayer, clock: Optional[MarketClock] = None):
        self._data = data_layer
        self._clock = clock or MarketClock()

        # Sub-engines
        self._micro = MicrostructureEngine()
        self._structure = MarketStructureEngine()
        self._volume = VolumeProfileEngine()
        self._vwap = {tf: VWAPCalculator() for tf in TIMEFRAMES}
        self._sessions = SessionFeatures(self._clock)
        self._mtf = MTFAlignmentEngine()
        self._cointegration = CointegrationEngine()
        self._halftrend = HalfTrendCalculator()
        self._registry = get_registry()

        # Technical indicator caches
        self._atr_cache: dict[Timeframe, float] = {}
        self._bb_cache: dict[Timeframe, dict] = {}

        # External data for cointegration (populated by ATLAS/DATA)
        self._external_prices: dict[str, np.ndarray] = {}

        # v2: tick velocity tracking
        self._last_tick_ts: float = 0.0
        self._tick_intervals: list[float] = []

    def on_tick(self, tick: Tick) -> dict:
        """Process a raw tick. Returns microstructure features."""
        # v2: tick velocity tracking
        now_ts = tick.timestamp.timestamp() if hasattr(tick.timestamp, 'timestamp') else float(tick.timestamp)
        if self._last_tick_ts > 0:
            interval = now_ts - self._last_tick_ts
            if interval > 0:
                self._tick_intervals.append(interval)
                if len(self._tick_intervals) > 200:
                    self._tick_intervals = self._tick_intervals[-200:]
        self._last_tick_ts = now_ts

        return self._micro.update(
            timestamp=tick.timestamp,
            bid=tick.bid,
            ask=tick.ask,
            last_price=tick.last,
            volume=tick.volume,
        ).vpin  # Return just VPIN for quick access; full features on bar

    def on_bar(self, bar: Bar) -> dict:
        """
        Process a completed bar. Computes all features for this timeframe.
        Returns the full feature dictionary.
        """
        tf = bar.timeframe
        df = self._data.get_bars_df(tf, count=500)

        if df.empty:
            return {}

        features = {}

        # 1. Microstructure (already updated per-tick, grab latest state)
        features.update(self._micro.to_dict())

        # 2. Market structure
        if len(df) >= 20:
            structure_features = self._structure.compute_all(df)
            features.update(structure_features)

        # 3. Volume profile
        self._volume.update_bar(bar.open, bar.high, bar.low, bar.close, bar.volume)
        self._volume.compute_session_profile(df)
        features.update(self._volume.to_dict())

        # 4. VWAP
        self._vwap[tf].update(bar.high, bar.low, bar.close, bar.volume)
        features.update(self._vwap[tf].to_dict())

        # 5. Technical indicators
        features.update(self._compute_technicals(df, tf))

        # 6. Session features
        features.update(self._sessions.compute())

        # 7. MTF alignment (only on M5+ bars to avoid noise)
        if tf != Timeframe.M1:
            bars_by_tf = {}
            for t in TIMEFRAMES:
                t_df = self._data.get_bars_df(t, count=100)
                if not t_df.empty:
                    bars_by_tf[t] = t_df
            features.update(self._mtf.compute(bars_by_tf))

        # 8. Cointegration (on H1 bars only — too expensive for M1)
        if tf == Timeframe.H1 and self._external_prices:
            xau_prices = df["close"].values
            self._external_prices["XAUUSD"] = xau_prices
            coint_features = self._cointegration.compute_all(self._external_prices)
            features["any_cointegrated"] = coint_features["any_cointegrated"]
            features["max_spread_zscore"] = coint_features["max_spread_zscore"]

        # 9. HalfTrend indicator
        if len(df) >= 101:
            ht_state = self._halftrend.compute(
                df["high"].values, df["low"].values, df["close"].values
            )
            features.update(self._halftrend.to_dict(ht_state))

        # 10. Tick velocity (ticks per second)
        if self._tick_intervals:
            avg_interval = np.mean(self._tick_intervals[-50:])
            features["tick_velocity"] = 1.0 / avg_interval if avg_interval > 0 else 0.0
        else:
            features["tick_velocity"] = 0.0

        # 11. Efficiency ratio (Kaufman) — directional movement / total movement
        _closes = df["close"].values
        if len(_closes) >= 20:
            direction_move = abs(_closes[-1] - _closes[-20])
            total_move = np.sum(np.abs(np.diff(_closes[-20:])))
            features["efficiency_ratio"] = direction_move / total_move if total_move > 0 else 0.0
        else:
            features["efficiency_ratio"] = 0.0

        # Add metadata
        features["timeframe"] = tf.value
        features["bar_timestamp"] = str(bar.timestamp)

        # v2: Sanitize — replace NaN/inf with safe defaults
        return self._validate_features(features)

    @staticmethod
    def _validate_features(features: dict) -> dict:
        """Replace NaN/inf values with 0.0 to prevent downstream model garbage."""
        sanitized = {}
        for key, value in features.items():
            if isinstance(value, float):
                if np.isnan(value) or np.isinf(value):
                    sanitized[key] = 0.0
                else:
                    sanitized[key] = value
            elif isinstance(value, np.floating):
                v = float(value)
                if np.isnan(v) or np.isinf(v):
                    sanitized[key] = 0.0
                else:
                    sanitized[key] = v
            else:
                sanitized[key] = value
        return sanitized

        return features

    def set_external_prices(self, symbol: str, prices: np.ndarray) -> None:
        """Set external asset prices for cointegration analysis."""
        self._external_prices[symbol] = prices

    def set_mtf_weights(self, weights: dict[Timeframe, float]) -> None:
        """Set dynamic MTF weights from MERIDIAN."""
        self._mtf.set_weights(weights)

    def reset_session(self) -> None:
        """Reset session-level accumulators (called at session open)."""
        for vwap in self._vwap.values():
            vwap.reset_session()
        self._volume.reset_session()

    def _compute_technicals(self, df: pd.DataFrame, tf: Timeframe) -> dict:
        """Compute standard technical indicators."""
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        volumes = df["volume"].values if "volume" in df.columns else np.ones(len(closes))

        features = {}

        # ATR (14-period)
        if len(df) >= 15:
            atr = self._compute_atr(highs, lows, closes, period=14)
            self._atr_cache[tf] = atr
            features["atr"] = atr

        # Bollinger Bands (20-period, 2 std)
        if len(closes) >= 20:
            bb = self._compute_bollinger(closes, period=20, std_mult=2.0)
            self._bb_cache[tf] = bb
            features.update(bb)

        # RSI (14-period)
        if len(closes) >= 15:
            features["rsi"] = self._compute_rsi(closes, period=14)

        # EMA 20, 50
        if len(closes) >= 50:
            features["ema_20"] = self._ema(closes, 20)
            features["ema_50"] = self._ema(closes, 50)
            features["ema_cross"] = 1 if features["ema_20"] > features["ema_50"] else -1

        # MACD (12, 26, 9)
        if len(closes) >= 35:
            features.update(self._compute_macd(closes))

        # Stochastic %K / %D (14, 3)
        if len(closes) >= 17:
            features.update(self._compute_stochastic(highs, lows, closes))

        # ADX (14-period)
        if len(df) >= 28:
            features["adx"] = self._compute_adx(highs, lows, closes, period=14)

        # OBV (On-Balance Volume)
        if len(closes) >= 2:
            features["obv"] = self._compute_obv(closes, volumes)

        # Rate of Change (10-period)
        if len(closes) >= 11:
            features["roc_10"] = ((closes[-1] - closes[-11]) / closes[-11]) * 100.0

        # ── Advanced Indicators (v3) ─────────────────────────────────────

        # Hurst exponent — trend vs mean-reversion detection
        if len(closes) >= 100:
            features["hurst_exponent"] = self._compute_hurst(closes[-256:] if len(closes) >= 256 else closes)

        # Keltner Channels (20-period EMA, 1.5 ATR)
        if len(closes) >= 20 and "atr" in features:
            kc = self._compute_keltner(closes, features["atr"], period=20, mult=1.5)
            features.update(kc)

        # Williams %R (14-period)
        if len(closes) >= 14:
            features["williams_r"] = self._compute_williams_r(highs, lows, closes, period=14)

        # Chaikin Money Flow (20-period)
        if len(closes) >= 20:
            features["cmf"] = self._compute_cmf(highs, lows, closes, volumes, period=20)

        # Fisher Transform (9-period)
        if len(closes) >= 10:
            ft = self._compute_fisher_transform(highs, lows, period=9)
            features.update(ft)

        # Fractal dimension (Higuchi method)
        if len(closes) >= 64:
            features["fractal_dimension"] = self._compute_fractal_dimension(closes[-128:] if len(closes) >= 128 else closes)

        # Price entropy (information content of returns)
        if len(closes) >= 50:
            features["price_entropy"] = self._compute_price_entropy(closes[-100:] if len(closes) >= 100 else closes)

        # Wavelet denoised trend (Daubechies-4 approximation)
        if len(closes) >= 32:
            wd = self._compute_wavelet_trend(closes)
            features.update(wd)

        # Squeeze detection (Bollinger inside Keltner)
        if "bb_upper" in features and "kc_upper" in features:
            features["squeeze_on"] = 1 if (
                features["bb_upper"] < features["kc_upper"]
                and features["bb_lower"] > features["kc_lower"]
            ) else 0

        return features

    @staticmethod
    def _compute_atr(highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, period: int = 14) -> float:
        """Average True Range."""
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        if len(tr) < period:
            return float(np.mean(tr)) if len(tr) > 0 else 0.0
        return float(np.mean(tr[-period:]))

    @staticmethod
    def _compute_bollinger(closes: np.ndarray, period: int = 20,
                           std_mult: float = 2.0) -> dict:
        """Bollinger Bands."""
        sma = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        upper = sma + std_mult * std
        lower = sma - std_mult * std
        width = (upper - lower) / sma if sma > 0 else 0

        return {
            "bb_upper": upper,
            "bb_middle": sma,
            "bb_lower": lower,
            "bb_width": width,
            "bb_percentile": (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5,
        }

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
        """Relative Strength Index using Wilder's exponential smoothing.
        Matches MT5/TradingView standard RSI calculation.
        """
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        if len(gains) < period:
            avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        else:
            # Wilder's smoothing: first average is SMA, subsequent are exponential
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        """Exponential Moving Average (last value)."""
        multiplier = 2.0 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = (price - ema) * multiplier + ema
        return float(ema)

    @staticmethod
    def _compute_macd(closes: np.ndarray,
                      fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        """MACD line, signal line, and histogram."""
        def _ema_series(arr: np.ndarray, period: int) -> np.ndarray:
            mult = 2.0 / (period + 1)
            out = np.empty_like(arr, dtype=float)
            out[0] = arr[0]
            for i in range(1, len(arr)):
                out[i] = (arr[i] - out[i - 1]) * mult + out[i - 1]
            return out

        ema_fast = _ema_series(closes, fast)
        ema_slow = _ema_series(closes, slow)
        macd_line = ema_fast - ema_slow
        signal_line = _ema_series(macd_line, signal)
        histogram = macd_line - signal_line

        return {
            "macd_line": float(macd_line[-1]),
            "macd_signal": float(signal_line[-1]),
            "macd_histogram": float(histogram[-1]),
        }

    @staticmethod
    def _compute_stochastic(highs: np.ndarray, lows: np.ndarray,
                            closes: np.ndarray,
                            k_period: int = 14, d_period: int = 3) -> dict:
        """Stochastic %K and %D."""
        n = len(closes)
        k_values = []
        for i in range(k_period - 1, n):
            high_n = np.max(highs[i - k_period + 1:i + 1])
            low_n = np.min(lows[i - k_period + 1:i + 1])
            if high_n == low_n:
                k_values.append(50.0)
            else:
                k_values.append(((closes[i] - low_n) / (high_n - low_n)) * 100.0)

        k_arr = np.array(k_values)
        if len(k_arr) >= d_period:
            d_val = float(np.mean(k_arr[-d_period:]))
        else:
            d_val = float(k_arr[-1]) if len(k_arr) > 0 else 50.0

        return {
            "stoch_k": float(k_arr[-1]) if len(k_arr) > 0 else 50.0,
            "stoch_d": d_val,
        }

    @staticmethod
    def _compute_adx(highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, period: int = 14) -> float:
        """Average Directional Index."""
        n = len(highs)
        if n < 2 * period:
            return 0.0

        up_moves = highs[1:] - highs[:-1]
        down_moves = lows[:-1] - lows[1:]

        plus_dm = np.where((up_moves > down_moves) & (up_moves > 0), up_moves, 0.0)
        minus_dm = np.where((down_moves > up_moves) & (down_moves > 0), down_moves, 0.0)

        tr_vals = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )

        # Wilder smoothing
        atr_sum = np.sum(tr_vals[:period])
        pdm_sum = np.sum(plus_dm[:period])
        mdm_sum = np.sum(minus_dm[:period])

        dx_list = []
        for i in range(period, len(tr_vals)):
            atr_sum = atr_sum - atr_sum / period + tr_vals[i]
            pdm_sum = pdm_sum - pdm_sum / period + plus_dm[i]
            mdm_sum = mdm_sum - mdm_sum / period + minus_dm[i]

            if atr_sum == 0:
                dx_list.append(0.0)
                continue
            plus_di = 100.0 * pdm_sum / atr_sum
            minus_di = 100.0 * mdm_sum / atr_sum
            di_sum = plus_di + minus_di
            if di_sum == 0:
                dx_list.append(0.0)
            else:
                dx_list.append(100.0 * abs(plus_di - minus_di) / di_sum)

        if not dx_list:
            return 0.0
        # Smooth DX → ADX
        adx = np.mean(dx_list[:period]) if len(dx_list) >= period else np.mean(dx_list)
        for i in range(period, len(dx_list)):
            adx = (adx * (period - 1) + dx_list[i]) / period
        return float(adx)

    @staticmethod
    def _compute_obv(closes: np.ndarray, volumes: np.ndarray) -> float:
        """On-Balance Volume (last value)."""
        obv = 0.0
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv += volumes[i]
            elif closes[i] < closes[i - 1]:
                obv -= volumes[i]
        return float(obv)

    # ── Advanced Algorithms (v3) ─────────────────────────────────────────

    @staticmethod
    def _compute_hurst(prices: np.ndarray) -> float:
        """
        Hurst exponent via Rescaled Range (R/S) analysis.
        H < 0.5 = mean-reverting, H = 0.5 = random walk, H > 0.5 = trending.
        Uses multiple window sizes for robust estimation.
        """
        n = len(prices)
        if n < 20:
            return 0.5

        log_returns = np.diff(np.log(np.maximum(prices, 1e-10)))
        # Use window sizes from 10 to n//2
        min_w = 10
        max_w = n // 2
        if max_w <= min_w:
            return 0.5

        window_sizes = np.unique(np.logspace(
            np.log10(min_w), np.log10(max_w), num=15, dtype=int
        ))
        window_sizes = window_sizes[window_sizes >= min_w]

        log_rs = []
        log_n = []

        for w in window_sizes:
            w = int(w)
            n_segments = len(log_returns) // w
            if n_segments < 1:
                continue

            rs_values = []
            for seg in range(n_segments):
                segment = log_returns[seg * w:(seg + 1) * w]
                mean_seg = np.mean(segment)
                deviations = np.cumsum(segment - mean_seg)
                r = np.max(deviations) - np.min(deviations)
                s = np.std(segment, ddof=1)
                if s > 1e-12:
                    rs_values.append(r / s)

            if rs_values:
                log_rs.append(np.log(np.mean(rs_values)))
                log_n.append(np.log(w))

        if len(log_rs) < 3:
            return 0.5

        # Linear regression: log(R/S) = H * log(n) + c
        log_n_arr = np.array(log_n)
        log_rs_arr = np.array(log_rs)
        A = np.column_stack([log_n_arr, np.ones(len(log_n_arr))])
        try:
            result = np.linalg.lstsq(A, log_rs_arr, rcond=None)
            hurst = float(result[0][0])
        except np.linalg.LinAlgError:
            return 0.5

        return max(0.0, min(1.0, hurst))

    @staticmethod
    def _compute_keltner(closes: np.ndarray, atr: float,
                         period: int = 20, mult: float = 1.5) -> dict:
        """Keltner Channels: EMA ± ATR multiplier."""
        multiplier = 2.0 / (period + 1)
        ema = closes[0]
        for price in closes[1:]:
            ema = (price - ema) * multiplier + ema

        upper = ema + mult * atr
        lower = ema - mult * atr
        width = (upper - lower) / ema if ema > 0 else 0.0

        return {
            "kc_upper": upper,
            "kc_middle": ema,
            "kc_lower": lower,
            "kc_width": width,
            "kc_percentile": (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5,
        }

    @staticmethod
    def _compute_williams_r(highs: np.ndarray, lows: np.ndarray,
                            closes: np.ndarray, period: int = 14) -> float:
        """Williams %R: momentum oscillator (-100 to 0)."""
        highest = np.max(highs[-period:])
        lowest = np.min(lows[-period:])
        if highest == lowest:
            return -50.0
        return float(-100.0 * (highest - closes[-1]) / (highest - lowest))

    @staticmethod
    def _compute_cmf(highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, volumes: np.ndarray,
                     period: int = 20) -> float:
        """
        Chaikin Money Flow: volume-weighted accumulation/distribution.
        Positive = buying pressure, Negative = selling pressure.
        """
        h = highs[-period:]
        l = lows[-period:]
        c = closes[-period:]
        v = volumes[-period:]

        hl_range = h - l
        # Money Flow Multiplier: ((close - low) - (high - close)) / (high - low)
        mfm = np.where(hl_range > 0, ((c - l) - (h - c)) / hl_range, 0.0)
        mf_volume = mfm * v
        total_vol = np.sum(v)

        if total_vol == 0:
            return 0.0
        return float(np.sum(mf_volume) / total_vol)

    @staticmethod
    def _compute_fisher_transform(highs: np.ndarray, lows: np.ndarray,
                                  period: int = 9) -> dict:
        """
        Fisher Transform: normalizes prices to Gaussian distribution.
        Produces clear turning-point signals.
        """
        n = len(highs)
        if n < period:
            return {"fisher": 0.0, "fisher_signal": 0.0}

        # Compute median price
        median_prices = (highs + lows) / 2.0

        # Normalize to [-1, 1] using rolling min/max
        highest = np.max(median_prices[-period:])
        lowest = np.min(median_prices[-period:])

        if highest == lowest:
            return {"fisher": 0.0, "fisher_signal": 0.0}

        # Raw value clamped to (-0.999, 0.999) to avoid log singularity
        raw = 2.0 * (median_prices[-1] - lowest) / (highest - lowest) - 1.0
        raw = max(-0.999, min(0.999, raw))

        # Fisher transform: 0.5 * ln((1+x)/(1-x)) with EMA smoothing
        fisher = 0.5 * np.log((1.0 + raw) / (1.0 - raw))

        # Signal line: previous value (simplified)
        raw_prev = 2.0 * (median_prices[-2] - lowest) / (highest - lowest) - 1.0 if n >= 2 else 0.0
        raw_prev = max(-0.999, min(0.999, raw_prev))
        fisher_signal = 0.5 * np.log((1.0 + raw_prev) / (1.0 - raw_prev))

        return {
            "fisher": float(fisher),
            "fisher_signal": float(fisher_signal),
        }

    @staticmethod
    def _compute_fractal_dimension(prices: np.ndarray, k_max: int = 10) -> float:
        """
        Higuchi Fractal Dimension — measures price curve complexity.
        FD ≈ 1.0 = smooth trend, FD ≈ 1.5 = random walk, FD ≈ 2.0 = very noisy.
        Reference: T. Higuchi, "Approach to an irregular time series on the
        basis of the fractal theory", Physica D, 1988.
        """
        n = len(prices)
        k_max = min(k_max, n // 4)
        if k_max < 2:
            return 1.5

        log_lengths = []
        log_k = []

        for k in range(1, k_max + 1):
            lengths_k = []
            for m in range(1, k + 1):
                # Construct sub-series x(m), x(m+k), x(m+2k), ...
                indices = np.arange(m - 1, n, k)
                if len(indices) < 2:
                    continue
                sub = prices[indices]
                # Length of this sub-series
                L = np.sum(np.abs(np.diff(sub))) * (n - 1) / (k * (len(indices) - 1) * k)
                lengths_k.append(L)

            if lengths_k:
                avg_length = np.mean(lengths_k)
                if avg_length > 0:
                    log_lengths.append(np.log(avg_length))
                    log_k.append(np.log(1.0 / k))

        if len(log_lengths) < 3:
            return 1.5

        # Linear regression slope = fractal dimension
        x = np.array(log_k)
        y = np.array(log_lengths)
        A = np.column_stack([x, np.ones(len(x))])
        try:
            result = np.linalg.lstsq(A, y, rcond=None)
            fd = float(result[0][0])
        except np.linalg.LinAlgError:
            return 1.5

        return max(1.0, min(2.0, fd))

    @staticmethod
    def _compute_price_entropy(prices: np.ndarray, n_bins: int = 20) -> float:
        """
        Shannon entropy of return distribution.
        High entropy = unpredictable; Low entropy = structured/trending.
        Normalized to [0, 1].
        """
        returns = np.diff(np.log(np.maximum(prices, 1e-10)))
        if len(returns) < 10:
            return 1.0

        # Histogram-based entropy
        counts, _ = np.histogram(returns, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]

        ent = float(sp_entropy(probs, base=2))
        max_entropy = np.log2(n_bins)

        return ent / max_entropy if max_entropy > 0 else 1.0

    @staticmethod
    def _compute_wavelet_trend(closes: np.ndarray) -> dict:
        """
        Haar wavelet decomposition for trend extraction.
        Uses iterative averaging (Haar wavelet) to extract low-frequency trend
        from noisy price data. Also computes denoised momentum.
        """
        n = len(closes)
        # Pad to nearest power of 2
        levels = 3  # 3 decomposition levels
        target_len = 1 << (int(np.ceil(np.log2(max(n, 8)))))
        padded = np.zeros(target_len)
        padded[:n] = closes
        padded[n:] = closes[-1]  # Pad with last value

        # Haar wavelet decomposition (manual for no pywt dependency)
        approx = padded.copy()
        details = []
        for _ in range(levels):
            half = len(approx) // 2
            if half < 1:
                break
            new_approx = np.zeros(half)
            detail = np.zeros(half)
            for j in range(half):
                new_approx[j] = (approx[2 * j] + approx[2 * j + 1]) / np.sqrt(2)
                detail[j] = (approx[2 * j] - approx[2 * j + 1]) / np.sqrt(2)
            details.append(detail)
            approx = new_approx

        # Reconstruct denoised signal (zero out highest-frequency detail)
        # This gives us the trend component
        denoised = approx
        for i in range(len(details) - 1, 0, -1):  # Skip finest detail [0]
            d = details[i]
            new_len = len(denoised) * 2
            reconstructed = np.zeros(new_len)
            for j in range(len(denoised)):
                reconstructed[2 * j] = (denoised[j] + d[j]) / np.sqrt(2)
                reconstructed[2 * j + 1] = (denoised[j] - d[j]) / np.sqrt(2)
            denoised = reconstructed

        # Pad back with zeros for finest detail
        d0 = np.zeros(len(denoised))  # Zero out finest detail for denoising
        new_len = len(denoised) * 2
        reconstructed = np.zeros(new_len)
        for j in range(len(denoised)):
            reconstructed[2 * j] = (denoised[j] + d0[j]) / np.sqrt(2)
            reconstructed[2 * j + 1] = (denoised[j] - d0[j]) / np.sqrt(2)
        denoised = reconstructed

        # Extract trend values at original series length
        trend = denoised[:n]
        trend_val = float(trend[-1]) if len(trend) > 0 else float(closes[-1])

        # Wavelet-denoised momentum: slope of the trend line
        if len(trend) >= 5:
            slope = float(trend[-1] - trend[-5]) / 5.0
        else:
            slope = 0.0

        # Noise ratio: std(detail) / std(signal)
        if details and len(details[0]) > 1:
            noise_energy = float(np.std(details[0]))
            signal_energy = float(np.std(closes)) if np.std(closes) > 0 else 1.0
            noise_ratio = noise_energy / signal_energy
        else:
            noise_ratio = 0.5

        return {
            "wavelet_trend": trend_val,
            "wavelet_momentum": slope,
            "wavelet_noise_ratio": float(min(noise_ratio, 2.0)),
            "wavelet_trend_strength": abs(slope) / (float(np.std(closes[-20:])) + 1e-10) if len(closes) >= 20 else 0.0,
        }
