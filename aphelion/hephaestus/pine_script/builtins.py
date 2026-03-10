"""
HEPHAESTUS — Pine Script built-in translations

Pure-numpy reference implementations of the most common Pine Script
built-in functions.  These are included in codegen prompts as
few-shot examples so the LLM generates correct implementations.
"""

from __future__ import annotations

# Each entry is a (function_name, numpy_implementation_source) pair
# that gets included verbatim in the code-generation prompt as reference.

BUILTIN_IMPLEMENTATIONS: dict[str, str] = {
    "compute_ema": """\
@staticmethod
def _compute_ema(data: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    ema = np.empty_like(data, dtype=np.float64)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema
""",

    "compute_sma": """\
@staticmethod
def _compute_sma(data: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(data, np.nan, dtype=np.float64)
    for i in range(period - 1, len(data)):
        out[i] = np.mean(data[i - period + 1 : i + 1])
    return out
""",

    "compute_rsi": """\
@staticmethod
def _compute_rsi(data: np.ndarray, period: int) -> np.ndarray:
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rsi = np.full(len(data), 50.0)
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss < 1e-10:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi
""",

    "compute_atr": """\
@staticmethod
def _compute_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int,
) -> np.ndarray:
    tr = np.empty(len(high), dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = np.empty_like(tr)
    atr[:period] = np.nan
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr
""",

    "compute_macd": """\
@staticmethod
def _compute_macd(
    data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _ema(src, p):
        a = 2.0 / (p + 1)
        out = np.empty_like(src, dtype=np.float64)
        out[0] = src[0]
        for i in range(1, len(src)):
            out[i] = a * src[i] + (1 - a) * out[i - 1]
        return out
    fast_ema = _ema(data, fast)
    slow_ema = _ema(data, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
""",

    "compute_bollinger": """\
@staticmethod
def _compute_bollinger(
    data: np.ndarray, period: int, mult: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = np.full_like(data, np.nan, dtype=np.float64)
    upper = np.full_like(data, np.nan, dtype=np.float64)
    lower = np.full_like(data, np.nan, dtype=np.float64)
    for i in range(period - 1, len(data)):
        window = data[i - period + 1 : i + 1]
        m = np.mean(window)
        s = np.std(window, ddof=0)
        basis[i] = m
        upper[i] = m + mult * s
        lower[i] = m - mult * s
    return upper, basis, lower
""",
}
