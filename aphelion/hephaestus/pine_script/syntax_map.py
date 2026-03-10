"""
HEPHAESTUS — Pine Script Syntax Map

Mapping from Pine Script v4/v5 concepts to Python/numpy equivalents.
Used by the parser to assist LLM understanding and by the validator
to verify generated code implements correct translations.
"""

from __future__ import annotations

# ─── Pine Script → Python concept mapping ────────────────────────────────────

PINE_TO_PYTHON: dict[str, str] = {
    # Data series
    "close":              "bars[:, 4]",
    "open":               "bars[:, 1]",
    "high":               "bars[:, 2]",
    "low":                "bars[:, 3]",
    "volume":             "bars[:, 5]",
    "time":               "bars[:, 0]",
    "hl2":                "(bars[:, 2] + bars[:, 3]) / 2",
    "hlc3":               "(bars[:, 2] + bars[:, 3] + bars[:, 4]) / 3",
    "ohlc4":              "(bars[:, 1] + bars[:, 2] + bars[:, 3] + bars[:, 4]) / 4",

    # Built-in functions
    "ta.ema(src, len)":   "compute_ema(src, len)",
    "ta.sma(src, len)":   "np.mean(src[-len:])",
    "ta.rsi(src, len)":   "compute_rsi(src, len)",
    "ta.atr(len)":        "compute_atr(high, low, close, len)",
    "ta.macd(src, f, s, sig)": "compute_macd(src, fast, slow, signal)",
    "ta.bb(src, len, m)": "compute_bollinger(src, len, mult)",
    "ta.stoch(c,h,l,k,d,sm)": "compute_stochastic(close, high, low, k, d, smooth)",

    # Crossover / crossunder
    "ta.crossover(a, b)": "a[-1] > b[-1] and a[-2] <= b[-2]",
    "ta.crossunder(a, b)": "a[-1] < b[-1] and a[-2] >= b[-2]",
    "ta.cross(a, b)":     "(a[-1] > b[-1] and a[-2] <= b[-2]) or (a[-1] < b[-1] and a[-2] >= b[-2])",

    # Utility
    "nz(x, repl)":        "x if not np.isnan(x) else repl",
    "na(x)":              "np.isnan(x)",
    "math.abs(x)":        "abs(x)",
    "math.max(a, b)":     "max(a, b)",
    "math.min(a, b)":     "min(a, b)",
    "math.sqrt(x)":       "np.sqrt(x)",
    "math.log(x)":        "np.log(x)",
    "math.round(x)":      "round(x)",

    # Strategy entries
    "strategy.entry(id, long)":  "return Vote(direction=1, ...)",
    "strategy.entry(id, short)": "return Vote(direction=-1, ...)",
    "strategy.close(id)":        "return Vote(direction=0, ...)",
    "strategy.exit(id, ...)":    "exit condition logic",

    # Higher timeframe
    "request.security(sym, tf, expr)": "MTF feature lookup via ATLAS",
    "security(sym, tf, expr)":         "MTF feature lookup via ATLAS",

    # Bar state
    "barstate.isrealtime": "True  # always True in live mode",
    "barstate.isconfirmed": "True  # APHELION processes confirmed bars only",
    "barstate.islast":      "True",
}

# ─── Pine Script keywords used for input-type detection ──────────────────────

PINE_KEYWORDS: set[str] = {
    "//@version",
    "indicator(",
    "study(",
    "strategy(",
    "ta.ema",
    "ta.sma",
    "ta.rsi",
    "ta.atr",
    "ta.macd",
    "ta.bb",
    "ta.crossover",
    "ta.crossunder",
    "ta.stoch",
    "request.security",
    "plot(",
    "plotshape(",
    "alertcondition(",
    "strategy.entry",
    "strategy.exit",
    "strategy.close",
    "input.int(",
    "input.float(",
    "input(",
    "bgcolor(",
    "hline(",
}

# ─── Forbidden Pine Script patterns that have no Python equivalent ───────────

PINE_VISUAL_ONLY: set[str] = {
    "plot(",
    "plotshape(",
    "plotchar(",
    "plotarrow(",
    "bgcolor(",
    "barcolor(",
    "hline(",
    "fill(",
    "table.new(",
    "label.new(",
    "line.new(",
}
