"""
APHELION Cross-Impact Matrix Features.

Models next-step impact on target asset (default XAUUSD) from lagged order-flow
proxies in correlated instruments via rolling ridge regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CrossImpactConfig:
    window: int = 180
    ridge: float = 1e-3
    target_symbol: str = "XAUUSD"


class CrossImpactMatrix:
    def __init__(self, config: Optional[CrossImpactConfig] = None):
        self._cfg = config or CrossImpactConfig()
        self._symbols: list[str] = []
        self._matrix: Optional[np.ndarray] = None

    @staticmethod
    def _safe_returns(prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=float)
        prices = np.maximum(prices, 1e-9)
        return np.diff(np.log(prices))

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        std = float(np.std(x))
        if std < 1e-12:
            return np.zeros_like(x)
        return (x - float(np.mean(x))) / std

    def fit(
        self,
        price_series: dict[str, np.ndarray],
        order_flow_series: Optional[dict[str, np.ndarray]] = None,
    ) -> dict[str, float]:
        # Require target plus at least one peer instrument.
        if self._cfg.target_symbol not in price_series or len(price_series) < 2:
            return self.zero_features()

        symbols = sorted(price_series.keys())

        # Align history length
        raw_lengths = [len(price_series[s]) for s in symbols]
        min_len = min(raw_lengths)
        needed = self._cfg.window + 2
        if min_len < needed:
            return self.zero_features()

        # Build aligned return and flow matrices.
        returns_map: dict[str, np.ndarray] = {}
        flow_map: dict[str, np.ndarray] = {}
        for sym in symbols:
            prices = np.asarray(price_series[sym], dtype=float)[-needed:]
            ret = self._safe_returns(prices)  # length window+1
            returns_map[sym] = ret

            if order_flow_series and sym in order_flow_series:
                flow_raw = np.asarray(order_flow_series[sym], dtype=float)
                if len(flow_raw) >= len(ret):
                    flow = flow_raw[-len(ret):]
                else:
                    padded = np.zeros(len(ret), dtype=float)
                    padded[-len(flow_raw):] = flow_raw
                    flow = padded
            else:
                # No explicit flow feed: use standardized return innovations as proxy.
                flow = ret

            flow_map[sym] = self._zscore(flow)

        # All series have same length now.
        n = len(symbols)
        t = len(next(iter(returns_map.values())))
        if t < 5:
            return self.zero_features()

        m = t - 1
        X = np.column_stack([flow_map[s][:-1] for s in symbols])  # lagged flows

        mat = np.zeros((n, n), dtype=float)
        ridge = self._cfg.ridge
        XtX = X.T @ X
        reg = XtX + ridge * np.eye(n)

        try:
            reg_inv = np.linalg.inv(reg)
        except np.linalg.LinAlgError:
            reg_inv = np.linalg.pinv(reg)

        for i, target in enumerate(symbols):
            y = returns_map[target][1:]
            beta = reg_inv @ (X.T @ y)
            mat[i] = beta

        self._symbols = symbols
        self._matrix = mat
        return self.current_features(flow_map)

    def current_features(self, flow_map: dict[str, np.ndarray]) -> dict[str, float]:
        if self._matrix is None or not self._symbols:
            return self.zero_features()

        symbols = self._symbols
        target = self._cfg.target_symbol
        if target not in symbols:
            return self.zero_features()

        idx = symbols.index(target)
        beta_row = self._matrix[idx]
        latest_flow = np.array([flow_map[s][-1] for s in symbols], dtype=float)

        pred_return = float(beta_row @ latest_flow)
        off_diag = np.array([abs(beta_row[j]) for j in range(len(symbols)) if j != idx], dtype=float)
        strength = float(np.linalg.norm(off_diag)) if len(off_diag) > 0 else 0.0
        max_beta = float(np.max(off_diag)) if len(off_diag) > 0 else 0.0

        entropy = 0.0
        if len(off_diag) > 0 and np.sum(off_diag) > 0:
            p = off_diag / np.sum(off_diag)
            entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p) + 1e-12))

        signal = float(np.tanh(pred_return * 50.0))

        features = self.zero_features()
        features.update(
            {
                "cross_impact_pred_return": pred_return,
                "cross_impact_signal": signal,
                "cross_impact_strength": strength,
                "cross_impact_entropy": entropy,
                "cross_impact_max_beta": max_beta,
            }
        )

        # Common macro/metal driver betas (stable feature names for HYDRA).
        sym_to_key = {
            "DXY": "cross_impact_beta_dxy",
            "TLT": "cross_impact_beta_tlt",
            "XAGUSD": "cross_impact_beta_xagusd",
            "CL": "cross_impact_beta_oil",
            "WTI": "cross_impact_beta_oil",
            "GLD": "cross_impact_beta_gld",
        }
        for j, sym in enumerate(symbols):
            if j == idx:
                continue
            key = sym_to_key.get(sym.upper())
            if key:
                features[key] = float(beta_row[j])

        return features

    @staticmethod
    def zero_features() -> dict[str, float]:
        return {
            "cross_impact_pred_return": 0.0,
            "cross_impact_signal": 0.0,
            "cross_impact_strength": 0.0,
            "cross_impact_entropy": 0.0,
            "cross_impact_max_beta": 0.0,
            "cross_impact_beta_dxy": 0.0,
            "cross_impact_beta_tlt": 0.0,
            "cross_impact_beta_xagusd": 0.0,
            "cross_impact_beta_oil": 0.0,
            "cross_impact_beta_gld": 0.0,
        }
