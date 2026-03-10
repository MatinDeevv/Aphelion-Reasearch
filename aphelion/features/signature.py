"""
APHELION Signature Transform Features.

Implements truncated path signatures (levels 1 and 2) for financial time
series paths. These iterated-integral features encode nonlinear path
structure beyond standard indicators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class SignatureConfig:
    window: int = 64
    eps: float = 1e-9


class SignatureTransform:
    """
    Truncated signature transform up to level 2 for piecewise-linear paths.

    For increment dx and current signature (S1, S2):
      S1 <- S1 + dx
      S2 <- S2 + outer(S1_prev, dx) + 0.5 * outer(dx, dx)
    """

    def __init__(self, config: SignatureConfig | None = None):
        self._cfg = config or SignatureConfig()

    @staticmethod
    def _zscore(x: np.ndarray, eps: float) -> np.ndarray:
        mean = float(np.mean(x))
        std = float(np.std(x))
        if std < eps:
            return x - mean
        return (x - mean) / std

    def _build_path(self, close: np.ndarray, volume: np.ndarray, spread: np.ndarray) -> np.ndarray:
        eps = self._cfg.eps
        c = self._zscore(np.log(np.maximum(close, eps)), eps)
        v = self._zscore(np.log(np.maximum(volume, eps)), eps)
        s = self._zscore(spread, eps)
        return np.column_stack([c, v, s]).astype(float)

    @staticmethod
    def _signature_level2(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, d = path.shape
        if n < 2:
            return np.zeros(d, dtype=float), np.zeros((d, d), dtype=float)

        s1 = np.zeros(d, dtype=float)
        s2 = np.zeros((d, d), dtype=float)

        for i in range(1, n):
            dx = path[i] - path[i - 1]
            s2 = s2 + np.outer(s1, dx) + 0.5 * np.outer(dx, dx)
            s1 = s1 + dx

        return s1, s2

    def compute(self, close: Iterable[float], volume: Iterable[float], spread: Iterable[float]) -> dict[str, float]:
        close_arr = np.asarray(list(close), dtype=float)
        vol_arr = np.asarray(list(volume), dtype=float)
        spr_arr = np.asarray(list(spread), dtype=float)

        if len(close_arr) == 0:
            return self.zero_features()

        n = min(len(close_arr), len(vol_arr), len(spr_arr), self._cfg.window)
        close_arr = close_arr[-n:]
        vol_arr = vol_arr[-n:]
        spr_arr = spr_arr[-n:]

        if n < 4:
            return self.zero_features()

        path = self._build_path(close_arr, vol_arr, spr_arr)
        s1, s2 = self._signature_level2(path)

        # Levy area terms (antisymmetric part) capture path ordering effects.
        levy_pv = 0.5 * (s2[0, 1] - s2[1, 0])
        levy_ps = 0.5 * (s2[0, 2] - s2[2, 0])
        levy_vs = 0.5 * (s2[1, 2] - s2[2, 1])

        return {
            "sig1_price": float(s1[0]),
            "sig1_volume": float(s1[1]),
            "sig1_spread": float(s1[2]),
            "sig2_price_price": float(s2[0, 0]),
            "sig2_volume_volume": float(s2[1, 1]),
            "sig2_spread_spread": float(s2[2, 2]),
            "sig2_price_volume": float(s2[0, 1]),
            "sig2_volume_price": float(s2[1, 0]),
            "sig2_price_spread": float(s2[0, 2]),
            "sig2_spread_price": float(s2[2, 0]),
            "sig2_volume_spread": float(s2[1, 2]),
            "sig2_spread_volume": float(s2[2, 1]),
            "sig_levy_pv": float(levy_pv),
            "sig_levy_ps": float(levy_ps),
            "sig_levy_vs": float(levy_vs),
            "sig_l2_frobenius": float(np.linalg.norm(s2, ord="fro")),
        }

    @staticmethod
    def zero_features() -> dict[str, float]:
        return {
            "sig1_price": 0.0,
            "sig1_volume": 0.0,
            "sig1_spread": 0.0,
            "sig2_price_price": 0.0,
            "sig2_volume_volume": 0.0,
            "sig2_spread_spread": 0.0,
            "sig2_price_volume": 0.0,
            "sig2_volume_price": 0.0,
            "sig2_price_spread": 0.0,
            "sig2_spread_price": 0.0,
            "sig2_volume_spread": 0.0,
            "sig2_spread_volume": 0.0,
            "sig_levy_pv": 0.0,
            "sig_levy_ps": 0.0,
            "sig_levy_vs": 0.0,
            "sig_l2_frobenius": 0.0,
        }
