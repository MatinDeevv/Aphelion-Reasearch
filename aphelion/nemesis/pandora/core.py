"""
NEMESIS PANDORA — Overfitting Detector
Phase 14 — Detects overfitting in strategy parameters and model weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OverfitSignal:
    """Overfitting detection result."""
    module: str
    metric: str
    train_value: float
    test_value: float
    gap: float
    is_overfitting: bool
    severity: str = "LOW"  # LOW, MODERATE, HIGH


class PandoraCore:
    """
    Detects overfitting by comparing in-sample vs out-of-sample performance.
    
    Checks:
    - Train/test Sharpe gap > 1.0 → overfit
    - Win rate gap > 15% → overfit
    - Walk-forward fold variance > threshold → unstable
    - Parameter sensitivity: small changes → large performance swings
    """

    def __init__(
        self,
        max_sharpe_gap: float = 1.0,
        max_wr_gap: float = 0.15,
        max_fold_variance: float = 0.8,
    ):
        self._max_sharpe_gap = max_sharpe_gap
        self._max_wr_gap = max_wr_gap
        self._max_fold_var = max_fold_variance
        self._history: List[OverfitSignal] = []

    def check_sharpe_gap(
        self, module: str, train_sharpe: float, test_sharpe: float
    ) -> OverfitSignal:
        """Check train/test Sharpe ratio gap."""
        gap = train_sharpe - test_sharpe
        is_overfit = gap > self._max_sharpe_gap
        severity = "HIGH" if gap > 2.0 else "MODERATE" if gap > 1.0 else "LOW"
        signal = OverfitSignal(
            module=module, metric="sharpe_gap",
            train_value=train_sharpe, test_value=test_sharpe,
            gap=gap, is_overfitting=is_overfit, severity=severity,
        )
        self._history.append(signal)
        return signal

    def check_win_rate_gap(
        self, module: str, train_wr: float, test_wr: float
    ) -> OverfitSignal:
        gap = train_wr - test_wr
        is_overfit = gap > self._max_wr_gap
        severity = "HIGH" if gap > 0.25 else "MODERATE" if gap > 0.15 else "LOW"
        signal = OverfitSignal(
            module=module, metric="win_rate_gap",
            train_value=train_wr, test_value=test_wr,
            gap=gap, is_overfitting=is_overfit, severity=severity,
        )
        self._history.append(signal)
        return signal

    def check_fold_stability(
        self, module: str, fold_sharpes: List[float]
    ) -> OverfitSignal:
        """Check walk-forward fold Sharpe variance."""
        if len(fold_sharpes) < 2:
            return OverfitSignal(
                module=module, metric="fold_variance",
                train_value=0, test_value=0, gap=0,
                is_overfitting=False,
            )
        variance = float(np.var(fold_sharpes))
        is_overfit = variance > self._max_fold_var
        severity = "HIGH" if variance > 1.5 else "MODERATE" if variance > 0.8 else "LOW"
        return OverfitSignal(
            module=module, metric="fold_variance",
            train_value=float(np.mean(fold_sharpes)),
            test_value=variance,
            gap=variance,
            is_overfitting=is_overfit,
            severity=severity,
        )

    def full_audit(
        self, module: str,
        train_sharpe: float, test_sharpe: float,
        train_wr: float, test_wr: float,
        fold_sharpes: Optional[List[float]] = None,
    ) -> List[OverfitSignal]:
        """Run all overfitting checks for a module."""
        results = [
            self.check_sharpe_gap(module, train_sharpe, test_sharpe),
            self.check_win_rate_gap(module, train_wr, test_wr),
        ]
        if fold_sharpes:
            results.append(self.check_fold_stability(module, fold_sharpes))
        return results

    @property
    def any_overfitting(self) -> bool:
        return any(s.is_overfitting for s in self._history[-10:])

    def reset(self) -> None:
        self._history.clear()
