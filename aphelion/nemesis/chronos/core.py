"""
NEMESIS CHRONOS — Temporal Anomaly Detection
Phase 14 — Detects time-based anomalies that indicate regime shifts.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TemporalAnomaly:
    """Detected temporal anomaly."""
    anomaly_type: str
    severity: float         # [0, 1]
    timestamp: float
    description: str = ""


class ChronosCore:
    """
    Detects temporal patterns that precede regime transitions:
    - Abnormal tick rate changes
    - Session boundary anomalies
    - Volume time-distribution shifts
    - Periodicity breaks
    """

    def __init__(self, window_size: int = 100):
        self._window = window_size
        self._tick_rates: List[float] = []
        self._volume_by_hour: Dict[int, List[float]] = {}
        self._anomalies: List[TemporalAnomaly] = []

    def record_tick_rate(self, ticks_per_second: float) -> Optional[TemporalAnomaly]:
        """Record tick rate and check for anomalies."""
        self._tick_rates.append(ticks_per_second)
        if len(self._tick_rates) > self._window * 2:
            self._tick_rates = self._tick_rates[-self._window:]

        if len(self._tick_rates) < self._window:
            return None

        recent = np.array(self._tick_rates[-20:])
        baseline = np.array(self._tick_rates[-self._window:-20])
        if len(baseline) < 10:
            return None

        z = (np.mean(recent) - np.mean(baseline)) / (np.std(baseline) + 1e-10)
        if abs(z) > 3.0:
            anomaly = TemporalAnomaly(
                anomaly_type="TICK_RATE_ANOMALY",
                severity=min(1.0, abs(z) / 5.0),
                timestamp=time.time(),
                description=f"Tick rate z-score: {z:.2f}",
            )
            self._anomalies.append(anomaly)
            return anomaly
        return None

    def record_volume(self, hour: int, volume: float) -> None:
        """Record volume for time-distribution tracking."""
        if hour not in self._volume_by_hour:
            self._volume_by_hour[hour] = []
        self._volume_by_hour[hour].append(volume)
        if len(self._volume_by_hour[hour]) > self._window:
            self._volume_by_hour[hour] = self._volume_by_hour[hour][-self._window:]

    def check_volume_anomaly(self, hour: int, current_volume: float) -> Optional[TemporalAnomaly]:
        """Check if current volume is anomalous for this time of day."""
        history = self._volume_by_hour.get(hour, [])
        if len(history) < 20:
            return None
        mean_vol = np.mean(history)
        std_vol = np.std(history)
        if std_vol < 1e-10:
            return None
        z = (current_volume - mean_vol) / std_vol
        if abs(z) > 3.0:
            return TemporalAnomaly(
                anomaly_type="VOLUME_TIME_ANOMALY",
                severity=min(1.0, abs(z) / 5.0),
                timestamp=time.time(),
                description=f"Volume at hour {hour}: z={z:.2f}",
            )
        return None

    @property
    def recent_anomalies(self) -> List[TemporalAnomaly]:
        return self._anomalies[-20:]

    def reset(self) -> None:
        self._tick_rates.clear()
        self._volume_by_hour.clear()
        self._anomalies.clear()
