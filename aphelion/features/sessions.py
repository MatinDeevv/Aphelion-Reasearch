"""
APHELION Session Features
Session flags, time calculations, news proximity, calendar features.
Section 5.4 of the Engineering Spec.
Wraps MarketClock for feature-engine integration.
"""

from aphelion.core.clock import MarketClock


class SessionFeatures:
    """Thin wrapper over MarketClock that produces a feature dict."""

    def __init__(self, clock: MarketClock | None = None):
        self._clock = clock or MarketClock()

    @property
    def clock(self) -> MarketClock:
        return self._clock

    def compute(self, dt=None) -> dict:
        """Return all session features as a flat dict."""
        return self._clock.session_features(dt)
