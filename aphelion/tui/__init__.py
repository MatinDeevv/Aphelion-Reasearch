"""
APHELION TUI — Terminal User Interface v2 (Bloomberg-grade)

Textual-powered interactive dashboard with Rich fallback.
Multi-view layout, keyboard navigation, real-time sparklines,
risk gauges, and performance analytics.
"""

from aphelion.tui.app import AphelionTUI, TUIConfig
from aphelion.tui.bridge import TUIBridge
from aphelion.tui.state import (
    TUIState,
    HydraSignalView,
    SentinelView,
    EquityView,
    PositionView,
    PriceView,
    AlertEntry,
    LogEntry,
)

__all__ = [
    "AphelionTUI",
    "TUIConfig",
    "TUIBridge",
    "TUIState",
    "HydraSignalView",
    "SentinelView",
    "EquityView",
    "PositionView",
    "PriceView",
    "AlertEntry",
    "LogEntry",
]
