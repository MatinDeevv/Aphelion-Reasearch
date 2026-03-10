"""
APHELION MACRO — Market Regime & Macro Intelligence
Phase 12 — Engineering Spec v3.0

Provides regime classification, economic event awareness, DXY correlation,
and seasonality context. Commander-tier ARES voter.
"""

from .analyzer import MacroAnalyzer, MacroSignal
from .regime import RegimeClassifier, Regime
from .dxy import DXYMonitor, DXYBias
from .seasonality import GoldSeasonality
from .event_calendar import EconomicCalendar
from .sentiment import SentimentAnalyzer
from .atlas.core import AtlasCore, MacroContext
from .atlas.dxy_feed import DXYLiveFeed, DXYSnapshot
from .atlas.cot_parser import COTParser, COTRecord
from .atlas.event_blocker import EventBlocker, BlockWindow
from .argus.core import ArgusCore, MarketAnomaly
from .herald.core import HeraldCore, NewsEvent
from .nexus.core import NexusCore, NexusOutput
from .oracle.core import OracleCore, Forecast

__all__ = [
    "MacroAnalyzer",
    "MacroSignal",
    "RegimeClassifier",
    "Regime",
    "DXYMonitor",
    "DXYBias",
    "GoldSeasonality",
    "EconomicCalendar",
    "SentimentAnalyzer",
    "AtlasCore",
    "MacroContext",
    "DXYLiveFeed",
    "DXYSnapshot",
    "COTParser",
    "COTRecord",
    "EventBlocker",
    "BlockWindow",
    "ArgusCore",
    "MarketAnomaly",
    "HeraldCore",
    "NewsEvent",
    "NexusCore",
    "NexusOutput",
    "OracleCore",
    "Forecast",
]
