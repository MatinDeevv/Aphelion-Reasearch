"""
APHELION TUI — Live Simulation Engine — REMOVED.

The simulated trading mode has been removed from Aphelion.
All trading sessions must use real market data from MT5 or replay feeds.
"""

raise ImportError(
    "LiveSimulator has been removed. "
    "Aphelion now requires real market data exclusively. "
    "Use paper trading with a live MT5 connection or replay historical data."
)
