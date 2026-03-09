"""
APHELION TUI — Live Price Ticker Widget

Bloomberg-style price ticker with bid/ask, spread, change indicator.
"""

from __future__ import annotations

from rich.text import Text

from aphelion.tui.state import PriceView


def render_price_ticker(price: PriceView) -> Text:
    """
    Compact price ticker for the header bar.

    Example: XAUUSD  2358.45 ▲+12.30 (+0.52%)  Spd: 0.15
    """
    txt = Text()
    txt.append(f" {price.symbol} ", style="bold bright_white on rgb(20,20,60)")
    txt.append("  ")

    # Last price
    if price.last > 0:
        txt.append(f"{price.last:,.2f}", style="bold bright_white")
    else:
        txt.append("-.--", style="dim")

    txt.append(" ")

    # Change with arrow
    if price.change > 0:
        txt.append(f"▲+{price.change:,.2f}", style="bold bright_green")
        txt.append(f" (+{price.change_pct:.2f}%)", style="bright_green")
    elif price.change < 0:
        txt.append(f"▼{price.change:,.2f}", style="bold bright_red")
        txt.append(f" ({price.change_pct:.2f}%)", style="bright_red")
    else:
        txt.append("◆ 0.00", style="bright_yellow")
        txt.append(" (0.00%)", style="bright_yellow")

    # Spread
    txt.append("  ")
    txt.append("Spd:", style="dim")
    spread_color = "bright_green" if price.spread < 0.3 else (
        "bright_yellow" if price.spread < 0.5 else "bright_red"
    )
    txt.append(f"{price.spread:.2f}", style=spread_color)

    # Bid/Ask
    if price.bid > 0:
        txt.append("  ")
        txt.append("B:", style="dim cyan")
        txt.append(f"{price.bid:,.2f}", style="cyan")
        txt.append(" A:", style="dim cyan")
        txt.append(f"{price.ask:,.2f}", style="cyan")

    # High/Low
    if price.high > 0:
        txt.append("  ")
        txt.append("H:", style="dim green")
        txt.append(f"{price.high:,.2f}", style="green")
        txt.append(" L:", style="dim red")
        txt.append(f"{price.low:,.2f}", style="red")

    return txt
