"""
MACRO Gold Seasonality Patterns
Historical seasonal tendencies for XAU/USD.
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional


@dataclass
class SeasonalBias:
    month_bias: str = "NEUTRAL"       # BULLISH, BEARISH, NEUTRAL
    week_of_month_bias: str = "NEUTRAL"
    day_of_week_bias: str = "NEUTRAL"
    strength: float = 0.0             # [0, 1]
    notes: str = ""


class GoldSeasonality:
    """
    Gold seasonal patterns based on historical data.
    
    Gold tends to:
    - Rally in Q1 (Jan-Mar) — jewelry demand, central bank buying
    - Dip in Q2 (Apr-Jun) — post-Q1 correction
    - Rally Aug-Sep — safe haven demand before autumn
    - Mixed Oct-Dec — depends on year-end flows
    
    Day-of-week:
    - Monday: Often range-bound (waiting for direction)
    - Tuesday-Thursday: Strongest moves
    - Friday: Reduced activity after 1500 UTC
    """

    # Monthly bias (from historical gold performance analysis)
    MONTHLY_BIAS = {
        1: ("BULLISH", 0.65),    # January: strong historically
        2: ("BULLISH", 0.55),
        3: ("NEUTRAL", 0.45),
        4: ("BEARISH", 0.55),    # Post-Q1 correction
        5: ("BEARISH", 0.50),
        6: ("NEUTRAL", 0.40),
        7: ("NEUTRAL", 0.45),
        8: ("BULLISH", 0.60),    # Safe haven demand
        9: ("BULLISH", 0.55),
        10: ("NEUTRAL", 0.45),
        11: ("NEUTRAL", 0.45),
        12: ("NEUTRAL", 0.40),   # Low liquidity
    }

    # Day of week bias (0=Mon, 4=Fri)
    DOW_BIAS = {
        0: ("NEUTRAL", 0.40),    # Monday
        1: ("BULLISH", 0.55),    # Tuesday: breakout day
        2: ("NEUTRAL", 0.50),    # Wednesday: FOMC days
        3: ("BULLISH", 0.50),    # Thursday: continuation
        4: ("BEARISH", 0.45),    # Friday: profit-taking
    }

    def get_bias(self, dt: Optional[datetime] = None) -> SeasonalBias:
        """Get seasonal bias for a given date."""
        if dt is None:
            dt = datetime.now(timezone.utc)

        month = dt.month
        dow = dt.weekday()
        week = (dt.day - 1) // 7 + 1  # Week 1-5 of month

        month_info = self.MONTHLY_BIAS.get(month, ("NEUTRAL", 0.40))
        dow_info = self.DOW_BIAS.get(dow, ("NEUTRAL", 0.40))

        # Week of month bias
        if week == 1:
            wom_bias = "NEUTRAL"  # First week: often range-setting
        elif week in (2, 3):
            wom_bias = "BULLISH"  # Mid-month: strongest moves
        else:
            wom_bias = "NEUTRAL"  # Month-end: rebalancing

        strength = (month_info[1] + dow_info[1]) / 2

        return SeasonalBias(
            month_bias=month_info[0],
            week_of_month_bias=wom_bias,
            day_of_week_bias=dow_info[0],
            strength=strength,
            notes=f"Month {month}: {month_info[0]}, {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow]}: {dow_info[0]}",
        )
