"""Position sizing logic with immutable SENTINEL risk caps.

Improvements:
- Volatility-adjusted sizing support
- Raises ValueError on zero entry price (was returning 0.01 silently)
- ATR-based lot calculation alternative
"""

from __future__ import annotations

from aphelion.core.config import KELLY_FRACTION, KELLY_MAX_F, SENTINEL


class PositionSizer:
    """Quarter-Kelly position sizing with hardcoded SENTINEL caps."""

    def __init__(self):
        pass

    def kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        if avg_win <= 0 or avg_loss <= 0:
            return 0.0

        full_kelly = (win_rate / avg_loss) - ((1.0 - win_rate) / avg_win)
        full_kelly = max(0.0, min(1.0, full_kelly))
        sized = full_kelly * KELLY_FRACTION
        return min(sized, KELLY_MAX_F)

    def compute_size_pct(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        confidence: float = 1.0,
        volatility_scalar: float = 1.0,
    ) -> float:
        """Compute position size as fraction of account.

        Args:
            volatility_scalar: 0.5=high vol (halve size), 1.0=normal, 1.5=low vol (increase)
        """
        base = self.kelly_fraction(win_rate, avg_win, avg_loss)
        clamped_confidence = max(0.0, min(1.0, confidence))
        clamped_vol = max(0.25, min(2.0, volatility_scalar))
        size = base * clamped_confidence * clamped_vol
        return min(size, SENTINEL.max_position_pct)

    def pct_to_lots(
        self,
        size_pct: float,
        account_equity: float,
        entry_price: float,
        pip_value_per_lot: float = 10.0,
    ) -> float:
        if entry_price <= 0:
            raise ValueError(f"entry_price must be > 0, got {entry_price}")
        if pip_value_per_lot <= 0:
            raise ValueError(f"pip_value_per_lot must be > 0, got {pip_value_per_lot}")

        risk_dollars = account_equity * size_pct
        lots = risk_dollars / (entry_price * pip_value_per_lot)
        rounded = round(lots, 2)
        return max(0.01, rounded)

    def atr_based_lots(
        self,
        account_equity: float,
        risk_pct: float,
        atr: float,
        lot_size_oz: float = 100.0,
    ) -> float:
        """Compute lot size based on ATR for volatility-adjusted sizing.

        risk_dollars = account_equity * risk_pct
        lots = risk_dollars / (atr * lot_size_oz)
        """
        if atr <= 0 or lot_size_oz <= 0:
            return 0.01
        risk_dollars = account_equity * min(risk_pct, SENTINEL.max_position_pct)
        lots = risk_dollars / (atr * lot_size_oz)
        return max(0.01, round(lots, 2))

    def validate_size(self, size_pct: float, current_exposure_pct: float) -> tuple[bool, str]:
        if size_pct > SENTINEL.max_position_pct:
            return (
                False,
                f"SIZE_EXCEEDED: {size_pct:.3f} > {SENTINEL.max_position_pct}",
            )

        max_total = SENTINEL.max_position_pct * SENTINEL.max_simultaneous_positions
        if current_exposure_pct + size_pct > max_total:
            total = current_exposure_pct + size_pct
            return (
                False,
                f"EXPOSURE_EXCEEDED: {total:.3f} > {max_total:.3f}",
            )

        return True, "OK"
