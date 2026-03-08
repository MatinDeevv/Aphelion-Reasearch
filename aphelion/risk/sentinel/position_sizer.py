"""Position sizing logic with immutable SENTINEL risk caps."""

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
    ) -> float:
        base = self.kelly_fraction(win_rate, avg_win, avg_loss)
        clamped_confidence = max(0.0, min(1.0, confidence))
        size = base * clamped_confidence
        return min(size, SENTINEL.max_position_pct)

    def pct_to_lots(
        self,
        size_pct: float,
        account_equity: float,
        entry_price: float,
        pip_value_per_lot: float = 10.0,
    ) -> float:
        if entry_price <= 0 or pip_value_per_lot <= 0:
            return 0.01

        risk_dollars = account_equity * size_pct
        lots = risk_dollars / (entry_price * pip_value_per_lot)
        rounded = round(lots, 2)
        return max(0.01, rounded)

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
