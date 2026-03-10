"""
ATLAS — COT (Commitment of Traders) Parser
Phase 19 — Engineering Spec v3.0

Parses CFTC COT data for gold futures to derive positioning bias.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class COTRecord:
    """One weekly COT record for gold futures."""
    report_date: datetime
    managed_money_long: int = 0
    managed_money_short: int = 0
    commercial_long: int = 0
    commercial_short: int = 0

    @property
    def net_speculative(self) -> int:
        return self.managed_money_long - self.managed_money_short

    @property
    def net_commercial(self) -> int:
        return self.commercial_long - self.commercial_short

    @property
    def speculative_bias(self) -> str:
        net = self.net_speculative
        if net > 50_000:
            return "EXTREME_LONG"
        if net > 20_000:
            return "BULLISH"
        if net < -50_000:
            return "EXTREME_SHORT"
        if net < -20_000:
            return "BEARISH"
        return "NEUTRAL"


class COTParser:
    """
    Parses COT JSON files and tracks weekly positioning changes.
    """

    def __init__(self, data_dir: str = "data/external"):
        self._data_dir = Path(data_dir)
        self._records: List[COTRecord] = []

    def load_from_json(self, filepath: Optional[str] = None) -> int:
        """Load COT data from a JSON file. Returns record count."""
        path = Path(filepath) if filepath else self._data_dir / "cot_gold.json"
        if not path.exists():
            logger.warning("COT file not found: %s", path)
            return 0

        with open(path, "r") as f:
            data = json.load(f)

        for entry in data:
            rec = COTRecord(
                report_date=datetime.fromisoformat(entry["date"]),
                managed_money_long=entry.get("mm_long", 0),
                managed_money_short=entry.get("mm_short", 0),
                commercial_long=entry.get("comm_long", 0),
                commercial_short=entry.get("comm_short", 0),
            )
            self._records.append(rec)

        self._records.sort(key=lambda r: r.report_date)
        return len(self._records)

    def add_record(self, record: COTRecord) -> None:
        self._records.append(record)

    @property
    def latest(self) -> Optional[COTRecord]:
        return self._records[-1] if self._records else None

    @property
    def positioning_change(self) -> int:
        """Week-over-week change in net speculative positions."""
        if len(self._records) < 2:
            return 0
        return self._records[-1].net_speculative - self._records[-2].net_speculative

    @property
    def records(self) -> List[COTRecord]:
        return list(self._records)
