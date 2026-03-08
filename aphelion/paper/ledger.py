"""
APHELION Paper Trading Ledger
Append-only JSON-Lines audit journal that records every event during
a paper trading session: fills, rejections, SL/TP exits, sentinel
status snapshots, and errors.

Each line is a self-contained JSON object with a UTC timestamp.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default ledger directory
_DEFAULT_DIR = Path("logs") / "paper"


class PaperLedger:
    """
    Append-only JSON-Lines file logger for paper trading audit trail.
    Thread-safe via single-writer design (only one session at a time).
    """

    def __init__(
        self,
        session_id: str,
        directory: Optional[Path] = None,
    ):
        self._session_id = session_id
        self._dir = directory or _DEFAULT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"paper_{session_id}.jsonl"
        self._entry_count: int = 0
        self._fh = open(self._path, "a", encoding="utf-8")  # noqa: SIM115

        # Write session header
        self._write("SESSION_START", {
            "session_id": session_id,
            "ledger_path": str(self._path),
        })
        logger.info("Paper ledger opened: %s", self._path)

    # ── Core writers ──────────────────────────────────────────────────────

    def log_fill(self, data: dict) -> None:
        """Record a simulated fill."""
        self._write("FILL", data)

    def log_rejection(self, data: dict) -> None:
        """Record an order rejection."""
        self._write("REJECTION", data)

    def log_exit(self, data: dict) -> None:
        """Record a position exit (SL, TP, force close)."""
        self._write("EXIT", data)

    def log_signal(self, data: dict) -> None:
        """Record a HYDRA signal (actionable or filtered)."""
        self._write("SIGNAL", data)

    def log_sentinel_status(self, data: dict) -> None:
        """Record a periodic sentinel status snapshot."""
        self._write("SENTINEL_STATUS", data)

    def log_error(self, message: str, details: Optional[dict] = None) -> None:
        """Record an error event."""
        self._write("ERROR", {"message": message, **(details or {})})

    def log_event(self, event_type: str, data: dict) -> None:
        """Record a generic event."""
        self._write(event_type, data)

    # ── Internal ──────────────────────────────────────────────────────────

    def _write(self, event_type: str, data: Any) -> None:
        """Write a single JSON-L entry."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session": self._session_id,
            "seq": self._entry_count,
            "type": event_type,
            "data": data,
        }
        try:
            line = json.dumps(entry, default=str)
            self._fh.write(line + "\n")
            self._fh.flush()
            self._entry_count += 1
        except Exception:
            logger.exception("Ledger write failed for event_type=%s", event_type)

    # ── Session end ───────────────────────────────────────────────────────

    def close(self, summary: Optional[dict] = None) -> None:
        """Write session footer and close the file handle."""
        self._write("SESSION_END", summary or {})
        try:
            self._fh.close()
        except Exception:
            pass
        logger.info("Paper ledger closed: %s (%d entries)", self._path, self._entry_count)

    # ── Stats ─────────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._path

    @property
    def entry_count(self) -> int:
        return self._entry_count

    # ── Reading ───────────────────────────────────────────────────────────

    @staticmethod
    def read_ledger(path: Path) -> list[dict]:
        """Read a JSON-L ledger file and return list of entries."""
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    # ── Context manager ───────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
