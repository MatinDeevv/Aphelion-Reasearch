#!/usr/bin/env python3
"""
APHELION — Bloomberg-grade Autonomous Trading System

Launch the full TUI application:

    python aphelion.py

Optional flags (environment variables):
    APHELION_CONFIG  — path to config JSON (default: config/aphelion.json)
    APHELION_VIEW    — initial view to show (default: launcher)
"""

from __future__ import annotations

import os
import sys
import logging

# ---------------------------------------------------------------------------
# Bootstrap logging
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "aphelion.log")),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger("aphelion")


def main() -> None:
    """Entry-point: load config → create controller → build TUI → run."""

    # -- config --
    from aphelion.tui.config import load_config

    config_path = os.environ.get("APHELION_CONFIG", "config/aphelion.json")
    config = load_config(config_path)

    errors = config.validate()
    if errors:
        for e in errors:
            logger.warning("Config warning: %s", e)

    # -- controller (single owner of all background work) --
    from aphelion.tui.controller import AphelionController

    controller = AphelionController(config)

    # -- TUI --
    from aphelion.tui.app import AphelionTUI, TUIConfig

    initial_view = os.environ.get("APHELION_VIEW", "launcher")
    tui_config = TUIConfig(initial_view=initial_view)
    tui = AphelionTUI(controller=controller, config=tui_config)

    # Wire controller ↔ TUI state
    controller.tui_state = tui.state

    logger.info("APHELION TUI starting  (view=%s)", initial_view)
    try:
        tui.run_sync()
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop_session()
        logger.info("APHELION TUI shutdown complete")


if __name__ == "__main__":
    main()
