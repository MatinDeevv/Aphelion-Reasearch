#!/usr/bin/env python3
"""
APHELION Paper Trading Launcher  —  run_paper.py

Single-command entry point for paper trading with full system wiring:
    MT5TickFeed → FeatureEngine → HYDRA → SENTINEL → TUI

Usage:
    python run_paper.py                          # Live MT5 tick feed (default)
    python run_paper.py --mode replay --bars 500 # Replay historical bars
    python run_paper.py --capital 25000          # Custom starting capital
    python run_paper.py --no-tui                 # Headless (logging only)

Requirements:
    pip install -e .
    For MT5 mode: MetaTrader5 Python package + terminal running on Windows
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional — fall back to real env vars

# Ensure project root is importable
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from aphelion.paper.feed import FeedConfig, FeedMode
from aphelion.paper.runner import PaperRunner, PaperRunnerConfig
from aphelion.paper.session import PaperSessionConfig
from aphelion.risk.sentinel.execution.mt5 import MT5Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_paper",
        description="APHELION Paper Trading — end-to-end launcher",
    )
    parser.add_argument(
        "--mode", type=str, default="live",
        choices=["mt5_tick", "live", "replay"],
        help="Data feed mode (default: live)",
    )
    parser.add_argument("--capital", type=float, default=10_000.0, help="Starting capital (USD)")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Trading symbol")
    parser.add_argument("--hydra-checkpoint", type=str, default="", help="Path to HYDRA ensemble .pt checkpoint")
    parser.add_argument("--mt5-login", type=int, default=int(os.getenv("MT5_LOGIN", "0")), help="MT5 account login (env: MT5_LOGIN)")
    parser.add_argument("--mt5-password", type=str, default=os.getenv("MT5_PASSWORD", ""), help="MT5 account password (env: MT5_PASSWORD)")
    parser.add_argument("--mt5-server", type=str, default=os.getenv("MT5_SERVER", ""), help="MT5 broker server (env: MT5_SERVER)")
    parser.add_argument("--mt5-terminal", type=str, default=os.getenv("MT5_TERMINAL", ""), help="Path to terminal64.exe (env: MT5_TERMINAL)")
    parser.add_argument("--mt5-timeout-ms", type=int, default=int(os.getenv("MT5_TIMEOUT_MS", "10000")), help="MT5 initialize timeout in ms (env: MT5_TIMEOUT_MS)")
    parser.add_argument("--mt5-retries", type=int, default=int(os.getenv("MT5_RETRIES", "3")), help="MT5 connection retry attempts (env: MT5_RETRIES)")
    parser.add_argument("--mt5-retry-delay", type=float, default=float(os.getenv("MT5_RETRY_DELAY", "5.0")), help="Seconds between MT5 retries (env: MT5_RETRY_DELAY)")
    parser.add_argument("--poll-ms", type=int, default=100, help="Tick poll interval (ms)")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup bars to pre-load")
    parser.add_argument("--no-tui", action="store_true", help="Disable TUI (console logging only)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/paper_run.log", mode="a", encoding="utf-8"),
        ],
    )
    # Quiet noisy loggers
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("MetaTrader5").setLevel(logging.WARNING)


def build_config(args: argparse.Namespace) -> PaperRunnerConfig:
    """Build the full runner config from CLI arguments."""
    mode_map = {
        "mt5_tick": FeedMode.MT5_TICK,
        "live": FeedMode.LIVE,
        "replay": FeedMode.REPLAY,
    }

    return PaperRunnerConfig(
        feed_mode=mode_map[args.mode],
        mt5_config=MT5Config(
            terminal_path=args.mt5_terminal,
            login=args.mt5_login,
            password=args.mt5_password,
            server=args.mt5_server,
            symbol=args.symbol,
            timeout_ms=args.mt5_timeout_ms,
            retry_attempts=args.mt5_retries,
            retry_delay_seconds=args.mt5_retry_delay,
        ),
        feed_config=FeedConfig(
            symbol=args.symbol,
            poll_interval_ms=args.poll_ms,
            warmup_bars=args.warmup,
        ),
        session_config=PaperSessionConfig(
            initial_capital=args.capital,
            symbol=args.symbol,
            hydra_checkpoint=args.hydra_checkpoint,
            warmup_bars=max(64, args.warmup),
        ),
        enable_tui=not args.no_tui,
    )


async def main() -> None:
    args = parse_args()

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    setup_logging(args.verbose)

    logger = logging.getLogger("aphelion.run_paper")
    logger.info("=" * 60)
    logger.info("APHELION Paper Trading System")
    logger.info("=" * 60)

    # MT5 availability check
    if args.mode in ("mt5_tick", "live"):
        try:
            import MetaTrader5  # noqa: F401
            logger.info("MetaTrader5 package: AVAILABLE")
        except ImportError:
            logger.error(
                "MetaTrader5 package not installed. "
                "Install with: pip install MetaTrader5"
            )
            sys.exit(1)

        if sys.platform != "win32":
            logger.error("MT5 is only available on Windows.")
            sys.exit(1)

    config = build_config(args)
    logger.info("Feed mode: %s", config.feed_mode.name)
    logger.info("Capital:   $%.2f", config.session_config.initial_capital)
    logger.info("Symbol:    %s", config.session_config.symbol)

    if config.session_config.hydra_checkpoint:
        logger.info("HYDRA:     %s", config.session_config.hydra_checkpoint)
    else:
        logger.info("HYDRA:     (no checkpoint — signals disabled)")

    runner = PaperRunner(config)

    try:
        result = await runner.run()
        print("\n" + result.summary())
    except asyncio.CancelledError:
        logger.info("Run cancelled — shutting down gracefully")
        await runner.stop()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        await runner.stop()
    except ConnectionError as exc:
        logger.error("Connection failed: %s", exc)
        sys.exit(1)
    except Exception:
        logger.exception("Fatal error during paper trading")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # asyncio.run may re-raise KeyboardInterrupt after cancelling pending tasks
        print("\nInterrupted by user.")
