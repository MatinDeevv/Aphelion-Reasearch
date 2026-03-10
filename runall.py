#!/usr/bin/env python3
"""
Run APHELION end-to-end demo:
1) Launch background HYDRA training process
2) Run simulated paper trading session
3) Show live TUI with real-time updates

Usage:
    python runall.py
    python runall.py --train-full
    python runall.py --no-training
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

from aphelion.paper.feed import FeedMode, SimulatedFeedConfig
from aphelion.paper.runner import PaperRunner, PaperRunnerConfig
from aphelion.paper.session import PaperSessionConfig
from aphelion.tui.app import AphelionTUI, TUIConfig

# Optional governance imports
try:
    from aphelion.ares.coordinator import AresCoordinator, AresConfig
    _HAS_ARES = True
except ImportError:
    _HAS_ARES = False

try:
    from aphelion.governance.council.sola import SOLA
    _HAS_SOLA = True
except ImportError:
    _HAS_SOLA = False


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APHELION all-in-one live demo")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Starting paper capital")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Trading symbol")
    parser.add_argument("--sim-bars", type=int, default=0, help="Sim bars (0 = infinite)")
    parser.add_argument("--warmup", type=int, default=64, help="Warmup bars")
    parser.add_argument("--no-training", action="store_true", help="Do not start background training")
    parser.add_argument("--train-full", action="store_true", help="Use full training config")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return parser.parse_args()


def _find_hydra_checkpoint() -> str:
    """Find best available checkpoint so simulation can place model-driven trades."""
    preferred = [
        "models/hydra/hydra_ensemble_best_sharpe.pt",
        "models/hydra/hydra_ensemble_best_loss.pt",
        "models/test_hydra/hydra_ensemble_best_sharpe.pt",
        "models/test_hydra/hydra_ensemble_best_loss.pt",
    ]
    for path in preferred:
        if Path(path).exists():
            return path

    # Fallback: search recursively
    for path in Path("models").rglob("hydra_ensemble_best_sharpe.pt"):
        return str(path)
    for path in Path("models").rglob("hydra_ensemble_best_loss.pt"):
        return str(path)

    return ""


def _start_training_subprocess(train_full: bool) -> subprocess.Popen:
    """Start training in background and write logs to logs/train_hydra_live.log."""
    Path("logs").mkdir(exist_ok=True)
    log_path = Path("logs/train_hydra_live.log")
    log_file = log_path.open("a", encoding="utf-8")

    cmd = [sys.executable, "scripts/train_hydra.py"]
    if train_full:
        cmd.append("--full")

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).resolve().parent),
    )
    return proc


async def _wait_for_runner_state(runner: PaperRunner, timeout_s: float = 5.0):
    """Wait until PaperRunner has initialized TUI state."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if runner.tui_state is not None:
            return runner.tui_state
        await asyncio.sleep(0.05)
    raise TimeoutError("Runner did not initialize TUI state in time")


async def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("aphelion.runall")

    logger.info("Starting APHELION all-in-one demo...")

    checkpoint = _find_hydra_checkpoint()
    if checkpoint:
        logger.info("Using HYDRA checkpoint: %s", checkpoint)
    else:
        logger.warning("No HYDRA checkpoint found. Session will run but may not place model-driven trades.")

    training_proc: Optional[subprocess.Popen] = None
    if not args.no_training:
        training_proc = _start_training_subprocess(args.train_full)
        logger.info("Background training started (PID %s). Logs: logs/train_hydra_live.log", training_proc.pid)

    # ── Build ARES governance pipeline ──────────────────────────────────
    ares = None
    if _HAS_ARES:
        sola = None
        if _HAS_SOLA:
            try:
                sola = SOLA()
                logger.info("SOLA sovereign intelligence layer initialized")
            except Exception as exc:
                logger.warning("SOLA init failed (non-fatal): %s", exc)

        ares = AresCoordinator(config=AresConfig(), sola=sola)
        logger.info(
            "ARES coordinator initialized — SOLA=%s",
            "active" if sola is not None else "disabled",
        )
    else:
        logger.warning("ARES not available — session will run without governance layer")

    config = PaperRunnerConfig(
        feed_mode=FeedMode.SIMULATED,
        session_config=PaperSessionConfig(
            initial_capital=args.capital,
            symbol=args.symbol,
            hydra_checkpoint=checkpoint,
            warmup_bars=max(32, args.warmup),
        ),
        sim_config=SimulatedFeedConfig(
            max_bars=args.sim_bars,
            symbol=args.symbol,
        ),
        enable_tui=True,
        ares=ares,
    )

    runner = PaperRunner(config)
    runner_task = asyncio.create_task(runner.run())

    try:
        state = await _wait_for_runner_state(runner)
        tui = AphelionTUI(
            state=state,
            config=TUIConfig(refresh_rate=0.25, initial_view="overview"),
        )

        logger.info("TUI started. Press 'q' in TUI or Ctrl+C to exit.")
        await tui.run()

    except asyncio.CancelledError:
        logger.info("Run cancelled")
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await runner.stop()

        if not runner_task.done():
            runner_task.cancel()
            try:
                await runner_task
            except asyncio.CancelledError:
                pass

        if training_proc is not None and training_proc.poll() is None:
            logger.info("Stopping background training process (PID %s)", training_proc.pid)
            training_proc.terminate()
            try:
                training_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                training_proc.kill()

        logger.info("APHELION runall demo stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
