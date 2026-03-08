"""
APHELION HYDRA Training Script (Phase 7)
Generates synthetic OHLCV data with regime patterns, trains the full ensemble,
and validates the pipeline end-to-end.

Usage:
    python scripts/train_hydra.py             # Quick validation (500 bars, 2 epochs)
    python scripts/train_hydra.py --full      # Full synthetic (10000 bars, 20 epochs)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from aphelion.intelligence.hydra.dataset import (
    CONTINUOUS_FEATURES,
    DatasetConfig,
    build_dataset_from_feature_dicts,
    create_dataloaders,
)
from aphelion.intelligence.hydra.ensemble import EnsembleConfig
from aphelion.intelligence.hydra.tft import TFTConfig
from aphelion.intelligence.hydra.lstm import LSTMConfig
from aphelion.intelligence.hydra.cnn import CNNConfig
from aphelion.intelligence.hydra.moe import MoEConfig
from aphelion.intelligence.hydra.trainer import HydraTrainer, TrainerConfig


def generate_synthetic_data(n_bars: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with defined regime patterns:
    - Trend regime (bars 1000-2000): positive drift
    - Range regime (bars 3000-4000): mean-reverting
    - Volatile regime (bars 4000-5000): high variance
    - News spike (bars 7500-7600): sudden move + reversion
    """
    logger.info(f"Generating {n_bars} synthetic bars for training...")

    np.random.seed(42)

    # 1. Base random walk
    returns = np.random.normal(0, 0.001, n_bars)

    # Trend regime
    if n_bars > 2000:
        returns[1000:2000] += 0.0005

    # Range regime (mean revert toward 0)
    if n_bars > 4000:
        returns[3000:4000] *= 0.3

    # Volatile regime
    if n_bars > 5000:
        returns[4000:5000] *= 3.0

    # News spike
    if n_bars > 7600:
        returns[7500] = 0.02
        returns[7501:7600] -= 0.0003  # Gradual reversion

    # Generate prices
    close = 2000.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.0005, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.0005, n_bars)))
    open_p = np.roll(close, 1)
    open_p[0] = 2000.0

    volume = np.random.randint(10, 1000, n_bars).astype(float)
    # Higher volume during volatile regime
    if n_bars > 5000:
        volume[4000:5000] *= 3

    df = pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    # Add categorical features
    sessions = ["ASIAN", "LONDON", "OVERLAP_LDN_NY", "NEW_YORK", "DEAD_ZONE"]
    days = ["MON", "TUE", "WED", "THU", "FRI"]
    df["session"] = [sessions[i % len(sessions)] for i in range(n_bars)]
    df["day_of_week"] = [days[(i // (24 * 60)) % len(days)] for i in range(n_bars)]

    # Generate remaining continuous features as correlated noise
    atr = np.abs(high - low)
    rsi = 50 + 10 * np.tanh(np.cumsum(returns) * 100)
    bb_width = atr / close * 100

    for feat in CONTINUOUS_FEATURES:
        if feat in df.columns:
            continue
        elif feat == "atr":
            df[feat] = atr
        elif feat == "rsi":
            df[feat] = rsi
        elif feat == "bb_width":
            df[feat] = bb_width
        elif feat == "vpin":
            df[feat] = np.abs(np.random.normal(0, 0.3, n_bars))
        elif "distance" in feat:
            df[feat] = np.random.exponential(0.01, n_bars)
        elif "delta" in feat:
            df[feat] = np.cumsum(np.random.normal(0, 100, n_bars))
        else:
            df[feat] = np.random.normal(0, 1, n_bars)

    return df


def build_small_ensemble_config() -> EnsembleConfig:
    """Small model config for fast CPU-only validation."""
    return EnsembleConfig(
        tft_config=TFTConfig(hidden_dim=32, lstm_layers=1, attention_heads=2),
        lstm_config=LSTMConfig(hidden_size=32, num_layers=1),
        cnn_config=CNNConfig(hidden_size=32),
        moe_config=MoEConfig(hidden_size=32),
        gate_hidden_size=32,
        dropout=0.1,
    )


def build_full_ensemble_config() -> EnsembleConfig:
    """Full-size model config for GPU training."""
    return EnsembleConfig(
        tft_config=TFTConfig(hidden_dim=256, lstm_layers=2, attention_heads=4),
        lstm_config=LSTMConfig(hidden_size=128, num_layers=2),
        cnn_config=CNNConfig(hidden_size=128),
        moe_config=MoEConfig(hidden_size=128),
        gate_hidden_size=256,
        dropout=0.2,
    )


def run_training(
    n_bars: int = 500,
    max_epochs: int = 2,
    batch_size: int = 32,
    full_model: bool = False,
    checkpoint_dir: str = "models/hydra",
) -> dict:
    """
    Run the full training pipeline.

    Returns:
        Training result metrics dict.
    """
    t0 = time.time()

    # 1. Generate data
    df = generate_synthetic_data(n_bars)
    feature_dicts = df.to_dict(orient="records")
    close_prices = df["close"].values

    # 2. Build datasets
    logger.info("Building HYDRA datasets...")
    ds_config = DatasetConfig(
        val_split=0.15,
        test_split=0.15,
        batch_size=batch_size,
        num_workers=0,
        lookback_bars=32 if not full_model else 64,
    )
    train_ds, val_ds, test_ds, means, stds = build_dataset_from_feature_dicts(
        feature_dicts, close_prices, config=ds_config,
    )
    logger.info(f"Dataset: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.error("Dataset too small, aborting.")
        return {"error": "Dataset too small"}

    train_dl, val_dl, test_dl = create_dataloaders(train_ds, val_ds, test_ds, config=ds_config)

    # 3. Configure model + trainer
    ens_config = build_full_ensemble_config() if full_model else build_small_ensemble_config()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer_config = TrainerConfig(
        max_epochs=max_epochs,
        learning_rate=5e-4 if full_model else 1e-3,
        use_amp=(device == "cuda"),
        ensemble_config=ens_config,
        checkpoint_dir=checkpoint_dir,
        save_every_n_epochs=max(1, max_epochs // 5),
        patience=max(5, max_epochs // 3),
    )
    trainer = HydraTrainer(trainer_config, device=device)

    # 4. Train
    logger.info(f"Training on {device} for {max_epochs} epochs...")
    results = trainer.train(train_dl, val_dl)

    elapsed = time.time() - t0
    logger.info(
        f"Training complete in {elapsed:.1f}s — "
        f"Final val loss: {results['final_val_loss']:.4f}, "
        f"Best Sharpe proxy: {results['best_val_sharpe']:.2f}, "
        f"Params: {results['model_params']:,}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="APHELION HYDRA Training")
    parser.add_argument("--full", action="store_true", help="Full training (10K bars, 20 epochs)")
    parser.add_argument("--bars", type=int, default=None, help="Override bar count")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--checkpoint-dir", type=str, default="models/hydra", help="Checkpoint dir")
    args = parser.parse_args()

    if args.full:
        n_bars = args.bars or 10000
        max_epochs = args.epochs or 20
        full_model = True
    else:
        n_bars = args.bars or 500
        max_epochs = args.epochs or 2
        full_model = False

    results = run_training(
        n_bars=n_bars,
        max_epochs=max_epochs,
        batch_size=args.batch_size,
        full_model=full_model,
        checkpoint_dir=args.checkpoint_dir,
    )

    if "error" not in results:
        logger.success(f"HYDRA training complete. {results['total_epochs']} epochs trained.")


if __name__ == "__main__":
    main()
