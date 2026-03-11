"""
APHELION HYDRA Training Script (Phase 7)
Trains the full ensemble on synthetic OR real market data.

Usage:
    python scripts/train_hydra.py                              # Quick validation (synthetic, 500 bars, 2 epochs)
    python scripts/train_hydra.py --full                       # Full synthetic (10000 bars, 20 epochs)
    python scripts/train_hydra.py --data data/bars/xauusd_m1.csv --full   # Real data from MT5 export
    python scripts/train_hydra.py --data data/bars/xauusd_m1.csv --epochs 50 --batch-size 64
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
    build_dataset_from_dataframe,
    build_dataset_from_feature_dicts,
    create_dataloaders,
)
from aphelion.intelligence.hydra.ensemble import EnsembleConfig
from aphelion.intelligence.hydra.tft import TFTConfig
from aphelion.intelligence.hydra.lstm import LSTMConfig
from aphelion.intelligence.hydra.cnn import CNNConfig
from aphelion.intelligence.hydra.moe import MoEConfig
from aphelion.intelligence.hydra.tcn import TCNConfig
from aphelion.intelligence.hydra.transformer import TransformerConfig
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


def load_real_data(csv_path: str) -> pd.DataFrame:
    """
    Load real OHLCV data from a CSV exported by export_mt5_data.py.
    Computes the same features that generate_synthetic_data produces
    so the dataset pipeline can consume it identically.
    """
    logger.info(f"Loading real market data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Normalise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Drop rows with NaN prices
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    n_bars = len(df)
    logger.info(f"Loaded {n_bars:,} bars — price range {df['close'].min():.2f} to {df['close'].max():.2f}")

    # ── Derive session & day_of_week if a timestamp column exists ──
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        hours = ts.dt.hour
        # Map UTC hour → session
        def _hour_to_session(h):
            if h < 8:
                return "ASIAN"
            elif h < 12:
                return "LONDON"
            elif h < 16:
                return "OVERLAP_LDN_NY"
            elif h < 21:
                return "NEW_YORK"
            else:
                return "DEAD_ZONE"

        df["session"] = hours.map(_hour_to_session)
        day_map = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI", 5: "SAT", 6: "SUN"}
        df["day_of_week"] = ts.dt.dayofweek.map(day_map)
    else:
        sessions = ["ASIAN", "LONDON", "OVERLAP_LDN_NY", "NEW_YORK", "DEAD_ZONE"]
        days = ["MON", "TUE", "WED", "THU", "FRI"]
        df["session"] = [sessions[i % len(sessions)] for i in range(n_bars)]
        df["day_of_week"] = [days[(i // (24 * 60)) % len(days)] for i in range(n_bars)]

    # ── Compute technical features from OHLCV ──
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)

    # ATR (14-period)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    atr_safe = np.where(atr > 1e-10, atr, 1.0)  # Avoid division by zero

    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss_arr = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=14, min_periods=1).mean().values
    avg_loss = pd.Series(loss_arr).ewm(span=14, min_periods=1).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss > 1e-10, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)

    # Bollinger Band width
    sma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    std20 = pd.Series(close).rolling(20, min_periods=1).std(ddof=0).values
    std20_safe = np.where(std20 > 1e-10, std20, 1.0)  # Avoid division by zero
    sma20_safe = np.where(np.abs(sma20) > 1e-10, sma20, 1.0)
    bb_width = (2 * std20_safe) / sma20_safe * 100

    # EMAs
    ema20 = pd.Series(close).ewm(span=20, min_periods=1).mean().values
    ema50 = pd.Series(close).ewm(span=50, min_periods=1).mean().values

    # VWAP (session-level approximation using cumulative)
    cum_vol = np.cumsum(volume)
    cum_pv = np.cumsum(close * volume)
    vwap = np.where(cum_vol > 0, cum_pv / cum_vol, close)

    # Cumulative delta (simple approximation from bar direction)
    bar_delta = np.where(close > np.roll(close, 1), volume, -volume)
    bar_delta[0] = 0
    cum_delta = np.cumsum(bar_delta)

    # Assign computed features — use safe denominators everywhere
    df["atr"] = atr
    df["rsi"] = rsi
    df["bb_width"] = bb_width
    df["bb_percentile"] = np.clip((close - (sma20 - std20_safe)) / (2 * std20_safe), -5, 5)
    df["ema_20"] = ema20
    df["ema_50"] = ema50
    df["ema_cross"] = (ema20 - ema50) / atr_safe
    df["vwap"] = vwap
    df["price_vs_vwap"] = (close - vwap) / atr_safe
    df["vwap_upper_1"] = vwap + std20_safe
    df["vwap_lower_1"] = vwap - std20_safe
    df["vwap_upper_2"] = vwap + 2 * std20_safe
    df["vwap_lower_2"] = vwap - 2 * std20_safe
    df["volume_delta"] = bar_delta
    df["cumulative_delta"] = cum_delta
    df["mtf_alignment_score"] = 0.0
    df["mtf_weighted_alignment"] = 0.0
    df["max_spread_zscore"] = 0.0

    # Fill remaining CONTINUOUS_FEATURES with 0 if not present
    for feat in CONTINUOUS_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    # ── CRITICAL: Clean NaN/Inf — these would destroy training ──
    for feat in CONTINUOUS_FEATURES:
        col = df[feat].values
        nan_count = np.isnan(col).sum() + np.isinf(col).sum()
        if nan_count > 0:
            logger.warning(f"  Feature '{feat}' has {nan_count} NaN/Inf values — filling with 0")
            df[feat] = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)

    total_nans = df[CONTINUOUS_FEATURES].isna().sum().sum()
    logger.info(f"Feature sanity check: {total_nans} NaN remaining (should be 0)")

    return df


def build_small_ensemble_config() -> EnsembleConfig:
    """Small model config for fast CPU-only validation."""
    return EnsembleConfig(
        tft_config=TFTConfig(hidden_dim=32, lstm_layers=1, attention_heads=2),
        lstm_config=LSTMConfig(hidden_size=32, num_layers=1, n_attention_heads=2),
        cnn_config=CNNConfig(hidden_size=32, channels=(16, 32, 64, 128)),
        moe_config=MoEConfig(hidden_size=32, expert_hidden_size=48, num_experts=4, top_k=2),
        tcn_config=TCNConfig(hidden_size=32, num_channels=[16, 32, 32]),
        transformer_config=TransformerConfig(hidden_size=32, n_heads=2, n_layers=2, dim_feedforward=64),
        gate_hidden_size=32,
        gate_n_heads=2,
        gate_n_interaction_layers=1,
        model_dropout=0.0,
        dropout=0.1,
    )


def build_full_ensemble_config() -> EnsembleConfig:
    """SUPER INSANE full-size model config for GPU training."""
    return EnsembleConfig(
        tft_config=TFTConfig(hidden_dim=512, lstm_layers=4, attention_heads=8),
        lstm_config=LSTMConfig(hidden_size=384, num_layers=4, n_attention_heads=8),
        cnn_config=CNNConfig(hidden_size=384, channels=(64, 128, 256, 512)),
        moe_config=MoEConfig(hidden_size=384, expert_hidden_size=512, num_experts=8, top_k=2),
        tcn_config=TCNConfig(hidden_size=384, num_channels=[64, 128, 128, 256, 256, 512]),
        transformer_config=TransformerConfig(hidden_size=384, n_heads=8, n_layers=8, dim_feedforward=1536),
        gate_hidden_size=512,
        gate_n_heads=8,
        gate_n_interaction_layers=2,
        model_dropout=0.1,
        dropout=0.15,
    )


def run_training(
    n_bars: int = 500,
    max_epochs: int = 2,
    batch_size: int = 32,
    full_model: bool = False,
    checkpoint_dir: str = "models/hydra",
    data_csv: str = "",
    num_workers: int = -1,
) -> dict:
    """
    Run the full training pipeline.

    Args:
        data_csv: Path to a CSV with real OHLCV data. If empty, uses synthetic data.
        num_workers: DataLoader worker count.  Pass ``-1`` (default) to auto-select:
            4 workers on CUDA, 0 on CPU (safe for Windows / notebook exec() mode).

    Returns:
        Training result metrics dict.
    """
    t0 = time.time()

    # 1. Load data — real CSV or synthetic
    if data_csv:
        df = load_real_data(data_csv)
        n_bars = len(df)
        logger.info(f"Using REAL data: {n_bars:,} bars from {data_csv}")
    else:
        df = generate_synthetic_data(n_bars)
        logger.info(f"Using SYNTHETIC data: {n_bars:,} bars")

    close_prices = df["close"].values

    # 2. Build datasets using the DataFrame path (no to_dict roundtrip)
    logger.info("Building HYDRA datasets...")

    # 3. Configure model + trainer
    ens_config = build_full_ensemble_config() if full_model else build_small_ensemble_config()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve num_workers: auto-select based on device when -1
    if num_workers < 0:
        num_workers = 4 if device == "cuda" else 0

    ds_config = DatasetConfig(
        val_split=0.15,
        test_split=0.15,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else 2,
        lookback_bars=32 if not full_model else 64,
    )
    train_ds, val_ds, test_ds, means, stds = build_dataset_from_dataframe(
        df, close_prices, config=ds_config,
    )
    logger.info(f"Dataset: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.error("Dataset too small, aborting.")
        return {"error": "Dataset too small"}

    train_dl, val_dl, test_dl = create_dataloaders(train_ds, val_ds, test_ds, config=ds_config)

    trainer_config = TrainerConfig(
        max_epochs=max_epochs,
        learning_rate=3e-4 if full_model else 1e-3,
        use_amp=(device == "cuda"),
        ensemble_config=ens_config,
        checkpoint_dir=checkpoint_dir,
        save_every_n_epochs=max(1, max_epochs // 5),
        patience=max(5, max_epochs // 3),
        warmup_epochs=10 if full_model else 0,
        gradient_accumulation_steps=4 if full_model else 1,
        mixup_alpha=0.2 if full_model else 0.0,
        swa_start_epoch=max(max_epochs - 50, max_epochs * 2) if full_model else 9999,
        label_smoothing=0.1 if full_model else 0.0,
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
    parser.add_argument("--data", type=str, default="", help="Path to real OHLCV CSV (from export_mt5_data.py)")
    parser.add_argument("--bars", type=int, default=None, help="Override bar count (synthetic only)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--checkpoint-dir", type=str, default="models/hydra", help="Checkpoint dir")
    parser.add_argument(
        "--num-workers", type=int, default=-1,
        help="DataLoader worker processes (-1 = auto: 4 on CUDA, 0 on CPU)",
    )
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
        data_csv=args.data,
        num_workers=args.num_workers,
    )

    if "error" not in results:
        logger.success(f"HYDRA training complete. {results['total_epochs']} epochs trained.")


if __name__ == "__main__":
    main()
