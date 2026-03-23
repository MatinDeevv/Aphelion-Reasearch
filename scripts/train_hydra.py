"""
APHELION HYDRA training entrypoint.

Supports:
  - real OHLCV CSV input
  - forged {symbol}_hydra.parquet when prepared train/val/test npz splits exist beside it
  - direct prepared dataset directory / train.npz path

Examples:
    python scripts/train_hydra.py --data data/raw/XAUUSD/xauusd_m1.csv
    python scripts/train_hydra.py --data data/processed/XAUUSD/xauusd_hydra.parquet --full
    python scripts/train_hydra.py --data data/processed/XAUUSD --full --epochs 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from aphelion.intelligence.hydra.dataset import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    DatasetConfig,
    HydraDataset,
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


def load_real_data(csv_path: str) -> pd.DataFrame:
    """
    Load real OHLCV data from a CSV exported by export_mt5_data.py
    or by the richer bulk downloader in aphelion_data.py.
    """
    logger.info(f"Loading real market data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Normalize column names.
    df.columns = [c.strip().lower() for c in df.columns]

    # Accept both the simple MT5 export format and the richer bulk-export format.
    rename_map = {}
    if "time" in df.columns and "timestamp" not in df.columns:
        rename_map["time"] = "timestamp"
    if "tick_vol" in df.columns and "volume" not in df.columns:
        rename_map["tick_vol"] = "volume"
    if "real_vol" in df.columns and "real_volume" not in df.columns:
        rename_map["real_vol"] = "real_volume"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    n_bars = len(df)
    logger.info(f"Loaded {n_bars:,} bars - price range {df['close'].min():.2f} to {df['close'].max():.2f}")

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        hours = ts.dt.hour

        def hour_to_session(hour: int) -> str:
            if hour < 8:
                return "ASIAN"
            if hour < 12:
                return "LONDON"
            if hour < 16:
                return "OVERLAP_LDN_NY"
            if hour < 21:
                return "NEW_YORK"
            return "DEAD_ZONE"

        df["session"] = hours.map(hour_to_session)
        day_map = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI", 5: "SAT", 6: "SUN"}
        df["day_of_week"] = ts.dt.dayofweek.map(day_map)
    else:
        sessions = ["ASIAN", "LONDON", "OVERLAP_LDN_NY", "NEW_YORK", "DEAD_ZONE"]
        days = ["MON", "TUE", "WED", "THU", "FRI"]
        df["session"] = [sessions[i % len(sessions)] for i in range(n_bars)]
        df["day_of_week"] = [days[(i // (24 * 60)) % len(days)] for i in range(n_bars)]

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)

    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    atr_safe = np.where(atr > 1e-10, atr, 1.0)

    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss_arr = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=14, min_periods=1).mean().values
    avg_loss = pd.Series(loss_arr).ewm(span=14, min_periods=1).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 1e-10, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)

    sma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    std20 = pd.Series(close).rolling(20, min_periods=1).std(ddof=0).values
    std20_safe = np.where(std20 > 1e-10, std20, 1.0)
    sma20_safe = np.where(np.abs(sma20) > 1e-10, sma20, 1.0)
    bb_width = (2 * std20_safe) / sma20_safe * 100

    ema20 = pd.Series(close).ewm(span=20, min_periods=1).mean().values
    ema50 = pd.Series(close).ewm(span=50, min_periods=1).mean().values

    cum_vol = np.cumsum(volume)
    cum_pv = np.cumsum(close * volume)
    vwap = np.where(cum_vol > 0, cum_pv / cum_vol, close)

    bar_delta = np.where(close > np.roll(close, 1), volume, -volume)
    bar_delta[0] = 0
    cum_delta = np.cumsum(bar_delta)

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

    for feat in CONTINUOUS_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    for feat in CONTINUOUS_FEATURES:
        col = df[feat].values
        bad_count = np.isnan(col).sum() + np.isinf(col).sum()
        if bad_count > 0:
            logger.warning(f"Feature '{feat}' has {bad_count} NaN/Inf values - filling with 0")
            df[feat] = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)

    total_nans = int(df[CONTINUOUS_FEATURES].isna().sum().sum())
    logger.info(f"Feature sanity check: {total_nans} NaN remaining (should be 0)")
    return df


def resolve_prepared_split_dir(data_path: str) -> Optional[Path]:
    """Resolve a directory containing train.npz / val.npz / test.npz."""
    path = Path(data_path)
    if path.is_dir():
        base = path
    elif path.suffix.lower() in {".npz", ".parquet"}:
        base = path.parent
    else:
        return None

    required = [base / "train.npz", base / "val.npz", base / "test.npz"]
    if all(p.exists() for p in required):
        return base
    return None


def resolve_first_key(keys: set[str], *candidates: str) -> str:
    for key in candidates:
        if key in keys:
            return key
    raise KeyError(f"None of the candidate keys were present: {candidates}")


def compute_raw_returns_from_close(close: np.ndarray, horizons: tuple[int, int, int] = (5, 15, 60)) -> np.ndarray:
    raw = np.zeros((len(close), len(horizons)), dtype=np.float32)
    close = close.astype(np.float32, copy=False)
    for i, horizon in enumerate(horizons):
        if len(close) > horizon:
            base = np.clip(close[:-horizon], 1e-10, None)
            raw[:-horizon, i] = (close[horizon:] - close[:-horizon]) / base * 100.0
    return raw


def dataset_from_npz(split_path: Path, lookback: int) -> tuple[HydraDataset, int, int]:
    data = np.load(split_path, allow_pickle=True)
    keys = set(data.files)

    x_cont = np.asarray(data["X_cont"], dtype=np.float32)
    x_cat = np.asarray(data["X_cat"], dtype=np.int64)
    close = np.asarray(data["close"], dtype=np.float32) if "close" in keys else np.zeros(len(x_cont), dtype=np.float32)

    y5 = np.asarray(data[resolve_first_key(keys, "y_label_5m", "y_label_5")], dtype=np.int64)
    y15 = np.asarray(data[resolve_first_key(keys, "y_label_15m", "y_label_15")], dtype=np.int64)
    y1h = np.asarray(data[resolve_first_key(keys, "y_label_60m", "y_label_1h", "y_label_60")], dtype=np.int64)

    if "raw_returns" in keys:
        raw_returns = np.asarray(data["raw_returns"], dtype=np.float32)
    else:
        future_5 = next((k for k in ("future_ret_5", "y_future_ret_5", "raw_ret_5") if k in keys), None)
        future_15 = next((k for k in ("future_ret_15", "y_future_ret_15", "raw_ret_15") if k in keys), None)
        future_60 = next((k for k in ("future_ret_60", "future_ret_60m", "y_future_ret_60", "raw_ret_60") if k in keys), None)
        if future_5 and future_15 and future_60:
            raw_returns = np.stack(
                [
                    np.asarray(data[future_5], dtype=np.float32),
                    np.asarray(data[future_15], dtype=np.float32),
                    np.asarray(data[future_60], dtype=np.float32),
                ],
                axis=1,
            )
        else:
            raw_returns = compute_raw_returns_from_close(close)

    if not (len(x_cont) == len(x_cat) == len(y5) == len(y15) == len(y1h) == len(raw_returns)):
        raise ValueError(f"Prepared split arrays in {split_path} do not have matching lengths.")

    max_horizon = 60
    valid_end = len(x_cont) - max_horizon
    if valid_end <= lookback:
        raise ValueError(
            f"Prepared split {split_path} is too small for lookback={lookback} and max_horizon={max_horizon}."
        )

    indices = list(range(lookback, valid_end))
    ds = HydraDataset(x_cont, x_cat, y5, y15, y1h, raw_returns, indices, lookback)
    return ds, int(x_cont.shape[1]), int(x_cat.shape[1])


def load_prepared_datasets(data_path: str, lookback: int) -> tuple[HydraDataset, HydraDataset, HydraDataset, int, int, Path]:
    base_dir = resolve_prepared_split_dir(data_path)
    if base_dir is None:
        raise ValueError(
            "Prepared HYDRA splits not found. Expected train.npz/val.npz/test.npz in the given directory or beside the parquet."
        )

    train_ds, n_cont, n_cat = dataset_from_npz(base_dir / "train.npz", lookback)
    val_ds, val_n_cont, val_n_cat = dataset_from_npz(base_dir / "val.npz", lookback)
    test_ds, test_n_cont, test_n_cat = dataset_from_npz(base_dir / "test.npz", lookback)

    if (n_cont, n_cat) != (val_n_cont, val_n_cat) or (n_cont, n_cat) != (test_n_cont, test_n_cat):
        raise ValueError("Prepared split feature dimensions do not match across train/val/test.")

    return train_ds, val_ds, test_ds, n_cont, n_cat, base_dir


def build_small_ensemble_config(
    n_continuous: Optional[int] = None,
    n_categorical: Optional[int] = None,
    lookback: int = 32,
) -> EnsembleConfig:
    """Small model config for fast validation and smoke tests."""
    n_continuous = n_continuous if n_continuous is not None else len(CONTINUOUS_FEATURES)
    n_categorical = n_categorical if n_categorical is not None else len(CATEGORICAL_FEATURES)
    return EnsembleConfig(
        tft_config=TFTConfig(
            n_continuous=n_continuous,
            n_categorical=n_categorical,
            hidden_dim=32,
            lstm_layers=1,
            attention_heads=2,
            lookback=lookback,
        ),
        lstm_config=LSTMConfig(
            n_continuous=n_continuous,
            n_categorical=n_categorical,
            hidden_size=32,
            num_layers=1,
            n_attention_heads=2,
        ),
        cnn_config=CNNConfig(n_continuous=n_continuous, lookback=lookback, hidden_size=32, channels=(16, 32, 64, 128)),
        moe_config=MoEConfig(
            n_continuous=n_continuous,
            n_categorical=n_categorical,
            hidden_size=32,
            expert_hidden_size=48,
            num_experts=4,
            top_k=2,
        ),
        tcn_config=TCNConfig(input_size=n_continuous, hidden_size=32, num_channels=[16, 32, 32]),
        transformer_config=TransformerConfig(
            input_size=n_continuous,
            hidden_size=32,
            n_heads=2,
            n_layers=2,
            dim_feedforward=64,
            max_seq_len=max(lookback, 64),
        ),
        gate_hidden_size=32,
        gate_n_heads=2,
        gate_n_interaction_layers=1,
        model_dropout=0.0,
        dropout=0.1,
    )


def build_full_ensemble_config(
    n_continuous: Optional[int] = None,
    n_categorical: Optional[int] = None,
    lookback: int = 64,
) -> EnsembleConfig:
    """Full-size model config for GPU training."""
    n_continuous = n_continuous if n_continuous is not None else len(CONTINUOUS_FEATURES)
    n_categorical = n_categorical if n_categorical is not None else len(CATEGORICAL_FEATURES)
    return EnsembleConfig(
        tft_config=TFTConfig(
            n_continuous=n_continuous,
            n_categorical=n_categorical,
            hidden_dim=512,
            lstm_layers=4,
            attention_heads=8,
            lookback=lookback,
        ),
        lstm_config=LSTMConfig(
            n_continuous=n_continuous,
            n_categorical=n_categorical,
            hidden_size=384,
            num_layers=4,
            n_attention_heads=8,
        ),
        cnn_config=CNNConfig(n_continuous=n_continuous, lookback=lookback, hidden_size=384, channels=(64, 128, 256, 512)),
        moe_config=MoEConfig(
            n_continuous=n_continuous,
            n_categorical=n_categorical,
            hidden_size=384,
            expert_hidden_size=512,
            num_experts=8,
            top_k=2,
        ),
        tcn_config=TCNConfig(input_size=n_continuous, hidden_size=384, num_channels=[64, 128, 128, 256, 256, 512]),
        transformer_config=TransformerConfig(
            input_size=n_continuous,
            hidden_size=384,
            n_heads=8,
            n_layers=8,
            dim_feedforward=1536,
            max_seq_len=max(lookback, 512),
        ),
        gate_hidden_size=512,
        gate_n_heads=8,
        gate_n_interaction_layers=2,
        model_dropout=0.1,
        dropout=0.15,
    )


def run_training(
    max_epochs: int = 2,
    batch_size: int = 32,
    full_model: bool = False,
    checkpoint_dir: str = "models/hydra",
    data_csv: str = "",
) -> dict:
    """
    Run the full training pipeline.

    data_csv may be:
      - a real OHLCV CSV
      - a forged xauusd_hydra.parquet path with sibling train/val/test npz files
      - a prepared data directory
      - a direct train.npz path (with sibling val.npz and test.npz)
    """
    t0 = time.time()
    lookback = 64 if full_model else 32

    if not data_csv:
        raise ValueError("Data path is required. Pass a real CSV or a prepared dataset path.")

    path = Path(data_csv)
    prepared_dir = resolve_prepared_split_dir(data_csv)

    if prepared_dir is not None and (path.is_dir() or path.suffix.lower() in {".npz", ".parquet"}):
        train_ds, val_ds, test_ds, n_cont, n_cat, base_dir = load_prepared_datasets(data_csv, lookback=lookback)
        logger.info(f"Using PREPARED HYDRA dataset from {base_dir} (n_cont={n_cont}, n_cat={n_cat})")
    else:
        if path.suffix.lower() == ".parquet":
            raise ValueError(
                f"{path} is a parquet feature file, but train.npz/val.npz/test.npz were not found beside it. "
                "Run the dataset preparation phase first."
            )

        df = load_real_data(data_csv)
        n_bars = len(df)
        logger.info(f"Using REAL data: {n_bars:,} bars from {data_csv}")

        feature_dicts = df.to_dict(orient="records")
        close_prices = df["close"].values
        logger.info("Building HYDRA datasets from CSV features...")
        ds_config = DatasetConfig(
            val_split=0.15,
            test_split=0.15,
            batch_size=batch_size,
            num_workers=0,
            lookback_bars=lookback,
        )
        train_ds, val_ds, test_ds, _, _ = build_dataset_from_feature_dicts(
            feature_dicts,
            close_prices,
            config=ds_config,
        )
        n_cont = len(CONTINUOUS_FEATURES)
        n_cat = len(CATEGORICAL_FEATURES)

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.error("Dataset too small, aborting.")
        return {"error": "Dataset too small"}

    logger.info(f"Dataset: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    ds_config = DatasetConfig(batch_size=batch_size, num_workers=0, lookback_bars=lookback)
    train_dl, val_dl, test_dl = create_dataloaders(train_ds, val_ds, test_ds, config=ds_config)
    _ = test_dl

    ens_config = (
        build_full_ensemble_config(n_continuous=n_cont, n_categorical=n_cat, lookback=lookback)
        if full_model
        else build_small_ensemble_config(n_continuous=n_cont, n_categorical=n_cat, lookback=lookback)
    )

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    logger.info(f"Training on {device} for {max_epochs} epochs...")
    results = trainer.train(train_dl, val_dl)

    elapsed = time.time() - t0
    logger.info(
        f"Training complete in {elapsed:.1f}s - "
        f"Final val loss: {results['final_val_loss']:.4f}, "
        f"Best Sharpe proxy: {results['best_val_sharpe']:.2f}, "
        f"Params: {results['model_params']:,}"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="APHELION HYDRA Training")
    parser.add_argument("--full", action="store_true", help="Full training (20 epochs)")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to real OHLCV CSV, forged parquet, prepared split directory, or train.npz path",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--checkpoint-dir", type=str, default="models/hydra", help="Checkpoint dir")
    args = parser.parse_args()

    if args.full:
        max_epochs = args.epochs or 20
        full_model = True
    else:
        max_epochs = args.epochs or 2
        full_model = False

    results = run_training(
        max_epochs=max_epochs,
        batch_size=args.batch_size,
        full_model=full_model,
        checkpoint_dir=args.checkpoint_dir,
        data_csv=args.data,
    )

    if "error" not in results:
        logger.success(f"HYDRA training complete. {results['total_epochs']} epochs trained.")


if __name__ == "__main__":
    main()
