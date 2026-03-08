"""
APHELION HYDRA Evaluation Script
Generates a vast synthetic OHLCV dataset with defined patterns to train the
HYDRA Full Ensemble, validating the architecture end-to-end for Phase 4.
"""

import os
import time
import numpy as np
import pandas as pd
from loguru import logger

from aphelion.intelligence.hydra.dataset import build_dataset_from_feature_dicts, create_dataloaders
from aphelion.intelligence.hydra.trainer import HydraTrainer, TrainerConfig
from aphelion.intelligence.hydra.ensemble import EnsembleConfig


def generate_synthetic_data(n_bars: int = 10000) -> pd.DataFrame:
    """Generate synthetic OHLCV data with some repeating patterns for MoE to learn."""
    logger.info(f"Generating {n_bars} synthetic bars for training...")
    
    np.random.seed(42)
    
    # 1. Base random walk
    returns = np.random.normal(0, 0.001, n_bars)
    
    # Add trend regime (Bars 1000-2000)
    returns[1000:2000] += 0.0005 
    
    # Add volatile regime (Bars 4000-5000)
    returns[4000:5000] *= 3.0
    
    # Generate prices
    close = 2000.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.0005, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.0005, n_bars)))
    open_p = np.roll(close, 1)
    open_p[0] = 2000.0
    
    df = pd.DataFrame({
        "timestamp_ms": np.arange(n_bars) * 60000,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": np.random.randint(10, 1000, n_bars),
        # Required categorical fields based on dict
        "day_of_week": (np.arange(n_bars) // (24 * 60)) % 5,
        "session_id": np.random.randint(0, 5, n_bars),
    })
    
    # Add the remaining 60+ continuous features required by HYDRA
    from aphelion.intelligence.hydra.dataset import CONTINUOUS_FEATURES
    
    for feat in CONTINUOUS_FEATURES:
        if feat not in df.columns:
            df[feat] = np.random.normal(0, 1, n_bars)
            
    return df

def test_train_loop():
    logger.info("Starting Phase 4 Validation...")
    df = generate_synthetic_data(500) # Tiny dataset for instant validation
    
    feature_dicts = df.to_dict(orient="records")
    close_prices = df["close"].values
    
    logger.info("Building HYDRA Datasets...")
    from aphelion.intelligence.hydra.dataset import DatasetConfig
    ds_config = DatasetConfig(val_split=0.2, test_split=0.1, num_workers=0)
    train_ds, val_ds, test_ds, _, _ = build_dataset_from_feature_dicts(
        feature_dicts, close_prices, config=ds_config
    )
    
    logger.info(f"Dataset generated. Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    ds_config.batch_size = 32
    train_dl, val_dl, test_dl = create_dataloaders(train_ds, val_ds, test_ds, config=ds_config)
    
    # Tiny Config to ensure it runs fast on CPU
    ens_config = EnsembleConfig()
    ens_config.tft_config.hidden_size = 16
    ens_config.lstm_config.hidden_size = 16
    ens_config.cnn_config.hidden_size = 16
    ens_config.moe_config.hidden_size = 16
    ens_config.gate_hidden_size = 32
    
    config = TrainerConfig(
        max_epochs=2, # Just test convergence
        learning_rate=1e-3,
        use_amp=False, # Disable AMP for CPU
        ensemble_config=ens_config,
        checkpoint_dir="models/test_hydra"
    )
    
    trainer = HydraTrainer(config, device="cpu")
    
    logger.info("Beginning Training...")
    metrics = trainer.train(train_dl, val_dl)
    
    logger.info("Training Complete!")
    logger.info(f"Final Val Loss: {metrics['final_val_loss']:.4f}")
    logger.info(f"Best Sharpe Proxy: {metrics['best_val_sharpe']:.2f}")

if __name__ == "__main__":
    test_train_loop()
