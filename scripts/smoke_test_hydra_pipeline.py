r"""
HYDRA pipeline smoke test.

Runs a minimal end-to-end check of:
  1. data load/generation
  2. dataset + dataloader creation
  3. model instantiation
  4. forward pass
  5. short trainer run
  6. checkpoint creation

Usage:
    .venv\Scripts\python.exe scripts\smoke_test_hydra_pipeline.py
    .venv\Scripts\python.exe scripts\smoke_test_hydra_pipeline.py --data data/bars/xauusd_m1.csv
    .venv\Scripts\python.exe scripts\smoke_test_hydra_pipeline.py --device cuda --epochs 2
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - hard failure is the point
    raise SystemExit("PyTorch is required for the HYDRA smoke test.") from exc

from aphelion.intelligence.hydra.dataset import (
    DatasetConfig,
    build_dataset_from_feature_dicts,
    create_dataloaders,
)
from aphelion.intelligence.hydra.ensemble import HydraGate
from aphelion.intelligence.hydra.trainer import HydraTrainer, TrainerConfig
from scripts.train_hydra import (
    build_small_ensemble_config,
    generate_synthetic_data,
    load_real_data,
)


REQUIRED_OUTPUT_KEYS = (
    "logits_5m",
    "logits_15m",
    "logits_1h",
    "quantiles_5m",
    "quantiles_15m",
    "quantiles_1h",
    "confidence",
    "uncertainty",
)


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested CUDA, but torch.cuda.is_available() is False.")
    return requested


def _load_frame(data_path: str, bars: int):
    if data_path:
        path = Path(data_path)
        if not path.exists():
            raise SystemExit(f"Data file not found: {path}")
        df = load_real_data(str(path))
        source = f"real:{path}"
    else:
        df = generate_synthetic_data(bars)
        source = f"synthetic:{bars}"
    if df.empty:
        raise SystemExit("Loaded dataframe is empty.")
    return df, source


def _build_dataloaders(df, batch_size: int, lookback: int):
    feature_dicts = df.to_dict(orient="records")
    close_prices = df["close"].values
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
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise SystemExit(
            "Dataset build failed: at least one split is empty. "
            "Increase --bars or provide a larger real dataset."
        )
    train_dl, val_dl, test_dl = create_dataloaders(train_ds, val_ds, test_ds, config=ds_config)
    return train_ds, val_ds, test_ds, train_dl, val_dl, test_dl


def _check_batch(train_dl, device: str):
    batch = next(iter(train_dl))
    if len(batch) != 6:
        raise SystemExit(f"Expected 6 tensors per batch, got {len(batch)}.")

    cont, cat, y5m, y15m, y1h, raw_ret = batch
    if cont.ndim != 3:
        raise SystemExit(f"Continuous features should be 3D, got {cont.ndim}D.")
    if cat.ndim not in (2, 3):
        raise SystemExit(f"Categorical features should be 2D or 3D, got {cat.ndim}D.")

    model = HydraGate(build_small_ensemble_config()).to(device)
    model.eval()

    with torch.inference_mode():
        outputs = model(cont.to(device), cat.to(device))

    missing = [key for key in REQUIRED_OUTPUT_KEYS if key not in outputs]
    if missing:
        raise SystemExit(f"Forward pass missing output keys: {missing}")

    print("Batch OK")
    print(f"  cont={tuple(cont.shape)}")
    print(f"  cat={tuple(cat.shape)}")
    print(f"  labels_5m={tuple(y5m.shape)}")
    print(f"  labels_15m={tuple(y15m.shape)}")
    print(f"  labels_1h={tuple(y1h.shape)}")
    print(f"  raw_ret={tuple(raw_ret.shape)}")
    print(f"  model_params={model.count_parameters():,}")


def _run_train(train_dl, val_dl, device: str, epochs: int, checkpoint_dir: str):
    ckpt_dir = Path(checkpoint_dir)
    existing = {p.name for p in ckpt_dir.glob("*.pt")} if ckpt_dir.exists() else set()

    config = TrainerConfig(
        max_epochs=epochs,
        learning_rate=1e-3,
        use_amp=(device == "cuda"),
        ensemble_config=build_small_ensemble_config(),
        checkpoint_dir=str(ckpt_dir),
        save_every_n_epochs=1,
        patience=max(3, epochs),
        warmup_epochs=0,
        gradient_accumulation_steps=1,
        mixup_alpha=0.0,
        swa_start_epoch=9999,
        label_smoothing=0.0,
    )
    trainer = HydraTrainer(config, device=device)
    results = trainer.train(train_dl, val_dl)

    required_result_keys = (
        "total_epochs",
        "best_val_sharpe",
        "best_val_loss",
        "final_train_loss",
        "final_val_loss",
        "model_params",
    )
    missing = [key for key in required_result_keys if key not in results]
    if missing:
        raise SystemExit(f"Trainer result missing keys: {missing}")

    finite_keys = ("best_val_sharpe", "best_val_loss", "final_train_loss", "final_val_loss")
    for key in finite_keys:
        if not math.isfinite(float(results[key])):
            raise SystemExit(f"Trainer result is not finite for {key}: {results[key]}")

    checkpoints = list(ckpt_dir.glob("*.pt"))
    if not checkpoints:
        raise SystemExit(f"No checkpoints created in {ckpt_dir}")

    new_files = sorted({p.name for p in checkpoints} - existing)
    print("Training OK")
    print(f"  results={results}")
    print(f"  checkpoint_dir={ckpt_dir}")
    print(f"  new_checkpoints={new_files or 'reused existing names'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the HYDRA training pipeline.")
    parser.add_argument("--data", type=str, default="", help="Optional path to real OHLCV CSV.")
    parser.add_argument("--bars", type=int, default=500, help="Synthetic bars to generate if --data is absent.")
    parser.add_argument("--epochs", type=int, default=2, help="Short training epochs to run.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for the smoke test.")
    parser.add_argument("--lookback", type=int, default=16, help="Lookback window for the dataset.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/hydra_smoke",
        help="Directory where smoke-test checkpoints are written.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. Defaults to auto.",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    df, source = _load_frame(args.data, args.bars)
    train_ds, val_ds, test_ds, train_dl, val_dl, _ = _build_dataloaders(
        df,
        batch_size=args.batch_size,
        lookback=args.lookback,
    )

    print("HYDRA pipeline smoke test")
    print(f"  torch={torch.__version__}")
    print(f"  device={device}")
    print(f"  source={source}")
    print(f"  rows={len(df):,}")
    print(f"  train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")
    print(f"  batch_size={args.batch_size} lookback={args.lookback} epochs={args.epochs}")
    if torch.cuda.is_available():
        print(f"  cuda_device={torch.cuda.get_device_name(0)}")

    numeric = df.select_dtypes(include=[np.number])
    if numeric.isna().any().any():
        raise SystemExit("Numeric dataframe columns still contain NaN values.")
    if np.isinf(numeric.to_numpy()).any():
        raise SystemExit("Numeric dataframe columns still contain Inf values.")

    _check_batch(train_dl, device)
    _run_train(train_dl, val_dl, device, args.epochs, args.checkpoint_dir)

    print("PASS: HYDRA pipeline completed end-to-end.")


if __name__ == "__main__":
    main()
