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
    load_prepared_datasets,
    load_real_data,
    resolve_prepared_split_dir,
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


def _load_frame(data_path: str):
    if not data_path:
        raise SystemExit(
            "A data path is required. Pass a real OHLCV CSV, a forged parquet path, "
            "a prepared split directory, or a train.npz path."
        )
    path = Path(data_path)
    if not path.exists():
        raise SystemExit(f"Data file not found: {path}")

    prepared_dir = resolve_prepared_split_dir(str(path))
    if prepared_dir is not None and (path.is_dir() or path.suffix.lower() in {".npz", ".parquet"}):
        return None, f"prepared:{prepared_dir}"

    df = load_real_data(str(path))
    if df.empty:
        raise SystemExit("Loaded dataframe is empty.")
    return df, f"real:{path}"


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


def _build_prepared_dataloaders(data_path: str, batch_size: int, lookback: int):
    train_ds, val_ds, test_ds, n_cont, n_cat, base_dir = load_prepared_datasets(data_path, lookback=lookback)
    ds_config = DatasetConfig(batch_size=batch_size, num_workers=0, lookback_bars=lookback)
    train_dl, val_dl, test_dl = create_dataloaders(train_ds, val_ds, test_ds, config=ds_config)
    return train_ds, val_ds, test_ds, train_dl, val_dl, test_dl, n_cont, n_cat, base_dir


def _check_batch(train_dl, device: str, n_cont: int, n_cat: int, lookback: int):
    batch = next(iter(train_dl))
    if len(batch) != 6:
        raise SystemExit(f"Expected 6 tensors per batch, got {len(batch)}.")

    cont, cat, y5m, y15m, y1h, raw_ret = batch
    if cont.ndim != 3:
        raise SystemExit(f"Continuous features should be 3D, got {cont.ndim}D.")
    if cat.ndim not in (2, 3):
        raise SystemExit(f"Categorical features should be 2D or 3D, got {cat.ndim}D.")

    model = HydraGate(
        build_small_ensemble_config(
            n_continuous=n_cont,
            n_categorical=n_cat,
            lookback=lookback,
        )
    ).to(device)
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


def _run_train(
    train_dl,
    val_dl,
    device: str,
    epochs: int,
    checkpoint_dir: str,
    n_cont: int,
    n_cat: int,
    lookback: int,
):
    ckpt_dir = Path(checkpoint_dir)
    existing = {p.name for p in ckpt_dir.glob("*.pt")} if ckpt_dir.exists() else set()

    config = TrainerConfig(
        max_epochs=epochs,
        learning_rate=1e-3,
        use_amp=(device == "cuda"),
        ensemble_config=build_small_ensemble_config(
            n_continuous=n_cont,
            n_categorical=n_cat,
            lookback=lookback,
        ),
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
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to real OHLCV CSV, forged parquet, prepared split directory, or train.npz path.",
    )
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
    df, source = _load_frame(args.data)
    if df is None:
        train_ds, val_ds, test_ds, train_dl, val_dl, _, n_cont, n_cat, base_dir = _build_prepared_dataloaders(
            args.data,
            batch_size=args.batch_size,
            lookback=args.lookback,
        )
        row_count = len(train_ds) + len(val_ds) + len(test_ds)
        source = f"prepared:{base_dir}"
    else:
        train_ds, val_ds, test_ds, train_dl, val_dl, _ = _build_dataloaders(
            df,
            batch_size=args.batch_size,
            lookback=args.lookback,
        )
        n_cont = train_ds.x_cont.shape[1]
        n_cat = train_ds.x_cat.shape[1]
        row_count = len(df)

    print("HYDRA pipeline smoke test")
    print(f"  torch={torch.__version__}")
    print(f"  device={device}")
    print(f"  source={source}")
    print(f"  rows={row_count:,}")
    print(f"  train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")
    print(f"  features_cont={n_cont} features_cat={n_cat}")
    print(f"  batch_size={args.batch_size} lookback={args.lookback} epochs={args.epochs}")
    if torch.cuda.is_available():
        print(f"  cuda_device={torch.cuda.get_device_name(0)}")

    if df is not None:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.isna().any().any():
            raise SystemExit("Numeric dataframe columns still contain NaN values.")
        if np.isinf(numeric.to_numpy()).any():
            raise SystemExit("Numeric dataframe columns still contain Inf values.")

    _check_batch(train_dl, device, n_cont, n_cat, args.lookback)
    _run_train(train_dl, val_dl, device, args.epochs, args.checkpoint_dir, n_cont, n_cat, args.lookback)

    print("PASS: HYDRA pipeline completed end-to-end.")


if __name__ == "__main__":
    main()
