"""
Train a tiny HYDRA model to validate the real data pipeline.

This is not a separate toy system. It uses the same HYDRA model family,
the same dataset objects, and the same trainer, but with a much smaller
configuration so you can quickly answer:

  - did the data land where we expect
  - are the splits chronological
  - do train/val/test timestamps overlap
  - was scaling fit on train only
  - can a tiny HYDRA train end-to-end on this dataset

Examples:
    python scripts/train_hydra_tiny.py
    python scripts/train_hydra_tiny.py --symbol XAUUSD
    python scripts/train_hydra_tiny.py --data data/processed/XAUUSD/xauusd_hydra.parquet
    python scripts/train_hydra_tiny.py --data data/raw/xauusd_m5.csv --epochs 2 --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as exc:  # pragma: no cover - hard failure is the point
    raise SystemExit("PyTorch is required. Install the ml extras first.") from exc

from aphelion.intelligence.hydra.dataset import DatasetConfig, create_dataloaders
from aphelion.intelligence.hydra.ensemble import EnsembleConfig, HydraGate
from aphelion.intelligence.hydra.trainer import HydraTrainer, TrainerConfig
from aphelion.intelligence.hydra.tft import TFTConfig
from aphelion.intelligence.hydra.lstm import LSTMConfig
from aphelion.intelligence.hydra.cnn import CNNConfig
from aphelion.intelligence.hydra.moe import MoEConfig
from aphelion.intelligence.hydra.tcn import TCNConfig
from aphelion.intelligence.hydra.transformer import TransformerConfig
from scripts.train_hydra import (
    build_dataset_from_feature_dicts,
    load_prepared_datasets,
    load_real_data,
    resolve_prepared_split_dir,
)

MAX_HORIZON = 60


def build_tiny_ensemble_config(
    n_continuous: int,
    n_categorical: int,
    lookback: int = 32,
) -> EnsembleConfig:
    """Very small HYDRA config for fast validation runs."""
    return EnsembleConfig(
        tft_config=TFTConfig(
            n_continuous=n_continuous,
            n_categorical=n_categorical,
            hidden_dim=16,
            lstm_layers=1,
            attention_heads=2,
            lookback=lookback,
        ),
        lstm_config=LSTMConfig(
            n_continuous=n_continuous,
            n_categorical=n_categorical,
            hidden_size=16,
            num_layers=1,
            n_attention_heads=2,
        ),
        cnn_config=CNNConfig(
            n_continuous=n_continuous,
            lookback=lookback,
            hidden_size=16,
            channels=(8, 16, 32, 64),
        ),
        moe_config=MoEConfig(
            n_continuous=n_continuous,
            n_categorical=n_categorical,
            hidden_size=16,
            expert_hidden_size=24,
            num_experts=2,
            top_k=1,
        ),
        tcn_config=TCNConfig(
            input_size=n_continuous,
            hidden_size=16,
            num_channels=[8, 16, 16],
        ),
        transformer_config=TransformerConfig(
            input_size=n_continuous,
            hidden_size=16,
            n_heads=2,
            n_layers=1,
            dim_feedforward=32,
            max_seq_len=max(lookback, 64),
        ),
        gate_hidden_size=16,
        gate_n_heads=2,
        gate_n_interaction_layers=1,
        model_dropout=0.0,
        dropout=0.05,
    )


def resolve_default_data_path(symbol: str) -> Path | None:
    lower = symbol.lower()
    candidates = [
        Path("data/processed") / symbol / f"{lower}_hydra.parquet",
        Path("data/processed") / symbol,
        Path("data/processed") / f"{lower}_hydra.parquet",
        Path("data/raw") / symbol / f"{lower}_m5.csv",
        Path("data/raw") / f"{lower}_m5.csv",
        Path("data/raw") / symbol / f"{lower}_m1.csv",
        Path("data/raw") / f"{lower}_m1.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested CUDA, but torch.cuda.is_available() is False.")
    return requested


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_raw_fetch_summary(symbol: str) -> dict[str, Any]:
    lower = symbol.lower()
    raw_root = Path("data/raw")
    nested_dir = raw_root / symbol

    flat_files = sorted(raw_root.glob(f"{lower}_*.csv"))
    nested_files = sorted(nested_dir.glob(f"{lower}_*.csv")) if nested_dir.exists() else []
    symbol_files = nested_files or flat_files
    all_csv = sorted(raw_root.rglob("*.csv"))

    return {
        "raw_root": str(raw_root.resolve()),
        "layout": "nested" if nested_files else ("flat" if flat_files else "missing"),
        "symbol_file_count": len(symbol_files),
        "all_csv_count": len(all_csv),
        "sample_files": [p.name for p in symbol_files[:5]],
    }


def parse_timestamp_array(values: np.ndarray) -> pd.DatetimeIndex:
    stamps = pd.to_datetime(np.asarray(values).astype(str), utc=True, errors="coerce")
    if stamps.isna().any():
        raise ValueError("One or more timestamps could not be parsed.")
    return pd.DatetimeIndex(stamps)


def summarize_prepared_split(split_path: Path, lookback: int) -> dict[str, Any]:
    data = np.load(split_path, allow_pickle=True)
    keys = set(data.files)
    if "X_cont" not in keys or "X_cat" not in keys:
        raise ValueError(f"{split_path} is missing X_cont or X_cat.")
    if "timestamps" not in keys:
        raise ValueError(f"{split_path} is missing timestamps.")

    x_cont = np.asarray(data["X_cont"], dtype=np.float32)
    x_cat = np.asarray(data["X_cat"], dtype=np.int64)
    timestamps = parse_timestamp_array(data["timestamps"])

    if len(x_cont) != len(x_cat) or len(x_cont) != len(timestamps):
        raise ValueError(f"{split_path} has mismatched row counts across arrays.")
    if len(timestamps) <= lookback + MAX_HORIZON:
        raise ValueError(
            f"{split_path} is too small for lookback={lookback} and horizon guard={MAX_HORIZON}."
        )
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"{split_path} timestamps are not monotonic increasing.")
    if not timestamps.is_unique:
        raise ValueError(f"{split_path} timestamps are not unique.")

    usable_centers = len(x_cont) - lookback - MAX_HORIZON
    last_center_idx = len(x_cont) - MAX_HORIZON - 1

    return {
        "path": str(split_path),
        "rows": int(len(x_cont)),
        "n_cont": int(x_cont.shape[1]),
        "n_cat": int(x_cat.shape[1]),
        "timestamps": timestamps,
        "usable_centers": int(usable_centers),
        "holdout_rows": MAX_HORIZON,
        "last_center_idx": int(last_center_idx),
        "start": str(timestamps[0]),
        "end": str(timestamps[-1]),
        "last_used_timestamp": str(timestamps[last_center_idx]),
    }


def validate_prepared_dataset(data_path: str, lookback: int) -> tuple[dict[str, Any], int, int]:
    base_dir = resolve_prepared_split_dir(data_path)
    if base_dir is None:
        raise ValueError("Prepared HYDRA splits were not found.")

    scaler = load_json_if_exists(base_dir / "scaler.json")
    meta = load_json_if_exists(base_dir / "dataset_meta.json")

    train = summarize_prepared_split(base_dir / "train.npz", lookback)
    val = summarize_prepared_split(base_dir / "val.npz", lookback)
    test = summarize_prepared_split(base_dir / "test.npz", lookback)

    if train["n_cont"] != val["n_cont"] or train["n_cont"] != test["n_cont"]:
        raise ValueError("Prepared splits do not agree on continuous feature count.")
    if train["n_cat"] != val["n_cat"] or train["n_cat"] != test["n_cat"]:
        raise ValueError("Prepared splits do not agree on categorical feature count.")
    if train["timestamps"][-1] >= val["timestamps"][0]:
        raise ValueError("Train and validation timestamps overlap or are out of order.")
    if val["timestamps"][-1] >= test["timestamps"][0]:
        raise ValueError("Validation and test timestamps overlap or are out of order.")

    if scaler is not None and scaler.get("fit_on") != "train_only":
        raise ValueError("scaler.json does not say fit_on=train_only.")
    if meta is not None and meta.get("split_method") != "chronological (no shuffle)":
        raise ValueError("dataset_meta.json does not say chronological (no shuffle).")

    report = {
        "mode": "prepared",
        "base_dir": str(base_dir.resolve()),
        "scaler_fit_on": scaler.get("fit_on") if scaler else "missing",
        "split_method": meta.get("split_method") if meta else "missing",
        "train": train,
        "val": val,
        "test": test,
    }
    return report, train["n_cont"], train["n_cat"]


def validate_csv_dataset(csv_path: str, lookback: int) -> tuple[dict[str, Any], Any, Any, Any, int, int]:
    df = load_real_data(csv_path)
    feature_dicts = df.to_dict(orient="records")
    close_prices = df["close"].values

    ds_config = DatasetConfig(
        lookback_bars=lookback,
        val_split=0.15,
        test_split=0.15,
        batch_size=16,
        num_workers=0,
    )
    train_ds, val_ds, test_ds, means, stds = build_dataset_from_feature_dicts(
        feature_dicts,
        close_prices,
        config=ds_config,
    )

    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(stds)):
        raise ValueError("CSV dataset normalization statistics contain NaN/Inf.")

    time_info = {
        "train_start": None,
        "train_end": None,
        "val_start": None,
        "val_end": None,
        "test_start": None,
        "test_end": None,
        "has_timestamps": False,
    }
    if "timestamp" in df.columns:
        timestamps = parse_timestamp_array(df["timestamp"].values)
        indices = list(range(lookback, len(df) - MAX_HORIZON))
        train_split_idx = int(len(indices) * (1 - ds_config.val_split - ds_config.test_split))
        val_split_idx = int(len(indices) * (1 - ds_config.test_split))
        train_indices = indices[:train_split_idx]
        val_indices = indices[train_split_idx:val_split_idx]
        test_indices = indices[val_split_idx:]
        if timestamps[train_indices[-1]] >= timestamps[val_indices[0]]:
            raise ValueError("CSV-derived train and validation windows overlap or are out of order.")
        if timestamps[val_indices[-1]] >= timestamps[test_indices[0]]:
            raise ValueError("CSV-derived validation and test windows overlap or are out of order.")
        time_info = {
            "train_start": str(timestamps[train_indices[0]]),
            "train_end": str(timestamps[train_indices[-1]]),
            "val_start": str(timestamps[val_indices[0]]),
            "val_end": str(timestamps[val_indices[-1]]),
            "test_start": str(timestamps[test_indices[0]]),
            "test_end": str(timestamps[test_indices[-1]]),
            "has_timestamps": True,
        }

    report = {
        "mode": "csv",
        "csv_path": str(Path(csv_path).resolve()),
        "rows": int(len(df)),
        "n_cont": int(train_ds._cont.shape[1]),
        "n_cat": int(train_ds._cat.shape[1]),
        "normalization_fit": "training slice only",
        "time_info": time_info,
    }
    return report, train_ds, val_ds, test_ds, int(train_ds._cont.shape[1]), int(train_ds._cat.shape[1])


def build_dataloaders_for_training(train_ds, val_ds, test_ds, batch_size: int, lookback: int):
    effective_batch = min(batch_size, len(train_ds))
    if effective_batch <= 0:
        raise ValueError("Training dataset is empty.")
    ds_config = DatasetConfig(batch_size=effective_batch, num_workers=0, lookback_bars=lookback)
    return create_dataloaders(train_ds, val_ds, test_ds, config=ds_config), effective_batch


def cap_dataset_centers(dataset, max_centers: int | None):
    if max_centers is None or max_centers <= 0 or len(dataset) <= max_centers:
        return dataset
    dataset._indices = dataset._indices[:max_centers]
    return dataset


def check_tiny_forward_pass(train_dl, device: str, n_cont: int, n_cat: int, lookback: int) -> int:
    model = HydraGate(build_tiny_ensemble_config(n_cont, n_cat, lookback)).to(device)
    batch = next(iter(train_dl))
    cont, cat = batch[0].to(device), batch[1].to(device)
    with torch.inference_mode():
        outputs = model(cont, cat)

    required_keys = {
        "logits_5m",
        "logits_15m",
        "logits_1h",
        "quantiles_5m",
        "quantiles_15m",
        "quantiles_1h",
        "confidence",
        "uncertainty",
    }
    missing = required_keys - set(outputs)
    if missing:
        raise ValueError(f"Tiny HYDRA forward pass is missing outputs: {sorted(missing)}")
    return model.count_parameters()


def run_tiny_training(
    train_ds,
    val_ds,
    test_ds,
    n_cont: int,
    n_cat: int,
    lookback: int,
    batch_size: int,
    epochs: int,
    device: str,
    checkpoint_dir: str,
    max_centers_per_split: int | None,
) -> dict[str, Any]:
    train_ds = cap_dataset_centers(train_ds, max_centers_per_split)
    val_ds = cap_dataset_centers(val_ds, max_centers_per_split)
    test_ds = cap_dataset_centers(test_ds, max_centers_per_split)
    (train_dl, val_dl, _), effective_batch = build_dataloaders_for_training(
        train_ds,
        val_ds,
        test_ds,
        batch_size=batch_size,
        lookback=lookback,
    )
    model_params = check_tiny_forward_pass(train_dl, device, n_cont, n_cat, lookback)

    trainer_cfg = TrainerConfig(
        max_epochs=epochs,
        learning_rate=2e-3,
        use_amp=(device == "cuda"),
        ensemble_config=build_tiny_ensemble_config(n_cont, n_cat, lookback),
        checkpoint_dir=checkpoint_dir,
        save_every_n_epochs=1,
        patience=max(3, epochs),
        warmup_epochs=0,
        gradient_accumulation_steps=1,
        mixup_alpha=0.0,
        swa_start_epoch=9999,
        label_smoothing=0.0,
    )
    trainer = HydraTrainer(trainer_cfg, device=device)
    results = trainer.train(train_dl, val_dl)
    for key in ("best_val_sharpe", "best_val_loss", "final_train_loss", "final_val_loss"):
        if not math.isfinite(float(results[key])):
            raise ValueError(f"Tiny HYDRA training produced a non-finite {key}.")
    results["model_params"] = model_params
    results["effective_batch_size"] = effective_batch
    return results


def print_fetch_summary(summary: dict[str, Any]) -> None:
    print("Fetch check")
    print(f"  raw_root={summary['raw_root']}")
    print(f"  layout={summary['layout']}")
    print(f"  symbol_csv_files={summary['symbol_file_count']}")
    print(f"  all_market_csv_files={summary['all_csv_count']}")
    print(f"  sample_files={summary['sample_files']}")


def print_validation_report(report: dict[str, Any], lookback: int) -> None:
    print("Leakage check")
    print(f"  mode={report['mode']}")
    if report["mode"] == "prepared":
        print(f"  data_dir={report['base_dir']}")
        print(f"  scaler_fit_on={report['scaler_fit_on']}")
        print(f"  split_method={report['split_method']}")
        for split_name in ("train", "val", "test"):
            split = report[split_name]
            print(
                f"  {split_name}: rows={split['rows']:,} "
                f"range={split['start']} -> {split['end']} "
                f"usable_centers={split['usable_centers']:,} "
                f"holdout_rows={split['holdout_rows']}"
            )
        print(f"  lookback_guard={lookback}")
        print(f"  horizon_guard={MAX_HORIZON}")
    else:
        print(f"  csv_path={report['csv_path']}")
        print(f"  rows={report['rows']:,}")
        print(f"  normalization_fit={report['normalization_fit']}")
        time_info = report["time_info"]
        if time_info["has_timestamps"]:
            print(
                f"  train_window={time_info['train_start']} -> {time_info['train_end']}"
            )
            print(
                f"  val_window={time_info['val_start']} -> {time_info['val_end']}"
            )
            print(
                f"  test_window={time_info['test_start']} -> {time_info['test_end']}"
            )
        else:
            print("  timestamps=missing in CSV, so ordering was checked by index only")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny HYDRA to validate the data pipeline.")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Primary symbol, default XAUUSD.")
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Prepared parquet/dir/npz path or raw CSV path. If omitted, the script resolves a sensible default.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Tiny training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Requested batch size.")
    parser.add_argument("--lookback", type=int, default=32, help="Lookback window.")
    parser.add_argument(
        "--max-centers-per-split",
        type=int,
        default=None,
        help="Optional cap on train/val/test sample centers for faster validation.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/hydra_tiny_validator",
        help="Directory for tiny validation checkpoints.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    data_path = Path(args.data) if args.data else resolve_default_data_path(args.symbol)
    if data_path is None:
        raise SystemExit(
            "No dataset path could be resolved. If you have prepared splits, point --data at the parquet or its "
            "directory. If you only have raw CSV, point --data at that file."
        )

    raw_summary = collect_raw_fetch_summary(args.symbol)
    print("HYDRA tiny validator")
    print(f"  device={device}")
    print(f"  data={data_path}")
    print_fetch_summary(raw_summary)

    prepared_dir = resolve_prepared_split_dir(str(data_path))
    if prepared_dir is not None and (data_path.is_dir() or data_path.suffix.lower() in {".parquet", ".npz"}):
        report, n_cont, n_cat = validate_prepared_dataset(str(data_path), lookback=args.lookback)
        train_ds, val_ds, test_ds, _, _, _ = load_prepared_datasets(str(data_path), lookback=args.lookback)
    else:
        report, train_ds, val_ds, test_ds, n_cont, n_cat = validate_csv_dataset(
            str(data_path),
            lookback=args.lookback,
        )

    print_validation_report(report, lookback=args.lookback)
    results = run_tiny_training(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        n_cont=n_cont,
        n_cat=n_cat,
        lookback=args.lookback,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        max_centers_per_split=args.max_centers_per_split,
    )

    print("Tiny training")
    print(f"  model_params={results['model_params']:,}")
    print(f"  effective_batch_size={results['effective_batch_size']}")
    print(f"  epochs={results['total_epochs']}")
    print(f"  final_train_loss={results['final_train_loss']:.4f}")
    print(f"  final_val_loss={results['final_val_loss']:.4f}")
    print(f"  best_val_sharpe={results['best_val_sharpe']:.4f}")
    print(f"  checkpoint_dir={Path(args.checkpoint_dir).resolve()}")
    print("PASS: tiny HYDRA validated this data path.")


if __name__ == "__main__":
    main()
