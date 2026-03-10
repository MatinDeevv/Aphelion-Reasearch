# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║              APHELION HYDRA — 8-HOUR ULTRA TRAINING SESSION                ║
# ║              160M Parameter 6-Model Neural Ensemble for XAUUSD             ║
# ║                                                                            ║
# ║   1. Run this cell                                                         ║
# ║   2. Upload Aphelion_data.zip when prompted (18 MB data only)              ║
# ║   3. Authorize Google Drive when prompted                                  ║
# ║   4. Walk away for 8 hours                                                 ║
# ║   5. Come back to a trained model saved to Google Drive                    ║
# ║                                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os, sys, time, shutil, zipfile, subprocess
from pathlib import Path

START_TIME = time.time()
MAX_RUNTIME = 8 * 3600  # 8 hours in seconds
SAFETY_MARGIN = 600     # Stop 10 min early to save final checkpoint

# ═══════════════════════════════════════════════════════════════
#  PHASE 1: GPU CHECK
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 1: GPU VERIFICATION")
print("=" * 70)

import torch
if not torch.cuda.is_available():
    raise RuntimeError("NO GPU! Go to Runtime → Change runtime type → A100 GPU")

gpu_name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
vram_bytes = getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)
if vram_bytes == 0:
    vram_bytes = torch.cuda.mem_get_info(0)[1]
vram_gb = vram_bytes / 1024**3
print(f"  GPU: {gpu_name}")
print(f"  VRAM: {vram_gb:.1f} GB")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.version.cuda}")

# Determine optimal batch size based on GPU
if vram_gb >= 35:      # A100 40/80GB
    BATCH_SIZE = 128
    NUM_WORKERS = 4
elif vram_gb >= 20:    # L4 24GB
    BATCH_SIZE = 96
    NUM_WORKERS = 2
else:                  # T4 16GB
    BATCH_SIZE = 64
    NUM_WORKERS = 2

print(f"  Optimal batch size: {BATCH_SIZE}")
print("  ✓ GPU ready")
print()

# ═══════════════════════════════════════════════════════════════
#  PHASE 2: CLONE REPO FROM GITHUB
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 2: CLONE REPO FROM GITHUB")
print("=" * 70)

REPO_URL = "https://github.com/MatinDeevv/Aphelion-Reasearch.git"
PROJECT = "/content/Aphelion-Reasearch"

if os.path.exists(os.path.join(PROJECT, 'aphelion', 'intelligence', 'hydra', 'ensemble.py')):
    print("  Repo already cloned, skipping...")
else:
    if os.path.exists(PROJECT):
        shutil.rmtree(PROJECT)
    print(f"  Cloning {REPO_URL} ...")
    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, PROJECT], check=True)
    print("  ✓ Clone complete")

os.chdir(PROJECT)
print(f"  Project root: {PROJECT}")
print(f"  Contents: {sorted(os.listdir('.')[:15])}")

# Verify critical files
assert os.path.exists('aphelion/intelligence/hydra/ensemble.py'), "Missing ensemble.py!"
assert os.path.exists('scripts/train_hydra.py'), "Missing train_hydra.py!"
print("  ✓ Project structure verified")
print()

# ═══════════════════════════════════════════════════════════════
#  PHASE 3: UPLOAD DATA (CSV files not in git)
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 3: UPLOAD TRAINING DATA")
print("=" * 70)

if os.path.exists('data/bars') and any(f.endswith('.csv') for f in os.listdir('data/bars')):
    print("  Data already present, skipping upload...")
else:
    from google.colab import files
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║  Click 'Choose Files' below and select:         ║")
    print("  ║  C:\\Users\\marti\\Aphelion_data.zip               ║")
    print("  ║  (18 MB — contains only CSV bar data)           ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()
    uploaded = files.upload()
    zip_name = list(uploaded.keys())[0]
    print(f"  Extracting {zip_name}...")
    with zipfile.ZipFile(zip_name, 'r') as z:
        z.extractall('.')
    print("  ✓ Data extracted")

# Verify data exists
assert os.path.exists('data/bars'), "data/bars/ not found after extraction!"
csv_count = len([f for f in os.listdir('data/bars') if f.endswith('.csv')])
assert csv_count > 0, "No CSV files found in data/bars/!"
print(f"  Found {csv_count} CSV files in data/bars/")
print()

# ═══════════════════════════════════════════════════════════════
#  PHASE 4: INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 4: INSTALLING DEPENDENCIES")
print("=" * 70)

subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[ml]', '-q'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'loguru', '-q'], check=True)

# Verify imports
sys.path.insert(0, PROJECT)
from aphelion.intelligence.hydra.ensemble import HydraGate, EnsembleConfig
from aphelion.intelligence.hydra.trainer import HydraTrainer, TrainerConfig
from aphelion.intelligence.hydra.tft import TFTConfig
from aphelion.intelligence.hydra.lstm import LSTMConfig
from aphelion.intelligence.hydra.cnn import CNNConfig
from aphelion.intelligence.hydra.moe import MoEConfig
from aphelion.intelligence.hydra.tcn import TCNConfig
from aphelion.intelligence.hydra.transformer import TransformerConfig
from aphelion.intelligence.hydra.dataset import (
    CONTINUOUS_FEATURES, DatasetConfig,
    build_dataset_from_feature_dicts, create_dataloaders,
)
from scripts.train_hydra import load_real_data, build_full_ensemble_config

print("  All imports OK ✓")
print()

# ═══════════════════════════════════════════════════════════════
#  PHASE 5: MOUNT GOOGLE DRIVE (auto-save checkpoints)
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 5: MOUNTING GOOGLE DRIVE FOR AUTO-SAVE")
print("=" * 70)

from google.colab import drive
drive.mount('/content/drive')

DRIVE_SAVE = '/content/drive/MyDrive/APHELION_HYDRA_CHECKPOINTS'
os.makedirs(DRIVE_SAVE, exist_ok=True)
print(f"  Checkpoints will auto-save to: {DRIVE_SAVE}")
print()

# ═══════════════════════════════════════════════════════════════
#  PHASE 6: VERIFY DATA
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 6: DATA VERIFICATION")
print("=" * 70)

import pandas as pd
data_dir = 'data/bars'
csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv') and 'tick' not in f])
total_bars = 0
for f in csv_files:
    path = os.path.join(data_dir, f)
    rows = len(pd.read_csv(path))
    total_bars += rows
    print(f"  {f:35s} {rows:>10,} bars")

print(f"\n  Total: {total_bars:,} bars")
print()

# ═══════════════════════════════════════════════════════════════
#  TRAINING PLAN
# ═══════════════════════════════════════════════════════════════
#
#  The smartest trading bot trains on MULTIPLE timeframes:
#
#  Round 1: XAUUSD M5  (100K bars) — 200 epochs — Primary signal
#  Round 2: XAUUSD M15 (50K bars)  — 150 epochs — Confirmation signal
#  Round 3: XAUUSD H1  (30K bars)  — 150 epochs — Trend structure
#  Round 4: XAUUSD M1  (100K bars) — 100 epochs — Micro-scalping
#  Round 5: XAGUSD M5  (100K bars) — 100 epochs — Cross-asset (gold/silver)
#  Round 6: EURUSD M5  (100K bars) —  80 epochs — Macro correlation
#
#  Each round saves its own checkpoint + copies to Google Drive
#  Time watchdog stops training cleanly before 8-hour limit
#
# ═══════════════════════════════════════════════════════════════

TRAINING_ROUNDS = [
    # (data_file, epochs, checkpoint_tag, description)
    ("data/bars/xauusd_m5.csv",  200, "xauusd_m5",  "XAUUSD M5 — Primary Signal (100K bars)"),
    ("data/bars/xauusd_m15.csv", 150, "xauusd_m15", "XAUUSD M15 — Confirmation (50K bars)"),
    ("data/bars/xauusd_h1.csv",  150, "xauusd_h1",  "XAUUSD H1 — Trend Structure (30K bars)"),
    ("data/bars/xauusd_m1.csv",  100, "xauusd_m1",  "XAUUSD M1 — Micro Entry (100K bars)"),
    ("data/bars/xagusd_m5.csv",  100, "xagusd_m5",  "XAGUSD M5 — Cross-Asset (100K bars)"),
    ("data/bars/eurusd_m5.csv",   80, "eurusd_m5",  "EURUSD M5 — Macro Proxy (100K bars)"),
]


def time_remaining():
    elapsed = time.time() - START_TIME
    return MAX_RUNTIME - SAFETY_MARGIN - elapsed


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


def save_to_drive(ckpt_dir, tag):
    """Copy checkpoints to Google Drive for persistence."""
    drive_subdir = os.path.join(DRIVE_SAVE, tag)
    os.makedirs(drive_subdir, exist_ok=True)
    for f in os.listdir(ckpt_dir):
        if f.endswith('.pt'):
            src = os.path.join(ckpt_dir, f)
            dst = os.path.join(drive_subdir, f)
            shutil.copy2(src, dst)
    print(f"  📁 Checkpoints saved to Drive: {drive_subdir}")


def run_one_round(data_file, max_epochs, tag, description):
    """Train one full round on a single dataset."""
    remaining = time_remaining()
    if remaining < 300:  # Less than 5 min left
        print(f"\n  ⏰ Only {format_time(remaining)} left — skipping {tag}")
        return None

    print(f"\n{'═' * 70}")
    print(f"  TRAINING: {description}")
    print(f"  Time remaining: {format_time(remaining)}")
    print(f"  Epochs: {max_epochs} | Batch: {BATCH_SIZE} | GPU: {gpu_name}")
    print(f"{'═' * 70}\n")

    if not os.path.exists(data_file):
        print(f"  ⚠ {data_file} not found, skipping...")
        return None

    # Load data
    df = load_real_data(data_file)
    n_bars = len(df)
    feature_dicts = df.to_dict(orient="records")
    close_prices = df["close"].values

    # Build datasets with A100-optimized settings
    ds_config = DatasetConfig(
        val_split=0.15,
        test_split=0.15,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        lookback_bars=64,
    )
    train_ds, val_ds, test_ds, means, stds = build_dataset_from_feature_dicts(
        feature_dicts, close_prices, config=ds_config,
    )
    print(f"  Dataset: Train={len(train_ds):,}, Val={len(val_ds):,}, Test={len(test_ds):,}")

    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"  ⚠ Dataset too small, skipping...")
        return None

    train_dl, val_dl, test_dl = create_dataloaders(train_ds, val_ds, test_ds, config=ds_config)

    # SUPER INSANE model config
    ens_config = build_full_ensemble_config()

    # Compute reasonable epoch limit for remaining time
    # Estimate ~2-5 min per epoch on A100 with 100K bars
    time_per_epoch_estimate = max(n_bars / 100000 * 120, 30)  # seconds
    max_epochs_by_time = int(remaining * 0.85 / time_per_epoch_estimate)
    actual_epochs = min(max_epochs, max_epochs_by_time)
    if actual_epochs < max_epochs:
        print(f"  ⏰ Time-limited: {actual_epochs} epochs (wanted {max_epochs})")

    ckpt_dir = f"models/hydra_{tag}"

    trainer_config = TrainerConfig(
        max_epochs=actual_epochs,
        learning_rate=3e-4,
        use_amp=True,
        ensemble_config=ens_config,
        checkpoint_dir=ckpt_dir,
        save_every_n_epochs=max(1, actual_epochs // 10),
        patience=max(15, actual_epochs // 4),
        warmup_epochs=10,
        gradient_accumulation_steps=4,
        mixup_alpha=0.2,
        swa_start_epoch=max(actual_epochs - 50, actual_epochs + 1),
        label_smoothing=0.1,
    )

    torch.cuda.empty_cache()
    trainer = HydraTrainer(trainer_config, device="cuda")
    param_count = trainer.model.count_parameters()
    print(f"  Model: {param_count:,} parameters")
    print(f"  Training for {actual_epochs} epochs...\n")

    t0 = time.time()
    results = trainer.train(train_dl, val_dl)
    elapsed = time.time() - t0

    print(f"\n  ✅ {tag} COMPLETE in {format_time(elapsed)}")
    print(f"     Epochs: {results['total_epochs']}")
    print(f"     Best Sharpe: {results['best_val_sharpe']:.4f}")
    print(f"     Best Loss:   {results['best_val_loss']:.4f}")
    print(f"     Final Loss:  {results['final_val_loss']:.4f}")

    # Save to Google Drive
    if os.path.exists(ckpt_dir):
        save_to_drive(ckpt_dir, tag)

    return results


# ═══════════════════════════════════════════════════════════════
#  PHASE 7: EXECUTE ALL TRAINING ROUNDS
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 7: STARTING 8-HOUR TRAINING MARATHON")
print(f"  Total time budget: {format_time(MAX_RUNTIME)}")
print(f"  Training rounds: {len(TRAINING_ROUNDS)}")
print(f"  Auto-saving to Google Drive every checkpoint")
print("=" * 70)

all_results = {}
for i, (data_file, epochs, tag, desc) in enumerate(TRAINING_ROUNDS):
    print(f"\n{'▓' * 70}")
    print(f"  ROUND {i+1}/{len(TRAINING_ROUNDS)}")
    print(f"{'▓' * 70}")

    result = run_one_round(data_file, epochs, tag, desc)
    if result:
        all_results[tag] = result

    # Check time
    remaining = time_remaining()
    if remaining < 300:
        print(f"\n  ⏰ TIME LIMIT REACHED — {format_time(time.time() - START_TIME)} elapsed")
        break

# ═══════════════════════════════════════════════════════════════
#  PHASE 8: FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
total_elapsed = time.time() - START_TIME

print(f"\n\n{'═' * 70}")
print(f"  APHELION HYDRA — 8-HOUR TRAINING COMPLETE")
print(f"  Total runtime: {format_time(total_elapsed)}")
print(f"{'═' * 70}\n")

if all_results:
    print(f"  {'Model':<20} {'Epochs':>8} {'Best Sharpe':>14} {'Best Loss':>12}")
    print(f"  {'─'*20} {'─'*8} {'─'*14} {'─'*12}")
    for tag, r in all_results.items():
        print(f"  {tag:<20} {r['total_epochs']:>8} {r['best_val_sharpe']:>14.4f} {r['best_val_loss']:>12.4f}")

print(f"\n  All checkpoints saved to Google Drive:")
print(f"  📁 {DRIVE_SAVE}")
print(f"\n  Checkpoint files per model:")
print(f"     hydra_ensemble_best_sharpe.pt  — Use this for trading")
print(f"     hydra_ensemble_best_loss.pt    — Lowest validation loss")
print(f"     hydra_ensemble_latest.pt       — Last epoch")
print()
print(f"  TO USE: Copy from Google Drive → C:\\Users\\marti\\Aphelion\\models\\hydra\\")
print(f"\n{'═' * 70}")
print(f"  DONE. Go check your Google Drive.")
print(f"{'═' * 70}")
