# Aphelion Data Reorganization — Symbol-Specific Directory Structure

## Overview

The Aphelion data pipeline has been reorganized to support multiple symbols with isolated directory structures. Each symbol now has its own data directories, making the project more scalable and maintainable.

## New Directory Structure

```
data/
├── raw/                          # Shared metadata (applies to all symbols)
│   ├── account_info.json         ✓ Global account info
│   └── all_symbols.csv           ✓ All broker symbols list
│
├── raw/XAUUSD/                   # XAUUSD symbol data (primary instrument)
│   ├── xauusd_m1.csv                    ✓ M1 OHLCV bars
│   ├── xauusd_m5.csv                    ✓ M5 OHLCV bars
│   ├── xauusd_m15.csv                   ✓ M15 OHLCV bars
│   ├── ... (all 21 timeframes)
│   ├── xauusd_symbol_info.json          ✓ Symbol metadata
│   ├── xauusd_live_tick.json            ✓ Current live tick info
│   ├── xauusd_ticks.csv                 ✓ Tick history (bid/ask/volume)
│   └── xauusd_m1_enriched.csv           ✓ M1 bars enriched from ticks
│
├── raw/EURUSD/                   # Context symbol data (EURUSD - 1 of 27)
│   ├── eurusd_m1.csv
│   ├── eurusd_m5.csv
│   └── ... (all 21 timeframes)
│
├── raw/{SYMBOL}/                 # Other context symbols (GBPUSD, USDCHF, etc.)
│   └── ...
│
├── processed/XAUUSD/             # Feature dataset & training splits
│   ├── xauusd_hydra.parquet              ✓ Full feature dataset
│   ├── feature_columns.txt               ✓ List of all 700+ features
│   ├── train.npz                         ✓ Training split
│   ├── val.npz                           ✓ Validation split
│   ├── test.npz                          ✓ Test split
│   ├── scaler.json                       ✓ Feature normalization params
│   └── dataset_meta.json                 ✓ Dataset metadata & splits
│
└── processed/{SYMBOL}/           # Other symbols (if multi-symbol training)
    └── ... (same structure)
```

## What Changed

### Removed
- ❌ No longer store all symbols' data in flat `data/raw/` directory
- ❌ No longer store features in flat `data/processed/` directory
- ❌ No hardcoded paths like `data/processed/xauusd_hydra.parquet`

### Added
- ✅ Symbol-aware directory helper: `get_data_dirs(symbol)`
- ✅ Each symbol's data isolated in its own subdirectories
- ✅ Seamless support for multi-symbol workflows
- ✅ Clearer data organization for scalability

## Updated Functions

### aphelion_data.py

**New Helper:**
```python
get_data_dirs(symbol: str) -> tuple[Path, Path]
    # Returns both raw and processed directories for a symbol
    # Creates directories automatically if they don't exist
```

**Updated Phase Functions:**
- `fetch_phase(mt5, symbol, ...)` — Now symbol-specific
- `build_features_phase(symbol, ...)` — Reads/writes symbol-specific paths
- `prepare_dataset_phase(symbol, ...)` — Symbol-specific dataset preparation
- `load_tf(symbol, tf)` — Loads symbol-specific timeframe data
- `_fetch_ticks(mt5, symbol, ..., symbol_raw)` — Symbol-specific tick storage
- `_build_enriched_m1(symbol, ..., symbol_raw)` — Symbol-specific enrichment

### runall.py

Updated to support symbol parameter:
```bash
python runall.py --symbol XAUUSD        # (default)
python runall.py --symbol EURUSD        # or any other symbol
```

The training subprocess automatically finds the correct symbol-specific data.

### scripts/train_hydra.py

Updated documentation in docstring to reflect new paths:
```bash
# Old paths (no longer used)
# python scripts/train_hydra.py --data data/processed/xauusd_hydra.parquet

# New paths (symbol-specific)
python scripts/train_hydra.py --data data/processed/XAUUSD/xauusd_hydra.parquet
python scripts/train_hydra.py --data data/processed/XAUUSD --full --epochs 50
```

## Usage Examples

### Fetch Data for a Single Symbol (Default: XAUUSD)
```bash
python aphelion_data.py --fetch-only
```
Output: Files in `data/raw/XAUUSD/` and context symbols in `data/raw/{SYMBOL}/`

### Fetch Data for a Different Symbol
```bash
python aphelion_data.py --fetch-only --symbol EURUSD
```
Output: Files in `data/raw/EURUSD/`

### Complete Pipeline for a Symbol
```bash
python aphelion_data.py --symbol XAUUSD
```
Runs all phases:
1. Fetch raw data → `data/raw/XAUUSD/`
2. Build features → `data/processed/XAUUSD/xauusd_hydra.parquet`
3. Prepare dataset → `data/processed/XAUUSD/{train,val,test}.npz`
4. Train HYDRA → `models/hydra/`

### Resume Download (Ticks Only)
```bash
python aphelion_data.py --resume --no-ticks   # Skip tick fetch
python aphelion_data.py --resume --fetch-only  # Resume from last successful tick date
```

## Migration from Old Structure

If you have data in the old flat structure, the recommendation is:

1. **Automatic cleanup:** Use `--from-scratch` to clear old data for a symbol
   ```bash
   python aphelion_data.py --from-scratch --symbol XAUUSD
   ```

2. **Manual migration:** Move files to symbol-specific directories
   ```bash
   # From Windows PowerShell or bash
   mkdir -p data/raw/XAUUSD data/processed/XAUUSD
   mv data/raw/xauusd_*.csv data/raw/XAUUSD/
   mv data/processed/xauusd_* data/processed/XAUUSD/
   ```

## Multi-Symbol Support

The new structure enables seamless multi-symbol workflows:

```python
# Fetch different symbols independently
for symbol in ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]:
    python aphelion_data.py --fetch-only --symbol {symbol}

# Each creates isolated directory structure:
# data/raw/{SYMBOL}/
# data/processed/{SYMBOL}/
```

## Key Benefits

1. **Scalability** — Add new symbols without conflicts
2. **Isolation** — Each symbol's data is self-contained
3. **Clarity** — Easy to see what data exists for which symbol
4. **Maintainability** — Symbol-specific directories = symbol-specific logic
5. **Resume Logic** — Works correctly per-symbol without interference
6. **Backup/Archive** — Easy to backup individual symbols

## Shared Files (Not Symbol-Specific)

Some files remain shared in `data/raw/`:
- `account_info.json` — Trading account metadata (same for all symbols)
- `all_symbols.csv` — List of all available broker symbols (same for all)

These apply globally and are fetched once per data pipeline run.

## Context Symbols

When fetching XAUUSD, the system automatically fetches 27 context symbols for feature engineering:

```
EURUSD, GBPUSD, USDCHF, USDJPY, USDCAD, AUDUSD, 
NZDUSD, CADJPY, CHFJPY, GBPJPY, EURJPY, EURGBP,
DXY (USDX), XAGUSD, XPTUSD, SPX500, US500, US30, 
GER40, NAS100, BTCUSD, ETHUSD, USOIL, BRENT
```

Each context symbol gets its own directory: `data/raw/{SYMBOL}/`

## FAQ

**Q: Do I need to re-download all data?**
A: Not necessarily. Use `--resume` to continue from your last successful point. Or use `--from-scratch` to start fresh.

**Q: Can I train on multiple symbols?**
A: Yes! The structure now supports multi-symbol workflows. Fetch each symbol separately, then the training script can be extended to use multiple parquet files.

**Q: What if I only care about XAUUSD?**
A: Nothing changes for you operationally. Just run `python aphelion_data.py` and data goes to `data/raw/XAUUSD/` and `data/processed/XAUUSD/`.

**Q: Are old flat-structure paths still supported?**
A: Only as fallback. The primary logic uses symbol-specific paths. Legacy code paths are maintained for gradual migration.

## Testing the Reorganization

```bash
# Test syntax
python test_syntax.py

# Test fetch phase only
python aphelion_data.py --fetch-only --symbol XAUUSD

# Verify directory structure
ls -la data/raw/XAUUSD/
ls -la data/processed/XAUUSD/

# Run full pipeline
python aphelion_data.py --symbol XAUUSD
```

## Related Files Updated

- ✅ `aphelion_data.py` — Main data pipeline (8 functions updated)
- ✅ `runall.py` — End-to-end demo (symbol parameter propagated)
- ✅ `scripts/train_hydra.py` — Documentation updated
- ✅ `test_syntax.py` — New syntax validation script

## Future Enhancements

Potential improvements enabled by this reorganization:

1. Parallel multi-symbol fetching
2. Cross-symbol feature engineering with isolated dependencies
3. Per-symbol model ensembles
4. Symbol-specific hyperparameter tuning
5. Easy data lifecycle management (archive old symbols, etc.)
