"""
APHELION — Export MT5 Historical Data for Colab Training

Run this LOCALLY on Windows with MT5 terminal open:
    python scripts/export_mt5_data.py
    python scripts/export_mt5_data.py --bars 50000 --timeframe M5
    python scripts/export_mt5_data.py --output data/bars/xauusd_m1.csv

Outputs a CSV file you can upload to Google Colab for training.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package not installed.")
    print("Run: pip install MetaTrader5")
    print("(Only works on Windows with MT5 terminal installed)")
    sys.exit(1)

import pandas as pd
import numpy as np


TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


def export_bars(
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    n_bars: int = 100_000,
    output: str = "",
) -> Path:
    """
    Fetch historical OHLCV bars from MT5 and save as CSV.
    
    Returns the path to the saved file.
    """
    # 1. Connect
    if not mt5.initialize():
        error = mt5.last_error()
        print(f"MT5 initialize failed: {error}")
        print("Make sure MT5 terminal is open and logged in.")
        sys.exit(1)

    print(f"MT5 connected — build {mt5.version()}")

    # 2. Check symbol
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"Symbol '{symbol}' not found. Available symbols:")
        symbols = mt5.symbols_get()
        gold_syms = [s.name for s in symbols if "XAU" in s.name or "GOLD" in s.name]
        print(f"  Gold-related: {gold_syms[:10]}")
        mt5.shutdown()
        sys.exit(1)

    if not info.visible:
        mt5.symbol_select(symbol, True)

    # 3. Fetch bars
    tf = TIMEFRAME_MAP.get(timeframe.upper())
    if tf is None:
        print(f"Unknown timeframe '{timeframe}'. Use one of: {list(TIMEFRAME_MAP.keys())}")
        mt5.shutdown()
        sys.exit(1)

    print(f"Fetching {n_bars:,} {timeframe} bars for {symbol}...")
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)

    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        print(f"No data returned: {error}")
        mt5.shutdown()
        sys.exit(1)

    # 4. Convert to DataFrame
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={
        "time": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "tick_volume": "volume",
        "spread": "spread",
        "real_volume": "real_volume",
    })

    # Keep useful columns
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    if "spread" in df.columns:
        cols.append("spread")
    if "real_volume" in df.columns and df["real_volume"].sum() > 0:
        cols.append("real_volume")
    df = df[cols]

    # 5. Save
    if not output:
        output = f"data/bars/{symbol.lower()}_{timeframe.lower()}.csv"

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    mt5.shutdown()

    print(f"\nExported {len(df):,} bars to {out_path}")
    print(f"  Date range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    print(f"  Price range: {df['close'].min():.2f} → {df['close'].max():.2f}")
    print(f"  File size: {out_path.stat().st_size / 1024:.0f} KB")
    print(f"\nUpload this file to Colab and use it with train_hydra.py")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export MT5 historical data for training")
    parser.add_argument("--symbol", default="XAUUSD", help="Trading symbol (default: XAUUSD)")
    parser.add_argument("--timeframe", default="M1", help="Timeframe: M1, M5, M15, M30, H1, H4, D1")
    parser.add_argument("--bars", type=int, default=100_000, help="Number of bars to fetch (default: 100000)")
    parser.add_argument("--output", default="", help="Output CSV path (auto-generated if empty)")
    args = parser.parse_args()

    export_bars(
        symbol=args.symbol,
        timeframe=args.timeframe,
        n_bars=args.bars,
        output=args.output,
    )


if __name__ == "__main__":
    main()
