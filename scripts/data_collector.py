"""
APHELION — Comprehensive MT5 Data Collector
Opens MT5 on your laptop and fetches EVERYTHING the system needs:

Symbols:
  - XAUUSD  (primary)
  - XAGUSD  (cointegration, cross-impact)
  - DXY / USDX  (macro correlation, cross-impact)
  - USTBOND / TLT  (cross-impact bond proxy)
  - USOIL / WTI / XTIUSD  (cross-impact oil)
  - EURUSD  (DXY proxy if DXY unavailable)

Timeframes:
  - M1, M5, M15, M30, H1, H4, D1, W1

Usage:
    python scripts/data_collector.py                    # Fetch everything (100K M1 bars default)
    python scripts/data_collector.py --bars 200000      # More history
    python scripts/data_collector.py --output-dir data/bars  # Custom output directory
    python scripts/data_collector.py --dry-run           # Show what would be fetched
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    print("=" * 60)
    print("ERROR: MetaTrader5 package not installed.")
    print("Run:   pip install MetaTrader5")
    print("(Windows only — MT5 terminal must be installed & open)")
    print("=" * 60)
    sys.exit(1)

import numpy as np
import pandas as pd


# ─── Configuration: Everything the system needs ─────────────────────────────

# MT5 timeframe mapping
TIMEFRAMES = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
}

# How many bars to fetch per timeframe (scaled — lower TFs get more bars)
DEFAULT_BAR_COUNTS = {
    "M1":  100_000,   # ~70 trading days
    "M5":  100_000,   # ~350 trading days
    "M15": 50_000,    # ~520 trading days
    "M30": 50_000,    # ~1040 trading days
    "H1":  30_000,    # ~1250 trading days
    "H4":  10_000,    # ~1660 trading days
    "D1":  5_000,     # ~20 years
    "W1":  2_000,     # ~38 years
}

# Symbols the system needs, with broker-specific alternatives
SYMBOL_GROUPS = {
    "XAUUSD": {
        "role": "PRIMARY — Gold vs USD (main trading instrument)",
        "alternatives": ["GOLD", "XAUUSDm", "XAUUSD.a", "XAUUSD_i", "Gold"],
        "required": True,
    },
    "XAGUSD": {
        "role": "COINTEGRATION + CROSS-IMPACT — Silver (gold/silver ratio)",
        "alternatives": ["SILVER", "XAGUSDm", "XAGUSD.a", "XAGUSD_i", "Silver"],
        "required": True,
    },
    "USDX": {
        "role": "MACRO — US Dollar Index (DXY correlation)",
        "alternatives": ["DXY", "DX", "DXY.f", "USDindex", "DOLLAR_INDX", "DX-SEP24", "DX-DEC24"],
        "required": False,  # EURUSD can proxy
    },
    "EURUSD": {
        "role": "MACRO PROXY — Inverse DXY proxy if DXY unavailable",
        "alternatives": ["EURUSDm", "EURUSD.a", "EURUSD_i"],
        "required": False,
    },
    "XTIUSD": {
        "role": "CROSS-IMPACT — Crude Oil",
        "alternatives": ["USOIL", "WTI", "CL", "USOUSD", "CLm", "USOIL.a", "WTI_OIL", "OIL"],
        "required": False,
    },
    "USTBOND": {
        "role": "CROSS-IMPACT — US Bonds (TLT proxy for rate sensitivity)",
        "alternatives": ["TLT", "US10Y", "US30Y", "T-NOTE", "TNOTE", "USBond", "US10Y_Bond"],
        "required": False,
    },
    "SPX500": {
        "role": "CROSS-IMPACT — S&P 500 (risk-on/risk-off regime)",
        "alternatives": ["US500", "SPX", "SP500", "SP500m", "US500.a", "S&P500"],
        "required": False,
    },
}


# ─── Helper functions ────────────────────────────────────────────────────────

def resolve_symbol(desired: str, alternatives: list[str]) -> Optional[str]:
    """
    Try to find a matching symbol in the MT5 terminal.
    Checks the desired name first, then each alternative.
    Returns the first match found, or None.
    """
    # Try exact match first
    for candidate in [desired] + alternatives:
        info = mt5.symbol_info(candidate)
        if info is not None:
            # Make sure it's visible / selectable
            if not info.visible:
                mt5.symbol_select(candidate, True)
            return candidate

    # Try partial match on all available symbols
    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        return None

    desired_lower = desired.lower()
    for sym in all_symbols:
        name_lower = sym.name.lower()
        if desired_lower in name_lower or name_lower in desired_lower:
            if not sym.visible:
                mt5.symbol_select(sym.name, True)
            return sym.name

    return None


def fetch_bars(symbol: str, tf_name: str, tf_mt5: int, count: int) -> Optional[pd.DataFrame]:
    """
    Fetch bars from MT5 and return as DataFrame.
    Automatically chunks large requests and retries with smaller counts
    since many brokers cap how many bars they return in a single call.
    """
    CHUNK_SIZE = 10_000  # Small safe chunks — works on all brokers

    all_frames: list[pd.DataFrame] = []

    if count <= CHUNK_SIZE:
        # Single request — try as-is, then fallback
        rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, count)
        if rates is None or len(rates) == 0:
            # Retry with progressively smaller counts
            for fallback in [50_000, 30_000, 20_000, 10_000, 5_000]:
                if fallback >= count:
                    continue
                rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, fallback)
                if rates is not None and len(rates) > 0:
                    break
        if rates is not None and len(rates) > 0:
            all_frames.append(pd.DataFrame(rates))
    else:
        # Chunk: fetch from newest to oldest using copy_rates_from_pos offset
        pos = 0
        remaining = count
        while remaining > 0:
            chunk = min(CHUNK_SIZE, remaining)
            rates = mt5.copy_rates_from_pos(symbol, tf_mt5, pos, chunk)
            if rates is None or len(rates) == 0:
                # Try smaller chunk
                for fallback in [20_000, 10_000, 5_000]:
                    if fallback >= chunk:
                        continue
                    rates = mt5.copy_rates_from_pos(symbol, tf_mt5, pos, fallback)
                    if rates is not None and len(rates) > 0:
                        break
                if rates is None or len(rates) == 0:
                    break  # No more data available at this offset
            got = len(rates)
            all_frames.append(pd.DataFrame(rates))
            pos += got
            remaining -= got
            if got < chunk:
                break  # Reached end of available history

    if not all_frames:
        return None

    # Combine all chunks (they come newest-first per chunk, so sort by time)
    raw = pd.concat(all_frames, ignore_index=True)
    raw = raw.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    if len(raw) == 0:
        return None

    df = raw.copy()
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

    # Keep relevant columns
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    if "spread" in df.columns:
        cols.append("spread")
    if "real_volume" in df.columns and df["real_volume"].sum() > 0:
        cols.append("real_volume")

    return df[cols]


def fetch_ticks(symbol: str, count: int = 500_000) -> Optional[pd.DataFrame]:
    """Fetch recent ticks for microstructure features."""
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    # Fetch ticks from last 24h (or as many as available)
    ticks = mt5.copy_ticks_from(symbol, now - timedelta(days=7), count, mt5.COPY_TICKS_ALL)

    if ticks is None or len(ticks) == 0:
        return None

    df = pd.DataFrame(ticks)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={
        "time": "timestamp",
        "bid": "bid",
        "ask": "ask",
        "last": "last",
        "volume": "volume",
        "flags": "flags",
    })

    cols_keep = ["timestamp", "bid", "ask"]
    if "last" in df.columns:
        cols_keep.append("last")
    if "volume" in df.columns:
        cols_keep.append("volume")

    return df[cols_keep]


def print_banner():
    print()
    print("=" * 70)
    print("    APHELION DATA COLLECTOR")
    print("    Fetching all data the system needs from MT5")
    print("=" * 70)
    print()


def print_summary(results: dict, output_dir: str, elapsed: float):
    print()
    print("=" * 70)
    print("    COLLECTION COMPLETE")
    print("=" * 70)
    print()

    total_bars = 0
    total_files = 0
    total_size = 0

    for symbol, tf_results in results.items():
        print(f"  {symbol}:")
        for tf_name, info in tf_results.items():
            if info["status"] == "ok":
                total_bars += info["bars"]
                total_files += 1
                total_size += info["size_kb"]
                print(f"    {tf_name:>4}: {info['bars']:>8,} bars  "
                      f"({info['date_from']} → {info['date_to']})  "
                      f"[{info['size_kb']:.0f} KB]")
            else:
                print(f"    {tf_name:>4}: SKIPPED — {info['reason']}")
        print()

    print(f"  Total: {total_bars:,} bars across {total_files} files")
    print(f"  Disk:  {total_size / 1024:.1f} MB")
    print(f"  Time:  {elapsed:.1f} seconds")
    print(f"  Dir:   {output_dir}")
    print()
    print("  Next steps:")
    print(f"    1. Upload the '{output_dir}' folder to Google Colab")
    print("    2. Train: python scripts/train_hydra.py \\")
    print(f"         --data {output_dir}/xauusd_m1.csv --full --epochs 50")
    print()


# ─── Main collector ──────────────────────────────────────────────────────────

def collect_all(
    output_dir: str = "data/bars",
    bar_multiplier: float = 1.0,
    fetch_tick_data: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Connect to MT5 and fetch all required data.

    Args:
        output_dir: Directory to save CSV files
        bar_multiplier: Scale default bar counts (e.g. 2.0 = double history)
        fetch_tick_data: Whether to also fetch raw tick data
        dry_run: If True, just print what would be fetched

    Returns:
        Dict of {symbol: {timeframe: result_info}}
    """
    print_banner()

    # ── 1. Connect to MT5 ──
    print("Connecting to MT5...")
    if not mt5.initialize():
        error = mt5.last_error()
        print(f"\n  FAILED: {error}")
        print("\n  Checklist:")
        print("    - Is MetaTrader 5 terminal open?")
        print("    - Are you logged into your account?")
        print("    - Is the terminal responding (not frozen)?")
        print("    - Try closing & reopening MT5, then run this again")
        sys.exit(1)

    terminal_info = mt5.terminal_info()
    account_info = mt5.account_info()

    print(f"  Connected to MT5 build {mt5.version()}")
    if terminal_info:
        print(f"  Terminal: {terminal_info.name}")
    if account_info:
        print(f"  Account:  {account_info.login} ({account_info.server})")
        print(f"  Balance:  {account_info.balance:.2f} {account_info.currency}")
    print()

    # ── 2. Resolve symbols ──
    print("Resolving symbols...")
    resolved_symbols: dict[str, str] = {}  # our_name → broker_name
    missing_required = []

    for our_name, spec in SYMBOL_GROUPS.items():
        broker_name = resolve_symbol(our_name, spec["alternatives"])
        if broker_name:
            resolved_symbols[our_name] = broker_name
            tag = "✓" if broker_name == our_name else f"✓ (as {broker_name})"
            print(f"  {our_name:<12} {tag:<25} — {spec['role']}")
        else:
            if spec["required"]:
                missing_required.append(our_name)
                print(f"  {our_name:<12} ✗ NOT FOUND (REQUIRED!)    — {spec['role']}")
            else:
                print(f"  {our_name:<12} ✗ not found (optional)     — {spec['role']}")

    print()

    if missing_required:
        print(f"  WARNING: Required symbols not found: {missing_required}")
        print("  Your broker may use different names. Check MT5 Market Watch.")
        print("  The collector will continue with what's available.\n")

    if not resolved_symbols:
        print("  ERROR: No symbols found at all. Check your MT5 terminal.")
        mt5.shutdown()
        sys.exit(1)

    # ── 3. Dry run check ──
    if dry_run:
        print("DRY RUN — would fetch:")
        for our_name, broker_name in resolved_symbols.items():
            print(f"\n  {our_name} (broker: {broker_name}):")
            for tf_name, count in DEFAULT_BAR_COUNTS.items():
                adj_count = int(count * bar_multiplier)
                print(f"    {tf_name}: {adj_count:,} bars")
        if fetch_tick_data and "XAUUSD" in resolved_symbols:
            print(f"\n  XAUUSD ticks: up to 500,000")
        mt5.shutdown()
        return {}

    # ── 4. Fetch everything ──
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    total_tasks = len(resolved_symbols) * len(TIMEFRAMES)
    done = 0

    for our_name, broker_name in resolved_symbols.items():
        results[our_name] = {}
        symbol_dir = out  # Flat structure: symbol_timeframe.csv

        for tf_name, tf_mt5 in TIMEFRAMES.items():
            done += 1
            base_count = DEFAULT_BAR_COUNTS.get(tf_name, 10_000)
            count = int(base_count * bar_multiplier)

            progress = f"[{done}/{total_tasks}]"
            print(f"  {progress} {our_name} {tf_name} — fetching {count:,} bars...", end="", flush=True)

            try:
                df = fetch_bars(broker_name, tf_name, tf_mt5, count)
            except Exception as e:
                print(f" ERROR: {e}")
                results[our_name][tf_name] = {"status": "error", "reason": str(e)}
                continue

            if df is None or len(df) == 0:
                print(f" no data available")
                results[our_name][tf_name] = {"status": "skip", "reason": "no data from MT5"}
                continue

            # Save CSV
            filename = f"{our_name.lower()}_{tf_name.lower()}.csv"
            filepath = symbol_dir / filename
            df.to_csv(filepath, index=False)

            size_kb = filepath.stat().st_size / 1024
            date_from = str(df["timestamp"].iloc[0])[:10]
            date_to = str(df["timestamp"].iloc[-1])[:10]

            print(f" {len(df):,} bars ({date_from} → {date_to}) [{size_kb:.0f} KB]")
            results[our_name][tf_name] = {
                "status": "ok",
                "bars": len(df),
                "date_from": date_from,
                "date_to": date_to,
                "size_kb": size_kb,
                "file": str(filepath),
            }

    # ── 5. Fetch tick data for XAUUSD (microstructure features) ──
    if fetch_tick_data and "XAUUSD" in resolved_symbols:
        print(f"\n  Fetching XAUUSD tick data for microstructure features...", end="", flush=True)
        try:
            tick_df = fetch_ticks(resolved_symbols["XAUUSD"], count=500_000)
            if tick_df is not None and len(tick_df) > 0:
                tick_path = out / "xauusd_ticks.csv"
                tick_df.to_csv(tick_path, index=False)
                size_kb = tick_path.stat().st_size / 1024
                print(f" {len(tick_df):,} ticks [{size_kb:.0f} KB]")
                results.setdefault("XAUUSD_TICKS", {})["ticks"] = {
                    "status": "ok",
                    "bars": len(tick_df),
                    "date_from": str(tick_df["timestamp"].iloc[0])[:10],
                    "date_to": str(tick_df["timestamp"].iloc[-1])[:10],
                    "size_kb": size_kb,
                    "file": str(tick_path),
                }
            else:
                print(" no tick data available")
        except Exception as e:
            print(f" ERROR: {e}")

    # ── 6. Save metadata ──
    meta = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "mt5_build": str(mt5.version()),
        "account": account_info.login if account_info else "?",
        "server": account_info.server if account_info else "?",
        "symbols_resolved": {k: v for k, v in resolved_symbols.items()},
        "bar_multiplier": bar_multiplier,
    }

    import json
    meta_path = out / "_collection_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    mt5.shutdown()
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="APHELION Data Collector — Fetch all required data from MT5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/data_collector.py                      # Fetch everything
  python scripts/data_collector.py --bars 200000        # 2x default history
  python scripts/data_collector.py --dry-run            # Preview what'll be fetched
  python scripts/data_collector.py --no-ticks           # Skip tick data (faster)
  python scripts/data_collector.py --output-dir my_data # Custom output dir
        """,
    )
    parser.add_argument(
        "--bars", type=int, default=100_000,
        help="Base M1 bar count (other TFs scale automatically). Default: 100000",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/bars",
        help="Output directory for CSV files (default: data/bars)",
    )
    parser.add_argument(
        "--no-ticks", action="store_true",
        help="Skip tick data collection (faster, but no microstructure features)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be fetched without actually fetching",
    )
    args = parser.parse_args()

    # bar_multiplier scales all timeframes proportionally
    bar_multiplier = args.bars / 100_000

    t0 = time.time()

    results = collect_all(
        output_dir=args.output_dir,
        bar_multiplier=bar_multiplier,
        fetch_tick_data=not args.no_ticks,
        dry_run=args.dry_run,
    )

    elapsed = time.time() - t0

    if results:
        print_summary(results, args.output_dir, elapsed)


if __name__ == "__main__":
    main()
