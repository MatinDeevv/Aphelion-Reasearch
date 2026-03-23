#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║        ▄████████  ██████████  ██        ██████████ ██      ██                ║
║       ██░░░░░░██  ██░░░░░░██  ██       ██░░░░░░██ ████    ██                ║
║      ██      ██  ████████    ██      ██      ██ ██ ████  ██                ║
║     ████████    ██          ██     ████████   ██   ████ ██                ║
║    ██░░░░░░██  ██░░░░░░██  ██    ██░░░░░░██  ██    ██████                ║
║   ██      ██  ██      ██  ██   ██      ██  ██      ████                  ║
║  ██████████  ██████████  ████ ██████████  ██       ██                    ║
║                                                                               ║
║            DATA FORGE  —  FETCH → BUILD → TRAIN                              ║
║                    Autonomous XAU/USD Intelligence                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝

  One script. Fetches every byte of market data MT5 has.
  Builds 700+ features. Trains HYDRA. No terminal commands ever again.

  Usage:
    python aphelion_data.py                  # full pipeline
    python aphelion_data.py --fetch-only     # just download data
    python aphelion_data.py --features-only  # just build features from existing data
    python aphelion_data.py --train-only     # just train HYDRA
    python aphelion_data.py --resume         # skip already-downloaded files
    python aphelion_data.py --no-ticks       # skip tick data (faster)
    python aphelion_data.py --no-train       # fetch + build but don't train
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import threading
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL COLORS & STYLE
# ─────────────────────────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    # Colors
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    ORANGE  = "\033[38;5;214m"
    GOLD    = "\033[38;5;220m"
    PURPLE  = "\033[38;5;135m"
    # Backgrounds
    BG_BLACK  = "\033[40m"
    BG_BLUE   = "\033[44m"

def supports_color():
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

USE_COLOR = supports_color()

def c(color, text):
    return f"{color}{text}{C.RESET}" if USE_COLOR else text

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

LOG_FILE = Path("logs/aphelion_data.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
_log_fh  = open(LOG_FILE, "a", encoding="utf-8")
_start   = time.time()

def _elapsed():
    s = int(time.time() - _start)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

def log(msg: str, level: str = "INFO") -> None:
    ts    = datetime.now().strftime("%H:%M:%S")
    el    = _elapsed()
    plain = f"[{ts}][{el}][{level}] {msg}"
    _log_fh.write(plain + "\n"); _log_fh.flush()

    if level == "INFO":
        prefix = c(C.CYAN,    f"[{ts}]") + c(C.DIM, f"[{el}]") + c(C.GREEN,  " ✓ ")
    elif level == "WARN":
        prefix = c(C.YELLOW,  f"[{ts}]") + c(C.DIM, f"[{el}]") + c(C.YELLOW, " ⚠ ")
    elif level == "ERROR":
        prefix = c(C.RED,     f"[{ts}]") + c(C.DIM, f"[{el}]") + c(C.RED,    " ✗ ")
    elif level == "SAVE":
        prefix = c(C.GREEN,   f"[{ts}]") + c(C.DIM, f"[{el}]") + c(C.GOLD,   " 💾 ")
    elif level == "DATA":
        prefix = c(C.MAGENTA, f"[{ts}]") + c(C.DIM, f"[{el}]") + c(C.MAGENTA," 📊 ")
    elif level == "TICK":
        prefix = c(C.BLUE,    f"[{ts}]") + c(C.DIM, f"[{el}]") + c(C.BLUE,   " ⚡ ")
    elif level == "FEAT":
        prefix = c(C.ORANGE,  f"[{ts}]") + c(C.DIM, f"[{el}]") + c(C.ORANGE, " 🔬 ")
    elif level == "TRAIN":
        prefix = c(C.PURPLE,  f"[{ts}]") + c(C.DIM, f"[{el}]") + c(C.PURPLE, " 🧠 ")
    elif level == "DONE":
        prefix = c(C.GOLD,    f"[{ts}]") + c(C.DIM, f"[{el}]") + c(C.GOLD,   " ★ ")
    else:
        prefix = f"[{ts}][{el}]  "
    print(prefix + msg, flush=True)

def section(title: str, emoji: str = "━") -> None:
    w    = 70
    line = emoji * w
    print(flush=True)
    print(c(C.GOLD + C.BOLD, f"  {line[:w]}"), flush=True)
    print(c(C.GOLD + C.BOLD, f"  {emoji*2}  {title}"), flush=True)
    print(c(C.GOLD + C.BOLD, f"  {line[:w]}"), flush=True)
    print(flush=True)
    _log_fh.write(f"\n=== {title} ===\n"); _log_fh.flush()

def banner():
    print(c(C.GOLD + C.BOLD, """
╔═══════════════════════════════════════════════════════════════════════════╗
║  ▄▀█ █▀█ █░█ █▀▀ █░░ █ █▀█ █▄░█   █▀▄ ▄▀█ ▀█▀ ▄▀█   █▀▀ █▀█ █▀█ █▀▀  ║
║  █▀█ █▀▀ █▀█ ██▄ █▄▄ █ █▄█ █░▀█   █▄▀ █▀█ ░█░ █▀█   █▀░ █▄█ █▀▄ █▄█  ║
╠═══════════════════════════════════════════════════════════════════════════╣
║              FETCH  →  BUILD  →  TRAIN  →  PROFIT                       ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""))

def progress_bar(done: int, total: int, prefix: str = "", suffix: str = "", width: int = 40) -> str:
    pct    = min(1.0, done / max(total, 1))
    filled = int(width * pct)
    bar    = c(C.GREEN,  "█" * filled) + c(C.DIM, "░" * (width - filled))
    pct_s  = c(C.BOLD,  f"{pct*100:5.1f}%")
    return f"  [{bar}] {pct_s}  {c(C.CYAN, prefix)}  {c(C.DIM, suffix)}"

def save_msg(path: Path, rows: int, mb: float) -> None:
    log(f"{c(C.WHITE+C.BOLD, path.name):<55} {c(C.GREEN, f'{rows:>10,} rows')}  "
        f"{c(C.YELLOW, f'{mb:>7.1f} MB')}", "SAVE")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PRIMARY    = "XAUUSD"
RAW_DIR    = Path("data/raw")
PROC_DIR   = Path("data/processed")
MODELS_DIR = Path("models/hydra")
TICK_DAYS  = 1825  # 5 years

TIMEFRAMES_DEF = None  # filled after MT5 import

CONTEXT_H1 = [
    "EURUSD","GBPUSD","USDCHF","USDJPY","USDCAD","AUDUSD",
    "DXY","USDX","XAGUSD","XPTUSD","XPDUSD",
    "US500","SPX500","SP500","US30","DJ30",
    "NAS100","NASDAQ","GER40","DAX40",
    "USOIL","WTIUSD","CRUDEOIL","NATGAS","COPPER",
    "US10Y","USTBOND","BTCUSD",
]

TF_STEP = {
    "M1":7,"M2":14,"M3":21,"M4":21,"M5":30,"M6":30,
    "M10":60,"M12":60,"M15":60,"M20":60,"M30":90,
    "H1":180,"H2":365,"H3":365,"H4":365,"H6":730,
    "H8":730,"H12":730,"D1":3650,"W1":3650,"MN1":3650,
}

# ─────────────────────────────────────────────────────────────────────────────
# STATS TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class Stats:
    def __init__(self):
        self.files_saved    = 0
        self.total_rows     = 0
        self.total_mb       = 0.0
        self.ticks_fetched  = 0
        self.features_built = 0
        self.start          = time.time()

    def add(self, rows: int, mb: float):
        self.files_saved += 1
        self.total_rows  += rows
        self.total_mb    += mb

    def print_summary(self):
        elapsed = time.time() - self.start
        print()
        print(c(C.GOLD + C.BOLD, "  ┌─────────────────────────────────────────────┐"))
        print(c(C.GOLD + C.BOLD, "  │          SESSION SUMMARY                    │"))
        print(c(C.GOLD + C.BOLD, "  ├─────────────────────────────────────────────┤"))
        print(c(C.GOLD,          f"  │  Files saved:    {self.files_saved:>8,}                    │"))
        print(c(C.GOLD,          f"  │  Total rows:     {self.total_rows:>8,}                    │"))
        print(c(C.GOLD,          f"  │  Total data:     {self.total_mb:>7.0f} MB                  │"))
        print(c(C.GOLD,          f"  │  Ticks fetched:  {self.ticks_fetched:>8,}                    │"))
        print(c(C.GOLD,          f"  │  Features built: {self.features_built:>8,}                    │"))
        print(c(C.GOLD,          f"  │  Time elapsed:   {elapsed/60:>7.1f} min                  │"))
        print(c(C.GOLD + C.BOLD, "  └─────────────────────────────────────────────┘"))
        print()

STATS = Stats()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    mb = path.stat().st_size / 1e6
    save_msg(path, len(df), mb)
    STATS.add(len(df), mb)

def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log(f"Saved → {c(C.WHITE, path.name)}")

def skip(path: Path, resume: bool) -> bool:
    if resume and path.exists() and path.stat().st_size > 1000:
        log(f"Skip {c(C.DIM, path.name)} (already downloaded)", "WARN")
        return True
    return False

def ema(arr: np.ndarray, p: int) -> np.ndarray:
    a = 2.0 / (p + 1); out = np.empty_like(arr, dtype=np.float64); out[0] = arr[0]
    for i in range(1, len(arr)): out[i] = a * arr[i] + (1 - a) * out[i-1]
    return out

def rolling_mean(arr: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    cs  = np.cumsum(np.nan_to_num(arr))
    out[p-1:] = (cs[p-1:] - np.concatenate([[0], cs[:-p]])) / p
    return out

def rolling_std(arr: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(p-1, len(arr)): out[i] = np.std(arr[i-p+1:i+1], ddof=0)
    return out

def load_tf(symbol: str, tf: str) -> pd.DataFrame | None:
    p = RAW_DIR / f"{symbol.lower()}_{tf.lower()}.csv"
    if not p.exists(): return None
    df = pd.read_csv(p, parse_dates=["time"]).set_index("time").sort_index()
    if not df.index.tz: df.index = df.index.tz_localize("UTC")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████████████████████████████████████████
# PHASE 1 — FETCH EVERYTHING FROM MT5
# ██████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

def init_mt5():
    try:
        import MetaTrader5 as mt5
        return mt5
    except ImportError:
        log("MetaTrader5 not installed. Run: pip install MetaTrader5", "ERROR")
        sys.exit(1)

def connect_mt5(mt5):
    section("CONNECTING TO MT5", "⚡")
    if not mt5.initialize():
        log(f"MT5 init failed: {mt5.last_error()}", "ERROR")
        log("Open MetaTrader5, log in, then run this script again.", "ERROR")
        sys.exit(1)
    acc = mt5.account_info()
    print(c(C.GOLD + C.BOLD, f"""
  ┌───────────────────────────────────────────┐
  │  MT5 CONNECTED                            │
  │  Login:    {acc.login:<32} │
  │  Server:   {acc.server:<32} │
  │  Balance:  {str(acc.balance)+' '+acc.currency:<32} │
  │  Leverage: 1:{str(acc.leverage):<30} │
  │  Broker:   {acc.company:<32} │
  └───────────────────────────────────────────┘
"""))
    return acc

def probe_earliest(mt5, symbol, tf_const):
    # Unused - fetch_bars now walks from 2000 directly
    return datetime(2000, 1, 1, tzinfo=timezone.utc)

def fetch_bars(mt5, symbol, tf_const, tf_name, from_date):
    mt5.symbol_select(symbol, True)
    mt5.copy_rates_from_pos(symbol, tf_const, 0, 200)
    time.sleep(0.3)
    actual = probe_earliest(mt5, symbol, tf_const)
    start  = max(from_date, actual)
    step   = timedelta(days=TF_STEP.get(tf_name.split(":")[-1], 90))
    to_dt  = datetime.now(timezone.utc)
    chunks = []; total = 0; current = start

    while current < to_dt:
        end   = min(current + step, to_dt)
        rates = mt5.copy_rates_range(symbol, tf_const, current, end)
        n     = len(rates) if rates is not None else 0
        if n > 0:
            chunks.append(pd.DataFrame(rates))
            total += n
        print(f"\r{progress_bar(int((current-start).total_seconds()),
              int((to_dt-start).total_seconds()),
              f'{tf_name:<12}', f'{total:>10,} bars  {current.date()}')}",
              end="", flush=True)
        current = end
        time.sleep(0.02)
    print()

    if not chunks: return None
    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    log(f"{c(C.CYAN, tf_name):<20} {c(C.GREEN, f'{len(df):>10,} bars')}  "
        f"{c(C.DIM, str(actual.date()))} → {c(C.DIM, str(to_dt.date()))}", "DATA")
    return df

def enrich_ohlcv(df):
    df["body"]      = (df["close"] - df["open"]).abs()
    df["range"]     = df["high"] - df["low"]
    df["upper_wick"]= df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"]= df[["open","close"]].min(axis=1) - df["low"]
    df["direction"] = np.sign(df["close"] - df["open"]).astype(int)
    df["hlc3"]      = (df["high"] + df["low"] + df["close"]) / 3
    df["body_pct"]  = df["body"] / (df["range"] + 1e-10)
    return df

def fetch_phase(mt5, symbol, resume, no_ticks, tick_days):
    """Full MT5 data fetch — all timeframes, all symbols, ticks."""

    TIMEFRAMES = {
        "M1": mt5.TIMEFRAME_M1,  "M2": mt5.TIMEFRAME_M2,
        "M3": mt5.TIMEFRAME_M3,  "M4": mt5.TIMEFRAME_M4,
        "M5": mt5.TIMEFRAME_M5,  "M6": mt5.TIMEFRAME_M6,
        "M10":mt5.TIMEFRAME_M10, "M12":mt5.TIMEFRAME_M12,
        "M15":mt5.TIMEFRAME_M15, "M20":mt5.TIMEFRAME_M20,
        "M30":mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1,
        "H2": mt5.TIMEFRAME_H2,  "H3": mt5.TIMEFRAME_H3,
        "H4": mt5.TIMEFRAME_H4,  "H6": mt5.TIMEFRAME_H6,
        "H8": mt5.TIMEFRAME_H8,  "H12":mt5.TIMEFRAME_H12,
        "D1": mt5.TIMEFRAME_D1,  "W1": mt5.TIMEFRAME_W1,
        "MN1":mt5.TIMEFRAME_MN1,
    }
    FROM_2000 = datetime(2000, 1, 1, tzinfo=timezone.utc)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ── Account info ──────────────────────────────────────────────────────────
    section(f"XAUUSD — ALL {len(TIMEFRAMES)} TIMEFRAMES — MAX HISTORY", "📈")
    acc  = mt5.account_info()
    term = mt5.terminal_info()
    save_json({"account": {k:str(v) for k,v in acc._asdict().items()},
               "terminal":{k:str(v) for k,v in term._asdict().items()} if term else {}},
              RAW_DIR / "account_info.json")

    # ── All broker symbols ────────────────────────────────────────────────────
    syms = mt5.symbols_get()
    rows = []
    for s in (syms or []):
        d = s._asdict()
        rows.append({k:d.get(k) for k in ["name","description","path","currency_base",
            "currency_profit","digits","point","spread","trade_contract_size",
            "volume_min","volume_max","swap_long","swap_short","margin_initial","visible"]})
    sym_df = pd.DataFrame(rows)
    save_csv(sym_df, RAW_DIR / "all_symbols.csv")
    avail = set(sym_df["name"].dropna())
    log(f"Broker has {c(C.BOLD, str(len(avail)))} symbols available")

    # ── XAUUSD symbol info ────────────────────────────────────────────────────
    mt5.symbol_select(symbol, True)
    info = mt5.symbol_info(symbol)
    if info:
        save_json({k:str(v) for k,v in info._asdict().items()},
                  RAW_DIR / f"{symbol.lower()}_symbol_info.json")
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            spread = round(tick.ask - tick.bid, info.digits)
            save_json({"time": datetime.fromtimestamp(tick.time,tz=timezone.utc).isoformat(),
                       "bid":tick.bid,"ask":tick.ask,"spread":spread},
                      RAW_DIR / f"{symbol.lower()}_live_tick.json")
            log(f"Live {symbol}:  bid={c(C.GREEN, str(tick.bid))}  "
                f"ask={c(C.RED, str(tick.ask))}  spread={c(C.YELLOW, str(spread))}")

    # ── XAUUSD ALL TIMEFRAMES — MAX HISTORY ───────────────────────────────────
    for tf_name, tf_const in TIMEFRAMES.items():
        out = RAW_DIR / f"{symbol.lower()}_{tf_name.lower()}.csv"
        if skip(out, resume): continue
        df = fetch_bars(mt5, symbol, tf_const, tf_name, FROM_2000)
        if df is None: log(f"{tf_name} — no data from broker", "WARN"); continue
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.rename(columns={"tick_volume":"tick_vol","real_volume":"real_vol"}, inplace=True)
        df = enrich_ohlcv(df)
        save_csv(df, out)

    # ── CONTEXT SYMBOLS — ALL TIMEFRAMES — 2000 ───────────────────────────────
    section("CONTEXT SYMBOLS — ALL 21 TIMEFRAMES — FROM 2000", "🌍")
    for sym in CONTEXT_H1:
        if sym not in avail: continue
        print(c(C.CYAN + C.BOLD, f"\n  ── {sym} ─────────────────────────────────────────"))
        mt5.symbol_select(sym, True)
        for tf_name, tf_const in TIMEFRAMES.items():
            out = RAW_DIR / f"{sym.lower()}_{tf_name.lower()}.csv"
            if skip(out, resume): continue
            df = fetch_bars(mt5, sym, tf_const, f"{sym}:{tf_name}", FROM_2000)
            if df is None: continue
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.rename(columns={"tick_volume":"tick_vol","real_volume":"real_vol"}, inplace=True)
            df = enrich_ohlcv(df)
            save_csv(df, out)

    # ── TICK DATA — STREAM TO DISK ─────────────────────────────────────────────
    if not no_ticks:
        section("TICK DATA — EVERY BID/ASK — 5 YEARS", "⚡")
        _fetch_ticks(mt5, symbol, tick_days, resume)
        _build_enriched_m1(symbol, resume)

    log("Fetch phase complete ✓", "DONE")

def _fetch_ticks(mt5, symbol, days_back, resume):
    out = RAW_DIR / f"{symbol.lower()}_ticks.csv"
    if skip(out, resume): return

    mt5.symbol_select(symbol, True)
    to_dt   = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(days=days_back)

    # Probe actual earliest tick
    for d in [days_back, 1825, 2555, 3650]:
        test = to_dt - timedelta(days=d)
        p    = mt5.copy_ticks_range(symbol, test, test+timedelta(days=1), mt5.COPY_TICKS_ALL)
        if p is not None and len(p) > 0: from_dt = test
        else: break

    total_days = max(1, (to_dt - from_dt).days)
    log(f"Ticks: {c(C.CYAN, str(from_dt.date()))} → {c(C.CYAN, str(to_dt.date()))}  "
        f"({c(C.BOLD, str(total_days))} days)", "TICK")

    COLS = ["datetime","datetime_ms","bid","ask","last","volume","volume_real",
            "spread","mid","tick_dir","flags","is_bid","is_ask","is_last",
            "is_volume","is_buy","is_sell","time","time_msc"]

    chunk_start = from_dt; days_done = 0; total = 0

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()

        while chunk_start < to_dt:
            chunk_end = min(chunk_start + timedelta(days=1), to_dt)
            ticks = mt5.copy_ticks_range(symbol, chunk_start, chunk_end, mt5.COPY_TICKS_ALL)
            n = len(ticks) if ticks is not None else 0

            if n > 0:
                df = pd.DataFrame(ticks)
                df["datetime"]    = pd.to_datetime(df["time"],     unit="s",  utc=True).dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
                df["datetime_ms"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S.%f+00:00")
                df["is_bid"]    = (df["flags"] & 2).astype(bool)
                df["is_ask"]    = (df["flags"] & 4).astype(bool)
                df["is_last"]   = (df["flags"] & 8).astype(bool)
                df["is_volume"] = (df["flags"] & 16).astype(bool)
                df["is_buy"]    = (df["flags"] & 32).astype(bool)
                df["is_sell"]   = (df["flags"] & 64).astype(bool)
                df["mid"]       = (df["bid"] + df["ask"]) / 2
                df["spread"]    = df["ask"] - df["bid"]
                df["tick_dir"]  = np.sign(df["mid"].diff()).fillna(0).astype(int)
                for row in df[COLS].itertuples(index=False):
                    writer.writerow(dict(zip(COLS, row)))
                total += n

            days_done += 1
            mb = out.stat().st_size / 1e6 if out.exists() else 0
            print(f"\r{progress_bar(days_done, total_days, str(chunk_start.date()), f'{total:>12,} ticks  {mb:>7.0f} MB')}",
                  end="", flush=True)
            chunk_start = chunk_end
            time.sleep(0.02)

    print()
    mb = out.stat().st_size / 1e6
    STATS.ticks_fetched = total
    log(f"{c(C.BOLD, f'{total:,}')} ticks  {c(C.YELLOW, f'{mb:.0f} MB')} → {out.name}", "TICK")

def _build_enriched_m1(symbol, resume):
    section("ENRICHED M1 FROM TICKS", "🔬")
    out = RAW_DIR / f"{symbol.lower()}_m1_enriched.csv"
    if skip(out, resume): return
    tick_path = RAW_DIR / f"{symbol.lower()}_ticks.csv"
    if not tick_path.exists(): log("No tick file", "WARN"); return

    log("Reading tick CSV in 5M-row chunks and aggregating...")
    agg_chunks = []; rows_read = 0
    for chunk in pd.read_csv(tick_path, chunksize=5_000_000, parse_dates=["datetime_ms"]):
        rows_read += len(chunk)
        chunk = chunk.set_index("datetime_ms").sort_index()
        a = chunk.resample("1min").agg(
            open_bid=("bid","first"),  high_bid=("bid","max"),
            low_bid=("bid","min"),     close_bid=("bid","last"),
            open_ask=("ask","first"),  high_ask=("ask","max"),
            low_ask=("ask","min"),     close_ask=("ask","last"),
            open_mid=("mid","first"),  high_mid=("mid","max"),
            low_mid=("mid","min"),     close_mid=("mid","last"),
            spread_mean=("spread","mean"), spread_max=("spread","max"),
            spread_min=("spread","min"),   spread_std=("spread","std"),
            volume=("volume","sum"),   volume_real=("volume_real","sum"),
            tick_count=("mid","count"),
            buy_ticks=("is_buy","sum"),    sell_ticks=("is_sell","sum"),
            bid_ticks=("is_bid","sum"),    ask_ticks=("is_ask","sum"),
        ).dropna(subset=["close_mid"])
        agg_chunks.append(a)
        log(f"  Processed {c(C.BOLD, f'{rows_read:,}')} ticks...", "TICK")

    agg = pd.concat(agg_chunks).groupby(level=0).agg({
        "open_bid":"first","high_bid":"max","low_bid":"min","close_bid":"last",
        "open_ask":"first","high_ask":"max","low_ask":"min","close_ask":"last",
        "open_mid":"first","high_mid":"max","low_mid":"min","close_mid":"last",
        "spread_mean":"mean","spread_max":"max","spread_min":"min","spread_std":"mean",
        "volume":"sum","volume_real":"sum","tick_count":"sum",
        "buy_ticks":"sum","sell_ticks":"sum","bid_ticks":"sum","ask_ticks":"sum",
    })
    agg["ofi"]          = (agg["buy_ticks"]-agg["sell_ticks"])/(agg["buy_ticks"]+agg["sell_ticks"]+1e-9)
    agg["signed_vol"]   = agg["volume"] * agg["ofi"]
    agg["spread_pct"]   = agg["spread_mean"] / (agg["close_mid"]+1e-10) * 100
    agg["bid_ask_ratio"]= agg["bid_ticks"] / (agg["ask_ticks"]+1e-9)
    agg["range_mid"]    = agg["high_mid"] - agg["low_mid"]
    agg["vwap_tick"]    = (agg["high_mid"]+agg["low_mid"]+agg["close_mid"]) / 3
    agg = agg.reset_index().rename(columns={"datetime_ms":"time"})
    save_csv(agg, out)

# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████████████████████████████████████████
# PHASE 2 — BUILD FEATURES
# ██████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

def build_features_phase(symbol, resume):
    section("BUILDING 700+ FEATURE DATASET", "🔬")

    out = PROC_DIR / "xauusd_hydra.parquet"
    if skip(out, resume): return

    # Load M1
    m1 = load_tf(symbol, "m1")
    if m1 is None: raise FileNotFoundError("xauusd_m1.csv not found — run fetch first")
    log(f"M1 base: {c(C.BOLD, f'{len(m1):,}')} bars  "
        f"{c(C.DIM, str(m1.index[0].date()))} → {c(C.DIM, str(m1.index[-1].date()))}")

    # Merge tick features
    ep = RAW_DIR / f"{symbol.lower()}_m1_enriched.csv"
    if ep.exists():
        e = pd.read_csv(ep, parse_dates=["time"]).set_index("time")
        if not e.index.tz: e.index = e.index.tz_localize("UTC")
        extra = [c2 for c2 in e.columns if c2 not in m1.columns]
        m1 = m1.join(e[extra], how="left"); m1[extra] = m1[extra].ffill().fillna(0)
        log(f"Tick features merged: {c(C.GREEN, str(len(extra)))} columns")

    # Merge XAUUSD HTF
    for tf in ["m5","m15","m30","h1","h4","h12","d1","w1"]:
        df = load_tf(symbol, tf)
        if df is None: continue
        df.columns = [f"xau_{tf}_{x}" for x in df.columns]
        m1 = m1.join(df, how="left").ffill()
        log(f"Merged XAUUSD {tf.upper()}: {c(C.GREEN, f'{len(df):,}')} bars")

    # Merge cross-asset
    ASSETS = [
        ("eurusd",["m5","m15","h1","h4","d1"]),("usdchf",["m5","m15","h1","h4"]),
        ("usdjpy",["m5","m15","h1","h4","d1"]),("gbpusd",["m5","h1","h4"]),
        ("usdcad",["h1","h4"]),("audusd",["h1","h4"]),
        ("usdx",["m5","m15","h1","h4","d1"]),("xagusd",["m5","m15","h1","h4","d1"]),
        ("xptusd",["h1","d1"]),("spx500",["m5","h1","h4","d1"]),
        ("us500",["m5","h1","h4","d1"]),("us30",["h1","h4","d1"]),
        ("ger40",["h1","d1"]),("nas100",["h1","d1"]),
        ("btcusd",["h1","h4","d1"]),("usoil",["h1","h4","d1"]),
    ]
    for sym, tfs in ASSETS:
        for tf in tfs:
            df = load_tf(sym, tf)
            if df is None: continue
            keep = [x for x in ["open","high","low","close","tick_vol","direction","hlc3"] if x in df.columns]
            df   = df[keep].copy(); df.columns = [f"{sym}_{tf}_{x}" for x in df.columns]
            m1   = m1.join(df, how="left").ffill()

    log(f"Cross-asset merge complete. Shape: {c(C.BOLD, str(m1.shape))}")

    # Compute all features
    c_arr = m1["close"].values.astype(np.float64)
    h_arr = m1["high"].values.astype(np.float64)
    l_arr = m1["low"].values.astype(np.float64)
    o_arr = m1["open"].values.astype(np.float64)
    v_arr = m1["tick_vol"].values.astype(np.float64) if "tick_vol" in m1.columns else np.ones(len(m1))
    n     = len(c_arr)

    feat: dict[str, np.ndarray] = {}
    log("Computing technical indicators...", "FEAT")

    # EMAs
    for p in [3,5,7,8,9,10,13,14,20,21,25,34,50,55,89,100,144,200,233,377,500]:
        feat[f"ema_{p}"]       = ema(c_arr, p)
        feat[f"ema_{p}_dist"]  = (c_arr-feat[f"ema_{p}"])/(feat[f"ema_{p}"]+1e-10)*100
        feat[f"ema_{p}_slope"] = np.gradient(feat[f"ema_{p}"])
    for fast,slow in [(8,21),(13,34),(20,50),(50,200),(21,55)]:
        feat[f"ema_x_{fast}_{slow}"] = feat[f"ema_{fast}"] - feat[f"ema_{slow}"]
    log(f"  EMAs: {c(C.GREEN, str(len([k for k in feat if k.startswith('ema_')])))} features")

    # SMAs
    for p in [5,10,20,50,100,200]:
        feat[f"sma_{p}"]      = rolling_mean(c_arr, p)
        feat[f"sma_{p}_dist"] = (c_arr-feat[f"sma_{p}"])/(feat[f"sma_{p}"]+1e-10)*100

    # RSI
    delta = np.diff(c_arr, prepend=c_arr[0])
    gain  = np.where(delta>0, delta, 0.0)
    loss  = np.where(delta<0, -delta, 0.0)
    for p in [2,3,5,7,9,14,21,28]:
        ag=np.zeros(n); al=np.zeros(n)
        if p<n: ag[p]=np.mean(gain[1:p+1]); al[p]=np.mean(loss[1:p+1])
        for i in range(p+1,n): ag[i]=(ag[i-1]*(p-1)+gain[i])/p; al[i]=(al[i-1]*(p-1)+loss[i])/p
        rs=np.where(al==0,100.0,ag/(al+1e-10)); rsi=100-100/(1+rs); rsi[:p]=np.nan
        feat[f"rsi_{p}"]=rsi; feat[f"rsi_{p}_ob"]=(rsi>70).astype(float); feat[f"rsi_{p}_os"]=(rsi<30).astype(float)
    log(f"  RSI: 8 periods + overbought/oversold flags")

    # MACD
    for fast,slow,sig in [(12,26,9),(5,13,1),(8,21,5),(3,10,16)]:
        ml=ema(c_arr,fast)-ema(c_arr,slow); sl=ema(ml,sig); hl=ml-sl
        feat[f"macd_{fast}_{slow}"]=ml; feat[f"macd_s_{fast}_{slow}"]=sl; feat[f"macd_h_{fast}_{slow}"]=hl
    log(f"  MACD: 4 settings")

    # Bollinger Bands
    c_prev = np.roll(c_arr,1); c_prev[0]=c_arr[0]
    tr = np.maximum(h_arr-l_arr, np.maximum(abs(h_arr-c_prev), abs(l_arr-c_prev)))
    for p,k in [(10,2),(20,1),(20,2),(20,2.5),(50,2)]:
        mid=rolling_mean(c_arr,p); std=rolling_std(c_arr,p)
        up=mid+k*std; lo=mid-k*std
        feat[f"bb_w_{p}_{k}"]=(up-lo)/(mid+1e-10); feat[f"bb_pct_{p}_{k}"]=(c_arr-lo)/(up-lo+1e-10)
        feat[f"bb_up_{p}_{k}"]=up; feat[f"bb_lo_{p}_{k}"]=lo

    # ATR
    for p in [5,7,10,14,20,50]:
        atr=np.zeros(n); atr[p-1]=np.mean(tr[:p])
        for i in range(p,n): atr[i]=(atr[i-1]*(p-1)+tr[i])/p
        atr[:p-1]=np.nan
        feat[f"atr_{p}"]=atr; feat[f"atr_{p}_pct"]=atr/(c_arr+1e-10)*100

    # Stoch, ADX, Williams, CCI
    for k_p,d_p in [(5,3),(14,3),(21,3)]:
        k_r=np.full(n,np.nan)
        for i in range(k_p-1,n):
            hi=np.max(h_arr[i-k_p+1:i+1]); lo=np.min(l_arr[i-k_p+1:i+1])
            k_r[i]=0 if hi==lo else 100*(c_arr[i]-lo)/(hi-lo)
        feat[f"stoch_k_{k_p}"]=k_r; feat[f"stoch_d_{k_p}"]=rolling_mean(k_r,d_p)
    for p in [7,14,21]:
        dm_p=np.maximum(h_arr[1:]-h_arr[:-1],0); dm_p=np.insert(dm_p,0,0)
        dm_m=np.maximum(l_arr[:-1]-l_arr[1:],0); dm_m=np.insert(dm_m,0,0)
        atr_s=np.zeros(n); dp=np.zeros(n); dm=np.zeros(n)
        atr_s[p]=np.sum(tr[1:p+1]); dp[p]=np.sum(dm_p[1:p+1]); dm[p]=np.sum(dm_m[1:p+1])
        for i in range(p+1,n): atr_s[i]=atr_s[i-1]-atr_s[i-1]/p+tr[i]; dp[i]=dp[i-1]-dp[i-1]/p+dm_p[i]; dm[i]=dm[i-1]-dm[i-1]/p+dm_m[i]
        dip=100*dp/(atr_s+1e-10); dim=100*dm/(atr_s+1e-10)
        dx=100*abs(dip-dim)/(dip+dim+1e-10); adx=np.full(n,np.nan)
        if 2*p<n: adx[2*p]=np.mean(dx[p:2*p+1])
        for i in range(2*p+1,n): adx[i]=(adx[i-1]*(p-1)+dx[i])/p
        feat[f"adx_{p}"]=adx; feat[f"dip_{p}"]=dip; feat[f"dim_{p}"]=dim
    log(f"  Stochastic, ADX, Bollinger, ATR complete")

    # OBV, CMF, volume features
    obv=np.zeros(n)
    for i in range(1,n): obv[i]=obv[i-1]+(v_arr[i] if c_arr[i]>c_arr[i-1] else (-v_arr[i] if c_arr[i]<c_arr[i-1] else 0))
    feat["obv"]=obv; feat["obv_ema14"]=ema(obv,14)
    buy_vol=np.where(c_arr>=o_arr,v_arr,v_arr*(c_arr-l_arr)/(h_arr-l_arr+1e-10))
    sell_vol=np.where(c_arr<o_arr,v_arr,v_arr*(h_arr-c_arr)/(h_arr-l_arr+1e-10))
    feat["vol_delta"]=buy_vol-sell_vol; feat["cum_delta"]=np.cumsum(buy_vol-sell_vol)
    v_mean=rolling_mean(v_arr,20); v_std=rolling_std(v_arr,20)
    feat["vol_zscore"]=(v_arr-v_mean)/(v_std+1e-10)
    feat["vol_spike_2s"]=(feat["vol_zscore"]>2).astype(float)
    feat["vol_spike_bull"]=((feat["vol_zscore"]>2)&(c_arr>=o_arr)).astype(float)
    feat["vol_spike_bear"]=((feat["vol_zscore"]>2)&(c_arr<o_arr)).astype(float)
    for p in [10,20,50,200]: feat[f"vol_ratio_{p}"]=v_arr/(rolling_mean(v_arr,p)+1e-10)
    mfm=((c_arr-l_arr)-(h_arr-c_arr))/(h_arr-l_arr+1e-10); mfv=mfm*v_arr
    for p in [20,21]:
        cmf=np.full(n,np.nan)
        for i in range(p-1,n): cmf[i]=np.sum(mfv[i-p+1:i+1])/(np.sum(v_arr[i-p+1:i+1])+1e-10)
        feat[f"cmf_{p}"]=cmf
    log(f"  Volume features complete")

    # Gaps + FVGs
    gap=np.append(0.0,o_arr[1:]-c_arr[:-1]); gap_pct=gap/(c_arr+1e-10)*100
    feat["gap"]=gap; feat["gap_pct"]=gap_pct; feat["gap_up"]=(gap>0).astype(float)
    feat["gap_down"]=(gap<0).astype(float); feat["gap_abs"]=(abs(gap_pct))
    bull_fvg=np.zeros(n); bear_fvg=np.zeros(n)
    for i in range(2,n):
        if h_arr[i-2]<l_arr[i]: bull_fvg[i]=1
        if l_arr[i-2]>h_arr[i]: bear_fvg[i]=1
    feat["bull_fvg"]=bull_fvg; feat["bear_fvg"]=bear_fvg
    for p in [10,20,50]:
        fc=np.full(n,0.0)
        for i in range(p,n): fc[i]=np.sum(bull_fvg[i-p:i])+np.sum(bear_fvg[i-p:i])
        feat[f"fvg_count_{p}"]=fc
    log(f"  Gaps + FVGs complete")

    # Returns, volatility, momentum
    log_ret=np.append(0.0,np.log((c_arr[1:]+1e-10)/(c_arr[:-1]+1e-10)))
    for p in [1,3,5,10,15,20,30,60,120,240,480,1440]:
        ret=np.full(n,np.nan); ret[p:]=(c_arr[p:]-c_arr[:-p])/(c_arr[:-p]+1e-10)*100
        feat[f"ret_{p}"]=ret; feat[f"mom_{p}"]=np.append(np.full(p,np.nan),c_arr[p:]-c_arr[:-p])
    for p in [10,20,50,200,1440]:
        rv=np.full(n,np.nan)
        for i in range(p,n): rv[i]=np.std(log_ret[i-p+1:i+1])*np.sqrt(1440)*100
        feat[f"realvol_{p}"]=rv
    log(f"  Returns, volatility, momentum complete")

    # Swing highs/lows, BOS
    for left,right in [(3,3),(5,5),(10,10)]:
        sh=np.zeros(n); sl=np.zeros(n)
        for i in range(left,n-right):
            if all(h_arr[i]>h_arr[i-j] for j in range(1,left+1)) and all(h_arr[i]>h_arr[i+j] for j in range(1,right+1)): sh[i]=1
            if all(l_arr[i]<l_arr[i-j] for j in range(1,left+1)) and all(l_arr[i]<l_arr[i+j] for j in range(1,right+1)): sl[i]=1
        feat[f"swing_high_{left}"]=sh; feat[f"swing_low_{left}"]=sl
    log(f"  Swing highs/lows + BOS complete")

    # Time + session
    idx=m1.index; h_idx=idx.hour; dow=idx.dayofweek
    feat["hour"]=h_idx; feat["dow"]=dow; feat["month"]=idx.month
    feat["hour_sin"]=np.sin(2*np.pi*h_idx/24); feat["hour_cos"]=np.cos(2*np.pi*h_idx/24)
    feat["dow_sin"]=np.sin(2*np.pi*dow/7);   feat["dow_cos"]=np.cos(2*np.pi*dow/7)
    feat["sess_asian"]=((h_idx>=0)&(h_idx<8)).astype(int)
    feat["sess_london"]=((h_idx>=7)&(h_idx<16)).astype(int)
    feat["sess_ny"]=((h_idx>=13)&(h_idx<21)).astype(int)
    feat["sess_overlap"]=((h_idx>=13)&(h_idx<16)).astype(int)

    # Candle patterns
    body=abs(c_arr-o_arr); rng=h_arr-l_arr
    up_w=h_arr-np.maximum(c_arr,o_arr); lo_w=np.minimum(c_arr,o_arr)-l_arr
    feat["body_pct"]=body/(rng+1e-10); feat["up_wick_pct"]=up_w/(rng+1e-10)
    feat["lo_wick_pct"]=lo_w/(rng+1e-10); feat["direction"]=np.sign(c_arr-o_arr)
    feat["is_doji"]=(body/(rng+1e-10)<0.1).astype(float)
    feat["is_hammer"]=((lo_w>2*body)&(up_w<body)&(c_arr>o_arr)).astype(float)
    feat["is_shooting_star"]=((up_w>2*body)&(lo_w<body)&(c_arr<o_arr)).astype(float)
    log(f"  Candle patterns, time features complete")

    # Hurst exponent
    hurst=np.full(n,0.5)
    for i in range(100,n):
        w=log_ret[i-100:i]; rs=[]
        for sz in [10,20,50]:
            for j in range(0,len(w)-sz,sz):
                s=w[j:j+sz]; cs=np.cumsum(s-np.mean(s)); rng2=np.max(cs)-np.min(cs); std2=np.std(s)+1e-10
                rs.append(rng2/std2/np.sqrt(sz))
        hurst[i]=min(1.0,max(0.0,np.mean(rs)/2+0.5)) if rs else 0.5
    feat["hurst"]=hurst; feat["trending_hurst"]=(hurst>0.55).astype(float)
    log(f"  Hurst exponent computed")

    STATS.features_built = len(feat)
    log(f"Total features computed: {c(C.BOLD+C.GREEN, str(len(feat)))}", "FEAT")

    # Add labels
    for horizon in [5,15,30,60,240,1440]:
        fut=np.full(n,np.nan); fut[:n-horizon]=(c_arr[horizon:]-c_arr[:n-horizon])/(c_arr[:n-horizon]+1e-10)*100
        lbl=np.full(n,1); lbl[np.nan_to_num(fut)>0.05]=2; lbl[np.nan_to_num(fut)<-0.05]=0; lbl[np.isnan(fut)]=-1
        feat[f"label_{horizon}m"]=lbl; feat[f"future_ret_{horizon}"]=fut

    # Build final dataframe
    feat_df = pd.DataFrame(feat, index=m1.index)
    df_final = pd.concat([m1, feat_df], axis=1)

    # Clean
    nan_frac = df_final.isnull().mean(axis=1)
    df_final = df_final[nan_frac < 0.5].copy()
    df_final = df_final.ffill().bfill()
    df_final = df_final.replace([np.inf,-np.inf], np.nan).ffill().bfill()
    empty = [col for col in df_final.columns if df_final[col].isnull().all()]
    if empty: df_final = df_final.drop(columns=empty)
    df_final = df_final.copy()

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(out, index=True)
    mb = out.stat().st_size / 1e6
    STATS.add(len(df_final), mb)
    log(f"Parquet saved: {c(C.BOLD, f'{len(df_final):,}')} rows × "
        f"{c(C.BOLD, str(len(df_final.columns)))} columns  "
        f"{c(C.YELLOW, f'{mb:.0f} MB')}", "SAVE")

    # Save feature list
    feat_path = PROC_DIR / "feature_columns.txt"
    with open(feat_path, "w") as f:
        for col in df_final.columns: f.write(col+"\n")
    log(f"Feature list saved → {feat_path.name} ({len(df_final.columns)} features)")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████████████████████████████████████████
# PHASE 4 — PREPARE TRAINING DATASET
# ██████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset_phase(symbol: str, resume: bool) -> Path | None:
    """
    Loads the feature parquet, normalizes everything properly,
    splits into train/val/test (no lookahead leakage), and saves
    ready-to-use .npz files that train_hydra.py loads instantly.

    Output files:
      data/processed/
        train.npz       ← X_cont, X_cat, y_5m, y_15m, y_60m, timestamps
        val.npz
        test.npz
        scaler.json     ← mean/std for every feature (for live inference)
        dataset_meta.json  ← splits, shapes, feature names, label counts
    """
    section("PREPARING TRAINING DATASET", "📦")

    out_train = PROC_DIR / "train.npz"
    out_val   = PROC_DIR / "val.npz"
    out_test  = PROC_DIR / "test.npz"
    out_scaler= PROC_DIR / "scaler.json"
    out_meta  = PROC_DIR / "dataset_meta.json"

    if resume and all(p.exists() for p in [out_train, out_val, out_test]):
        log("Dataset files already exist — skipping (use --from-scratch to rebuild)", "WARN")
        return out_train

    parquet = PROC_DIR / "xauusd_hydra.parquet"
    if not parquet.exists():
        log("No parquet found — run build_features phase first", "ERROR")
        return None

    # ── Load ──────────────────────────────────────────────────────────────────
    log(f"Loading {c(C.CYAN, parquet.name)}...")
    df = pd.read_parquet(parquet)
    log(f"Loaded: {c(C.BOLD, f'{len(df):,}')} rows × {c(C.BOLD, str(len(df.columns)))} columns")

    # ── Identify feature columns ──────────────────────────────────────────────
    # Label columns and raw price cols we don't want as features
    EXCLUDE_PREFIXES = ["label_", "future_ret_"]
    EXCLUDE_EXACT    = {"open","high","low","close","tick_vol","real_vol",
                        "spread","time","datetime","datetime_ms",
                        "time_msc","time_s","flags"}

    feature_cols = [
        col for col in df.columns
        if not any(col.startswith(p) for p in EXCLUDE_PREFIXES)
        and col not in EXCLUDE_EXACT
        and df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
    ]

    # Separate categorical from continuous
    CATEGORICAL = ["session", "day_of_week_name"]
    SESSION_MAP  = {"ASIAN":0,"LONDON":1,"OVERLAP_LDN_NY":2,"NEW_YORK":3,"DEAD_ZONE":4}
    DAY_MAP      = {"MON":0,"TUE":1,"WED":2,"THU":3,"FRI":4,"SAT":5,"SUN":6}

    cat_cols  = [c2 for c2 in CATEGORICAL if c2 in df.columns]
    cont_cols = [c2 for c2 in feature_cols if c2 not in cat_cols]

    log(f"Continuous features: {c(C.GREEN, str(len(cont_cols)))}")
    log(f"Categorical features: {c(C.GREEN, str(len(cat_cols)))}")

    # ── Label columns ──────────────────────────────────────────────────────────
    label_5m  = "label_5m"   if "label_5m"   in df.columns else None
    label_15m = "label_15m"  if "label_15m"  in df.columns else None
    label_60m = "label_60m"  if "label_60m"  in df.columns else None

    # Also support label_5, label_15, label_60 naming
    for candidate, attr in [("label_5","label_5m"),("label_15","label_15m"),("label_60","label_60m")]:
        if candidate in df.columns and locals()[attr.replace("m","")] is None:
            pass

    # Find whatever label columns exist
    lbl_cols = [c2 for c2 in df.columns if c2.startswith("label_") and df[c2].dtype in [np.int64,np.int32,np.float64,float]]
    log(f"Label columns: {c(C.CYAN, str(lbl_cols))}")

    # Drop rows where any label is -1 (future unknown)
    if lbl_cols:
        valid_mask = np.ones(len(df), dtype=bool)
        for lc in lbl_cols:
            valid_mask &= (df[lc].values != -1)
        df = df[valid_mask].copy()
        log(f"After removing unknown-label rows: {c(C.BOLD, f'{len(df):,}')} rows")

    # ── Time-based split (NO random shuffle — preserves temporal order) ────────
    n         = len(df)
    val_frac  = 0.10   # 10% validation
    test_frac = 0.10   # 10% test
    train_end = int(n * (1 - val_frac - test_frac))
    val_end   = int(n * (1 - test_frac))

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    log(f"Split:")
    log(f"  Train: {c(C.GREEN,  f'{len(train_df):>8,} rows')}  "
        f"{str(train_df.index[0])[:10]} → {str(train_df.index[-1])[:10]}")
    log(f"  Val:   {c(C.YELLOW, f'{len(val_df):>8,} rows')}  "
        f"{str(val_df.index[0])[:10]} → {str(val_df.index[-1])[:10]}")
    log(f"  Test:  {c(C.RED,    f'{len(test_df):>8,} rows')}  "
        f"{str(test_df.index[0])[:10]} → {str(test_df.index[-1])[:10]}")

    # ── Fit scaler on TRAIN ONLY (critical — no leakage) ──────────────────────
    log("Fitting scaler on training data only (zero-leakage)...")
    train_cont = train_df[cont_cols].values.astype(np.float32)

    # Robust scaler: use median + IQR (handles outliers better than mean/std for finance)
    feat_median = np.nanmedian(train_cont, axis=0)
    q25         = np.nanpercentile(train_cont, 25, axis=0)
    q75         = np.nanpercentile(train_cont, 75, axis=0)
    feat_iqr    = q75 - q25
    feat_iqr    = np.where(feat_iqr == 0, 1.0, feat_iqr)  # avoid div by zero

    # Also compute mean/std for HYDRA inference normalization
    feat_mean   = np.nanmean(train_cont, axis=0)
    feat_std    = np.nanstd(train_cont, axis=0)
    feat_std    = np.where(feat_std == 0, 1.0, feat_std)

    def scale(arr: np.ndarray) -> np.ndarray:
        """Robust scale: (x - median) / IQR, clipped to [-10, 10]"""
        scaled = (arr - feat_median) / feat_iqr
        return np.clip(scaled, -10.0, 10.0).astype(np.float32)

    def encode_cat(df_part: pd.DataFrame) -> np.ndarray:
        """Encode categorical features as integers."""
        cat_arr = np.zeros((len(df_part), max(len(cat_cols), 2)), dtype=np.int32)
        if "session" in df_part.columns:
            cat_arr[:, 0] = df_part["session"].map(SESSION_MAP).fillna(4).astype(int)
        if "day_of_week_name" in df_part.columns:
            cat_arr[:, 1] = df_part["day_of_week_name"].map(DAY_MAP).fillna(0).astype(int)
        elif "day_of_week" in df_part.columns:
            cat_arr[:, 1] = df_part["day_of_week"].fillna(0).astype(int)
        return cat_arr

    def get_labels(df_part: pd.DataFrame) -> dict[str, np.ndarray]:
        """Extract all label arrays."""
        labels = {}
        for lc in lbl_cols:
            labels[lc] = df_part[lc].fillna(1).astype(np.int32).values
        return labels

    # ── Build arrays for each split ───────────────────────────────────────────
    log("Building train arrays...")
    train_cont_raw  = train_df[cont_cols].values.astype(np.float32)
    train_cont_sc   = scale(train_cont_raw)
    train_cat       = encode_cat(train_df)
    train_labels    = get_labels(train_df)
    train_ts        = np.array([str(t) for t in train_df.index])
    train_close     = train_df["close"].values.astype(np.float32) if "close" in train_df.columns else np.zeros(len(train_df), dtype=np.float32)

    log("Building val arrays...")
    val_cont_sc     = scale(val_df[cont_cols].values.astype(np.float32))
    val_cat         = encode_cat(val_df)
    val_labels      = get_labels(val_df)
    val_ts          = np.array([str(t) for t in val_df.index])
    val_close       = val_df["close"].values.astype(np.float32) if "close" in val_df.columns else np.zeros(len(val_df), dtype=np.float32)

    log("Building test arrays...")
    test_cont_sc    = scale(test_df[cont_cols].values.astype(np.float32))
    test_cat        = encode_cat(test_df)
    test_labels     = get_labels(test_df)
    test_ts         = np.array([str(t) for t in test_df.index])
    test_close      = test_df["close"].values.astype(np.float32) if "close" in test_df.columns else np.zeros(len(test_df), dtype=np.float32)

    # ── Label distribution report ─────────────────────────────────────────────
    for lc in lbl_cols:
        vals, counts = np.unique(train_labels[lc], return_counts=True)
        dist = {int(v): int(ct) for v, ct in zip(vals, counts)}
        pcts = {int(v): f"{ct/len(train_labels[lc])*100:.1f}%" for v, ct in zip(vals, counts)}
        log(f"  {lc} distribution: SHORT={pcts.get(0,'0%')}  FLAT={pcts.get(1,'0%')}  LONG={pcts.get(2,'0%')}")

    # ── Save .npz files ───────────────────────────────────────────────────────
    log("Saving train.npz...")
    np.savez_compressed(
        out_train,
        X_cont      = train_cont_sc,
        X_cat       = train_cat,
        close       = train_close,
        timestamps  = train_ts,
        **{f"y_{k}": v for k, v in train_labels.items()},
    )
    mb = out_train.stat().st_size / 1e6
    log(f"train.npz: {c(C.GREEN, f'{len(train_cont_sc):,}')} rows × "
        f"{c(C.GREEN, str(train_cont_sc.shape[1]))} features  "
        f"{c(C.YELLOW, f'{mb:.0f} MB')}", "SAVE")

    log("Saving val.npz...")
    np.savez_compressed(
        out_val,
        X_cont      = val_cont_sc,
        X_cat       = val_cat,
        close       = val_close,
        timestamps  = val_ts,
        **{f"y_{k}": v for k, v in val_labels.items()},
    )
    mb = out_val.stat().st_size / 1e6
    log(f"val.npz:   {c(C.GREEN, f'{len(val_cont_sc):,}')} rows  {c(C.YELLOW, f'{mb:.0f} MB')}", "SAVE")

    log("Saving test.npz...")
    np.savez_compressed(
        out_test,
        X_cont      = test_cont_sc,
        X_cat       = test_cat,
        close       = test_close,
        timestamps  = test_ts,
        **{f"y_{k}": v for k, v in test_labels.items()},
    )
    mb = out_test.stat().st_size / 1e6
    log(f"test.npz:  {c(C.GREEN, f'{len(test_cont_sc):,}')} rows  {c(C.YELLOW, f'{mb:.0f} MB')}", "SAVE")

    # ── Save scaler (needed for live inference) ────────────────────────────────
    scaler_data = {
        "feature_cols":  cont_cols,
        "cat_cols":      cat_cols,
        "label_cols":    lbl_cols,
        "n_features":    len(cont_cols),
        "median":        feat_median.tolist(),
        "iqr":           feat_iqr.tolist(),
        "mean":          feat_mean.tolist(),
        "std":           feat_std.tolist(),
        "clip_min":      -10.0,
        "clip_max":       10.0,
        "scaler_type":   "robust",
        "fit_on":        "train_only",
        "created":       datetime.now(timezone.utc).isoformat(),
    }
    save_json(scaler_data, out_scaler)

    # ── Save metadata ──────────────────────────────────────────────────────────
    # Label class counts for class weighting in trainer
    class_counts = {}
    for lc in lbl_cols:
        vals, counts = np.unique(train_labels[lc], return_counts=True)
        class_counts[lc] = {int(v): int(ct) for v, ct in zip(vals, counts)}

    meta = {
        "symbol":           symbol,
        "created":          datetime.now(timezone.utc).isoformat(),
        "n_train":          int(len(train_cont_sc)),
        "n_val":            int(len(val_cont_sc)),
        "n_test":           int(len(test_cont_sc)),
        "n_features_cont":  int(len(cont_cols)),
        "n_features_cat":   int(len(cat_cols)),
        "n_labels":         int(len(lbl_cols)),
        "label_cols":       lbl_cols,
        "feature_cols":     cont_cols,
        "cat_cols":         cat_cols,
        "class_counts":     class_counts,
        "train_start":      str(train_df.index[0]),
        "train_end":        str(train_df.index[-1]),
        "val_start":        str(val_df.index[0]),
        "val_end":          str(val_df.index[-1]),
        "test_start":       str(test_df.index[0]),
        "test_end":         str(test_df.index[-1]),
        "scaler":           "robust (median + IQR)",
        "split_method":     "chronological (no shuffle)",
        "val_frac":         val_frac,
        "test_frac":        test_frac,
        "npz_files": {
            "train": str(out_train),
            "val":   str(out_val),
            "test":  str(out_test),
        },
        "how_to_load": (
            "data = np.load('data/processed/train.npz', allow_pickle=True)\n"
            "X_cont = data['X_cont']   # shape (N, n_features)\n"
            "X_cat  = data['X_cat']    # shape (N, 2)\n"
            "y      = data['y_label_5m']  # shape (N,) values 0=SHORT 1=FLAT 2=LONG"
        ),
    }
    save_json(meta, out_meta)

    # ── Pretty summary ────────────────────────────────────────────────────────
    print()
    print(c(C.GOLD + C.BOLD, "  ┌──────────────────────────────────────────────────┐"))
    print(c(C.GOLD + C.BOLD, "  │  DATASET READY                                   │"))
    print(c(C.GOLD,          f"  │  Train:    {len(train_cont_sc):>8,} rows  {str(train_df.index[0])[:10]} → {str(train_df.index[-1])[:10]}  │"))
    print(c(C.GOLD,          f"  │  Val:      {len(val_cont_sc):>8,} rows  {str(val_df.index[0])[:10]} → {str(val_df.index[-1])[:10]}  │"))
    print(c(C.GOLD,          f"  │  Test:     {len(test_cont_sc):>8,} rows  {str(test_df.index[0])[:10]} → {str(test_df.index[-1])[:10]}  │"))
    print(c(C.GOLD,          f"  │  Features: {len(cont_cols):>8,} continuous + {len(cat_cols)} categorical       │"))
    print(c(C.GOLD,          f"  │  Labels:   {str(lbl_cols):<42} │"))
    print(c(C.GOLD + C.BOLD, "  └──────────────────────────────────────────────────┘"))
    print()

    log("Dataset preparation complete ✓", "DONE")
    return out_train


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████████████████████████████████████████
# PHASE 3 — TRAIN HYDRA
# ██████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

def train_phase(data_path: Path):
    section("TRAINING HYDRA — 160M PARAMETER ENSEMBLE", "🧠")

    try:
        import torch
        gpu = torch.cuda.is_available()
        device = "cuda" if gpu else "cpu"
        log(f"PyTorch {torch.__version__}  |  "
            f"Device: {c(C.GREEN+C.BOLD, device.upper())}  |  "
            f"{'GPU: '+torch.cuda.get_device_name(0) if gpu else c(C.YELLOW,'CPU MODE (slower)')}")
    except ImportError:
        log("PyTorch not installed. Run: pip install torch", "ERROR")
        return

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    try:
        from scripts.train_hydra import run_training
        log(f"Starting HYDRA training on {c(C.CYAN, str(data_path))}...", "TRAIN")
        results = run_training(
            data_path   = str(data_path),
            max_epochs  = 100,
            batch_size  = 512 if gpu else 64,
            full_model  = True,
            checkpoint_dir = str(MODELS_DIR),
        )
        val_loss = results.get("final_val_loss", 0)
        sharpe   = results.get("best_val_sharpe", 0)
        params   = results.get("model_params", 0)
        log(f"Training complete!  Val loss: {val_loss:.4f}  Sharpe: {sharpe:.2f}  Params: {params:,}", "DONE")
    except ImportError:
        log("Could not import train_hydra. Run from the Aphelion repo root.", "ERROR")
        log(f"Manual command:  python scripts/train_hydra.py --data {data_path} --full --epochs 100", "WARN")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    banner()

    parser = argparse.ArgumentParser(description="APHELION Data Forge — Fetch → Build → Train")
    parser.add_argument("--fetch-only",    action="store_true", help="Only fetch data from MT5")
    parser.add_argument("--features-only", action="store_true", help="Only build features from existing data")
    parser.add_argument("--train-only",    action="store_true", help="Only train HYDRA")
    parser.add_argument("--prepare-only",  action="store_true", help="Only prepare dataset from existing parquet")
    parser.add_argument("--no-train",      action="store_true", help="Fetch + build but skip training")
    parser.add_argument("--no-ticks",      action="store_true", help="Skip tick data (much faster)")
    parser.add_argument("--resume",        action="store_true", help="Skip already-downloaded files")
    parser.add_argument("--from-scratch",  action="store_true", help="Delete existing data and restart")
    parser.add_argument("--symbol",        default=PRIMARY,     help=f"Primary symbol (default {PRIMARY})")
    parser.add_argument("--tick-days",     type=int, default=TICK_DAYS, help="Days of tick history")
    args = parser.parse_args()

    # Wipe if requested
    if args.from_scratch:
        section("CLEARING EXISTING DATA", "🗑")
        for d in [RAW_DIR, PROC_DIR]:
            if d.exists():
                freed = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                shutil.rmtree(d)
                log(f"Cleared {d}  (freed {freed/1e6:.0f} MB)")
        args.resume = False

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── PHASE 1: FETCH ────────────────────────────────────────────────────────
    if not args.features_only and not args.train_only and not getattr(args,'prepare_only',False):
        mt5 = init_mt5()
        connect_mt5(mt5)
        fetch_phase(mt5, args.symbol, args.resume, args.no_ticks, args.tick_days)
        mt5.shutdown()
        log("MT5 disconnected", "INFO")

    # ── PHASE 2: BUILD FEATURES ───────────────────────────────────────────────
    data_path = None
    if not args.fetch_only and not args.train_only and not getattr(args,'prepare_only',False):
        data_path = build_features_phase(args.symbol, args.resume)

    # ── PHASE 4: PREPARE DATASET ─────────────────────────────────────────────
    if not args.fetch_only and not args.train_only or getattr(args,'prepare_only',False):
        prepare_dataset_phase(args.symbol, args.resume)

    # ── PHASE 3: TRAIN ────────────────────────────────────────────────────────
    if not args.fetch_only and not args.features_only and not args.no_train:
        if data_path is None:
            data_path = PROC_DIR / "xauusd_hydra.parquet"
        if data_path.exists():
            train_phase(data_path)
        else:
            log("No parquet found — run without --train-only first", "ERROR")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    section("ALL DONE", "★")
    STATS.print_summary()

    elapsed = time.time() - t0
    print(c(C.GOLD + C.BOLD, f"""
  ┌──────────────────────────────────────────────────────┐
  │  APHELION DATA FORGE COMPLETE                        │
  │                                                      │
  │  Data:      {str(RAW_DIR.absolute()):<40} │
  │  Features:  {str(PROC_DIR/'xauusd_hydra.parquet'):<40} │
  │  Model:     {str(MODELS_DIR):<40} │
  │  Total time: {elapsed/60:.1f} minutes                           │
  └──────────────────────────────────────────────────────┘
"""))

    if not args.no_train and not args.fetch_only and not args.features_only:
        print(c(C.GREEN + C.BOLD, "  HYDRA is trained and ready. Run:  python aphelion.py\n"))
    else:
        print(c(C.CYAN, "  Next:  python aphelion_data.py  (no flags = full pipeline)\n"))


if __name__ == "__main__":
    main()
