# PHASE 23 — TUI SUPREME
## Full App Experience: Zero Terminal Commands Required
### APHELION Engineering Specification v1.0

---

> *"A normal person opens a terminal, types `python aphelion.py`, and never needs to touch the command line again."*

---

## THE PROBLEM

Right now APHELION requires a developer to run it:

```bash
python run_paper.py --mode mt5_tick --capital 10000 --mt5-login 7938565 \
  --mt5-password abc123 --mt5-server Eightcap-Demo --warmup 200
```

That's not a product. That's a script.

**Phase 23 turns APHELION into an app.** One command launches the TUI. Everything else — configuration, starting/stopping trading, training HYDRA, forging strategies with HEPHAESTUS, backtesting — happens through interactive screens.

---

## THE NEW ENTRY POINT

**Before Phase 23:**
```bash
python run_paper.py --mode simulated --capital 10000 --no-tui --warmup 50
```

**After Phase 23:**
```bash
python aphelion.py
```

That's it. Full stop.

---

## ARCHITECTURE OVERVIEW

### Current Architecture (broken)
```
Terminal CMD → run_paper.py (args) → PaperRunner → TUI (read-only display)
```

### Phase 23 Architecture
```
python aphelion.py
    ↓
TUI LAUNCHER SCREEN
    ↓
User configures everything via screens
    ↓
TUI starts/stops PaperRunner internally
    ↓
TUI displays live data AND accepts user input
```

The TUI is now the **orchestrator**, not just a display.

---

## NEW SCREENS (7 new screens added to the 5 existing views)

### Total after Phase 23: 12 navigable screens

```
F1  — Overview       (EXISTING — enhanced)
F2  — HYDRA          (EXISTING — enhanced)
F3  — Risk           (EXISTING — enhanced)
F4  — Analytics      (EXISTING)
F5  — Logs           (EXISTING)
F6  — LAUNCHER       (NEW — system start/stop/mode)
F7  — SETUP          (NEW — all config: MT5, capital, symbol)
F8  — HEPHAESTUS     (NEW — paste Pine Script, forge strategies)
F9  — TRAINING       (NEW — train HYDRA from TUI)
F10 — BACKTEST       (NEW — run backtests from TUI)
F11 — EVOLUTION      (NEW — PROMETHEUS / NEMESIS status)
F12 — SOLA           (NEW — AI self-monitoring panel)
```

---

## SCREEN DESIGNS

---

### SCREEN F6 — LAUNCHER

The **first screen you see** when APHELION opens (before any session starts).

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  ████████████████████████████████████████████████████████████████████████   ║
║  █                                                                       █   ║
║  █          ▄████████  ██████████  ██        ██████████ ██      ██       █   ║
║  █         ██░░░░░░██  ██░░░░░░██  ██       ██░░░░░░██ ████    ██       █   ║
║  █        ██      ██  ████████    ██      ██      ██ ██ ████  ██       █   ║
║  █       ████████    ██          ██     ████████   ██   ████ ██       █   ║
║  █      ██░░░░░░██  ██░░░░░░██  ██    ██░░░░░░██  ██    ██████       █   ║
║  █     ██      ██  ██      ██  ██   ██      ██  ██      ████        █   ║
║  █    ██████████  ██████████  ████ ██████████  ██       ██         █   ║
║  █                        AUTONOMOUS XAU/USD TRADING SYSTEM              █   ║
║  ████████████████████████████████████████████████████████████████████████   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   SYSTEM STATUS                          LAST SESSION                        ║
║   ─────────────────────────────          ──────────────────────────────      ║
║   HYDRA:    ✅ Checkpoint loaded          Date:    2026-03-09 14:22           ║
║   MT5:      ✅ Connected (Eightcap)       Capital: $10,234.55                 ║
║   SENTINEL: ✅ Online                     PnL:     +$234.55 (+2.3%)           ║
║   SOLA:     ✅ Active                     Trades:  14 (64% WR)                ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   SELECT MODE                                                                ║
║                                                                              ║
║   ┌─────────────────────┐   ┌─────────────────────┐   ┌──────────────────┐  ║
║   │  [ENTER] LIVE PAPER │   │   [S] SIMULATED     │   │  [B] BACKTEST    │  ║
║   │                     │   │                     │   │                  │  ║
║   │  Real MT5 data.     │   │  Fake price feed.   │   │  Historical run. │  ║
║   │  Eightcap demo.     │   │  No MT5 needed.     │   │  No live feed.   │  ║
║   │  0.01 lot min.      │   │  Instant bars.      │   │  Full metrics.   │  ║
║   └─────────────────────┘   └─────────────────────┘   └──────────────────┘  ║
║                                                                              ║
║   [C] Configure  [T] Train HYDRA  [H] HEPHAESTUS Forge  [Q] Quit            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

**Bindings on LAUNCHER screen:**
- `ENTER` → Start live paper trading with current config → switches to Overview
- `S` → Start simulated session → switches to Overview
- `B` → Jump to BACKTEST screen
- `C` → Jump to SETUP screen (configure)
- `T` → Jump to TRAINING screen
- `H` → Jump to HEPHAESTUS screen
- `Q` → Quit

**System Status indicators** auto-check:
- HYDRA: looks for `models/hydra_checkpoint.pt` → green/red
- MT5: tries connection with saved credentials → green/red/DEMO
- SENTINEL: always green (in-process)
- SOLA: always green (in-process)

---

### SCREEN F7 — SETUP (Configuration)

All the args from `run_paper.py` become form fields. Tab between fields. Press `S` to save.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  APHELION SETUP                                          [F7]  [ESC: back]   ║
╠═════════════════════════════╦════════════════════════════════════════════════╣
║  SECTION                    ║  SETTINGS                                      ║
║  ─────────────────────────  ║  ─────────────────────────────────────────     ║
║  ▶ Trading                  ║                                                ║
║    MT5 Connection           ║   TRADING                                      ║
║    Risk                     ║   ──────────────────────────────────────────   ║
║    HYDRA                    ║   Symbol        [ XAUUSD              ]        ║
║    System                   ║   Mode          [ Paper ▼             ]        ║
║                             ║   Starting Capital  [ $10,000.00      ]        ║
║                             ║   Warmup Bars   [ 200                 ]        ║
║                             ║                                                ║
║                             ║   MT5 CONNECTION                               ║
║                             ║   ──────────────────────────────────────────   ║
║                             ║   Login         [ 7938565             ]        ║
║                             ║   Password      [ ••••••••            ]        ║
║                             ║   Server        [ Eightcap-Demo       ]        ║
║                             ║   Terminal Path [ C:\MT5\terminal64.exe ]      ║
║                             ║                                                ║
║                             ║   RISK                                         ║
║                             ║   ──────────────────────────────────────────   ║
║                             ║   Max Daily DD  [ 3.0%                ]        ║
║                             ║   Max Exposure  [ 6.0%                ]        ║
║                             ║   Max Positions [ 3                   ]        ║
║                             ║   Risk Per Trade[ 1.0%                ]        ║
║                             ║                                                ║
║                             ║   HYDRA                                        ║
║                             ║   ──────────────────────────────────────────   ║
║                             ║   Checkpoint    [ models/hydra.pt     ] [📂]   ║
║                             ║   Min Confidence[ 0.65                ]        ║
╠═════════════════════════════╩════════════════════════════════════════════════╣
║  [TAB] Next field   [S] Save & Apply   [R] Reset to defaults   [ESC] Back    ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

**Config persistence:** Settings saved to `config/aphelion.json` on `[S]`. Loaded on startup. Never need to set again.

```json
// config/aphelion.json (auto-created, auto-loaded)
{
  "trading": {
    "symbol": "XAUUSD",
    "capital": 10000.0,
    "warmup_bars": 200
  },
  "mt5": {
    "login": 7938565,
    "password": "encrypted:abc123",
    "server": "Eightcap-Demo",
    "terminal_path": "C:\\MT5\\terminal64.exe"
  },
  "risk": {
    "max_daily_dd": 0.03,
    "max_exposure": 0.06,
    "max_positions": 3,
    "risk_per_trade": 0.01
  },
  "hydra": {
    "checkpoint": "models/hydra.pt",
    "min_confidence": 0.65
  }
}
```

---

### SCREEN F8 — HEPHAESTUS FORGE

The killer screen. Paste Pine Script directly into the TUI, watch it get forged live.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  HEPHAESTUS — Strategy Forge                              [F8]  [ESC: back]  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PASTE INDICATOR CODE  (Pine Script / Python / Plain English)                ║
║  ╔════════════════════════════════════════════════════════════════════════╗  ║
║  ║ //@version=5                                                           ║  ║
║  ║ indicator("My RSI Reversal", overlay=false)                            ║  ║
║  ║ rsiLen = input.int(14, "RSI Length")                                   ║  ║
║  ║ rsi = ta.rsi(close, rsiLen)                                            ║  ║
║  ║ longCondition = ta.crossover(rsi, 30)                                  ║  ║
║  ║ shortCondition = ta.crossunder(rsi, 70)                                ║  ║
║  ║ plotshape(longCondition, style=shape.triangleup)                       ║  ║
║  ║ plotshape(shortCondition, style=shape.triangledown)                    ║  ║
║  ║ █  (cursor)                                                            ║  ║
║  ╚════════════════════════════════════════════════════════════════════════╝  ║
║  [F] Forge Strategy   [CTRL+A] Select All   [CTRL+V] Paste   [CTRL+X] Clear ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FORGE PROGRESS                                                              ║
║                                                                              ║
║  ▶ Parsing logic...    [████████████░░░░░░░░]  60%   12s elapsed             ║
║    → Detected: Pine Script v5                                                ║
║    → Name: "RSI Reversal"                                                    ║
║    → Conditions: crossover(RSI,30) BUY | crossunder(RSI,70) SELL            ║
║    → Parameters: rsiLen=14                                                   ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DEPLOYED STRATEGIES (3 active)                                              ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │ NAME                  │ SHARPE │  WR  │ MODE   │ TRADES │ STATUS    │   ║
║  ├──────────────────────────────────────────────────────────────────────┤   ║
║  │ EMA_Cross_8_21        │  1.52  │ 61%  │ FULL   │  2,847 │ ✅ LIVE   │   ║
║  │ BB_Squeeze_Break      │  1.71  │ 63%  │ FULL   │  1,204 │ ✅ LIVE   │   ║
║  │ MACD_Momentum_12_26   │  1.29  │ 55%  │ SHADOW │     89 │ ⏳ SHADOW │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║  RECENT REJECTIONS                                                           ║
║  ✗ ADX_Trend_Filter   Walk-forward variance too high (1.9)                  ║
║  ✗ VWAP_Bounce        Insufficient trades (62 < 150 minimum)                ║
║  [D] Deploy Selected   [R] Remove Selected   [V] View Report                ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

**Forge flow:**
1. User pastes Pine Script into the text area
2. Presses `F` to forge
3. Progress bar fills with live status messages
4. On PASS → strategy appears in deployed table as SHADOW
5. On FAIL → rejection report pops up as modal

**Forge status messages (shown live as LLM works):**
```
▶ Detecting input type...    → Pine Script v5
▶ Parsing logic...           → "RSI Reversal" (confidence: 0.87)
▶ Generating Python class... → RSIReversalVoter (342 lines)
▶ Running sandbox tests...   → 8/8 edge cases passed
▶ Backtesting (2yr data)...  → Sharpe 1.44, WR 58%, 312 trades
▶ Walk-forward (12 folds)... → 9/12 folds passed, median Sharpe 1.31
▶ Monte Carlo (1000 sims)... → P5 Sharpe 0.91, P95 DD 18.2%
▶ TITAN gate...              → PASS ✅
▶ Correlation check...       → max corr 0.42 (diverse) ✅
▶ Deploying to ARES...       → SHADOW mode (500 trades probation)
✅ DEPLOYED: RSI_Reversal_14 — Sharpe 1.44, WR 58%
```

---

### SCREEN F9 — TRAINING

Train HYDRA directly from the TUI. No Python script needed.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  HYDRA TRAINING CENTER                                    [F9]  [ESC: back]  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DATA SOURCE                                                                 ║
║  ○ Synthetic data (no MT5 needed, quick validation)                          ║
║  ● Real data     Path: [ data/processed/xauusd_features.parquet    ] [📂]   ║
║                  Bars: 2,600,000 detected                                    ║
║                                                                              ║
║  TRAINING CONFIG                                                             ║
║  Epochs:      [ 100  ]   Batch Size: [ 512  ]   GPU: [✅ L4 24GB detected]  ║
║  Train Split: [ 80%  ]   Val Split:  [ 10%  ]   Test: [ 10% ]               ║
║  Save Path:   [ models/hydra_checkpoint.pt                       ]           ║
║                                                                              ║
║  QUICK PRESETS:                                                              ║
║  [1] Quick test (500 bars, 2 epochs, ~2min)                                  ║
║  [2] Full synthetic (10k bars, 20 epochs, ~15min)                            ║
║  [3] Full real data (2.6M bars, 100 epochs, ~75min) ← RECOMMENDED           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  TRAINING PROGRESS                                     Elapsed: 00:23:14     ║
║                                                                              ║
║  [████████████████████░░░░░░░░░░░░░]  Epoch 61/100                          ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────────────┐ ║
║  │ MODEL          │ TRAIN LOSS │ VAL LOSS │ SHARPE │ STATUS               │ ║
║  ├────────────────────────────────────────────────────────────────────────┤ ║
║  │ TFT            │   0.412    │  0.438   │  1.24  │ ✅ Training          │ ║
║  │ LSTM           │   0.398    │  0.421   │  1.31  │ ✅ Training          │ ║
║  │ CNN            │   0.445    │  0.467   │  1.18  │ ✅ Training          │ ║
║  │ TCN            │   0.389    │  0.411   │  1.38  │ ✅ Training          │ ║
║  │ Transformer    │   0.421    │  0.449   │  1.22  │ ✅ Training          │ ║
║  │ MoE            │   0.403    │  0.425   │  1.29  │ ✅ Training          │ ║
║  └────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  Loss curve: ▁▂▃▄▅▅▄▄▃▃▃▂▂▂▂▂▂▁▁▁▁▁▁▁ (converging)                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  [ENTER] Start Training   [P] Pause   [STOP] Abort   [L] Load checkpoint    ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

Training runs in a **background thread** so the TUI stays responsive. Progress updates stream in via a queue.

---

### SCREEN F10 — BACKTEST

Run a full backtest from the TUI. See results immediately.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  BACKTEST ENGINE                                         [F10]  [ESC: back]  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CONFIG                                                                      ║
║  Symbol:   [ XAUUSD ]   From: [ 2023-01-01 ]  To: [ 2025-01-01 ]           ║
║  Capital:  [ $10,000 ]  Commission: [ 0.35 pips ]  Slippage: [ ADAPTIVE ]   ║
║  Data:     [ data/raw/xauusd_m1_full.csv                        ] [📂]      ║
║                                                                              ║
║  [ENTER] Run Backtest    [W] Walk-Forward    [M] Monte Carlo    [CTRL+C] Stop║
╠══════════════════════════════════════════════════════════════════════════════╣
║  RESULTS                                          Run: 2026-03-09 22:14      ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │ PERFORMANCE          │ RISK                 │ TRADE STATS           │    ║
║  ├─────────────────────────────────────────────────────────────────────┤    ║
║  │ Total Return: +184%  │ Max Drawdown:  8.2%  │ Total Trades:  2,847  │    ║
║  │ Ann. Return:  +92%   │ Sharpe:        1.84  │ Win Rate:      62.3%  │    ║
║  │ CAGR:         +92%   │ Sortino:       2.41  │ Avg R:         1.82   │    ║
║  │ Final Equity: $28.4k │ Calmar:        11.2  │ Profit Factor: 2.14   │    ║
║  │                      │ Ulcer Index:   1.8%  │ Avg Hold:      14 min │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  EQUITY CURVE                                                                ║
║  $28k ┤                                                         ╭──          ║
║  $24k ┤                                               ╭────────╯            ║
║  $20k ┤                                     ╭────────╯                      ║
║  $15k ┤                           ╭────────╯                                ║
║  $10k ┤───────────────────────────╯                                          ║
║       └────────────────────────────────────────────────────────────→         ║
║       Jan-23                                                    Jan-25       ║
║                                                                              ║
║  [E] Export CSV   [P] Print Report   [S] Save to registry                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

### ENHANCED SCREEN F1 — OVERVIEW (during active session)

When a session is running, Overview gets a **session control bar** at the top:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  ● LIVE SESSION  │  SIMULATED  │  $10,234.55  │  +2.3% today  │  [STOP: Q]  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ... (existing dashboard layout) ...                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

`Q` during an active session now shows a **confirmation modal**:
```
╔════════════════════════════════════╗
║  STOP TRADING SESSION?             ║
║                                    ║
║  Open positions: 1                 ║
║  Unrealized PnL: +$42.30           ║
║                                    ║
║  [Y] Stop & Close All              ║
║  [K] Stop & Keep Positions         ║
║  [ESC] Cancel (keep trading)       ║
╚════════════════════════════════════╝
```

---

## ARCHITECTURAL CHANGES

### New Entry Point: `aphelion.py`

```python
# aphelion.py — the ONLY file a user ever needs
#!/usr/bin/env python3
"""
APHELION — Autonomous XAU/USD Trading System
Usage: python aphelion.py
"""
from aphelion.tui.app import AphelionTUI
from aphelion.tui.config import load_config

def main():
    config = load_config("config/aphelion.json")
    app = AphelionTUI(config=config)
    app.run_sync()

if __name__ == "__main__":
    main()
```

### Config Persistence: `aphelion/tui/config.py`

```python
class AphelionConfig:
    """Persistent config. Loaded at startup, saved on any change."""
    
    def load(path: str) -> AphelionConfig: ...
    def save(self) -> None: ...
    def to_paper_runner_config(self) -> PaperRunnerConfig: ...
    def to_training_config(self) -> TrainingConfig: ...
```

### Session Controller: `aphelion/tui/controller.py`

```python
class SessionController:
    """
    Bridges TUI controls with the PaperRunner.
    The TUI calls methods here; it never imports PaperRunner directly.
    
    Runs PaperRunner in a background asyncio task.
    Streams status back to TUI via asyncio.Queue.
    """
    
    async def start_session(self, config: AphelionConfig, mode: str) -> None:
        """Start PaperRunner in background."""
    
    async def stop_session(self, close_positions: bool = True) -> None:
        """Stop session gracefully."""
    
    async def start_training(self, config: TrainingConfig) -> None:
        """Start HYDRA training in background thread."""
    
    def get_training_progress(self) -> TrainingProgress: ...
    def is_session_active(self) -> bool: ...
    def is_training_active(self) -> bool: ...
```

### Forge Progress Streaming: `aphelion/hephaestus/progress.py`

```python
class ForgeProgressStream:
    """
    Streams forge progress messages to TUI in real-time.
    HEPHAESTUS agent calls emit() at each stage.
    TUI polls get() to update the progress bar.
    """
    
    def emit(self, stage: str, message: str, pct: float) -> None: ...
    def get_latest(self) -> list[ForgeUpdate]: ...
```

---

## NEW FILES TO CREATE

```
aphelion.py                              ← NEW entry point (root)
config/
  aphelion.json                         ← AUTO-CREATED on first run
  aphelion.default.json                 ← Bundled defaults

aphelion/tui/
  config.py                             ← Config load/save/validate
  controller.py                         ← Session + training orchestrator
  screens/
    launcher.py                         ← F6: Launcher screen
    setup.py                            ← F7: Config form
    hephaestus_panel.py                 ← F8: Forge screen
    training_panel.py                   ← F9: Training screen
    backtest_panel.py                   ← F10: Backtest screen
    evolution_panel.py                  ← F11: PROMETHEUS/NEMESIS (EXISTS, enhance)
    sola_panel.py                       ← F12: SOLA panel (EXISTS, enhance)
  widgets/
    text_area.py                        ← Editable text area for Pine Script input
    progress_bar.py                     ← Animated forge/training progress bar
    modal.py                            ← Confirmation/alert modals
    form_field.py                       ← Interactive form input field
    file_picker.py                      ← File browser (checkpoint, data paths)

aphelion/hephaestus/
  progress.py                           ← Forge progress streaming to TUI
```

---

## FILES TO MODIFY

```
aphelion/tui/app.py                     ← Add F6-F12 bindings, initial_screen=LAUNCHER
aphelion/tui/screens/dashboard.py       ← Add session control bar to Overview
aphelion/hephaestus/agent.py            ← Emit progress to ForgeProgressStream
scripts/train_hydra.py                  ← Accept TrainingConfig object (not just args)
```

---

## KEYBOARD MAP (complete)

```
═══════════════════════════════════════════════════
  GLOBAL (any screen)
═══════════════════════════════════════════════════
  F1        → Overview (or Launcher if no session)
  F2        → HYDRA detail
  F3        → Risk / SENTINEL
  F4        → Analytics / Performance
  F5        → Full logs
  F6        → Launcher
  F7        → Setup / Config
  F8        → HEPHAESTUS Forge
  F9        → HYDRA Training
  F10       → Backtest
  F11       → Evolution (PROMETHEUS)
  F12       → SOLA panel
  Q         → Quit (with confirmation if session active)

═══════════════════════════════════════════════════
  LAUNCHER SCREEN (F6)
═══════════════════════════════════════════════════
  ENTER     → Start live paper session
  S         → Start simulated session
  B         → Go to Backtest
  C         → Go to Setup
  T         → Go to Training
  H         → Go to HEPHAESTUS

═══════════════════════════════════════════════════
  HEPHAESTUS SCREEN (F8)
═══════════════════════════════════════════════════
  F (focus on textarea) → Forge current code
  CTRL+V    → Paste from clipboard
  CTRL+A    → Select all
  CTRL+X    → Clear textarea
  D         → Deploy selected strategy
  R         → Remove selected strategy
  V         → View rejection report

═══════════════════════════════════════════════════
  TRAINING SCREEN (F9)
═══════════════════════════════════════════════════
  1         → Quick preset
  2         → Full synthetic preset
  3         → Full real data preset
  ENTER     → Start training
  P         → Pause training
  CTRL+C    → Abort training
  L         → Load existing checkpoint

═══════════════════════════════════════════════════
  ACTIVE SESSION (Overview)
═══════════════════════════════════════════════════
  Q         → Stop session (confirmation modal)
  P         → Pause / Resume (halt new orders)
  E         → Emergency stop (close all positions)

═══════════════════════════════════════════════════
  SETUP SCREEN (F7)
═══════════════════════════════════════════════════
  TAB       → Next field
  SHIFT+TAB → Previous field
  S         → Save & Apply
  R         → Reset to defaults
  ESC       → Back (discard changes)
```

---

## MODAL DIALOGS

### Quit Confirmation (active session)
```
╔══════════════════════════╗
║  Stop trading session?   ║
║                          ║
║  Positions open: 1       ║
║  Unrealized PnL: +$42    ║
║                          ║
║  [Y] Stop + Close All    ║
║  [K] Stop + Keep Open    ║
║  [ESC] Cancel            ║
╚══════════════════════════╝
```

### Forge Complete (PASS)
```
╔══════════════════════════════╗
║  ✅ STRATEGY FORGED          ║
║                              ║
║  Name:    RSI_Reversal_14    ║
║  Sharpe:  1.44               ║
║  Win Rate: 58%               ║
║  Trades:  312                ║
║                              ║
║  Status: SHADOW MODE         ║
║  (500 trades before full)    ║
║                              ║
║  [ENTER] OK                  ║
╚══════════════════════════════╝
```

### Forge Complete (FAIL)
```
╔══════════════════════════════════════════════╗
║  ✗ STRATEGY REJECTED                         ║
║                                              ║
║  Name: ADX_Trend_Filter                      ║
║  Failed at: WALK-FORWARD                     ║
║                                              ║
║  Reasons:                                    ║
║  • WF variance 1.9 > limit 1.2              ║
║  • Only 7/12 folds passed (need 7) ← BORDER ║
║                                              ║
║  Suggestions:                                ║
║  • Add a volatility regime filter            ║
║  • Reduce ADX period from 14 to 10           ║
║                                              ║
║  [ENTER] OK   [V] View Full Report           ║
╚══════════════════════════════════════════════╝
```

### First-Run Wizard
On first launch (no config file found):
```
╔══════════════════════════════════════════════╗
║  Welcome to APHELION                         ║
║                                              ║
║  First run detected. Quick setup:            ║
║                                              ║
║  1. Do you have MetaTrader 5 available?      ║
║     [Y] Yes, set up MT5 connection           ║
║     [N] No, use simulated mode               ║
║                                              ║
║  2. Starting capital: [ $10,000 ]            ║
║                                              ║
║  3. HYDRA checkpoint:                        ║
║     [A] I have a checkpoint (browse...)      ║
║     [B] Train from scratch (go to Training)  ║
║     [C] Skip for now (simulated mode only)   ║
║                                              ║
║  [ENTER] Continue                            ║
╚══════════════════════════════════════════════╝
```

---

## TEST COVERAGE

### New tests to write: ~80 tests

```
tests/tui/
  test_config.py          # load/save/validate config, field types, defaults
  test_controller.py      # session start/stop, training start/stop, state machine
  test_launcher.py        # key bindings, status checks, mode selection
  test_setup.py           # form navigation, validation, save/load
  test_hephaestus_panel.py# paste, forge trigger, progress streaming, table
  test_training_panel.py  # preset selection, progress updates, abort
  test_backtest_panel.py  # config, run, results display, export
  test_modals.py          # quit confirm, forge pass/fail, first-run wizard
```

---

## BUILD ORDER

Phase 23 should be built in this order (each step is independently testable):

**Step 1 — Config layer** (1-2 hrs)
`aphelion/tui/config.py` + `config/aphelion.default.json`
Config saves/loads. All fields validated. Converts to PaperRunnerConfig.

**Step 2 — Controller layer** (2-3 hrs)
`aphelion/tui/controller.py`
SessionController starts/stops PaperRunner in background. TrainingController does same for training. TUI never imports PaperRunner directly.

**Step 3 — Widgets** (1-2 hrs)
`text_area.py`, `progress_bar.py`, `modal.py`, `form_field.py`
Reusable building blocks for the new screens.

**Step 4 — Launcher screen** (1 hr)
`screens/launcher.py`
Status checks, mode buttons, navigation. This is the new "home".

**Step 5 — Setup screen** (1-2 hrs)
`screens/setup.py`
Form fields, validation, save. Uses `form_field.py` widget.

**Step 6 — HEPHAESTUS screen** (2-3 hrs)
`screens/hephaestus_panel.py` + `hephaestus/progress.py`
Text area, forge button, progress streaming, deployed table.

**Step 7 — Training screen** (1-2 hrs)
`screens/training_panel.py`
Preset buttons, per-model table, loss curve, background thread.

**Step 8 — Backtest screen** (1-2 hrs)
`screens/backtest_panel.py`
Config inputs, equity curve render, results table.

**Step 9 — Wire everything** (1 hr)
Update `app.py` F-key bindings, add session control bar to Overview, add quit confirmation modal.

**Step 10 — Entry point** (15 min)
`aphelion.py` at root. `python aphelion.py` works.

---

## DEFINITION OF DONE

Phase 23 is complete when:

1. `python aphelion.py` launches the app with no other commands needed
2. A person with zero Python knowledge can start a simulated session in < 30 seconds
3. MT5 credentials saved and never need re-entering
4. Pine Script paste-and-forge works end-to-end from TUI
5. HYDRA training can be started and monitored from TUI
6. Backtest can be run and results viewed from TUI
7. Quit confirmation prevents accidental position closure
8. First-run wizard handles zero-config cold start
9. All 80 new tests pass

---

*Phase 23 — TUI SUPREME*
*APHELION Engineering Specification v1.0*
*Author: Matin Deev*
*Status: Ready to Build*
