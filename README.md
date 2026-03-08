# APHELION

Autonomous XAU/USD trading system with strict risk governance, engineered feature pipelines, backtesting infrastructure, and an in-progress neural intelligence layer.

> Last repository audit: **2026-03-08 (America/Toronto)**
> 
> Spec reference: `docs/specs/APHELION_Engineering_Spec_v1.docx`

---

## Table of Contents

- [APHELION](#aphelion)
  - [Table of Contents](#table-of-contents)
  - [1. What This Repository Is](#1-what-this-repository-is)
  - [2. Current Reality vs Engineering Spec](#2-current-reality-vs-engineering-spec)
    - [High-level status summary](#high-level-status-summary)
    - [Important README correction](#important-readme-correction)
  - [3. Quick Start](#3-quick-start)
    - [Prerequisites](#prerequisites)
    - [Install](#install)
  - [4. Development Commands](#4-development-commands)
    - [Run tests](#run-tests)
    - [Fast test run](#fast-test-run)
    - [Package install metadata](#package-install-metadata)
  - [5. Project Structure](#5-project-structure)
  - [6. Architecture Overview](#6-architecture-overview)
    - [Tiered governance model (from config)](#tiered-governance-model-from-config)
    - [Runtime pipeline (implemented path)](#runtime-pipeline-implemented-path)
  - [7. Phase-by-Phase Compliance Audit](#7-phase-by-phase-compliance-audit)
    - [Audit legend](#audit-legend)
    - [Notes on phase mismatch](#notes-on-phase-mismatch)
  - [8. Module Status Matrix (20-Module Plan)](#8-module-status-matrix-20-module-plan)
  - [9. Implemented Systems](#9-implemented-systems)
  - [9.1 Core Runtime (Phase 1)](#91-core-runtime-phase-1)
  - [9.2 Feature Engine (Phase 1)](#92-feature-engine-phase-1)
  - [9.3 Risk System / SENTINEL (Phase 2)](#93-risk-system--sentinel-phase-2)
  - [9.4 Backtest Layer (Phase 3)](#94-backtest-layer-phase-3)
  - [10. HYDRA (Phase 4+) Status](#10-hydra-phase-4-status)
  - [11. Testing Status](#11-testing-status)
  - [12. Gaps to Reach Full Spec](#12-gaps-to-reach-full-spec)
    - [Required to close Phase 3 formally](#required-to-close-phase-3-formally)
    - [Required to close Phase 4 formally](#required-to-close-phase-4-formally)
    - [Required for Phases 5-16](#required-for-phases-5-16)
  - [13. Suggested Near-Term Build Order](#13-suggested-near-term-build-order)
  - [14. Contribution Workflow](#14-contribution-workflow)
    - [Branching and commits](#branching-and-commits)
    - [Before opening a PR](#before-opening-a-pr)
    - [Coding conventions](#coding-conventions)
  - [15. License and Confidentiality](#15-license-and-confidentiality)
  - [Appendix A: Phase Requirements Snapshot (from DOCX)](#appendix-a-phase-requirements-snapshot-from-docx)
  - [Appendix B: Directory Completeness Snapshot](#appendix-b-directory-completeness-snapshot)

---

## 1. What This Repository Is

APHELION is designed as a multi-module autonomous trading platform centered on **XAU/USD**. The spec defines a 16-phase build with 20 major modules and 100+ sub-components.

This repository currently contains:

- A solid **core runtime** (event bus, data layer, clock, registry)
- A substantial **feature stack** (microstructure, market structure, volume, VWAP, MTF, cointegration)
- A strong **risk layer** (`SENTINEL`) with hardcoded limits and enforcement components
- A substantial **backtest stack** (engine, broker sim, portfolio, metrics, Monte Carlo, walk-forward, analytics)
- An in-progress **HYDRA** neural layer (TFT + LSTM + CNN + MoE + ensemble scaffolding)
- Many other planned modules still as package scaffolding only

---

## 2. Current Reality vs Engineering Spec

The engineering spec (`docs/specs/APHELION_Engineering_Spec_v1.docx`) is a **target architecture** and roadmap. The repository is currently a **partial implementation** of that roadmap.

### High-level status summary

- Phase 1 (Data Foundation): **Implemented and tested**
- Phase 2 (Risk Layer): **Implemented and tested**
- Phase 3 (Backtesting Engine): **Implemented in code, partially validated**
- Phase 4 (HYDRA v1): **Implemented in code, not yet validated against spec acceptance criteria**
- Phases 5-16: **Mostly not implemented yet (scaffold only)**

### Important README correction

Older README sections marked Phase 3 and Phase 4 as planned only. That is outdated relative to the current codebase. This README reflects the current repository state.

---

## 3. Quick Start

### Prerequisites

- Python `>=3.11`
- Optional: MetaTrader5 terminal for live/paper connectivity
- Optional: CUDA-capable GPU for neural training/inference

### Install

```bash
git clone https://github.com/MatinDeevv/Aphelion-Reasearch.git
cd Aphelion-Reasearch

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
# source venv/bin/activate

pip install -e .
# Optional extras
pip install -e ".[ml]"
pip install -e ".[broker]"
pip install -e ".[tui]"
pip install -e ".[dev]"
# Everything
pip install -e ".[all]"
```

---

## 4. Development Commands

### Run tests

```bash
pytest tests/ -v
```

Current observed result during audit:

- `191` collected
- `144 passed`
- `1 xfailed`
- `46 xpassed`

### Fast test run

```bash
pytest tests/ -q
```

### Package install metadata

From `pyproject.toml`:

- Core deps: `numpy`, `pandas`, `polars`, `scipy`, `statsmodels`
- Optional ML deps: `torch`, `xgboost`, `lightgbm`, `optuna`, `shap`
- Optional broker dep: `MetaTrader5`

---

## 5. Project Structure

For a quick top-level navigation map, see `docs/REPOSITORY_STRUCTURE.md`.

```text
aphelion/
  core/                 implemented
  features/             implemented
  risk/
    sentinel/           implemented
    titan/              scaffold only
  backtest/             implemented
  intelligence/
    hydra/              implemented (partial validation)
    kronos/             scaffold only
    echo/               scaffold only
    forge/              scaffold only
    shadow/             scaffold only
  evolution/
    prometheus/         scaffold only
    cipher/             scaffold only
    meridian/           scaffold only
    zeus/               scaffold only
  flow/
    phantom/            scaffold only
    specter/            scaffold only
  macro/
    oracle/             scaffold only
    atlas/              scaffold only
    nexus/              scaffold only
    argus/              scaffold only
    herald/             scaffold only
  money/                scaffold only
  ares/                 scaffold only
  nemesis/
    pandora/            scaffold only
    leviathan/          scaffold only
    chronos/            scaffold only
    verdict/            scaffold only
  governance/
    olympus/            scaffold only
    council/            scaffold only
  tui/                  scaffold only
  aphelion_model/       scaffold only

tests/
  core/
  features/
  risk/
  integration/
```

---

## 6. Architecture Overview

### Tiered governance model (from config)

| Tier | Vote Weight | Role | Examples |
|---|---:|---|---|
| Sovereign | inf | System authority | Event bus, clock, registry |
| Council | 100 | Strategic governance | OLYMPUS, SENTINEL, ARES |
| Minister | 40 | Intelligence systems | HYDRA, PROMETHEUS, PHANTOM, NEMESIS |
| Commander | 10 | Execution and strategy modules | BACKTEST, VENOM, REAPER, APEX |
| Operator | 1 | Support/reporting | FUND |

### Runtime pipeline (implemented path)

```text
Tick/Bar -> DataLayer -> FeatureEngine -> Strategy -> SENTINEL Validator/Enforcer -> BrokerSim -> Portfolio -> Metrics
```

---

## 7. Phase-by-Phase Compliance Audit

This section compares the spec phase requirements (from the DOCX roadmap table) against current repository evidence.

### Audit legend

- `DONE`: implemented and validated in-repo
- `PARTIAL`: implemented in code, but acceptance criteria not yet proven in-repo
- `NOT STARTED`: package scaffold only or missing

| Phase | Spec Requirement (short) | Repo Evidence | Status |
|---|---|---|---|
| 1 | Data Foundation (`DATA`) | `aphelion/core/*`, `aphelion/features/*`, passing tests in `tests/core` + `tests/features` | DONE |
| 2 | Risk Layer (`SENTINEL`) | `risk/sentinel/core.py`, `validator.py`, `position_sizer.py`, `circuit_breaker.py`, integration tests | DONE |
| 3 | Backtesting Engine (`BACKTEST`) | `aphelion/backtest/engine.py`, `walk_forward.py`, `monte_carlo.py`, `metrics.py`, `analytics.py` | PARTIAL |
| 4 | HYDRA v1 (`HYDRA-TFT`) | `intelligence/hydra/tft.py`, `dataset.py`, `trainer.py`, `inference.py`, `strategy.py` | PARTIAL |
| 5 | Paper Trading | `aphelion/paper/session.py`, `execution/paper.py`, `execution/mt5.py`, `paper/feed.py`, `paper/ledger.py`, 39 tests | IN PROGRESS |
| 6 | TUI v1 | `aphelion/tui/` present but no implemented screens/app modules | NOT STARTED |
| 7 | HYDRA Full Ensemble | `hydra/lstm.py`, `cnn.py`, `moe.py`, `ensemble.py` exist, but no acceptance evidence | PARTIAL |
| 8 | PROMETHEUS v1 (NEAT) | `aphelion/evolution/prometheus/` scaffold only | NOT STARTED |
| 9 | Money Makers | `aphelion/money/` scaffold only | NOT STARTED |
| 10 | ARES Integration | `aphelion/ares/` scaffold only | NOT STARTED |
| 11 | Full PROMETHEUS | evolution stack scaffold only | NOT STARTED |
| 12 | Flow Intelligence | `aphelion/flow/*` scaffold only | NOT STARTED |
| 13 | Macro Intelligence | `aphelion/macro/*` scaffold only | NOT STARTED |
| 14 | Advanced ML (KRONOS/ECHO/FORGE/SHADOW) | subpackages exist but implementation missing | NOT STARTED |
| 15 | NEMESIS | `aphelion/nemesis/*` scaffold only | NOT STARTED |
| 16 | Full System Optimization | OLYMPUS/TITAN/GHOST/CIPHER/MERIDIAN not implemented | NOT STARTED |

### Notes on phase mismatch

- Spec acceptance criteria include performance proofs (for example Sharpe thresholds, OOS window requirements, paper-trading duration). Those proofs are not committed as reproducible artifacts yet.
- Therefore phases 3/4/7 are marked `PARTIAL` even though substantial code exists.

---

## 8. Module Status Matrix (20-Module Plan)

| Module | Code Status | Test Coverage in Repo |
|---|---|---|
| DATA | Implemented | Yes |
| SENTINEL | Implemented | Yes |
| BACKTEST | Implemented | Indirect (no dedicated backtest test suite yet) |
| HYDRA | Implemented (partial validation) | No dedicated HYDRA tests |
| OLYMPUS | Scaffold | No |
| ARES | Scaffold | No |
| PROMETHEUS | Scaffold | No |
| VENOM | Scaffold | No |
| REAPER | Scaffold | No |
| APEX | Scaffold | No |
| WRAITH | Scaffold | No |
| KRONOS | Scaffold | No |
| ECHO | Scaffold | No |
| FORGE | Scaffold | No |
| SHADOW | Scaffold | No |
| PHANTOM | Scaffold | No |
| SPECTER | Scaffold | No |
| NEMESIS | Scaffold | No |
| TITAN | Scaffold | No |
| GHOST | Scaffold | No |

---

## 9. Implemented Systems

## 9.1 Core Runtime (Phase 1)

Implemented files:

- `aphelion/core/event_bus.py`
- `aphelion/core/data_layer.py`
- `aphelion/core/clock.py`
- `aphelion/core/registry.py`
- `aphelion/core/config.py`

Implemented features include:

- Async pub/sub event bus with priority handling
- Tick and bar ingestion/aggregation
- Market session and lockout logic
- Component health registry
- Global system constants and immutable SENTINEL limits

## 9.2 Feature Engine (Phase 1)

Implemented files:

- `features/engine.py`
- `microstructure.py`
- `market_structure.py`
- `volume_profile.py`
- `vwap.py`
- `sessions.py`
- `mtf.py`
- `cointegration.py`

Outputs include microstructure, profile, session, MTF, and technical context used by both backtest and HYDRA.

## 9.3 Risk System / SENTINEL (Phase 2)

Implemented files:

- `risk/sentinel/core.py`
- `risk/sentinel/validator.py`
- `risk/sentinel/position_sizer.py`
- `risk/sentinel/circuit_breaker.py`
- `risk/sentinel/execution/enforcer.py`
- `risk/sentinel/monitor.py`

Highlights:

- Hardcoded limits are immutable (`core/config.py`)
- Trade proposals validated before execution
- Drawdown-triggered escalation via circuit breaker
- Runtime rejection telemetry in enforcer

## 9.4 Backtest Layer (Phase 3)

Implemented files:

- `backtest/engine.py`
- `backtest/broker_sim.py`
- `backtest/portfolio.py`
- `backtest/order.py`
- `backtest/metrics.py`
- `backtest/analytics.py`
- `backtest/monte_carlo.py`
- `backtest/walk_forward.py`

Capabilities present:

- Event-driven bar simulation
- Broker simulation with risk validation hooks
- Position and equity tracking
- Risk/performance metrics
- Monte Carlo resampling
- Walk-forward validation engine

Important caveat vs spec:

- Spec phase 3 explicitly asks for tick-level replay validation against MT5 history. Current engine architecture is primarily bar-driven and does not yet include committed proof artifacts for that acceptance criterion.

---

## 10. HYDRA (Phase 4+) Status

Implemented HYDRA files:

- `intelligence/hydra/tft.py`
- `intelligence/hydra/lstm.py`
- `intelligence/hydra/cnn.py`
- `intelligence/hydra/moe.py`
- `intelligence/hydra/ensemble.py`
- `intelligence/hydra/dataset.py`
- `intelligence/hydra/trainer.py`
- `intelligence/hydra/inference.py`
- `intelligence/hydra/strategy.py`

What is already in code:

- TFT model and sequence processing path
- Additional submodels (LSTM/CNN/MoE)
- Ensemble gate scaffolding
- Dataset prep and dataloader helpers
- Training loop and checkpoint logic
- Inference and strategy adapter

What is still missing for spec acceptance:

- Reproducible evidence of required OOS Sharpe thresholds
- Committed validation reports for 2-year OOS and minimum trade count
- Robust test coverage for HYDRA modules
- A fully healthy package export surface (`aphelion/intelligence/hydra/__init__.py` currently falls back to empty `__all__` if any import fails)

---

## 11. Testing Status

Current automated test footprint is focused on:

- Core infrastructure
- Features
- SENTINEL
- Integration pipeline

Current observed test summary:

```text
191 collected
144 passed
1 xfailed
46 xpassed
```

Current testing gaps:

- No dedicated test suite for `aphelion/backtest/*`
- No dedicated test suite for `aphelion/intelligence/hydra/*`
- No tests yet for phases 5+ modules because implementations are mostly absent

---

## 12. Gaps to Reach Full Spec

### Required to close Phase 3 formally

- Add reproducible benchmark for tick-level replay parity with MT5 historical data
- Add backtest-focused tests (engine, broker sim, portfolio, metrics, walk-forward)
- Commit acceptance artifacts for walk-forward quality checks

### Required to close Phase 4 formally

- Make `HYDRA` package exports robust (avoid silent empty-export fallback)
- Add unit/integration tests for HYDRA data, model, trainer, inference, strategy
- Produce and store reproducible OOS validation artifacts
- Enforce acceptance gates from spec (`Sharpe > 1.0`, minimum trade count)

### Required for Phases 5-16

- Implement module code in currently scaffold-only packages
- Add tests and operational glue between modules
- Add runtime orchestration and deployment pipelines

---

## 13. Suggested Near-Term Build Order

A practical path based on current code maturity:

1. Stabilize Phase 4 package ergonomics and tests (`hydra/__init__.py`, data/model/inference test suite)
2. Harden Phase 3 acceptance artifacts (backtest and walk-forward reproducibility)
3. Implement Phase 5 paper trading execution loop
4. Build minimal Phase 6 TUI screens for live risk/PnL visibility
5. Start Phase 8 PROMETHEUS v1 (NEAT-only) after Phases 3-6 are acceptance-validated

---

## 14. Contribution Workflow

### Branching and commits

- Use focused branches (`feature/<name>`, `fix/<name>`)
- Keep each commit scoped to one concern
- Include tests with behavior changes

### Before opening a PR

Run:

```bash
pytest tests/ -v
```

Include in PR description:

- What changed
- Why it changed
- How it was validated
- Any risk or migration notes

### Coding conventions

- Python 3.11+
- Prefer type hints
- Keep module boundaries clean
- Do not bypass SENTINEL risk constraints in any execution path

---

## 15. License and Confidentiality

This project is marked confidential/proprietary in the engineering spec. Treat strategy logic, model architecture, and risk logic as sensitive intellectual property.

---

## Appendix A: Phase Requirements Snapshot (from DOCX)

From the engineering roadmap section of the DOCX spec:

- Phase 1: Data Foundation, DATA module, features + tests
- Phase 2: Risk Layer, SENTINEL core rules and kill switches
- Phase 3: Backtesting engine with walk-forward validation
- Phase 4: HYDRA v1 (TFT-only acceptance criteria)
- Phase 5+: progressive integration through paper trading, TUI, full ensemble, evolution engine, macro/flow intelligence, NEMESIS, and full optimization

This README intentionally tracks the **actual repository state** against those requirements.

---

## Appendix B: Directory Completeness Snapshot

Non-`__init__.py` Python file counts observed during audit:

```text
core: 5
features: 8
risk/sentinel: 6
backtest: 8
intelligence/hydra: 9
intelligence/kronos: 0
intelligence/echo: 0
intelligence/forge: 0
intelligence/shadow: 0
evolution/prometheus: 0
evolution/cipher: 0
evolution/meridian: 0
evolution/zeus: 0
money: 0
ares: 0
flow/phantom: 0
flow/specter: 0
macro/oracle: 0
macro/atlas: 0
macro/nexus: 0
macro/argus: 0
macro/herald: 0
nemesis/pandora: 0
nemesis/leviathan: 0
nemesis/chronos: 0
nemesis/verdict: 0
governance/olympus: 0
governance/council: 0
risk/titan: 0
tui: 0
```

This snapshot is the main reason phases 5-16 are currently marked not started.

