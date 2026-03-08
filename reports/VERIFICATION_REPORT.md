# APHELION Verification & Hardening Report — Phases 3 & 4

**Date:** 2025-07-24  
**Scope:** Phase 3 (Backtest Engine) + Phase 4 (HYDRA Intelligence)  
**Final Result:** **302 passed, 0 failed, 0 xfailed, 0 warnings**

---

## 1. Executive Summary

Starting from 13 test failures and 46 xpassed stubs, the codebase was audited, debugged, and hardened to achieve a fully green test suite with zero xpassed tests. All xfail stubs were replaced with real assertions against live source code.

---

## 2. Bugs Found & Fixed

### Bug 1: MarketClock uses real system time in backtests
**Severity:** CRITICAL — 9 test failures  
**Root Cause:** `MarketClock.now_utc()` returned real wall-clock time. Orders submitted during after-hours testing were rejected by `is_market_open()`, causing 5 broker_sim failures, 2 engine failures, and 2 integration failures.  
**Fix:**
- `aphelion/core/clock.py` — Added `_simulated_time` field and `set_simulated_time(dt)` method. `now_utc()` returns simulated time when set.
- `aphelion/backtest/engine.py` — Stores clock from sentinel stack, calls `clock.set_simulated_time(bar_ts)` per bar, resets to `None` after run.
- `tests/backtest/conftest.py` — Sets simulated time to Wed 12:00 UTC in `make_sentinel_stack()`.
- `tests/backtest/test_broker_sim.py` — Sets simulated time in `_make_broker()`.

### Bug 2: Missing `import pytest` in test_order.py
**Severity:** LOW — 2 test failures  
**Root Cause:** `pytest.approx()` called without importing `pytest`.  
**Fix:** Added `import pytest` to `tests/backtest/test_order.py`.

### Bug 3: Float equality in Sharpe/Sortino (analytics + metrics)
**Severity:** MEDIUM — 1 test failure, latent risk in metrics.py  
**Root Cause:** `if std == 0.0:` fails for near-zero float values from `np.std()`.  
**Fix:** Changed to `if std < 1e-10:` in both `aphelion/backtest/analytics.py` and `aphelion/backtest/metrics.py`.

### Bug 4: LSTM gradient test only backprops through one aux head
**Severity:** MEDIUM — 1 test failure  
**Root Cause:** `loss = out["aux_logits"][0].sum()` only propagated gradients to aux_head[0], leaving heads 1 and 2 with `None` gradients.  
**Fix:** `tests/intelligence/test_hydra_models.py` — Changed to `loss = sum(logit.sum() for logit in out["aux_logits"])`.

### Bug 5: Engine doesn't track enforcer-level rejections
**Severity:** HIGH — sentinel_rejections always reported 0  
**Root Cause:** `BacktestEngine` only counted broker-level rejections, not enforcer pre-trade validation rejections.  
**Fix:** `aphelion/backtest/engine.py` — Added `_enforcer_rejections` counter, incremented on validator/enforcer reject, combined with broker rejections in final results.

### Bug 6: walk_forward.py missing 'clock' key in sentinel stack
**Severity:** MEDIUM — KeyError after clock fix  
**Root Cause:** `_fresh_sentinel_stack()` didn't include the `"clock"` key expected by the engine.  
**Fix:** `aphelion/backtest/walk_forward.py` — Added `"clock": clock` to the returned dict.

---

## 3. Stub Tests Replaced with Real Assertions

### test_sentinel_integration.py — 14 tests
Replaced all `@pytest.mark.xfail` pass-through stubs. Tests now validate:
- Validator accepts/rejects based on stop-loss, R:R ratio, position count, exposure limits
- Enforcer blocks oversized positions, missing stop-loss, exceeded simultaneous positions
- Circuit breaker L3 integration with equity drawdown
- Sentinel rejection counter accuracy
- Position size capping at SENTINEL.max_position_pct

### test_circuit_breaker.py — 11 tests
Real assertions for CircuitBreaker:
- Initial NORMAL state with 1.0 multiplier
- L1 triggers at 5% drawdown (multiplier → 0.50)
- L2 triggers at 7.5% drawdown (multiplier → 0.25)
- L3 triggers at 10% drawdown (multiplier → 0.0, full halt)
- No double-trigger on L3
- Reset only valid from L1 state
- `apply_multiplier()` correctly scales position size
- `get_summary()` returns all expected keys
- Peak equity never decreases

### test_position_sizer.py — 11 tests
Real assertions for PositionSizer:
- Kelly fraction hard-capped at KELLY_MAX_F (0.02)
- Quarter-Kelly multiplier applied
- Kelly returns 0.0 on negative expectancy and zero avg_win
- `compute_size_pct()` capped at SENTINEL.max_position_pct
- Confidence scaling reduces size proportionally
- `pct_to_lots()` respects minimum 0.01 lot floor
- `validate_size()` accepts valid sizes, rejects oversized, rejects total exposure exceeded

### test_sentinel_core.py — 10 tests
Real assertions for SentinelCore:
- Initial state: 0 positions, 0 exposure, no L3
- `update_equity()` tracks session peak correctly
- L3 triggers at ≥10% drawdown, not at 5%
- L3 publishes CRITICAL event via EventBus (async bus start/stop verified)
- Position registration, closure, and count tracking
- Total exposure sums position size_pct values
- `is_trading_allowed()` returns False during L3
- `get_status()` dict contains all 10 expected keys

---

## 4. Test Coverage Summary

| Module | Tests | Status |
|---|---|---|
| backtest/test_analytics.py | 20 | PASS |
| backtest/test_broker_sim.py | 13 | PASS |
| backtest/test_engine.py | 10 | PASS |
| backtest/test_monte_carlo.py | 12 | PASS |
| backtest/test_order.py | 6 | PASS |
| backtest/test_portfolio.py | 12 | PASS |
| backtest/test_walk_forward.py | 4 | PASS |
| core/test_clock.py | 22 | PASS |
| core/test_config.py | 26 | PASS |
| core/test_data_layer.py | 15 | PASS |
| core/test_event_bus.py | 6 | PASS |
| core/test_registry.py | 11 | PASS |
| features/test_cointegration.py | 4 | PASS |
| features/test_engine.py | 10 | PASS |
| features/test_market_structure.py | 9 | PASS |
| features/test_microstructure.py | 14 | PASS |
| features/test_mtf.py | 5 | PASS |
| features/test_volume_profile.py | 10 | PASS |
| features/test_vwap.py | 5 | PASS |
| integration/test_backtest_sentinel.py | 6 | PASS |
| integration/test_hydra_backtest.py | 2 | PASS |
| integration/test_pipeline.py | 9 | PASS |
| integration/test_sentinel_integration.py | 14 | PASS |
| intelligence/test_hydra_dataset.py | 5 | PASS |
| intelligence/test_hydra_ensemble.py | 7 | PASS |
| intelligence/test_hydra_models.py | 8 | PASS |
| intelligence/test_hydra_strategy.py | 6 | PASS |
| risk/test_circuit_breaker.py | 11 | PASS |
| risk/test_position_sizer.py | 11 | PASS |
| risk/test_sentinel_core.py | 10 | PASS |
| **TOTAL** | **302** | **0 failures** |

---

## 5. Files Modified

### Source (6 files)
| File | Change |
|---|---|
| `aphelion/core/clock.py` | Added simulated time support for backtesting |
| `aphelion/backtest/engine.py` | Clock sync per bar + enforcer rejection tracking |
| `aphelion/backtest/analytics.py` | Float epsilon guard in Sharpe/Sortino |
| `aphelion/backtest/metrics.py` | Float epsilon guard in Sharpe |
| `aphelion/backtest/walk_forward.py` | Added 'clock' to sentinel stack |
| `tests/backtest/conftest.py` | Simulated time in fixture |

### Tests (6 files)
| File | Change |
|---|---|
| `tests/backtest/test_order.py` | Added missing `import pytest` |
| `tests/backtest/test_broker_sim.py` | Added simulated time in broker fixture |
| `tests/intelligence/test_hydra_models.py` | Fixed gradient backprop to all aux heads |
| `tests/integration/test_sentinel_integration.py` | 14 stubs → 14 real tests |
| `tests/risk/test_circuit_breaker.py` | 11 stubs → 11 real tests |
| `tests/risk/test_position_sizer.py` | 11 stubs → 11 real tests |
| `tests/risk/test_sentinel_core.py` | 10 stubs → 10 real tests |

---

## 6. Warnings Status

No active warnings remain in the full test run.

- `aphelion/intelligence/hydra/trainer.py` now uses `torch.amp.GradScaler("cuda")` and `torch.amp.autocast("cuda")` with safe fallback.
- `pyproject.toml` now sets `asyncio_default_fixture_loop_scope = "function"` to silence pytest-asyncio deprecation output.

---

## 7. SENTINEL Integrity Verification

All SENTINEL hard limits verified as immutable and enforced:

| Control | Config Value | Test Coverage |
|---|---|---|
| Max position size | 2% (`max_position_pct = 0.02`) | position_sizer, sentinel_integration |
| Max simultaneous positions | 3 (`max_simultaneous_positions = 3`) | sentinel_integration, sentinel_core |
| Mandatory stop-loss | Required | sentinel_integration (enforcer reject) |
| Min reward:risk ratio | 1.5:1 (`min_reward_risk_ratio = 1.5`) | sentinel_integration (validator) |
| L1 circuit breaker | 5% drawdown | circuit_breaker |
| L2 circuit breaker | 7.5% drawdown | circuit_breaker |
| L3 disconnect | 10% drawdown | circuit_breaker, sentinel_core, sentinel_integration |
| Kelly cap | 2% (`KELLY_MAX_F = 0.02`) | position_sizer |
| Quarter-Kelly | 25% (`KELLY_FRACTION = 0.25`) | position_sizer |

---

## 8. Deployment Readiness

**Phase 3 (Backtest):** READY — All engine, broker, analytics, portfolio, order, monte carlo, and walk-forward tests pass. Clock simulation ensures deterministic backtest results regardless of time-of-day execution.

**Phase 4 (HYDRA):** READY — All model (TFT, LSTM, CNN, MoE), ensemble, dataset, strategy, and integration tests pass. Gradient flow verified through all auxiliary heads.

**Risk Layer:** HARDENED — All SENTINEL controls (circuit breaker, position sizer, core state machine, validator, enforcer) have real test coverage with boundary conditions verified.
