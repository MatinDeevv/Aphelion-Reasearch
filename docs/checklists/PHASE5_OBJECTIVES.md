# Phase 5 — Paper Trading: Objectives & Acceptance Criteria

**Start Date:** _(fill when session first runs)_  
**Target Duration:** 2 weeks of live paper trading  
**Spec Reference:** Section 11 — "Paper trading live for 2 weeks. All SENTINEL rules respected. Zero errors."

---

## Deliverables

| # | Deliverable | File(s) | Status |
|---|-------------|---------|--------|
| 1 | Paper trading fill simulator | `aphelion/risk/sentinel/execution/paper.py` | NOT STARTED |
| 2 | MT5 broker connection wrapper | `aphelion/risk/sentinel/execution/mt5.py` | NOT STARTED |
| 3 | Data feed abstraction (live / replay) | `aphelion/paper/feed.py` | NOT STARTED |
| 4 | Paper trade ledger & audit journal | `aphelion/paper/ledger.py` | NOT STARTED |
| 5 | Paper trading session orchestrator | `aphelion/paper/session.py` | NOT STARTED |
| 6 | Phase 5 test suite | `tests/paper/` | NOT STARTED |

---

## Acceptance Criteria (all must pass to exit Phase 5)

### Hard Gates (must be TRUE)

- [ ] Paper trading session ran continuously for **≥ 2 weeks** (logged timestamps prove uptime).
- [ ] **Zero runtime errors** during the 2-week window (no unhandled exceptions, no crashes).
- [ ] **All SENTINEL rules respected** — no trade bypassed validation, no position exceeded 2%, no more than 3 simultaneous positions, all trades had mandatory SL.
- [ ] **L3 circuit breaker** fired correctly when 10% drawdown was reached (if it occurred), halting all trading and closing positions.
- [ ] **Friday lockout** closed all positions 30 min before market close every Friday.
- [ ] **News lockout** blocked trading during configured pre/post event windows.
- [ ] ExecutionEnforcer approved/rejected every order with logged reason.

### Soft Metrics (tracked, not gating)

- [ ] Total trades executed in paper mode.
- [ ] Win rate, average R-multiple, Sharpe ratio of paper trades.
- [ ] HYDRA signal quality — % of signals filtered by confidence/agreement thresholds.
- [ ] Average latency from bar close → order fill (should be < 500ms).
- [ ] SENTINEL rejection rate and top rejection reasons.
- [ ] Maximum drawdown during paper period.

---

## Architecture

```
Market Data (MT5 / Replay)
        │
        ▼
   DataLayer + BarAggregator
        │
        ▼
   FeatureEngine (60+ features)
        │
        ▼
   HydraInference → HydraSignal
        │
        ▼
   HydraStrategy → Order
        │
        ▼
   ExecutionEnforcer (CB + Validator)
        │
        ▼
   PaperExecutor (virtual fills)
        │
        ▼
   Portfolio + SentinelCore (state tracking)
        │
        ▼
   Ledger (JSON-L audit log)
```

---

## Risk Controls (inherited from Phase 2-4, enforced in paper mode)

| Control | Value | Enforcement Point |
|---------|-------|-------------------|
| Max position size | 2% of account | TradeValidator + PositionSizer |
| Max simultaneous positions | 3 | TradeValidator |
| Mandatory stop-loss | Every trade | TradeValidator |
| Min risk:reward | 1.5:1 | TradeValidator |
| L1 drawdown (5%) | Size × 0.5 | CircuitBreaker |
| L2 drawdown (7.5%) | Size × 0.25 | CircuitBreaker |
| L3 drawdown (10%) | Full halt + disconnect | SentinelCore |
| Pre-news lockout | 5 min | MarketClock |
| Post-news lockout | 2 min | MarketClock |
| Friday close | 30 min before 21:00 UTC | MarketClock + Session |
