# Phase 5 Entry Checklist

Status as of 2026-03-08:

- [x] Full test suite green (`302 passed, 0 failed, 0 xfailed`)
- [x] Previous `xfail` removed (`tests/features/test_engine.py::test_session_reset`)
- [x] Warning cleanup completed (AMP deprecations + pytest-asyncio scope)
- [x] Verification report updated (`reports/VERIFICATION_REPORT.md`)
- [x] Reproducibility rerun completed in fresh terminal session

## Baseline Freeze (recommended before coding Phase 5)

Use one of the following:

1. Create a git tag:
   - `git tag -a phase4-hardened-2026-03-08 -m "Phase 4 hardened baseline"`
   - `git push origin phase4-hardened-2026-03-08`

2. Or create a release branch:
   - `git checkout -b release/phase4-hardened`
   - `git push -u origin release/phase4-hardened`

## Phase 5 Gate Criteria

Only proceed when all are true:

- `pytest tests/ -v` remains fully green after any pre-Phase-5 merge
- No xfail/xpass in core risk/backtest/hydra test suites
- SENTINEL hard limits unchanged and still tested
- Phase 5 objectives documented with measurable pass/fail metrics
