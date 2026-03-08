# Repository Structure Guide

This layout keeps source code, scripts, docs, and generated reports separated for easier navigation.

## Top-Level Conventions

- `aphelion/` — all runtime source modules
- `tests/` — all test suites
- `scripts/` — standalone helper scripts (for example, training or migration utilities)
- `docs/` — specifications, checklists, architecture notes
- `reports/` — generated verification artifacts and run outputs
- `data/`, `models/`, `logs/`, `genomes/` — runtime datasets and artifacts

## Current Organized Locations

- Engineering spec: `docs/specs/APHELION_Engineering_Spec_v1.docx`
- Parsed spec text: `docs/specs/spec_text.txt`
- Phase checklist: `docs/checklists/PHASE5_ENTRY_CHECKLIST.md`
- Phase 5 objectives: `docs/checklists/PHASE5_OBJECTIVES.md`
- Verification report: `reports/VERIFICATION_REPORT.md`
- Historical pytest outputs: `reports/tests/`
- Historical HYDRA training logs: `reports/training/`
- HYDRA local training helper: `scripts/train_hydra.py`

## Phase 5 — Paper Trading Modules

- **Paper executor** (virtual fills): `aphelion/risk/sentinel/execution/paper.py`
- **MT5 broker connection**: `aphelion/risk/sentinel/execution/mt5.py`
- **Data feed abstraction**: `aphelion/paper/feed.py` (Live, Replay, Simulated)
- **Audit ledger**: `aphelion/paper/ledger.py` (JSON-Lines journal)
- **Session orchestrator**: `aphelion/paper/session.py` (async main loop)
- **Tests**: `tests/paper/test_paper.py` (39 tests)
