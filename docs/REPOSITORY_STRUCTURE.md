# Repository Structure Guide

This layout keeps runtime code, developer tooling, documentation, and generated artifacts separate so the repo is easier to scan and maintain.

## Top-Level Conventions

- `aphelion/` - runtime source modules
- `tests/` - automated test suites
- `scripts/` - helper scripts and maintenance utilities
- `docs/` - guides, specs, architecture notes, and checklists
- `reports/` - generated verification artifacts and saved outputs
- `data/`, `models/`, `logs/`, `genomes/` - runtime datasets and artifacts

Keep the repository root focused on package metadata and primary entrypoints such as `aphelion.py`, `aphelion_data.py`, `runall.py`, and `run_paper.py`.

## Documentation Layout

- `docs/guides/` - operator and developer walkthroughs
- `docs/architecture/` - repository and data-layout notes
- `docs/specs/` - engineering specs and long-form design docs
- `docs/checklists/` - phase objectives and readiness checklists

## Current Organized Locations

- Engineering spec v1: `docs/specs/APHELION_Engineering_Spec_v1.docx`
- Engineering spec v3: `docs/specs/APHELION_ENGINEERING_SPEC_v3.md`
- Phase 23 TUI spec: `docs/specs/APHELION_PHASE23_TUI_SUPREME.md`
- Parsed spec text: `docs/specs/spec_text.txt`
- How-to guide: `docs/guides/HOW_TO_USE.md`
- Data layout note: `docs/architecture/DATA_REORGANIZATION.md`
- Paper trading guide: `docs/PAPER_TRADING_SETUP.md`
- Phase checklist: `docs/checklists/PHASE5_ENTRY_CHECKLIST.md`
- Phase 5 objectives: `docs/checklists/PHASE5_OBJECTIVES.md`
- Verification report: `reports/VERIFICATION_REPORT.md`
- Historical pytest outputs: `reports/tests/`
- Historical HYDRA training logs: `reports/training/`
- HYDRA training helper: `scripts/train_hydra.py`
- Maintenance rewrite helpers: `scripts/maintenance/`

## Phase 5 Paper Trading Modules

- Paper executor (virtual fills): `aphelion/risk/sentinel/execution/paper.py`
- MT5 broker connection: `aphelion/risk/sentinel/execution/mt5.py`
- Data feed abstraction: `aphelion/paper/feed.py`
- Audit ledger: `aphelion/paper/ledger.py`
- Session orchestrator: `aphelion/paper/session.py`
- Tests: `tests/paper/test_paper.py`
