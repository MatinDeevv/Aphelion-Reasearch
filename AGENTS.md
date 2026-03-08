# Repository Guidelines

## Project Structure & Module Organization
Core Python code lives in `aphelion/`, organized by domain:
- `core/` (event bus, clock, config, data layer, registry)
- `features/` (feature-engine pipeline and sub-engines)
- module namespaces such as `risk/`, `evolution/`, `macro/`, `flow/`, `intelligence/`, `governance/`, and `tui/`

Tests are under `tests/` and mirror runtime modules:
- `tests/core/`
- `tests/features/`
- `tests/integration/`

Runtime artifacts and datasets are stored in `data/`, `logs/`, `models/`, `genomes/`, and `aphelion_training_data/`.

## Build, Test, and Development Commands
- `python -m venv venv` then `venv\Scripts\activate` (Windows)  
  Create and activate a local virtual environment.
- `pip install -e .`  
  Install the package in editable mode.
- `pip install -e ".[dev]"`  
  Install pytest and async test tooling.
- `pytest tests/ -v`  
  Run the full test suite.
- `pytest tests/features/test_engine.py -v`  
  Run a focused test module during development.

## Coding Style & Naming Conventions
Use Python 3.11+ style with 4-space indentation and type hints (match existing signatures).  
Prefer clear dataclasses and small, composable engines. Keep module docstrings concise and descriptive.

Naming patterns:
- files/modules: `snake_case.py`
- classes: `PascalCase`
- functions/variables/tests: `snake_case`
- constants/enums: `UPPER_SNAKE_CASE`

No formatter/linter is enforced in config; follow existing repository style and keep imports deterministic.

## Testing Guidelines
Framework: `pytest` with `pytest-asyncio` (`asyncio_mode = auto`).  
Name tests as `test_<behavior>.py` and `test_<expected_outcome>()`.  
For new logic, include positive-path and edge-case assertions (validation failures, priority ordering, timeframe boundaries).  
Keep or improve current suite health before opening a PR.

## Commit & Pull Request Guidelines
History includes a `feat:`-prefixed commit; use that pattern consistently (`feat:`, `fix:`, `test:`, `docs:`) with imperative summaries.

PRs should include:
- concise problem/solution description
- linked issue (if available)
- test evidence (`pytest` output or targeted run)
- notes on risk-sensitive changes (especially `core/`, `risk/`, or `features/engine.py`)

## Security & Configuration Tips
Do not commit credentials, broker tokens, or local terminal paths.  
Treat `SENTINEL` limits in `aphelion/core/config.py` as immutable risk controls and avoid bypass logic in tests or runtime code.
