"""
TUI screens package.

All screens are importable via their individual modules:
  from aphelion.tui.screens.dashboard import build_dashboard_layout
  from aphelion.tui.screens.equity import build_equity_panel
  etc.

Avoid eager imports here to prevent circular import with app.py.
"""

__all__ = [
    "build_dashboard_layout",
    "build_equity_panel",
    "build_log_panel",
    "build_header",
    "build_hydra_panel",
    "build_performance_panel",
    "build_positions_panel",
    "build_sentinel_panel",
    "build_system_health_panel",
    "build_sola_panel",
    "build_omega_panel",
    "build_evolution_panel",
    "build_replay_panel",
]


def __getattr__(name: str):
    """Lazy import to avoid circular dependency with app.py."""
    _map = {
        "build_dashboard_layout": ".dashboard",
        "build_equity_panel": ".equity",
        "build_log_panel": ".event_log",
        "build_header": ".header",
        "build_hydra_panel": ".hydra_panel",
        "build_performance_panel": ".performance",
        "build_positions_panel": ".positions",
        "build_sentinel_panel": ".sentinel_panel",
        "build_system_health_panel": ".system_health",
        "build_sola_panel": ".sola_panel",
        "build_omega_panel": ".omega_panel",
        "build_evolution_panel": ".evolution_panel",
        "build_replay_panel": ".replay_panel",
    }
    if name in _map:
        import importlib
        mod = importlib.import_module(_map[name], __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
