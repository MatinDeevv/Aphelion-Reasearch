"""
HEPHAESTUS — Data Models

All dataclasses and enumerations used across the Hephaestus autonomous
strategy forge.  Models are pure data containers with no external
dependencies beyond the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ─── Enumerations ────────────────────────────────────────────────────────────


class InputType(Enum):
    """Classification of source code submitted to the forge."""
    PINE_SCRIPT = "PINE_SCRIPT"
    PYTHON = "PYTHON"
    PSEUDOCODE = "PSEUDOCODE"
    PLAIN_ENGLISH = "PLAIN_ENGLISH"
    UNKNOWN = "UNKNOWN"


class ForgeStatus(Enum):
    """Lifecycle stages of a forged strategy."""
    PENDING = "PENDING"
    PARSING = "PARSING"
    GENERATING = "GENERATING"
    TESTING = "TESTING"
    VALIDATING = "VALIDATING"
    DEPLOYED = "DEPLOYED"
    SHADOW = "SHADOW"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


class ShadowEvaluation(Enum):
    """Outcome of shadow-mode evaluation."""
    PROMOTE = "PROMOTE"
    CONTINUE_SHADOW = "CONTINUE_SHADOW"
    REJECT = "REJECT"


# ─── Parser output ───────────────────────────────────────────────────────────


@dataclass
class StrategySpec:
    """Structured description of an indicator extracted by the LLM parser.

    This is the intermediate representation between raw source and Python code.
    """

    # Identity
    name: str
    source_type: InputType
    description: str

    # Signal logic
    entry_long_conditions: list[str] = field(default_factory=list)
    entry_short_conditions: list[str] = field(default_factory=list)
    exit_conditions: list[str] = field(default_factory=list)

    # Indicator computations
    indicators_used: list[str] = field(default_factory=list)
    lookback_bars: int = 50
    timeframe: str = "M5"

    # Parameters (will become configurable)
    parameters: dict[str, float] = field(default_factory=dict)
    parameter_ranges: dict[str, tuple] = field(default_factory=dict)

    # Risk
    suggested_stop_loss: str = "2 × ATR below entry"
    suggested_take_profit: str = "3 × ATR above entry"
    suggested_r_ratio: float = 1.5

    # Quality metadata
    complexity_score: float = 0.5
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)


# ─── Code-generation output ─────────────────────────────────────────────────


@dataclass
class ForgedStrategy:
    """The LLM-generated Python strategy code, ready for sandbox execution."""

    spec: StrategySpec
    python_code: str
    class_name: str
    version: int = 1
    generation_prompt_tokens: int = 0
    fix_history: list[str] = field(default_factory=list)


# ─── Sandbox ─────────────────────────────────────────────────────────────────


@dataclass
class ASTCheckResult:
    """Result of a static AST safety check."""
    safe: bool
    reason: str = ""


@dataclass
class SandboxResult:
    """Output from a sandboxed execution of generated code."""
    success: bool
    output: str = ""
    error_message: str = ""
    execution_ms: float = 0.0


@dataclass
class TestResult:
    """Aggregated unit-test result from the sandbox."""
    all_passed: bool
    passed: int = 0
    failed: int = 0
    failure_summary: str = ""


# ─── Validation ──────────────────────────────────────────────────────────────


@dataclass
class CorrelationReport:
    """Correlation of a new voter with existing ARES voters."""
    max_correlation: float = 0.0
    most_correlated_voter: str = ""
    adds_diversity: bool = True


@dataclass
class ValidationReport:
    """Complete validation results for a forged strategy."""

    # Syntax & runtime
    syntax_valid: bool = False
    runtime_errors: list[str] = field(default_factory=list)
    unit_tests_passed: int = 0
    unit_tests_total: int = 0

    # Backtest results
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    avg_r_multiple: float = 0.0

    # Walk-forward
    wf_folds_passed: int = 0
    wf_folds_total: int = 0
    wf_median_sharpe: float = 0.0
    wf_sharpe_variance: float = 0.0

    # Monte Carlo
    mc_5th_pct_sharpe: float = 0.0
    mc_95th_pct_max_dd: float = 0.0

    # TITAN gate
    titan_passed: bool = False
    titan_failures: list[str] = field(default_factory=list)

    # Correlation with existing voters
    max_correlation_with_existing: float = 0.0
    correlated_voter: str = ""
    adds_diversity: bool = True

    # Overall
    passed: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


# ─── Forge result ────────────────────────────────────────────────────────────


@dataclass
class ForgeResult:
    """Final result of the HEPHAESTUS forge process."""

    strategy_id: str
    status: ForgeStatus

    spec: Optional[StrategySpec] = None
    forged: Optional[ForgedStrategy] = None
    validation: Optional[ValidationReport] = None

    # Timing
    submitted_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    completed_at: Optional[datetime] = None
    total_seconds: float = 0.0

    # Agent stats
    parse_attempts: int = 0
    generation_attempts: int = 0
    fix_attempts: int = 0
    total_llm_calls: int = 0
    total_tokens_used: int = 0

    # If deployed
    ares_voter_id: Optional[str] = None
    deployment_mode: str = "SHADOW"


# ─── Rejection report ────────────────────────────────────────────────────────


@dataclass
class RejectionReport:
    """Detailed rejection report with diagnosis and recommendations."""

    strategy_id: str
    strategy_name: str
    source_snippet: str = ""

    failed_at: str = ""  # PARSE / GENERATE / SANDBOX / BACKTEST / WF / MC / TITAN / CORRELATION
    reasons: list[str] = field(default_factory=list)

    # Metrics (even for failed strategies — useful for learning)
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0

    suggestions: list[str] = field(default_factory=list)


# ─── Vote (ARES-compatible) ──────────────────────────────────────────────────


@dataclass
class Vote:
    """A single strategy vote — compatible with ARES ``StrategyVote``."""
    direction: int  # 1=BUY, -1=SELL, 0=FLAT
    confidence: float  # [0, 1]
    reason: str
    metadata: dict = field(default_factory=dict)


# ─── TITAN thresholds for HEPHAESTUS strategies ─────────────────────────────


HEPHAESTUS_TITAN_REQUIREMENTS: dict[str, float] = {
    # Backtest
    "min_sharpe_ratio": 1.3,
    "min_win_rate": 0.50,
    "max_drawdown": 0.15,
    "min_profit_factor": 1.2,
    "min_trades_for_significance": 150,
    # Walk-forward
    "wf_min_folds_passing": 7,
    "wf_min_median_sharpe": 1.0,
    "wf_max_sharpe_variance": 1.2,
    # Monte Carlo
    "mc_5th_percentile_sharpe": 0.7,
    "mc_95th_percentile_max_dd": 0.30,
    # Correlation
    "max_correlation_with_existing": 0.75,
}
