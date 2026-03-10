"""
HEPHAESTUS — Parser

Extracts the mathematical intent of an indicator from raw source code
(Pine Script, Python, pseudocode, or plain English) and returns a
structured ``StrategySpec``.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from aphelion.hephaestus.llm_client import HephaestusLLMClient
from aphelion.hephaestus.models import InputType, StrategySpec
from aphelion.hephaestus.pine_script.syntax_map import PINE_KEYWORDS
from aphelion.hephaestus.prompts import PARSER_SYSTEM_PROMPT, PARSER_USER_TEMPLATE

logger = logging.getLogger(__name__)


# ─── Input-type detection heuristics ─────────────────────────────────────────


def detect_input_type(source_code: str) -> InputType:
    """Classify source code without an LLM call (fast heuristics).

    - Contains Pine Script keywords → PINE_SCRIPT
    - Contains ``def `` and ``import`` → PYTHON
    - Contains ``when price`` / ``if candle`` → PSEUDOCODE
    - Otherwise → PLAIN_ENGLISH
    """
    text = source_code.strip()
    if not text:
        return InputType.UNKNOWN

    lower = text.lower()

    # Pine Script detection (most specific first)
    pine_hits = sum(1 for kw in PINE_KEYWORDS if kw.lower() in lower)
    if pine_hits >= 2:
        return InputType.PINE_SCRIPT

    # Python detection
    has_def = bool(re.search(r"\bdef\s+\w+", text))
    has_import = bool(re.search(r"^(import |from \w)", text, re.MULTILINE))
    if has_def and has_import:
        return InputType.PYTHON
    if has_def or has_import:
        # Single signal — could still be Python
        return InputType.PYTHON

    # Pseudocode detection
    pseudo_patterns = [
        r"\bwhen\s+price\b",
        r"\bif\s+candle\b",
        r"\bthen\s+buy\b",
        r"\bthen\s+sell\b",
        r"\bentry\s+when\b",
        r"\bexit\s+when\b",
    ]
    pseudo_hits = sum(1 for pat in pseudo_patterns if re.search(pat, lower))
    if pseudo_hits >= 1:
        return InputType.PSEUDOCODE

    # If it looks like sentences → plain English
    words = lower.split()
    if len(words) >= 5:
        return InputType.PLAIN_ENGLISH

    return InputType.UNKNOWN


# ─── Parser validator ────────────────────────────────────────────────────────


class ParserValidator:
    """Validates that a ``StrategySpec`` is coherent and complete."""

    MINIMUM_CONFIDENCE: float = 0.50
    MINIMUM_LOOKBACK: int = 5
    MAXIMUM_LOOKBACK: int = 5000

    def validate(self, spec: StrategySpec) -> tuple[bool, list[str]]:
        issues: list[str] = []

        if not spec.entry_long_conditions and not spec.entry_short_conditions:
            issues.append(
                "No entry conditions found — indicator may be display-only"
            )

        if spec.lookback_bars < self.MINIMUM_LOOKBACK:
            issues.append(
                f"Lookback {spec.lookback_bars} too small — likely extraction error"
            )

        if spec.lookback_bars > self.MAXIMUM_LOOKBACK:
            issues.append(f"Lookback {spec.lookback_bars} suspiciously large")

        if spec.confidence < self.MINIMUM_CONFIDENCE:
            issues.append(
                f"Parser confidence {spec.confidence:.2f} too low to proceed safely"
            )

        if not (0.5 <= spec.suggested_r_ratio <= 20.0):
            issues.append(
                f"R ratio {spec.suggested_r_ratio} outside reasonable range [0.5, 20]"
            )

        return len(issues) == 0, issues


# ─── Main parser ─────────────────────────────────────────────────────────────


class HephaestusParser:
    """Extracts mathematical logic from indicator source code using an LLM.

    Supports Pine Script v4/v5, Python, pseudocode, and plain English.
    Output: ``StrategySpec`` — a structured, language-agnostic description.
    """

    def __init__(self, llm_client: Optional[HephaestusLLMClient] = None) -> None:
        self._llm = llm_client or HephaestusLLMClient()
        self._validator = ParserValidator()

    # ── Public API ───────────────────────────────────────────────────────

    def parse(self, source_code: str) -> Optional[StrategySpec]:
        """Parse source code into a ``StrategySpec``.

        Returns *None* if parsing fails or the spec is invalid.
        """
        source_type = detect_input_type(source_code)
        user_prompt = PARSER_USER_TEMPLATE.format(
            source_type=source_type.value,
            source_code=source_code,
        )

        raw = self._llm.call_parser(PARSER_SYSTEM_PROMPT, user_prompt)
        if not raw:
            logger.warning("Parser: LLM returned empty response")
            return None

        data = HephaestusLLMClient.extract_json(raw)
        if data is None:
            logger.warning("Parser: could not extract JSON from LLM response")
            return None

        return self._dict_to_spec(data, source_type)

    def detect_input_type(self, source_code: str) -> InputType:
        """Thin wrapper around the module-level heuristic."""
        return detect_input_type(source_code)

    # ── Internal ─────────────────────────────────────────────────────────

    def _dict_to_spec(self, data: dict, source_type: InputType) -> Optional[StrategySpec]:
        """Convert a parsed JSON dict to a ``StrategySpec``."""
        try:
            spec = StrategySpec(
                name=str(data.get("name", "Unnamed")),
                source_type=source_type,
                description=str(data.get("description", "")),
                entry_long_conditions=list(data.get("entry_long_conditions", [])),
                entry_short_conditions=list(data.get("entry_short_conditions", [])),
                exit_conditions=list(data.get("exit_conditions", [])),
                indicators_used=list(data.get("indicators_used", [])),
                lookback_bars=int(data.get("lookback_bars", 50)),
                timeframe=str(data.get("timeframe", "M5")),
                parameters={
                    str(k): float(v)
                    for k, v in data.get("parameters", {}).items()
                },
                parameter_ranges={
                    str(k): tuple(v)
                    for k, v in data.get("parameter_ranges", {}).items()
                },
                suggested_stop_loss=str(
                    data.get("suggested_stop_loss", "2 × ATR below entry")
                ),
                suggested_take_profit=str(
                    data.get("suggested_take_profit", "3 × ATR above entry")
                ),
                suggested_r_ratio=float(data.get("suggested_r_ratio", 1.5)),
                complexity_score=float(data.get("complexity_score", 0.5)),
                confidence=float(data.get("confidence", 0.0)),
                warnings=list(data.get("warnings", [])),
            )
            return spec
        except (TypeError, ValueError, KeyError) as exc:
            logger.warning("Parser: failed to build StrategySpec — %s", exc)
            return None

    def validate_spec(self, spec: StrategySpec) -> tuple[bool, list[str]]:
        """Expose the validator for external use."""
        return self._validator.validate(spec)
