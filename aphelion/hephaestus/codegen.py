"""
HEPHAESTUS — Code Generator

Takes a ``StrategySpec`` and produces a production-grade Python class
that implements ``BaseARESVoter``.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

from aphelion.hephaestus.llm_client import HephaestusLLMClient
from aphelion.hephaestus.models import ForgedStrategy, StrategySpec
from aphelion.hephaestus.pine_script.examples import EXAMPLES
from aphelion.hephaestus.prompts import CODEGEN_SYSTEM_PROMPT, CODEGEN_USER_TEMPLATE

logger = logging.getLogger(__name__)


def _sanitize_class_name(name: str) -> str:
    """Convert an arbitrary name to a valid PascalCase Python class name."""
    # Strip non-alphanumeric, split on separators
    parts = re.split(r"[^a-zA-Z0-9]+", name)
    pascal = "".join(p.capitalize() for p in parts if p)
    if not pascal:
        pascal = "ForgedStrategy"
    # Ensure it starts with a letter
    if pascal[0].isdigit():
        pascal = "Heph" + pascal
    return pascal + "Voter"


class HephaestusCodegen:
    """Generates Python voter classes from ``StrategySpec`` via LLM.

    Includes few-shot examples to ground the LLM's output format.
    """

    def __init__(self, llm_client: Optional[HephaestusLLMClient] = None) -> None:
        self._llm = llm_client or HephaestusLLMClient()

    # ── Public API ───────────────────────────────────────────────────────

    def generate(self, spec: StrategySpec) -> Optional[ForgedStrategy]:
        """Generate a ``ForgedStrategy`` from a validated spec."""
        class_name = _sanitize_class_name(spec.name)

        # Build the few-shot enriched system prompt
        system = self._build_system_prompt()

        user = CODEGEN_USER_TEMPLATE.format(
            name=spec.name,
            description=spec.description,
            entry_long_conditions="\n".join(
                f"  - {c}" for c in spec.entry_long_conditions
            ),
            entry_short_conditions="\n".join(
                f"  - {c}" for c in spec.entry_short_conditions
            ),
            exit_conditions="\n".join(
                f"  - {c}" for c in spec.exit_conditions
            ),
            indicators_used=", ".join(spec.indicators_used),
            lookback_bars=spec.lookback_bars,
            parameters=spec.parameters,
            suggested_stop_loss=spec.suggested_stop_loss,
            suggested_take_profit=spec.suggested_take_profit,
            class_name=class_name,
        )

        raw = self._llm.call_codegen(system, user)
        if not raw:
            logger.warning("Codegen: LLM returned empty response")
            return None

        code = self._clean_code(raw)
        if not code:
            return None

        return ForgedStrategy(
            spec=spec,
            python_code=code,
            class_name=class_name,
        )

    # ── Internal ─────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Inject few-shot examples into the system prompt."""
        examples_text = ""
        for i, ex in enumerate(EXAMPLES, 1):
            examples_text += (
                f"\n--- EXAMPLE {i} ---\n"
                f"Pine Script:\n{ex['pine_source']}\n"
                f"Spec summary: {ex['spec_summary']}\n"
                f"Generated class:\n{ex['generated_class']}\n"
            )
        return CODEGEN_SYSTEM_PROMPT + "\n\nFEW-SHOT EXAMPLES:\n" + examples_text

    @staticmethod
    def _clean_code(raw: str) -> str:
        """Strip markdown fences and trailing commentary from LLM output."""
        cleaned = raw.strip()
        # Remove markdown fences
        for fence in ("```python", "```py", "```"):
            if cleaned.startswith(fence):
                cleaned = cleaned[len(fence):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Basic sanity — must contain 'class' and 'def vote'
        if "class " not in cleaned or "def vote" not in cleaned:
            logger.warning("Codegen: generated code missing class or vote method")
            return ""
        return cleaned

    @staticmethod
    def compute_code_hash(code: str) -> str:
        """SHA-256 digest of generated code (for deduplication)."""
        return hashlib.sha256(code.encode()).hexdigest()[:16]
