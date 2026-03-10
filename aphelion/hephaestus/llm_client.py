"""
HEPHAESTUS — LLM Client

Thin wrapper around the Anthropic / OpenAI-compatible API used by the
Hephaestus forge.  When no API key is configured the client degrades to a
deterministic *stub* mode that returns empty strings — this allows the full
module to be imported and unit-tested without a live LLM backend.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Optional anthropic SDK import
try:
    import anthropic  # type: ignore[import-untyped]
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """LLM call configuration."""

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.1
    api_key: str = ""  # Read from env if empty


# Token budgets per task type
TOKEN_BUDGETS: dict[str, int] = {
    "parse": 1000,
    "generate": 3000,
    "fix": 2000,
    "diagnose": 1000,
}


# ─── Client ──────────────────────────────────────────────────────────────────


class HephaestusLLMClient:
    """Claude API wrapper for HEPHAESTUS.

    Provides three high-level methods that correspond to the three LLM
    call-sites in the forge pipeline: ``call_parser``, ``call_codegen``,
    and ``call_fixer``.
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self._config = config or LLMConfig()
        self._api_key = self._config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client: Optional[object] = None
        self._total_tokens: int = 0
        self._total_calls: int = 0

        if _HAS_ANTHROPIC and self._api_key:
            try:
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except Exception:
                logger.warning("HEPHAESTUS LLM client: anthropic SDK init failed — stub mode")

    # ── Public helpers ───────────────────────────────────────────────────

    @property
    def is_live(self) -> bool:
        """True when a real LLM backend is configured."""
        return self._client is not None

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_calls(self) -> int:
        return self._total_calls

    # ── Core call ────────────────────────────────────────────────────────

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Send a prompt to the LLM and return the text response.

        Falls back to an empty string when no backend is available.
        """
        self._total_calls += 1
        max_tok = max_tokens or self._config.max_tokens
        temp = temperature if temperature is not None else self._config.temperature

        if not self.is_live:
            logger.debug("HEPHAESTUS LLM stub call (no backend)")
            return ""

        try:
            response = self._client.messages.create(  # type: ignore[union-attr]
                model=self._config.model,
                max_tokens=max_tok,
                temperature=temp,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = response.content[0].text  # type: ignore[index]
            usage = getattr(response, "usage", None)
            if usage:
                self._total_tokens += getattr(usage, "input_tokens", 0) + getattr(
                    usage, "output_tokens", 0
                )
            return text
        except Exception as exc:
            logger.error("HEPHAESTUS LLM call failed: %s", exc)
            return ""

    # ── Task-specific wrappers ───────────────────────────────────────────

    def call_parser(self, system_prompt: str, user_prompt: str) -> str:
        """LLM call for the parser stage (compact JSON extraction)."""
        return self.call(system_prompt, user_prompt, max_tokens=TOKEN_BUDGETS["parse"])

    def call_codegen(self, system_prompt: str, user_prompt: str) -> str:
        """LLM call for code generation (full Python class)."""
        return self.call(system_prompt, user_prompt, max_tokens=TOKEN_BUDGETS["generate"])

    def call_fixer(self, system_prompt: str, user_prompt: str) -> str:
        """LLM call for the fixer (targeted error repair)."""
        return self.call(system_prompt, user_prompt, max_tokens=TOKEN_BUDGETS["fix"])

    # ── JSON extraction helper ───────────────────────────────────────────

    @staticmethod
    def extract_json(text: str) -> Optional[dict]:
        """Best-effort JSON extraction from LLM output.

        Handles common issues: markdown fences, trailing commas, etc.
        """
        if not text:
            return None

        # Strip markdown code fences
        cleaned = text.strip()
        for fence in ("```json", "```", "```python"):
            if cleaned.startswith(fence):
                cleaned = cleaned[len(fence):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Attempt direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try extracting first JSON object from freeform text
        brace_depth = 0
        start: Optional[int] = None
        for i, ch in enumerate(cleaned):
            if ch == "{":
                if brace_depth == 0:
                    start = i
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0 and start is not None:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        pass
                    start = None

        return None
