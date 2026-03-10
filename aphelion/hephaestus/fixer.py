"""
HEPHAESTUS — Fixer

LLM-powered code fixer.  Takes broken code + error message and produces
a minimally-changed fix.
"""

from __future__ import annotations

import logging
from typing import Optional

from aphelion.hephaestus.llm_client import HephaestusLLMClient
from aphelion.hephaestus.models import ForgedStrategy
from aphelion.hephaestus.prompts import FIXER_SYSTEM_PROMPT, FIXER_USER_TEMPLATE

logger = logging.getLogger(__name__)


class HephaestusFixer:
    """LLM-powered code fixer.

    Takes broken code + error message → fixed code (minimal changes).
    """

    def __init__(self, llm_client: Optional[HephaestusLLMClient] = None) -> None:
        self._llm = llm_client or HephaestusLLMClient()

    def fix(self, forged: ForgedStrategy, error: str) -> Optional[str]:
        """Attempt to fix broken code.

        Returns the fixed code string, or *None* if the LLM produces nothing.
        """
        user_prompt = FIXER_USER_TEMPLATE.format(
            code=forged.python_code,
            error=error,
        )

        raw = self._llm.call_fixer(FIXER_SYSTEM_PROMPT, user_prompt)
        if not raw:
            logger.warning("Fixer: LLM returned empty response")
            return None

        return self._clean_fix(raw)

    # ── Internals ────────────────────────────────────────────────────────

    @staticmethod
    def _clean_fix(raw: str) -> Optional[str]:
        """Strip markdown fences and validate the fix looks like Python."""
        cleaned = raw.strip()
        for fence in ("```python", "```py", "```"):
            if cleaned.startswith(fence):
                cleaned = cleaned[len(fence):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Minimal sanity — must still contain a class
        if "class " not in cleaned:
            logger.warning("Fixer: fixed code missing class definition")
            return None

        return cleaned
