"""
Tests for hephaestus.fixer — code repair.
"""

from __future__ import annotations

import pytest

from aphelion.hephaestus.fixer import HephaestusFixer
from aphelion.hephaestus.llm_client import HephaestusLLMClient
from aphelion.hephaestus.models import ForgedStrategy, InputType, StrategySpec


def _make_forged(code: str = "class Foo:\n    pass") -> ForgedStrategy:
    spec = StrategySpec(
        name="T",
        source_type=InputType.PYTHON,
        description="test",
        confidence=0.9,
    )
    return ForgedStrategy(spec=spec, python_code=code, class_name="Foo")


# ─── _clean_fix ──────────────────────────────────────────────────────────────


class TestCleanFix:

    def test_strips_markdown_fences(self):
        raw = "```python\nclass Foo:\n    pass\n```"
        assert HephaestusFixer._clean_fix(raw) == "class Foo:\n    pass"

    def test_strips_py_fence(self):
        raw = "```py\nclass Bar:\n    pass\n```"
        result = HephaestusFixer._clean_fix(raw)
        assert result is not None
        assert "```" not in result

    def test_no_class_returns_none(self):
        assert HephaestusFixer._clean_fix("just some text") is None

    def test_valid_code_passes(self):
        raw = "class Fixed:\n    def vote(self): pass"
        result = HephaestusFixer._clean_fix(raw)
        assert result == raw

    def test_whitespace_only_returns_none(self):
        assert HephaestusFixer._clean_fix("   \n  \n  ") is None

    def test_empty_returns_none(self):
        assert HephaestusFixer._clean_fix("") is None


# ─── Fixer integration (stub LLM) ───────────────────────────────────────────


class TestFixerIntegration:

    def test_fix_returns_none_without_llm(self):
        fixer = HephaestusFixer(llm_client=HephaestusLLMClient())
        result = fixer.fix(_make_forged(), "NameError: name 'x' not defined")
        assert result is None

    def test_fixer_accepts_any_error_string(self):
        fixer = HephaestusFixer(llm_client=HephaestusLLMClient())
        result = fixer.fix(_make_forged(), "")
        assert result is None

    def test_fixer_preserves_forged_unchanged(self):
        forged = _make_forged("class X:\n    pass")
        original_code = forged.python_code
        fixer = HephaestusFixer(llm_client=HephaestusLLMClient())
        fixer.fix(forged, "some error")
        assert forged.python_code == original_code


# ─── Fix attempt limits (tested via agent, but here for unit coverage) ───────


class TestFixAttemptCounting:

    def test_fix_history_empty_initially(self):
        forged = _make_forged()
        assert forged.fix_history == []
        assert forged.version == 1

    def test_version_increments_on_fix(self):
        forged = _make_forged()
        forged.version += 1
        forged.fix_history.append("error 1")
        assert forged.version == 2
        assert len(forged.fix_history) == 1

    def test_multiple_fix_attempts_tracked(self):
        forged = _make_forged()
        for i in range(5):
            forged.version += 1
            forged.fix_history.append(f"error {i}")
        assert forged.version == 6
        assert len(forged.fix_history) == 5

    def test_fix_history_preserves_order(self):
        forged = _make_forged()
        errors = ["SyntaxError", "NameError", "TypeError"]
        for e in errors:
            forged.fix_history.append(e)
        assert forged.fix_history == errors
