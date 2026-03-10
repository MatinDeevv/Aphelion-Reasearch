"""
Tests for hephaestus.codegen — code generation and cleaning.
"""

from __future__ import annotations

import pytest

from aphelion.hephaestus.codegen import HephaestusCodegen, _sanitize_class_name
from aphelion.hephaestus.llm_client import HephaestusLLMClient
from aphelion.hephaestus.models import InputType, StrategySpec


# ─── _sanitize_class_name ────────────────────────────────────────────────────


class TestSanitizeClassName:
    def test_simple_name(self):
        assert _sanitize_class_name("ema crossover") == "EmaCrossoverVoter"

    def test_special_characters(self):
        assert _sanitize_class_name("RSI-reversal_14") == "RsiReversal14Voter"

    def test_leading_digit(self):
        result = _sanitize_class_name("14_period_ema")
        assert result.startswith("Heph")
        assert result.endswith("Voter")
        assert result[0].isalpha()

    def test_empty_name(self):
        assert _sanitize_class_name("") == "ForgedStrategyVoter"

    def test_all_special_chars(self):
        assert _sanitize_class_name("---") == "ForgedStrategyVoter"

    def test_single_word(self):
        assert _sanitize_class_name("MACD") == "MacdVoter"

    def test_already_pascal(self):
        assert _sanitize_class_name("BollingerBands") == "BollingerbandsVoter"


# ─── Codegen._clean_code ────────────────────────────────────────────────────


class TestCleanCode:

    def test_strips_markdown_fences(self):
        raw = "```python\nclass Foo:\n    def vote(self): pass\n```"
        result = HephaestusCodegen._clean_code(raw)
        assert "```" not in result
        assert "class Foo" in result

    def test_strips_py_fence(self):
        raw = "```py\nclass Foo:\n    def vote(self): pass\n```"
        result = HephaestusCodegen._clean_code(raw)
        assert "class Foo" in result

    def test_no_class_returns_empty(self):
        raw = "def vote(self): pass"
        result = HephaestusCodegen._clean_code(raw)
        assert result == ""

    def test_no_vote_returns_empty(self):
        raw = "class Foo:\n    def compute(self): pass"
        result = HephaestusCodegen._clean_code(raw)
        assert result == ""

    def test_valid_code_passes_through(self):
        raw = "class MyVoter:\n    def vote(self, bars, ctx): pass"
        result = HephaestusCodegen._clean_code(raw)
        assert result == raw


# ─── HephaestusCodegen.compute_code_hash ─────────────────────────────────────


class TestCodeHash:
    def test_deterministic(self):
        code = "class Foo:\n    pass"
        h1 = HephaestusCodegen.compute_code_hash(code)
        h2 = HephaestusCodegen.compute_code_hash(code)
        assert h1 == h2
        assert len(h1) == 16

    def test_different_code_different_hash(self):
        h1 = HephaestusCodegen.compute_code_hash("class A: pass")
        h2 = HephaestusCodegen.compute_code_hash("class B: pass")
        assert h1 != h2


# ─── Codegen integration (stub LLM) ─────────────────────────────────────────


class TestCodegenIntegration:

    def _make_spec(self) -> StrategySpec:
        return StrategySpec(
            name="EMA Cross",
            source_type=InputType.PINE_SCRIPT,
            description="Buy on EMA crossover",
            entry_long_conditions=["EMA(8) crosses above EMA(21)"],
            entry_short_conditions=["EMA(8) crosses below EMA(21)"],
            indicators_used=["EMA(8)", "EMA(21)"],
            lookback_bars=30,
            confidence=0.9,
        )

    def test_generate_returns_none_without_llm(self):
        codegen = HephaestusCodegen(llm_client=HephaestusLLMClient())
        result = codegen.generate(self._make_spec())
        assert result is None  # stub LLM returns empty → None

    def test_build_system_prompt_includes_examples(self):
        codegen = HephaestusCodegen()
        prompt = codegen._build_system_prompt()
        assert "EXAMPLE 1" in prompt
        assert "EMACrossoverVoter" in prompt
