"""
Tests for hephaestus.parser — input-type detection, spec parsing, validation.
"""

from __future__ import annotations

import pytest

from aphelion.hephaestus.models import InputType, StrategySpec
from aphelion.hephaestus.parser import (
    HephaestusParser,
    ParserValidator,
    detect_input_type,
)
from aphelion.hephaestus.llm_client import HephaestusLLMClient


# ─── detect_input_type ───────────────────────────────────────────────────────


class TestDetectInputType:
    """Heuristic input-type classification."""

    def test_pine_script_v5(self):
        code = '//@version=5\nindicator("My Strat")\nrsi = ta.rsi(close, 14)'
        assert detect_input_type(code) == InputType.PINE_SCRIPT

    def test_pine_script_strategy(self):
        code = '//@version=5\nstrategy("EMA Cross")\nta.ema(close, 8)'
        assert detect_input_type(code) == InputType.PINE_SCRIPT

    def test_python_code(self):
        code = "import numpy as np\ndef compute_signal(bars):\n    return 0"
        assert detect_input_type(code) == InputType.PYTHON

    def test_python_def_only(self):
        code = "def my_strategy():\n    pass"
        assert detect_input_type(code) == InputType.PYTHON

    def test_pseudocode(self):
        code = "when price crosses above EMA(200) then buy"
        assert detect_input_type(code) == InputType.PSEUDOCODE

    def test_plain_english(self):
        code = "Buy when the price is above the 200 period moving average and RSI is below 60"
        assert detect_input_type(code) == InputType.PLAIN_ENGLISH

    def test_empty_input(self):
        assert detect_input_type("") == InputType.UNKNOWN

    def test_short_gibberish(self):
        assert detect_input_type("xyz") == InputType.UNKNOWN

    def test_pine_with_study(self):
        code = 'study("old style")\nta.sma(close, 14)\nta.crossover(a, b)'
        assert detect_input_type(code) == InputType.PINE_SCRIPT


# ─── ParserValidator ─────────────────────────────────────────────────────────


class TestParserValidator:
    """StrategySpec validation rules."""

    def _make_spec(self, **overrides) -> StrategySpec:
        defaults = dict(
            name="Test",
            source_type=InputType.PINE_SCRIPT,
            description="A test strategy",
            entry_long_conditions=["price > EMA(200)"],
            entry_short_conditions=["price < EMA(200)"],
            lookback_bars=50,
            confidence=0.8,
            suggested_r_ratio=1.5,
        )
        defaults.update(overrides)
        return StrategySpec(**defaults)

    def test_valid_spec_passes(self):
        v = ParserValidator()
        ok, issues = v.validate(self._make_spec())
        assert ok is True
        assert issues == []

    def test_no_entry_conditions(self):
        v = ParserValidator()
        ok, issues = v.validate(
            self._make_spec(entry_long_conditions=[], entry_short_conditions=[])
        )
        assert ok is False
        assert any("No entry conditions" in i for i in issues)

    def test_lookback_too_small(self):
        v = ParserValidator()
        ok, issues = v.validate(self._make_spec(lookback_bars=2))
        assert ok is False
        assert any("too small" in i for i in issues)

    def test_lookback_too_large(self):
        v = ParserValidator()
        ok, issues = v.validate(self._make_spec(lookback_bars=6000))
        assert ok is False
        assert any("suspiciously large" in i for i in issues)

    def test_low_confidence(self):
        v = ParserValidator()
        ok, issues = v.validate(self._make_spec(confidence=0.2))
        assert ok is False
        assert any("too low" in i for i in issues)

    def test_bad_r_ratio_low(self):
        v = ParserValidator()
        ok, issues = v.validate(self._make_spec(suggested_r_ratio=0.1))
        assert ok is False
        assert any("outside reasonable range" in i for i in issues)

    def test_bad_r_ratio_high(self):
        v = ParserValidator()
        ok, issues = v.validate(self._make_spec(suggested_r_ratio=25.0))
        assert ok is False

    def test_minimum_thresholds(self):
        """Boundary: exactly at MINIMUM_CONFIDENCE and MINIMUM_LOOKBACK."""
        v = ParserValidator()
        ok, _ = v.validate(self._make_spec(confidence=0.50, lookback_bars=5))
        assert ok is True


# ─── HephaestusParser (stub LLM — no API key) ───────────────────────────────


class TestHephaestusParser:
    """Parser integration (stub mode: LLM returns empty → parse returns None)."""

    def test_parse_returns_none_without_llm(self):
        parser = HephaestusParser(llm_client=HephaestusLLMClient())
        result = parser.parse("some code")
        assert result is None

    def test_detect_input_type_method(self):
        parser = HephaestusParser()
        assert parser.detect_input_type('//@version=5\nindicator("x")\nta.ema(close,8)') == InputType.PINE_SCRIPT

    def test_validate_spec_delegates(self):
        parser = HephaestusParser()
        spec = StrategySpec(
            name="T",
            source_type=InputType.PYTHON,
            description="test",
            entry_long_conditions=["buy"],
            confidence=0.8,
            lookback_bars=10,
        )
        ok, issues = parser.validate_spec(spec)
        assert ok is True

    def test_dict_to_spec_valid(self):
        parser = HephaestusParser()
        data = {
            "name": "EMA Cross",
            "description": "Buy on EMA cross",
            "entry_long_conditions": ["EMA8 > EMA21"],
            "entry_short_conditions": [],
            "exit_conditions": [],
            "indicators_used": ["EMA(8)", "EMA(21)"],
            "lookback_bars": 30,
            "timeframe": "M5",
            "parameters": {"fast": 8, "slow": 21},
            "parameter_ranges": {"fast": [5, 15]},
            "suggested_r_ratio": 1.5,
            "complexity_score": 0.3,
            "confidence": 0.9,
            "warnings": [],
        }
        spec = parser._dict_to_spec(data, InputType.PINE_SCRIPT)
        assert spec is not None
        assert spec.name == "EMA Cross"
        assert spec.lookback_bars == 30
        assert spec.parameters["fast"] == 8.0

    def test_dict_to_spec_missing_fields_uses_defaults(self):
        parser = HephaestusParser()
        data = {"name": "Minimal", "description": "bare minimum"}
        spec = parser._dict_to_spec(data, InputType.UNKNOWN)
        assert spec is not None
        assert spec.lookback_bars == 50  # default
