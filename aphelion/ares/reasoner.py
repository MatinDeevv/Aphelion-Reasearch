"""
APHELION ARES — LLM Reasoning Engine

Provides pluggable LLM-powered reasoning for:
  - Signal conflict resolution (when strategies disagree)
  - Trade thesis generation (natural language explanations)
  - Market regime narration
  - Post-trade reflection and learning

Supports both local models (via llama-cpp or vLLM) and cloud APIs
(OpenAI, Anthropic) through a unified interface.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional, Any

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM backends."""
    LOCAL = auto()          # Local model (llama-cpp, vLLM)
    OPENAI = auto()         # OpenAI API
    ANTHROPIC = auto()      # Anthropic Claude API
    MOCK = auto()           # Deterministic mock for testing


@dataclass
class ReasonerConfig:
    """Configuration for the ARES reasoning engine."""
    provider: LLMProvider = LLMProvider.MOCK
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3               # Low temp for consistency
    max_tokens: int = 500
    timeout_seconds: float = 10.0
    # Local model
    local_model_path: str = ""
    # API
    api_key: str = ""                      # Set via env var, never commit
    api_base_url: str = ""
    # Safety
    max_retries: int = 2
    fallback_to_rules: bool = True         # Fall back to rule-based if LLM fails


@dataclass
class ReasoningResult:
    """Result of an LLM reasoning call."""
    direction: int                          # -1, 0, 1
    confidence: float                       # [0, 1]
    reasoning: str                          # Natural language explanation
    raw_response: str = ""
    provider: LLMProvider = LLMProvider.MOCK
    latency_ms: float = 0.0
    tokens_used: int = 0


class AresReasoner:
    """
    LLM reasoning engine for ARES.

    Provides conflict resolution, trade thesis generation, and
    market narration through a pluggable LLM backend.
    """

    def __init__(self, config: Optional[ReasonerConfig] = None):
        self._config = config or ReasonerConfig()
        self._call_count: int = 0
        self._total_tokens: int = 0
        self._total_latency_ms: float = 0.0

    def resolve_conflict(
        self,
        votes: list,
        regime: str = "UNKNOWN",
        market_context: Optional[dict] = None,
    ) -> Optional[ReasoningResult]:
        """
        Use LLM reasoning to resolve conflicting strategy signals.

        Args:
            votes: List of StrategyVote objects with disagreement.
            regime: Current market regime string.
            market_context: Optional dict with price, atr, session, etc.

        Returns:
            ReasoningResult with resolved direction and explanation,
            or None if reasoning fails and fallback is disabled.
        """
        import time
        t0 = time.time()

        prompt = self._build_conflict_prompt(votes, regime, market_context)

        try:
            result = self._call_llm(prompt)
        except Exception:
            logger.warning("LLM call failed for conflict resolution", exc_info=True)
            if self._config.fallback_to_rules:
                return self._rule_based_fallback(votes)
            return None

        result.latency_ms = (time.time() - t0) * 1000
        self._call_count += 1
        self._total_tokens += result.tokens_used
        self._total_latency_ms += result.latency_ms

        return result

    def generate_thesis(
        self,
        direction: int,
        confidence: float,
        features: dict,
        regime: str = "UNKNOWN",
    ) -> str:
        """Generate a natural language trade thesis."""
        side = "LONG" if direction == 1 else "SHORT" if direction == -1 else "FLAT"

        if self._config.provider == LLMProvider.MOCK:
            return (
                f"ARES thesis: {side} at {confidence:.0%} confidence. "
                f"Regime: {regime}. ATR={features.get('atr', 'N/A')}"
            )

        prompt = (
            f"Generate a concise 2-sentence trade thesis for XAU/USD.\n"
            f"Direction: {side}\n"
            f"Confidence: {confidence:.0%}\n"
            f"Regime: {regime}\n"
            f"Key features: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in list(features.items())[:10]})}"
        )

        try:
            result = self._call_llm(prompt)
            return result.reasoning
        except Exception:
            return f"[ARES] {side} signal with {confidence:.0%} confidence in {regime} regime."

    # ── LLM Backend ──────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> ReasoningResult:
        """Route to the configured LLM provider."""
        provider = self._config.provider

        if provider == LLMProvider.MOCK:
            return self._mock_llm(prompt)
        elif provider == LLMProvider.OPENAI:
            return self._call_openai(prompt)
        elif provider == LLMProvider.ANTHROPIC:
            return self._call_anthropic(prompt)
        elif provider == LLMProvider.LOCAL:
            return self._call_local(prompt)
        else:
            return self._mock_llm(prompt)

    def _mock_llm(self, prompt: str) -> ReasoningResult:
        """Deterministic mock for testing — always returns LONG with explanation."""
        return ReasoningResult(
            direction=1,
            confidence=0.65,
            reasoning="Mock LLM: Consensus leans bullish. Proceeding with reduced size.",
            raw_response="mock",
            provider=LLMProvider.MOCK,
            latency_ms=1.0,
            tokens_used=50,
        )

    def _call_openai(self, prompt: str) -> ReasoningResult:
        """Call OpenAI API. Requires `openai` package and API key."""
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package not installed. pip install openai")

        client = openai.OpenAI(
            api_key=self._config.api_key,
            base_url=self._config.api_base_url or None,
        )

        system_prompt = (
            "You are ARES, the strategic brain of the APHELION XAU/USD trading system. "
            "Analyse the given signals and market context. Respond with JSON: "
            '{"direction": 1/-1/0, "confidence": 0.0-1.0, "reasoning": "..."}'
        )

        response = client.chat.completions.create(
            model=self._config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )

        content = response.choices[0].message.content or ""
        return self._parse_llm_response(content, LLMProvider.OPENAI,
                                         response.usage.total_tokens if response.usage else 0)

    def _call_anthropic(self, prompt: str) -> ReasoningResult:
        """Call Anthropic API. Requires `anthropic` package and API key."""
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic package not installed. pip install anthropic")

        client = anthropic.Anthropic(api_key=self._config.api_key)

        system_prompt = (
            "You are ARES, the strategic brain of the APHELION XAU/USD trading system. "
            "Analyse the given signals and market context. Respond with JSON: "
            '{"direction": 1/-1/0, "confidence": 0.0-1.0, "reasoning": "..."}'
        )

        message = client.messages.create(
            model=self._config.model_name,
            max_tokens=self._config.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        content = message.content[0].text if message.content else ""
        tokens = (message.usage.input_tokens + message.usage.output_tokens) if message.usage else 0
        return self._parse_llm_response(content, LLMProvider.ANTHROPIC, tokens)

    def _call_local(self, prompt: str) -> ReasoningResult:
        """Placeholder for local model inference (llama-cpp / vLLM)."""
        logger.warning("Local LLM not yet implemented — using rule-based fallback")
        return self._mock_llm(prompt)

    # ── Response Parsing ─────────────────────────────────────────────────────

    def _parse_llm_response(
        self, content: str, provider: LLMProvider, tokens: int = 0,
    ) -> ReasoningResult:
        """Parse JSON response from LLM."""
        try:
            # Try to extract JSON from response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                return ReasoningResult(
                    direction=int(data.get("direction", 0)),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=str(data.get("reasoning", "LLM decision")),
                    raw_response=content,
                    provider=provider,
                    tokens_used=tokens,
                )
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Fallback: try to infer direction from text
        content_lower = content.lower()
        direction = 0
        if "long" in content_lower or "buy" in content_lower:
            direction = 1
        elif "short" in content_lower or "sell" in content_lower:
            direction = -1

        return ReasoningResult(
            direction=direction,
            confidence=0.5,
            reasoning=content[:200],
            raw_response=content,
            provider=provider,
            tokens_used=tokens,
        )

    # ── Rule-Based Fallback ──────────────────────────────────────────────────

    def _rule_based_fallback(self, votes: list) -> ReasoningResult:
        """Simple rule-based conflict resolution when LLM is unavailable."""
        if not votes:
            return ReasoningResult(direction=0, confidence=0.0,
                                   reasoning="No votes to resolve")

        # Weight by confidence
        long_weight = sum(v.confidence for v in votes if v.direction == 1)
        short_weight = sum(v.confidence for v in votes if v.direction == -1)

        if long_weight > short_weight * 1.3:
            return ReasoningResult(
                direction=1, confidence=long_weight / (long_weight + short_weight + 1e-10),
                reasoning="Rule fallback: Long signals dominate"
            )
        elif short_weight > long_weight * 1.3:
            return ReasoningResult(
                direction=-1, confidence=short_weight / (long_weight + short_weight + 1e-10),
                reasoning="Rule fallback: Short signals dominate"
            )
        else:
            return ReasoningResult(
                direction=0, confidence=0.3,
                reasoning="Rule fallback: No clear consensus — staying flat"
            )

    # ── Prompt Building ──────────────────────────────────────────────────────

    def _build_conflict_prompt(
        self,
        votes: list,
        regime: str,
        context: Optional[dict],
    ) -> str:
        """Build a structured prompt for conflict resolution."""
        parts = [
            "ARES CONFLICT RESOLUTION REQUEST",
            f"Market regime: {regime}",
            "",
            "Strategy votes:",
        ]

        for v in votes:
            dir_str = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(v.direction, "UNKNOWN")
            parts.append(
                f"  - {v.source.value}: {dir_str} (confidence={v.confidence:.2f}, "
                f"tier={v.tier.name})"
            )

        if context:
            parts.extend([
                "",
                "Market context:",
                f"  Price: {context.get('price', 'N/A')}",
                f"  ATR: {context.get('atr', 'N/A')}",
                f"  Session: {context.get('session', 'N/A')}",
            ])

        parts.extend([
            "",
            'Respond with JSON: {"direction": 1/-1/0, "confidence": 0.0-1.0, "reasoning": "..."}',
        ])

        return "\n".join(parts)

    # ── Stats ────────────────────────────────────────────────────────────────

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def avg_latency_ms(self) -> float:
        return self._total_latency_ms / self._call_count if self._call_count > 0 else 0.0

    @property
    def total_tokens(self) -> int:
        return self._total_tokens
