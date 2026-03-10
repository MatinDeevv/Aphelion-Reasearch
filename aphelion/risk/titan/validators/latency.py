"""
TITAN — Latency Validator
Ensures system pipeline latency meets requirements.
"""

from __future__ import annotations

from typing import Dict, List

from aphelion.risk.titan.gate import ValidationResult


class LatencyValidator:
    """Validates pipeline latency meets production requirements."""

    def validate(
        self,
        latency_buckets: Dict[str, float],  # {operation: p99_ms}
        max_total_p99_ms: float = 500.0,
        max_single_p99_ms: float = 200.0,
    ) -> List[ValidationResult]:
        results = []

        total_p99 = sum(latency_buckets.values())
        results.append(ValidationResult(
            check_name="total_pipeline_p99",
            passed=total_p99 <= max_total_p99_ms,
            actual_value=total_p99,
            threshold=max_total_p99_ms,
            message=f"Sum of all pipeline stage p99 latencies",
        ))

        for operation, p99 in latency_buckets.items():
            results.append(ValidationResult(
                check_name=f"latency_{operation}",
                passed=p99 <= max_single_p99_ms,
                actual_value=p99,
                threshold=max_single_p99_ms,
            ))

        return results
