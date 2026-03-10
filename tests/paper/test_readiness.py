"""Tests for Paper v2 — LiveReadinessGate and LatencyProfile."""

import pytest

from aphelion.paper.readiness import LatencyProfile, LiveReadinessGate, ReadinessCheck


# ── LatencyProfile ──────────────────────────────────────────────────────────

class TestLatencyProfile:

    def test_empty_profile(self):
        lp = LatencyProfile()
        assert lp.total_samples == 0
        assert lp.operations == []

    def test_record_and_get_bucket(self):
        lp = LatencyProfile()
        for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            lp.record("tick_ingest", ms)
        bucket = lp.get_bucket("tick_ingest")
        assert bucket.samples == 10
        assert bucket.p50_ms > 0
        assert bucket.p95_ms >= bucket.p50_ms
        assert bucket.p99_ms >= bucket.p95_ms
        assert bucket.max_ms == 100.0

    def test_get_bucket_empty(self):
        lp = LatencyProfile()
        bucket = lp.get_bucket("nonexistent")
        assert bucket.samples == 0
        assert bucket.p50_ms == 0.0

    def test_total_samples(self):
        lp = LatencyProfile()
        for _ in range(5):
            lp.record("op_a", 10.0)
        for _ in range(3):
            lp.record("op_b", 20.0)
        assert lp.total_samples == 8

    def test_operations_list(self):
        lp = LatencyProfile()
        lp.record("tick_ingest", 5.0)
        lp.record("model_inference", 15.0)
        assert set(lp.operations) == {"tick_ingest", "model_inference"}

    def test_get_all_buckets(self):
        lp = LatencyProfile()
        lp.record("a", 1.0)
        lp.record("b", 2.0)
        buckets = lp.get_all_buckets()
        assert "a" in buckets
        assert "b" in buckets

    def test_total_pipeline_p99(self):
        lp = LatencyProfile()
        for _ in range(50):
            lp.record("tick", 10.0)
            lp.record("model", 20.0)
        total = lp.get_total_pipeline_p99()
        assert total > 0

    def test_reset(self):
        lp = LatencyProfile()
        lp.record("tick", 5.0)
        lp.reset()
        assert lp.total_samples == 0


# ── LiveReadinessGate ───────────────────────────────────────────────────────

class TestLiveReadinessGate:

    def _passing_kwargs(self):
        return dict(
            total_trades=200,
            win_rate=0.60,
            sharpe=1.5,
            profit_factor=1.5,
            max_drawdown_pct=0.03,
            sentinel_l2=False,
            sentinel_l3=False,
            pipeline_p99_ms=100,
            paper_trading_days=15,
            models_loaded=True,
            feed_connected=True,
        )

    def test_all_pass(self):
        gate = LiveReadinessGate()
        checks = gate.evaluate(**self._passing_kwargs())
        assert gate.is_ready(checks) is True

    def test_fails_min_trades(self):
        gate = LiveReadinessGate(min_trades=100)
        kw = self._passing_kwargs()
        kw["total_trades"] = 50
        checks = gate.evaluate(**kw)
        assert gate.is_ready(checks) is False
        failed = [c for c in checks if not c.passed]
        assert any(c.name == "min_trades" for c in failed)

    def test_fails_win_rate(self):
        gate = LiveReadinessGate(min_win_rate=0.55)
        kw = self._passing_kwargs()
        kw["win_rate"] = 0.45
        checks = gate.evaluate(**kw)
        assert gate.is_ready(checks) is False

    def test_fails_drawdown(self):
        gate = LiveReadinessGate(max_drawdown_pct=0.05)
        kw = self._passing_kwargs()
        kw["max_drawdown_pct"] = 0.08
        checks = gate.evaluate(**kw)
        assert gate.is_ready(checks) is False

    def test_fails_latency(self):
        gate = LiveReadinessGate(max_pipeline_p99_ms=200)
        kw = self._passing_kwargs()
        kw["pipeline_p99_ms"] = 500
        checks = gate.evaluate(**kw)
        assert gate.is_ready(checks) is False

    def test_sentinel_l2_blocks(self):
        gate = LiveReadinessGate()
        kw = self._passing_kwargs()
        kw["sentinel_l2"] = True
        checks = gate.evaluate(**kw)
        assert gate.is_ready(checks) is False

    def test_summary_output(self):
        gate = LiveReadinessGate()
        checks = gate.evaluate(**self._passing_kwargs())
        text = gate.summary(checks)
        assert "Live Readiness Gate" in text
        assert "READY" in text

    def test_paper_duration_gate(self):
        gate = LiveReadinessGate(min_paper_days=10)
        kw = self._passing_kwargs()
        kw["paper_trading_days"] = 3
        checks = gate.evaluate(**kw)
        assert gate.is_ready(checks) is False
