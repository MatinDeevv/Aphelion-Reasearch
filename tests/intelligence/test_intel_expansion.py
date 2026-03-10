"""Tests for Intelligence expansion — KRONOS analytics, ECHO matcher, FORGE scheduler, SHADOW scenarios."""

import pytest
from datetime import datetime, timedelta, timezone

from aphelion.intelligence.kronos.analytics import KronosAnalytics, TradeRecord, PerformanceSnapshot
from aphelion.intelligence.kronos.report_generator import KronosReportGenerator
from aphelion.intelligence.echo.matcher import PatternMatcher
from aphelion.intelligence.forge.scheduler import ForgeScheduler, ScheduledJob, create_default_schedule
from aphelion.intelligence.forge.parameter_space import ParameterSpace, create_default_parameter_space
from aphelion.intelligence.shadow.regime_simulator import AdvancedRegimeSimulator, RegimeScenario
from aphelion.intelligence.shadow.stress_scenarios import StressScenarioLibrary


# ── KronosAnalytics ─────────────────────────────────────────────────────────

class TestKronosAnalytics:

    def _add_trades(self, ka, n=20):
        for i in range(n):
            won = i % 3 != 0
            pnl = 50.0 if won else -30.0
            ka.add_trade(TradeRecord(
                trade_id=f"T{i}",
                direction="BUY" if i % 2 == 0 else "SELL",
                entry_price=1900.0 + i,
                exit_price=1905.0 + i if won else 1895.0 + i,
                entry_time=datetime(2024, 1, 1 + i, 10, 0, tzinfo=timezone.utc),
                exit_time=datetime(2024, 1, 1 + i, 11, 0, tzinfo=timezone.utc),
                pnl=pnl,
                pnl_pct=pnl / 10000.0,
                regime="TRENDING" if i < 10 else "RANGING",
                session="LONDON" if i % 2 == 0 else "NEWYORK",
            ))

    def test_compute_snapshot_empty(self):
        ka = KronosAnalytics()
        snap = ka.compute_snapshot()
        assert snap.win_rate == 0.0

    def test_compute_snapshot_basic(self):
        ka = KronosAnalytics()
        self._add_trades(ka, 20)
        snap = ka.compute_snapshot()
        assert isinstance(snap, PerformanceSnapshot)
        assert 0.0 <= snap.win_rate <= 1.0
        assert ka.total_trades == 20

    def test_compute_by_regime(self):
        ka = KronosAnalytics()
        self._add_trades(ka, 20)
        by_regime = ka.compute_by_regime()
        assert "TRENDING" in by_regime
        assert "RANGING" in by_regime

    def test_compute_by_session(self):
        ka = KronosAnalytics()
        self._add_trades(ka, 20)
        by_session = ka.compute_by_session()
        assert "LONDON" in by_session
        assert "NEWYORK" in by_session


# ── KronosReportGenerator ──────────────────────────────────────────────────

class TestKronosReportGenerator:

    def _make_analytics(self):
        ka = KronosAnalytics()
        for i in range(20):
            won = i % 3 != 0
            pnl = 50.0 if won else -30.0
            ka.add_trade(TradeRecord(
                trade_id=f"T{i}",
                direction="BUY",
                entry_price=1900.0,
                exit_price=1905.0 if won else 1895.0,
                entry_time=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                exit_time=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
                pnl=pnl,
                pnl_pct=pnl / 10000.0,
                regime="TRENDING",
                session="LONDON",
            ))
        return ka

    def test_text_report(self):
        gen = KronosReportGenerator(self._make_analytics())
        text = gen.generate_text_report()
        assert "KRONOS" in text
        assert "Win Rate" in text

    def test_json_report(self):
        gen = KronosReportGenerator(self._make_analytics())
        data = gen.generate_json_report()
        assert "overall" in data
        assert data["overall"]["total_trades"] == 20


# ── PatternMatcher ──────────────────────────────────────────────────────────

class TestPatternMatcher:

    def test_construction(self):
        pm = PatternMatcher()
        assert pm is not None

    def test_construction_with_params(self):
        pm = PatternMatcher(top_k=3, min_similarity=0.80)
        assert pm._top_k == 3
        assert pm._min_similarity == 0.80


# ── ForgeScheduler ──────────────────────────────────────────────────────────

class TestForgeScheduler:

    def test_create_default_schedule(self):
        scheduler = create_default_schedule()
        assert scheduler.total_jobs >= 3

    def test_register_job(self):
        s = ForgeScheduler()
        s.register_job(ScheduledJob(
            job_id="TEST", target="test_module",
            schedule_interval=timedelta(hours=1),
        ))
        assert s.total_jobs == 1

    def test_check_due_jobs(self):
        s = ForgeScheduler()
        # Job with next_run in the past → should be due
        job = ScheduledJob(
            job_id="DUE", target="test",
            schedule_interval=timedelta(hours=1),
            next_run=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        s.register_job(job)
        due = s.check_due_jobs()
        assert len(due) >= 1

    def test_start_and_complete_job(self):
        s = ForgeScheduler()
        job = ScheduledJob(
            job_id="J1", target="test",
            schedule_interval=timedelta(hours=1),
        )
        s.register_job(job)
        s.start_job("J1")
        assert job.running is True
        s.complete_job("J1")
        assert job.running is False
        assert job.last_run is not None


# ── ParameterSpace ──────────────────────────────────────────────────────────

class TestParameterSpace:

    def test_default_has_modules(self):
        ps = create_default_parameter_space()
        assert len(ps.module_names) >= 4

    def test_alpha_params_exist(self):
        ps = create_default_parameter_space()
        alpha = ps.get_module_params("ALPHA")
        assert len(alpha) > 0

    def test_omega_params_exist(self):
        ps = create_default_parameter_space()
        omega = ps.get_module_params("OMEGA")
        assert len(omega) > 0

    def test_sentinel_params_exist(self):
        ps = create_default_parameter_space()
        sentinel = ps.get_module_params("SENTINEL")
        assert len(sentinel) > 0

    def test_ares_params_exist(self):
        ps = create_default_parameter_space()
        ares = ps.get_module_params("ARES")
        assert len(ares) > 0

    def test_unknown_module_empty(self):
        ps = create_default_parameter_space()
        assert ps.get_module_params("NONEXISTENT") == []

    def test_total_dimensions(self):
        ps = create_default_parameter_space()
        assert ps.total_dimensions >= 10


# ── AdvancedRegimeSimulator ─────────────────────────────────────────────────

class TestAdvancedRegimeSimulator:

    def test_generate_trending(self):
        sim = AdvancedRegimeSimulator()
        scenario = RegimeScenario(name="trend", regime="TRENDING", n_bars=100)
        bars = sim.generate_scenario(scenario)
        assert len(bars) == 100

    def test_generate_ranging(self):
        sim = AdvancedRegimeSimulator()
        scenario = RegimeScenario(name="range", regime="RANGING", n_bars=100)
        bars = sim.generate_scenario(scenario)
        assert len(bars) == 100

    def test_generate_volatile(self):
        sim = AdvancedRegimeSimulator()
        scenario = RegimeScenario(name="vol", regime="VOLATILE", n_bars=50)
        bars = sim.generate_scenario(scenario)
        assert len(bars) == 50

    def test_generate_crisis(self):
        sim = AdvancedRegimeSimulator()
        scenario = RegimeScenario(name="crisis", regime="CRISIS", n_bars=50)
        bars = sim.generate_scenario(scenario)
        assert len(bars) == 50

    def test_transition_scenario(self):
        sim = AdvancedRegimeSimulator()
        phases = [
            RegimeScenario(name="trend", regime="TRENDING", n_bars=50),
            RegimeScenario(name="range", regime="RANGING", n_bars=50),
        ]
        multi = sim.generate_transition(phases)
        assert len(multi) >= 100


# ── StressScenarioLibrary ──────────────────────────────────────────────────

class TestStressScenarioLibrary:

    def test_list_scenarios(self):
        lib = StressScenarioLibrary()
        names = lib.list_scenarios()
        assert len(names) >= 5

    def test_generate_scenario(self):
        lib = StressScenarioLibrary()
        for name in lib.list_scenarios():
            bars = lib.generate_scenario(name)
            assert bars is not None
            assert len(bars) > 0

    def test_unknown_scenario_raises(self):
        lib = StressScenarioLibrary()
        with pytest.raises(ValueError):
            lib.generate_scenario("nonexistent")

    def test_flash_crash_has_drop(self):
        lib = StressScenarioLibrary()
        bars = lib.generate_scenario("flash_crash")
        assert len(bars) > 0
        closes = [b.close for b in bars]
        assert min(closes) < closes[0]  # Price drops

    def test_generate_all(self):
        lib = StressScenarioLibrary()
        all_scenarios = lib.generate_all()
        assert len(all_scenarios) >= 5
