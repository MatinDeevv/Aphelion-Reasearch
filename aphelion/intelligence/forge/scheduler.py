"""
FORGE — Optimization Scheduler
Phase 13 — Engineering Spec v3.0

Schedules periodic optimization runs for FORGE, PROMETHEUS, and HYDRA retraining.
Manages optimization quotas, cooldowns, and parallelism.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScheduledJob:
    """A scheduled optimization job."""
    job_id: str
    target: str              # "FORGE", "PROMETHEUS", "HYDRA"
    schedule_interval: timedelta
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    running: bool = False
    max_duration_minutes: int = 60
    callback: Optional[Callable] = None

    @property
    def is_due(self) -> bool:
        if self.next_run is None:
            return True
        return datetime.now(timezone.utc) >= self.next_run


class ForgeScheduler:
    """
    Manages periodic optimization scheduling.

    Default schedule (from spec):
    - FORGE parameter optimization: every 2 weeks, 200 trials
    - PROMETHEUS genome evolution: every 1 week, 100 generations
    - HYDRA retraining: every 3 days, if performance degrades
    """

    def __init__(self):
        self._jobs: Dict[str, ScheduledJob] = {}
        self._history: List[dict] = []

    def register_job(self, job: ScheduledJob) -> None:
        """Register an optimization job."""
        if job.next_run is None:
            job.next_run = datetime.now(timezone.utc) + job.schedule_interval
        self._jobs[job.job_id] = job
        logger.info("Registered optimization job: %s (interval=%s)", job.job_id, job.schedule_interval)

    def check_due_jobs(self) -> List[ScheduledJob]:
        """Return list of jobs that are due to run."""
        due = []
        for job in self._jobs.values():
            if job.is_due and not job.running:
                due.append(job)
        return due

    def start_job(self, job_id: str) -> bool:
        """Mark a job as started."""
        job = self._jobs.get(job_id)
        if job is None or job.running:
            return False
        job.running = True
        job.last_run = datetime.now(timezone.utc)
        logger.info("Started optimization job: %s", job_id)
        return True

    def complete_job(self, job_id: str, result: Optional[dict] = None) -> None:
        """Mark a job as completed and reschedule."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.running = False
        job.next_run = datetime.now(timezone.utc) + job.schedule_interval
        self._history.append({
            "job_id": job_id,
            "completed": datetime.now(timezone.utc).isoformat(),
            "result": result or {},
        })
        logger.info("Completed optimization job: %s, next run: %s", job_id, job.next_run)

    def get_status(self) -> Dict[str, dict]:
        """Get status of all jobs."""
        return {
            jid: {
                "target": j.target,
                "is_due": j.is_due,
                "running": j.running,
                "last_run": j.last_run.isoformat() if j.last_run else None,
                "next_run": j.next_run.isoformat() if j.next_run else None,
            }
            for jid, j in self._jobs.items()
        }

    @property
    def active_jobs(self) -> int:
        return sum(1 for j in self._jobs.values() if j.running)

    @property
    def total_jobs(self) -> int:
        return len(self._jobs)


def create_default_schedule() -> ForgeScheduler:
    """Create the default optimization schedule per spec."""
    scheduler = ForgeScheduler()
    scheduler.register_job(ScheduledJob(
        job_id="forge_param_opt",
        target="FORGE",
        schedule_interval=timedelta(weeks=2),
        max_duration_minutes=120,
    ))
    scheduler.register_job(ScheduledJob(
        job_id="prometheus_evolution",
        target="PROMETHEUS",
        schedule_interval=timedelta(weeks=1),
        max_duration_minutes=90,
    ))
    scheduler.register_job(ScheduledJob(
        job_id="hydra_retrain",
        target="HYDRA",
        schedule_interval=timedelta(days=3),
        max_duration_minutes=60,
    ))
    return scheduler
