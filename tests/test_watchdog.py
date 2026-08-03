"""
Watchdog scan logic tests — verifies crash recovery behavior.

Architecture defense:

  Why a VPS-side watchdog instead of self-healing on the GPU pod?
    The GPU pod IS the thing that crashed. A dead process can't detect its own
    death. The VPS is always-on ($20/mo) and has access to Postgres (truth)
    and Redis (queue). It's the natural place for health monitoring.

  Why mark stale jobs as 'failed' instead of re-enqueuing?
    A stale job's training state is unknown — it may have partially written
    to S3, corrupted the adapter directory, or left GPU memory in a bad state.
    Re-enqueuing blindly risks non-deterministic behavior. Failing is safe;
    the user can manually retry with a fresh job.

  Why re-enqueue unclaimed queued jobs?
    After a Redis restart, the RQ queue (which is just a Redis list) is empty.
    But Postgres still has jobs with status='queued' and no worker_id. These
    jobs were accepted but never dispatched. The atomic claim guard
    (UPDATE WHERE status='queued') prevents duplicate execution even if the
    same job is enqueued twice.
"""

import uuid

import pytest
import pytest_asyncio

import db
from watchdog import _scan_once


@pytest_asyncio.fixture(autouse=True)
async def init_db_pool(clean_db):
    db.pool = clean_db
    yield


@pytest.mark.asyncio
class TestWatchdogScans:
    async def test_stale_job_marked_failed(self, clean_db):
        """Pod crashed mid-training → heartbeat stopped → watchdog catches it."""
        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        await db.update_job_running(job_id, "worker-1")

        # Simulate stale heartbeat
        async with clean_db.acquire() as conn:
            await conn.execute(
                "UPDATE jobs SET last_alive_at = NOW() - INTERVAL '5 minutes' WHERE id = $1",
                job_id,
            )

        await _scan_once()

        job = await db.get_job(job_id)
        assert job["status"] == "failed"
        assert "heartbeat lost" in job["error_message"]

    async def test_fresh_job_not_marked_failed(self, clean_db):
        """Running job with recent heartbeat must NOT be touched."""
        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        await db.update_job_running(job_id, "worker-1")
        # last_alive_at just set — fresh

        await _scan_once()

        job = await db.get_job(job_id)
        assert job["status"] == "running"  # untouched
