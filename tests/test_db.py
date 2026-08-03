"""
Database lifecycle integration tests — requires running Postgres.

Architecture notes for interview defense:

  Atomic claiming (test_atomic_claim_only_one_wins):
    Two workers calling update_job_running on the same job must not both succeed.
    The WHERE clause `status = 'queued'` is the guard — the first UPDATE flips it
    to 'running', so the second UPDATE matches 0 rows. This is cheaper than a
    SELECT FOR UPDATE and avoids explicit row locking.

  Heartbeat (test_heartbeat_updates_last_alive_at):
    The heartbeat thread updates last_alive_at every 30s. If it stops (pod crash),
    the watchdog detects the stale timestamp and marks the job failed. This is
    crash recovery without requiring the dead process to clean up after itself.
"""

import uuid

import pytest
import pytest_asyncio

import db


@pytest_asyncio.fixture(autouse=True)
async def init_db_pool(clean_db):
    """Point the db module's pool at our test pool."""
    db.pool = clean_db
    yield


@pytest.mark.asyncio
class TestJobLifecycle:
    async def test_create_and_get(self):
        job_id = uuid.uuid4()
        returned_id = await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="classify emails",
            dataset_s3_key="jobs/test/dataset.jsonl",
            template_version="alpaca_v1",
        )
        assert returned_id == job_id

        job = await db.get_job(job_id)
        assert job["status"] == "queued"
        assert job["num_examples"] == 50
        assert job["task_description"] == "classify emails"

    async def test_get_nonexistent_returns_none(self):
        job = await db.get_job(uuid.uuid4())
        assert job is None

    async def test_update_job_running(self):
        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        rows = await db.update_job_running(job_id, "worker-1")
        assert rows == 1

        job = await db.get_job(job_id)
        assert job["status"] == "running"
        assert job["worker_id"] == "worker-1"
        assert job["started_at"] is not None
        assert job["last_alive_at"] is not None

    async def test_atomic_claim_only_one_wins(self):
        """Two workers claim the same job — only the first succeeds.

        This is THE core concurrency guarantee. Without it, two workers could
        train the same adapter simultaneously, wasting GPU time and producing
        non-deterministic results.
        """
        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )

        rows_1 = await db.update_job_running(job_id, "worker-1")
        rows_2 = await db.update_job_running(job_id, "worker-2")

        assert rows_1 == 1
        assert rows_2 == 0  # second claim fails — 0 rows updated

        job = await db.get_job(job_id)
        assert job["worker_id"] == "worker-1"  # first claimer wins

    async def test_complete_job(self):
        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        await db.update_job_running(job_id, "worker-1")
        await db.complete_job(job_id, "s3://bucket/adapter/", 1.23, 28.5)

        job = await db.get_job(job_id)
        assert job["status"] == "complete"
        assert job["adapter_path"] == "s3://bucket/adapter/"
        assert job["eval_loss"] == pytest.approx(1.23)
        assert job["adapter_size_mb"] == pytest.approx(28.5)
        assert job["completed_at"] is not None

    async def test_fail_job(self):
        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        await db.fail_job(job_id, "OOM during training")

        job = await db.get_job(job_id)
        assert job["status"] == "failed"
        assert job["error_message"] == "OOM during training"


@pytest.mark.asyncio
class TestHeartbeat:
    async def test_heartbeat_updates_last_alive_at(self):
        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        await db.update_job_running(job_id, "worker-1")

        job_before = await db.get_job(job_id)
        await db.heartbeat_job(job_id)
        job_after = await db.get_job(job_id)

        assert job_after["last_alive_at"] >= job_before["last_alive_at"]


@pytest.mark.asyncio
class TestWatchdogQueries:
    async def test_stale_running_jobs_detected(self):
        """Watchdog finds jobs with expired heartbeat — crash recovery path."""
        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        await db.update_job_running(job_id, "worker-1")

        # Manually set last_alive_at to 5 minutes ago
        async with db.pool.acquire() as conn:
            await conn.execute(
                "UPDATE jobs SET last_alive_at = NOW() - INTERVAL '5 minutes' WHERE id = $1",
                job_id,
            )

        stale = await db.get_stale_running_jobs(stale_seconds=120)
        stale_ids = [row["id"] for row in stale]
        assert job_id in stale_ids

    async def test_fresh_running_jobs_not_detected(self):
        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        await db.update_job_running(job_id, "worker-1")
        # last_alive_at was just set by update_job_running — fresh

        stale = await db.get_stale_running_jobs(stale_seconds=120)
        stale_ids = [row["id"] for row in stale]
        assert job_id not in stale_ids

    async def test_unclaimed_queued_jobs_detected(self):
        """Recovers jobs lost after a Redis restart — RQ queue is ephemeral,
        Postgres is the source of truth.
        """
        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )

        # Manually set created_at to 10 minutes ago
        async with db.pool.acquire() as conn:
            await conn.execute(
                "UPDATE jobs SET created_at = NOW() - INTERVAL '10 minutes' WHERE id = $1",
                job_id,
            )

        unclaimed = await db.get_unclaimed_queued_jobs(grace_seconds=300)
        unclaimed_ids = [row["id"] for row in unclaimed]
        assert job_id in unclaimed_ids

    async def test_recently_queued_jobs_not_detected(self):
        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        # just created — within grace window

        unclaimed = await db.get_unclaimed_queued_jobs(grace_seconds=300)
        unclaimed_ids = [row["id"] for row in unclaimed]
        assert job_id not in unclaimed_ids


@pytest.mark.asyncio
class TestLoraConfig:
    async def test_update_lora_config(self):
        """Rank selector stores config on the job record for dashboard display."""
        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        await db.update_lora_config(
            job_id,
            lora_r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        job = await db.get_job(job_id)
        assert job["lora_r"] == 16
        assert job["lora_alpha"] == 32
        assert job["target_modules"] == "q_proj,k_proj,v_proj,o_proj"


@pytest.mark.asyncio
class TestListJobs:
    async def test_list_jobs_returns_newest_first(self):
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()
        await db.create_job(
            id=id1,
            num_examples=50,
            task_description="first",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        await db.create_job(
            id=id2,
            num_examples=100,
            task_description="second",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )

        jobs = await db.list_jobs(limit=10, offset=0)
        # newest (id2) should come first
        assert len(jobs) >= 2
        job_ids = [j["id"] for j in jobs]
        assert job_ids.index(id2) < job_ids.index(id1)

    async def test_list_jobs_pagination(self):
        for i in range(5):
            await db.create_job(
                id=uuid.uuid4(),
                num_examples=50,
                task_description=f"job {i}",
                dataset_s3_key="k",
                template_version="alpaca_v1",
            )

        page1 = await db.list_jobs(limit=2, offset=0)
        page2 = await db.list_jobs(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        # no overlap
        ids1 = {j["id"] for j in page1}
        ids2 = {j["id"] for j in page2}
        assert ids1.isdisjoint(ids2)

    async def test_count_jobs(self):
        initial = await db.count_jobs()
        await db.create_job(
            id=uuid.uuid4(),
            num_examples=50,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        assert await db.count_jobs() == initial + 1
