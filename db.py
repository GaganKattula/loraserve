import asyncpg
from config import settings


"""
create_tables()        — CREATE TABLE IF NOT EXISTS for jobs and training_events
create_job(...)        — INSERT a new job, return the UUID
get_job(job_id)        — SELECT one job by ID
update_job_status(...) — UPDATE status, started_at, worker_id
complete_job(...)      — UPDATE status=complete, adapter_path, eval_loss
fail_job(...)          — UPDATE status=failed, error_message
insert_training_event(...)  — INSERT one training event row

"""


pool = None


async def init_pool():
    global pool
    pool = await asyncpg.create_pool(settings.database_url, min_size=2, max_size=10)


async def create_tables():
    conn = await asyncpg.connect(settings.database_url)
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                status           VARCHAR(32) NOT NULL DEFAULT 'queued',
                created_at       TIMESTAMPTZ DEFAULT NOW(),
                started_at       TIMESTAMPTZ,
                completed_at     TIMESTAMPTZ,
                worker_id        VARCHAR,
                num_examples     INTEGER,
                task_description TEXT,
                dataset_s3_key   VARCHAR,
                adapter_path     VARCHAR,
                adapter_size_mb  FLOAT,
                eval_loss        FLOAT,
                error_message    TEXT,
                template_version VARCHAR,
                last_alive_at    TIMESTAMPTZ,
                retry_count      INTEGER DEFAULT 0,
                lora_r           INTEGER,
                lora_alpha       INTEGER,
                target_modules   TEXT
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS training_events (
                id         BIGSERIAL PRIMARY KEY,
                job_id     UUID REFERENCES jobs(id),
                step       INTEGER,
                epoch      FLOAT,
                train_loss FLOAT,
                eval_loss  FLOAT,
                grad_norm  FLOAT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        print("Tables created")
    finally:
        await conn.close()


async def create_job(
    id, num_examples, task_description, dataset_s3_key, template_version
):

    async with pool.acquire() as conn:
        job_id = await conn.fetchval(
            """
            INSERT INTO jobs (id, num_examples, task_description, dataset_s3_key, template_version)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            id,
            num_examples,
            task_description,
            dataset_s3_key,
            template_version,
        )
    return job_id


async def get_job(job_id):

    async with pool.acquire() as conn:
        job = await conn.fetchrow(
            """
            SELECT * FROM jobs WHERE id = $1
            """,
            job_id,
        )
        return job


async def update_job_running(job_id, worker_id):

    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE jobs
            SET status = 'running',
                started_at = NOW(),
                last_alive_at = NOW(),
                worker_id = $2
            WHERE id = $1
            AND status = 'queued'
            """,
            job_id,
            worker_id,
        )
        return int(result.split()[-1])


async def update_lora_config(job_id, lora_r, lora_alpha, target_modules):
    """Store the selected LoRA config on the job record after rank selection."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE jobs
            SET lora_r = $2,
                lora_alpha = $3,
                target_modules = $4
            WHERE id = $1
            """,
            job_id,
            lora_r,
            lora_alpha,
            ",".join(target_modules),
        )


async def complete_job(job_id, adapter_path, eval_loss, adapter_size_mb):

    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE jobs
            SET status = 'complete',
                completed_at = NOW(),
                adapter_size_mb = $4,
                eval_loss = $3,
                adapter_path = $2

            WHERE id = $1


            """,
            job_id,
            adapter_path,
            eval_loss,
            adapter_size_mb,
        )
        return int(result.split()[-1])


async def fail_job(job_id, error_message):

    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE jobs
            SET status = 'failed',
                error_message = $2

            WHERE id = $1

            """,
            job_id,
            error_message,
        )
        return int(result.split()[-1])


async def insert_training_event(job_id, step, epoch, train_loss, eval_loss, grad_norm):

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT into training_events(job_id, step, epoch, train_loss, eval_loss, grad_norm)
            VALUES ($1,$2, $3, $4, $5, $6)
            
            """,
            job_id,
            step,
            epoch,
            train_loss,
            eval_loss,
            grad_norm,
        )


async def fetch_training_events(job_id):
    async with pool.acquire() as conn:
        events = await conn.fetch(
            "SELECT * FROM training_events WHERE job_id = $1 ORDER BY id ASC", job_id
        )
    return events


async def list_jobs(limit: int = 20, offset: int = 0):
    """Return recent jobs, newest first. For the dashboard."""
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT id, status, created_at, started_at, completed_at,
                   num_examples, task_description, eval_loss,
                   adapter_size_mb, lora_r, lora_alpha, target_modules,
                   error_message
            FROM jobs
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )


async def count_jobs():
    """Total job count for pagination."""
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT COUNT(*) FROM jobs")


async def heartbeat_job(job_id):
    """Update last_alive_at for a running job. Called by worker heartbeat thread."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE jobs SET last_alive_at = NOW() WHERE id = $1 AND status = 'running'",
            job_id,
        )


async def get_stale_running_jobs(stale_seconds: int = 120):
    """Find jobs stuck in 'running' with no heartbeat for stale_seconds."""
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT id FROM jobs
            WHERE status = 'running'
            AND last_alive_at < NOW() - INTERVAL '1 second' * $1
            """,
            stale_seconds,
        )


async def get_unclaimed_queued_jobs(grace_seconds: int = 300):
    """Find queued jobs that have sat unclaimed past the grace window."""
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT id FROM jobs
            WHERE status = 'queued'
            AND created_at < NOW() - INTERVAL '1 second' * $1
            AND worker_id IS NULL
            """,
            grace_seconds,
        )


async def get_upload_pending_jobs(max_retries: int = 3):
    """Find jobs where adapter trained but S3 upload failed."""
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT id FROM jobs
            WHERE status = 'upload_pending'
            AND retry_count < $1
            """,
            max_retries,
        )


async def increment_retry_count(job_id):
    """Bump retry_count for upload retry tracking."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE jobs SET retry_count = retry_count + 1 WHERE id = $1", job_id
        )
