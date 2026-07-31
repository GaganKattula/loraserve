"""
Shared fixtures for the lora-serve test suite.

Tests that need Postgres or Redis expect the compose services to be running:
    docker compose up -d postgres redis

The CI pipeline (.github/workflows/ci.yml) provides these as service containers
automatically.
"""

import os
import pytest
import pytest_asyncio
import asyncpg
from redis import Redis

# ── Use the same env vars the app uses (CI sets these; local dev uses .env) ────
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://gagan@localhost/lora_serve")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

_tables_created = False


@pytest_asyncio.fixture
async def pg_pool():
    """Async Postgres connection pool — created per test function."""
    global _tables_created
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=5)

    if not _tables_created:
        async with pool.acquire() as conn:
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
        _tables_created = True

    yield pool
    await pool.close()


@pytest_asyncio.fixture
async def clean_db(pg_pool):
    """Truncate all test data before each test — isolation between tests."""
    async with pg_pool.acquire() as conn:
        await conn.execute("DELETE FROM training_events")
        await conn.execute("DELETE FROM jobs")
    yield pg_pool


@pytest.fixture
def redis_client():
    """Sync Redis client — cleaned before each test."""
    r = Redis.from_url(REDIS_URL)
    r.flushdb()
    yield r
    r.flushdb()
    r.close()
