"""
API endpoint integration tests — requires Postgres + Redis.

Tests exercise the HTTP layer without touching torch/serving engine.
The inference endpoint's serving engine import is deferred (lazy import),
so these tests can run on a CPU-only machine.

Architecture note: The shared-secret auth test validates that the GPU pod
can't be accessed without the pre-shared token. In production, the VPS
proxies to the pod with this header — if it's missing, the request was
sent directly to the pod, bypassing the VPS's job-status validation.
"""

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import db


@pytest_asyncio.fixture
async def app(clean_db):
    """Create a fresh FastAPI app with test DB pool."""
    # Patch db.pool before importing the app
    db.pool = clean_db

    from api.main import app as fastapi_app

    yield fastapi_app


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
class TestHealthEndpoint:
    async def test_health_returns_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["db"] is True
        assert data["redis"] is True


@pytest.mark.asyncio
class TestJobEndpoints:
    async def test_get_nonexistent_job_returns_404(self, client):
        resp = await client.get(f"/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Job not found"

    async def test_get_existing_job(self, client, clean_db):
        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        resp = await client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"


@pytest.mark.asyncio
class TestInferEndpoint:
    async def test_infer_nonexistent_job_returns_404(self, client):
        resp = await client.post(
            f"/infer/{uuid.uuid4()}",
            json={"text": "test", "max_new_tokens": 5},
        )
        assert resp.status_code == 404

    async def test_infer_incomplete_job_returns_400(self, client, clean_db):
        """Inference on a queued/running job must fail — the adapter doesn't exist yet."""
        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        resp = await client.post(
            f"/infer/{job_id}",
            json={"text": "test", "max_new_tokens": 5},
        )
        assert resp.status_code == 400
        assert "not ready" in resp.json()["detail"]

    async def test_infer_returns_503_when_gpu_locked(self, client, clean_db, redis_client):
        """GPU lock held by training → inference gets 503 with Retry-After.

        This is the core safety mechanism: training and inference never run
        concurrently on the same GPU. The 503 with Retry-After tells the client
        to back off, not to fail permanently.
        """
        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        await db.update_job_running(job_id, "worker-1")
        await db.complete_job(job_id, "s3://bucket/adapter/", 1.0, 10.0)

        # Set GPU lock
        redis_client.set("gpu:lock", "train:worker-1:test", px=60000)

        resp = await client.post(
            f"/infer/{job_id}",
            json={"text": "test", "max_new_tokens": 5},
        )
        assert resp.status_code == 503
        assert "Retry-After" in resp.headers

    async def test_shared_secret_rejects_bad_token(self, client, clean_db, monkeypatch):
        """Missing/wrong secret → 401. Protects the GPU pod from direct access."""
        monkeypatch.setenv("GPU_SHARED_SECRET", "correct-secret")
        # Force reload settings
        from config import Settings

        monkeypatch.setattr("api.main.settings", Settings())

        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        await db.update_job_running(job_id, "worker-1")
        await db.complete_job(job_id, "s3://bucket/adapter/", 1.0, 10.0)

        resp = await client.post(
            f"/infer/{job_id}",
            json={"text": "test", "max_new_tokens": 5},
            headers={"Authorization": "Bearer wrong-secret"},
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
class TestJobListEndpoint:
    async def test_list_jobs_empty(self, client):
        resp = await client.get("/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    async def test_list_jobs_returns_created_jobs(self, client, clean_db):
        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=50,
            task_description="classify emails",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        resp = await client.get("/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["jobs"][0]["job_id"] == str(job_id)
        assert data["jobs"][0]["task_description"] == "classify emails"
        assert data["jobs"][0]["num_examples"] == 50

    async def test_list_jobs_pagination(self, client, clean_db):
        for i in range(5):
            await db.create_job(
                id=uuid.uuid4(),
                num_examples=50,
                task_description=f"job {i}",
                dataset_s3_key="k",
                template_version="alpaca_v1",
            )
        resp = await client.get("/jobs?limit=2&offset=0")
        data = resp.json()
        assert len(data["jobs"]) == 2
        assert data["total"] == 5

    async def test_get_job_includes_lora_metadata(self, client, clean_db):
        """GET /jobs/{id} returns training metadata for dashboard display."""
        job_id = uuid.uuid4()
        await db.create_job(
            id=job_id,
            num_examples=200,
            task_description="test",
            dataset_s3_key="k",
            template_version="alpaca_v1",
        )
        await db.update_lora_config(job_id, 16, 32, ["q_proj", "k_proj", "v_proj", "o_proj"])

        resp = await client.get(f"/jobs/{job_id}")
        data = resp.json()
        assert data["lora_r"] == 16
        assert data["lora_alpha"] == 32
        assert data["target_modules"] == "q_proj,k_proj,v_proj,o_proj"
        assert data["num_examples"] == 200
