"""
1. Fetch job from Postgres using get_job(job_id)
2. Claim it atomically — update_job_running(job_id, worker_id)
   if 0 rows updated → already claimed, return early
3. Select LoRA config — select_lora_config(job.num_examples)
4. Call train_lora() with job fields
5. Upload adapter to S3 — upload_adapter(job_id, result['adapter_local_path'])
6. Mark job complete in Postgres — complete_job(...)
7. On any exception → fail_job(job_id, str(e))

"""
import multiprocessing
import asyncio
import psycopg2
import uuid
import db
from db import get_job, update_job_running, complete_job, fail_job, init_pool
from storage import upload_adapter
from worker.trainer import train_lora
from worker.rank_selector import select_lora_config
from redis import Redis
from config import settings

multiprocessing.set_start_method("spawn", force=True)
r = Redis.from_url(settings.redis_url)
pg = psycopg2.connect(settings.database_url)

WORKER_ID = str(uuid.uuid4())

async def _run_job(job_id: str):
    await init_pool()
    try:
        # 1. fetch job
        job = await get_job(job_id)

        # 2. claim atomically
        rows = await update_job_running(job_id, WORKER_ID)
        if rows == 0:
            return   # already claimed

        # 3. select lora config — sync, no await
        lora_cfg = select_lora_config(job["num_examples"])

        # 4. train — sync, no await, runs for minutes
        result = train_lora(
            job_id=job_id,
            dataset_s3_key=job["dataset_s3_key"],
            lora_config=lora_cfg,
            template_version=job["template_version"],
            redis_client=r,
            pg_conn=pg
        )

        # 5. upload adapter to S3 — sync boto3
        adapter_path = upload_adapter(job_id, result["adapter_local_path"])

        # 6. mark complete
        await complete_job(
            job_id,
            adapter_path,
            result["eval_loss"],
            result["adapter_size_mb"]
        )

    except Exception as e:
        await fail_job(job_id, str(e))
        raise 

    finally:
        await db.pool.close()


def train_job(job_id: str):
    asyncio.run(_run_job(job_id))
