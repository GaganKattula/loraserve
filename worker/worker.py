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
import threading
import psycopg2
import uuid
import db
from db import get_job, update_job_running, update_lora_config, complete_job, fail_job, init_pool, heartbeat_job
from storage import upload_adapter
from worker.trainer import train_lora
from worker.rank_selector import select_lora_config
from redis import Redis
from config import settings
from serving.engine import evict_all
import gpu_lock

multiprocessing.set_start_method("spawn", force=True)
r = Redis.from_url(settings.redis_url)
pg = psycopg2.connect(settings.database_url)

WORKER_ID = str(uuid.uuid4())
HEARTBEAT_INTERVAL = 30  # seconds


class HeartbeatThread(threading.Thread):
    """Background thread that renews the GPU lock TTL and updates last_alive_at every 30s."""

    def __init__(self, job_id: str, lock_token: str):
        super().__init__(daemon=True)
        self._job_id = job_id
        self._lock_token = lock_token
        self._stop_event = threading.Event()
        self._loop = None

    def run(self):
        self._loop = asyncio.new_event_loop()
        try:
            while not self._stop_event.wait(HEARTBEAT_INTERVAL):
                gpu_lock.renew(self._lock_token)
                self._loop.run_until_complete(heartbeat_job(self._job_id))
        finally:
            self._loop.close()

    def stop(self):
        self._stop_event.set()


async def _run_job(job_id: str):
    await init_pool()
    lock_token = f"train:{WORKER_ID}:{job_id}"
    heartbeat = None
    try:
        # 1. fetch job
        job = await get_job(job_id)

        # 2. claim atomically
        rows = await update_job_running(job_id, WORKER_ID)
        if rows == 0:
            return   # already claimed

        # 3. select lora config — sync, no await
        lora_cfg = select_lora_config(job["num_examples"])
        await update_lora_config(job_id, lora_cfg.r, lora_cfg.alpha, lora_cfg.target_modules)

        # 3.5. acquire GPU lock — blocks inference while training
        if not gpu_lock.acquire(lock_token):
            await fail_job(job_id, "GPU lock held by another process")
            return

        # 3.6. flush inference cache — free VRAM for training
        evict_all()

        # 3.7. start heartbeat — renews GPU lock TTL + Postgres last_alive_at every 30s
        heartbeat = HeartbeatThread(job_id, lock_token)
        heartbeat.start()

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
        if heartbeat is not None:
            heartbeat.stop()
        gpu_lock.release(lock_token)
        await db.pool.close()


import signal
import logging
from sqs import receive_job, delete_job_message

logger = logging.getLogger("worker")
_running = True


def _handle_sigterm(signum, frame):
    global _running
    logger.info("SIGTERM received, finishing current job then exiting")
    _running = False


def run_poll_loop():
    """SQS long-poll loop. Replaces the RQ worker process."""
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)
    logger.info("Worker poll loop started")

    while _running:
        try:
            msg = receive_job(wait_seconds=20)
            if msg is None:
                continue

            job_id = msg["job_id"]
            receipt_handle = msg["_receipt_handle"]
            logger.info("Received job %s (receive_count=%s)", job_id, msg["_receive_count"])

            try:
                asyncio.run(_run_job(job_id))
                delete_job_message(receipt_handle)
                logger.info("Job %s complete, message deleted", job_id)
            except Exception:
                logger.exception("Job %s failed — message will redeliver after visibility timeout", job_id)

        except Exception:
            logger.exception("Poll loop error — continuing")

    logger.info("Worker poll loop exiting")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_poll_loop()
