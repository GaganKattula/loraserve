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

import asyncio
import logging
import multiprocessing
import os
import signal
import socket
import threading
import uuid

import psycopg2
from redis import Redis

import db
import gpu_lock
from config import settings
from db import (
    complete_job,
    fail_job,
    get_job,
    init_pool,
    update_job_running,
    update_lora_config,
)
from serving.engine import evict_all
from sqs import delete_job_message, receive_job
from storage import upload_adapter
from worker.rank_selector import select_lora_config
from worker.trainer import train_lora

multiprocessing.set_start_method("spawn", force=True)
r = Redis.from_url(settings.redis_url)
pg = psycopg2.connect(settings.database_url)

WORKER_ID = str(uuid.uuid4())
HEARTBEAT_INTERVAL = 30  # seconds
POD_HOST_TTL = 90  # 3x heartbeat — auto-expires if pod dies


def _get_pod_ip() -> str:
    """Get this pod's IP for VPS discovery. Env var override for containers."""
    return os.environ.get("POD_HOST", socket.gethostbyname(socket.gethostname()))


def _register_pod():
    """Write this pod's IP to Redis so the VPS knows where to proxy inference."""
    pod_ip = _get_pod_ip()
    r.set("gpu:pod:host", pod_ip, ex=POD_HOST_TTL)
    return pod_ip


class HeartbeatThread(threading.Thread):
    """Background thread that renews the GPU lock TTL and updates last_alive_at every 30s.

    Uses the sync psycopg2 connection (pg) instead of the async pool to avoid
    event loop conflicts — the async pool is bound to _run_job's event loop,
    which is different from the heartbeat thread's context.
    """

    def __init__(self, job_id: str, lock_token: str):
        super().__init__(daemon=True)
        self._job_id = job_id
        self._lock_token = lock_token
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.wait(HEARTBEAT_INTERVAL):
            gpu_lock.renew(self._lock_token)
            _register_pod()
            try:
                with pg.cursor() as cur:
                    cur.execute(
                        "UPDATE jobs SET last_alive_at = NOW()"
                        " WHERE id = %s AND status = 'running'",
                        (self._job_id,),
                    )
                pg.commit()
            except Exception:
                logger.warning("Heartbeat DB update failed for job %s", self._job_id, exc_info=True)

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
            return  # already claimed

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
            pg_conn=pg,
        )

        # 5. upload adapter to S3 — sync boto3
        adapter_path = upload_adapter(job_id, result["adapter_local_path"])

        # 6. mark complete
        await complete_job(job_id, adapter_path, result["eval_loss"], result["adapter_size_mb"])

    except Exception as e:
        await fail_job(job_id, str(e))
        raise

    finally:
        if heartbeat is not None:
            heartbeat.stop()
        gpu_lock.release(lock_token)
        await db.pool.close()


logger = logging.getLogger("worker")
_running = True


_inference_server_started = False


def _start_inference_server():
    """Start the GPU pod inference FastAPI app on :8001 in a background thread."""
    global _inference_server_started
    if _inference_server_started:
        return
    _inference_server_started = True

    import socket as _socket

    import uvicorn

    from serving.app import app as infer_app

    # Pre-bind the socket so uvicorn doesn't attempt its own bind
    # (which double-binds under multiprocessing spawn mode)
    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 8001))
    sock.set_inheritable(True)

    def _run():
        config = uvicorn.Config(
            infer_app, host="0.0.0.0", port=8001, log_level="info", fd=sock.fileno()
        )
        server = uvicorn.Server(config)
        server.run()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    logger.info("Inference server started on :8001")


def _handle_sigterm(signum, frame):
    global _running
    logger.info("SIGTERM received, finishing current job then exiting")
    _running = False


def run_poll_loop():
    """SQS long-poll loop. Replaces the RQ worker process."""
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    # Register this pod's IP in Redis for VPS inference proxy discovery
    pod_ip = _register_pod()
    logger.info("Registered pod at %s (Redis key gpu:pod:host, TTL %ds)", pod_ip, POD_HOST_TTL)

    # Start the GPU pod inference server in a background thread
    _start_inference_server()

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
                logger.exception(
                    "Job %s failed — message will redeliver after visibility timeout",
                    job_id,
                )

        except Exception:
            logger.exception("Poll loop error — continuing")

    logger.info("Worker poll loop exiting")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_poll_loop()
