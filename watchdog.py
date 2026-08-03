"""
VPS-side watchdog — runs every 2 minutes as a FastAPI background task.

Three scans (per LLD-v3 mermaid spec):
  1. Stale running jobs   — heartbeat lost, mark failed
  2. Unclaimed queued jobs — re-enqueue via SQS (recovers from send_message failure)
  3. Upload pending jobs   — retry S3 upload (pod still alive, local files exist)
"""

import asyncio
import logging

from db import (
    fail_job,
    get_stale_running_jobs,
    get_unclaimed_queued_jobs,
    get_upload_pending_jobs,
    increment_retry_count,
)
from sqs import enqueue_job

logger = logging.getLogger("watchdog")

SCAN_INTERVAL = 120  # seconds between scans
STALE_THRESHOLD = 120  # seconds without heartbeat → stale
UNCLAIMED_GRACE = 300  # seconds before re-enqueuing a queued job


async def _scan_once():
    """Run all three watchdog scans."""

    # Scan 1 — stale running jobs (heartbeat lost)
    stale = await get_stale_running_jobs(STALE_THRESHOLD)
    for row in stale:
        job_id = row["id"]
        logger.warning("Watchdog: marking stale job %s as failed (heartbeat lost)", job_id)
        await fail_job(job_id, "worker heartbeat lost")

    # Scan 2 — unclaimed queued jobs (e.g. SQS send_message failed after DB insert)
    unclaimed = await get_unclaimed_queued_jobs(UNCLAIMED_GRACE)
    for row in unclaimed:
        job_id = row["id"]
        logger.warning("Watchdog: re-enqueuing unclaimed job %s to SQS", job_id)
        enqueue_job(str(job_id))

    # Scan 3 — upload pending (adapter trained, S3 upload failed, pod still up)
    pending = await get_upload_pending_jobs(max_retries=3)
    for row in pending:
        job_id = row["id"]
        logger.warning("Watchdog: retrying upload for job %s", job_id)
        await increment_retry_count(job_id)
        enqueue_job(str(job_id))


async def run_watchdog():
    """Loop forever, scanning every SCAN_INTERVAL seconds."""
    logger.info("Watchdog started (interval=%ds)", SCAN_INTERVAL)
    while True:
        try:
            await _scan_once()
        except Exception:
            logger.exception("Watchdog scan error")
        await asyncio.sleep(SCAN_INTERVAL)
