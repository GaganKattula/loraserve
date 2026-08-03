"""
Lambda orchestration handler for lora-serve.

Triggered via async invocation from the API when a new training job is enqueued.
Handles notification side effects without touching the training path.

Payload: {"job_id": "..."}
"""

import json
import os
import logging
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

NOTIFICATION_TOPIC_ARN = os.environ.get("NOTIFICATION_TOPIC_ARN", "")


def handler(event, context):
    job_id = event.get("job_id", "unknown")
    logger.info("Orchestrator invoked for job %s", job_id)

    # Hook 1 — send notification via SNS (email/Slack/webhook)
    if NOTIFICATION_TOPIC_ARN:
        sns = boto3.client("sns")
        sns.publish(
            TopicArn=NOTIFICATION_TOPIC_ARN,
            Subject=f"LoRA training job queued: {job_id}",
            Message=json.dumps(
                {
                    "job_id": job_id,
                    "event": "job_queued",
                }
            ),
        )
        logger.info("SNS notification sent for job %s", job_id)

    # Hook 2 — future: auto-scale GPU instance
    # _maybe_scale_gpu(job_id)

    return {
        "statusCode": 200,
        "body": json.dumps({"job_id": job_id, "status": "notified"}),
    }
