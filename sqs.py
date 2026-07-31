"""
SQS client for lora-serve job queue.

enqueue_job(job_id)            → str   (SQS MessageId)
receive_job(wait_seconds=20)   → dict | None
delete_job_message(receipt_handle) → None
notify_lambda(job_id)          → None  (fire-and-forget async invocation)
"""
import json
import datetime
import logging
import boto3
from config import settings

logger = logging.getLogger("sqs")

_profile = settings.aws_profile if settings.aws_profile else None
_session = boto3.Session(profile_name=_profile)
_sqs = _session.client("sqs", region_name=settings.aws_default_region)


def enqueue_job(job_id: str) -> str:
    """Send a job dispatch message to SQS. Returns the SQS MessageId."""
    resp = _sqs.send_message(
        QueueUrl=settings.sqs_queue_url,
        MessageBody=json.dumps({
            "job_id": job_id,
            "enqueued_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }),
    )
    logger.info("Enqueued job %s → SQS MessageId %s", job_id, resp["MessageId"])
    return resp["MessageId"]


def receive_job(wait_seconds: int = 20) -> dict | None:
    """Long-poll SQS for one message. Returns parsed body dict or None."""
    resp = _sqs.receive_message(
        QueueUrl=settings.sqs_queue_url,
        MaxNumberOfMessages=1,
        WaitTimeSeconds=wait_seconds,
        VisibilityTimeout=settings.sqs_visibility_timeout,
        AttributeNames=["ApproximateReceiveCount"],
    )
    messages = resp.get("Messages", [])
    if not messages:
        return None

    msg = messages[0]
    body = json.loads(msg["Body"])
    body["_receipt_handle"] = msg["ReceiptHandle"]
    body["_receive_count"] = int(
        msg.get("Attributes", {}).get("ApproximateReceiveCount", 1)
    )
    return body


def delete_job_message(receipt_handle: str) -> None:
    """Delete a message from SQS after successful processing."""
    _sqs.delete_message(
        QueueUrl=settings.sqs_queue_url,
        ReceiptHandle=receipt_handle,
    )


def notify_lambda(job_id: str) -> None:
    """Fire-and-forget Lambda invocation for orchestration hooks.
    Skips silently if orchestrator_lambda_name is not configured."""
    if not settings.orchestrator_lambda_name:
        return
    lambda_client = _session.client("lambda", region_name=settings.aws_default_region)
    lambda_client.invoke(
        FunctionName=settings.orchestrator_lambda_name,
        InvocationType="Event",
        Payload=json.dumps({"job_id": job_id}).encode(),
    )
    logger.info("Lambda notified for job %s", job_id)
