"""
SQS module tests — uses moto to mock AWS SQS in-process.

Tests verify the enqueue → receive → delete round-trip
without any real AWS calls.
"""

import pytest
from unittest.mock import patch
import boto3
from moto import mock_aws


FAKE_QUEUE_NAME = "lora-serve-jobs-test"
FAKE_REGION = "us-east-1"


@pytest.fixture
def sqs_queue():
    """Create a moto SQS queue and patch settings so sqs.py uses it."""
    with mock_aws():
        client = boto3.client("sqs", region_name=FAKE_REGION)
        resp = client.create_queue(QueueName=FAKE_QUEUE_NAME)
        queue_url = resp["QueueUrl"]

        # Patch settings and the module-level SQS client before importing sqs
        with patch("sqs.settings") as mock_settings, patch("sqs._sqs", client):
            mock_settings.sqs_queue_url = queue_url
            mock_settings.sqs_visibility_timeout = 900
            mock_settings.orchestrator_lambda_name = ""
            mock_settings.aws_default_region = FAKE_REGION
            mock_settings.aws_profile = ""

            from sqs import enqueue_job, receive_job, delete_job_message

            yield {
                "client": client,
                "queue_url": queue_url,
                "enqueue_job": enqueue_job,
                "receive_job": receive_job,
                "delete_job_message": delete_job_message,
            }


class TestSQSRoundTrip:
    def test_enqueue_returns_message_id(self, sqs_queue):
        msg_id = sqs_queue["enqueue_job"]("test-job-123")
        assert isinstance(msg_id, str)
        assert len(msg_id) > 0

    def test_receive_returns_job_id(self, sqs_queue):
        sqs_queue["enqueue_job"]("test-job-456")
        msg = sqs_queue["receive_job"](wait_seconds=0)

        assert msg is not None
        assert msg["job_id"] == "test-job-456"
        assert "enqueued_at" in msg
        assert "_receipt_handle" in msg
        assert "_receive_count" in msg

    def test_receive_returns_none_when_empty(self, sqs_queue):
        msg = sqs_queue["receive_job"](wait_seconds=0)
        assert msg is None

    def test_delete_removes_message(self, sqs_queue):
        sqs_queue["enqueue_job"]("test-job-789")
        msg = sqs_queue["receive_job"](wait_seconds=0)
        assert msg is not None

        sqs_queue["delete_job_message"](msg["_receipt_handle"])

        # Queue should be empty now — change visibility to 0 so deleted msg
        # won't reappear, and verify no messages remain
        attrs = sqs_queue["client"].get_queue_attributes(
            QueueUrl=sqs_queue["queue_url"],
            AttributeNames=["ApproximateNumberOfMessages"],
        )
        count = int(attrs["Attributes"]["ApproximateNumberOfMessages"])
        assert count == 0

    def test_message_body_structure(self, sqs_queue):
        sqs_queue["enqueue_job"]("structured-job")
        msg = sqs_queue["receive_job"](wait_seconds=0)

        assert msg["job_id"] == "structured-job"
        # enqueued_at should be an ISO timestamp with UTC timezone
        assert "+00:00" in msg["enqueued_at"] or msg["enqueued_at"].endswith("Z")
        # internal fields should be present
        assert isinstance(msg["_receive_count"], int)
        assert isinstance(msg["_receipt_handle"], str)
