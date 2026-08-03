"""
Pydantic model validation tests.

The 20-example minimum is a business constraint: LoRA fine-tuning with fewer
examples produces adapters that overfit to noise. The validator catches this
at the API boundary before any S3 upload or RQ enqueue happens.
"""

import pytest
from pydantic import ValidationError
from api.models import JobRequest, InferRequest


def _make_examples(n: int) -> list[dict]:
    return [{"text": f"sample text {i}", "label": "positive"} for i in range(n)]


class TestJobRequest:
    def test_rejects_under_20_examples(self):
        """<20 examples → ValidationError. Prevents low-quality adapters."""
        with pytest.raises(ValidationError, match="Need at least 20 examples"):
            JobRequest(
                task_description="classify sentiment",
                examples=_make_examples(19),
            )

    def test_accepts_exactly_20(self):
        req = JobRequest(
            task_description="classify sentiment",
            examples=_make_examples(20),
        )
        assert len(req.examples) == 20

    def test_accepts_large_dataset(self):
        req = JobRequest(
            task_description="classify sentiment",
            examples=_make_examples(500),
        )
        assert len(req.examples) == 500

    def test_rejects_empty_examples(self):
        with pytest.raises(ValidationError):
            JobRequest(
                task_description="classify sentiment",
                examples=[],
            )


class TestInferRequest:
    def test_default_max_new_tokens(self):
        req = InferRequest(text="hello world")
        assert req.max_new_tokens == 20

    def test_custom_max_new_tokens(self):
        req = InferRequest(text="hello world", max_new_tokens=50)
        assert req.max_new_tokens == 50
