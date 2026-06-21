from pydantic import BaseModel, field_validator
from uuid import UUID
from typing import Optional


"""
Example          — one labeled example: text + label
JobRequest       — POST /jobs body: task_description + list of Examples
                   with a validator that enforces minimum 20 examples
JobResponse      — POST /jobs response: job_id + status + stream_url + infer_url
JobStatusResponse — GET /jobs/{id} response: job_id + status + eval_loss + adapter_path
"""
class Example(BaseModel):
    text: str
    label: str

class JobRequest(BaseModel):
    task_description: str
    examples: list[Example]
    
    @field_validator("examples")
    @classmethod
    def min_examples(cls, v):
        if len(v) < 20:
            raise ValueError("Need at least 20 examples")
        return v

class JobResponse(BaseModel):
    job_id: UUID
    status: str
    stream_url: str
    infer_url: str


class JobStatusResponse(BaseModel):
    job_id: UUID
    status: str
    eval_loss: Optional[float] = None
    adapter_path: Optional[str] = None

class InferRequest(BaseModel):
    text: str
    max_new_tokens: int = 20

class InferResponse(BaseModel):
    output: str
    job_id: UUID
    latency_ms: int
    cache_hit: bool