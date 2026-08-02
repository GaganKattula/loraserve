from pydantic import BaseModel, field_validator
from uuid import UUID
from typing import Optional
from datetime import datetime


"""
Example          — one labeled example: text + label
JobRequest       — POST /jobs body: task_description + list of Examples
                   with a validator that enforces minimum 20 examples
JobResponse      — POST /jobs response: job_id + status + stream_url + infer_url
JobStatusResponse — GET /jobs/{id} response: job_id + status + eval_loss + adapter_path + training metadata
JobSummary       — one job in the GET /jobs list
JobListResponse  — GET /jobs response: paginated job list
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
    num_examples: Optional[int] = None
    adapter_size_mb: Optional[float] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    target_modules: Optional[str] = None
    created_at: Optional[datetime] = None
    error_message: Optional[str] = None


class JobSummary(BaseModel):
    job_id: UUID
    status: str
    task_description: Optional[str] = None
    num_examples: Optional[int] = None
    eval_loss: Optional[float] = None
    adapter_size_mb: Optional[float] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    created_at: Optional[datetime] = None
    error_message: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: list[JobSummary]
    total: int
    limit: int
    offset: int

class InferRequest(BaseModel):
    text: str
    max_new_tokens: int = 20

class InferResponse(BaseModel):
    output: str
    job_id: UUID
    latency_ms: int
    cache_hit: bool


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_new_tokens: int = 50

class ChatResponse(BaseModel):
    output: str
    job_id: UUID
    latency_ms: int
    cache_hit: bool