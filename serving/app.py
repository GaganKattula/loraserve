"""
GPU pod inference server — runs alongside the SQS worker on the GPU pod.

This is NOT the VPS API (api/main.py). This app runs on the GPU pod at :8001
and imports serving.engine directly (torch/peft available here).

The VPS API proxies inference requests to this app via httpx.

Request flow:
  Browser → Nginx → VPS API (proxy) → THIS APP → serving.engine → response
"""

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

import gpu_lock
from config import settings

app = FastAPI()


# ── Auth — same pattern as api/main.py:25-31 ──
async def verify_gpu_secret(authorization: str | None = Header(default=None)):
    secret = settings.gpu_shared_secret
    if not secret:
        return
    if authorization != f"Bearer {secret}":
        raise HTTPException(status_code=401, detail="Invalid or missing GPU shared secret")


# ── Request/Response models (local to GPU pod, not shared with VPS) ──
class InferBody(BaseModel):
    text: str
    max_new_tokens: int = 20


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatBody(BaseModel):
    messages: list[ChatMessage]
    max_new_tokens: int = 50


# ── Endpoints ──


@app.get("/health")
async def health():
    return {"status": "ok", "gpu_lock_held": gpu_lock.is_held()}


@app.post("/infer/{job_id}")
async def infer_job(
    job_id: str,
    body: InferBody,
    adapter_path: str,
    adapter_size_mb: float,
    authorization: str | None = Header(default=None),
):
    """Single-shot inference. VPS proxy passes adapter_path and adapter_size_mb as query params."""
    await verify_gpu_secret(authorization)

    if gpu_lock.is_held():
        raise HTTPException(
            status_code=503,
            headers={"Retry-After": "30"},
            detail="GPU busy (training in progress)",
        )

    from serving.engine import InsufficientVRAMError, run_inference

    try:
        result = await run_inference(
            job_id=job_id,
            adapter_path=adapter_path,
            text=body.text,
            adapter_size_mb=adapter_size_mb,
            max_new_tokens=body.max_new_tokens,
        )
    except InsufficientVRAMError:
        raise HTTPException(
            status_code=503, headers={"Retry-After": "30"}, detail="Insufficient VRAM"
        )

    return result


@app.post("/chat/{job_id}")
async def chat_job(
    job_id: str,
    body: ChatBody,
    adapter_path: str,
    adapter_size_mb: float,
    authorization: str | None = Header(default=None),
):
    """Multi-turn chat inference. VPS proxy passes adapter metadata as query params."""
    await verify_gpu_secret(authorization)

    if gpu_lock.is_held():
        raise HTTPException(
            status_code=503,
            headers={"Retry-After": "30"},
            detail="GPU busy (training in progress)",
        )

    from serving.engine import InsufficientVRAMError, run_chat

    try:
        result = await run_chat(
            job_id=job_id,
            adapter_path=adapter_path,
            adapter_size_mb=adapter_size_mb,
            messages=[{"role": m.role, "content": m.content} for m in body.messages],
            max_new_tokens=body.max_new_tokens,
        )
    except InsufficientVRAMError:
        raise HTTPException(
            status_code=503, headers={"Retry-After": "30"}, detail="Insufficient VRAM"
        )

    return result
