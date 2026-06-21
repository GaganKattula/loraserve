"""
POST /jobs      → validate, upload dataset to S3, INSERT to Postgres, enqueue RQ
GET  /jobs/{id} → SELECT job from Postgres, return status
POST /infer/{id} → validate job complete, run inference
"""
import db
import uuid
from uuid import UUID
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from db import create_tables, init_pool
from db import  create_job as db_create_job
from db import get_job as db_get_job
from storage import upload_dataset
from worker.worker import train_job
from redis import Redis
from rq import Queue
from config import settings
from api.models import JobRequest, JobResponse, JobStatusResponse, InferRequest, InferResponse, Example
from serving.engine import run_inference
import redis.asyncio as aioredis
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles


def serialize_examples(examples: list) -> bytes:
    lines = [json.dumps({"text": e.text, "label": e.label}) for e in examples]
    return "\n".join(lines).encode("utf-8")

redis_conn = Redis.from_url(settings.redis_url)
q = Queue(connection=redis_conn)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    await init_pool()
    yield
    await db.pool.close()

app = FastAPI(lifespan=lifespan)

@app.post('/jobs', status_code=202)
async def create_job(request: JobRequest):
    
    content = serialize_examples(request.examples)
    # upload dataset to S3
    job_id = uuid.uuid4()   # generate UUID before inserting
    key = upload_dataset(str(job_id), content)

    # INSERT to Postgres
    job_id = await db_create_job(id=job_id, num_examples=len(request.examples), task_description=request.task_description,
                               dataset_s3_key=key,template_version='alpaca_v1')


    # enqueue RedisQueue

    q.enqueue(train_job, kwargs={"job_id": str(job_id)})
    
    return JobResponse(
    job_id=job_id,
    status="queued",
    stream_url=f"/jobs/{job_id}/stream",
    infer_url=f"/infer/{job_id}"
    )


@app.get('/jobs/{job_id}')
async def get_job(job_id: UUID):
    """
    Takes job_id: UUID as path parameter
    Calls await db_get_job(job_id) — import it as db_get_job to avoid name collision
    If job is None → raise HTTPException(status_code=404, detail="Job not found")
    Return JobStatusResponse
    """

    job = await db_get_job(job_id=job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    eval_loss=job["eval_loss"] if job["eval_loss"] is not None else None
    adapter_path=job["adapter_path"] if job["adapter_path"] is not None else None

    return JobStatusResponse(job_id=job_id, status=job['status'], 
                           eval_loss=eval_loss,
                           adapter_path=adapter_path)



@app.post('/infer/{job_id}')
async def infer_job(job_id: UUID, request: InferRequest):
    """
    Takes job_id: UUID as path parameter and request: InferRequest as body
    Fetch job from Postgres
    If job status is not "complete" → raise HTTPException(status_code=400, detail="Job not ready")
    Call serving engine — import and call directly for Phase 1
    Return InferResponse
    """
    job = await db_get_job(job_id=job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job['status'] != 'complete':
        raise HTTPException(status_code=400, detail="Job not ready")
    
    # insert serving engine call
    result = run_inference(
    job_id=str(job_id),
    adapter_path=job["adapter_path"],
    text=request.text,
    max_new_tokens=request.max_new_tokens
    )

    return InferResponse(
    output=result["output"],
    job_id=job_id,
    latency_ms=result["latency_ms"],
    cache_hit=result["cache_hit"]
    )




@app.get('/jobs/{job_id}/stream')
async def stream_job(job_id: UUID):
    async def event_generator():
        # Step 1 — subscribe FIRST, before any status check (TOCTOU fix)
        r = aioredis.from_url(settings.redis_url)
        pubsub = r.pubsub()
        await pubsub.subscribe(f"job:{job_id}:events")

        # Step 2 — replay historical events from Postgres
        events = await db.fetch_training_events(job_id)
        max_step_seen = 0
        for event in events:
            payload = {
                "type": "eval" if event["step"] is None else "step",
                "step": event["step"],
                "epoch": event["epoch"],
                "train_loss": event["train_loss"],
                "eval_loss": event["eval_loss"],
                "grad_norm": event["grad_norm"],
            }
            yield f"data: {json.dumps(payload)}\n\n"
            if event["step"] is not None:
                max_step_seen = max(max_step_seen, event["step"])

        # Step 3 — check current status
        job = await db_get_job(job_id=job_id)
        if job["status"] in ("complete", "failed"):
            yield f"data: {json.dumps({'type': 'done', 'status': job['status'], 'eval_loss': job['eval_loss']})}\n\n"
            await pubsub.unsubscribe(f"job:{job_id}:events")
            await r.close()
            return

        # Step 4 — drain live events buffered since Step 1, with dedup
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            data = json.loads(message["data"])
            if data.get("step") is not None and data["step"] <= max_step_seen:
                continue   # skip duplicate already replayed from Postgres
            yield f"data: {json.dumps(data)}\n\n"
            if data.get("type") == "done":
                break

        await pubsub.unsubscribe(f"job:{job_id}:events")
        await r.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


app.mount("/static", StaticFiles(directory="frontend"), name="static")