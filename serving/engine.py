import asyncio
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import settings
from peft import get_peft_model
from peft import LoraConfig as PeftLoraConfig, TaskType
from collections import OrderedDict
from storage import download_adapter

bootstrap_config = PeftLoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # arbitrary — never actually used for real inference
    target_modules=["q_proj", "v_proj"],  # arbitrary — same reason
)

VRAM_BUDGET_MB = 10240
tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    settings.base_model, dtype=settings.dtype, device_map=settings.device
)
cache_lock = asyncio.Lock()
cache = OrderedDict()


model = get_peft_model(base_model, bootstrap_config)  # needs bootstrap PeftLoraConfig


INFERENCE_TEMPLATES = {
    "alpaca_v1": lambda text: f"### Instruction:\n{text}\n\n### Response:\n"
}

"""
Phase 3 — multi-adapter serving with an LRU cache, replacing Phase 1's
per-request PeftModel.from_pretrained() (which cost ~4.5s every call).

INITIAL SETUP (runs once, at module import):
    - load tokenizer + frozen base model
    - get_peft_model(base_model, bootstrap_config) → ONE shared PeftModel
      for the life of the process. bootstrap_config's values are arbitrary
      and never used for real inference — every real adapter brings its
      own config from its saved adapter_config.json via load_adapter().
    - cache = OrderedDict()      — tracks which adapters are loaded + their VRAM cost
    - cache_lock = asyncio.Lock() — serializes all cache mutations
    - VRAM_BUDGET_MB              — ceiling for total adapter VRAM usage

ON EVERY REQUEST — get_or_load_adapter(job_id, adapter_path, adapter_size_mb):
    - check cache for job_id
    - IF HIT:
        move key to end (mark as most-recently-used)
        set_adapter(job_id) to activate it — no download, no load, no S3 call
    - IF MISS:
        Stage A (cheap estimate, no real allocation attempted yet):
            while estimated_total + new_adapter_estimate > VRAM_BUDGET_MB:
                evict LRU entry (popitem + delete_adapter)
        Stage B (ground truth — the real allocation, with retry):
            try: load_adapter()
            except OutOfMemoryError:
                if cache is empty: raise HTTPException(503, Retry-After)
                else: evict one more LRU entry, empty_cache(), retry load_adapter()
            once load succeeds: set_adapter(job_id), insert into cache

THEN, REGARDLESS OF HIT OR MISS — run_inference() continues:
    1. format prompt via INFER_TEMPLATES[template_version](text)
    2. tokenizer(prompt, return_tensors='pt').to(settings.device) → inputs
    3. model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    4. tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    5. return {"output": text, "latency_ms": int, "cache_hit": bool}
"""


class InsufficientVRAMError(Exception):
    """Raised when all cached adapters have been evicted and VRAM is still insufficient."""

    pass


def total_cached_vram_mb():
    return sum(entry["vram_mb"] for entry in cache.values())


def evict_all():
    """Flush every cached adapter and free VRAM. Call before training."""
    while cache:
        evicted_id, _ = cache.popitem(last=False)
        model.delete_adapter(evicted_id)
    torch.cuda.empty_cache()


async def run_inference(
    job_id,
    adapter_path,
    adapter_size_mb,
    text,
    max_new_tokens,
    template_version: str = "alpaca_v1",
) -> dict:

    t0 = time.monotonic()  # start time

    cache_hit = await get_or_load_adapter(job_id, adapter_path, adapter_size_mb)

    # Format input text using inference template

    prompt = INFERENCE_TEMPLATES[template_version](text)
    # tokenizer step
    inputs = tokenizer(text=prompt, return_tensors="pt").to(settings.device)
    input_length = inputs["input_ids"].shape[1]  # number of input tokens

    # model generate step
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    result = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    latency_ms = int((time.monotonic() - t0) * 1000)  # end time | calculate time delta

    return {"output": result, "latency_ms": latency_ms, "cache_hit": cache_hit}


async def run_chat(
    job_id, adapter_path, adapter_size_mb, messages: list[dict], max_new_tokens
) -> dict:
    """Multi-turn chat inference using the model's native chat template (ChatML).

    Args:
        messages: list of {"role": "user"|"assistant"|"system", "content": str}
    """
    t0 = time.monotonic()

    cache_hit = await get_or_load_adapter(job_id, adapter_path, adapter_size_mb)

    # Format using the tokenizer's built-in chat template (ChatML for SmolLM2-Instruct)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text=prompt, return_tensors="pt").to(settings.device)
    input_length = inputs["input_ids"].shape[1]  # (1, T) → T

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    result = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    latency_ms = int((time.monotonic() - t0) * 1000)

    return {"output": result, "latency_ms": latency_ms, "cache_hit": cache_hit}


async def get_or_load_adapter(job_id, adapter_path, adapter_size_mb):

    adapter_dir = download_adapter(job_id=job_id, adapter_s3_path=adapter_path)
    async with cache_lock:  # serialize concurrent requests
        if job_id in cache:
            cache.move_to_end(job_id)  # cache hit — promote to MRU
            model.set_adapter(job_id)  # activate this adapter on the shared model
            return True

        # cache miss — make room using the cheap estimate BEFORE attempting load
        estimated_vram_mb = adapter_size_mb * 3.0
        while total_cached_vram_mb() + estimated_vram_mb > VRAM_BUDGET_MB:
            if len(cache) == 0:
                break  # nothing left to evict by estimate — let the real load attempt decide
            evicted_id, evicted_meta = cache.popitem(last=False)
            model.delete_adapter(evicted_id)

        # now attempt the real load, with the OOM-retry loop as the ground-truth fallback
        while True:
            try:
                # compute memory before loading adapter
                mem_before = torch.cuda.memory_allocated()
                # load adapter
                model.load_adapter(adapter_dir, adapter_name=job_id)
                # compute memory after loading adapter
                mem_after = torch.cuda.memory_allocated()
                # compute delta
                measured_vram_mb = (mem_after - mem_before) / 1e6

                break
            except torch.cuda.OutOfMemoryError:
                if len(cache) == 0:
                    raise InsufficientVRAMError(
                        f"Cannot load adapter for job {job_id} — insufficient VRAM after full cache eviction"
                    )
                evicted_id, evicted_meta = cache.popitem(last=False)
                model.delete_adapter(evicted_id)
                torch.cuda.empty_cache()

        # store actual VRAM used by adapter
        cache[job_id] = {"vram_mb": measured_vram_mb, "adapter_path": adapter_dir}

        model.set_adapter(job_id)

        return False
